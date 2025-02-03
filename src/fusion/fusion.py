"""Python 3.7 SDK for J.P. Morgan's Fusion platform."""
from __future__ import annotations

import json as js
import logging
import re
import sys
import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from joblib import Parallel, delayed
from tabulate import tabulate
from tqdm import tqdm

from .attributes import Attribute, Attributes
from .credentials import FusionCredentials
from .dataflow import InputDataFlow, OutputDataFlow
from .dataset import Dataset
from .exceptions import APIResponseError
from .fusion_filesystem import FusionHTTPFileSystem
from .fusion_types import Types
from .product import Product
from .report import Report
from .utils import (
    RECOGNIZED_FORMATS,
    cpu_count,
    distribution_to_filename,
    distribution_to_url,
    get_default_fs,
    get_session,
    is_dataset_raw,
    normalise_dt_param_str,
    path_to_url,
    requests_raise_for_status,
    tqdm_joblib,
    upload_files,
    validate_file_names,
)

if TYPE_CHECKING:
    import fsspec
    import requests


logger = logging.getLogger(__name__)
VERBOSE_LVL = 25


class Fusion:
    """Core Fusion class for API access."""

    @staticmethod
    def _call_for_dataframe(url: str, session: requests.Session) -> pd.DataFrame:
        """Private function that calls an API endpoint and returns the data as a pandas dataframe.

        Args:
            url (str): URL for an API endpoint with valid parameters.
            session (requests.Session): Session object for making the API call.

        Returns:
            pandas.DataFrame: A dataframe containing the requested data.
        """
        response = session.get(url)
        response.raise_for_status()
        table = response.json()["resources"]
        ret_df = pd.DataFrame(table).reset_index(drop=True)
        return ret_df

    @staticmethod
    def _call_for_bytes_object(url: str, session: requests.Session) -> BytesIO:
        """Private function that calls an API endpoint and returns the data as a bytes object in memory.

        Args:
            url (str): URL for an API endpoint with valid parameters.
            session (requests.Session): Session object for making the API call.

        Returns:
            io.BytesIO: In-memory file content.
        """
        response = session.get(url)
        response.raise_for_status()
        return BytesIO(response.content)

    def __init__(
        self,
        credentials: Union[str, dict] = "config/client_credentials.json",
        root_url: str = "https://fusion.jpmorgan.com/api/v1/",
        download_folder: str = "downloads",
        *,
        log_level: int = logging.ERROR,
        log_path: str = ".",
        fs: fsspec.filesystem = None,
    ) -> None:
        """Constructor to instantiate a new Fusion object.

        Args:
            credentials (Union[str, dict]): A path to a credentials file or a dictionary containing credentials.
                Defaults to 'config/client_credentials.json'.
            root_url (str): The API root URL. Defaults to "https://fusion.jpmorgan.com/api/v1/".
            download_folder (str): The folder path where downloaded data files are saved. Defaults to "downloads".
            log_level (int): Set the logging level. Defaults to logging.ERROR.
            log_path (str): The folder path where the log is stored. Defaults to the current directory.
            fs (fsspec.filesystem): filesystem.
        """
        self._default_catalog = "common"

        self.root_url = root_url
        self.download_folder = download_folder
        Path(download_folder).mkdir(parents=True, exist_ok=True)

        if logger.hasHandlers():
            logger.handlers.clear()
        file_handler = logging.FileHandler(filename=f"{log_path}/fusion_sdk.log")
        logging.addLevelName(VERBOSE_LVL, "VERBOSE")
        stdout_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(name)s:%(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)
        logger.setLevel(log_level)

        if isinstance(credentials, FusionCredentials):
            self.credentials = credentials
        elif isinstance(credentials, str):
            self.credentials = FusionCredentials.from_file(Path(credentials))  # type: ignore
        else:
            raise ValueError("credentials must be a path to a credentials file or a dictionary")

        self.session = get_session(self.credentials, self.root_url)
        self.fs = fs if fs else get_default_fs()

    def __repr__(self) -> str:
        """Object representation to list all available methods."""
        methods = [
            method_name
            for method_name in dir(Fusion)
            if callable(getattr(Fusion, method_name)) and not method_name.startswith("_")
        ]
        return "Fusion object \nAvailable methods:\n" + tabulate(
            [[method_name] for method_name in methods],
            headers=["Method Name"],
            tablefmt="psql",
        )

    @property
    def default_catalog(self) -> str:
        """Returns the default catalog."""
        return self._default_catalog

    @default_catalog.setter
    def default_catalog(self, catalog: str) -> None:
        """Allow the default catalog, which is "common," to be overridden.

        Args:
            catalog (str): The catalog to use as the default.
        """
        self._default_catalog = catalog

    def _use_catalog(self, catalog: Optional[str]) -> str:
        """Determine which catalog to use in an API call.

        Args:
            catalog (Optional[str]): The catalog value passed as an argument to an API function wrapper.

        Returns:
            str: The catalog to use.
        """
        return catalog if catalog else self.default_catalog

    def get_fusion_filesystem(self) -> FusionHTTPFileSystem:
        """Creates Fusion Filesystem.

        Returns: Fusion Filesystem

        """
        return FusionHTTPFileSystem(client_kwargs={"root_url": self.root_url, "credentials": self.credentials})

    def list_catalogs(self, output: bool = False) -> pd.DataFrame:
        """Lists the catalogs available to the API account.

        Args:
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            class:`pandas.DataFrame`: A dataframe with a row for each catalog
        """
        url = f"{self.root_url}catalogs/"
        cat_df = Fusion._call_for_dataframe(url, self.session)

        if output:
            pass

        return cat_df

    def catalog_resources(self, catalog: Optional[str] = None, output: bool = False) -> pd.DataFrame:
        """List the resources contained within the catalog, for example products and datasets.

        Args:
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
           class:`pandas.DataFrame`: A dataframe with a row for each resource within the catalog
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}"
        cat_df = Fusion._call_for_dataframe(url, self.session)

        if output:
            pass

        return cat_df

    def list_products(
            self,
            contains: Union[str, List[str], None] = None,
            id_contains: bool = False,
            catalog: Optional[str] = None,
            output: bool = False,
            max_results: int = -1,
            display_all_columns: bool = False,
    ) -> pd.DataFrame:
        """Get the products contained in a catalog. A product is a grouping of datasets.

        Args:
            contains (Union[str, list], optional): A string or a list of strings that are product
                identifiers to filter the products list. If a list is provided then it will return
                products whose identifier matches any of the strings. Defaults to None.
            id_contains (bool): Filter datasets only where the string(s) are contained in the identifier,
                ignoring description.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.
            max_results (int, optional): Limit the number of rows returned in the dataframe.
                Defaults to -1 which returns all results.
            display_all_columns (bool, optional): If True displays all columns returned by the API,
                otherwise only the key columns are displayed

        Returns:
            class:`pandas.DataFrame`: a dataframe with a row for each product
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/products"
        full_prod_df: pd.DataFrame = Fusion._call_for_dataframe(url, self.session)

        if contains:
            if isinstance(contains, list):  # noqa: ignore
                contains = "|".join(f"{s}" for s in contains)
            if id_contains:
                filtered_df = full_prod_df[full_prod_df["identifier"].str.contains(contains, case=False)]
            else:
                filtered_df = full_prod_df[
                    full_prod_df["identifier"].str.contains(contains, case=False)
                    | full_prod_df["description"].str.contains(contains, case=False)
                    ]
        else:
            filtered_df = full_prod_df

        filtered_df["category"] = filtered_df.category.str.join(", ")
        filtered_df["region"] = filtered_df.region.str.join(", ")
        if not display_all_columns:
            filtered_df = filtered_df[
                filtered_df.columns.intersection(
                    [
                        "identifier",
                        "title",
                        "region",
                        "category",
                        "status",
                        "description",
                    ]
                )
            ]

        if max_results > -1:
            filtered_df = filtered_df[0:max_results]

        if output:
            pass

        return filtered_df

    def list_datasets(  # noqa: PLR0913
            self,
            contains: Union[str, List[str], None] = None,
            id_contains: bool = False,
            product: Union[str, List[str], None] = None,
            catalog: Optional[str] = None,
            output: bool = False,
            max_results: int = -1,
            display_all_columns: bool = False,
            status: Optional[str] = None,
            dataset_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get the datasets contained in a catalog.

        Args:
            contains (Union[str, list], optional): A string or a list of strings that are dataset
                identifiers to filter the datasets list. If a list is provided then it will return
                datasets whose identifier matches any of the strings. Defaults to None.
            id_contains (bool): Filter datasets only where the string(s) are contained in the identifier,
                ignoring description.
            product (Union[str, list], optional): A string or a list of strings that are product
                identifiers to filter the datasets list. Defaults to None.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.
            max_results (int, optional): Limit the number of rows returned in the dataframe.
                Defaults to -1 which returns all results.
            display_all_columns (bool, optional): If True displays all columns returned by the API,
                otherwise only the key columns are displayed
            status (str, optional): filter the datasets by status, default is to show all results.
            dataset_type (str, optional): filter the datasets by type, default is to show all results.

        Returns:
            class:`pandas.DataFrame`: a dataframe with a row for each dataset.
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets"
        ds_df = Fusion._call_for_dataframe(url, self.session)

        if contains:
            if isinstance(contains, list):
                contains = "|".join(f"{s}" for s in contains)
            if id_contains:
                ds_df = ds_df[ds_df["identifier"].str.contains(contains, case=False)]
            else:
                ds_df = ds_df[
                    ds_df["identifier"].str.contains(contains, case=False)
                    | ds_df["description"].str.contains(contains, case=False)
                    ]

        if product:
            url = f"{self.root_url}catalogs/{catalog}/productDatasets"
            prd_df = Fusion._call_for_dataframe(url, self.session)
            prd_df = (
                prd_df[prd_df["product"] == product]
                if isinstance(product, str)
                else prd_df[prd_df["product"].isin(product)]
            )
            ds_df = ds_df[ds_df["identifier"].str.lower().isin(prd_df["dataset"].str.lower())].reset_index(drop=True)

        if max_results > -1:
            ds_df = ds_df[0:max_results]

        ds_df["category"] = ds_df.category.str.join(", ")
        ds_df["region"] = ds_df.region.str.join(", ")
        if not display_all_columns:
            cols = [
                "identifier",
                "title",
                "containerType",
                "region",
                "category",
                "coverageStartDate",
                "coverageEndDate",
                "description",
                "status",
                "type",
            ]
            cols = [c for c in cols if c in ds_df.columns]
            ds_df = ds_df[cols]

        if status is not None:
            ds_df = ds_df[ds_df["status"] == status]

        if dataset_type is not None:
            ds_df = ds_df[ds_df["type"] == dataset_type]

        if output:
            pass

        return ds_df

    def dataset_resources(self, dataset: str, catalog: Optional[str] = None, output: bool = False) -> pd.DataFrame:
        """List the resources available for a dataset, currently this will always be a datasetseries.

        Args:
            dataset (str): A dataset identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            class:`pandas.DataFrame`: A dataframe with a row for each resource
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}"
        ds_res_df = Fusion._call_for_dataframe(url, self.session)

        if output:
            pass

        return ds_res_df

    def list_dataset_attributes(
            self,
            dataset: str,
            catalog: Optional[str] = None,
            output: bool = False,
            display_all_columns: bool = False,
    ) -> pd.DataFrame:
        """Returns the list of attributes that are in the dataset.

        Args:
            dataset (str): A dataset identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.
            display_all_columns (bool, optional): If True displays all columns returned by the API,
                otherwise only the key columns are displayed

        Returns:
            class:`pandas.DataFrame`: A dataframe with a row for each attribute
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}/attributes"
        ds_attr_df = Fusion._call_for_dataframe(url, self.session)

        if "index" in ds_attr_df.columns:
            ds_attr_df = ds_attr_df.sort_values(by="index").reset_index(drop=True)

        if not display_all_columns:
            ds_attr_df = ds_attr_df[
                ds_attr_df.columns.intersection(
                    [
                        "identifier",
                        "title",
                        "dataType",
                        "isDatasetKey",
                        "description",
                        "source",
                    ]
                )
            ]

        if output:
            pass

        return ds_attr_df

    def list_datasetmembers(
            self,
            dataset: str,
            catalog: Optional[str] = None,
            output: bool = False,
            max_results: int = -1,
    ) -> pd.DataFrame:
        """List the available members in the dataset series.

        Args:
            dataset (str): A dataset identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.
            max_results (int, optional): Limit the number of rows returned in the dataframe.
                Defaults to -1 which returns all results.

        Returns:
            class:`pandas.DataFrame`: a dataframe with a row for each dataset member.
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries"
        ds_members_df = Fusion._call_for_dataframe(url, self.session)

        if max_results > -1:
            ds_members_df = ds_members_df[0:max_results]

        if output:
            pass

        return ds_members_df

    def datasetmember_resources(
            self,
            dataset: str,
            series: str,
            catalog: Optional[str] = None,
            output: bool = False,
    ) -> pd.DataFrame:
        """List the available resources for a datasetseries member.

        Args:
            dataset (str): A dataset identifier
            series (str): The datasetseries identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            class:`pandas.DataFrame`: A dataframe with a row for each datasetseries member resource.
                Currently, this will always be distributions.
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}"
        ds_mem_res_df = Fusion._call_for_dataframe(url, self.session)

        if output:
            pass

        return ds_mem_res_df

    def list_distributions(
            self,
            dataset: str,
            series: str,
            catalog: Optional[str] = None,
            output: bool = False,
    ) -> pd.DataFrame:
        """List the available distributions (downloadable instances of the dataset with a format type).

        Args:
            dataset (str): A dataset identifier
            series (str): The datasetseries identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            class:`pandas.DataFrame`: A dataframe with a row for each distribution.
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions"
        distros_df = Fusion._call_for_dataframe(url, self.session)

        if output:
            pass

        return distros_df

    def _resolve_distro_tuples(
            self,
            dataset: str,
            dt_str: str = "latest",
            dataset_format: str = "parquet",
            catalog: Optional[str] = None,
    ) -> list:
        """Resolve distribution tuples given specification params.

        A private utility function to generate a list of distribution tuples.
        Each tuple is a distribution, identified by catalog, dataset id,
        datasetseries member id, and the file format.

        Args:
            dataset (str): A dataset identifier
            dt_str (str, optional): Either a single date or a range identified by a start or end date,
                or both separated with a ":". Defaults to 'latest' which will return the most recent
                instance of the dataset.
            dataset_format (str, optional): The file format, e.g. CSV or Parquet. Defaults to 'parquet'.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.

        Returns:
            list: a list of tuples, one for each distribution
        """
        catalog = self._use_catalog(catalog)

        datasetseries_list = self.list_datasetmembers(dataset, catalog)
        if len(datasetseries_list) == 0:
            raise AssertionError(f"There are no dataset members for dataset {dataset} in catalog {catalog}")

        if datasetseries_list.empty:
            raise APIResponseError(  # pragma: no cover
                f"No data available for dataset {dataset}. "
                f"Check that a valid dataset identifier and date/date range has been set."
            )

        if dt_str == "latest":
            dt_str = (
                datasetseries_list[
                    datasetseries_list["createdDate"] == datasetseries_list["createdDate"].to_numpy().max()
                    ]
                .sort_values(by="identifier")
                .iloc[-1]["identifier"]
            )
            datasetseries_list = datasetseries_list[datasetseries_list["identifier"] == dt_str]
        else:
            parsed_dates = normalise_dt_param_str(dt_str)
            if len(parsed_dates) == 1:
                parsed_dates = (parsed_dates[0], parsed_dates[0])

            if parsed_dates[0]:
                datasetseries_list = datasetseries_list[
                    pd.Series([pd.to_datetime(i, errors="coerce") for i in datasetseries_list["identifier"]])
                    >= pd.to_datetime(parsed_dates[0])
                    ].reset_index()

            if parsed_dates[1]:
                datasetseries_list = datasetseries_list[
                    pd.Series([pd.to_datetime(i, errors="coerce") for i in datasetseries_list["identifier"]])
                    <= pd.to_datetime(parsed_dates[1])
                    ].reset_index()

        if len(datasetseries_list) == 0:
            raise APIResponseError(  # pragma: no cover
                f"No data available for dataset {dataset} in catalog {catalog}.\n"
                f"Check that a valid dataset identifier and date/date range has been set."
            )

        required_series = list(datasetseries_list["@id"])
        tups = [(catalog, dataset, series, dataset_format) for series in required_series]

        return tups

    def download(  # noqa: PLR0912, PLR0913
            self,
            dataset: str,
            dt_str: str = "latest",
            dataset_format: str = "parquet",
            catalog: Optional[str] = None,
            n_par: Optional[int] = None,
            show_progress: bool = True,
            force_download: bool = False,
            download_folder: Optional[str] = None,
            return_paths: bool = False,
            partitioning: Optional[str] = None,
            preserve_original_name: bool = False,
    ) -> Optional[List[Tuple[bool, str, Optional[str]]]]:
        """Downloads the requested distributions of a dataset to disk.

        Args:
            dataset (str): A dataset identifier
            dt_str (str, optional): Either a single date or a range identified by a start or end date,
                or both separated with a ":". Defaults to 'latest' which will return the most recent
                instance of the dataset. If more than one series member exists on the latest date, the
                series member identifiers will be sorted alphabetically and the last one will be downloaded.
            dataset_format (str, optional): The file format, e.g. CSV or Parquet. Defaults to 'parquet'.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            n_par (int, optional): Specify how many distributions to download in parallel.
                Defaults to all cpus available.
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            force_download (bool, optional): If True then will always download a file even
                if it is already on disk. Defaults to True.
            download_folder (str, optional): The path, absolute or relative, where downloaded files are saved.
                Defaults to download_folder as set in __init__
            return_paths (bool, optional): Return paths and success statuses of the downloaded files.
            partitioning (str, optional): Partitioning specification.
            preserve_original_name (bool, optional): Preserve the original name of the file. Defaults to False.

        Returns:

        """
        catalog = self._use_catalog(catalog)

        valid_date_range = re.compile(r"^(\d{4}\d{2}\d{2})$|^((\d{4}\d{2}\d{2})?([:])(\d{4}\d{2}\d{2})?)$")

        if valid_date_range.match(dt_str) or dt_str == "latest":
            required_series = self._resolve_distro_tuples(dataset, dt_str, dataset_format, catalog)
        else:
            # sample data is limited to csv
            if dt_str == "sample":
                dataset_format = self.list_distributions(dataset, dt_str, catalog)["identifier"].iloc[0]
            required_series = [(catalog, dataset, dt_str, dataset_format)]

        if dataset_format not in RECOGNIZED_FORMATS + ["raw"]:
            raise ValueError(f"Dataset format {dataset_format} is not supported")

        if not download_folder:
            download_folder = self.download_folder

        download_folders = [download_folder] * len(required_series)

        if partitioning == "hive":
            members = [series[2].strip("/") for series in required_series]
            download_folders = [
                f"{download_folders[i]}/{series[0]}/{series[1]}/{members[i]}"
                for i, series in enumerate(required_series)
            ]

        for d in download_folders:
            if not self.fs.exists(d):
                self.fs.mkdir(d, create_parents=True)

        n_par = cpu_count(n_par)
        download_spec = [
            {
                "lfs": self.fs,
                "rpath": distribution_to_url(
                    self.root_url,
                    series[1],
                    series[2],
                    series[3],
                    series[0],
                    is_download=True,
                ),
                "lpath": distribution_to_filename(
                    download_folders[i],
                    series[1],
                    series[2],
                    series[3],
                    series[0],
                    partitioning=partitioning,
                ),
                "overwrite": force_download,
                "preserve_original_name": preserve_original_name,
            }
            for i, series in enumerate(required_series)
        ]

        logger.log(
            VERBOSE_LVL,
            f"Beginning {len(download_spec)} downloads in batches of {n_par}",
        )
        if show_progress:
            with tqdm_joblib(tqdm(total=len(download_spec))):
                res = Parallel(n_jobs=n_par)(
                    delayed(self.get_fusion_filesystem().download)(**spec) for spec in download_spec
                )
        else:
            res = Parallel(n_jobs=n_par)(
                delayed(self.get_fusion_filesystem().download)(**spec) for spec in download_spec
            )

        if (len(res) > 0) and (not all(r[0] for r in res)):
            for r in res:
                if not r[0]:
                    warnings.warn(f"The download of {r[1]} was not successful", stacklevel=2)
        return res if return_paths else None

    def to_bytes(
            self,
            dataset: str,
            series_member: str,
            dataset_format: str = "parquet",
            catalog: Optional[str] = None,
    ) -> BytesIO:
        """Returns an instance of dataset (the distribution) as a bytes object.

        Args:
            dataset (str): A dataset identifier
            series_member (str,): A dataset series member identifier
            dataset_format (str, optional): The file format, e.g. CSV or Parquet. Defaults to 'parquet'.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
        """

        catalog = self._use_catalog(catalog)

        url = distribution_to_url(
            self.root_url,
            dataset,
            series_member,
            dataset_format,
            catalog,
        )

        return Fusion._call_for_bytes_object(url, self.session)

    def upload(  # noqa: PLR0913
            self,
            path: str,
            dataset: Optional[str] = None,
            dt_str: str = "latest",
            catalog:Optional[str] = None,
            n_par: Optional[int] = None,
            show_progress: bool = True,
            return_paths: bool = False,
            multipart: bool = True,
            chunk_size: int = 5 * 2 ** 20,
            from_date: Optional[str] = None,
            to_date: Optional[str] = None,
            preserve_original_name: Optional[bool] = False,
            additional_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[List[Tuple[bool, str, Optional[str]]]]:
        """Uploads the requested files/files to Fusion.

        Args:
            path (str): path to a file or a folder with files
            dataset (str, optional): Dataset identifier to which the file will be uploaded (for single file only).
                                    If not provided the dataset will be implied from file's name.
            dt_str (str, optional): A file name. Can be any string but is usually a date.
                                    Defaults to 'latest' which will return the most recent.
                                    Relevant for a single file upload only. If not provided the dataset will
                                    be implied from file's name.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            n_par (int, optional): Specify how many distributions to download in parallel.
                Defaults to all cpus available.
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            return_paths (bool, optional): Return paths and success statuses of the downloaded files.
            multipart (bool, optional): Is multipart upload.
            chunk_size (int, optional): Maximum chunk size.
            from_date (str, optional): start of the data date range contained in the distribution,
                defaults to upoad date
            to_date (str, optional): end of the data date range contained in the distribution,
                defaults to upload date.
            preserve_original_name (bool, optional): Preserve the original name of the file. Defaults to False.

        Returns:


        """
        catalog = self._use_catalog(catalog)

        if not self.fs.exists(path):
            raise RuntimeError("The provided path does not exist")

        fs_fusion = self.get_fusion_filesystem()
        if self.fs.info(path)["type"] == "directory":
            file_path_lst = self.fs.find(path)
            local_file_validation = validate_file_names(file_path_lst, fs_fusion)
            file_path_lst = [f for flag, f in zip(local_file_validation, file_path_lst) if flag]
            file_name = [f.split("/")[-1] for f in file_path_lst]
            is_raw_lst = is_dataset_raw(file_path_lst, fs_fusion)
            local_url_eqiv = [path_to_url(i, r) for i, r in zip(file_path_lst, is_raw_lst)]
        else:
            file_path_lst = [path]
            if not catalog or not dataset:
                local_file_validation = validate_file_names(file_path_lst, fs_fusion)
                file_path_lst = [f for flag, f in zip(local_file_validation, file_path_lst) if flag]
                is_raw_lst = is_dataset_raw(file_path_lst, fs_fusion)
                local_url_eqiv = [path_to_url(i, r) for i, r in zip(file_path_lst, is_raw_lst)]
                if preserve_original_name:
                    raise ValueError("preserve_original_name can only be used when catalog and dataset are provided.")
            else:
                date_identifier = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
                if date_identifier.match(dt_str):
                    dt_str = dt_str if dt_str != "latest" else pd.Timestamp("today").date().strftime("%Y%m%d")
                    dt_str = pd.Timestamp(dt_str).date().strftime("%Y%m%d")

                if catalog not in fs_fusion.ls("") or dataset not in [
                    i.split("/")[-1] for i in fs_fusion.ls(f"{catalog}/datasets")
                ]:
                    msg = (
                        f"File file has not been uploaded, one of the catalog: {catalog} "
                        f"or dataset: {dataset} does not exit."
                    )
                    warnings.warn(msg, stacklevel=2)
                    return [(False, path, msg)]
                file_format = path.split(".")[-1]
                file_name = [path.split("/")[-1]]
                file_format = "raw" if file_format not in RECOGNIZED_FORMATS else file_format

                local_url_eqiv = [
                    "/".join(distribution_to_url("", dataset, dt_str, file_format, catalog, False).split("/")[1:])
                ]

        if not preserve_original_name:
            data_map_df = pd.DataFrame([file_path_lst, local_url_eqiv]).T
            data_map_df.columns = pd.Index(["path", "url"])
        else:
            data_map_df = pd.DataFrame([file_path_lst, local_url_eqiv, file_name]).T
            data_map_df.columns = pd.Index(["path", "url", "file_name"])

        n_par = cpu_count(n_par)
        parallel = len(data_map_df) > 1
        res = upload_files(
            fs_fusion,
            self.fs,
            data_map_df,
            parallel=parallel,
            n_par=n_par,
            multipart=multipart,
            chunk_size=chunk_size,
            show_progress=show_progress,
            from_date=from_date,
            to_date=to_date,
            additional_headers=additional_headers,
        )

        if not all(r[0] for r in res):
            failed_res = [r for r in res if not r[0]]
            msg = f"Not all uploads were successfully completed. The following failed:\n{failed_res}"
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)

        return res if return_paths else None

    def from_bytes(  # noqa: PLR0913
            self,
            data: BytesIO,
            dataset: str,
            series_member: str = "latest",
            catalog: Optional[str] = None,
            distribution: str = "parquet",
            show_progress: bool = True,
            return_paths: bool = False,
            chunk_size: int = 5 * 2 ** 20,
            from_date: Optional[str] = None,
            to_date: Optional[str] = None,
            file_name: Optional[str] = None,
            **kwargs: Any,  # noqa: ARG002
    ) -> Optional[List[Tuple[bool, str, Optional[str]]]]:
        """Uploads data from an object in memory.

        Args:
            data (str): an object in memory to upload
            dataset (str): Dataset name to which the bytes will be uploaded.
            series_member (str, optional): A single date or label. Defaults to 'latest' which will return
                the most recent.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            distribution (str, optional): A distribution type, e.g. a file format or raw
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            return_paths (bool, optional): Return paths and success statuses of the downloaded files.
            chunk_size (int, optional): Maximum chunk size.
            from_date (str, optional): start of the data date range contained in the distribution,
                defaults to upload date
            to_date (str, optional): end of the data date range contained in the distribution, defaults to upload date.
            file_name (str, optional): file name to be used for the uploaded file. Defaults to Fusion standard naming.

        Returns:
            Optional[list[tuple[bool, str, Optional[str]]]: a list of tuples, one for each distribution

        """
        catalog = self._use_catalog(catalog)

        fs_fusion = self.get_fusion_filesystem()
        if distribution not in RECOGNIZED_FORMATS + ["raw"]:
            raise ValueError(f"Dataset format {distribution} is not supported")

        is_raw = js.loads(fs_fusion.cat(f"{catalog}/datasets/{dataset}"))["isRawData"]
        local_url_eqiv = path_to_url(f"{dataset}__{catalog}__{series_member}.{distribution}", is_raw)

        data_map_df = pd.DataFrame(["", local_url_eqiv, file_name]).T
        data_map_df.columns = ["path", "url", "file_name"]  # type: ignore

        res = upload_files(
            fs_fusion,
            data,
            data_map_df,
            parallel=False,
            n_par=1,
            multipart=False,
            chunk_size=chunk_size,
            show_progress=show_progress,
            from_date=from_date,
            to_date=to_date,
        )

        if not all(r[0] for r in res):
            failed_res = [r for r in res if not r[0]]
            msg = f"Not all uploads were successfully completed. The following failed:\n{failed_res}"
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)

        return res if return_paths else None

    def list_dataset_lineage(
        self,
        dataset_id: str,
        catalog: Optional[str] = None,
        output: bool = False,
        max_results: int = -1,
    ) -> pd.DataFrame:
        """List the upstream and downstream lineage of the dataset.

        Args:
            dataset_id (str): A dataset identifier.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True, then print the dataframe. Defaults to False.
            max_results (int, optional): Limit the number of rows returned in the dataframe.
                Defaults to -1, which returns all results.

        Returns:
            pd.DataFrame: A dataframe with a row for each resource.

        Raises:
            HTTPError: If the dataset is not found in the catalog.
        """
        catalog = self._use_catalog(catalog)

        url_dataset = f"{self.root_url}catalogs/{catalog}/datasets/{dataset_id}"
        resp_dataset = self.session.get(url_dataset)
        resp_dataset.raise_for_status()

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset_id}/lineage"
        resp = self.session.get(url)
        data = resp.json()
        relations_data = data["relations"]

        restricted_datasets = [
            dataset_metadata["identifier"]
            for dataset_metadata in data["datasets"]
            if dataset_metadata.get("status", None) == "Restricted"
        ]

        data_dict = {}

        for entry in relations_data:
            source_dataset_id = entry["source"]["dataset"]
            source_catalog = entry["source"]["catalog"]
            destination_dataset_id = entry["destination"]["dataset"]
            destination_catalog = entry["destination"]["catalog"]

            if destination_dataset_id == dataset_id:
                for dataset in data["datasets"]:
                    if dataset["identifier"] == source_dataset_id and dataset.get("status", None) != "Restricted":
                        source_dataset_title = dataset["title"]
                    elif dataset["identifier"] == source_dataset_id and dataset.get("status", None) == "Restricted":
                        source_dataset_title = "Access Restricted"
                data_dict[source_dataset_id] = (
                    "source",
                    source_catalog,
                    source_dataset_title,
                )

            if source_dataset_id == dataset_id:
                for dataset in data["datasets"]:
                    if dataset["identifier"] == destination_dataset_id and dataset.get("status", None) != "Restricted":
                        destination_dataset_title = dataset["title"]
                    elif (
                        dataset["identifier"] == destination_dataset_id and dataset.get("status", None) == "Restricted"
                    ):
                        destination_dataset_title = "Access Restricted"
                data_dict[destination_dataset_id] = (
                    "produced",
                    destination_catalog,
                    destination_dataset_title,
                )

        output_data = {
            "type": [v[0] for v in data_dict.values()],
            "dataset_identifier": list(data_dict.keys()),
            "title": [v[2] for v in data_dict.values()],
            "catalog": [v[1] for v in data_dict.values()],
        }

        lineage_df = pd.DataFrame(output_data)
        lineage_df.loc[
            lineage_df["dataset_identifier"].isin(restricted_datasets),
            ["dataset_identifier", "catalog", "title"],
        ] = "Access Restricted"

        if max_results > -1:
            lineage_df = lineage_df[0:max_results]

        if output:
            print(lineage_df)

        return lineage_df


    def create_dataset_lineage(
            self: Fusion,
            base_dataset: str,
            source_dataset_catalog_mapping: Union[pd.DataFrame, List[Dict[str, str]]],
            catalog: Optional[str] = None,
            return_resp_obj: bool = False,
        ) -> Optional[requests.Response]:
        """Upload lineage to a dataset.

        Args:
            base_dataset (str): A dataset identifier to which you want to add lineage.
            source_dataset_catalog_mapping (Union[pd.DataFrame, list[dict[str]]]): Mapping for the dataset
                identifier(s) and catalog(s) from which to add lineage.
            catalog (Optional[str], optional): Catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True, then return the response object. Defaults to False.

        Raises:
            ValueError: If source_dataset_catalog_mapping is not a pandas DataFrame or a list of dictionaries.
            HTTPError: If the request is unsuccessful.

        Examples:
            Creating lineage from a pandas DataFrame.
            >>> data = [{"dataset": "a", "catalog": "a"}, {"dataset": "b", "catalog": "b"}]
            >>> df = pd.DataFrame(data)
            >>> fusion = Fusion()
            >>> fusion.create_dataset_lineage(base_dataset="c", source_dataset_catalog_mapping=df, catalog="c")

            Creating lineage from a list of dictionaries.
            >>> data = [{"dataset": "a", "catalog": "a"}, {"dataset": "b", "catalog": "b"}]
            >>> fusion = Fusion()
            >>> fusion.create_dataset_lineage(base_dataset="c", source_dataset_catalog_mapping=data, catalog="c")

        """
        catalog = self._use_catalog(catalog)

        if isinstance(source_dataset_catalog_mapping, pd.DataFrame):
            dataset_mapping_list = [
                {"dataset": row["dataset"], "catalog": row["catalog"]}
                for _, row in source_dataset_catalog_mapping.iterrows()
            ]
        elif isinstance(source_dataset_catalog_mapping, list):
            dataset_mapping_list = source_dataset_catalog_mapping
        else:
            raise ValueError("source_dataset_catalog_mapping must be a pandas DataFrame or a list of dictionaries.")

        data = {"source": dataset_mapping_list}

        url = f"{self.root_url}catalogs/{catalog}/datasets/{base_dataset}/lineage"

        resp = self.session.post(url, json=data)
        resp.raise_for_status()

        return resp if return_resp_obj else None

    def list_product_dataset_mapping(
            self: Fusion,
            dataset: Union[str, List[str], None] = None,
            product: Union[str, List[str], None] = None,
            catalog: Union[str, None] = None,
        ) -> pd.DataFrame:
            """Get the product to dataset linking contained in a catalog. A product is a grouping of datasets.

            Args:
                dataset (Union[str, List[str], None], optional): A string or list of strings that are dataset
                identifiers to filter the output. If a list is provided then it will return
                datasets whose identifier matches any of the strings. Defaults to None.
                product (Union[str, List[str], None], optional): A string or list of strings that are product
                identifiers to filter the output. If a list is provided then it will return
                products whose identifier matches any of the strings. Defaults to None.
                catalog (Union[str, None], optional): A catalog identifier. Defaults to 'common'.

            Returns:
                pd.DataFrame: A dataframe with a row for each dataset to product mapping.
            """
            catalog = self._use_catalog(catalog)
            url = f"{self.root_url}catalogs/{catalog}/productDatasets"
            mapping_df = pd.DataFrame(self._call_for_dataframe(url, self.session))

            if dataset:
                if isinstance(dataset, list):
                    contains = "|".join(f"{s}" for s in dataset)
                    mapping_df = mapping_df[mapping_df["dataset"].str.contains(contains, case=False)]
                if isinstance(dataset, str):
                    mapping_df = mapping_df[mapping_df["dataset"].str.contains(dataset, case=False)]
            if product:
                if isinstance(product, list):
                    contains = "|".join(f"{s}" for s in product)
                    mapping_df = mapping_df[mapping_df["product"].str.contains(contains, case=False)]
                if isinstance(product, str):
                    mapping_df = mapping_df[mapping_df["product"].str.contains(product, case=False)]
            return mapping_df

    def product(
            self: Fusion,
            identifier: str,
            title: str = "",
            category: Union[str, List[str], None] = None,
            short_abstract: str = "",
            description: str = "",
            is_active: bool = True,
            is_restricted: Union[bool, None] = None,
            maintainer: Union[str, List[str], None] = None,
            region: Union[str, List[str]] = "Global",
            publisher: str = "J.P. Morgan",
            sub_category: Union[str, List[str], None] = None,
            tag: Union[str, List[str], None] = None,
            delivery_channel: Union[str, List[str]] = "API",
            theme: Union[str, None] = None,
            release_date: Union[str, None] = None,
            language: str = "English",
            status: str = "Available",
            image: str = "",
            logo: str = "",
            dataset: Union[str, List[str], None] = None,
            **kwargs: Any,
        ) -> Product:
            """Instantiate a Product object with this client for metadata creation."""
            product_obj = Product(
                identifier=identifier,
                title=title,
                category=category,
                short_abstract=short_abstract,
                description=description,
                is_active=is_active,
                is_restricted=is_restricted,
                maintainer=maintainer,
                region=region,
                publisher=publisher,
                sub_category=sub_category,
                tag=tag,
                delivery_channel=delivery_channel,
                theme=theme,
                release_date=release_date,
                language=language,
                status=status,
                image=image,
                logo=logo,
                dataset=dataset,
                **kwargs,
            )
            product_obj.client = self
            return product_obj

    def dataset(
            self: Fusion,
            identifier: str,
            title: str = "",
            category: Union[str, List[str], None] = None,
            description: str = "",
            frequency: str = "Once",
            is_internal_only_dataset: bool = False,
            is_third_party_data: bool = True,
            is_restricted: Union[bool, None] = None,
            is_raw_data: bool = True,
            maintainer: Union[str, None] = "J.P. Morgan Fusion",
            source: Union[str, List[str], None] = None,
            region: Union[str, List[str], None] = None,
            publisher: str = "J.P. Morgan",
            product: Union[str, List[str], None] = None,
            sub_category: Union[str, List[str], None] = None,
            tags: Union[str, List[str], None] = None,
            created_date: Union[str, None] = None,
            modified_date: Union[str, None] = None,
            delivery_channel: Union[str, List[str]] = "API",
            language: str = "English",
            status: str = "Available",
            type_: Union[str, None] = "Source",
            container_type: Union[str, None] = "Snapshot-Full",
            snowflake: Union[str, None] = None,
            complexity: Union[str, None] = None,
            is_immutable: Union[bool, None] = None,
            is_mnpi: Union[bool, None] = None,
            is_pci: Union[bool, None] = None,
            is_pii: Union[bool, None] = None,
            is_client: Union[bool, None] = None,
            is_public: Union[bool, None] = None,
            is_internal: Union[bool, None] = None,
            is_confidential: Union[bool, None] = None,
            is_highly_confidential: Union[bool, None] = None,
            is_active: Union[bool, None] = None,
            owners: Union[List[str], None] = None,
            application_id: Union[str, Dict[str, str], None] = None,
            **kwargs: Any,
        ) -> Dataset:
            """Instantiate a Dataset object with this client for metadata creation."""
            dataset_obj = Dataset(
                identifier=identifier,
                title=title,
                category=category,
                description=description,
                frequency=frequency,
                is_internal_only_dataset=is_internal_only_dataset,
                is_third_party_data=is_third_party_data,
                is_restricted=is_restricted,
                is_raw_data=is_raw_data,
                maintainer=maintainer,
                source=source,
                region=region,
                publisher=publisher,
                product=product,
                sub_category=sub_category,
                tags=tags,
                created_date=created_date,
                modified_date=modified_date,
                delivery_channel=delivery_channel,
                language=language,
                status=status,
                type_=type_,
                container_type=container_type,
                snowflake=snowflake,
                complexity=complexity,
                is_immutable=is_immutable,
                is_mnpi=is_mnpi,
                is_pci=is_pci,
                is_pii=is_pii,
                is_client=is_client,
                is_public=is_public,
                is_internal=is_internal,
                is_confidential=is_confidential,
                is_highly_confidential=is_highly_confidential,
                is_active=is_active,
                owners=owners,
                application_id=application_id,
                **kwargs,
            )
            dataset_obj.client = self
            return dataset_obj


    def attribute(  # noqa: PLR0913
            self,
            identifier: str,
            index: int,
            data_type: Union[str, Types] = "String",
            title: str = "",
            description: str = "",
            is_dataset_key: bool = False,
            source: Optional[str] = None,
            source_field_id: Optional[str] = None,
            is_internal_dataset_key: Optional[bool] = None,
            is_externally_visible: Optional[bool] = True,
            unit: Optional[Any] = None,
            multiplier: float = 1.0,
            is_propagation_eligible: Optional[bool] = None,
            is_metric: Optional[bool] = None,
            available_from: Optional[str] = None,
            deprecated_from: Optional[str] = None,
            term: str = "bizterm1",
            dataset: Optional[int] = None,
            attribute_type: Optional[str] = None,
            application_id: Optional[Union[str, dict]] = None,
            **kwargs: Any,
        ) -> Attribute:
        data_type = Types[str(data_type).strip().rsplit(".", maxsplit=1)[-1].title()]
        attribute_obj = Attribute(
            identifier=identifier,
            index=index,
            data_type=data_type,
            title=title,
            description=description,
            is_dataset_key=is_dataset_key,
            source=source,
            source_field_id=source_field_id,
            is_internal_dataset_key=is_internal_dataset_key,
            is_externally_visible=is_externally_visible,
            unit=unit,
            multiplier=multiplier,
            is_propagation_eligible=is_propagation_eligible,
            is_metric=is_metric,
            available_from=available_from,
            deprecated_from=deprecated_from,
            term=term,
            dataset=dataset,
            attribute_type=attribute_type,
            application_id=application_id,
            **kwargs,
        )
        attribute_obj.client = self
        return attribute_obj

    def attributes(
            self,
            attributes: Optional[List[Attribute]] = None,
        ) -> Attributes:
        attributes_obj = Attributes(attributes=attributes or [])
        attributes_obj.client = self
        return attributes_obj

    def delete_datasetmembers(
            self,
            dataset: str,
            series_members: Union[str, List[str]],
            catalog: Optional[str] = None,
            return_resp_obj: bool = False,
        ) -> Optional[List[requests.Response]]:
        catalog = self._use_catalog(catalog)
        if isinstance(series_members, str):
            series_members = [series_members]
        responses = []
        for series_member in series_members:
            url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series_member}"
            resp = self.session.delete(url)
            requests_raise_for_status(resp)
            responses.append(resp)
        return responses if return_resp_obj else None

    def delete_all_datasetmembers(
            self,
            dataset: str,
            catalog: Optional[str] = None,
            return_resp_obj: bool = False,
        ) -> Optional[requests.Response]:
        catalog = self._use_catalog(catalog)
        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries"
        resp = self.session.delete(url)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def list_registered_attributes(
            self,
            catalog: Optional[str] = None,
            output: bool = False,
            display_all_columns: bool = False,
        ) -> pd.DataFrame:
        catalog = self._use_catalog(catalog)
        url = f"{self.root_url}catalogs/{catalog}/attributes"
        ds_attr_df = Fusion._call_for_dataframe(url, self.session).reset_index(drop=True)
        if not display_all_columns:
            ds_attr_df = ds_attr_df[
                ds_attr_df.columns.intersection(
                    [
                        "identifier",
                        "title",
                        "dataType",
                        "description",
                        "publisher",
                        "applicationId",
                    ]
                )
            ]
        if output:
            pass
        return ds_attr_df

    def report(  # noqa: PLR0913
            self,
            identifier: str,
            title: str = "",
            category: Union[str, List[str], None] = None,
            description: str = "",
            frequency: str = "Once",
            is_internal_only_dataset: bool = False,
            is_third_party_data: bool = True,
            is_restricted: Optional[bool] = None,
            is_raw_data: bool = True,
            maintainer: Optional[str] = "J.P. Morgan Fusion",
            source: Union[str, List[str], None] = None,
            region: Union[str, List[str], None] = None,
            publisher: str = "J.P. Morgan",
            product: Union[str, List[str], None] = None,
            sub_category: Union[str, List[str], None] = None,
            tags: Union[str, List[str], None] = None,
            created_date: Optional[str] = None,
            modified_date: Optional[str] = None,
            delivery_channel: Union[str, List[str]] = "API",
            language: str = "English",
            status: str = "Available",
            type_: Optional[str] = "Report",
            container_type: Optional[str] = "Snapshot-Full",
            snowflake: Optional[str] = None,
            complexity: Optional[str] = None,
            is_immutable: Optional[bool] = None,
            is_mnpi: Optional[bool] = None,
            is_pci: Optional[bool] = None,
            is_pii: Optional[bool] = None,
            is_client: Optional[bool] = None,
            is_public: Optional[bool] = None,
            is_internal: Optional[bool] = None,
            is_confidential: Optional[bool] = None,
            is_highly_confidential: Optional[bool] = None,
            is_active: Optional[bool] = None,
            owners: Optional[List[str]] = None,
            application_id: Union[str, Dict[str, str], None] = None,
            report: Optional[Dict[str, str]] = None,
            **kwargs: Any,
        ) -> Report:
            """Instantiate Report object with this client for metadata creation
            for managing regulatory reporting metadata.

            Args:
                identifier (str): Dataset identifier.
                title (str, optional): Dataset title. If not provided, defaults to identifier.
                category (Union[str, List[str], None], optional): A category or list of categories for the dataset.
                    Defaults to None.
                description (str, optional): Dataset description. If not provided, defaults to identifier.
                frequency (str, optional): The frequency of the dataset. Defaults to "Once".
                is_internal_only_dataset (bool, optional): Flag for internal datasets. Defaults to False.
                is_third_party_data (bool, optional): Flag for third party data. Defaults to True.
                is_restricted (Optional[bool], optional): Flag for restricted datasets. Defaults to None.
                is_raw_data (bool, optional): Flag for raw datasets. Defaults to True.
                maintainer (Optional[str], optional): Dataset maintainer. Defaults to "J.P. Morgan Fusion".
                source (Union[str, List[str], None], optional): Name of data vendor which provided the data.
                    Defaults to None.
                region (Union[str, List[str], None], optional): Region. Defaults to None.
                publisher (str, optional): Name of vendor that publishes the data. Defaults to "J.P. Morgan".
                product (Union[str, List[str], None], optional): Product to associate dataset with. Defaults to None.
                sub_category (Union[str, List[str], None], optional): Sub-category. Defaults to None.
                tags (Union[str, List[str], None], optional): Tags used for search purposes. Defaults to None.
                created_date (Optional[str], optional): Created date. Defaults to None.
                modified_date (Optional[str], optional): Modified date. Defaults to None.
                delivery_channel (Union[str, List[str]], optional): Delivery channel. Defaults to "API".
                language (str, optional): Language. Defaults to "English".
                status (str, optional): Status. Defaults to "Available".
                type_ (Optional[str], optional): Dataset type. Defaults to "Source".
                container_type (Optional[str], optional): Container type. Defaults to "Snapshot-Full".
                snowflake (Optional[str], optional): Snowflake account connection. Defaults to None.
                complexity (Optional[str], optional): Complexity. Defaults to None.
                is_immutable (Optional[bool], optional): Flag for immutable datasets. Defaults to None.
                is_mnpi (Optional[bool], optional): is_mnpi. Defaults to None.
                is_pci (Optional[bool], optional): is_pci. Defaults to None.
                is_pii (Optional[bool], optional): is_pii. Defaults to None.
                is_client (Optional[bool], optional): is_client. Defaults to None.
                is_public (Optional[bool], optional): is_public. Defaults to None.
                is_internal (Optional[bool], optional): is_internal. Defaults to None.
                is_confidential (Optional[bool], optional): is_confidential. Defaults to None.
                is_highly_confidential (Optional[bool], optional): is_highly_confidential. Defaults to None.
                is_active (Optional[bool], optional): is_active. Defaults to None.
                owners (Optional[List[str]], optional): The owners of the dataset. Defaults to None.
                application_id (Union[str, Dict[str, str], None], optional): The application ID of the dataset.
                    Defaults to None.
                report (Optional[Dict[str, str]], optional): The report metadata. Specifies the tier of the report.
                    Required for registered reports to the catalog.

            Returns:
                Report: Fusion Report class.

            Examples:
                >>> from fusion37 import Fusion
                >>> fusion = Fusion()
                >>> dataset = fusion.report(identifier="DATASET_1")

            Note:
                See the dataset module for more information on functionalities of report objects.

            """
            report_obj = Report(
                identifier=identifier,
                title=title,
                category=category,
                description=description,
                frequency=frequency,
                is_internal_only_dataset=is_internal_only_dataset,
                is_third_party_data=is_third_party_data,
                is_restricted=is_restricted,
                is_raw_data=is_raw_data,
                maintainer=maintainer,
                source=source,
                region=region,
                publisher=publisher,
                product=product,
                sub_category=sub_category,
                tags=tags,
                created_date=created_date,
                modified_date=modified_date,
                delivery_channel=delivery_channel,
                language=language,
                status=status,
                type_=type_,
                container_type=container_type,
                snowflake=snowflake,
                complexity=complexity,
                is_immutable=is_immutable,
                is_mnpi=is_mnpi,
                is_pci=is_pci,
                is_pii=is_pii,
                is_client=is_client,
                is_public=is_public,
                is_internal=is_internal,
                is_confidential=is_confidential,
                is_highly_confidential=is_highly_confidential,
                is_active=is_active,
                owners=owners,
                application_id=application_id,
                report=report,
                **kwargs,
            )
            report_obj.client = self
            return report_obj

    def input_dataflow(  # noqa: PLR0913
        self,
        identifier: str,
        title: str = "",
        category: Union[str, List[str], None] = None,
        description: str = "",
        frequency: str = "Once",
        is_internal_only_dataset: bool = False,
        is_third_party_data: bool = True,
        is_restricted: Optional[bool] = None,
        is_raw_data: bool = True,
        maintainer: Optional[str] = "J.P. Morgan Fusion",
        source: Union[str, List[str], None] = None,
        region: Union[str, List[str], None] = None,
        publisher: str = "J.P. Morgan",
        product: Union[str, List[str], None] = None,
        sub_category: Union[str, List[str], None] = None,
        tags: Union[str, List[str], None] = None,
        created_date: Optional[str] = None,
        modified_date: Optional[str] = None,
        delivery_channel: Union[str, List[str]] = "API",
        language: str = "English",
        status: str = "Available",
        type_: Optional[str] = "Flow",
        container_type: Optional[str] = "Snapshot-Full",
        snowflake: Optional[str] = None,
        complexity: Optional[str] = None,
        is_immutable: Optional[bool] = None,
        is_mnpi: Optional[bool] = None,
        is_pci: Optional[bool] = None,
        is_pii: Optional[bool] = None,
        is_client: Optional[bool] = None,
        is_public: Optional[bool] = None,
        is_internal: Optional[bool] = None,
        is_confidential: Optional[bool] = None,
        is_highly_confidential: Optional[bool] = None,
        is_active: Optional[bool] = None,
        owners: Optional[List[str]] = None,
        application_id: Union[str, Dict[str, str], None] = None,
        producer_application_id: Optional[Dict[str, str]] = None,
        consumer_application_id: Union[List[Dict[str, str]], Dict[str, str], None] = None,
        flow_details: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> InputDataFlow:
        flow_details = {"flowDirection": "Input"} if flow_details is None else flow_details
        dataflow_obj = InputDataFlow(
            identifier=identifier,
            title=title,
            category=category,
            description=description,
            frequency=frequency,
            is_internal_only_dataset=is_internal_only_dataset,
            is_third_party_data=is_third_party_data,
            is_restricted=is_restricted,
            is_raw_data=is_raw_data,
            maintainer=maintainer,
            source=source,
            region=region,
            publisher=publisher,
            product=product,
            sub_category=sub_category,
            tags=tags,
            created_date=created_date,
            modified_date=modified_date,
            delivery_channel=delivery_channel,
            language=language,
            status=status,
            type_=type_,
            container_type=container_type,
            snowflake=snowflake,
            complexity=complexity,
            is_immutable=is_immutable,
            is_mnpi=is_mnpi,
            is_pci=is_pci,
            is_pii=is_pii,
            is_client=is_client,
            is_public=is_public,
            is_internal=is_internal,
            is_confidential=is_confidential,
            is_highly_confidential=is_highly_confidential,
            is_active=is_active,
            owners=owners,
            application_id=application_id,
            producer_application_id=producer_application_id,
            consumer_application_id=consumer_application_id,
            flow_details=flow_details,
            **kwargs,
        )
        dataflow_obj.client = self
        return dataflow_obj

    def output_dataflow(  # noqa: PLR0913
        self,
        identifier: str,
        title: str = "",
        category: Union[str, List[str], None] = None,
        description: str = "",
        frequency: str = "Once",
        is_internal_only_dataset: bool = False,
        is_third_party_data: bool = True,
        is_restricted: Optional[bool] = None,
        is_raw_data: bool = True,
        maintainer: Optional[str] = "J.P. Morgan Fusion",
        source: Union[str, List[str], None] = None,
        region: Union[str, List[str], None] = None,
        publisher: str = "J.P. Morgan",
        product: Union[str, List[str], None] = None,
        sub_category: Union[str, List[str], None] = None,
        tags: Union[str, List[str], None] = None,
        created_date: Optional[str] = None,
        modified_date: Optional[str] = None,
        delivery_channel: Union[str, List[str]] = "API",
        language: str = "English",
        status: str = "Available",
        type_: Optional[str] = "Flow",
        container_type: Optional[str] = "Snapshot-Full",
        snowflake: Optional[str] = None,
        complexity: Optional[str] = None,
        is_immutable: Optional[bool] = None,
        is_mnpi: Optional[bool] = None,
        is_pci: Optional[bool] = None,
        is_pii: Optional[bool] = None,
        is_client: Optional[bool] = None,
        is_public: Optional[bool] = None,
        is_internal: Optional[bool] = None,
        is_confidential: Optional[bool] = None,
        is_highly_confidential: Optional[bool] = None,
        is_active: Optional[bool] = None,
        owners: Optional[List[str]] = None,
        application_id: Union[str, Dict[str, str], None] = None,
        producer_application_id: Optional[Dict[str, str]] = None,
        consumer_application_id: Union[List[Dict[str, str]], Dict[str, str], None] = None,
        flow_details: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> OutputDataFlow:
        flow_details = {"flowDirection": "Output"} if flow_details is None else flow_details
        dataflow_obj = OutputDataFlow(
            identifier=identifier,
            title=title,
            category=category,
            description=description,
            frequency=frequency,
            is_internal_only_dataset=is_internal_only_dataset,
            is_third_party_data=is_third_party_data,
            is_restricted=is_restricted,
            is_raw_data=is_raw_data,
            maintainer=maintainer,
            source=source,
            region=region,
            publisher=publisher,
            product=product,
            sub_category=sub_category,
            tags=tags,
            created_date=created_date,
            modified_date=modified_date,
            delivery_channel=delivery_channel,
            language=language,
            status=status,
            type_=type_,
            container_type=container_type,
            snowflake=snowflake,
            complexity=complexity,
            is_immutable=is_immutable,
            is_mnpi=is_mnpi,
            is_pci=is_pci,
            is_pii=is_pii,
            is_client=is_client,
            is_public=is_public,
            is_internal=is_internal,
            is_confidential=is_confidential,
            is_highly_confidential=is_highly_confidential,
            is_active=is_active,
            owners=owners,
            application_id=application_id,
            producer_application_id=producer_application_id,
            consumer_application_id=consumer_application_id,
            flow_details=flow_details,
            **kwargs,
        )
        dataflow_obj.client = self
        return dataflow_obj

    def to_df(  # noqa: PLR0913
            self,
            dataset: str,
            dt_str: str = "latest",
            dataset_format: str = "parquet",
            catalog: str | None = None,
            n_par: int | None = None,
            show_progress: bool = True,
            columns: list[str] | None = None,
            filters: Any | None = None,
            force_download: bool = False,
            download_folder: str | None = None,
            dataframe_type: str = "pandas",
            **kwargs: Any,
    ) -> pd.DataFrame:
        """Gets distributions for a specified date or date range and returns the data as a dataframe.

        Args:
            dataset (str): A dataset identifier
            dt_str (str, optional): Either a single date or a range identified by a start or end date,
                or both separated with a ":". Defaults to 'latest' which will return the most recent
                instance of the dataset.
            dataset_format (str, optional): The file format, e.g. CSV or Parquet. Defaults to 'parquet'.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            n_par (int, optional): Specify how many distributions to download in parallel.
                Defaults to all cpus available.
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            columns (List, optional): A list of columns to return from a parquet file. Defaults to None
            filters (List, optional): List[Tuple] or List[List[Tuple]] or None (default)
                Rows which do not match the filter predicate will be removed from scanned data.
                Partition keys embedded in a nested directory structure will be exploited to avoid
                loading files at all if they contain no matching rows. If use_legacy_dataset is True,
                filters can only reference partition keys and only a hive-style directory structure
                is supported. When setting use_legacy_dataset to False, also within-file level filtering
                and different partitioning schemes are supported.
                More on https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
            force_download (bool, optional): If True then will always download a file even
                if it is already on disk. Defaults to False.
            download_folder (str, optional): The path, absolute or relative, where downloaded files are saved.
                Defaults to download_folder as set in __init__
            dataframe_type (str, optional): Type
        Returns:
            class:`pandas.DataFrame`: a dataframe containing the requested data.
                If multiple dataset instances are retrieved then these are concatenated first.
        """
        raise NotImplementedError("Method not implemented")

    def to_table(  # noqa: PLR0913
            self,
            dataset: str,
            dt_str: str = "latest",
            dataset_format: str = "parquet",
            catalog: str | None = None,
            n_par: int | None = None,
            show_progress: bool = True,
            columns: list[str] | None = None,
            filters: Any | None = None,
            force_download: bool = False,
            download_folder: str | None = None,
            **kwargs: Any,
    ) -> Any:
        """Gets distributions for a specified date or date range and returns the data as an arrow table.

        Args:
            dataset (str): A dataset identifier
            dt_str (str, optional): Either a single date or a range identified by a start or end date,
                or both separated with a ":". Defaults to 'latest' which will return the most recent
                instance of the dataset.
            dataset_format (str, optional): The file format, e.g. CSV or Parquet. Defaults to 'parquet'.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            n_par (int, optional): Specify how many distributions to download in parallel.
                Defaults to all cpus available.
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            columns (List, optional): A list of columns to return from a parquet file. Defaults to None
            filters (List, optional): List[Tuple] or List[List[Tuple]] or None (default)
                Rows which do not match the filter predicate will be removed from scanned data.
                Partition keys embedded in a nested directory structure will be exploited to avoid
                loading files at all if they contain no matching rows. If use_legacy_dataset is True,
                filters can only reference partition keys and only a hive-style directory structure
                is supported. When setting use_legacy_dataset to False, also within-file level filtering
                and different partitioning schemes are supported.
                More on https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
            force_download (bool, optional): If True then will always download a file even
                if it is already on disk. Defaults to False.
            download_folder (str, optional): The path, absolute or relative, where downloaded files are saved.
                Defaults to download_folder as set in __init__
        Returns:
            class:`pyarrow.Table`: a dataframe containing the requested data.
                If multiple dataset instances are retrieved then these are concatenated first.
        """
        raise NotImplementedError("Method not implemented")

    def listen_to_events(
            self,
            last_event_id: str | None = None,
            catalog: str | None = None,
            url: str = "https://fusion.jpmorgan.com/api/v1/",
    ) -> None | pd.DataFrame:
        """Run server sent event listener in the background. Retrieve results by running get_events.

        Args:
            last_event_id (str): Last event ID (exclusive).
            catalog (str): catalog.
            url (str): subscription url.
        Returns:
            Union[None, class:`pandas.DataFrame`]: If in_background is True then the function returns no output.
                If in_background is set to False then pandas DataFrame is output upon keyboard termination.
        """
        raise NotImplementedError("Method not implemented")

    def get_events(
            self,
            last_event_id: str | None = None,
            catalog: str | None = None,
            in_background: bool = True,
            url: str = "https://fusion.jpmorgan.com/api/v1/",
    ) -> None | pd.DataFrame:
        """Run server sent event listener and print out the new events. Keyboard terminate to stop.

        Args:
            last_event_id (str): id of the last event.
            catalog (str): catalog.
            in_background (bool): execute event monitoring in the background (default = True).
            url (str): subscription url.
        Returns:
            Union[None, class:`pandas.DataFrame`]: If in_background is True then the function returns no output.
                If in_background is set to False then pandas DataFrame is output upon keyboard termination.
        """
        raise NotImplementedError("Method not implemented")