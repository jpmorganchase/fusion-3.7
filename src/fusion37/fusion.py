"""Python 3.7 SDK for J.P. Morgan's Fusion platform."""
import json as js
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests
from tabulate import tabulate

from .credentials import FusionCredentials
from .utils import get_session, normalise_dt_param_str
from .exceptions import *

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
        log_level: int = logging.ERROR,
        log_path: str = ".",
    ) -> None:
        """Constructor to instantiate a new Fusion object.

        Args:
            credentials (Union[str, dict]): A path to a credentials file or a dictionary containing credentials.
                Defaults to 'config/client_credentials.json'.
            root_url (str): The API root URL. Defaults to "https://fusion.jpmorgan.com/api/v1/".
            download_folder (str): The folder path where downloaded data files are saved. Defaults to "downloads".
            log_level (int): Set the logging level. Defaults to logging.ERROR.
            log_path (str): The folder path where the log is stored. Defaults to the current directory.
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
            self.credentials = FusionCredentials.from_file(Path(credentials))
        else:
            raise ValueError("credentials must be a path to a credentials file or a dictionary")

        self.session = get_session(self.credentials, self.root_url)

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
            contains: Union[str,list,None] = None,
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
            contains: Union[str, list, None] = None,
            id_contains: bool = False,
            product: Union[str, list, None] = None,
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
        ds_attr_df = Fusion._call_for_dataframe(url, self.session).sort_values(by="index").reset_index(drop=True)

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
