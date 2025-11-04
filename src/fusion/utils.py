

"""Fusion utilities."""

from __future__ import annotations

import contextlib
import json as js
import logging
import multiprocessing as mp
import os
import re
import ssl
import zipfile
from contextlib import nullcontext
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast
from urllib.parse import urlparse, urlunparse

import aiohttp
import certifi
import fsspec
import joblib
import pandas as pd
import requests
from dateutil import parser
from requests import Session
from tqdm import tqdm
from urllib3.util.retry import Retry

from fusion.exceptions import APIResponseError

from .authentication import FusionAiohttpSession, FusionOAuthAdapter

if TYPE_CHECKING:
    from collections.abc import Generator

    import pyarrow as pa

    from .credentials import FusionCredentials


logger = logging.getLogger(__name__)
VERBOSE_LVL = 25
DT_YYYYMMDD_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
DT_YYYYMMDDTHHMM_RE = re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{4})$")
DT_YYYY_MM_DD_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
DEFAULT_CHUNK_SIZE = 2**16
DEFAULT_THREAD_POOL_SIZE = 5
RECOGNIZED_FORMATS = [
    "csv",
    "parquet",
    "psv",
    "json",
    "pdf",
    "txt",
    "doc",
    "docx",
    "htm",
    "html",
    "xls",
    "xlsx",
    "xlsm",
    "dot",
    "dotx",
    "docm",
    "dotm",
    "rtf",
    "odt",
    "xltx",
    "xlsb",
    "jpg",
    "jpeg",
    "bmp",
    "png",
    "tif",
    "gif",
    "mp3",
    "wav",
    "mp4",
    "mov",
    "mkv",
    "gz",
    "xml",
]

re_str_1 = re.compile("(.)([A-Z][a-z]+)")
re_str_2 = re.compile("([a-z0-9])([A-Z])")


def get_default_fs() -> fsspec.filesystem:
    """Retrieve default filesystem.

    Returns: filesystem

    """
    protocol = os.environ.get("FS_PROTOCOL", "file")
    if "S3_ENDPOINT" in os.environ and "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
        endpoint = os.environ["S3_ENDPOINT"]
        fs = fsspec.filesystem(
            "s3",
            client_kwargs={"endpoint_url": f"https://{endpoint}"},
            key=os.environ["AWS_ACCESS_KEY_ID"],
            secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
    else:
        fs = fsspec.filesystem(protocol)
    return fs


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> Generator[tqdm, None, None]:  # type: ignore
    # pragma: no cover
    """Progress bar sensitive to exceptions during the batch processing.

    Args:
        tqdm_object (tqdm.tqdm): tqdm object.

    Yields: tqdm.tqdm object

    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):  # type: ignore
        """Tqdm execution wrapper."""

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            n = 0
            lst = args[0]._result if hasattr(args[0], "_result") else args[0]
            for i in lst:
                try:
                    if i[0] is True:
                        n += 1
                except Exception as _:  # noqa: F841, PERF203, BLE001
                    n += 1
            tqdm_object.update(n=n)
            return super().__call__(*args, **kwargs)

    old_batch_callback: type[joblib.parallel.BatchCompletionCallBack] = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def cpu_count(thread_pool_size: Optional[int] = None, is_threading: bool = False) -> int:
    """Determine the number of cpus/threads for parallelization.

    Args:
        thread_pool_size (int): override argument for number of cpus/threads.
        is_threading: use threads instead of CPUs

    Returns: number of cpus/threads to use.

    """
    if os.environ.get("NUM_THREADS") is not None:
        return int(os.environ["NUM_THREADS"])
    if thread_pool_size:
        return thread_pool_size
    if is_threading:
        return 10

    thread_pool_size = mp.cpu_count() if mp.cpu_count() else DEFAULT_THREAD_POOL_SIZE
    return thread_pool_size

def _normalise_dt_param(dt: str | int | datetime | date) -> str:
    """Convert dates into a normalised string representation.

    Args:
        dt (Union[str, int, datetime, date]): A date represented in various types.

    Returns:
        str: A normalized date string.
    """

    if isinstance(dt, (date, datetime)):
        return dt.strftime("%Y-%m-%d")

    if isinstance(dt, int):
        dt = str(dt)

    if not isinstance(dt, str):
        raise ValueError(f"{dt} is not in a recognised data format")

    dt_clean = re.sub(r"[ \-:T]", "", dt)

    for candidate in (dt, dt_clean):
        try:
            ts = pd.to_datetime(candidate, errors="raise")
        except (ValueError, TypeError):
            continue
        if ts.time() != datetime.min.time():
            return ts.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            return ts.strftime("%Y-%m-%d")

    raise ValueError(f"{dt} is not in a recognised data format")

def normalise_dt_param_str(dt: str) -> Tuple[str, ...]:
    """Convert a date parameter which may be a single date or a date range into a tuple.

    Args:
        dt (str): Either a single date or a date range separated by a ":".

    Returns:
        tuple: A tuple of dates.
    """
    date_parts = dt.split(":")
    max_date_seg_cnt = 2
    if not date_parts or len(date_parts) > max_date_seg_cnt:
        raise ValueError(f"Unable to parse {dt} as either a date or an interval")

    return tuple(_normalise_dt_param(dt_part) if dt_part else dt_part for dt_part in date_parts)


def distribution_to_filename(
    root_folder: str,
    dataset: str,
    datasetseries: str,
    file_format: str,
    catalog: str = "common",
    partitioning: Optional[str] = None,
    file_name: Optional[str] = None,
) -> str:
    """Returns a filename representing a dataset distribution.

    Args:
        root_folder (str): The root folder path.
        dataset (str): The dataset identifier.
        datasetseries (str): The datasetseries instance identifier.
        file_format (str): The file type, e.g. CSV or Parquet
        catalog (str, optional): The data catalog containing the dataset. Defaults to "common".
        partitioning (str, optional): Partitioning specification.
        file_name (str, optional): Explicit filename override.  Defaults to None.

    Returns:
        str: FQ distro file name
    """
    if datasetseries[-1] == "/" or datasetseries[-1] == "\\":
        datasetseries = datasetseries[0:-1]

    if file_name and file_name != file_format:
        # Use explicit filename directly
        final_name = f"{file_name}.{file_format}"
    elif partitioning == "hive":        
        final_name = f"{dataset}.{file_format}"
    else:        
        final_name = f"{dataset}__{catalog}__{datasetseries}.{file_format}"

    sep = "/"
    if "\\" in root_folder:
        sep = "\\"
    return f"{root_folder}{sep}{final_name}"


def _filename_to_distribution(file_name: str) -> Tuple[str, str, str, str]:
    """Breaks a filename down into the components that represent a dataset distribution in the catalog.

    Args:
        file_name (str): The filename to decompose.

    Returns:
        tuple: A tuple consisting of catalog id, dataset id, datasetseries instane id, and file format (e.g. CSV).
    """
    dataset, catalog, series_format = Path(file_name).name.split("__")
    datasetseries, file_format = series_format.split(".")
    return catalog, dataset, datasetseries, file_format


def distribution_to_url(
    root_url: str,
    dataset: str,
    datasetseries: str,
    file_format: str,
    catalog: str = "common",
    is_download: bool = False,
    file_name: Optional[str] = None,
) -> str:
    """Returns the API URL to download a dataset distribution.

    Args:
        root_url (str): The base url for the API.
        dataset (str): The dataset identifier.
        datasetseries (str): The datasetseries instance identifier.
        file_format (str): The file type, e.g. CSV or Parquet
        catalog (str, optional): The data catalog containing the dataset. Defaults to "common".
        is_download (bool, optional): Is url for download
        file_name (str, optional): Explicit filename override.  Defaults to None.

    Returns:
        str: A URL for the API distribution endpoint.
    """
    if datasetseries[-1] == "/" or datasetseries[-1] == "\\":
        datasetseries = datasetseries[0:-1]

    if datasetseries == "sample":
        return f"{root_url}catalogs/{catalog}/datasets/{dataset}/sample/distributions/{file_format}"
    if is_download:
        return (
            f"{root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/"
            f"{datasetseries}/distributions/{file_format}/files/operationType/download?file={file_name}"
        )
    return (
        f"{root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/" f"{datasetseries}/distributions/{file_format}"
    )


def _get_canonical_root_url(any_url: str) -> str:
    """Get the full URL for the API endpoint.

    Args:
        any_url (str): A valid URL or URL part

    Returns:
        str: A complete root URL

    """
    url_parts = urlparse(any_url)
    root_url = urlunparse((url_parts[0], url_parts[1], "", "", "", ""))
    return root_url


async def get_client(credentials: FusionCredentials, **kwargs: Any) -> FusionAiohttpSession:  # noqa: PLR0915
    """Gets session for async.

    Args:
        credentials: Credentials.
        **kwargs: Kwargs.

    Returns:

    """

    async def on_request_start_token(session: Any, _trace_config_ctx: Any, params: Any) -> None:
        if params.url:
            params.headers.update(session.credentials.get_fusion_token_headers(str(params.url)))

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(on_request_start_token)

    if "timeout" in kwargs:
        timeout = aiohttp.ClientTimeout(total=kwargs["timeout"])
    else:
        timeout = aiohttp.ClientTimeout(total=60 * 60)  # default 60min timeout

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    session = FusionAiohttpSession(
        trace_configs=[trace_config], trust_env=True, timeout=timeout, connector=aiohttp.TCPConnector(ssl=ssl_context)
    )
    session.post_init(credentials=credentials)
    return session


def get_session(
    credentials: FusionCredentials, root_url: str, get_retries: Optional[Union[int, Retry]] = None
) -> requests.Session:
    """Create a new http session and set parameters.

    Args:
        credentials (FusionCredentials): Valid user credentials to provide an access token
        root_url (str): The URL to call.

    Returns:
        requests.Session(): A HTTP Session object

    """
    if not get_retries:
        get_retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504])
    else:
        get_retries = Retry.from_int(get_retries)
    session = requests.Session()
    if credentials.proxies:
        session.proxies.update(credentials.proxies)
    try:
        mount_url = _get_canonical_root_url(root_url)
    except Exception:  # noqa: BLE001
        mount_url = "https://"
    auth_handler = FusionOAuthAdapter(credentials, max_retries=get_retries, mount_url=mount_url)
    session.mount(mount_url, auth_handler)
    return session


def validate_file_names(paths: List[str]) -> List[bool]:
    """Validate if the file name format adheres to the standard.

    Args:
        paths (list): List of file paths.
        fs_fusion: Fusion filesystem.

    Returns (list): List of booleans.

    """
    file_names = [i.split("/")[-1].split(".")[0] for i in paths]
    validation = []
    file_seg_cnt = 3
    for i, f_n in enumerate(file_names):
        tmp = f_n.split("__")        
        if len(tmp) != file_seg_cnt or not tmp[0] or not tmp[1]:
            logger.warning(
            f"The file in {paths[i]} has a non-compliant name and will not be processed. "
            f"Please rename the file to dataset__catalog__yyyymmdd.format"
            )
            validation.append(False)
        else:
            validation.append(True)

    return validation


def is_dataset_raw(paths: list[str], fs_fusion: fsspec.AbstractFileSystem) -> list[bool]:
    """Check if the files correspond to a raw dataset.

    Args:
        paths (list): List of file paths.
        fs_fusion: Fusion filesystem.

    Returns (list): List of booleans.

    """
    file_names = [i.split("/")[-1].split(".")[0] for i in paths]
    ret = []
    is_raw = {}
    for _i, f_n in enumerate(file_names):
        tmp = f_n.split("__")
        if tmp[0] not in is_raw:
            is_raw[tmp[0]] = js.loads(fs_fusion.cat(f"{tmp[1]}/datasets/{tmp[0]}"))["isRawData"]
        ret.append(is_raw[tmp[0]])

    return ret


def path_to_url(x: str, is_raw: bool = False, is_download: bool = False) -> str:
    """Convert file name to fusion url.

    Args:
        x (str): File path.
        is_raw(bool, optional): Is the dataset raw.
        is_download(bool, optional): Is the url for download.

    Returns (str): Fusion url string.

    """
    catalog, dataset, date, ext = _filename_to_distribution(x.split("/")[-1])
    ext = "raw" if is_raw and ext not in RECOGNIZED_FORMATS else ext
    return "/".join(distribution_to_url("", dataset, date, ext, catalog, is_download).split("/")[1:])


def upload_files(  # noqa: PLR0913
    fs_fusion: fsspec.AbstractFileSystem,
    fs_local: fsspec.AbstractFileSystem,
    loop: pd.DataFrame,    
    multipart: bool = True,
    chunk_size: int = 5 * 2**20,
    show_progress: bool = True,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    additional_headers: Optional[Dict[str, str]]= None,
) -> List[Tuple[Optional[Union[bool, str]]]]:
    """Upload file into Fusion.

    Args:
        fs_fusion: Fusion filesystem.
        fs_local: Local filesystem.
        loop (pd.DataFrame): DataFrame of files to iterate through.        
        multipart (bool): Is multipart upload.
        chunk_size (int): Maximum chunk size.
        show_progress (bool): Show progress bar
        from_date (str, optional): earliest date of data contained in distribution.
        to_date (str, optional): latest date of data contained in distribution.
        additional_headers (dict, optional): Additional headers to include in the request.

    Returns: List of update statuses.

    """

    def _upload(p_url: str, path: str, file_name: str | None = None) -> tuple[bool, str, str | None]:
        try:
            
            if isinstance(fs_local, BytesIO):
                fs_local.seek(0, 2)
                size_in_bytes = fs_local.tell()
                mp = multipart and size_in_bytes > chunk_size
                fs_local.seek(0)
                fs_fusion.put(
                    fs_local,
                    p_url,
                    chunk_size=chunk_size,
                    method="put",
                    multipart=mp,
                    from_date=from_date,
                    to_date=to_date,
                    file_name=file_name,
                    additional_headers=additional_headers,
                )
            else:
                mp = multipart and fs_local.size(path) > chunk_size
                with fs_local.open(path, "rb") as file_local:
                    fs_fusion.put(
                        file_local,
                        p_url,
                        chunk_size=chunk_size,
                        method="put",
                        multipart=mp,
                        from_date=from_date,
                        to_date=to_date,
                        file_name=file_name,
                        additional_headers=additional_headers,
                    )
            return (True, path, None)
        except Exception as ex:  # noqa: BLE001
            logger.log(
                VERBOSE_LVL,
                f"Failed to upload {path}.",
                exc_info=True,
            )
            return (False, path, str(ex))

    res = [None] * len(loop)
    if show_progress:
        with tqdm(total=len(loop)) as p:
            for i, (_, row) in enumerate(loop.iterrows()):
                r = _upload(row["url"], row["path"])
                res[i] = r
                if r[0] is True:
                    p.update(1)
    else:
        res = [_upload(row["url"], row["path"]) for _, row in loop.iterrows()]

    return res  # type: ignore


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s1 = re.sub(re_str_1, r"\1_\2", name)
    return re.sub(re_str_2, r"\1_\2", s1).lower()


class CamelCaseMeta(type):
    """Metaclass to support both snake and camel case typing."""

    def __new__(cls: Any, name: str, bases: Any, dct: dict[str, Any]) -> Any:
        new_namespace = {}
        annotations = dct.get("__annotations__", {})
        new_annotations = {}
        for attr_name, attr_value in dct.items():
            if not attr_name.startswith("__"):
                snake_name = camel_to_snake(attr_name)
                new_namespace[snake_name] = attr_value
            else:
                new_namespace[attr_name] = attr_value
        for anno_name, anno_type in annotations.items():
            snake_name = camel_to_snake(anno_name)
            new_annotations[snake_name] = anno_type
        new_namespace["__annotations__"] = new_annotations
        new_cls = super().__new__(cls, name, bases, new_namespace)
        return new_cls

    def __call__(cls: Any, *args: Any, **kwargs: Any) -> Any:
        # Convert keyword arguments to snake_case before initialization
        snake_kwargs = {camel_to_snake(k): v for k, v in kwargs.items()}
        return super().__call__(*args, **snake_kwargs)


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.lower().split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def tidy_string(x: str) -> str:
    """Tidy string.

    Args:
        x (str): String to tidy.

    Returns (str): Tidy string.

    """
    x = x.strip()
    x = re.sub(" +", " ", x)
    x = re.sub("/ +", "/", x)

    return x


def make_list(obj: Any) -> list[str]:
    """Make list."""
    if isinstance(obj, list):
        lst = obj
    elif isinstance(obj, str):
        lst = obj.split(",")
        for i, _ in enumerate(lst):
            lst[i] = lst[i].strip()
    else:
        lst = [cast(str, obj)]

    return lst


def make_bool(obj: Any) -> bool:
    """Make boolean."""
    if isinstance(obj, str):
        false_strings = ["F", "FALSE", "0"]
        obj = obj.strip().upper()
        if obj in false_strings:
            obj = False

    bool_obj = bool(obj)
    return bool_obj

def convert_date_format(date_str: str) -> Any:
    """Convert date to YYYY-MM-DD format."""
    if pd.isna(date_str) or not isinstance(date_str, str):
        return None

    try:
        desired_format = "%Y-%m-%d"
        date_obj = parser.parse(date_str)
        formatted_date = date_obj.strftime(desired_format)
        return formatted_date
    except(ValueError, TypeError):
        return None


def _is_json(data: str) -> bool:
    """Check if the data is in JSON format."""
    try:
        js.loads(data)
    except Exception:  # noqa: BLE001
        return False
    return True


def requests_raise_for_status(response: requests.Response) -> None:
    """Send response text into raise for status."""
   
    real_reason = ""
    try:
        real_reason = response.text
        response.reason = real_reason
    finally:
        response.raise_for_status()

def validate_file_formats(fs_fusion: fsspec.AbstractFileSystem, path: str) -> None:
    """
    Validate file formats in the given folder and subfolders.
    Raise an error if more than one raw (unrecognized) file is found.

    Args:
        fs (fsspec.AbstractFileSystem): Filesystem instance (local, S3, etc.)
        folder_path (str): Root folder path to search

    Raises:
        ValueError: If more than one raw file is found.
    """
    if not fs_fusion.exists(path):
        raise FileNotFoundError(f"The folder '{path}' does not exist.")

    all_files = [f for f in fs_fusion.find(path) if fs_fusion.info(f)["type"] == "file"]
    raw_files = [
        f for f in all_files
        if f.split(".")[-1].lower() not in RECOGNIZED_FORMATS
    ]

    if len(raw_files) > 1:
        raise ValueError(
            f"Multiple raw files detected in '{path}':\n"
            f"{chr(10).join(raw_files)}\n\n"
            f"Only one raw file is allowed. Supported formats are:\n"
            f"{', '.join(RECOGNIZED_FORMATS)}"
        )
    
def file_name_to_url(
    file_name: str, dataset: str, catalog: str, is_download: bool = False
) -> str:
    """Construct a distribution URL using the constructed file name as the series member.

    Args:
        file_name (str): Flattened file name.
        dataset (str): Dataset name.
        catalog (str): Catalog name.
        is_download (bool, optional): Whether the URL is for download. Defaults to False.

    Returns:
        str: Relative distribution URL string.
    """
    parts = file_name.rsplit(".", 1)

    datasetseries = parts[0]  # Use the full base name (excluding extension) as the date
    ext = (
        "raw"
        if len(parts) == 1 or parts[1].lower() not in RECOGNIZED_FORMATS
        else parts[1].lower()
    )

    return "/".join(
        distribution_to_url("", dataset, datasetseries, ext, catalog, is_download).split("/")[1:]
    )


def handle_paginated_request(session: Session, url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """
    Fetches all pages from a paginated API endpoint using the x-jpmc-next-token header,
    merges the results, and returns a single combined response dictionary.
    """
    all_responses: list[dict[str, Any]] = []
    current_headers = headers.copy() if headers else {}

    next_token = None

    while True:
        if next_token:
            current_headers["x-jpmc-next-token"] = next_token

        response = session.get(url, headers=current_headers)
        response.raise_for_status()
        response_data = response.json()
        all_responses.append(response_data)

        next_token = response.headers.get("x-jpmc-next-token")
        if not next_token:
            break

    return _merge_responses(all_responses)


def _merge_responses(responses: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Merges all top-level list fields from a list of response dictionaries.
    For each top-level key that is a list, concatenates the lists across all responses.
    Non-list fields are taken from the first response.
    """
    if not responses:
        return {}

    merged = responses[0].copy()
    # Find all top-level keys that are lists in the first response
    list_keys = [k for k, v in merged.items() if isinstance(v, list)]

    for key in list_keys:
        for response in responses[1:]:
            if key in response and isinstance(response[key], list):
                merged[key].extend(response[key])

    return merged

def ensure_resources(resp: dict[str, Any]) -> None:
    """Raise APIResponseError if 'resources' key is missing or empty in the response."""
    if "resources" not in resp or not resp["resources"]:
        raise APIResponseError(ValueError("No data found"))
    

def csv_to_table(
    path: str,
    fs: fsspec.filesystem  | None = None,
    columns: list[str] | None = None,
) -> pa.Table:
    """Reads csv data to pyarrow table.

    Args:
        path (str): path to the file.
        fs: filesystem object.
        columns: columns to read.

    Returns:
        class:`pyarrow.Table` pyarrow table with the data.
    """
    import pyarrow as pa
    from pyarrow import csv
    
    # Read CSV file
    with fs.open(path) if fs else nullcontext(path) as f:
        tbl = csv.read_csv(f)
    
    # Select columns if specified (PyArrow 0.17.1 compatible method)
    if columns is not None:
        # PyArrow 0.17.1 doesn't have .select(), use column indexing
        schema = pa.schema([tbl.schema.field(col) for col in columns])
        arrays = [tbl.column(col) for col in columns]
        tbl = pa.Table.from_arrays(arrays, schema=schema)
    
    return tbl


def json_to_table(
    path: str,
    fs: fsspec.filesystem | None = None,
    columns: list[str] | None = None,
) -> pa.Table:
    """Reads json data to pyarrow table.

    Args:
        path: path to json file.
        fs: filesystem.
        columns: columns to read.

    Returns:
        class:`pyarrow.Table` pyarrow table with the data.
    """
    import pyarrow as pa
    from pyarrow import json
    
    # Read JSON file
    with fs.open(path) if fs else nullcontext(path) as f:
        tbl = json.read_json(f)
    
    # Select columns if specified (PyArrow 0.17.1 compatible method)
    if columns is not None:
        # PyArrow 0.17.1 doesn't have .select(), use column indexing
        schema = pa.schema([tbl.schema.field(col) for col in columns])
        arrays = [tbl.column(col) for col in columns]
        tbl = pa.Table.from_arrays(arrays, schema=schema)
    
    return tbl


PathLikeT = Union[str, Path]


def parquet_to_table(
    path: PathLikeT | list[PathLikeT],
    fs: fsspec.filesystem | None = None,
    columns: list[str] | None = None,
) -> pa.Table:
    """Reads parquet data to pyarrow table.

    Args:
        path: path to parquet file or list of paths.
        fs: filesystem.
        columns: columns to read.

    Returns:
        class:`pyarrow.Table` pyarrow table with the data.
    """
    import pyarrow.parquet as pq

    # Read parquet dataset
    return pq.ParquetDataset(
        path,
        filesystem=fs,
        memory_map=True,
    ).read(columns=columns)


def read_csv(  # noqa: PLR0912
    path: str | zipfile.ZipFile,
    columns: list[str] | None = None,
    fs: fsspec.filesystem | None = None,
    dataframe_type: str = "pandas",
) -> pd.DataFrame | pa.Table:
    """Reads csv with column selection.

    Args:
        path (str): path to the csv file.
        columns: list of columns to read.
        fs: filesystem object.
        dataframe_type (str, optional): Dataframe type 'pandas' or 'polars'.

    Returns:
        Union[pandas.DataFrame, polars.DataFrame]: a dataframe containing the data.

    """
    
    if isinstance(path, zipfile.ZipExtFile):
        res = pd.read_csv(path, usecols=columns, index_col=False)
    elif isinstance(path, str):
        try:
            try:
                res = csv_to_table(path, fs, columns=columns)

                if dataframe_type == "pandas":
                    res = res.to_pandas()
                elif dataframe_type == "polars":
                    import polars as pl

                    res = pl.from_arrow(res)
                else:
                    raise ValueError(f"Unknown DataFrame type {dataframe_type}")
            except Exception as err:
                logger.log(
                    VERBOSE_LVL,
                    "Failed to read %s, with comma delimiter.",
                    path,
                    exc_info=True,
                )
                raise ValueError from err

        except Exception:  # noqa: BLE001
            logger.log(
                VERBOSE_LVL,
                "Could not parse %s properly. Trying with pandas csv reader.",
                path,
                exc_info=True,
            )
            try:  # pragma: no cover
                with fs.open(path) if fs else nullcontext(path) as f:
                    if dataframe_type == "pandas":
                        res = pd.read_csv(f, usecols=columns, index_col=False)
                    elif dataframe_type == "polars":
                        import polars as pl

                        res = pl.read_csv(f, columns=columns)
                    else:
                        raise ValueError(f"Unknown DataFrame type {dataframe_type}")  # noqa: W0707

            except Exception as err:  # noqa: BLE001
                logger.log(
                    VERBOSE_LVL,
                    "Could not parse %s properly. Trying with pandas csv reader pandas engine.",
                    path,
                    exc_info=True,
                )
                with fs.open(path) if fs else nullcontext(path) as f:
                    if dataframe_type == "pandas":
                        res = pd.read_table(  # pragma: no cover
                            f,
                            usecols=columns,
                            index_col=False,
                            engine="python",
                            delimiter=None,
                        )
                    else:
                        raise ValueError(f"Unknown DataFrame type {dataframe_type}") from err
    return res


def read_json(
    path: str,
    columns: list[str] | None = None,
    fs: fsspec.filesystem | None = None,
    dataframe_type: str = "pandas",
) -> pd.DataFrame | pa.Table:
    """Read json file(s) with column selection.

    Args:
        path (str): path to json file.
        columns (list): list of columns to read.
        fs: filesystem object.
        dataframe_type (str, optional): Dataframe type 'pandas' or 'polars'.

    Returns:
        Union[pandas.DataFrame, polars.DataFrame]: a dataframe containing the data.
    """

    try:
        try:
            res = json_to_table(path, fs, columns=columns)
            if dataframe_type == "pandas":
                res = res.to_pandas()
            elif dataframe_type == "polars":
                import polars as pl

                res = pl.from_arrow(res)
            else:
                raise ValueError(f"Unknown DataFrame type {dataframe_type}")
        except Exception as err:
            logger.log(
                VERBOSE_LVL,
                "Failed to read %s, with arrow reader. %s",
                path,
                err,
            )
            raise err

    except Exception:  # noqa: BLE001
        logger.log(
            VERBOSE_LVL,
            "Could not parse %s properly. Trying with pandas json reader.",
            path,
            exc_info=True,
        )
        try:  # pragma: no cover
            with fs.open(path) if fs else nullcontext(path) as f:
                if dataframe_type == "pandas":
                    res = pd.read_json(f)
                elif dataframe_type == "polars":
                    import polars as pl

                    res = pl.read_json(f)
                else:
                    raise ValueError(f"Unknown DataFrame type {dataframe_type}")

        except Exception as err:
            logger.log(VERBOSE_LVL, f"Could not parse {path} properly. ", exc_info=True)
            raise err  # pragma: no cover
    return res


def read_parquet(
    path: PathLikeT,
    columns: list[str] | None = None,
    fs: fsspec.filesystem | None = None,
    dataframe_type: str = "pandas",
) -> pd.DataFrame | pa.Table:
    """Read parquet file(s) with column selection.

    Args:
        path (PathLikeT): path or list of paths to parquet files.
        columns (list): list of columns to read.
        fs: filesystem object.
        dataframe_type (str, optional): Dataframe type 'pandas' or 'polars'.

    Returns:
        Union[pandas.DataFrame, polars.DataFrame]: a dataframe containing the data.

    """
    import pyarrow.parquet as pq

    paths = path if isinstance(path, list) else [path]
    dataframes = []
    for p in paths:
        if fs:
            with fs.open(p, 'rb') as f:
                table = pq.read_table(f, columns=columns)
                df = table.to_pandas()
        else:
            table = pq.read_table(p, columns=columns)
            df = table.to_pandas()
        dataframes.append(df)
    
    df = pd.concat(dataframes, ignore_index=True) if len(dataframes) > 1 else dataframes[0]
    
    if dataframe_type == "polars":
        import polars as pl
        return pl.from_pandas(df)
    
    return df
