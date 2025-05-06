
"""Fusion utilities."""

from __future__ import annotations

import contextlib
import json as js
import logging
import multiprocessing as mp
import os
import re
import ssl
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
from tqdm import tqdm
from urllib3.util.retry import Retry

from .authentication import FusionAiohttpSession, FusionOAuthAdapter

if TYPE_CHECKING:
    from collections.abc import Generator

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


def _normalise_dt_param(dt: Union[str, int, datetime, date]) -> str:
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

    matches = DT_YYYY_MM_DD_RE.match(dt)
    if matches:
        yr = matches.group(1)
        mth = matches.group(2).zfill(2)
        day = matches.group(3).zfill(2)
        return f"{yr}-{mth}-{day}"

    matches = DT_YYYYMMDD_RE.match(dt)

    if matches:
        return "-".join(matches.groups())

    matches = DT_YYYYMMDDTHHMM_RE.match(dt)

    if matches:
        return "-".join(matches.groups())

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
) -> str:
    """Returns a filename representing a dataset distribution.

    Args:
        root_folder (str): The root folder path.
        dataset (str): The dataset identifier.
        datasetseries (str): The datasetseries instance identifier.
        file_format (str): The file type, e.g. CSV or Parquet
        catalog (str, optional): The data catalog containing the dataset. Defaults to "common".
        partitioning (str, optional): Partitioning specification.

    Returns:
        str: FQ distro file name
    """
    if datasetseries[-1] == "/" or datasetseries[-1] == "\\":
        datasetseries = datasetseries[0:-1]

    if partitioning == "hive":
        file_name = f"{dataset}.{file_format}"
    else:
        file_name = f"{dataset}__{catalog}__{datasetseries}.{file_format}"

    sep = "/"
    if "\\" in root_folder:
        sep = "\\"
    return f"{root_folder}{sep}{file_name}"


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
) -> str:
    """Returns the API URL to download a dataset distribution.

    Args:
        root_url (str): The base url for the API.
        dataset (str): The dataset identifier.
        datasetseries (str): The datasetseries instance identifier.
        file_format (str): The file type, e.g. CSV or Parquet
        catalog (str, optional): The data catalog containing the dataset. Defaults to "common".
        is_download (bool, optional): Is url for download

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
            f"{datasetseries}/distributions/{file_format}/operationType/download"
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


def validate_file_names(paths: List[str], fs_fusion: fsspec.AbstractFileSystem) -> List[bool]:
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
        if len(tmp) == file_seg_cnt:
            try:
                val = tmp[1] in [i.split("/")[0] for i in fs_fusion.ls(tmp[1])]
            except FileNotFoundError:
                val = False
            if not val:
                validation.append(False)
            else:
                # check if dataset exists
                try:
                    val = tmp[0] in js.loads(fs_fusion.cat(f"{tmp[1]}/datasets/{tmp[0]}"))["identifier"]
                except Exception:  # noqa: BLE001
                    val = False
                validation.append(val)
        else:
            validation.append(False)
        if not validation[-1] and len(tmp) == file_seg_cnt:
            logger.warning(
                "You might not have access to the catalog %s or dataset %s. "
                "Please check your permission on the platform.",
                tmp[1],
                tmp[0],
            )
        if not validation[-1] and len(tmp) != file_seg_cnt:
            logger.warning(
                f"The file in {paths[i]} has a non-compliant name and will not be processed. "
                f"Please rename the file to dataset__catalog__yyyymmdd.format"
            )

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
    parallel: bool = True,
    n_par: int = -1,
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
        parallel (bool): Is parallel mode enabled.
        n_par (int): Number of subprocesses.
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
