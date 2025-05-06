import datetime
import io
import json
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Generator, List, Tuple
from unittest.mock import MagicMock, patch

import fsspec
import joblib
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from fusion import Fusion
from fusion.authentication import FusionOAuthAdapter
from fusion.credentials import FusionCredentials
from fusion.utils import (
    _filename_to_distribution,
    _normalise_dt_param,
    convert_date_format,
    cpu_count,
    distribution_to_filename,
    distribution_to_url,
    get_session,
    is_dataset_raw,
    make_bool,
    make_list,
    normalise_dt_param_str,
    path_to_url,
    snake_to_camel,
    tidy_string,
    upload_files,
    validate_file_names,
)


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("col1,col2\nvalue1,value2\n")
    return csv_path


@pytest.fixture
def sample_csv_path_str(sample_csv_path: Path) -> str:
    return str(sample_csv_path)


@pytest.fixture
def sample_json_path(tmp_path: Path) -> Path:
    json_path = tmp_path / "sample.json"
    json_path.write_text('{"col1": "value1", "col2": "value2"}\n')
    return json_path


@pytest.fixture
def sample_json_path_str(sample_json_path: Path) -> str:
    return str(sample_json_path)


@pytest.fixture
def sample_parquet_path(tmp_path: Path) -> Path:
    parquet_path = tmp_path / "sample.parquet"

    def generate_sample_parquet_file(parquet_path: Path) -> None:
        data = {"col1": ["value1"], "col2": ["value2"]}
        test_df = pd.DataFrame(data)
        test_df.to_parquet(parquet_path)

    generate_sample_parquet_file(parquet_path)
    return parquet_path


@pytest.fixture
def sample_parquet_paths(tmp_path: Path) -> List[Path]:
    parquet_paths = []
    for i in range(3):
        parquet_path = tmp_path / f"sample_{i}.parquet"

        def generate_sample_parquet_file(parquet_path: Path) -> None:
            data = {"col1": ["value1"], "col2": ["value2"]}
            test_df = pd.DataFrame(data)
            test_df.to_parquet(parquet_path)

        generate_sample_parquet_file(parquet_path)
        parquet_paths.append(parquet_path)
    return parquet_paths


@pytest.fixture
def sample_parquet_paths_str(sample_parquet_paths: List[Path]) -> List[str]:
    return [str(p) for p in sample_parquet_paths]


def test_cpu_count() -> None:
    assert cpu_count() > 0

def test_cpu_count_with_num_threads_env_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    test_num_threads = 8
    monkeypatch.setenv("NUM_THREADS", str(test_num_threads))
    assert cpu_count() == test_num_threads

def test_cpu_count_with_default_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NUM_THREADS", raising=False)
    assert cpu_count() == mp.cpu_count()

def test_normalise_dt_param_with_datetime() -> None:
    dt = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_date() -> None:
    dt = datetime.date(2022, 1, 1)
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_integer() -> None:
    dt = 20220101
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_valid_string_format_1() -> None:
    dt = "2022-01-01"
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_valid_string_format_2() -> None:
    dt = "20220101"
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_valid_string_format_3() -> None:
    dt = "20220101T1200"
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01-1200"


def test_normalise_dt_param_with_invalid_format() -> None:
    dt = "2022/01/01"
    with pytest.raises(ValueError, match="is not in a recognised data format"):
        _normalise_dt_param(dt)


def test_normalise_dt_param_with_invalid_type() -> None:
    dt = 32.23
    with pytest.raises(ValueError, match="is not in a recognised data format"):
        _normalise_dt_param(dt)  # type: ignore


def test_normalise_dt_param_str() -> None:
    dt = "2022-01-01"
    result = normalise_dt_param_str(dt)
    assert result == ("2022-01-01",)

    dt = "2022-01-01:2022-01-31"
    result = normalise_dt_param_str(dt)
    assert result == ("2022-01-01", "2022-01-31")

    dt = "2022-01-01:2022-01-01:2022-01-01"
    with pytest.raises(ValueError, match=f"Unable to parse {dt} as either a date or an interval"):
        normalise_dt_param_str(dt)


@pytest.fixture
def fs_fusion() -> MagicMock:
    return MagicMock()


@pytest.fixture
def fs_local() -> MagicMock:
    return MagicMock()


@pytest.fixture
def loop() -> pd.DataFrame:
    data = {"url": ["url1", "url2"], "path": ["path1", "path2"]}
    return pd.DataFrame(data)


def test_path_to_url() -> None:
    result = path_to_url("path/to/dataset__catalog__datasetseries.csv")
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv"

    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_raw=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv"

    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_download=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv/operationType/download"

    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_raw=True, is_download=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv/operationType/download"

    result = path_to_url("path/to/dataset__catalog__datasetseries.pt", is_raw=True, is_download=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/raw/operationType/download"


def test_filename_to_distribution() -> None:
    file_name = "dataset__catalog__datasetseries.csv"
    catalog, dataset, datasetseries, file_format = _filename_to_distribution(file_name)
    assert catalog == "catalog"
    assert dataset == "dataset"
    assert datasetseries == "datasetseries"
    assert file_format == "csv"

    file_name = "anotherdataset__anothercatalog__anotherdatasetseries.parquet"
    catalog, dataset, datasetseries, file_format = _filename_to_distribution(file_name)
    assert catalog == "anothercatalog"
    assert dataset == "anotherdataset"
    assert datasetseries == "anotherdatasetseries"
    assert file_format == "parquet"


def test_distribution_to_url() -> None:
    root_url = "https://api.fusion.jpmc.com/"
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    bad_series_chs = ["/", "\\"]
    exp_res = (
        f"{root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/" f"{datasetseries}/distributions/{file_format}"
    )
    for ch in bad_series_chs:
        datasetseries = f"2020-04-04{ch}"
        result = distribution_to_url(root_url, dataset, datasetseries, file_format, catalog)
        assert result == exp_res

    datasetseries = "2020-04-04"
    exp_res = (
        f"{root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/"
        f"{datasetseries}/distributions/{file_format}/operationType/download"
    )
    for ch in bad_series_chs:
        datasetseries_mod = f"2020-04-04{ch}"
        result = distribution_to_url(root_url, dataset, datasetseries_mod, file_format, catalog, is_download=True)
        assert result == exp_res

    exp_res = f"{root_url}catalogs/{catalog}/datasets/{dataset}/sample/distributions/csv"
    datasetseries = "sample"
    assert distribution_to_url(root_url, dataset, datasetseries, file_format, catalog) == exp_res


def test_distribution_to_filename() -> None:
    root_dir = "/tmp"
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    exp_res = f"{root_dir}/{dataset}__{catalog}__{datasetseries}.{file_format}"
    bad_series_chs = ["/", "\\"]
    for ch in bad_series_chs:
        datasetseries_mod = f"2020-04-04{ch}"
        res = distribution_to_filename(root_dir, dataset, datasetseries_mod, file_format, catalog)
        assert res == exp_res

    exp_res = f"{root_dir}/{dataset}.{file_format}"
    for ch in bad_series_chs:
        datasetseries_mod = f"2020-04-04{ch}"
        res = distribution_to_filename(root_dir, dataset, datasetseries_mod, file_format, catalog, partitioning="hive")
        assert res == exp_res

    root_dir = "c:\\tmp"
    exp_res = f"{root_dir}\\{dataset}__{catalog}__{datasetseries}.{file_format}"
    res = distribution_to_filename(root_dir, dataset, datasetseries, file_format, catalog)
    assert res == exp_res


TmpFsT = Tuple[fsspec.spec.AbstractFileSystem, str]


@pytest.fixture
def temp_fs() -> Generator[TmpFsT, None, None]:
    with tempfile.TemporaryDirectory() as tmpdirname, patch(
        "fsspec.filesystem", return_value=fsspec.filesystem("file", auto_mkdir=True, root_path=tmpdirname)
    ) as mock_fs:
        yield mock_fs, tmpdirname


def gen_binary_data(n: int, pad_len: int) -> List[bytes]:
    return [bin(i)[2:].zfill(pad_len).encode() for i in range(n)]


def test_progress_update() -> None:
    num_inputs = 100
    inputs = list(range(num_inputs))

    def true_if_even(x: int) -> Tuple[bool, int]:
        return (x % 2 == 0, x)

    with joblib.parallel_backend("threading"):
        res = joblib.Parallel(n_jobs=10)(joblib.delayed(true_if_even)(i) for i in inputs)

    assert len(res) == num_inputs


@pytest.fixture
def mock_fs_fusion() -> MagicMock:
    fs = MagicMock()
    fs.ls.side_effect = lambda path: {
        "catalog1": ["catalog1", "catalog2"],
        "catalog2": ["catalog1", "catalog2"],
        "catalog1/datasets": ["dataset1", "dataset2"],
        "catalog2/datasets": ["dataset3"],
    }.get(path, [])

    fs.cat.side_effect = lambda path: json.dumps({
        "identifier": "dataset1"
    }) if path == "catalog1/datasets/dataset1" else json.dumps({})
    return fs


# Validation tests

def test_validate_correct_file_names(mock_fs_fusion: MagicMock) -> None:
    paths = ["path/to/dataset1__catalog1__20230101.csv"]
    expected = [True]
    assert validate_file_names(paths, mock_fs_fusion) == expected

def test_validate_incorrect_format_file_names(mock_fs_fusion: MagicMock) -> None:
    paths = ["path/to/incorrectformatfile.csv"]
    expected = [False]
    assert validate_file_names(paths, mock_fs_fusion) == expected


def test_validate_non_existing_catalog(mock_fs_fusion: MagicMock) -> None:
    paths = ["path/to/dataset1__catalog3__20230101.csv"]
    expected = [False]
    assert validate_file_names(paths, mock_fs_fusion) == expected


def test_validate_non_existing_dataset(mock_fs_fusion: MagicMock) -> None:
    paths = ["path/to/dataset4__catalog1__20230101.csv"]
    expected = [False]
    assert validate_file_names(paths, mock_fs_fusion) == expected


def test_validate_error_paths(mock_fs_fusion: MagicMock) -> None:
    paths = ["path/to/catalog1__20230101.csv"]
    expected = [False]
    assert validate_file_names(paths, mock_fs_fusion) == expected


def test_empty_input_list(mock_fs_fusion: MagicMock) -> None:
    paths: List[str] = []
    expected: List[bool] = []
    assert validate_file_names(paths, mock_fs_fusion) == expected


def test_filesystem_exceptions(mock_fs_fusion: MagicMock) -> None:
    mock_fs_fusion.ls.side_effect = Exception("Failed to list directories")
    paths = ["path/to/dataset1__catalog1__20230101.csv"]
    with pytest.raises(Exception, match="Failed to list directories"):
        validate_file_names(paths, mock_fs_fusion)


def test_get_session(mocker: MockerFixture, credentials: FusionCredentials, fusion_obj: Fusion) -> None:
    session = get_session(credentials, fusion_obj.root_url)
    assert session

    session = get_session(credentials, fusion_obj.root_url, get_retries=3)
    assert session

    # Mock out the request to raise an exception
    mocker.patch("fusion.utils._get_canonical_root_url", side_effect=Exception("Failed to get canonical root url"))
    session = get_session(credentials, fusion_obj.root_url)
    for mnt, adapter_obj in session.adapters.items():
        if isinstance(adapter_obj, FusionOAuthAdapter):
            assert mnt == "https://"

@pytest.fixture
def mock_fs_fusion_w_cat() -> MagicMock:
    fs = MagicMock()
    # Mock the 'cat' method to return JSON strings as bytes
    fs.cat.side_effect = lambda path: {
        "catalog1/datasets/dataset1": b'{"isRawData": true}',
        "catalog1/datasets/dataset2": b'{"isRawData": false}',
    }.get(path, b"{}")  # Default empty JSON if path not found
    return fs


def test_is_dataset_raw(mock_fs_fusion_w_cat: MagicMock) -> None:
    paths = ["path/to/dataset1__catalog1.csv"]
    expected = [True]
    assert is_dataset_raw(paths, mock_fs_fusion_w_cat) == expected


def test_is_dataset_raw_fail(mock_fs_fusion_w_cat: MagicMock) -> None:
    paths = ["path/to/dataset2__catalog1.csv"]
    expected = [False]
    assert is_dataset_raw(paths, mock_fs_fusion_w_cat) == expected


def test_is_dataset_raw_empty_input_list(mock_fs_fusion_w_cat: MagicMock) -> None:
    paths: List[str] = []
    expected: List[bool] = []
    assert is_dataset_raw(paths, mock_fs_fusion_w_cat) == expected


def test_is_dataset_raw_filesystem_exceptions(mock_fs_fusion_w_cat: MagicMock) -> None:
    mock_fs_fusion_w_cat.cat.side_effect = Exception("File not found")
    paths = ["path/to/dataset1__catalog1.csv"]
    with pytest.raises(Exception, match="File not found"):
        is_dataset_raw(paths, mock_fs_fusion_w_cat)


def test_is_dataset_raw_caching_of_results(mock_fs_fusion_w_cat: MagicMock) -> None:
    paths = ["path/to/dataset1__catalog1.csv", "path/to/dataset1__catalog1.csv"]
    is_dataset_raw(paths, mock_fs_fusion_w_cat)
    mock_fs_fusion_w_cat.cat.assert_called_once()


@pytest.fixture
def setup_fs() -> Tuple[fsspec.AbstractFileSystem, fsspec.AbstractFileSystem]:
    fs_fusion = MagicMock(spec=fsspec.AbstractFileSystem)
    fs_local = MagicMock(spec=fsspec.AbstractFileSystem)
    fs_local.size.return_value = 4 * 2**20  # Less than chunk_size to test single-part upload
    fs_fusion.put.return_value = None
    return fs_fusion, fs_local


@pytest.fixture
def upload_row() -> pd.Series:  # type: ignore
    return pd.Series({"url": "http://example.com/file", "path": "localfile/path/file.txt"})


@pytest.fixture
def upload_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "url": ["http://example.com/file1", "http://example.com/file2", "http://example.com/file3"],
            "path": ["localfile/path/file1.txt", "localfile/path/file2.txt", "localfile/path/file3.txt"],
        }
    )


def test_upload_public(
    setup_fs: Tuple[fsspec.AbstractFileSystem, fsspec.AbstractFileSystem], upload_rows: pd.DataFrame
) -> None:
    fs_fusion, fs_local = setup_fs

    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=True, parallel=False)
    assert res
    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=False, parallel=False)
    assert res

    fs_local.size.return_value = 5 * 2**20
    fs_local = io.BytesIO(b"some data to simulate file content" * 100)
    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=False, parallel=False)
    assert res


def test_upload_public_parallel(
    setup_fs: Tuple[fsspec.AbstractFileSystem, fsspec.AbstractFileSystem], upload_rows: pd.DataFrame
) -> None:
    fs_fusion, fs_local = setup_fs

    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=False, parallel=True)
    assert res

    fs_local.size.return_value = 5 * 2**20
    fs_local = io.BytesIO(b"some data to simulate file content" * 100)
    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=False, parallel=True)
    assert res


def test_tidy_string() -> None:
    bad_string = " string with  spaces  and  multiple  spaces  "
    assert tidy_string(bad_string) == "string with spaces and multiple spaces"


def test_make_list_from_string() -> None:
    string_obj = "Hello, hi, hey"
    string_to_list = make_list(string_obj)
    assert isinstance(string_to_list, list)
    assert len(string_to_list) == 3
    assert string_to_list == ["Hello", "hi", "hey"]


def test_make_list_from_list() -> None:
    list_obj = ["hi", "hi"]
    list_to_list = make_list(list_obj)
    assert isinstance(list_to_list, list)
    assert list_to_list == ["hi", "hi"]


def test_make_list_from_nonstring() -> None:
    """Test make list from non string."""
    any_obj = 1
    obj_to_list = make_list(any_obj)
    assert isinstance(obj_to_list, list)
    assert obj_to_list == [1]


def test_make_bool_string() -> None:
    """Test make bool."""
    assert make_bool("string") is True


def test_make_bool_hidden_false() -> None:
    """Test make bool."""
    assert make_bool("False") is False
    assert make_bool("false") is False
    assert make_bool("FALSE") is False
    assert make_bool("0") is False


def test_make_bool_bool() -> None:
    """Test make bool."""
    assert make_bool(True) is True


def test_make_bool_1() -> None:
    """Test make bool."""
    assert make_bool(1) is True


def test_make_bool_0() -> None:
    """Test make bool."""
    assert make_bool(0) is False


def test_convert_date_format_month() -> None:
    """Test convert date format."""
    assert convert_date_format("May 6, 2024") == "2024-05-06"


def test_convert_format_one_string() -> None:
    """Test convert date format."""
    assert convert_date_format("20240506") == "2024-05-06"


def test_convert_format_slash() -> None:
    """Test convert date format."""
    assert convert_date_format("2024/05/06") == "2024-05-06"


def test_snake_to_camel() -> None:
    """Test snake to camel."""
    assert snake_to_camel("this_is_snake") == "thisIsSnake"