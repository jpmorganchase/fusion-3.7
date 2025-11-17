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
    _merge_responses,
    _normalise_dt_param,
    convert_date_format,
    cpu_count,
    distribution_to_filename,
    distribution_to_url,
    ensure_resources,
    file_name_to_url,
    get_session,
    handle_paginated_request,
    is_dataset_raw,
    make_bool,
    make_list,
    normalise_dt_param_str,
    path_to_url,
    snake_to_camel,
    tidy_string,
    upload_files,
    validate_file_formats,
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
    assert result == "2022-01-01T12:00:00"


def test_normalise_dt_param_with_invalid_format() -> None:
    invalid_dates = [
        "2022-13-01",      # invalid month
        "2022-01-32",      # invalid day
        "not-a-date",      # not a date at all
        "",                # empty string
    ]
    for dt in invalid_dates:
        with pytest.raises(
            ValueError,
            match="is not in a recognised data format|NaTType does not support time",
        ):
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
    assert result == (
        "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv/"
        "files/operationType/download?file=None"
    )

    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_raw=True, is_download=True)    
    assert result == (
        "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv/"
        "files/operationType/download?file=None"
    )

    result = path_to_url("path/to/dataset__catalog__datasetseries.pt", is_raw=True, is_download=True)    
    assert result == (
        "catalog/datasets/dataset/datasetseries/datasetseries/distributions/raw/"
        "files/operationType/download?file=None"
    )

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
        f"{datasetseries}/distributions/{file_format}/files/operationType/download?file=file"
    )
    for ch in bad_series_chs:
        datasetseries_mod = f"2020-04-04{ch}"
        result = distribution_to_url(
            root_url,
            dataset,
            datasetseries_mod,
            file_format,
            catalog,
            is_download=True,
            file_name="file",
        )
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

def test_validate_correct_file_names() -> None:
    paths = ["path/to/dataset1__catalog1__20230101.csv"]
    expected = [True]
    assert validate_file_names(paths) == expected

def test_validate_incorrect_format_file_names() -> None:
    paths = ["path/to/incorrectformatfile.csv"]
    expected = [False]
    assert validate_file_names(paths) == expected

def test_validate_error_paths() -> None:
    paths = ["path/to/catalog1__20230101.csv"]
    expected = [False]
    assert validate_file_names(paths) == expected


def test_empty_input_list() -> None:
    paths: List[str] = []
    expected: List[bool] = []
    assert validate_file_names(paths) == expected

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

    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=True)
    assert res
    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=False)
    assert res

    fs_local.size.return_value = 5 * 2**20
    fs_local = io.BytesIO(b"some data to simulate file content" * 100)
    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=False)
    assert res


def test_upload_public_parallel(
    setup_fs: Tuple[fsspec.AbstractFileSystem, fsspec.AbstractFileSystem], upload_rows: pd.DataFrame
) -> None:
    fs_fusion, fs_local = setup_fs

    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=False)
    assert res

    fs_local.size.return_value = 5 * 2**20
    fs_local = io.BytesIO(b"some data to simulate file content" * 100)
    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=False)
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

def test_folder_does_not_exist(mock_fs_fusion: MagicMock)  -> None:
    mock_fs_fusion.exists.return_value = False
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_file_formats(mock_fs_fusion, "/nonexistent")

def test_single_raw_file(mock_fs_fusion: MagicMock) -> None:
    mock_fs_fusion.makedirs("/test")
    mock_fs_fusion.touch("/test/file1.raw")
    # Should not raise
    validate_file_formats(mock_fs_fusion, "/test")

def test_only_supported_files(mock_fs_fusion: MagicMock) -> None:
    mock_fs_fusion.makedirs("/data")
    mock_fs_fusion.touch("/data/file1.csv")
    mock_fs_fusion.touch("/data/file2.json")
    mock_fs_fusion.touch("/data/file3.xlsx")
    # Should not raise
    validate_file_formats(mock_fs_fusion, "/data")

def test_multiple_raw_files(mock_fs_fusion: MagicMock) -> None:
    mock_fs_fusion.exists.return_value = True
    mock_fs_fusion.find.return_value = [
        "/mixed/file1.unknown",
        "/mixed/file2.custom",
        "/mixed/readme.txt"
    ]
    mock_fs_fusion.info.side_effect = lambda _: {"type": "file"}

    with pytest.raises(ValueError, match="Multiple raw files detected"):
        validate_file_formats(mock_fs_fusion, "/mixed")

@pytest.mark.parametrize(
    (
        "file_name",
        "dataset",
        "catalog",
        "is_download",
        "expected_ext",
        "expected_series",
    ),
    [
        ("folder__file.csv", "my_dataset", "my_catalog", False, "csv", "folder__file"),
        ("data__deep.custom", "my_dataset", "my_catalog", False, "raw", "data__deep"),
        ("no__ext", "my_dataset", "my_catalog", False, "raw", "no__ext"),
        ("some__leaf.json", "my_dataset", "my_catalog", True, "json", "some__leaf"),
        ("folder__20200101/extra.csv", "my_dataset", "my_catalog", False, "csv", "folder__20200101/extra"),
    ],
)
def test_file_name_to_url(
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
    dataset: str,
    catalog: str,
    is_download: bool,
    expected_ext: str,
    expected_series: str,
) -> None:

    def mock_distribution_to_url(
        root_url: str, # noqa: ARG001
        dataset_arg: str,
        series: str,
        ext: str,
        catalog_arg: str,
        is_download_arg: bool,
    ) -> str:
        return f"/mock/{catalog_arg}/{dataset_arg}/{series}.{ext}?dl={is_download_arg}"

    monkeypatch.setattr("fusion.utils.distribution_to_url", mock_distribution_to_url)

    result = file_name_to_url(file_name, dataset, catalog, is_download)
    expected_url = f"mock/{catalog}/{dataset}/{expected_series}.{expected_ext}?dl={is_download}"
    assert result == expected_url

class DummyResponse:
    def __init__(self, json_data: object, headers: dict = None) -> None:
        self._json_data = json_data
        self.headers = headers or {}
        self.status_code = 200
    def json(self) -> object:
        return self._json_data
    def raise_for_status(self) -> None:
        pass

class DummySession:
    def __init__(self, responses: list) -> None:
        self.responses = responses
        self.call_count = 0
    def get(self, url: str, headers: dict = None) -> object:
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

def test_handle_paginated_request_merges_all_pages() -> None:
    page1 = DummyResponse({'resources': [1, 2]}, headers={'x-jpmc-next-token': 'token2'})
    page2 = DummyResponse({'resources': [3, 4]}, headers={})
    session = DummySession([page1, page2])
    url = 'http://fake-url'
    result = handle_paginated_request(session, url)
    assert 'resources' in result
    assert result['resources'] == [1, 2, 3, 4]

def test_handle_paginated_request_no_pagination() -> None:
    page1 = DummyResponse({'resources': [1, 2]}, headers={})
    session = DummySession([page1])
    url = 'http://fake-url'
    result = handle_paginated_request(session, url)
    assert 'resources' in result
    assert result['resources'] == [1, 2]

def test_handle_paginated_request_with_headers() -> None:
    page1 = DummyResponse({'resources': [1]}, headers={'x-jpmc-next-token': 'token2'})
    page2 = DummyResponse({'resources': [2]}, headers={})
    session = DummySession([page1, page2])
    url = 'http://fake-url'
    custom_headers = {'Authorization': 'Bearer token'}
    result = handle_paginated_request(session, url, headers=custom_headers)
    assert result['resources'] == [1, 2]

def test_handle_paginated_request_empty() -> None:
    page1 = DummyResponse({}, headers={})
    session = DummySession([page1])
    url = 'http://fake-url'
    result = handle_paginated_request(session, url)
    assert result == {}

def test_handle_paginated_request_merges_non_list_fields() -> None:
    page1 = DummyResponse({'resources': [1], 'meta': 'foo'}, headers={'x-jpmc-next-token': 'token2'})
    page2 = DummyResponse({'resources': [2], 'meta': 'bar'}, headers={})
    session = DummySession([page1, page2])
    url = 'http://fake-url'
    result = handle_paginated_request(session, url)
    assert result['resources'] == [1, 2]
    assert result['meta'] == 'foo'

def test_merge_responses_merges_lists() -> None:
    responses = [
        {'resources': [1, 2], 'meta': 'foo'},
        {'resources': [3, 4], 'meta': 'bar'},
        {'resources': [5], 'meta': 'baz'}
    ]
    merged = _merge_responses(responses)
    assert merged['resources'] == [1, 2, 3, 4, 5]
    assert merged['meta'] == 'foo'

def test_merge_responses_empty() -> None:
    assert _merge_responses([]) == {}

def test_merge_responses_no_lists() -> None:
    responses = [
        {'meta': 'foo', 'count': 1},
        {'meta': 'bar', 'count': 2}
    ]
    merged = _merge_responses(responses)
    assert merged['meta'] == 'foo'
    assert merged['count'] == 1

def test_merge_responses_mixed_types() -> None:
    responses = [
        {'resources': [1], 'meta': 'foo', 'other': 123},
        {'resources': [], 'meta': 'bar', 'other': 456},
        {'resources': [2, 3], 'meta': 'baz', 'other': 789}
    ]
    merged = _merge_responses(responses)
    assert merged['resources'] == [1, 2, 3]
    assert merged['meta'] == 'foo'
    assert merged['other'] == 123


def test_csv_to_table(tmp_path: Path) -> None:
    """Test csv_to_table reads CSV and returns pyarrow Table."""
    pa = pytest.importorskip("pyarrow")
    from fusion.utils import csv_to_table
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
    
    table = csv_to_table(str(csv_file))
    
    assert isinstance(table, pa.Table)
    assert len(table) == 2
    assert table.column_names == ["col1", "col2", "col3"]
    assert table.column("col1").to_pylist() == [1, 4]


def test_csv_to_table_with_columns(tmp_path: Path) -> None:
    """Test csv_to_table with column selection."""
    pa = pytest.importorskip("pyarrow")
    from fusion.utils import csv_to_table
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
    
    table = csv_to_table(str(csv_file), columns=["col1", "col3"])
    
    assert isinstance(table, pa.Table)
    assert len(table) == 2
    assert table.column_names == ["col1", "col3"]


def test_csv_to_table_with_filesystem(tmp_path: Path) -> None:
    """Test csv_to_table with fsspec filesystem."""
    pa = pytest.importorskip("pyarrow")
    from fusion.utils import csv_to_table
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2\n1,2\n3,4\n")
    
    fs = fsspec.filesystem("file")
    table = csv_to_table(str(csv_file), fs=fs)
    
    assert isinstance(table, pa.Table)
    assert len(table) == 2
    assert table.column_names == ["col1", "col2"]


def test_json_to_table(tmp_path: Path) -> None:
    """Test json_to_table reads JSON and returns pyarrow Table."""
    pa = pytest.importorskip("pyarrow")
    from fusion.utils import json_to_table
    
    json_file = tmp_path / "test.json"
    json_data = [
        {"col1": 1, "col2": 2, "col3": 3},
        {"col1": 4, "col2": 5, "col3": 6}
    ]
    json_file.write_text("\n".join(json.dumps(row) for row in json_data))
    
    table = json_to_table(str(json_file))
    
    assert isinstance(table, pa.Table)
    assert len(table) == 2
    assert set(table.column_names) == {"col1", "col2", "col3"}


def test_json_to_table_with_columns(tmp_path: Path) -> None:
    """Test json_to_table with column selection."""
    pa = pytest.importorskip("pyarrow")
    from fusion.utils import json_to_table
    
    json_file = tmp_path / "test.json"
    json_data = [
        {"col1": 1, "col2": 2, "col3": 3},
        {"col1": 4, "col2": 5, "col3": 6}
    ]
    json_file.write_text("\n".join(json.dumps(row) for row in json_data))
    
    table = json_to_table(str(json_file), columns=["col1", "col3"])
    
    assert isinstance(table, pa.Table)
    assert len(table) == 2
    assert set(table.column_names) == {"col1", "col3"}


def test_json_to_table_with_filesystem(tmp_path: Path) -> None:
    """Test json_to_table with fsspec filesystem."""
    pa = pytest.importorskip("pyarrow")
    from fusion.utils import json_to_table
    
    json_file = tmp_path / "test.json"
    json_data = [
        {"col1": 1, "col2": 2},
        {"col1": 3, "col2": 4}
    ]
    json_file.write_text("\n".join(json.dumps(row) for row in json_data))
    
    fs = fsspec.filesystem("file")
    table = json_to_table(str(json_file), fs=fs)
    
    assert isinstance(table, pa.Table)
    assert len(table) == 2


def test_parquet_to_table(tmp_path: Path) -> None:
    """Test parquet_to_table reads Parquet and returns pyarrow Table."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from fusion.utils import parquet_to_table
    
    # Create a test Parquet file
    parquet_file = tmp_path / "test.parquet"
    table = pa.table({
        "col1": [1, 4, 7],
        "col2": [2, 5, 8],
        "col3": [3, 6, 9]
    })
    pq.write_table(table, str(parquet_file))
    
    result = parquet_to_table(str(parquet_file))
    
    assert isinstance(result, pa.Table)
    assert len(result) == 3
    assert result.column_names == ["col1", "col2", "col3"]
    assert result.column("col1").to_pylist() == [1, 4, 7]


def test_parquet_to_table_with_columns(tmp_path: Path) -> None:
    """Test parquet_to_table with column selection."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from fusion.utils import parquet_to_table
    
    parquet_file = tmp_path / "test.parquet"
    table = pa.table({
        "col1": [1, 4, 7],
        "col2": [2, 5, 8],
        "col3": [3, 6, 9]
    })
    pq.write_table(table, str(parquet_file))
    
    result = parquet_to_table(str(parquet_file), columns=["col1", "col3"])
    
    assert isinstance(result, pa.Table)
    assert len(result) == 3
    assert result.column_names == ["col1", "col3"]
    assert result.column("col1").to_pylist() == [1, 4, 7]
    assert result.column("col3").to_pylist() == [3, 6, 9]


def test_parquet_to_table_multiple_files(tmp_path: Path) -> None:
    """Test parquet_to_table with multiple files."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from fusion.utils import parquet_to_table
    
    parquet_file1 = tmp_path / "test1.parquet"
    parquet_file2 = tmp_path / "test2.parquet"
    
    table1 = pa.table({"col1": [1, 2], "col2": [3, 4]})
    table2 = pa.table({"col1": [5, 6], "col2": [7, 8]})
    
    pq.write_table(table1, str(parquet_file1))
    pq.write_table(table2, str(parquet_file2))
    
    result = parquet_to_table([str(parquet_file1), str(parquet_file2)])
    
    assert isinstance(result, pa.Table)
    assert len(result) == 4
    assert result.column_names == ["col1", "col2"]
    assert result.column("col1").to_pylist() == [1, 2, 5, 6]


def test_parquet_to_table_with_filesystem(tmp_path: Path) -> None:
    """Test parquet_to_table with fsspec filesystem."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from fusion.utils import parquet_to_table
    
    parquet_file = tmp_path / "test.parquet"
    table = pa.table({"col1": [1, 2], "col2": [3, 4]})
    pq.write_table(table, str(parquet_file))
    
    fs = fsspec.filesystem("file")
    result = parquet_to_table(str(parquet_file), fs=fs)
    
    assert isinstance(result, pa.Table)
    assert len(result) == 2
    assert result.column_names == ["col1", "col2"]


def test_cpu_count_with_threading() -> None:
    """Test cpu_count returns 10 when is_threading is True."""
    result = cpu_count(is_threading=True)
    assert result == 10


def test_ensure_resources_missing() -> None:
    """Test ensure_resources raises error when resources key is missing."""
    from fusion.exceptions import APIResponseError
    
    with pytest.raises(APIResponseError):
        ensure_resources({})
    
    with pytest.raises(APIResponseError):
        ensure_resources({"resources": []})


def test_convert_date_format_invalid_input() -> None:
    """Test convert_date_format returns None for invalid inputs."""
    import numpy as np
    
    assert convert_date_format(np.nan) is None
    assert convert_date_format(123) is None  # type: ignore


def test_convert_date_format_exception() -> None:
    """Test convert_date_format returns None when parsing fails."""
    assert convert_date_format("not-a-date-at-all-xyz") is None


def test_distribution_to_filename_with_hive() -> None:
    """Test distribution_to_filename with hive partitioning."""
    root_dir = "/data"
    dataset = "my_dataset"
    datasetseries = "2024-01-01"
    file_format = "parquet"
    catalog = "my_catalog"
    
    result = distribution_to_filename(
        root_dir, dataset, datasetseries, file_format, catalog, partitioning="hive"
    )
    
    expected = f"{root_dir}/{dataset}.{file_format}"
    assert result == expected


def test_distribution_to_filename_with_explicit_filename() -> None:
    """Test distribution_to_filename with explicit file_name parameter."""
    root_dir = "/data"
    dataset = "my_dataset"
    datasetseries = "2024-01-01"
    file_format = "csv"
    catalog = "my_catalog"
    
    result = distribution_to_filename(
        root_dir, dataset, datasetseries, file_format, catalog, file_name="custom_name"
    )
    
    expected = f"{root_dir}/custom_name.{file_format}"
    assert result == expected


def test_distribution_to_filename_with_backslash_separator() -> None:
    """Test distribution_to_filename with Windows-style paths."""
    root_dir = "C:\\data"
    dataset = "my_dataset"
    datasetseries = "2024-01-01"
    file_format = "csv"
    catalog = "my_catalog"
    
    result = distribution_to_filename(root_dir, dataset, datasetseries, file_format, catalog)
    
    expected = f"{root_dir}\\{dataset}__{catalog}__{datasetseries}.{file_format}"
    assert result == expected


def test__is_json_valid() -> None:
    """Test _is_json returns True for valid JSON."""
    from fusion.utils import _is_json
    
    assert _is_json('{"key": "value"}') is True
    assert _is_json('["item1", "item2"]') is True
    assert _is_json('123') is True
    assert _is_json('"string"') is True


def test__is_json_invalid() -> None:
    """Test _is_json returns False for invalid JSON."""
    from fusion.utils import _is_json
    
    assert _is_json('not json') is False
    assert _is_json('{invalid}') is False
    assert _is_json('') is False


def test_cpu_count_fallback_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test cpu_count falls back to DEFAULT_THREAD_POOL_SIZE when mp.cpu_count() returns None."""
    import multiprocessing as mp

    from fusion.utils import DEFAULT_THREAD_POOL_SIZE
    
    monkeypatch.delenv("NUM_THREADS", raising=False)
    
    monkeypatch.setattr(mp, "cpu_count", lambda: None)
    
    result = cpu_count()
    assert result == DEFAULT_THREAD_POOL_SIZE


def test_cpu_count_with_thread_pool_size_param() -> None:
    """Test cpu_count returns the thread_pool_size parameter when provided."""
    result = cpu_count(thread_pool_size=5)
    assert result == 5


def test_tidy_string_with_spaces() -> None:
    """Test tidy_string removes extra spaces."""
    assert tidy_string(" hello   world  ") == "hello world"
    assert tidy_string("  multiple   spaces   ") == "multiple spaces"


def test_make_list_various_inputs() -> None:
    """Test make_list with various edge cases."""
    assert make_list("") == ['']
    result = make_list("   ")
    assert len(result) > 0

    assert make_list(" value ") == ["value"]


def test_snake_to_camel_edge_cases() -> None:
    """Test snake_to_camel with edge cases."""
    assert snake_to_camel("single") == "single"
    assert snake_to_camel("two_words") == "twoWords"
    assert snake_to_camel("three_word_name") == "threeWordName"
    assert snake_to_camel("UPPER_CASE") == "upperCase"


def test_normalise_dt_param_int_conversion() -> None:
    """Test _normalise_dt_param converts integer to string and processes it."""
    result = _normalise_dt_param(20240115)
    assert result == "2024-01-15"


def test_tidy_string_with_slashes() -> None:
    """Test tidy_string handles slashes with spaces."""
    assert tidy_string("path/ subpath") == "path/subpath"
    assert tidy_string("a/ b/ c") == "a/b/c"


def test_make_bool_with_false_strings() -> None:
    """Test make_bool handles FALSE strings."""
    assert make_bool("F") is False
    assert make_bool("FALSE") is False
    assert make_bool("0") is False
    assert make_bool("  false  ") is False  # with whitespace


def test_normalise_dt_param_str_with_range() -> None:
    """Test normalise_dt_param_str with date range."""
    result = normalise_dt_param_str("2024-01-01:2024-01-31")
    assert len(result) == 2
    assert result[0] == "2024-01-01"
    assert result[1] == "2024-01-31"


def test_normalise_dt_param_str_single_date() -> None:
    """Test normalise_dt_param_str with single date."""
    result = normalise_dt_param_str("2024-01-15")
    assert len(result) == 1
    assert result[0] == "2024-01-15"


def test_make_list_from_comma_separated() -> None:
    """Test make_list splits comma-separated values."""
    result = make_list("value1, value2, value3")
    assert result == ["value1", "value2", "value3"]
    
    # Test with extra spaces
    result2 = make_list("  val1  ,  val2  ")
    assert result2 == ["val1", "val2"]


def test_normalise_dt_param_str_empty_parts() -> None:
    """Test normalise_dt_param_str with range having empty end."""
    result = normalise_dt_param_str("2024-01-01:")
    assert len(result) == 2
    assert result[0] == "2024-01-01"
    assert result[1] == ""


def test_normalise_dt_param_with_datetime_min_time() -> None:
    """Test _normalise_dt_param with datetime having min time returns date only."""
    from datetime import datetime as dt_class
    
    dt_no_time = dt_class(2024, 1, 15)  # No time component, defaults to 00:00:00
    result = _normalise_dt_param(dt_no_time)
    assert result == "2024-01-15"


def test_normalise_dt_param_with_date_object() -> None:
    """Test _normalise_dt_param with date object."""
    import datetime
    
    d = datetime.date(2024, 3, 25)
    result = _normalise_dt_param(d)
    assert result == "2024-03-25"


def test_tidy_string_multiple_operations() -> None:
    """Test tidy_string performs all tidying operations."""
    # Test with multiple spaces and slashes
    input_str = "  hello    world  /  path  "
    result = tidy_string(input_str)
    assert result == "hello world /path"


def test_make_bool_with_true_values() -> None:
    """Test make_bool with various true values."""
    assert make_bool("TRUE") is True
    assert make_bool("true") is True
    assert make_bool("1") is True
    assert make_bool(1) is True
    assert make_bool("yes") is True


def test_distribution_to_filename_trailing_slash_removal() -> None:
    """Test that distribution_to_filename removes trailing slashes from datasetseries."""
    root_dir = "/data"
    dataset = "test_dataset"
    datasetseries_with_slash = "2024-01-01/"
    datasetseries_with_backslash = "2024-01-01\\"
    file_format = "csv"
    catalog = "test_catalog"
    
    result1 = distribution_to_filename(root_dir, dataset, datasetseries_with_slash, file_format, catalog)
    result2 = distribution_to_filename(root_dir, dataset, datasetseries_with_backslash, file_format, catalog)
    
    expected = f"{root_dir}/{dataset}__{catalog}__2024-01-01.{file_format}"
    assert result1 == expected
    assert result2 == expected


def test_normalise_dt_param_various_formats() -> None:
    """Test _normalise_dt_param with various date string formats."""
    # Test YYYYMMDD format
    assert _normalise_dt_param("20240115") == "2024-01-15"
    
    # Test YYYY-MM-DD format
    assert _normalise_dt_param("2024-01-15") == "2024-01-15"
    
    # Test with time component - should return timestamp format
    result = _normalise_dt_param("2024-01-15 14:30:00")
    assert result == "2024-01-15T14:30:00"
    
    # Test with time and T separator
    result2 = _normalise_dt_param("2024-01-15T14:30:00")
    assert result2 == "2024-01-15T14:30:00"


def test_make_bool_edge_cases() -> None:
    """Test make_bool with various edge cases."""
    # Test with integer 0
    assert make_bool(0) is False
    
    # Test with integer 1
    assert make_bool(1) is True
    
    # Test with empty string
    assert make_bool("") is False
    
    # Test with non-empty string
    assert make_bool("anything") is True


def test_tidy_string_edge_cases() -> None:
    """Test tidy_string with various edge cases."""
    # Already tidy string
    assert tidy_string("clean") == "clean"
    
    # Only spaces
    assert tidy_string("     ") == ""
    
    # Multiple operations needed
    assert tidy_string("  a  b  / c  ") == "a b /c"


def test_snake_to_camel_simple() -> None:
    """Test snake_to_camel with simple cases."""
    # No underscores
    assert snake_to_camel("test") == "test"
    
    # Multiple underscores
    assert snake_to_camel("this_is_a_test") == "thisIsATest"
    
    # Uppercase input
    assert snake_to_camel("TEST_VALUE") == "testValue"


def test_distribution_to_url_sample() -> None:
    """Test distribution_to_url with sample datasetseries."""
    root_url = "https://api.example.com/"
    dataset = "my_dataset"
    datasetseries = "sample"
    file_format = "csv"
    catalog = "test_catalog"
    
    result = distribution_to_url(root_url, dataset, datasetseries, file_format, catalog)
    
    # Sample data uses special endpoint
    assert "/sample/distributions/" in result
    assert f"catalogs/{catalog}" in result
    assert f"datasets/{dataset}" in result


def test_normalise_dt_param_str_error_handling() -> None:
    """Test normalise_dt_param_str with invalid input."""
    # Too many colons should raise ValueError
    with pytest.raises(ValueError, match="Unable to parse"):
        normalise_dt_param_str("2024-01-01:2024-01-15:extra")


def test_normalise_dt_param_error_handling() -> None:
    """Test _normalise_dt_param with invalid input."""
    # Invalid type should raise ValueError
    with pytest.raises(ValueError, match="not in a recognised data format"):
        _normalise_dt_param([])  # type: ignore
    
    # Invalid date string should raise ValueError
    with pytest.raises(ValueError, match="not in a recognised data format"):
        _normalise_dt_param("not-a-date")

