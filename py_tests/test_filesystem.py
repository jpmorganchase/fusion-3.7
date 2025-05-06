import io
import json
from pathlib import Path
from typing import Any, Dict
from unittest import mock
from unittest.mock import MagicMock, patch

import asynctest
import fsspec
import pytest
from aiohttp import ClientResponse
from asynctest import CoroutineMock
from pytest_mock import MockerFixture
from typing_extensions import Literal

from fusion.credentials import FusionCredentials
from fusion.exceptions import APIResponseError
from fusion.fusion_filesystem import FusionHTTPFileSystem

AsyncMock = asynctest.CoroutineMock

@pytest.fixture
def http_fs_instance(credentials_examples: Path) -> FusionHTTPFileSystem:
    """Fixture to create a new instance for each test."""
    creds = FusionCredentials.from_file(credentials_examples)
    return FusionHTTPFileSystem(credentials=creds)


def test_filesystem(
    example_creds_dict: Dict[str, Any], example_creds_dict_https_pxy: Dict[str, Any], tmp_path: Path
) -> None:
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)
    assert FusionHTTPFileSystem(creds)

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict_https_pxy, f)
    creds = FusionCredentials.from_file(credentials_file)
    assert FusionHTTPFileSystem(creds)

    kwargs = {"client_kwargs": {"credentials": creds}}

    assert FusionHTTPFileSystem(None, **kwargs)

    kwargs = {"client_kwargs": {"credentials": 3.14}}  # type: ignore
    with pytest.raises(ValueError, match="Credentials not provided"):
        FusionHTTPFileSystem(None, **kwargs)


@pytest.mark.asyncio
async def test_not_found_status(http_fs_instance: FusionHTTPFileSystem) -> None:
    response = asynctest.MagicMock(spec=ClientResponse)
    response.status = 404
    response.text = asynctest.CoroutineMock(return_value="404 NotFound")

    with pytest.raises(FileNotFoundError):
        await http_fs_instance._async_raise_not_found_for_status(response, "http://example.com")

@pytest.mark.asyncio
async def test_other_error_status(credentials: FusionCredentials) -> None:
    response = mock.MagicMock(spec=ClientResponse)
    response.status = 500
    response.text = asynctest.CoroutineMock(return_value="Internal server error")

    instance = FusionHTTPFileSystem(credentials)

    with mock.patch.object(instance, "_raise_not_found_for_status", side_effect=Exception("Custom error")):
        with pytest.raises(Exception, match="Custom error"):
            await instance._async_raise_not_found_for_status(response, "http://example.com")

        response.text.assert_awaited_once()
        assert response.reason == "Internal server error", "The reason should be updated to the response text"


@pytest.mark.asyncio
async def test_successful_status(http_fs_instance: FusionHTTPFileSystem) -> None:
    response = mock.MagicMock(spec=ClientResponse)
    response.status = 200
    response.text = asynctest.CoroutineMock(return_value="Success response body")

    try:
        await http_fs_instance._async_raise_not_found_for_status(response, "http://example.com")
    except FileNotFoundError as e:
        pytest.fail(f"No exception should be raised for a successful response, but got: {e}")


@pytest.mark.asyncio
async def test_decorate_url_with_http_async(http_fs_instance: FusionHTTPFileSystem) -> None:
    url = "resource/path"
    exp_res = f"{http_fs_instance.client_kwargs['root_url']}catalogs/{url}"
    result = await http_fs_instance._decorate_url_a(url)
    assert result == exp_res


@pytest.mark.asyncio
async def test_isdir_true(http_fs_instance: FusionHTTPFileSystem) -> None:
    http_fs_instance._decorate_url = asynctest.CoroutineMock(return_value="decorated_path_dir")  # type: ignore
    http_fs_instance._info = asynctest.CoroutineMock(return_value={"type": "directory"})
    result = await http_fs_instance._isdir("path_dir")
    assert result


@pytest.mark.asyncio
async def test_isdir_false(http_fs_instance: FusionHTTPFileSystem) -> None:
    http_fs_instance._decorate_url = asynctest.CoroutineMock(return_value="decorated_path_file")  # type: ignore
    http_fs_instance._info = asynctest.CoroutineMock(return_value={"type": "file"})
    result = await http_fs_instance._isdir("path_file")
    assert not result

@pytest.mark.asyncio
async def test_check_sess_open(http_fs_instance: FusionHTTPFileSystem) -> None:
    new_fs_session_closed =  not http_fs_instance._check_session_open()
    assert new_fs_session_closed

    # Corresponds to running .set_session()
    session_mock = MagicMock()
    session_mock.closed = False
    http_fs_instance._session = session_mock
    fs_session_open = http_fs_instance._check_session_open()
    assert fs_session_open

    # Corresponds to situation where session was closed
    session_mock2 = MagicMock()
    session_mock2.closed = True
    http_fs_instance._session = session_mock2
    fs_session_closed = not http_fs_instance._check_session_open()
    assert fs_session_closed


@pytest.mark.asyncio
async def test_async_startup(http_fs_instance: FusionHTTPFileSystem) -> None:
    http_fs_instance._session = None

    with patch("fusion.fusion_filesystem.FusionHTTPFileSystem.set_session") as SetSessionMock:
        with pytest.raises(RuntimeError) as re:
            await http_fs_instance._async_startup()

        SetSessionMock.assert_called_once()
        assert "FusionFS session closed before operation" in str(re.value)

    # Mock an open session
    MockClient = MagicMock()
    MockClient.closed = False
    http_fs_instance._session = MockClient

    with patch("fusion.fusion_filesystem.FusionHTTPFileSystem.set_session") as SetSessionMock2:
        await http_fs_instance._async_startup()
        SetSessionMock2.assert_called_once()


@pytest.mark.asyncio
async def test_exists_methods(http_fs_instance: FusionHTTPFileSystem) -> None:
    with patch("fusion.fusion_filesystem.HTTPFileSystem.exists") as MockExists:
        MockExists.return_value = True
        exists_out = http_fs_instance.exists("dummy_path")
        MockExists.assert_called_once()
        assert exists_out

    with patch(
            "fusion.fusion_filesystem.HTTPFileSystem._exists", new_callable=CoroutineMock # type: ignore
        ) as Mock_Exists:
            with patch(
                "fusion.fusion_filesystem.FusionHTTPFileSystem._async_startup", new_callable=CoroutineMock # type: ignore
            ) as MockStartup:
                Mock_Exists.return_value = True
                _exists_out = await http_fs_instance._exists("dummy_path")

            assert Mock_Exists.await_count == 1
            assert MockStartup.await_count == 1
            assert _exists_out

@patch("requests.Session")
def test_stream_single_file(mock_session_class: MagicMock, example_creds_dict: Dict[str, Any], tmp_path: Path) -> None:
    url = "http://example.com/data"
    output_file = MagicMock(spec=fsspec.spec.AbstractBufferedFile)
    output_file.path = "./output_file_path/file.txt"
    output_file.name = "file.txt"

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Create a mock response object with the necessary context manager methods
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content = MagicMock(return_value=[b"0123456789", b""])
    # Mock the __enter__ method to return the mock response itself
    mock_response.__enter__.return_value = mock_response
    # Mock the __exit__ method to do nothing
    mock_response.__exit__.return_value = None

    # Set up the mock session to return the mock response
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    # Create an instance of FusionHTTPFileSystem
    http_fs_instance = FusionHTTPFileSystem(credentials=creds)
    http_fs_instance.sync_session = mock_session

    # Run the function
    results = http_fs_instance.stream_single_file(url, output_file)

    # Assertions to verify the behavior
    output_file.write.assert_any_call(b"0123456789")
    assert results == (True, output_file.path, None)


@patch("requests.Session")
def test_stream_single_file_exception(
    mock_session_class: MagicMock, example_creds_dict: Dict[str, Any], tmp_path: Path
) -> None:
    url = "http://example.com/data"
    output_file = MagicMock(spec=fsspec.spec.AbstractBufferedFile)
    output_file.path = "./output_file_path/file.txt"
    output_file.name = "file.txt"

    # Create a mock response object with the necessary context manager methods
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content = MagicMock(side_effect=Exception("Test exception"))
    # Mock the __enter__ method to return the mock response itself
    mock_response.__enter__.return_value = mock_response
    # Mock the __exit__ method to do nothing
    mock_response.__exit__.return_value = None

    # Set up the mock session to return the mock response
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Create an instance of FusionHTTPFileSystem
    http_fs_instance = FusionHTTPFileSystem(credentials=creds)
    http_fs_instance.sync_session = mock_session

    # Run the function and catch the exception
    results = http_fs_instance.stream_single_file(url, output_file)

    # Assertions to verify the behavior
    output_file.close.assert_called_once()
    assert results == (False, output_file.path, "Test exception")

@pytest.mark.asyncio
async def test_download_single_file_async(
    mocker: MockerFixture,  # Add type annotation for mocker
    example_creds_dict: Dict[str, Any], 
    tmp_path: Path
) -> None:
    # Patching the function _run_coros_in_chunks with a mock
    mock_run_coros_in_chunks = mocker.patch("fsspec.asyn._run_coros_in_chunks", new_callable=mocker.AsyncMock)
    
    mock_run_coros_in_chunks.return_value = [True, True, True]

    url = "http://example.com/data"
    output_file = MagicMock(spec=io.IOBase)
    output_file.path = "./output_file_path/file.txt"
    file_size = 20
    chunk_size = 10
    n_threads = 3

    # Mock credentials file
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Create the FusionHTTPFileSystem instance
    http_fs_instance = FusionHTTPFileSystem(credentials=creds)
    http_fs_instance.set_session = mocker.AsyncMock(return_value=mocker.AsyncMock())
    http_fs_instance._fetch_range = mocker.AsyncMock()

    # Run the download function
    result = await http_fs_instance._download_single_file_async(url, output_file, file_size, chunk_size, n_threads)

    # Assertions for successful download
    assert result == (True, output_file.path, None)
    output_file.close.assert_called_once()

    # Simulate an exception during the function call
    mock_run_coros_in_chunks.return_value = [Exception("Test exception")]
    result = await http_fs_instance._download_single_file_async(url, output_file, file_size, chunk_size, n_threads)

    # Assertions for failed download
    assert result == (False, output_file.path, "Test exception")
    output_file.close.assert_called()

@pytest.mark.asyncio
async def test_fetch_range_exception(
    mocker: MockerFixture,  # Add type annotation for mocker
    example_creds_dict: Dict[str, Any], 
    tmp_path: Path
) -> None:
    # Mock output file
    output_file = MagicMock(spec=io.IOBase)
    output_file.path = "./output_file_path/file.txt"
    output_file.seek = MagicMock()
    output_file.write = MagicMock()

    # Mock the response with an exception on read
    mock_response = mocker.MagicMock()
    mock_response.raise_for_status = mocker.MagicMock()
    mock_response.read = mocker.MagicMock(side_effect=Exception("Test exception"))
    mock_response.status = 500
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = None

    # Mock the ClientSession
    mock_client_session = mocker.MagicMock()
    mock_client_session.get.return_value = mock_response
    mocker.patch("aiohttp.ClientSession", return_value=mock_client_session)

    # Mock credentials file
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Create the FusionHTTPFileSystem instance
    http_fs_instance = FusionHTTPFileSystem(credentials=creds)
    http_fs_instance.kwargs = {}

    # Run the function (assuming you're testing a method that handles this exception)
    # You should assert what the function would do in the case of an exception
    await http_fs_instance._fetch_range(mock_client_session, "http://example.com/data", 0, 10, output_file)

    # Assertions
    output_file.seek.assert_not_called()
    output_file.write.assert_not_called()

@pytest.mark.asyncio
async def test_fetch_range_success(
    mocker: MockerFixture,  # Add type annotation for mocker
    example_creds_dict: Dict[str, Any], 
    tmp_path: Path
) -> None:
    url = "http://example.com/data"
    output_file = MagicMock(spec=io.IOBase)
    output_file.path = "./output_file_path/file.txt"
    output_file.seek = MagicMock()
    output_file.write = MagicMock()
    start = 0
    end = 10

    # Mocking the response
    mock_response = mocker.MagicMock()
    mock_response.raise_for_status = mocker.MagicMock()
    mock_response.read = mocker.MagicMock(return_value=b"some data")
    mock_response.status = 200
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = None

    # Mocking the session
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mocker.patch("aiohttp.ClientSession", return_value=mock_session)

    # Mocking credentials file
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Creating the FusionHTTPFileSystem instance
    http_fs_instance = FusionHTTPFileSystem(credentials=creds)
    http_fs_instance.kwargs = {}

    # Run the function
    await http_fs_instance._fetch_range(mock_session, url, start, end, output_file)

    # Assertions
    output_file.seek.assert_called_once_with(0)
    output_file.write.assert_called_once_with(b"some data")
    mock_response.raise_for_status.assert_not_called()
    mock_session.get.assert_called_once_with(url + f"?downloadRange=bytes={start}-{end-1}", **http_fs_instance.kwargs)


@pytest.mark.parametrize(
    ("n_threads", "is_local_fs", "expected_method"),
    [
        (10, False, "stream_single_file"),
        (10, True, "_download_single_file_async"),
    ],
)
@mock.patch("fusion.utils.get_default_fs")
@mock.patch("fsspec.asyn.sync")
@mock.patch.object(FusionHTTPFileSystem, "stream_single_file", new_callable=AsyncMock)
@mock.patch.object(FusionHTTPFileSystem, "_download_single_file_async", new_callable=AsyncMock)
def test_get(  # noqa: PLR0913
    mock_download_single_file_async:AsyncMock,
    mock_stream_single_file: AsyncMock,
    mock_sync: AsyncMock,
    mock_get_default_fs: MagicMock,
    n_threads: int,
    is_local_fs: bool,
    expected_method: str,
    example_creds_dict: Dict[str, Any],
    tmp_path: Path,
) -> None:
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Arrange
    fs = FusionHTTPFileSystem(credentials=creds)
    rpath = "http://example.com/data"
    chunk_size = 5 * 2**20
    kwargs = {"n_threads": n_threads, "is_local_fs": is_local_fs, "headers": {"Content-Length": "100"}}

    mock_file = AsyncMock(spec=fsspec.spec.AbstractBufferedFile)
    mock_default_fs = MagicMock()
    mock_default_fs.open.return_value = mock_file
    mock_get_default_fs.return_value = mock_default_fs
    mock_sync.side_effect = lambda _, func, *args, **kwargs: func(*args, **kwargs)
    
    # Act
    _ = fs.get(rpath, mock_file, chunk_size, **kwargs)

    # Assert
    if expected_method == "stream_single_file":
        mock_stream_single_file.assert_called_once_with(str(rpath), mock_file, block_size=chunk_size)
        mock_download_single_file_async.assert_not_called()
    else:
        mock_download_single_file_async.assert_called_once_with(
            str(rpath) + "/operationType/download", mock_file, 100, chunk_size, n_threads
        )
        mock_stream_single_file.assert_not_called()

    mock_get_default_fs.return_value.open.assert_not_called()

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overwrite", "preserve_original_name", "expected_lpath"),
    [
        (True, False, "local_file.txt"),
        (False, False, "local_file.txt"),
        (True, True, "original_file.txt"),
        (False, True, "original_file.txt"),
    ],
)
@patch.object(FusionHTTPFileSystem, "get", return_value=("mocked_return", "mocked_lpath", "mocked_extra"))
@patch.object(FusionHTTPFileSystem, "set_session", new_callable=asynctest.CoroutineMock)
@patch("fsspec.AbstractFileSystem", autospec=True)
@patch("aiohttp.ClientSession")
def test_download(  # noqa: PLR0913
    mock_client_session: MagicMock,
    mock_fs_class: MagicMock,
    mock_set_session: asynctest.CoroutineMock,
    mock_get: MagicMock,
    overwrite: bool,
    preserve_original_name: bool,
    expected_lpath: Literal["local_file.txt", "original_file.txt"],
    example_creds_dict: Dict[str, Any],
    tmp_path: Path,
) -> None:
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Arrange
    fs = FusionHTTPFileSystem(credentials=creds)
    lfs = mock_fs_class.return_value
    rpath = "http://example.com/data"
    lpath = "local_file.txt"
    chunk_size = 5 * 2**20

    mock_session = asynctest.CoroutineMock()
    mock_set_session.return_value = mock_session

    # Use MagicMock and explicitly set async context manager for Python 3.7
    mock_response = MagicMock()
    mock_response.__aenter__ = asynctest.CoroutineMock(return_value=mock_response)
    mock_response.__aexit__ = asynctest.CoroutineMock(return_value=None)
    mock_response.raise_for_status = asynctest.CoroutineMock()
    mock_response.headers = {"Content-Length": "100", "x-jpmc-file-name": "original_file.txt"}

    mock_session.head.return_value = mock_response 

    # Act
    result = fs.download(lfs, rpath, lpath, chunk_size, overwrite, preserve_original_name)

    # Assert
    if overwrite:
        assert result == ("mocked_return", "mocked_lpath", "mocked_extra")
        mock_get.assert_called_once_with(
            str(rpath),
            lfs.open(expected_lpath, "wb"),
            chunk_size=chunk_size,
            headers={"Content-Length": "100", "x-jpmc-file-name": "original_file.txt"},
            is_local_fs=False,
        )
    elif preserve_original_name:
        assert result == (True, Path(expected_lpath), None)
    else:
        assert result == (True, lpath, None)

@patch.object(FusionHTTPFileSystem, "get", return_value=("mocked_return", "mocked_lpath", "mocked_extra"))
@patch.object(FusionHTTPFileSystem, "set_session", new_callable=asynctest.CoroutineMock)
@patch("fsspec.AbstractFileSystem", autospec=True)
def test_raise_not_found_for_status_raises_wrapped_exception(
    mock_fs_class: MagicMock,
    mock_set_session: asynctest.CoroutineMock,
    mock_get: MagicMock,
    example_creds_dict: Dict[str, Any],
    tmp_path: Path,
) -> None:
    response = MagicMock()
    response.status = 403
    original_exception = ValueError("original error")
    response.raise_for_status.side_effect = original_exception  # ⬅️ simulate internal failure

    credentials_file = tmp_path / "client_credentials.json"
    with credentials_file.open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    fs = FusionHTTPFileSystem(credentials=creds)

    with pytest.raises(APIResponseError) as exc_info:
        fs._raise_not_found_for_status(response, "http://example.com")

    assert exc_info.value.status_code == 403
    assert "Error when accessing http://example.com" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)

@pytest.mark.asyncio
async def test_changes_returns_valid_json() -> None:
    fs = FusionHTTPFileSystem(credentials={})
    fs._decorate_url = lambda x: x
    fs.set_session = AsyncMock()

    mock_session = AsyncMock()
    fs.set_session.return_value = mock_session

    mock_response = AsyncMock()
    mock_response.json.return_value = {"key": "value"}
    mock_response.__aenter__.return_value = mock_response
    mock_session.get.return_value = mock_response

    fs._raise_not_found_for_status = MagicMock()

    result = await fs._changes("http://test.com/api/changes")
    assert result == {"key": "value"}
    fs._raise_not_found_for_status.assert_called_once_with(mock_response, "http://test.com/api/changes")

@pytest.mark.asyncio
async def test_changes_json_parsing_failure() -> None:
    fs = FusionHTTPFileSystem(credentials={})
    fs._decorate_url = lambda x: x
    fs.set_session = AsyncMock()
    fs._raise_not_found_for_status = MagicMock()

    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.json.side_effect = ValueError("bad json")
    mock_response.__aenter__.return_value = mock_response
    mock_session.get.return_value = mock_response
    fs.set_session.return_value = mock_session

    result = await fs._changes("http://test.com/api/changes")
    assert result == {}

@pytest.mark.asyncio
async def test_changes_raises_exception_on_failure() -> None:
    fs = FusionHTTPFileSystem(credentials={})
    fs._decorate_url = lambda x: x
    fs.set_session = AsyncMock(side_effect=RuntimeError("bad session"))

    with pytest.raises(RuntimeError):
        await fs._changes("http://test.com/api/changes")

@pytest.mark.asyncio
async def test_ls_real_distribution_file() -> None:
    fs = FusionHTTPFileSystem(credentials={})
    fs.set_session = AsyncMock()

    fs.client_kwargs = {"root_url": "https://mysite.com/"}
    fs.kwargs = {}

    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.headers = {"Content-Length": "1234"}
    mock_response.__aenter__.return_value = mock_response
    fs.set_session.return_value = mock_session
    fs._raise_not_found_for_status = MagicMock()

    mock_session.head.return_value = mock_response

    url = "https://mysite.com/catalogs/abc/distributions/123/resources/456/data.csv"
    result = await fs._ls_real(url, detail=True)

    assert result == [{"name": "abc-123-456.data.csv", "size": 1234, "type": "file"}]

@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "get", return_value=("mocked_return", "mocked_lpath", "mocked_extra"))
@patch.object(FusionHTTPFileSystem, "set_session", new_callable=asynctest.CoroutineMock)
@patch("fsspec.AbstractFileSystem", autospec=True)
async def test_ls_real_directory(
    mock_fs_class: MagicMock,
    mock_set_session: asynctest.CoroutineMock,
    mock_get: MagicMock,
    example_creds_dict: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """
    Test the _ls_real method for directory listing.
    """
    # Mock the session and response for _ls_real
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.json.return_value = {"resources": [{"identifier": "folder1"}, {"identifier": "file2.csv"}]}
    mock_response.__aenter__.return_value = mock_response
    mock_session.get.return_value = mock_response
    mock_set_session.return_value = mock_session

    credentials_file = tmp_path / "client_credentials.json"
    with credentials_file.open("w") as f:
        json.dump(example_creds_dict, f)
    
    creds = FusionCredentials.from_file(credentials_file)
    fs = FusionHTTPFileSystem(credentials=creds)

    # Test directory listing without detail
    url = "https://mysite.com/catalogs/dataset/resources"
    result = await fs._ls_real(url, detail=False)
    
    expected = ["https://mysite.com/catalogs/dataset/resources/folder1", "https://mysite.com/catalogs/dataset/resources/file2.csv"]
    assert result == expected

    # Test directory listing with detail
    result = await fs._ls_real(url, detail=True)
    expected_detail = [
        {"name": "https://mysite.com/catalogs/dataset/resources/folder1", "size": None, "type": "directory"},
        {"name": "https://mysite.com/catalogs/dataset/resources/file2.csv", "size": None, "type": "file"},
    ]
    assert result == expected_detail

@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "get", return_value=("mocked_return", "mocked_lpath", "mocked_extra"))
@patch.object(FusionHTTPFileSystem, "set_session", new_callable=asynctest.CoroutineMock)
@patch("fsspec.AbstractFileSystem", autospec=True)
async def test_ls_real_file(
    mock_fs_class: MagicMock,
    mock_set_session: asynctest.CoroutineMock,
    mock_get: MagicMock,
    example_creds_dict: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """
    Test the _ls_real method for file listing.
    """
    # Mock the session and response for a file
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.headers = {"Content-Length": "1234"}
    mock_response.__aenter__.return_value = mock_response
    mock_session.head.return_value = mock_response
    mock_set_session.return_value = mock_session

    credentials_file = tmp_path / "client_credentials.json"
    with credentials_file.open("w") as f:
        json.dump(example_creds_dict, f)
    
    creds = FusionCredentials.from_file(credentials_file)
    fs = FusionHTTPFileSystem(credentials=creds)

    # Test file listing
    url = "https://mysite.com/catalogs/dataset/distributions/123/resources/456/data.csv"
    result = await fs._ls_real(url, detail=True)

    expected = [{"name": "123-456.data.csv", "size": 1234, "type": "file"}]
    assert result == expected

@patch.object(FusionHTTPFileSystem, "_raise_not_found_for_status", side_effect=FileNotFoundError("not found"))
@patch("fsspec.AbstractFileSystem", autospec=True)
@pytest.mark.asyncio
async def test_async_raise_not_found_for_status_404_raises_wrapped_exception(
    mock_fs_class: MagicMock,
    mock_raise: MagicMock,
    example_creds_dict: Dict[str, Any],
    tmp_path: Path,
) -> None:
    response = MagicMock()
    response.status = 404

    credentials_file = tmp_path / "client_credentials.json"
    with credentials_file.open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    fs = FusionHTTPFileSystem(credentials=creds)

    with pytest.raises(APIResponseError) as exc_info:
        await fs._async_raise_not_found_for_status(response, "http://example.com/foo.csv")

    assert exc_info.value.status_code == 404
    assert "Error when accessing http://example.com/foo.csv" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


@patch.object(FusionHTTPFileSystem, "_raise_not_found_for_status", side_effect=RuntimeError("unexpected"))
@patch("fsspec.AbstractFileSystem", autospec=True)
@pytest.mark.asyncio
async def test_async_raise_not_found_for_status_non_404_raises_wrapped_exception(
    mock_fs_class: MagicMock,
    mock_raise: MagicMock,
    example_creds_dict: Dict[str, Any],
    tmp_path: Path,
) -> None:
    response = asynctest.CoroutineMock()
    response.status = 500
    response.text = asynctest.CoroutineMock(return_value="Internal Error")

    credentials_file = tmp_path / "client_credentials.json"
    with credentials_file.open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    fs = FusionHTTPFileSystem(credentials=creds)

    with pytest.raises(APIResponseError) as exc_info:
        await fs._async_raise_not_found_for_status(response, "http://example.com/bar.csv")

    assert exc_info.value.status_code == 500
    assert "Error when accessing http://example.com/bar.csv" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert response.reason == "Internal Error"

@pytest.mark.asyncio
async def test_async_raise_not_found_for_status_403_raises_wrapped_exception() -> None:
    fs = FusionHTTPFileSystem(credentials=None)

    # Simulate response with non-404 that still errors
    response = MagicMock()
    response.status = 403
    response.text = asynctest.CoroutineMock(return_value="Forbidden")
    
    # Trigger actual raise from inside _raise_not_found_for_status by setting response.raise_for_status
    response.raise_for_status.side_effect = ValueError("original error")

    with pytest.raises(APIResponseError) as exc_info:
        await fs._async_raise_not_found_for_status(response, "http://example.com")

    assert exc_info.value.status_code == 403
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "http://example.com" in str(exc_info.value)