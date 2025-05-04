import json
import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generator, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import fsspec
import pytest
from requests.adapters import HTTPAdapter

from fusion.authentication import FusionOAuthAdapter
from fusion.credentials import FusionCredentials
from fusion.exceptions import APIResponseError, CredentialError
from fusion.utils import get_default_fs

from .conftest import change_dir


def test_pickle_fusion_credentials(tmp_path: Path) -> None:
    creds = FusionCredentials.from_client_id(
        client_id="my_client_id",
        client_secret="my_client_secret",
        resource="my_resource",
        auth_url="my_auth_url",
        proxies={},
        fusion_e2e=None,
    )
    creds.put_bearer_token("some_token", 1234)

    creds_file = tmp_path / "creds.pkl"
    with creds_file.open("wb") as file:
        pickle.dump(creds, file)

    with creds_file.open("rb") as file:
        creds_loaded = pickle.load(file)

    assert isinstance(creds_loaded, FusionCredentials)
    assert creds_loaded.client_id == "my_client_id"
    assert creds_loaded.client_secret == "my_client_secret"
    assert creds_loaded.resource == "my_resource"
    assert creds_loaded.auth_url == "my_auth_url"


def test_from_file_relative_path_walkup_exists(tmp_path: Path, good_json: str) -> None:
    dir_down_path = Path(tmp_path / "level_1" / "level_2" / "level_3")
    dir_down_path.mkdir(parents=True)
    (Path(tmp_path / "level_1" / "config")).mkdir()
    credentials_file = dir_down_path.parent.parent / "config" / "client_credentials.json"
    credentials_file.write_text(good_json)

    with change_dir(dir_down_path):
        credentials = FusionCredentials.from_file(file_path=Path("client_credentials.json"))

        assert isinstance(credentials, FusionCredentials)
        assert credentials.client_id
        assert credentials.client_secret


def test_from_file_file_not_found(tmp_path: Path) -> None:
    missing_creds_file = tmp_path / "client_credentials.json"
    with pytest.raises(FileNotFoundError):
        FusionCredentials.from_file(file_path=missing_creds_file)


def test_from_file_empty_file(tmp_path: Path) -> None:
    credentials_file = tmp_path / "client_credentials.json"
    credentials_file.touch()

    with pytest.raises(CredentialError):
        FusionCredentials.from_file(file_path=credentials_file)


def test_from_file_invalid_json(tmp_path: Path) -> None:
    credentials_file = tmp_path / "client_credentials.json"
    credentials_file.write_text('{"client_id": "my_client_id"}')

    with pytest.raises(CredentialError):
        FusionCredentials.from_file(file_path=credentials_file)


class MockResponse:
    def __init__(self, status_code: int = 200, json_data: Optional[Dict[str, Any]] = None) -> None:
        self.status_code = status_code
        self.json_data = json_data

    def json(self) -> Optional[Dict[str, Any]]:
        return self.json_data


@pytest.mark.skip(reason="Legacy code")
def test_fusion_oauth_adapter_send_no_bearer_token_exp(fusion_oauth_adapter: FusionOAuthAdapter) -> None:
    fusion_oauth_adapter.credentials.bearer_token = None
    with pytest.raises(CredentialError):
        fusion_oauth_adapter.send(Mock())


@pytest.fixture
def local_fsspec_fs() -> Generator[Tuple[fsspec.filesystem, str], None, None]:
    with TemporaryDirectory() as tmp_dir, patch("fsspec.filesystem") as mock_fs:
        local_fs = fsspec.filesystem("file", auto_mkdir=True)
        mock_fs.return_value = local_fs
        yield local_fs, tmp_dir


def test_default_filesystem() -> None:
    with patch.dict(os.environ, {}, clear=True), patch("fsspec.filesystem") as mock_fs:
        mock_fs.return_value = MagicMock()
        fs = get_default_fs()
        mock_fs.assert_called_once_with("file")
        assert isinstance(fs, MagicMock)


def test_s3_filesystem() -> None:
    env_vars = {
        "FS_PROTOCOL": "s3",
        "S3_ENDPOINT": "s3.example.com",
        "AWS_ACCESS_KEY_ID": "key",
        "AWS_SECRET_ACCESS_KEY": "secret",
    }
    with patch.dict(os.environ, env_vars), patch("fsspec.filesystem") as mock_fs:
        mock_fs.return_value = MagicMock()
        fs = get_default_fs()
        mock_fs.assert_called_once_with(
            "s3", client_kwargs={"endpoint_url": "https://s3.example.com"}, key="key", secret="secret"
        )
        assert isinstance(fs, MagicMock)


def test_from_object_with_json_file(tmp_path: Path) -> None:
    credentials_file = tmp_path / "credentials.json"
    credentials = {
        "grant_type": "client_credentials",
        "client_id": "my_client_id",
        "client_secret": "my_client_secret",
        "resource": "my_resource",
        "auth_url": "my_auth_url",
    }

    with Path(credentials_file).open("w") as file:
        json.dump(credentials, file)

    creds = FusionCredentials.from_file(credentials_file)

    assert isinstance(creds, FusionCredentials)
    assert creds.client_id == "my_client_id"
    assert creds.client_secret == "my_client_secret"
    assert creds.resource == "my_resource"
    assert creds.auth_url == "my_auth_url"

def test_send_raises_api_response_error_on_generic_exception() -> None:
    request = Mock()
    request.url = "https://example.com/data"
    request.headers = {}

    mock_credentials = Mock(spec=FusionCredentials)
    mock_credentials.get_fusion_token_headers.return_value = {}
    mock_credentials.proxies = {}

    adapter = FusionOAuthAdapter(credentials=mock_credentials)

    with patch.object(HTTPAdapter, "send", side_effect=RuntimeError("Unexpected failure")), \
     pytest.raises(APIResponseError) as exc_info:
        adapter.send(request)

    assert "Unexpected error while sending request" in str(exc_info.value)
    HTTP_STATUS_INTERNAL_SERVER_ERROR = 500
    assert exc_info.value.status_code == HTTP_STATUS_INTERNAL_SERVER_ERROR

def test_send_raises_api_response_error_on_connection_error() -> None:
    request = Mock()
    request.url = "https://example.com/data"
    request.headers = {}

    mock_credentials = Mock(spec=FusionCredentials)
    mock_credentials.get_fusion_token_headers.return_value = {}
    mock_credentials.proxies = {}

    adapter = FusionOAuthAdapter(credentials=mock_credentials)

    with patch.object(HTTPAdapter, "send", side_effect=ConnectionError("Connection failed")), \
     pytest.raises(APIResponseError) as exc_info:
        adapter.send(request)

    assert "Connection error while sending request" in str(exc_info.value)
    HTTP_STATUS_SERVICE_UNAVAILABLE = 503
    assert exc_info.value.status_code == HTTP_STATUS_SERVICE_UNAVAILABLE
