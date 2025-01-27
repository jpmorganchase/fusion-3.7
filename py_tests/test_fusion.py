import datetime
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pandas as pd
import pytest
import requests
import requests_mock

from fusion37 import Fusion
from fusion37.credentials import FusionCredentials
from fusion37.utils import _normalise_dt_param


@pytest.fixture
def mock_response_data() -> Dict[str, Any]:
    """Fixture providing mock API response data."""
    return {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}


def test_call_for_dataframe(mock_response_data: Dict[str, Any]) -> None:
    """Test `_call_for_dataframe` static method."""
    url = "https://api.example.com/data"

    with patch("requests.Session.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response_data
        mock_get.return_value.raise_for_status = lambda: None

        session = requests.Session()
        df = Fusion._call_for_dataframe(url, session)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["id", "name"]
        assert df.iloc[0]["id"] == 1
        assert df.iloc[0]["name"] == "Resource 1"


def test_call_for_bytes_object() -> None:
    """Test `_call_for_bytes_object` static method."""
    url = "https://api.example.com/file"
    mock_content = b"Mock file content"

    with patch("requests.Session.get") as mock_get:
        mock_get.return_value.content = mock_content
        mock_get.return_value.raise_for_status = lambda: None

        session = requests.Session()
        byte_obj = Fusion._call_for_bytes_object(url, session)

        assert isinstance(byte_obj, BytesIO)
        assert byte_obj.read() == mock_content


def test_fusion_init_with_credentials(example_creds_dict: Dict[str, str]) -> None:
    """Test `Fusion` class initialization with credentials."""
    credentials = FusionCredentials(bearer_token=example_creds_dict['token'])
    fusion = Fusion(credentials=credentials)
    assert isinstance(fusion, Fusion)
    assert fusion.root_url == "https://fusion.jpmorgan.com/api/v1/"
    assert fusion.download_folder == "downloads"


def test_fusion_init_with_path(example_creds_dict: Dict[str, str], tmp_path: Path) -> None:
    """Test `Fusion` class initialization with a credentials file."""
    example_creds_dict.update({
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
        "username": "test_user",
        "password": "test_password",
    })
    credentials_file = tmp_path / "credentials.json"
    with credentials_file.open("w") as f:
        json.dump(example_creds_dict, f)

    fusion = Fusion(credentials=str(credentials_file))
    assert isinstance(fusion, Fusion)
    assert fusion.root_url == "https://fusion.jpmorgan.com/api/v1/"
    assert fusion.download_folder == "downloads"


def test_fusion_repr(example_creds_dict: Dict[str, str]) -> None:
    """Test the `__repr__` method of the `Fusion` class."""
    credentials = FusionCredentials(bearer_token=example_creds_dict['token'])
    fusion = Fusion(credentials=credentials)
    repr_str = repr(fusion)
    assert "Fusion object" in repr_str
    assert "Available methods" in repr_str


def test_default_catalog_property(example_creds_dict: Dict[str, str]) -> None:
    """Test the `default_catalog` property of the `Fusion` class."""
    credentials = FusionCredentials(bearer_token=example_creds_dict['token'])
    fusion = Fusion(credentials=credentials)
    assert fusion.default_catalog == "common"

    fusion.default_catalog = "new_catalog"
    assert fusion.default_catalog == "new_catalog"


def test_use_catalog(example_creds_dict: Dict[str, str]) -> None:
    """Test the `_use_catalog` method."""
    credentials = FusionCredentials(bearer_token=example_creds_dict['token'])
    fusion = Fusion(credentials=credentials)
    fusion.default_catalog = "default_cat"

    assert fusion._use_catalog(None) == "default_cat"
    assert fusion._use_catalog("specific_cat") == "specific_cat"

def test_date_parsing() -> None:
    assert _normalise_dt_param(20201212) == "2020-12-12"
    assert _normalise_dt_param("20201212") == "2020-12-12"
    assert _normalise_dt_param("2020-12-12") == "2020-12-12"
    assert _normalise_dt_param(datetime.date(2020, 12, 12)) == "2020-12-12"
    dtm = datetime.datetime(2020, 12, 12, 23, 55, 59, 342380, tzinfo=datetime.timezone.utc)
    assert _normalise_dt_param(dtm) == "2020-12-12"

@pytest.mark.parametrize("ref_int", [-1, 0, 1, 2])
@pytest.mark.parametrize("pluraliser", [None, "s", "es"])
def test_res_plural(ref_int: int, pluraliser: str) -> None:
    from fusion37.authentication import _res_plural

    res = _res_plural(ref_int, pluraliser)
    if abs(ref_int) == 1:
        assert res == ""
    else:
        assert res == pluraliser

def test_is_url() -> None:
    from fusion37.authentication import _is_url

    assert _is_url("https://www.google.com")
    assert _is_url("http://www.google.com/some/path?qp1=1&qp2=2")
    assert not _is_url("www.google.com")
    assert not _is_url("google.com")
    assert not _is_url("google")
    assert not _is_url("googlecom")
    assert not _is_url("googlecom.")
    assert not _is_url(3.141)  # type: ignore

def test_fusion_class(fusion_obj: Fusion) -> None:
    assert fusion_obj
    assert repr(fusion_obj)
    assert fusion_obj.default_catalog == "common"
    fusion_obj.default_catalog = "other"
    assert fusion_obj.default_catalog == "other"