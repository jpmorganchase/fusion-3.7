import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from pyfusion.fusion import Fusion


@pytest.fixture
def example_creds_dict() -> Dict[str, str]:
    """Fixture providing example credentials."""
    return {
        "token": "test_token",
    }


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


def test_create_session(example_creds_dict: Dict[str, str]) -> None:
    """Test `_create_session` static method."""
    session = Fusion._create_session(example_creds_dict)
    assert isinstance(session, requests.Session)
    assert session.headers["Authorization"] == f"Bearer {example_creds_dict['token']}"


def test_fusion_init_with_credentials(example_creds_dict: Dict[str, str]) -> None:
    """Test `Fusion` class initialization with credentials."""
    fusion = Fusion(credentials=example_creds_dict)
    assert isinstance(fusion, Fusion)
    assert fusion.root_url == "https://fusion.jpmorgan.com/api/v1/"
    assert fusion.download_folder == "downloads"


def test_fusion_init_with_path(example_creds_dict: Dict[str, str], tmp_path: Path) -> None:
    """Test `Fusion` class initialization with a credentials file."""
    credentials_file = tmp_path / "credentials.json"
    with credentials_file.open("w") as f:
        json.dump(example_creds_dict, f)

    fusion = Fusion(credentials=str(credentials_file))
    assert isinstance(fusion, Fusion)
    assert fusion.root_url == "https://fusion.jpmorgan.com/api/v1/"
    assert fusion.download_folder == "downloads"


def test_fusion_repr(example_creds_dict: Dict[str, str]) -> None:
    """Test the `__repr__` method of the `Fusion` class."""
    fusion = Fusion(credentials=example_creds_dict)
    repr_str = repr(fusion)
    assert "Fusion object" in repr_str
    assert "Available methods" in repr_str


def test_default_catalog_property(example_creds_dict: Dict[str, str]) -> None:
    """Test the `default_catalog` property of the `Fusion` class."""
    fusion = Fusion(credentials=example_creds_dict)
    assert fusion.default_catalog == "common"

    fusion.default_catalog = "new_catalog"
    assert fusion.default_catalog == "new_catalog"


def test_use_catalog(example_creds_dict: Dict[str, str]) -> None:
    """Test the `_use_catalog` method."""
    fusion = Fusion(credentials=example_creds_dict)
    fusion.default_catalog = "default_cat"

    assert fusion._use_catalog(None) == "default_cat"
    assert fusion._use_catalog("specific_cat") == "specific_cat"
