import datetime
import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests
import requests_mock
from pandas.testing import assert_frame_equal
from pytest_mock import MockerFixture

from fusion import Fusion
from fusion.attributes import Attribute, Types
from fusion.credentials import FusionCredentials
from fusion.exceptions import APIResponseError, CredentialError, FileFormatError
from fusion.fusion import logger
from fusion.report import Report
from fusion.utils import _normalise_dt_param, distribution_to_url


@pytest.fixture
def example_creds_dict_token() -> Dict[str, str]:
    """Fixture providing example credentials."""
    return {
        "token": "test_token",
    }


@pytest.fixture
def mock_response_data() -> Dict[str, Any]:
    """Fixture providing mock API response data."""
    return {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}


def test_call_for_dataframe(requests_mock: Any, mock_response_data: Dict[str, Any]) -> None:
    """Test `_call_for_dataframe` static method with requests_mock."""
    url = "https://api.example.com/data"
    requests_mock.get(url, json=mock_response_data)

    session = requests.Session()
    df = Fusion._call_for_dataframe(url, session)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["id", "name"]
    assert df.iloc[0]["id"] == 1
    assert df.iloc[0]["name"] == "Resource 1"


def test_call_for_dataframe_error(requests_mock: Any) -> None:
    """Test `_call_for_dataframe` static method error handling."""
    url = "https://api.example.com/data"
    requests_mock.get(url, status_code=500)
    session = requests.Session()
    with pytest.raises(requests.exceptions.HTTPError):
        Fusion._call_for_dataframe(url, session)


def test_call_for_bytes_object(requests_mock: Any) -> None:
    """Test `_call_for_bytes_object` static method with requests_mock."""
    url = "https://api.example.com/file"
    mock_content = b"Mock file content"
    requests_mock.get(url, content=mock_content)
    session = requests.Session()
    byte_obj = Fusion._call_for_bytes_object(url, session)
    assert isinstance(byte_obj, BytesIO)
    assert byte_obj.read() == mock_content


def test_call_for_bytes_object_fail(requests_mock: Any) -> None:
    """Test `_call_for_bytes_object` static method error handling."""
    url = "https://api.example.com/file"
    requests_mock.get(url, status_code=500)
    session = requests.Session()
    with pytest.raises(requests.exceptions.HTTPError):
        Fusion._call_for_bytes_object(url, session)


def test_fusion_init_with_credentials(example_creds_dict_token: Dict[str, str]) -> None:
    """Test `Fusion` class initialization with credentials."""
    credentials = FusionCredentials(bearer_token=example_creds_dict_token['token'])
    fusion = Fusion(credentials=credentials)
    assert isinstance(fusion, Fusion)
    assert fusion.root_url == "https://fusion.jpmorgan.com/api/v1/"
    assert fusion.download_folder == "downloads"


def test_fusion_init_with_path(example_creds_dict_token: Dict[str, str], tmp_path: Path) -> None:
    """Test `Fusion` class initialization with a credentials file."""
    example_creds_dict_token.update({
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
        "username": "test_user",
        "password": "test_password",
    })
    credentials_file = tmp_path / "credentials.json"
    with credentials_file.open("w") as f:
        json.dump(example_creds_dict_token, f)

    fusion = Fusion(credentials=str(credentials_file))
    assert isinstance(fusion, Fusion)
    assert fusion.root_url == "https://fusion.jpmorgan.com/api/v1/"
    assert fusion.download_folder == "downloads"


def test_fusion_repr(example_creds_dict_token: Dict[str, str]) -> None:
    """Test the `__repr__` method of the `Fusion` class."""
    credentials = FusionCredentials(bearer_token=example_creds_dict_token['token'])
    fusion = Fusion(credentials=credentials)
    repr_str = repr(fusion)
    assert "Fusion object" in repr_str
    assert "Available methods" in repr_str


def test_default_catalog_property(example_creds_dict_token: Dict[str, str]) -> None:
    """Test the `default_catalog` property of the `Fusion` class."""
    credentials = FusionCredentials(bearer_token=example_creds_dict_token['token'])
    fusion = Fusion(credentials=credentials)
    assert fusion.default_catalog == "common"

    fusion.default_catalog = "new_catalog"
    assert fusion.default_catalog == "new_catalog"


def test_use_catalog(example_creds_dict_token: Dict[str, str]) -> None:
    """Test the `_use_catalog` method."""
    credentials = FusionCredentials(bearer_token=example_creds_dict_token['token'])
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

def test_is_url() -> None:
    from fusion.authentication import _is_url

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

def test_get_fusion_filesystem(fusion_obj: Fusion) -> None:
    filesystem = fusion_obj.get_fusion_filesystem()
    assert filesystem is not None

def test__call_for_dataframe_success(requests_mock: requests_mock.Mocker) -> None:
    # Mock the response from the API endpoint
    url = "https://fusion.jpmorgan.com/api/v1/a_given_resource"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Create a mock session
    session = requests.Session()

    # Call the _call_for_dataframe function
    test_df = Fusion._call_for_dataframe(url, session)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)

def test__call_for_dataframe_error(requests_mock: requests_mock.Mocker) -> None:
    # Mock the response from the API endpoint with an error status code
    url = "https://fusion.jpmorgan.com/api/v1/a_given_resource"
    requests_mock.get(url, status_code=500)

    # Create a mock session
    session = requests.Session()

    # Call the _call_for_dataframe function and expect an exception to be raised
    with pytest.raises(requests.exceptions.HTTPError):
        Fusion._call_for_dataframe(url, session)


def test__call_for_bytes_object_success(requests_mock: requests_mock.Mocker) -> None:
    # Mock the response from the API endpoint
    url = "https://fusion.jpmorgan.com/api/v1/a_given_resource"
    expected_data = b"some binary data"
    requests_mock.get(url, content=expected_data)

    # Create a mock session
    session = requests.Session()

    # Call the _call_for_bytes_object function
    data = Fusion._call_for_bytes_object(url, session)

    # Check if the data is returned correctly
    assert data.getbuffer() == expected_data

def test__call_for_bytes_object_fail(requests_mock: requests_mock.Mocker) -> None:
    # Mock the response from the API endpoint with an error status code
    url = "https://fusion.jpmorgan.com/api/v1/a_given_resource"
    requests_mock.get(url, status_code=500)

    # Create a mock session
    session = requests.Session()

    # Call the _call_for_dataframe function and expect an exception to be raised
    with pytest.raises(requests.exceptions.HTTPError):
        Fusion._call_for_bytes_object(url, session)


def test_list_catalogs_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    # Mock the response from the API endpoint
    url = "https://fusion.jpmorgan.com/api/v1/catalogs/"
    expected_data = {"resources": [{"id": 1, "name": "Catalog 1"}, {"id": 2, "name": "Catalog 2"}]}
    requests_mock.get(url, json=expected_data)

    # Call the list_catalogs method
    test_df = fusion_obj.list_catalogs()

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)

def test_list_catalogs_pagination(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    # Mock paginated responses from the API endpoint
    url = "https://fusion.jpmorgan.com/api/v1/catalogs/"
    page1 = {"resources": [{"id": 1, "name": "Catalog 1"}]}
    page2 = {"resources": [{"id": 2, "name": "Catalog 2"}]}
    # First page returns a next token
    requests_mock.get(url, json=page1, headers={"x-jpmc-next-token": "token2"})
    # Second page is requested with the next token
    requests_mock.get(
        url,
        additional_matcher=lambda request: request.headers.get("x-jpmc-next-token") == "token2",
        json=page2
    )

    # Call the list_catalogs method
    test_df = fusion_obj.list_catalogs()

    # Check if the dataframe is created correctly and includes all paginated results
    expected_df = pd.DataFrame([{"id": 1, "name": "Catalog 1"}, {"id": 2, "name": "Catalog 2"}])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_catalogs_fail(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    # Mock the response from the API endpoint with an error status code
    url = "https://fusion.jpmorgan.com/api/v1/catalogs/"
    requests_mock.get(url, status_code=500)

    # Call the list_catalogs method and expect an exception to be raised
    with pytest.raises(requests.exceptions.HTTPError):
        fusion_obj.list_catalogs()

def test_catalog_resources_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    # Mock the response from the API endpoint

    new_catalog = "catalog_id"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Call the catalog_resources method
    test_df = fusion_obj.catalog_resources(new_catalog)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_products_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/products"
    server_mock_data = {
        "resources": [{"category": ["FX"], "region": ["US"]}, {"category": ["FX"], "region": ["US", "EU"]}]
    }
    expected_data = {"resources": [{"category": "FX", "region": "US"}, {"category": "FX", "region": "US, EU"}]}

    requests_mock.get(url, json=server_mock_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_products(catalog=new_catalog, max_results=2)
    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)

def test_list_products_contains_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/products"
    server_mock_data = {
        "resources": [
            {"identifier": "1", "description": "some desc", "category": ["FX"], "region": ["US"]},
            {"identifier": "2", "description": "some desc", "category": ["FX"], "region": ["US", "EU"]},
        ]
    }
    expected_data = {
        "resources": [
            {"identifier": "1", "description": "some desc", "category": "FX", "region": "US"},
        ]
    }
    expected_df = pd.DataFrame(expected_data["resources"])

    requests_mock.get(url, json=server_mock_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_products(catalog=new_catalog, max_results=2, contains=["1"])
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)

    test_df = fusion_obj.list_products(catalog=new_catalog, max_results=2, contains="1")
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)

    test_df = fusion_obj.list_products(catalog=new_catalog, max_results=2, contains="1", id_contains=True)
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)

def test_list_datasets_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets"
    server_mock_data = {
        "resources": [{"category": ["FX"], "region": ["US"]}, {"category": ["FX"], "region": ["US", "EU"]}]
    }
    expected_data = {"resources": [{"region": "US", "category": "FX"}, {"region": "US, EU", "category": "FX"}]}

    requests_mock.get(url, json=server_mock_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2)
    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df, check_like=True)

def test_list_datasets_pagination(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test that list_datasets handles paginated API responses."""
    url = f"{fusion_obj.root_url}catalogs/common/datasets"
    # Simulate two pages of results
    page1 = {
        "resources": [
            {"identifier": "ONE", "name": "Dataset 1", "category": ["FX"], "region": ["US"]}
        ],
        "next": "token2"
    }
    page2 = {
        "resources": [
            {"identifier": "TWO", "name": "Dataset 2", "category": ["FX"], "region": ["EU"]}
        ],
        "next": None
    }

    # First page returns a next token
    requests_mock.get(url, json=page1, headers={"x-jpmc-next-token": "token2"})
    # Second page is requested with the next token
    requests_mock.get(
        url,
        additional_matcher=lambda request: request.headers.get("x-jpmc-next-token") == "token2",
        json=page2
    )

    # Call the list_datasets method
    test_df = fusion_obj.list_datasets()

    # Check if the dataframe is created correctly and includes all paginated results
    expected_df = pd.DataFrame([
        {"identifier": "ONE", "name": "Dataset 1", "category": "FX", "region": "US"},
        {"identifier": "TWO", "name": "Dataset 2", "category": "FX", "region": "EU"}
    ])
    # Only check relevant columns
    pd.testing.assert_frame_equal(
        test_df[["identifier", "category", "region"]].reset_index(drop=True),
        expected_df[["identifier", "category", "region"]].reset_index(drop=True),
        check_like=True
    )

def test_list_datasets_type_filter(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets"
    server_mock_data = {
        "resources": [
            {
                "identifier": "ONE",
                "description": "some desc",
                "category": ["FX"],
                "region": ["US"],
                "status": "active",
                "type": "type1",
            },
            {
                "identifier": "TWO",
                "description": "some desc",
                "category": ["FX"],
                "region": ["US", "EU"],
                "status": "inactive",
                "type": "type2",
            },
        ]
    }
    expected_data = {
        "resources": [
            {
                "identifier": "ONE",
                "region": "US",
                "category": "FX",
                "description": "some desc",
                "status": "active",
                "type": "type1",
            }
        ]
    }

    expected_df = pd.DataFrame(expected_data["resources"])

    requests_mock.get(url, json=server_mock_data)

    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2, dataset_type="type1")

    pd.testing.assert_frame_equal(test_df, expected_df, check_like=True)

def test_list_datasets_contains_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets"
    server_mock_data = {
        "resources": [
            {"identifier": "ONE", "description": "some desc", "category": ["FX"], "region": ["US"], "status": "active"},
            {
                "identifier": "TWO",
                "description": "some desc",
                "category": ["FX"],
                "region": ["US", "EU"],
                "status": "inactive",
            },
        ]
    }
    expected_data = {
        "resources": [
            {"identifier": "ONE", "region": "US", "category": "FX", "description": "some desc", "status": "active"}
        ]
    }

    expected_df = pd.DataFrame(expected_data["resources"])

    requests_mock.get(url, json=server_mock_data)
    requests_mock.get(
    f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets/ONE",
        json={
            "identifier": "ONE",
            "description": "some desc",
            "category": ["FX"],
            "region": ["US"],
            "status": "active",
            "title": None,
            "containerType": None,
            "coverageStartDate": None,
            "coverageEndDate": None,
            "type": None,
        }
    )

    expected_df_exact_match = pd.DataFrame([{
    "identifier": "ONE",
    "title": None,
    "containerType": None,
    "region": ["US"],
    "category": ["FX"],
    "coverageStartDate": None,
    "coverageEndDate": None,
    "description": "some desc",
    "status": "active",
    "type": None
    }])

    select_prod = "prod_a"
    prod_url = f"{fusion_obj.root_url}catalogs/{new_catalog}/productDatasets"
    server_prod_mock_data = {
        "resources": [
            {"product": select_prod, "dataset": "one"},
            {"product": "prod_b", "dataset": "two"},
        ]
    }
    requests_mock.get(prod_url, json=server_prod_mock_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2, contains=["ONE"])
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df, check_like=True)

    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2, contains="ONE")
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df_exact_match, check_like=True)

    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2, contains="ONE", id_contains=True)
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df_exact_match, check_like=True)

    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2, product=select_prod)
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df, check_like=True)

    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2, status="active")
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df, check_like=True)


def test_dataset_resources_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    dataset = "my_dataset"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets/{dataset}"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Call the catalog_resources method
    test_df = fusion_obj.dataset_resources(dataset, new_catalog)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_dataset_attributes(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    dataset = "my_dataset"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets/{dataset}/attributes"

    core_cols = [
        "identifier",
        "title",
        "dataType",
        "isDatasetKey",
        "description",
        "source",
    ]

    server_mock_data = {
        "resources": [
            {
                "index": 0,
                "identifier": "attr_1",
                "title": "some title",
                "dataType": "string",
                "other_meta_attr": "some val",
                "status": "active",
            },
            {
                "index": 1,
                "identifier": "attr_2",
                "title": "some title",
                "dataType": "int",
                "other_meta_attr": "some val",
                "status": "active",
            },
        ]
    }
    expected_data = {
        "resources": [
            {
                "identifier": "attr_1",
                "title": "some title",
                "dataType": "string",
            },
            {
                "identifier": "attr_2",
                "title": "some title",
                "dataType": "int",
            },
        ]
    }

    expected_df = pd.DataFrame(expected_data["resources"])

    requests_mock.get(url, json=server_mock_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_dataset_attributes(dataset, catalog=new_catalog)
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)
    assert all(col in core_cols for col in test_df.columns)

    ext_expected_data = {
        "resources": [
            {
                "index": 0,
                "identifier": "attr_1",
                "title": "some title",
                "dataType": "string",
                "other_meta_attr": "some val",
                "status": "active",
            },
            {
                "index": 1,
                "identifier": "attr_2",
                "title": "some title",
                "dataType": "int",
                "other_meta_attr": "some val",
                "status": "active",
            },
        ]
    }

    ext_expected_df = pd.DataFrame(ext_expected_data["resources"])
    # Call the catalog_resources method
    test_df = fusion_obj.list_dataset_attributes(dataset, catalog=new_catalog, display_all_columns=True)

    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, ext_expected_df)

def test_list_datasetmembers_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    dataset = "my_dataset"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets/{dataset}/datasetseries"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Call the list_datasetmembers method
    test_df = fusion_obj.list_datasetmembers(dataset, new_catalog, max_results=2)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_datasetmember_resources_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    dataset = "my_dataset"
    series = "2022-02-02"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets/{dataset}/datasetseries/{series}"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Call the datasetmember_resources method
    test_df = fusion_obj.datasetmember_resources(dataset, series, new_catalog)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_distributions_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    dataset = "my_dataset"
    series = "2022-02-02"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets/{dataset}/datasetseries/{series}/distributions"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Call the list_distributions method
    test_df = fusion_obj.list_distributions(dataset, series, new_catalog)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test__resolve_distro_tuples(mocker: MockerFixture, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    with pytest.raises(AssertionError), mocker.patch.object(
        fusion_obj, "list_datasetmembers", return_value=pd.DataFrame()
    ):
        fusion_obj._resolve_distro_tuples("dataset", "catalog", "series")

    valid_ds_members = pd.DataFrame(
        {
            "@id": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "identifier": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "dataset": ["dataset", "dataset", "dataset"],
            "createdDate": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    exp_tuples = [
        (catalog, "dataset", "2020-01-01", "parquet"),
        (catalog, "dataset", "2020-01-02", "parquet"),
        (catalog, "dataset", "2020-01-03", "parquet"),
    ]

    with mocker.patch.object(fusion_obj, "list_datasetmembers", return_value=valid_ds_members):
        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog, dt_str="2020-01-01:2020-01-03")
        assert res == exp_tuples

        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog, dt_str=":")
        assert res == exp_tuples

        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog, dt_str="2020-01-01:2020-01-02")
        assert res == exp_tuples[:2]

        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog, dt_str="2020-01-02:2020-01-03")
        assert res == exp_tuples[1:]

        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog)
        assert res == [exp_tuples[-1]]

        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog, dt_str="2020-01-03")
        assert res == [exp_tuples[-1]]

        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog, dt_str="latest")
        assert res == [exp_tuples[-1]]

def test_to_bytes_multiple_files(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    mock_data = {
        "resources": [
            {
                "description": "Sample file 1",
                "fileExtension": ".parquet",
                "identifier": "file1",
                "title": "File 1",
                "@id": "file1"
            },
            {
                "description": "Sample file 2",
                "fileExtension": ".parquet",
                "identifier": "file2",
                "title": "File 2",
                "@id": "file2"
            }
        ]
    }  

    distri_files_url = (
        f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/"
        f"{datasetseries}/distributions/{file_format}/files"
    )

    requests_mock.get(distri_files_url, json=mock_data)
    file1_url = distribution_to_url(
    fusion_obj.root_url,
    dataset,
    datasetseries,
    file_format,
    catalog,
    is_download=True,
    file_name="file1",
    )
    file2_url = distribution_to_url(
    fusion_obj.root_url,
    dataset,
    datasetseries,
    file_format,
    catalog,
    is_download=True,
    file_name="file2",
    )
    expected_data = b"some binary data"
    requests_mock.get(file1_url, content=expected_data)
    requests_mock.get(file2_url, content=expected_data)

    data = fusion_obj.to_bytes(dataset, datasetseries, file_format, catalog)

    # Check if the data is returned correctly
    if isinstance(data, list):
        assert all(isinstance(d, BytesIO) for d in data)
        for d in data:
            assert d.getvalue() == expected_data

def test_to_bytes_single_file(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    mock_data = {
        "resources": [
            {
                "description": "Sample file 1",
                "fileExtension": ".parquet",
                "identifier": "file1",
                "title": "File 1",
                "@id": "file1"
            }
        ]
    }

    distri_files_url = (
        f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/"
        f"{datasetseries}/distributions/{file_format}/files"
    )

    requests_mock.get(distri_files_url, json=mock_data)

    file1_url = distribution_to_url(
        fusion_obj.root_url,
        dataset,
        datasetseries,
        file_format,
        catalog,
        is_download=True,
        file_name="file1",
    )
    expected_data = b"some binary data"
    requests_mock.get(file1_url, content=expected_data)

    data = fusion_obj.to_bytes(dataset, datasetseries, file_format, catalog)

    # Check if the data is returned correctly
    if isinstance(data, BytesIO):
        assert data.getbuffer() == expected_data


def test_to_bytes_with_filename(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"

    file1_url = distribution_to_url(
        fusion_obj.root_url,
        dataset,
        datasetseries,
        file_format,
        catalog,
        is_download=True,
        file_name="file1"
    )
    expected_data = b"some binary data"
    requests_mock.get(file1_url, content=expected_data)

    data = fusion_obj.to_bytes(dataset, datasetseries, file_format, catalog, file_name="file1")

    # Check if the data is returned correctly
    if isinstance(data, BytesIO):
        assert data.getbuffer() == expected_data


def test_validate_format_single_format(mocker: MockerFixture, fusion_obj: Fusion) -> None:
    """Test _validate_format returns the only available format when dataset_format is None."""
    catalog = "my_catalog"
    dataset = "my_dataset"
    
    mock_df = pd.DataFrame({"format": ["csv"]})
    mocker.patch.object(fusion_obj, "list_datasetmembers_distributions", return_value=mock_df)
    
    result = fusion_obj._validate_format(dataset, catalog, None)
    assert result == "csv"


def test_validate_format_valid_format(mocker: MockerFixture, fusion_obj: Fusion) -> None:
    """Test _validate_format returns the requested format when it's available."""
    catalog = "my_catalog"
    dataset = "my_dataset"
    
    mock_df = pd.DataFrame({"format": ["csv", "parquet", "json"]})
    mocker.patch.object(fusion_obj, "list_datasetmembers_distributions", return_value=mock_df)
    
    result = fusion_obj._validate_format(dataset, catalog, "parquet")
    
    assert result == "parquet"


@pytest.mark.skip(reason="MUST FIX")
def test_download_main(mocker: MockerFixture, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    dt_str = "20200101:20200103"
    file_format = "csv"

    # Mock dataset access check
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"
    expected_data = {
        "catalog": {"@id": f"{catalog}/"},
        "title": "Test Dataset",
        "identifier": dataset,
        "status": "Subscribed",  # User has access
        "@id": f"{dataset}/",
    }
    requests_mock.get(url, json=expected_data)
    
    # Mock list_datasetmembers (datasetseries endpoint)
    url_datasetseries = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries"
    requests_mock.get(url_datasetseries, json={
        "resources": [
            {
                "identifier": "series1",
                "@id": "series1/",
                "title": "Test Series 1"
            },
            {
                "identifier": "sample",
                "@id": "sample/",
                "title": "Sample Series"
            }
        ]
    })

    # Mock list_datasetmembers_distributions for format validation
    url_distributions = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}"
    requests_mock.get(
        url_distributions,
        json={
            "datasets": [
                {
                    "key": dataset,
                    "distributions": [
                        {
                            "key": f"{dataset}/2020-01-01/distribution.csv",
                            "values": ["2020-01-01T10:00:00Z", "1000", "hash1", catalog, dataset, "2020-01-01", "csv", 
                                       "bucket", "version1"],
                        },
                        {
                            "key": f"{dataset}/2020-01-02/distribution.csv", 
                            "values": ["2020-01-02T10:00:00Z", "1000", "hash2", catalog, dataset, "2020-01-02", "csv", 
                                       "bucket", "version2"],
                        },
                        {
                            "key": f"{dataset}/2020-01-03/distribution.csv",
                            "values": ["2020-01-03T10:00:00Z", "1000", "hash3", catalog, dataset, "2020-01-03", "csv", 
                                       "bucket", "version3"],
                        },
                    ],
                }
            ]
        },
    )

    # Dates for mocking _resolve_distro_tuples response
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    patch_res = [(catalog, dataset, dt, file_format) for dt in dates]

    # Mock _resolve_distro_tuples method
    mocker.patch.object(
        fusion_obj,
        "_resolve_distro_tuples",
        return_value=patch_res,
    )

    # Mock list_distribution_files to return single file per distribution
    mock_files_df = pd.DataFrame({"@id": ["file1"], "identifier": ["file1"]})
    mocker.patch.object(
        fusion_obj,
        "list_distribution_files",
        return_value=mock_files_df,
    )

    # Mock filesystem download method
    download_result = (True, f"{fusion_obj.download_folder}/test_file.{file_format}", None)
    mock_filesystem = mocker.MagicMock()
    mock_filesystem.download.return_value = download_result
    mocker.patch.object(fusion_obj, "get_fusion_filesystem", return_value=mock_filesystem)

    # Test 1: Basic download without return_paths
    res = fusion_obj.download(dataset=dataset, dt_str=dt_str, dataset_format=file_format, catalog=catalog)
    assert res is None

    # Test 2: Download with return_paths
    res = fusion_obj.download(
        dataset=dataset, dt_str=dt_str, dataset_format=file_format, catalog=catalog, return_paths=True
    )
    assert res is not None
    assert len(res) == len(dates)

    # Test 3: Download with show_progress=False
    res = fusion_obj.download(
        dataset=dataset,
        dt_str=dt_str,
        dataset_format=file_format,
        catalog=catalog,
        return_paths=True,
        show_progress=False,
    )
    assert res is not None
    assert len(res) == len(dates)

    # Test 4: Download with hive partitioning
    res = fusion_obj.download(
        dataset=dataset,
        dt_str=dt_str,
        dataset_format=file_format,
        catalog=catalog,
        return_paths=True,
        partitioning="hive",
    )
    assert res is not None
    assert len(res) == len(dates)

    # Test 5: Download with "latest" dt_str
    res = fusion_obj.download(
        dataset=dataset, dt_str="latest", dataset_format=file_format, catalog=catalog, return_paths=True
    )
    assert res is not None
    assert len(res) == len(dates)

    # Test 6: Error handling - mock failed downloads
    failed_download_result = (False, "my_file.dat", "Some Error")
    mock_filesystem.download.return_value = failed_download_result
    
    res = fusion_obj.download(
        dataset=dataset,
        dt_str=dt_str,
        dataset_format=file_format,
        catalog=catalog,
        return_paths=True,
        show_progress=False,
    )
    assert res is not None
    assert len(res) == len(dates)
    for r in res:
        assert not r[0]  # All downloads should have failed

    # Test 7: Sample download
    # Mock sample-specific behavior
    mocker.patch.object(
        fusion_obj,
        "list_distributions",
        return_value=pd.DataFrame({"identifier": [file_format]}),
    )
    
    # Reset filesystem mock for successful download
    sample_download_result = (True, f"{fusion_obj.download_folder}/{dataset}__sample__{catalog}.{file_format}", None)
    mock_filesystem.download.return_value = sample_download_result
    
    res = fusion_obj.download(
        dataset=dataset, dt_str="sample", dataset_format=file_format, catalog=catalog, return_paths=True
    )
    assert res is not None
    assert len(res) == 1
    assert res[0][0]  # Should be successful
    assert "sample" in res[0][1]

def test_download_no_access(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    dt_str = "20200101"
    file_format = "csv"

    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"

    expected_data = {
        "catalog": {
            "@id": "my_catalog/",
            "description": "my catalog",
            "title": "my catalog",
            "identifier": "my_catalog",
        },
        "title": "Test Dataset",
        "identifier": "TEST_DATASET",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "frequency": "Once",
        "isInternalOnlyDataset": False,
        "isThirdPartyData": True,
        "isRestricted": False,
        "isRawData": True,
        "maintainer": "maintainer",
        "source": "source",
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tags": ["tag1", "tag2"],
        "createdDate": "2020-05-05",
        "modifiedDate": "2020-05-05",
        "deliveryChannel": ["API"],
        "language": "English",
        "status": "Available",
        "type": "Source",
        "containerType": "Snapshot-Full",
        "snowflake": "snowflake",
        "complexity": "complexity",
        "isImmutable": False,
        "isMnpi": False,
        "isPii": False,
        "isPci": False,
        "isClient": False,
        "isPublic": False,
        "isInternal": False,
        "isConfidential": False,
        "isHighlyConfidential": False,
        "isActive": False,
        "@id": "TEST_DATASET/",
    }

    requests_mock.get(url, json=expected_data)

    with pytest.raises(
        CredentialError, match="You are not subscribed to TEST_DATASET in catalog my_catalog. Please request access."
    ):
        fusion_obj.download(dataset=dataset, dt_str=dt_str, dataset_format=file_format, catalog=catalog)


def test_download_format_not_available(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    dt_str = "20200101"
    file_format = "pdf"

    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"

    expected_data = {
        "catalog": {
            "@id": "my_catalog/",
            "description": "my catalog",
            "title": "my catalog",
            "identifier": "my_catalog",
        },
        "title": "Test Dataset",
        "identifier": "TEST_DATASET",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "frequency": "Once",
        "isInternalOnlyDataset": False,
        "isThirdPartyData": True,
        "isRestricted": False,
        "isRawData": True,
        "maintainer": "maintainer",
        "source": "source",
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tags": ["tag1", "tag2"],
        "createdDate": "2020-05-05",
        "modifiedDate": "2020-05-05",
        "deliveryChannel": ["API"],
        "language": "English",
        "status": "Subscribed",
        "type": "Source",
        "containerType": "Snapshot-Full",
        "snowflake": "snowflake",
        "complexity": "complexity",
        "isImmutable": False,
        "isMnpi": False,
        "isPii": False,
        "isPci": False,
        "isClient": False,
        "isPublic": False,
        "isInternal": False,
        "isConfidential": False,
        "isHighlyConfidential": False,
        "isActive": False,
        "@id": "TEST_DATASET/",
    }

    requests_mock.get(url, json=expected_data)

    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}"

    expected_resp = {
        "lastModified": "2025-03-18T09:04:22Z",
        "checksum": "SHA-256=vFdIF:HSLDBV:VBLHD/xe8Mom9yqooZA=-1",
        "metadata": {
            "fields": [
                "lastModified",
                "size",
                "checksum",
                "catalog",
                "dataset",
                "seriesMember",
                "distribution",
                "storageProvider",
                "version",
            ]
        },
        "datasets": [
            {
                "key": "TEST_DATASET",
                "lastModified": "2025-03-18T09:04:22Z",
                "checksum": "SHA-256=vSLKFGNSDFGJBADFGsjfgl/xe8Mom9yqooZA=-1",
                "distributions": [
                    {
                        "key": "TEST_DATASET/20250317/distribution.csv",
                        "values": [
                            "2025-03-18T09:04:22Z",
                            "3054",
                            "SHA-256=vlfaDJFb:VbSdfOHLvnL/xe8Mom9yqooZA=-1",
                            "my_catalog",
                            "TEST_DATASET",
                            "20250317",
                            "csv",
                            "api-bucket",
                            "SJLDHGF;eflSBVLS",
                        ],
                    },
                    {
                        "key": "TEST_DATASET/20250317/distribution.parquet",
                        "values": [
                            "2025-03-18T09:04:19Z",
                            "3076",
                            "SHA-256=7yfQDQq/M1VE4S0SKJDHfblDHFVBldvLXlv5Q=-1",
                            "my_catalog",
                            "TEST_DATASET",
                            "20250317",
                            "parquet",
                            "api-bucket",
                            "SJDFB;IUEBRF;dvbuLSDVc",
                        ],
                    },
                ],
            }
        ],
    }

    requests_mock.get(url, json=expected_resp)

    with pytest.raises(
        FileFormatError,
        match=re.escape(
            "Dataset format pdf is not available for TEST_DATASET in catalog my_catalog. "
            "Available formats are ['csv', 'parquet']."
        ),
    ):
        fusion_obj.download(dataset=dataset, dt_str=dt_str, dataset_format=file_format, catalog=catalog)


def test_download_multiple_format_error(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    dt_str = "20200101"

    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"

    expected_data = {
        "catalog": {
            "@id": "my_catalog/",
            "description": "my catalog",
            "title": "my catalog",
            "identifier": "my_catalog",
        },
        "title": "Test Dataset",
        "identifier": "TEST_DATASET",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "frequency": "Once",
        "isInternalOnlyDataset": False,
        "isThirdPartyData": True,
        "isRestricted": False,
        "isRawData": True,
        "maintainer": "maintainer",
        "source": "source",
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tags": ["tag1", "tag2"],
        "createdDate": "2020-05-05",
        "modifiedDate": "2020-05-05",
        "deliveryChannel": ["API"],
        "language": "English",
        "status": "Subscribed",
        "type": "Source",
        "containerType": "Snapshot-Full",
        "snowflake": "snowflake",
        "complexity": "complexity",
        "isImmutable": False,
        "isMnpi": False,
        "isPii": False,
        "isPci": False,
        "isClient": False,
        "isPublic": False,
        "isInternal": False,
        "isConfidential": False,
        "isHighlyConfidential": False,
        "isActive": False,
        "@id": "TEST_DATASET/",
    }

    requests_mock.get(url, json=expected_data)

    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}"

    expected_resp = {
        "lastModified": "2025-03-18T09:04:22Z",
        "checksum": "SHA-256=vFdIF:HSLDBV:VBLHD/xe8Mom9yqooZA=-1",
        "metadata": {
            "fields": [
                "lastModified",
                "size",
                "checksum",
                "catalog",
                "dataset",
                "seriesMember",
                "distribution",
                "storageProvider",
                "version",
            ]
        },
        "datasets": [
            {
                "key": "TEST_DATASET",
                "lastModified": "2025-03-18T09:04:22Z",
                "checksum": "SHA-256=vSLKFGNSDFGJBADFGsjfgl/xe8Mom9yqooZA=-1",
                "distributions": [
                    {
                        "key": "TEST_DATASET/20250317/distribution.csv",
                        "values": [
                            "2025-03-18T09:04:22Z",
                            "3054",
                            "SHA-256=vlfaDJFb:VbSdfOHLvnL/xe8Mom9yqooZA=-1",
                            "my_catalog",
                            "TEST_DATASET",
                            "20250317",
                            "csv",
                            "api-bucket",
                            "SJLDHGF;eflSBVLS",
                        ],
                    },
                    {
                        "key": "TEST_DATASET/20250317/distribution.parquet",
                        "values": [
                            "2025-03-18T09:04:19Z",
                            "3076",
                            "SHA-256=7yfQDQq/M1VE4S0SKJDHfblDHFVBldvLXlv5Q=-1",
                            "my_catalog",
                            "TEST_DATASET",
                            "20250317",
                            "parquet",
                            "api-bucket",
                            "SJDFB;IUEBRF;dvbuLSDVc",
                        ],
                    },
                ],
            }
        ],
    }

    requests_mock.get(url, json=expected_resp)

    with pytest.raises(
        FileFormatError,
        match=re.escape(
            "Multiple formats found for TEST_DATASET in catalog my_catalog. Dataset format is required to"
            "download. Available formats are ['csv', 'parquet']."
        ),
    ):
        fusion_obj.download(dataset=dataset, dt_str=dt_str, dataset_format=None, catalog=catalog)

def test_download_invalid_dt_str_format(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test download method when dt_str is not a valid date time format and not available in datasetseries."""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    dt_str = "invalid_date_format"  # Invalid date format
    file_format = "csv"

    # Mock dataset access check
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"
    expected_data = {
        "catalog": {
            "@id": "my_catalog/",
            "description": "my catalog",
            "title": "my catalog",
            "identifier": "my_catalog",
        },
        "title": "Test Dataset",
        "identifier": "TEST_DATASET",
        "status": "Subscribed",  # User has access
        "@id": "TEST_DATASET/",
    }
    requests_mock.get(url, json=expected_data)

    # Mock list_datasetmembers to return some dataset members (but not our invalid date)
    url_members = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries"
    requests_mock.get(
        url_members,
        json={
            "resources": [
                {
                    "@id": "2020-01-01",
                    "identifier": "2020-01-01",
                    "dataset": dataset,
                    "createdDate": "2020-01-01",
                }
            ]
        },
    )

    # Mock list_datasetmembers_distributions (changes endpoint) - required for format validation
    url_distributions = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}"
    requests_mock.get(
        url_distributions,
        json={
            "datasets": [
                {
                    "key": dataset,
                    "distributions": [
                        {
                            "key": f"{dataset}/20200101/distribution.csv",
                            "values": [
                                "2020-01-01T10:00:00Z",
                                "1000",
                                "hash1",
                                catalog,
                                dataset,
                                "20200101",
                                "csv",
                                "bucket",
                                "version1",
                            ],
                        }
                    ],
                }
            ]
        },
    )

    # The download should not create any directories or download any files
    # It should raise an error because the invalid dt_str doesn't match the regex pattern
    # and also doesn't exist in the datasetseries
    with pytest.raises(APIResponseError, match=f"datasetseries '{dt_str}' not found for dataset"):
        fusion_obj.download(dataset=dataset, dt_str=dt_str, dataset_format=file_format, catalog=catalog)


def test_download_valid_dt_str_format_not_in_datasetseries(
    requests_mock: requests_mock.Mocker, fusion_obj: Fusion
) -> None:
    """Test download method when dt_str is a valid date time format but not available in datasetseries."""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    dt_str = "2020-01-01"  # Valid date format but not available
    file_format = "csv"

    # Mock dataset access check
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"
    expected_data = {
        "catalog": {
            "@id": "my_catalog/",
            "description": "my catalog",
            "title": "my catalog",
            "identifier": "my_catalog",
        },
        "title": "Test Dataset",
        "identifier": "TEST_DATASET",
        "status": "Subscribed",  # User has access
        "@id": "TEST_DATASET/",
    }
    requests_mock.get(url, json=expected_data)

    # Mock list_datasetmembers to return different dates (not including our target date)
    url_members = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries"
    requests_mock.get(
        url_members,
        json={
            "resources": [
                {
                    "@id": "2020-02-01",
                    "identifier": "2020-02-01",
                    "dataset": dataset,
                    "createdDate": "2020-02-01",
                },
                {
                    "@id": "2020-03-01",
                    "identifier": "2020-03-01",
                    "dataset": dataset,
                    "createdDate": "2020-03-01",
                },
            ]
        },
    )

    # Mock list_datasetmembers_distributions to return available formats
    url_distributions = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}"
    requests_mock.get(
        url_distributions,
        json={
            "datasets": [
                {
                    "key": dataset,
                    "distributions": [
                        {
                            "key": f"{dataset}/2020-02-01/distribution.csv",
                            "values": [
                                "2020-02-01T10:00:00Z",
                                "1000",
                                "hash1",
                                catalog,
                                dataset,
                                "2020-02-01",
                                "csv",
                                "bucket",
                                "version1",
                            ],
                        },
                        {
                            "key": f"{dataset}/2020-03-01/distribution.csv",
                            "values": [
                                "2020-03-01T10:00:00Z",
                                "1000",
                                "hash2",
                                catalog,
                                dataset,
                                "2020-03-01",
                                "csv",
                                "bucket",
                                "version2",
                            ],
                        },
                    ],
                }
            ]
        },
    )

    # The download should not create any directories or download any files
    # It should raise an error because while the date format is valid,
    # no datasetseries exists for the requested date
    with pytest.raises(
        APIResponseError,
        match=f"datasetseries '{dt_str}' not found for dataset '{dataset}' in catalog '{catalog}'",
    ):
        fusion_obj.download(dataset=dataset, dt_str=dt_str, dataset_format=file_format, catalog=catalog)

def test_to_df(mocker: MockerFixture, fusion_obj: Fusion, tmp_path: Path) -> None:
    """Test to_df method returns pandas DataFrame."""
    catalog = "my_catalog"
    dataset = "my_dataset"
    dt_str = "2020-04-04"
    file_format = "csv"
    
    mock_download_res = [(True, str(tmp_path / f"{dataset}__test.{file_format}"), None)]
    mocker.patch.object(fusion_obj, "download", return_value=mock_download_res)
    
    csv_file = tmp_path / f"{dataset}__test.{file_format}"
    csv_file.write_text("col1,col2\n1,2\n3,4\n")
    
    result = fusion_obj.to_df(
        dataset=dataset,
        dt_str=dt_str,
        dataset_format=file_format,
        catalog=catalog
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ["col1", "col2"]


def test_to_df_parquet(mocker: MockerFixture, fusion_obj: Fusion, tmp_path: Path) -> None:
    """Test to_df method with parquet format."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    
    catalog = "my_catalog"
    dataset = "my_dataset"
    dt_str = "2020-04-04"
    file_format = "parquet"
    
    parquet_file = tmp_path / f"{dataset}__test.{file_format}"
    mock_download_res = [(True, str(parquet_file), None)]
    mocker.patch.object(fusion_obj, "download", return_value=mock_download_res)
    
    table = pa.table({"col1": [1, 3], "col2": [2, 4]})
    pq.write_table(table, str(parquet_file))
    
    result = fusion_obj.to_df(
        dataset=dataset,
        dt_str=dt_str,
        dataset_format=file_format,
        catalog=catalog
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ["col1", "col2"]


def test_to_df_with_columns(mocker: MockerFixture, fusion_obj: Fusion, tmp_path: Path) -> None:
    """Test to_df method with column selection."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    
    catalog = "my_catalog"
    dataset = "my_dataset"
    dt_str = "2020-04-04"
    file_format = "parquet"
    
    parquet_file = tmp_path / f"{dataset}__test.{file_format}"
    mock_download_res = [(True, str(parquet_file), None)]
    mocker.patch.object(fusion_obj, "download", return_value=mock_download_res)
    
    table = pa.table({"col1": [1, 3], "col2": [2, 4], "col3": [5, 6]})
    pq.write_table(table, str(parquet_file))
    
    result = fusion_obj.to_df(
        dataset=dataset,
        dt_str=dt_str,
        dataset_format=file_format,
        catalog=catalog,
        columns=["col1", "col2"]
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ["col1", "col2"]


def test_to_df_json(mocker: MockerFixture, fusion_obj: Fusion, tmp_path: Path) -> None:
    """Test to_df method with JSON format."""
    catalog = "my_catalog"
    dataset = "my_dataset"
    dt_str = "2020-04-04"
    file_format = "json"
    
    json_file = tmp_path / f"{dataset}__test.{file_format}"
    mock_download_res = [(True, str(json_file), None)]
    mocker.patch.object(fusion_obj, "download", return_value=mock_download_res)
    
    import json
    json_data = [{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}]
    json_file.write_text(json.dumps(json_data))
    
    result = fusion_obj.to_df(
        dataset=dataset,
        dt_str=dt_str,
        dataset_format=file_format,
        catalog=catalog
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_to_df_sample(mocker: MockerFixture, fusion_obj: Fusion, tmp_path: Path) -> None:
    """Test to_df method with sample data."""
    catalog = "my_catalog"
    dataset = "my_dataset"
    dt_str = "sample"
    
    csv_file = tmp_path / f"{dataset}__sample.csv"
    mock_download_res = [(True, str(csv_file), None)]
    mocker.patch.object(fusion_obj, "download", return_value=mock_download_res)
    
    csv_file.write_text("col1,col2\n10,20\n30,40\n")
    
    result = fusion_obj.to_df(
        dataset=dataset,
        dt_str=dt_str,
        catalog=catalog
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_to_df_no_download_results(mocker: MockerFixture, fusion_obj: Fusion) -> None:
    """Test to_df raises ValueError when download returns no results."""
    catalog = "my_catalog"
    dataset = "my_dataset"
    dt_str = "2020-04-04"
    file_format = "csv"
    
    mocker.patch.object(fusion_obj, "download", return_value=None)
    
    with pytest.raises(ValueError, match="Must specify 'return_paths=True'"):
        fusion_obj.to_df(
            dataset=dataset,
            dt_str=dt_str,
            dataset_format=file_format,
            catalog=catalog
        )


def test_to_df_failed_downloads(mocker: MockerFixture, fusion_obj: Fusion) -> None:
    """Test to_df raises Exception when downloads fail."""
    catalog = "my_catalog"
    dataset = "my_dataset"
    dt_str = "2020-04-04"
    file_format = "csv"
    
    mock_download_res = [(False, "failed_file.csv", "Download error")]
    mocker.patch.object(fusion_obj, "download", return_value=mock_download_res)
    
    with pytest.raises(Exception, match="Not all downloads were successfully completed"):
        fusion_obj.to_df(
            dataset=dataset,
            dt_str=dt_str,
            dataset_format=file_format,
            catalog=catalog
        )


def test_to_table_minimal_params(fusion_obj: Fusion) -> None:
    """Test to_table raises NotImplementedError with minimal parameters."""
    pytest.importorskip("pyarrow")
    
    with pytest.raises(NotImplementedError, match="Method not implemented"):
        fusion_obj.to_table(dataset="my_dataset")


def test_to_table_typical_usage(fusion_obj: Fusion) -> None:
    """Test to_table raises NotImplementedError with typical parameters."""
    pytest.importorskip("pyarrow")
    
    with pytest.raises(NotImplementedError, match="Method not implemented"):
        fusion_obj.to_table(
            dataset="my_dataset",
            dt_str="2020-04-04",
            dataset_format="parquet",
            catalog="my_catalog",
            columns=["col1", "col2"]
        )


def test_to_table_all_params(fusion_obj: Fusion, tmp_path: Path) -> None:
    """Test to_table raises NotImplementedError with all parameters."""
    pytest.importorskip("pyarrow")
    
    with pytest.raises(NotImplementedError, match="Method not implemented"):
        fusion_obj.to_table(
            dataset="my_dataset",
            dt_str="2020-04-04:2020-04-10",
            dataset_format="csv",
            catalog="my_catalog",
            n_par=4,
            show_progress=False,
            columns=["col1", "col2"],
            filters=[("col1", ">", 100)],
            force_download=True,
            download_folder=str(tmp_path)
        )

def test_listen_to_events(fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    # expect raise NotImplementedError
    with pytest.raises(NotImplementedError):
        fusion_obj.listen_to_events(catalog=catalog)

def test_get_events(fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    # expect raise NotImplementedError
    with pytest.raises(NotImplementedError):
        fusion_obj.get_events(catalog=catalog)

def test_list_dataset_lineage(requests_mock: requests_mock.Mocker, fusion_obj: Any) -> None:
    dataset = "dataset_id"
    catalog = "catalog_id"
    url_dataset = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"
    requests_mock.get(url_dataset, status_code=200)
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/lineage"
    expected_data: Dict[str, Any] = {
        "relations": [
            {
                "source": {"dataset": "source_dataset", "catalog": "source_catalog"},
                "destination": {"dataset": dataset, "catalog": catalog},
            },
            {
                "source": {"dataset": dataset, "catalog": catalog},
                "destination": {"dataset": "destination_dataset", "catalog": "destination_catalog"},
            },
        ],
        "datasets": [
            {"identifier": "source_dataset", "title": "Source Dataset"},
            {"identifier": "destination_dataset", "status": "Active", "title": "Destination Dataset"},
        ],
    }
    requests_mock.get(url, json=expected_data)

    # Call the list_dataset_lineage method
    test_df = fusion_obj.list_dataset_lineage(dataset, catalog=catalog)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(
        {
            "type": ["source", "produced"],
            "dataset_identifier": ["source_dataset", "destination_dataset"],
            "title": ["Source Dataset", "Destination Dataset"],
            "catalog": ["source_catalog", "destination_catalog"],
        }
    )
    pd.testing.assert_frame_equal(test_df, expected_df)

def test_list_dataset_lineage_max_results(requests_mock: requests_mock.Mocker, fusion_obj: Any) -> None:
    dataset = "dataset_id"
    catalog = "catalog_id"
    url_dataset = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"
    requests_mock.get(url_dataset, status_code=200)
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/lineage"
    expected_data: Dict[str, Any] = {
        "relations": [
            {
                "source": {"dataset": "source_dataset", "catalog": "source_catalog"},
                "destination": {"dataset": dataset, "catalog": catalog},
            },
            {
                "source": {"dataset": dataset, "catalog": catalog},
                "destination": {"dataset": "destination_dataset", "catalog": "destination_catalog"},
            },
        ],
        "datasets": [
            {"identifier": "source_dataset", "status": "Active", "title": "Source Dataset"},
            {"identifier": "destination_dataset", "status": "Active", "title": "Destination Dataset"},
        ],
    }
    requests_mock.get(url, json=expected_data)

    # Call the list_dataset_lineage method
    test_df = fusion_obj.list_dataset_lineage(dataset, catalog=catalog, max_results=1)

    # Check if the dataframe is created correctly
    assert len(test_df) == 1


def test_list_dataset_lineage_restricted(requests_mock: requests_mock.Mocker, fusion_obj: Any) -> None:
    dataset_id = "dataset_id"
    catalog = "catalog_id"
    url_dataset = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset_id}"
    requests_mock.get(url_dataset, status_code=200)
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset_id}/lineage"

    expected_data: Dict[str, Any] = {
        "relations": [
            {
                "source": {"dataset": "source_dataset", "catalog": "source_catalog"},
                "destination": {"dataset": dataset_id, "catalog": catalog},
            },
            {
                "source": {"dataset": dataset_id, "catalog": catalog},
                "destination": {"dataset": "destination_dataset", "catalog": "destination_catalog"},
            },
        ],
        "datasets": [
            {"identifier": "source_dataset", "status": "Restricted"},
            {"identifier": "destination_dataset", "status": "Active", "title": "Destination Dataset"},
        ],
    }
    requests_mock.get(url, json=expected_data)

    # Call the list_dataset_lineage method
    test_df = fusion_obj.list_dataset_lineage(dataset_id, catalog=catalog)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(
        {
            "type": ["source", "produced"],
            "dataset_identifier": ["Access Restricted", "destination_dataset"],
            "title": ["Access Restricted", "Destination Dataset"],
            "catalog": ["Access Restricted", "destination_catalog"],
        }
    )
    pd.testing.assert_frame_equal(test_df, expected_df)

    def test_list_dataset_lineage_dataset_not_found(requests_mock: Any, fusion_obj: Any) -> None:
        dataset_id = "dataset_id"
        catalog = "catalog_id"
        url_dataset = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset_id}"
        requests_mock.get(url_dataset, status_code=404)

        with pytest.raises(requests.exceptions.HTTPError):
            fusion_obj.list_dataset_lineage(dataset_id, catalog=catalog)


def test_create_dataset_lineage_from_df(requests_mock: Any, fusion_obj: Any) -> None:
    base_dataset = "base_dataset"
    source_dataset = "source_dataset"
    source_dataset_catalog = "source_catalog"
    catalog = "common"
    status_code = 200
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{base_dataset}/lineage"
    expected_data: Dict[str, List[Dict[str, str]]] = {
        "source": [{"dataset": source_dataset, "catalog": source_dataset_catalog}]
    }
    requests_mock.post(url, json=expected_data)

    data = [{"dataset": "source_dataset", "catalog": "source_catalog"}]
    df_input = pd.DataFrame(data)

    # Call the create_dataset_lineage method
    resp = fusion_obj.create_dataset_lineage(
        base_dataset=base_dataset, source_dataset_catalog_mapping=df_input, catalog=catalog, return_resp_obj=True
    )

    # Check if the response is correct
    assert resp is not None
    if resp is not None:
        assert resp.status_code == status_code


def test_create_dataset_lineage_from_list(requests_mock: Any, fusion_obj: Any) -> None:
    base_dataset = "base_dataset"
    source_dataset = "source_dataset"
    source_dataset_catalog = "source_catalog"
    catalog = "common"
    status_code = 200
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{base_dataset}/lineage"
    expected_data: Dict[str, List[Dict[str, str]]] = {
        "source": [{"dataset": source_dataset, "catalog": source_dataset_catalog}]
    }
    requests_mock.post(url, json=expected_data)

    data = [{"dataset": "source_dataset", "catalog": "source_catalog"}]

    # Call the create_dataset_lineage method
    resp = fusion_obj.create_dataset_lineage(
        base_dataset=base_dataset, source_dataset_catalog_mapping=data, catalog=catalog, return_resp_obj=True
    )

    # Check if the response is correct
    assert resp is not None
    if resp is not None:
        assert resp.status_code == status_code


def test_create_dataset_lineage_valueerror(requests_mock: Any, fusion_obj: Any) -> None:
    base_dataset = "base_dataset"
    source_dataset = "source_dataset"
    source_dataset_catalog = "source_catalog"
    catalog = "common"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{base_dataset}/lineage"
    expected_data: Dict[str, List[Dict[str, str]]] = {
        "source": [{"dataset": source_dataset, "catalog": source_dataset_catalog}]
    }
    requests_mock.post(url, json=expected_data)

    data = {"dataset": "source_dataset", "catalog": "source_catalog"}

    with pytest.raises(
        ValueError, match="source_dataset_catalog_mapping must be a pandas DataFrame or a list of dictionaries."
    ):
        fusion_obj.create_dataset_lineage(
            base_dataset=base_dataset,
            source_dataset_catalog_mapping=data,  # type: ignore
            catalog=catalog,
        )

def test_create_dataset_lineage_httperror(requests_mock: Any, fusion_obj: Any) -> None:
    base_dataset = "base_dataset"
    source_dataset = "source_dataset"
    source_dataset_catalog = "source_catalog"
    catalog = "common"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{base_dataset}/lineage"
    expected_data = {"source": [{"dataset": source_dataset, "catalog": source_dataset_catalog}]}
    data = [{"dataset": "source_dataset", "catalog": "source_catalog"}]
    requests_mock.post(url, status_code=500, json=expected_data)

    with pytest.raises(requests.exceptions.HTTPError):
        fusion_obj.create_dataset_lineage(
            base_dataset=base_dataset, source_dataset_catalog_mapping=data, catalog=catalog
        )


def test_list_product_dataset_mapping_dataset_list(requests_mock: Any, fusion_obj: Any) -> None:
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data = {
        "resources": [
            {"product": "P00001", "dataset": "D00001"},
            {"product": "P00002", "dataset": "D00002"},
        ]
    }
    requests_mock.get(url, json=expected_data)

    resp = fusion_obj.list_product_dataset_mapping(dataset=["D00001"], catalog=catalog)
    expected_df = pd.DataFrame({"product": ["P00001"], "dataset": ["D00001"]})

    # Ensure column order is the same before comparison
    assert_frame_equal(resp[expected_df.columns].reset_index(drop=True), expected_df.reset_index(drop=True))

def test_list_product_dataset_mapping_dataset_str(requests_mock: Any, fusion_obj: Any) -> None:
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data = {
        "resources": [
            {"product": "P00001", "dataset": "D00001"},
            {"product": "P00002", "dataset": "D00002"},
        ]
    }
    requests_mock.get(url, json=expected_data)

    resp = fusion_obj.list_product_dataset_mapping(dataset="D00001", catalog=catalog)

   # Convert expected_data to a DataFrame for comparison
    expected_df = pd.DataFrame(expected_data["resources"])

    # Filter the expected DataFrame to match the dataset "D00001"
    expected_df = expected_df[expected_df["dataset"] == "D00001"]

    # Use assert_frame_equal for proper DataFrame comparison
    assert_frame_equal(resp.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_list_product_dataset_mapping_product_list(requests_mock: Any, fusion_obj: Any) -> None:
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data = {
        "resources": [
            {"product": "P00001", "dataset": "D00001"},
            {"product": "P00002", "dataset": "D00002"},
        ]
    }
    requests_mock.get(url, json=expected_data)

    resp = fusion_obj.list_product_dataset_mapping(product=["P00001"], catalog=catalog)
    expected_df = pd.DataFrame({"product": ["P00001"], "dataset": ["D00001"]})

    # Use assert_frame_equal for comparing DataFrames
    assert_frame_equal(
    resp[expected_df.columns].reset_index(drop=True),
    expected_df.reset_index(drop=True)
    )


def test_list_product_dataset_mapping_product_no_filter(requests_mock: Any, fusion_obj: Any) -> None:
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data = {"resources": [{"product": "P00001", "dataset": "D00001"}]}
    requests_mock.get(url, json=expected_data)

    resp = fusion_obj.list_product_dataset_mapping(catalog=catalog)
     # Convert expected_data["resources"] to DataFrame
    expected_df = pd.DataFrame(expected_data["resources"])

    # Ensure column order matches before comparison
    resp = resp[expected_df.columns]

    # Use assert_frame_equal for DataFrame comparison
    assert_frame_equal(resp.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_fusion_product(fusion_obj: Any) -> None:
    test_product = fusion_obj.product(title="Test Product", identifier="Test Product", releaseDate="May 5, 2020")
    assert test_product.title == "Test Product"
    assert test_product.identifier == "Test_Product"
    assert test_product.category is None
    assert test_product.shortAbstract == "Test Product"
    assert test_product.description == "Test Product"
    assert test_product.isActive is True
    assert test_product.isRestricted is None
    assert test_product.maintainer is None
    assert test_product.region == ["Global"]
    assert test_product.publisher == "J.P. Morgan"
    assert test_product.subCategory is None
    assert test_product.tag is None
    assert test_product.deliveryChannel == ["API"]
    assert test_product.theme is None
    assert test_product.releaseDate == "2020-05-05"
    assert test_product.language == "English"
    assert test_product.status == "Available"
    assert test_product.image == ""
    assert test_product.logo == ""
    assert test_product.dataset is None
    assert test_product._client == fusion_obj

def test_fusion_dataset(fusion_obj: Fusion) -> None:
    """Test Fusion Dataset class from client"""
    test_dataset = fusion_obj.dataset(
        title="Test Dataset",
        identifier="Test Dataset",
        category="Test",
        product="TEST_PRODUCT",
    )

    assert str(test_dataset)
    assert repr(test_dataset)
    assert test_dataset.title == "Test Dataset"
    assert test_dataset.identifier == "Test_Dataset"
    assert test_dataset.category == ["Test"]
    assert test_dataset.description == "Test Dataset"
    assert test_dataset.frequency == "Once"
    assert test_dataset.is_internal_only_dataset is False
    assert test_dataset.is_third_party_data is True
    assert test_dataset.is_restricted is None
    assert test_dataset.is_raw_data is True
    assert test_dataset.maintainer == "J.P. Morgan Fusion"
    assert test_dataset.source is None
    assert test_dataset.region is None
    assert test_dataset.publisher == "J.P. Morgan"
    assert test_dataset.product == ["TEST_PRODUCT"]
    assert test_dataset.sub_category is None
    assert test_dataset.tags is None
    assert test_dataset.created_date is None
    assert test_dataset.modified_date is None
    assert test_dataset.delivery_channel == ["API"]
    assert test_dataset.language == "English"
    assert test_dataset.status == "Available"
    assert test_dataset.type_ == "Source"
    assert test_dataset.container_type == "Snapshot-Full"
    assert test_dataset.snowflake is None
    assert test_dataset.complexity is None
    assert test_dataset.is_immutable is None
    assert test_dataset.is_mnpi is None
    assert test_dataset.is_pii is None
    assert test_dataset.is_pci is None
    assert test_dataset.is_client is None
    assert test_dataset.is_public is None
    assert test_dataset.is_internal is None
    assert test_dataset.is_confidential is None
    assert test_dataset.is_highly_confidential is None
    assert test_dataset.is_active is None
    assert test_dataset.client == fusion_obj


def test_fusion_attribute(fusion_obj: Fusion) -> None:
    """Test Fusion Attribute class from client."""
    test_attribute = fusion_obj.attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        isDatasetKey=True,
        dataType="String",
        availableFrom="May 5, 2020",
    )
    assert str(test_attribute)
    assert repr(test_attribute)
    assert test_attribute.title == "Test Attribute"
    assert test_attribute.identifier == "Test_Attribute"
    assert test_attribute.index == 0
    assert test_attribute.isDatasetKey
    assert test_attribute.dataType == Types.String
    assert test_attribute.description == "Test Attribute"
    assert test_attribute.source is None
    assert test_attribute.sourceFieldId == "Test_Attribute"
    assert test_attribute.isInternalDatasetKey is None
    assert test_attribute.isExternallyVisible is True
    assert test_attribute.unit is None
    assert test_attribute.multiplier == 1.0
    assert test_attribute.isMetric is None
    assert test_attribute.isPropagationEligible is None
    assert test_attribute.availableFrom == "2020-05-05"
    assert test_attribute.deprecatedFrom is None
    assert test_attribute.term == "bizterm1"
    assert test_attribute.dataset is None
    assert test_attribute.attributeType is None
    assert test_attribute._client == fusion_obj


def test_fusion_attributes(fusion_obj: Fusion) -> None:
    """Test Fusion Attributes class from client."""
    test_attributes = fusion_obj.attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                is_dataset_key=True,
                data_type="String",  # Adjusted for Python 3.7.9 compatibility
                available_from="May 5, 2020",
            )
        ]
    )
    assert str(test_attributes)
    assert repr(test_attributes)
    assert test_attributes.attributes[0].title == "Test Attribute"
    assert test_attributes.attributes[0].identifier == "Test_Attribute"
    assert test_attributes.attributes[0].index == 0
    assert test_attributes.attributes[0].dataType == Types.String
    assert test_attributes.attributes[0].description == "Test Attribute"
    assert test_attributes.attributes[0].source is None
    assert test_attributes.attributes[0].sourceFieldId == "Test_Attribute"
    assert test_attributes.attributes[0].isInternalDatasetKey is None
    assert test_attributes.attributes[0].isExternallyVisible is True
    assert test_attributes.attributes[0].unit is None
    assert test_attributes.attributes[0].multiplier == 1.0
    assert test_attributes.attributes[0].isMetric is None
    assert test_attributes.attributes[0].isPropagationEligible is None
    assert test_attributes.attributes[0].availableFrom == "2020-05-05"
    assert test_attributes.attributes[0].deprecatedFrom is None
    assert test_attributes.attributes[0].term == "bizterm1"
    assert test_attributes.attributes[0].dataset is None
    assert test_attributes.attributes[0].attributeType is None
    assert test_attributes._client == fusion_obj

def test_fusion_create_product(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test create product from client."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/products/TEST_PRODUCT"
    expected_data = {
        "title": "Test Product",
        "identifier": "TEST_PRODUCT",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "isActive": True,
        "isRestricted": False,
        "maintainer": ["maintainer"],
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tag": ["tag1", "tag2"],
        "deliveryChannel": ["API"],
        "theme": "theme",
        "releaseDate": "2020-05-05",
        "language": "English",
        "status": "Available",
        "image": "",
        "logo": "",
    }
    requests_mock.post(url, json=expected_data)

    my_product = fusion_obj.product(
        title="Test Product",
        identifier="TEST_PRODUCT",
        category=["category"],
        shortAbstract="short abstract",
        description="description",
        isActive=True,
        isRestricted=False,
        maintainer=["maintainer"],
        region=["region"],
        publisher="publisher",
        subCategory=["subCategory"],
        tag=["tag1", "tag2"],
        deliveryChannel=["API"],
        theme="theme",
        releaseDate="2020-05-05",
        language="English",
        status="Available",
        image="",
        logo="",
    )
    status_code = 200
    resp = my_product.create(catalog=catalog, client=fusion_obj, return_resp_obj=True)
    assert isinstance(resp, requests.models.Response)
    assert resp.status_code == status_code

def test_fusion_create_dataset_dict(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test create dataset from client."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/TEST_DATASET"
    expected_data = {
        "title": "Test Dataset",
        "identifier": "TEST_DATASET",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "frequency": "Once",
        "isInternalOnlyDataset": False,
        "isThirdPartyData": True,
        "isRestricted": False,
        "isRawData": False,
        "maintainer": "maintainer",
        "source": "source",
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tags": ["tag1", "tag2"],
        "createdDate": "2020-05-05",
        "modifiedDate": "2020-05-05",
        "deliveryChannel": ["API"],
        "language": "English",
        "status": "Available",
        "type": "Source",
        "containerType": "Snapshot-Full",
        "snowflake": "snowflake",
        "complexity": "complexity",
        "isImmutable": False,
        "isMnpi": False,
        "isPii": False,
        "isPci": False,
        "isClient": False,
        "isPublic": False,
        "isInternal": False,
        "isConfidential": False,
        "isHighlyConfidential": False,
        "isActive": False,
    }
    requests_mock.post(url, json=expected_data)

    dataset_dict = {
        "title": "Test Dataset",
        "identifier": "TEST_DATASET",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "frequency": "Once",
        "isInternalOnlyDataset": False,
        "isThirdPartyData": True,
        "isRestricted": False,
        "isRawData": False,
        "maintainer": "maintainer",
        "source": "source",
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tags": ["tag1", "tag2"],
        "createdDate": "2020-05-05",
        "modifiedDate": "2020-05-05",
        "deliveryChannel": ["API"],
        "language": "English",
        "status": "Available",
        "type": "Source",
        "containerType": "Snapshot-Full",
        "snowflake": "snowflake",
        "complexity": "complexity",
        "isImmutable": False,
        "isMnpi": False,
        "isPii": False,
        "isPci": False,
        "isClient": False,
        "isPublic": False,
        "isInternal": False,
        "isConfidential": False,
        "isHighlyConfidential": False,
        "isActive": False,
    }
    dataset_obj = fusion_obj.dataset(identifier="TEST_DATASET").from_object(dataset_dict)
    resp = dataset_obj.create(client=fusion_obj, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert isinstance(resp, requests.models.Response)
    assert resp.status_code == status_code


def test_fusion_create_attributes(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test create attributes from client."""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/attributes"

    expected_data = {
        "attributes": [
            {
                "title": "Test Attribute",
                "identifier": "Test Attribute",
                "index": 0,
                "isDatasetKey": True,
                "dataType": "string",
                "description": "Test Attribute",
                "source": None,
                "sourceFieldId": "test_attribute",
                "isInternalDatasetKey": None,
                "isExternallyVisible": True,
                "unit": None,
                "multiplier": 1.0,
                "isMetric": None,
                "isPropagationEligible": None,
                "availableFrom": "2020-05-05",
                "deprecatedFrom": None,
                "term": "bizterm1",
                "dataset": None,
                "attributeType": None,
            }
        ]
    }

    requests_mock.put(url, json=expected_data)

    test_attributes = fusion_obj.attributes(
        [
            fusion_obj.attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                isDatasetKey=True,
                dataType="String",
                availableFrom="May 5, 2020",
            )
        ]
    )
    resp = test_attributes.create(client=fusion_obj, catalog=catalog, dataset=dataset, return_resp_obj=True)
    status_code = 200
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_fusion_delete_datasetmembers(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test delete datasetmembers"""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    datasetseries = "20200101"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{datasetseries}"
    requests_mock.delete(url, status_code=200)

    resp = fusion_obj.delete_datasetmembers(dataset, datasetseries, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert resp is not None
    assert isinstance(resp[0], requests.Response)
    assert resp[0].status_code == status_code
    resp_len = 1
    assert len(resp) == resp_len


def test_fusion_delete_datasetmembers_multiple(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test delete datasetmembers"""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    datasetseries = ["20200101", "20200101"]
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{datasetseries[0]}"
    requests_mock.delete(url, status_code=200)

    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{datasetseries[1]}"
    requests_mock.delete(url, status_code=200)

    resp = fusion_obj.delete_datasetmembers(dataset, datasetseries, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert resp is not None
    assert isinstance(resp[0], requests.Response)
    assert resp[0].status_code == status_code
    assert isinstance(resp[1], requests.Response)
    assert resp[1].status_code == status_code
    resp_len = 2
    assert len(resp) == resp_len


def test_fusion_delete_all_datasetmembers(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test delete datasetmembers"""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries"
    requests_mock.delete(url, status_code=200)

    resp = fusion_obj.delete_all_datasetmembers(dataset, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert resp is not None
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_list_registered_attributes(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list registered attributes."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/attributes"
    core_cols = [
        "identifier",
        "title",
        "dataType",
        "publisher",
        "description",
        "applicationId",
    ]

    server_mock_data = {
        "resources": [
            {
                "identifier": "attr_1",
                "title": "some title",
                "dataType": "string",
                "publisher": "J.P Morgan",
                "applicationId": {"id": "12345", "type": "application"},
                "catalog": {"@id": "12345/", "description": "catalog"},
            },
            {
                "identifier": "attr_2",
                "title": "some title",
                "dataType": "int",
                "publisher": "J.P Morgan",
                "applicationId": {"id": "12345", "type": "application"},
                "catalog": {"@id": "12345/", "description": "catalog"},
            },
        ]
    }
    expected_data = {
        "resources": [
            {
                "identifier": "attr_1",
                "title": "some title",
                "dataType": "string",
                "publisher": "J.P Morgan",
                "applicationId": {"id": "12345", "type": "application"},
            },
            {
                "identifier": "attr_2",
                "title": "some title",
                "dataType": "int",
                "publisher": "J.P Morgan",
                "applicationId": {"id": "12345", "type": "application"},
            },
        ]
    }

    expected_df = pd.DataFrame(expected_data["resources"])

    requests_mock.get(url, json=server_mock_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_registered_attributes(catalog=catalog)
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)
    assert all(col in core_cols for col in test_df.columns)


def test_fusion_report(fusion_obj: Fusion) -> None:
    """Test Fusion Report object creation using a dummy Report (mocked)."""
    dummy_report = Report(
        title="Quarterly Risk Report",
        description="Q1 Risk report for compliance",
        frequency="Quarterly",
        category="Risk Management",
        sub_category="Operational Risk",
        data_node_id={"name": "ComplianceTable", "dataNodeType": "Table"},
        regulatory_related=True,
        domain={"name": "Risk"},
        tier_type="Tier 1",
        lob="Global Markets",
        is_bcbs239_program=True,
        sap_code="SAP123",
        region="EMEA",
    )
    dummy_report.client = fusion_obj

    # Patch fusion_obj.report to return the dummy report
    fusion_obj.report = MagicMock(return_value=dummy_report)

    report = fusion_obj.report(
        title="Quarterly Risk Report",
        description="Q1 Risk report for compliance",
        frequency="Quarterly",
        category="Risk Management",
        sub_category="Operational Risk",
        data_node_id={"name": "ComplianceTable", "dataNodeType": "Table"},
        regulatory_related=True,
        domain={"name": "Risk"},
        tier_type="Tier 1",
        lob="Global Markets",
        is_bcbs239_program=True,
        sap_code="SAP123",
        region="EMEA",
    )

    assert isinstance(report, Report)
    assert report.title == "Quarterly Risk Report"
    assert report.description == "Q1 Risk report for compliance"
    assert report.client == fusion_obj
    assert report.domain == {"name": "Risk"}
    assert report.tier_type == "Tier 1"
    assert report.is_bcbs239_program is True
    assert report.region == "EMEA"
    assert report.data_node_id["name"] == "ComplianceTable"

def test_fusion_dataflow_id_only(fusion_obj: Fusion) -> None:
    """ID-only path: instantiate with just an id; nodes remain None."""
    flow = fusion_obj.dataflow(id="abc-123")
    assert flow.id == "abc-123"
    assert flow.providerNode is None
    assert flow.consumerNode is None
    assert flow.client == fusion_obj


def test_fusion_dataflow_full(fusion_obj: Fusion) -> None:
    """Full constructor path: provider/consumer plus optional fields."""
    provider = {"name": "CRM_DB", "nodeType": "Database"}
    consumer = {"name": "DWH", "nodeType": "Database"}
    flow = fusion_obj.dataflow(
        provider_node=provider,
        consumer_node=consumer,
        description="CRM → DWH nightly load",
        frequency="Daily",
        transport_type="Batch",
    )
    assert flow.providerNode == provider
    assert flow.consumerNode == consumer
    assert flow.description == "CRM → DWH nightly load"
    assert flow.frequency == "Daily"
    assert flow.transportType == "Batch"
    assert flow.client == fusion_obj


def test_list_dataflows_success(requests_mock: "requests_mock.Mocker", fusion_obj: Fusion) -> None:
    """list_dataflows returns a normalized dataframe when the API responds 200."""
    import pandas as pd

    flow_id = "abc-123"
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{flow_id}"
    server_json = {
        "id": flow_id,
        "description": "sample flow",
        "providerNode": {"name": "A", "nodeType": "DB"},
        "consumerNode": {"name": "B", "nodeType": "DB"},
        "frequency": "Daily",
    }
    requests_mock.get(url, json=server_json, status_code=200)

    test_df = fusion_obj.list_dataflows(flow_id)
    assert isinstance(test_df, pd.DataFrame)
    assert len(test_df) == 1
    assert test_df.loc[0, "id"] == flow_id
    assert test_df.loc[0, "description"] == "sample flow"
    assert "providerNode.name" in test_df.columns or "providerNode" in test_df.columns
    assert "consumerNode.name" in test_df.columns or "consumerNode" in test_df.columns
    # confirm nested normalization
    if "providerNode.name" in test_df.columns:
        assert test_df.loc[0, "providerNode.name"] == "A"
    if "consumerNode.name" in test_df.columns:
        assert test_df.loc[0, "consumerNode.name"] == "B"


def test_list_dataflows_http_error(requests_mock: "requests_mock.Mocker", fusion_obj: Fusion) -> None:
    """list_dataflows raises for non-200 responses."""
    import requests

    flow_id = "does-not-exist"
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{flow_id}"
    requests_mock.get(url, status_code=404)

    with pytest.raises(requests.exceptions.HTTPError):
        fusion_obj.list_dataflows(flow_id)


def test_fusion_init_logging_to_specified_file(credentials: FusionCredentials, tmp_path: str) -> None:
    log_path = tmp_path / "custom_log_folder"
    if not log_path.exists():
        log_path.mkdir(parents=True)

    # Clear handlers to avoid test contamination
    logger.handlers.clear()

    Fusion(credentials=credentials, enable_logging=True, log_path=log_path)

    # Check that StreamHandler and FileHandler were added
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    # Confirm log file exists
    log_file = log_path / "fusion_sdk.log"
    assert log_file.exists()

    # Clean up for other tests
    logger.handlers.clear()


def test_fusion_init_logging_enabled_to_stdout_and_file(credentials: FusionCredentials, tmp_path: str) -> None:
    log_path = tmp_path / "logs"
    if not log_path.exists():
        log_path.mkdir(parents=True)

    # Clear logger handlers to avoid contamination
    logger.handlers.clear()

    Fusion(credentials=credentials, enable_logging=True, log_path=log_path)

    # Ensure the logger is configured with both handlers
    assert any(type(handler) is logging.StreamHandler for handler in logger.handlers)
    assert any(type(handler) is logging.FileHandler for handler in logger.handlers)
    assert not any(type(handler) is logging.NullHandler for handler in logger.handlers)

    # Verify the log file exists
    log_file = log_path / "fusion_sdk.log"
    assert log_file.exists()

    logger.handlers.clear()


def test_fusion_init_logging_disabled(credentials: FusionCredentials) -> None:
    # Clear logger handlers to avoid contamination
    logger.handlers.clear()

    # Create the Fusion object with logging disabled
    Fusion(credentials=credentials, enable_logging=False)

    # No additional handlers should be added
    assert any(type(handler) is logging.StreamHandler for handler in logger.handlers)
    assert all(type(handler) is not logging.FileHandler for handler in logger.handlers)
    assert not any(type(handler) is logging.NullHandler for handler in logger.handlers)

    # Clean up
    logger.handlers.clear()


def test_fusion_preserves_existing_handlers(credentials: FusionCredentials) -> None:
    logger.handlers.clear()
    custom_handler = logging.StreamHandler()
    logger.addHandler(custom_handler)

    Fusion(credentials=credentials, enable_logging=False)

    assert logger.handlers == [custom_handler]
    logger.handlers.clear()


def test_fusion_adds_streamhandler_when_no_handlers(credentials: FusionCredentials) -> None:
    logger.handlers.clear()

    Fusion(credentials=credentials, enable_logging=False)

    assert len(logger.handlers) == 1
    assert type(logger.handlers[0]) is logging.StreamHandler

    logger.handlers.clear()

def test_list_distribution_files_with_max_results(fusion_obj: Fusion, requests_mock: requests_mock.Mocker) -> None:
    mock_data = {
        "resources": [
            {
                "description": "Sample file 1",
                "fileExtension": ".parquet",
                "identifier": "file1",
                "title": "File 1",
                "@id": "file1"
            },
            {
                "description": "Sample file 2",
                "fileExtension": ".parquet",
                "identifier": "file2",
                "title": "File 2",
                "@id": "file2"
            }
        ]
    }

    catalog = "common"
    dataset = "test_dataset"
    series = "test_series"
    file_format = "parquet"

    url = (
        f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/"
        f"{series}/distributions/{file_format}/files"
    )

    requests_mock.get(url, json=mock_data)

    # Case 1: Default (max_results = -1) → return all rows
    df_all = fusion_obj.list_distribution_files(
        dataset=dataset,
        series=series,
        file_format=file_format,
        catalog=catalog,
        max_results=-1
    )
    assert isinstance(df_all, pd.DataFrame)
    TOTAL_FILES = 2
    assert len(df_all) == TOTAL_FILES
    assert set(df_all.columns) >= {"description", "fileExtension", "identifier", "title", "@id"}
    assert df_all["@id"].tolist() == ["file1", "file2"]

    # Case 2: Limit to max_results = 1
    MAX_RESULTS = 1
    df_limited = fusion_obj.list_distribution_files(
        dataset=dataset,
        series=series,
        file_format=file_format,
        catalog=catalog,
        max_results=MAX_RESULTS
    )
    assert len(df_limited) == MAX_RESULTS
    assert df_limited["@id"].tolist() == ["file1"]

    # Case 3: Limit higher than available rows → returns all
    df_over_limit = fusion_obj.list_distribution_files(
        dataset=dataset,
        series=series,
        file_format=file_format,
        catalog=catalog,
        max_results=10
    )
    assert len(df_over_limit) == TOTAL_FILES
    assert df_over_limit["@id"].tolist() == ["file1", "file2"]


def test_to_df_invalid_format(mocker: MockerFixture, fusion_obj: Fusion) -> None:
    """Test to_df raises Exception for unsupported format."""
    catalog = "my_catalog"
    dataset = "my_dataset"
    dt_str = "2020-04-04"
    file_format = "unsupported_format"
    
    mock_download_res = [(True, "file.unsupported", None)]
    mocker.patch.object(fusion_obj, "download", return_value=mock_download_res)
    
    with pytest.raises(Exception, match="No pandas function to read file in format"):
        fusion_obj.to_df(
            dataset=dataset,
            dt_str=dt_str,
            dataset_format=file_format,
            catalog=catalog
        )


def test_get_new_root_url_with_api_v1_suffix(example_creds_dict_token: Dict[str, str]) -> None:
    """Test _get_new_root_url removes /api/v1/ suffix."""
    credentials = FusionCredentials(bearer_token=example_creds_dict_token['token'])
    fusion = Fusion(credentials=credentials, root_url="https://example.com/api/v1/")
    normalized = fusion._get_new_root_url()
    assert normalized == "https://example.com"


def test_get_new_root_url_with_v1_suffix(example_creds_dict_token: Dict[str, str]) -> None:
    """Test _get_new_root_url removes /v1/ suffix."""
    credentials = FusionCredentials(bearer_token=example_creds_dict_token['token'])
    fusion = Fusion(credentials=credentials, root_url="https://example.com/v1/")
    normalized = fusion._get_new_root_url()
    assert normalized == "https://example.com"


def test_report_attribute_creation(example_creds_dict_token: Dict[str, str]) -> None:
    """Test report_attribute method creates ReportAttribute with client."""
    credentials = FusionCredentials(bearer_token=example_creds_dict_token['token'])
    fusion = Fusion(credentials=credentials)
    attr = fusion.report_attribute(
        sourceIdentifier="test_id",
        title="Test Title",
        description="Test Description",
        technicalDataType="String",
        path="Test.Path"
    )
    assert attr.sourceIdentifier == "test_id"
    assert attr.title == "Test Title"
    assert attr.client == fusion


def test_report_attributes_creation(example_creds_dict_token: Dict[str, str]) -> None:
    """Test report_attributes method creates ReportAttributes with client."""
    credentials = FusionCredentials(bearer_token=example_creds_dict_token['token'])
    fusion = Fusion(credentials=credentials)
    attrs = fusion.report_attributes()
    assert len(attrs.attributes) == 0
    assert attrs.client == fusion


def test_reports_wrapper_creation(example_creds_dict_token: Dict[str, str]) -> None:
    """Test reports method creates ReportsWrapper with client."""
    credentials = FusionCredentials(bearer_token=example_creds_dict_token['token'])
    fusion = Fusion(credentials=credentials)
    reports = fusion.reports()
    assert reports.client == fusion


def test_link_attributes_to_terms_with_default_iskde(
    example_creds_dict_token: Dict[str, str], mocker: MockerFixture
) -> None:
    """Test link_attributes_to_terms adds default isKDE=True."""
    credentials = FusionCredentials(bearer_token=example_creds_dict_token['token'])
    fusion = Fusion(credentials=credentials)
    
    mock_link = mocker.patch("fusion.report.Report.link_attributes_to_terms", return_value={})
    
    mappings = [
        {
            "attribute": {"id": "attr1"},
            "term": {"id": "term1"}
        }
    ]
    
    fusion.link_attributes_to_terms(report_id="test_report", mappings=mappings)
    
    call_args = mock_link.call_args
    assert call_args[1]["mappings"][0]["isKDE"] is True




