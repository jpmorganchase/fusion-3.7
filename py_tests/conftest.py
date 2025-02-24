import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Union
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fusion.authentication import FusionOAuthAdapter
from fusion.credentials import FusionCredentials
from fusion.fusion import Fusion

PathLike = Union[str, Path]

def pytest_addoption(parser: Any) -> None:
    parser.addoption("--experiments", action="store_true", default=False, help="Run tests marked as experiments")


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    if not config.getoption("--experiments"):
        skip_experiments = pytest.mark.skip(reason="need --experiments option to run")
        for item in items:
            if "experiments" in item.keywords:
                item.add_marker(skip_experiments)


@contextmanager
def change_dir(destination: PathLike) -> Generator[None, None, None]:
    try:
        # Save the current working directory
        cwd = Path.cwd()
        # Change the working directory
        os.chdir(destination)
        yield
    finally:
        # Change back to the original directory
        os.chdir(cwd)


@pytest.fixture
def example_creds_dict() -> Dict[str, Any]:
    return {
        "client_id": "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        },
    }


@pytest.fixture
def example_creds_dict_from_env(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    example_client_id = "vf3tdjK0jdp7MdY3"
    example_client_secret = "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y"
    monkeypatch.setenv("FUSION_CLIENT_ID", example_client_id)
    monkeypatch.setenv("FUSION_CLIENT_SECRET", example_client_secret)

    return {
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "client_id": example_client_id,
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        },
    }


@pytest.fixture
def example_creds_dict_https_pxy() -> Dict[str, Any]:
    return {
        "client_id": "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "https": "http://myproxy.com:8081",
        },
    }


@pytest.fixture
def example_creds_dict_https_pxy_e2e() -> Dict[str, Any]:
    return {
        "client_id": "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "https": "http://myproxy.com:8081",
        },
        "fusion_e2e": "fusion-e2e-token",
    }


@pytest.fixture
def example_creds_dict_no_pxy(example_creds_dict: Dict[str, Any]) -> Dict[str, Any]:
    example_creds_dict.pop("proxies")
    return example_creds_dict


@pytest.fixture
def example_creds_dict_empty_pxy(example_creds_dict: Dict[str, Any]) -> Dict[str, Any]:
    example_creds_dict["proxies"].pop("http")
    example_creds_dict["proxies"].pop("https")
    return example_creds_dict


@pytest.fixture(
    params=[
        "example_creds_dict",
        "example_creds_dict_https_pxy",
        "example_creds_dict_no_pxy",
        "example_creds_dict_empty_pxy",
        "example_creds_dict_https_pxy_e2e",
    ]
)
def credentials_examples(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """Parameterized fixture to return credentials from different sources."""
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(request.getfixturevalue(request.param), f)
    return credentials_file


@pytest.fixture
def good_json() -> str:
    return """{
        "client_id": "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081"
            }
        }"""


@pytest.fixture
def credentials(example_creds_dict: Dict[str, Any], tmp_path: Path) -> FusionCredentials:
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)
    creds.put_bearer_token("my_token", 1800)
    return creds


@pytest.fixture
def fusion_oauth_adapter(credentials: FusionCredentials) -> FusionOAuthAdapter:
    return FusionOAuthAdapter(credentials)


@pytest.fixture
def fusion_obj(credentials: FusionCredentials) -> Fusion:
    fusion = Fusion(credentials=credentials)
    return fusion


@pytest.fixture
def mock_product_pd_read_csv() -> Generator[pd.DataFrame, Any, None]:
    """Mock the pd.read_csv function."""
    product_df = pd.DataFrame(
        {
            "title": ["Test Product"],
            "identifier": ["TEST_PRODUCT"],
        },
        index=[0],
    )
    with patch("fusion.fusion.pd.read_csv", return_value=product_df) as mock:
        yield mock


@pytest.fixture
def mock_dataset_pd_read_csv() -> Generator[pd.DataFrame, Any, None]:
    """Mock the pd.read_csv function."""
    dataset_df = pd.DataFrame(
        {"title": ["Test Dataset"], "identifier": ["TEST_DATASET"], "category": ["Test"], "product": ["TEST_PRODUCT"]},
        index=[0],
    )
    with patch("fusion.fusion.pd.read_csv", return_value=dataset_df) as mock:
        yield mock


@pytest.fixture
def mock_attributes_pd_to_csv() -> Generator[pd.DataFrame, Any, None]:
    """Mock the pd.to_csv function."""
    with patch("fusion.fusion.pd.DataFrame.to_csv") as mock:
        yield mock

@pytest.fixture
def mock_attributes_pd_read_csv() -> Generator[pd.DataFrame, Any, None]:
    """Mock the pd.read_csv function."""
    attributes_df = pd.DataFrame(
    {
        "applicationId": [None],
        "attributeType": [None],
        "availableFrom": [None],
        "dataType": ["String"],
        "dataset": [None],
        "deprecatedFrom": [None],
        "description": ["Example Attribute"],
        "identifier": ["example_attribute"],
        "index": [0],
        "isCriticalDataElement": [None],
        "isDatasetKey": [False],
        "isExternallyVisible": [True],
        "isInternalDatasetKey": [None],
        "isMetric": [None],
        "isPropagationEligible": [None],
        "multiplier": [1.0],
        "publisher": [None],
        "source": [None],
        "sourceFieldId": ["example_attribute"],
        "term": ["bizterm1"],
        "title": ["Example Attribute"],
        "unit": [None]
    },
    index=[0]
)
    for col in attributes_df.columns:            
        if attributes_df[col].dtype == "bool":
            attributes_df[col] = attributes_df[col].astype("object")  # Convert boolean to object
        elif np.issubdtype(attributes_df[col].dtype, np.integer):
            attributes_df[col] = attributes_df[col].astype("float")  # Convert integer to float

    attributes_df = attributes_df.replace(to_replace=np.nan, value=None)
    attributes_df = attributes_df.reset_index() if "index" not in attributes_df.columns else attributes_df
    with patch("fusion.fusion.pd.read_csv", return_value=attributes_df) as mock:
        yield mock


@pytest.fixture
def data_table_as_csv(data_table: pd.DataFrame) -> str:
    # Write the DataFrame to a CSV string
    from io import StringIO

    buffer = StringIO()
    data_table.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def data_table() -> pd.DataFrame:
    # Create a simple DataFrame
    return pd.DataFrame(
        {
            "col_1": range(10),
            "col_2": [str(x) for x in range(10)],
            "col_3": [x / 3.14159 for x in range(10)],
        }
    )