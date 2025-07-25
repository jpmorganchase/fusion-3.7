"""Test cases for report_attributes module."""

import json
from typing import Any

import pandas as pd
import pytest
import requests
from requests_mock import Mocker

from fusion import Fusion
from fusion.report_attributes import ReportAttribute, ReportAttributes


def test_report_attribute_client_value_error() -> None:
    """Test _use_client raises ValueError when no client provided."""
    attr = ReportAttribute(title="Test")
    with pytest.raises(ValueError, match="A Fusion client object is required."):
        attr._use_client(None)



def test_report_attributes_to_dict_dataframe() -> None:
    """Test to_dict and to_dataframe."""
    attr = ReportAttribute(title="A", sourceIdentifier="S")
    ra = ReportAttributes(attributes=[attr])

    dict_data = ra.to_dict()
    assert dict_data == {"attributes": [attr.to_dict()]}

    df = ra.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "title" in df.columns


def test_report_attributes_from_dict_list() -> None:
    """Test from_dict_list."""
    data = [
        {"title": "A", "sourceIdentifier": "S"},
        {"title": "B", "sourceIdentifier": "S2"},
    ]
    ra = ReportAttributes().from_dict_list(data)
    assert len(ra.attributes) == 2
    assert ra.attributes[0].title == "A"


def test_report_attributes_from_dataframe() -> None:
    """Test from_dataframe."""
    df = pd.DataFrame(
        [{"title": "A", "sourceIdentifier": "S"}, {"title": "B", "sourceIdentifier": "S2"}]
    )
    ra = ReportAttributes().from_dataframe(df)
    assert len(ra.attributes) == 2
    assert ra.attributes[1].title == "B"


def test_report_attributes_from_csv(tmp_path: Any) -> None:
    """Test from_csv."""
    csv_file = tmp_path / "attrs.csv"
    df = pd.DataFrame(
        [
            {
                "Local Data Element Reference ID": "SRC",
                "Data Element Name": "Name",
                "Data Element Description": "Desc",
            }
        ]
    )
    df.to_csv(csv_file, index=False)

    ra = ReportAttributes().from_csv(str(csv_file))
    assert len(ra.attributes) == 1
    assert ra.attributes[0].title == "Name"


def test_report_attributes_from_object_list_attrs() -> None:
    """Test from_object with list of ReportAttribute."""
    attr = ReportAttribute(title="A")
    ra = ReportAttributes().from_object([attr])
    assert isinstance(ra, ReportAttributes)
    assert ra.attributes[0].title == "A"


def test_report_attributes_from_object_list_dicts() -> None:
    """Test from_object with list of dicts."""
    data = [{"title": "A"}, {"title": "B"}]
    ra = ReportAttributes().from_object(data)
    assert len(ra.attributes) == 2
    assert ra.attributes[1].title == "B"


def test_report_attributes_from_object_dataframe() -> None:
    """Test from_object with DataFrame."""
    df = pd.DataFrame([{"title": "X"}])
    ra = ReportAttributes().from_object(df)
    assert ra.attributes[0].title == "X"


def test_report_attributes_from_object_csv(tmp_path: Any) -> None:
    """Test from_object with CSV file."""
    csv_file = tmp_path / "attrs.csv"
    pd.DataFrame(
        [
            {
                "Local Data Element Reference ID": "SRC",
                "Data Element Name": "Name",
                "Data Element Description": "Desc",
            }
        ]
    ).to_csv(csv_file, index=False)

    ra = ReportAttributes().from_object(str(csv_file))
    assert ra.attributes[0].title == "Name"


def test_report_attributes_from_object_json() -> None:
    """Test from_object with JSON string."""
    json_str = json.dumps([{"title": "J", "sourceIdentifier": "S"}])
    ra = ReportAttributes().from_object(json_str)
    assert ra.attributes[0].title == "J"


def test_report_attributes_from_object_invalid_type() -> None:
    """Test from_object with invalid type raises TypeError."""
    with pytest.raises(TypeError):
        ReportAttributes().from_object(123)  # type: ignore


def test_report_attributes_from_object_invalid_list() -> None:
    """Test from_object with mixed list raises TypeError."""
    with pytest.raises(TypeError):
        ReportAttributes().from_object([{"title": "A"}, "invalid"])  # type: ignore


def test_report_attributes_from_object_invalid_string() -> None:
    """Test from_object with invalid string raises ValueError."""
    with pytest.raises(ValueError):
        ReportAttributes().from_object("invalid_string")


def test_report_attributes_create(requests_mock: Mocker, fusion_obj: Fusion) -> None:
    """Test create method with requests_mock."""
    ra = ReportAttributes(
        attributes=[ReportAttribute(title="T", sourceIdentifier="S")]
    )
    ra.client = fusion_obj

    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/reports/rep123/reportElements"
    requests_mock.post(url, json={"result": "ok"})

    resp = ra.create("rep123", return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    assert resp.status_code == 200
