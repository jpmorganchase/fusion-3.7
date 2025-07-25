"""Test cases for Report and Reports classes."""

import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture

from fusion.report import Report, Reports


def test_report_from_dict_valid() -> None:
    data = {
        "title": "Liquidity Report",
        "description": "Monthly liquidity assessment",
        "frequency": "Monthly",
        "category": "Risk",
        "subCategory": "Liquidity",
        "dataNodeId": {"name": "ABC123", "dataNodeType": "Application (SEAL)"},
        "domain": {"name": "Finance"},
        "regulatoryRelated": True,
    }
    report = Report.from_dict(data)
    assert report.title == "Liquidity Report"
    assert report.description == "Monthly liquidity assessment"
    assert report.frequency == "Monthly"
    assert report.category == "Risk"
    assert report.sub_category == "Liquidity"
    assert report.data_node_id["name"] == "ABC123"
    assert report.domain["name"] == "Finance"
    assert report.regulatory_related is True
    assert str(report)
    assert repr(report)


def test_report_from_dict_missing_required() -> None:
    data = {
        "title": "Liquidity Report",
        "description": "Monthly liquidity assessment",
        # missing fields
    }
    report = Report.from_dict(data)
    with pytest.raises(ValueError, match="Missing required fields in Report:"):
        report.validate()




def test_report_from_dataframe_valid() -> None:
    df = pd.DataFrame([{
        "Report/Process Name": "Liquidity Report",
        "Report/Process Description": "Monthly liquidity assessment",
        "Activity Type": "Tier 1",
        "Frequency": "Monthly",
        "Category": "Risk",
        "Sub Category": "Liquidity",
        "Regulatory Designated": "Yes",
        "CDO Office": "Finance",
        "Application ID": "ABC123",
        "Application Type": "Application (SEAL)",
    }])
    reports = Report.from_dataframe(df)
    assert len(reports) == 1
    report = reports[0]
    assert report.title == "Liquidity Report"
    assert report.category == "Risk"
    assert report.domain == {"name": "Finance"}
    assert report.data_node_id["name"] == "ABC123"
    assert report.data_node_id["dataNodeType"] == "Application (SEAL)"
    assert report.regulatory_related is True


def test_report_from_dataframe_invalid_row_skipped(caplog: LogCaptureFixture) -> None:
    df = pd.DataFrame([{
        "Report/Process Name": "",
        "Report/Process Description": "desc",
        "Category": "Risk",
        "Frequency": "Monthly",
        "Sub Category": "Liquidity",
        "Regulatory Designated": "Yes",
        "CDO Office": "Finance",
        "Application ID": "ABC123",
        "Application Type": "Application (SEAL)",
    }])
    reports = Report.from_dataframe(df)
    assert len(reports) == 0
    assert "Skipping invalid row" in caplog.text



def test_report_from_object_invalid_type() -> None:
    with pytest.raises(TypeError, match="source must be a DataFrame, list of dicts, or string"):
        Report.from_object(123)  # type: ignore


def test_reports_wrapper_behavior() -> None:
    reports = Reports()
    assert isinstance(reports, Reports)
    assert len(reports) == 0
    sample = Report(
        title="Liquidity Report",
        description="desc",
        frequency="Monthly",
        category="Risk",
        sub_category="Liquidity",
        data_node_id={"name": "ABC123", "dataNodeType": "Application (SEAL)"},
        domain={"name": "Finance"},
        regulatory_related=True,
    )
    reports.reports.append(sample)
    assert len(reports) == 1
    assert reports[0].title == "Liquidity Report"
