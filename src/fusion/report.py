"""Fusion Report class and functions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
from typing_extensions import TypedDict

from .utils import (
    CamelCaseMeta,
    camel_to_snake,
    make_bool,
    requests_raise_for_status,
    snake_to_camel,
    tidy_string,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import requests

    from fusion import Fusion

logger = logging.getLogger(__name__)

@dataclass
class Report(metaclass=CamelCaseMeta):
    title: str
    data_node_id: Dict[str, str]
    description: str
    frequency: str
    category: str
    sub_category: str
    domain: Dict[str, str]
    regulatory_related: bool

    lob: Optional[str] = None
    sub_lob: Optional[str] = None
    tier_type: Optional[str] = None
    is_bcbs239_program: Optional[bool] = None
    risk_stripe: Optional[str] = None
    risk_area: Optional[str] = None
    sap_code: Optional[str] = None
    tier_designation: Optional[str] = None
    alternative_id: Optional[Dict[str, str]] = None
    region: Optional[str] = None
    mnpi_indicator: Optional[bool] = None
    country_of_reporting_obligation: Optional[str] = None
    primary_regulator: Optional[str] = None

    _client: Optional[Fusion] = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        self.title = tidy_string(self.title) if self.title else None
        self.description = tidy_string(self.description) if self.description else None

    def __getattr__(self, name: str) -> Any:
        snake_name = camel_to_snake(name)
        return self.__dict__.get(snake_name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "client":
            object.__setattr__(self, name, value)
        else:
            snake_name = camel_to_snake(name)
            self.__dict__[snake_name] = value

    @property
    def client(self) -> Optional[Fusion]:
        return self._client

    @client.setter
    def client(self, client: Optional[Fusion]) -> None:
        self._client = client

    def _use_client(self, client: Optional[Fusion]) -> Fusion:
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    @classmethod
    def from_dict(cls: Type[Report], data: Dict[str, Any]) -> Report:
        def normalize_value(val: Any) -> Any:
            if isinstance(val, str) and val.strip() == "":
                return None
            return val

        def convert_keys(d: Dict[str, Any]) -> Dict[str, Any]:
            converted = {}
            for k, v in d.items():
                key = k if k == "isBCBS239Program" else camel_to_snake(k)
                if isinstance(v, dict) and not isinstance(v, str):
                    converted[key] = convert_keys(v)
                else:
                    converted[key] = normalize_value(v)
            return converted

        converted_data = convert_keys(data)
        if "isBCBS239Program" in converted_data:
            converted_data["isBCBS239Program"] = make_bool(converted_data["isBCBS239Program"])

        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in converted_data.items() if k in valid_fields}

        report = cls.__new__(cls)
        for field_single in fields(cls):
            setattr(report, field_single.name, filtered_data.get(field_single.name, None))
        report.__post_init__()
        return report

    def validate(self) -> None:
        required_fields = [
            "title", "data_node_id", "category", "frequency", "description", "sub_category", "domain"
        ]
        missing = [f for f in required_fields if getattr(self, f, None) in [None, ""]]
        if missing:
            raise ValueError(f"Missing required fields in Report: {', '.join(missing)}")

    def to_dict(self) -> Dict[str, Any]:
        report_dict = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if k == "is_bcbs239_program":
                report_dict["isBCBS239Program"] = v
            elif k == "regulatory_related":
                report_dict["regulatoryRelated"] = v
            else:
                report_dict[snake_to_camel(k)] = v
        return report_dict

    @classmethod
    def map_application_type(cls, app_type: str) -> str:
        mapping = {
            "Application (SEAL)": "Application (SEAL)",
            "Intelligent Solutions": "Intelligent Solutions",
            "User Tool": "User Tool"
        }
        return mapping.get(app_type)

    @classmethod
    def map_tier_type(cls, tier_type: str) -> str:
        tier_mapping = {
            "Tier 1": "Tier 1",
            "Non Tier 1": "Non Tier 1"
        }
        return tier_mapping.get(tier_type)

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, client: Optional[Fusion] = None) -> List[Report]:
        data = data.rename(columns=Report.COLUMN_MAPPING)
        data = data.replace([np.nan, np.inf, -np.inf], None)
        data = data.where(data.notna(), None)

        reports = []
        for _, row in data.iterrows():
            report_data = row.to_dict()
            report_data["domain"] = {"name": report_data.pop("domain_name", None)}
            report_data["data_node_id"] = {
                "name": report_data.pop("data_node_name", None),
                "dataNodeType": cls.map_application_type(report_data.pop("data_node_type", None)),
            }


            report_data["is_bcbs239_program"] = make_bool(report_data.get("is_bcbs239_program"))
            report_data["mnpi_indicator"] = make_bool(report_data.get("mnpi_indicator"))
            report_data["regulatory_related"] = make_bool(report_data.get("regulatory_related"))



            tier_val = report_data.get("tier_designation")
            report_data["tier_designation"] = cls.map_tier_type(tier_val) if tier_val else None

            valid_fields = {f.name for f in fields(cls)}
            report_data = {k: v for k, v in report_data.items() if k in valid_fields}

            report_obj = cls(**report_data)
            report_obj.client = client
            try:
                report_obj.validate()
                reports.append(report_obj)
            except ValueError as e:
                logger.warning(f"Skipping invalid row: {e}")

        return reports

    @classmethod
    def from_csv(cls, file_path: str, client: Optional[Fusion] = None) -> List[Report]:
        data = pd.read_csv(file_path)
        return cls.from_dataframe(data, client=client)

    @classmethod
    def from_object(
        cls,
        source: Union[pd.DataFrame, List[Dict[str, Any]], str],
        client: Optional[Fusion] = None
    ) -> Reports:
        import json
        if isinstance(source, pd.DataFrame):
            return cls.from_dataframe(source, client=client)
        elif isinstance(source, list) and all(isinstance(item, dict) for item in source):
            return cls.from_dataframe(pd.DataFrame(source), client=client)
        elif isinstance(source, str):
            if source.lower().endswith(".csv") and Path(source).exists():
                return cls.from_csv(source, client=client)
            elif source.strip().startswith("[{"):
                data = json.loads(source)
                return cls.from_dataframe(pd.DataFrame(data), client=client)
            else:
                raise ValueError("Unsupported string input — must be .csv path or JSON array string")
        raise TypeError("source must be a DataFrame, list of dicts, or string")

    def create(
        self,
        client: Optional[Fusion] = None,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
        client = self._use_client(client)
        data = self.to_dict()
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/reports"
        resp = client.session.post(url, json=data)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    class AttributeTermMapping(TypedDict):
        attribute: Dict[str, str]
        term: Dict[str, str]
        isKDE: bool

    @classmethod
    def link_attributes_to_terms(
        cls,
        report_id: str,
        mappings: List[Report.AttributeTermMapping],
        client: Fusion,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
  
        base_url = client._get_new_root_url()
        endpoint = f"/api/corelineage-service/v1/reports/{report_id}/reportElements/businessTerms"
        url = f"{base_url}{endpoint}"

        resp = client.session.post(url, json=mappings)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None


# Add COLUMN_MAPPING
Report.COLUMN_MAPPING = {
    "Report/Process Name": "title",
    "Report/Process Description": "description",
    "Activity Type": "tier_type",
    "Frequency": "frequency",
    "Category": "category",
    "Report/Process Owner SID": "report_owner",
    "Sub Category": "sub_category",
    "LOB": "lob",
    "Sub-LOB": "sub_lob",
    "JPMSE BCBS Related": "is_bcbs239_program",
    "Report Type": "risk_stripe",
    "Tier Type": "tier_designation",
    "Region": "region",
    "MNPI Indicator": "mnpi_indicator",
    "Country of Reporting Obligation": "country_of_reporting_obligation",
    "Regulatory Designated": "regulatory_related",
    "Primary Regulator": "primary_regulator",
    "CDO Office": "domain_name",
    "Application ID": "data_node_name",
    "Application Type": "data_node_type",
}


class Reports:
    def __init__(self, reports: Optional[List[Report]] = None) -> None:
        self.reports = reports or []
        self._client: Optional[Fusion] = None

    def __getitem__(self, index: int) -> Report:
        return self.reports[index]

    def __iter__(self) -> Iterator[Report]:
        return iter(self.reports)

    def __len__(self) -> int:
        return len(self.reports)

    @property
    def client(self) -> Optional[Fusion]:
        return self._client

    @client.setter
    def client(self, client: Optional[Fusion]) -> None:
        self._client = client
        for report in self.reports:
            report.client = client

    @classmethod
    def from_csv(cls, file_path: str, client: Optional[Fusion] = None) -> Reports:
        data = pd.read_csv(file_path)
        return cls.from_dataframe(data, client=client)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, client: Optional[Fusion] = None) -> Reports:
        report_objs = Report.from_dataframe(df, client=client)
        obj = cls(report_objs)
        obj.client = client
        return obj

    def create_all(self) -> None:
        for report in self.reports:
            report.create()

    @classmethod
    def from_object(cls, source: Union[pd.DataFrame, List[Dict[str, Any]], str], client: Optional[Fusion] = None) -> Reports: # noqa: E501
        import json
        if isinstance(source, pd.DataFrame):
            return cls.from_dataframe(source, client=client)
        elif isinstance(source, list) and all(isinstance(item, dict) for item in source):
            return cls.from_dataframe(pd.DataFrame(source), client=client)
        elif isinstance(source, str):
            if source.lower().endswith(".csv") and Path(source).exists():
                return cls.from_csv(source, client=client)
            elif source.strip().startswith("[{"):
                dict_list = json.loads(source)
                return cls.from_dataframe(pd.DataFrame(dict_list), client=client)
            else:
                raise ValueError("Unsupported string input — must be .csv path or JSON array string")
        raise TypeError("source must be a DataFrame, list of dicts, or string (.csv path or JSON)")


class ReportsWrapper(Reports):
    def __init__(self, client: Fusion) -> None:
        super().__init__([])
        self.client = client

    def from_csv(self, file_path: str) -> Reports:
        return Reports.from_csv(file_path, client=self.client)

    def from_dataframe(self, df: pd.DataFrame) -> Reports:
        return Reports.from_dataframe(df, client=self.client)

    def from_object(self, source: Union[pd.DataFrame, List[Dict[str, Any]], str]) -> Reports:
        return Reports.from_object(source, client=self.client)
