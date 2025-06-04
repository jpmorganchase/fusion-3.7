"""Fusion Report class and functions."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from .utils import (
    CamelCaseMeta,
    camel_to_snake,
    make_bool,
    requests_raise_for_status,
    snake_to_camel,
    tidy_string,
)

if TYPE_CHECKING:
    import requests

    from fusion import Fusion


@dataclass
class Report(metaclass=CamelCaseMeta):
    """Fusion Report class for managing report metadata."""

    name: str
    tier_type: str
    lob: str
    data_node_id: Dict[str, str]
    alternative_id: Dict[str, str]

    # Optional fields
    title: Optional[str] = None
    alternate_id: Optional[str] = None
    description: Optional[str] = None
    frequency: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    report_inventory_name: Optional[str] = None
    report_inventory_id: Optional[str] = None
    report_owner: Optional[str] = None
    sub_lob: Optional[str] = None
    is_bcbs239_program: Optional[bool] = None
    risk_area: Optional[str] = None
    risk_stripe: Optional[str] = None
    sap_code: Optional[str] = None
    sourced_object: Optional[str] = None
    domain: Optional[Dict[str, Any]] = None  # Allow mixed str/bool values
    data_model_id: Optional[Dict[str, str]] = None

    _client: Optional[Fusion] = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        self.name = tidy_string(self.name)
        self.title = tidy_string(self.title) if self.title else None
        self.description = tidy_string(self.description) if self.description else None

    def __getattr__(self, name: str) -> Any:
        snake_name = camel_to_snake(name)
        if snake_name in self.__dict__:
            return self.__dict__[snake_name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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
        """Instantiate a Report object from a dictionary."""

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
        for fieldsingle in fields(cls):
            setattr(report, fieldsingle.name, filtered_data.get(fieldsingle.name, None))

        report.__post_init__()
        return report

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Report instance to a dictionary."""
        report_dict = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if k == "is_bcbs239_program":
                report_dict["isBCBS239Program"] = v
            else:
                report_dict[snake_to_camel(k)] = v
        return report_dict

    def create(
        self,
        client: Optional[Fusion] = None,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
        """Upload a new report to a Fusion catalog."""
        client = self._use_client(client)
        data = self.to_dict()
        url = f"{client.get_new_root_url()}/api/corelineage-service/v1/reports"
        resp: requests.Response = client.session.post(url, json=data)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None
