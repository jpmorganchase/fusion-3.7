from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import pandas as pd

from fusion.utils import (
    CamelCaseMeta,
    camel_to_snake,
    requests_raise_for_status,
)

if TYPE_CHECKING:
    import requests

    from fusion import Fusion

@dataclass
class ReportAttribute(metaclass=CamelCaseMeta):
    title: str
    sourceIdentifier: Optional[str] = None
    description: Optional[str] = None
    technicalDataType: Optional[str] = None
    path: Optional[str] = None

    _client: Optional[Fusion] = field(init=False, repr=False, compare=False, default=None)

    def __str__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return "ReportAttribute(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"

    def __repr__(self) -> str:
        return self.__str__()

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sourceIdentifier": self.sourceIdentifier,
            "title": self.title,
            "description": self.description,
            "technicalDataType": self.technicalDataType,
            "path": self.path,
        }

    def _use_client(self, client: Optional[Fusion]) -> Fusion:
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res


@dataclass
class ReportAttributes:
    attributes: List[ReportAttribute] = field(default_factory=list)
    _client: Optional[Fusion] = None

    def __str__(self) -> str:
        return "[\n" + ",\n ".join(f"{attr.__repr__()}" for attr in self.attributes) + "\n]" if self.attributes else "[]" # noqa: E501

    def __repr__(self) -> str:
        return self.__str__()

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

    def add_attribute(self, attribute: ReportAttribute) -> None:
        self.attributes.append(attribute)

    def remove_attribute(self, name: str) -> bool:
        for attr in self.attributes:
            if attr.name == name:
                self.attributes.remove(attr)
                return True
        return False

    def get_attribute(self, name: str) -> Optional[ReportAttribute]:
        for attr in self.attributes:
            if attr.name == name:
                return attr
        return None

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        return {"attributes": [attr.to_dict() for attr in self.attributes]}

    def from_dict_list(self, data: List[Dict[str, Any]]) -> ReportAttributes:
        attributes = [ReportAttribute(**attr_data) for attr_data in data]
        result = ReportAttributes(attributes=attributes)
        result.client = self._client
        return result

    def from_dataframe(self, data: pd.DataFrame) -> ReportAttributes:
        data = data.where(data.notna(), None)
        attributes = [ReportAttribute(**series.dropna().to_dict()) for _, series in data.iterrows()]
        result = ReportAttributes(attributes=attributes)
        result.client = self._client
        return result

    def from_csv(self, file_path: str) -> ReportAttributes:
        df = pd.read_csv(file_path)

        column_map = {
            "Local Data Element Reference ID": "sourceIdentifier",
            "Data Element Name": "title",
            "Data Element Description": "description",
        }

        df = df[[col for col in column_map if col in df.columns]]
        df = df.rename(columns=column_map)

        for col in ["technicalDataType", "path"]:
            if col not in df:
                df[col] = None

        df = df.where(pd.notna(df), None)

        return self.from_dataframe(df)

    def from_object(
        self,
        attributes_source: Union[
            List[ReportAttribute],
            List[Dict[str, Any]],
            pd.DataFrame,
            str
        ],
    ) -> ReportAttributes:
        import json

        if isinstance(attributes_source, list):
            if all(isinstance(attr, ReportAttribute) for attr in attributes_source):
                attributes_obj = ReportAttributes(attributes=cast(List[ReportAttribute], attributes_source))
            elif all(isinstance(attr, dict) for attr in attributes_source):
                attributes_obj = self.from_dict_list(cast(List[Dict[str, Any]], attributes_source))
            else:
                raise TypeError("List must contain either ReportAttribute instances or dicts.")
        elif isinstance(attributes_source, pd.DataFrame):
            attributes_obj = self.from_dataframe(attributes_source)
        elif isinstance(attributes_source, str):
            if attributes_source.strip().endswith(".csv"):
                attributes_obj = self.from_csv(attributes_source)
            elif attributes_source.strip().startswith("[{"):
                dict_list = json.loads(attributes_source)
                attributes_obj = self.from_dict_list(dict_list)
            else:
                raise ValueError("String must be a .csv path or JSON array string.")
        else:
            raise TypeError("Unsupported type for attributes_source.")

        attributes_obj.client = self._client
        return attributes_obj

    def to_dataframe(self) -> pd.DataFrame:
        data = [attr.to_dict() for attr in self.attributes]
        return pd.DataFrame(data)

    def create(
        self,
        report_id: str,
        client: Optional[Fusion] = None,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
        client = self._use_client(client)

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/reports/{report_id}/reportElements"

        payload = [attr.to_dict() for attr in self.attributes]

        resp = client.session.post(url, json=payload)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None
