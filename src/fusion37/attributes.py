"""Fusion Attributes class and functions."""

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union, cast

import numpy as np
import pandas as pd
import requests

from fusion37.fusion_types import Types
from fusion37.utils import (
    CamelCaseMeta,
    camel_to_snake,
    convert_date_format,
    make_bool,
    requests_raise_for_status,
    snake_to_camel,
    tidy_string,
)

if TYPE_CHECKING:
    import requests

    from fusion37.fusion import Fusion


@dataclass
class Attribute(metaclass=CamelCaseMeta):
    identifier: str
    index: int
    data_type: Types = cast(Types, Types.String)
    title: str = ""
    description: str = ""
    is_dataset_key: bool = False
    source: Optional[str] = None
    source_field_id: Optional[str] = None
    is_internal_dataset_key: Optional[bool] = None
    is_externally_visible: Optional[bool] = True
    unit: Optional[Any] = None
    multiplier: float = 1.0
    is_propagation_eligible: Optional[bool] = None
    is_metric: Optional[bool] = None
    available_from: Optional[str] = None
    deprecated_from: Optional[str] = None
    term: str = "bizterm1"
    dataset: Optional[int] = None
    attribute_type: Optional[str] = None
    application_id: Optional[Dict[str, str]] = None
    publisher: Optional[str] = None
    is_key_data_element: Optional[bool] = None

    _client: Optional["Fusion"] = field(init=False, repr=False, compare=False, default=None)

    def __str__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Attribute(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"

    def __repr__(self) -> str:
        s = ", ".join(f"{getattr(self, f.name)!r}" for f in fields(self) if not f.name.startswith("_"))
        return "(" + s + ")"

    def __post_init__(self) -> None:
        self.is_dataset_key = make_bool(self.is_dataset_key)
        self.identifier = tidy_string(self.identifier).lower().replace(" ", "_")
        self.title = tidy_string(self.title) if self.title != "" else self.identifier.replace("_", " ").title()
        self.description = tidy_string(self.description) if self.description and self.description != "" else self.title
        self.source_field_id = (
            tidy_string(self.source_field_id).lower().replace(" ", "_") if self.source_field_id else self.identifier
        )
        print(f"Before Conversion: {self.available_from} (type: {type(self.available_from)})")
        if isinstance(self.available_from, str) and not pd.isna(self.available_from):
            self.available_from = convert_date_format(self.available_from)
        print(f"After Conversion: {self.available_from} (type: {type(self.available_from)})")
           
        self.deprecated_from = convert_date_format(self.deprecated_from) if self.deprecated_from else None
        self.data_type = Types[str(self.data_type).strip().rsplit(".", maxsplit=1)[-1].title()]
        self.application_id = (
            {"id": str(self.application_id), "type": "Application (SEAL)"}
            if isinstance(self.application_id, str)
            else self.application_id
        )

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
    def client(self) -> Optional["Fusion"]:
        return self._client

    @client.setter
    def client(self, client: Optional["Fusion"]) -> None:
        self._client = client

    def _use_client(self, client: Optional["Fusion"]) -> "Fusion":
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    @classmethod
    def _from_series(
        cls: Type["Attribute"],
        series: pd.Series,
    ) -> "Attribute":
        series = series.rename(lambda x: x.replace(" ", "").replace("_", "").lower()).replace(
            to_replace=np.nan, value=None
        )
        data_type = series.get("datatype", cast(Types, Types.String))
        data_type = series.get("type", cast(Types, Types.String)) if data_type is None else data_type
        source = series.get("source", None)
        source = source.strip() if isinstance(source, str) else source

        is_propagation_eligible = series.get("ispropagationeligible", None)
        is_propagation_eligible = (
            make_bool(is_propagation_eligible) if is_propagation_eligible is not None else is_propagation_eligible
        )
        is_metric = series.get("ismetric", None)
        is_metric = make_bool(is_metric) if is_metric is not None else is_metric
        is_internal_dataset_key = series.get("isinternaldatasetkey", None)
        is_internal_dataset_key = (
            make_bool(is_internal_dataset_key) if is_internal_dataset_key is not None else is_internal_dataset_key
        )
        is_externally_visible = series.get("isexternallyvisible", True)
        is_externally_visible = (
            make_bool(is_externally_visible) if is_externally_visible is not None else is_externally_visible
        )

        return cls(
            identifier=series.get("identifier", "").strip(),
            index=series.get("index", -1),
            data_type=Types[data_type.strip().split(".")[-1].title()],
            title=series.get("title", ""),
            description=series.get("description", ""),
            is_dataset_key=series.get("isdatasetkey", False),
            source=source,
            source_field_id=series.get("sourcefieldid", None),
            is_internal_dataset_key=is_internal_dataset_key,
            is_externally_visible=is_externally_visible,
            unit=series.get("unit", None),
            multiplier=series.get("multiplier", 1.0),
            is_propagation_eligible=is_propagation_eligible,
            is_metric=is_metric,
            available_from=series.get("availablefrom", None),
            deprecated_from=series.get("deprecatedfrom", None),
            term=series.get("term", "bizterm1"),
            dataset=series.get("dataset", None),
            attribute_type=series.get("attributetype", None),
        )
    
    @classmethod
    def _from_dict(cls: Type['Attribute'], data: Dict[str, Any]) -> 'Attribute':
        """Instantiate an Attribute object from a dictionary.

        Args:
            data (Dict[str, Any]): Attribute metadata as a dictionary.

        Returns:
            Attribute: Attribute object.

        Examples:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> data = {
            ...     "identifier": "my_attribute",
            ...     "index": 0,
            ...     "data_type": "String",
            ...     "title": "My Attribute",
            ...     "description": "My attribute description"
            ... }
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)._from_dict(data)

        """
        keys = [f.name for f in fields(cls)]
        data = {camel_to_snake(k): v for k, v in data.items()}
        data = {k: (None if pd.isna(v) else v) for k, v in data.items() if k in keys}
        if "data_type" in data:
            data["data_type"] = Types[data["data_type"].strip().rsplit(".", maxsplit=1)[-1].title()]
        return cls(**data)

    def from_object(
        self,
        attribute_source: Union['Attribute', Dict[str, Any], pd.Series],
    ) -> 'Attribute':
        """Instatiate an Attribute from an Attribute object, dictionary or pandas Series.

        Args:
            attribute_source (Union[Attribute, Dict[str, Any], pd.Series]): Attribute metadata source.

        Raises:
            TypeError: If the object provided is not an Attribute object, dictionary or pandas Series.

        Returns:
            Attribute: Attribute object.

        Examples:

            Instatiating a Attribute from a dictionary:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> data = {
            ...     "identifier": "my_attribute",
            ...     "index": 0,
            ...     "data_type": "String",
            ...     "title": "My Attribute",
            ...     "description": "My attribute description"
            ... }
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0).from_object(data)

            Instatiating a Attribute from a pandas Series:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> import pandas as pd
            >>> series = pd.Series({
            ...     "identifier": "my_attribute",
            ...     "index": 0,
            ...     "data_type": "String",
            ...     "title": "My Attribute",
            ...     "description": "My attribute description"
            ... })
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0).from_object(series)

        """
        if isinstance(attribute_source, Attribute):
            attribute = attribute_source
        elif isinstance(attribute_source, dict):
            attribute = self._from_dict(attribute_source)
        elif isinstance(attribute_source, pd.Series):
            attribute = self._from_series(attribute_source)
        else:
            raise ValueError(f"Could not resolve the object provided: {attribute_source}")
        attribute.client = self._client
        return attribute

    def to_dict(self: 'Attribute') -> Dict[str, Any]:
            """Convert object to dictionary.

            Returns:
                Dict[str, Any]: Attribute metadata as a dictionary.

            Examples:

                >>> from fusion37 import Fusion
                >>> fusion = Fusion()
                >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
                >>> attribute_dict = attribute.to_dict()

            """
            result = {snake_to_camel(k): v for k, v in self.__dict__.items() if not k.startswith("_")}
            result["unit"] = str(self.unit) if self.unit is not None else None
            result["dataType"] = self.data_type.name
            if "isKeyDataElement" in result:
                result["isCriticalDataElement"] = result.pop("isKeyDataElement")
            return result   
        
    def create(
            self,
            dataset: str,
            catalog: Optional[str] = None,
            client: Optional["Fusion"] = None,
            return_resp_obj: bool = False,
        ) -> Optional[requests.Response]:
            """Upload a new attribute to a Fusion catalog.

            Args:
                dataset (str): Dataset identifier.
                client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
                    If instantiated from a Fusion object, then the client is set automatically.
                catalog (str, optional): A catalog identifier. Defaults to None.
                return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

            Returns:
                Optional[requests.Response]: The response object from the API call 
                if return_resp_obj is True, otherwise None.

            Examples:

                Individually, from scratch:

                >>> from fusion37 import Fusion
                >>> fusion = Fusion()
                >>> attribute0 = fusion.attribute(identifier="my_attribute_0", index=0)
                >>> attribute.create(dataset="my_dataset", catalog="my_catalog")
                >>> attribute1 = fusion.attribute(identifier="my_attribute_1", index=1)
                >>> attribute.create(dataset="my_dataset", catalog="my_catalog")

                Individually, from a dictionary:

                >>> from fusion37 import Fusion
                >>> fusion = Fusion()
                >>> data = {
                ...     "identifier": "my_attribute",
                ...     "index": 0,
                ...     "data_type": "String",
                ...     "title": "My Attribute",
                ...     "description": "My attribute description"
                ...    }
                >>> attribute = fusion.attribute(identifier="my_attribute", index=0).from_object(data)
                >>> attribute.create(dataset="my_dataset", catalog="my_catalog")

                Individually, from a pandas Series:

                >>> from fusion37 import Fusion
                >>> fusion = Fusion()
                >>> import pandas as pd
                >>> series = pd.Series({
                ...     "identifier": "my_attribute",
                ...     "index": 0,
                ...     "data_type": "String",
                ...     "title": "My Attribute",
                ...     "description": "My attribute description"
                ... })
                >>> attribute = fusion.attribute(identifier="my_attribute", index=0).from_object(series)
                >>> attribute.create(dataset="my_dataset", catalog="my_catalog")

            """
            client = self._use_client(client)
            catalog = client._use_catalog(catalog)
            data = self.to_dict()
            url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{self.identifier}"
            resp = client.session.put(url, json=data)
            requests_raise_for_status(resp)
            return resp if return_resp_obj else None
    
    def delete(
            self,
            dataset: str,
            catalog: Optional[str] = None,
            client: Optional["Fusion"] = None,
            return_resp_obj: bool = False,
        ) -> Optional[requests.Response]:
        """Delete an Attribute from a Fusion catalog.

        Args:
            dataset (str): Dataset identifier.
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
            catalog (str, optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            Optional[requests.Response]: The response object from the API call if return_resp_obj is True, 
            otherwise None.

        Examples:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> fusion.attribute(identifier="my_attribute", index=0).delete(dataset="my_dataset", catalog="my_catalog")

        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{self.identifier}"
        resp = client.session.delete(url)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def set_lineage(
            self,
            attributes: List['Attribute'],
            catalog: Optional[str] = None,
            client: Optional["Fusion"] = None,
            return_resp_obj: bool = False,
        ) -> Optional[requests.Response]:
        """Map an attribute to existing registered attributes in a Fusion catalog. Attributes from an output data flow
            can be mapped to existing registered input data flow attributes. This supports the case in which the
            generating application and receiving application store their attributes with different names.

        Args:
            attributes (List[Attribute]): List of Attribute objects to establish upstream lineage from.
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            catalog (str, optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            Optional[requests.Response]: The response object from the API call if return_resp_obj is True, 
            otherwise None.

        Examples:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> my_attr1 = fusion.attribute(identifier="my_attribute1", index=0, application_id="12345")
            >>> my_attr2 = fusion.attribute(identifier="my_attribute2", index=0, application_id="12345")
            >>> my_attr3 = fusion.attribute(identifier="my_attribute3", index=0, application_id="12345")
            >>> attrs = [my_attr1, my_attr2]
            >>> my_attr3.set_lineage(attributes=attrs, catalog="my_catalog")

        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)

        if self.application_id is None:
            raise ValueError("The 'application_id' attribute is required for setting lineage.")
        target_attributes = []
        for attribute in attributes:
            if attribute.application_id is None:
                raise ValueError(f"The 'application_id' attribute is required for setting lineage.")
            attr_dict = {
                    "catalog": catalog,
                    "attribute": attribute.identifier,
                    "applicationId": attribute.application_id
                }
            target_attributes.append(attr_dict)

        url = f"{client.root_url}catalogs/{catalog}/attributes/lineage"
        data = [
            {
                "source": {
                    "catalog": catalog,
                    "attribute": self.identifier,
                    "applicationId": self.application_id
            },
            "targets": target_attributes
        }
        ]
        resp = client.session.post(url, json=data)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None


@dataclass
class Attributes:
    """Class representing a collection of Attribute instances for managing attribute metadata in a Fusion catalog.

    Attributes:
        attributes (List[Attribute]): List of Attribute instances.
        _client (Optional[Fusion]): Fusion client object.

    """

    attributes: List[Attribute] = field(default_factory=list)

    _client: Optional["Fusion"] = None

    def __str__(self) -> str:
        """String representation of the Attributes collection."""
        return (
            f"[\n" + ",\n ".join(f"{attr.__repr__()}" for attr in self.attributes) + "\n]" if self.attributes else "[]"
        )

    def __repr__(self) -> str:
        """Object representation of the Attributes collection."""
        return self.__str__()

    @property
    def client(self) -> Optional["Fusion"]:
        """Return the client."""
        return self._client

    @client.setter
    def client(self, client: Optional["Fusion"]) -> None:
        """Set the client for the Dataset. Set automatically if the Dataset is instantiated from a Fusion object.

        Args:
            client (Any): Fusion client object.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> attributes = fusion.attributes()
            >>> attributes.client = fusion

        """
        self._client = client

    def _use_client(self, client: Optional["Fusion"]) -> "Fusion":
        """Determine client."""
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    def add_attribute(self, attribute: Attribute) -> None:
        """Add an Attribute instance to the collection.

        Args:
            attribute (Attribute): Attribute instance to add.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attributes = fusion.attributes()
            >>> attributes.add_attribute(attribute)

        """
        self.attributes.append(attribute)

    def remove_attribute(self, identifier: str) -> bool:
        """Remove an Attribute instance from the collection by identifier.

        Args:
            identifier (str): Identifier of the Attribute to remove.

        Returns:
            bool: True if the Attribute was removed, False otherwise.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attributes = fusion.attributes(attributes=[attribute])
            >>> attributes.remove_attribute("my_attribute")

        """
        for attr in self.attributes:
            if attr.identifier == identifier:
                self.attributes.remove(attr)
                return True
        return False

    def get_attribute(self, identifier: str) -> Optional[Attribute]:
        """Get an Attribute instance from the collection by identifier.

        Args:
            identifier (str): Identifier of the Attribute to retrieve.

        Returns:
            Optional[Attribute]: The Attribute instance if found, None otherwise.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attributes = fusion.attributes(attributes=[attribute])
            >>> retrieved_attribute = attributes.get_attribute("my_attribute")

        """
        for attr in self.attributes:
            if attr.identifier == identifier:
                return attr
        return None

    def to_dict(self: 'Attributes') -> Dict[str, List[Dict[str, Any]]]:
        """Convert the collection of Attribute instances to a list of dictionaries.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Collection of Attribute instances as a dictionary.

        Examples:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attributes = fusion.attributes(attributes=[attribute])
            >>> attributes_dict = attributes.to_dict()

        """
        dict_out = {"attributes": [attr.to_dict() for attr in self.attributes]}
        return dict_out

    @classmethod
    def _from_dict_list(cls: Type['Attributes'], data: List[Dict[str, Any]]) -> 'Attributes':
        """Create an Attributes instance from a list of dictionaries.

        Args:
            cls (Type[Attributes]): The class type.
            data (List[Dict[str, Any]]): List of dictionaries representing Attribute instances.

        Returns:
            Attributes: Attributes instance.

        Examples:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> data = [
            ...     {
            ...         "identifier": "my_attribute",
            ...         "index": 0,
            ...         "data_type": "String",
            ...         "title": "My Attribute",
            ...         "description": "My attribute description"
            ...     }
            ... ]
            >>> attributes = fusion.attributes()._from_dict_list(data)

        """
        attributes = [Attribute._from_dict(attr_data) for attr_data in data]
        return Attributes(attributes=attributes)

    @classmethod
    def _from_dataframe(cls: Type['Attributes'], data: pd.DataFrame) -> 'Attributes':
        """Create an Attributes instance from a pandas DataFrame.

        Args:
            data (pd.DataFrame): DataFrame representing Attribute instances.

        Returns:
            Attributes: Attributes instance.

        Examples:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> import pandas as pd
            >>> data = pd.DataFrame([
            ...     {
            ...         "identifier": "my_attribute",
            ...         "index": 0,
            ...         "data_type": "String",
            ...         "title": "My Attribute",
            ...         "description": "My attribute description"
            ...     }
            ... ])
            >>> attributes = fusion.attributes()._from_dataframe(data)

        """
        # Make a copy to avoid modifying the original DataFrame
        data = data.copy()

        # Convert boolean and integer columns to object type for compatibility
        for col in data.columns:
            if data[col].dtype == "bool":
                data[col] = data[col].astype("object")  # Convert boolean to object
            elif np.issubdtype(data[col].dtype, np.integer):
                data[col] = data[col].astype("float")  # Convert integer to float

        data = data.replace(to_replace=np.nan, value=None)
        data = data.reset_index() if "index" not in data.columns else data
        attributes = [Attribute._from_series(series) for _, series in data.iterrows()]
        return Attributes(attributes=attributes)

    def from_object(
        self: 'Attributes',
        attributes_source: Union[List[Attribute], List[Dict[str, Any]], pd.DataFrame],
    ) -> 'Attributes':
        """Instantiate an Attributes object from a list of Attribute objects, dictionaries or pandas DataFrame.

        Args:
            attributes_source (Union[List[Attribute], List[Dict[str, Any]], pd.DataFrame]): Attributes metadata source.

        Raises:
            TypeError: If the object provided is not a list of Attribute objects, dictionaries or pandas DataFrame.

        Returns:
            Attributes: Attributes object.

        Examples:

            Instantiating Attributes from a list of dictionaries:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> data = [
            ...     {
            ...         "identifier": "my_attribute",
            ...         "index": 0,
            ...         "data_type": "String",
            ...         "title": "My Attribute",
            ...         "description": "My attribute description"
            ...     }
            ... ]
            >>> attributes = fusion.attributes().from_object(data)

            Instantiating Attributes from a pandas DataFrame:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> import pandas as pd
            >>> data = pd.DataFrame([
            ...     {
            ...         "identifier": "my_attribute",
            ...         "index": 0,
            ...         "data_type": "String",
            ...         "title": "My Attribute",
            ...         "description": "My attribute description"
            ...     }
            ... ])
            >>> attributes = fusion.attributes().from_object(data)

        """
        if isinstance(attributes_source, list):
            if all(isinstance(attr, Attribute) for attr in attributes_source):
                attributes = Attributes(cast(List[Attribute], attributes_source))
            elif all(isinstance(attr, dict) for attr in attributes_source):
                attributes = Attributes._from_dict_list(cast(List[Dict[str, Any]], attributes_source))
        elif isinstance(attributes_source, pd.DataFrame):
            attributes = Attributes._from_dataframe(attributes_source)
        else:
            raise ValueError(f"Could not resolve the object provided: {attributes_source}")
        attributes.client = self._client
        return attributes


    def to_dataframe(self: 'Attributes') -> pd.DataFrame:
        """Convert the collection of Attribute instances to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame representing the collection of Attribute instances.

        Examples:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> import pandas as pd
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attributes = fusion.attributes(attributes=[attribute])
            >>> attributes_df = attributes.to_dataframe()

        """
        if len(self.attributes) == 0:
            self.attributes = [Attribute(identifier="example_attribute", index=0)]
        data = [attr.to_dict() for attr in self.attributes]
        return pd.DataFrame(data)

    def from_catalog(
        self: 'Attributes', 
        dataset: str, 
        catalog: Union[str, None] = None, 
        client: Union["Fusion", None] = None
    ) -> 'Attributes':
        """Instantiate an Attributes object from a dataset's attributes in a Fusion catalog.

        Args:
            dataset (str): The dataset identifier.
            catalog (Union[str, None], optional): The catalog identifier. Defaults to None.
            client (Union[Fusion, None], optional): Fusion session. Defaults to None.
                If instantiated from a Fusion object, then the client is set automatically.

        Returns:
            Attributes: An instance of the Attributes class with the attributes from the catalog.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> attributes = fusion.attributes().from_catalog(dataset="my_dataset", catalog="my_catalog")

        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes"
        response = client.session.get(url)
        requests_raise_for_status(response)
        list_attributes = response.json()["resources"]
        list_attributes = sorted(list_attributes, key=lambda x: x["index"])

        self.attributes = [Attribute._from_dict(attr_data) for attr_data in list_attributes]
        return self

    def create(
        self: 'Attributes',
        dataset: Union[str, None] = None,
        catalog: Union[str, None] = None,
        client: Union["Fusion", None] = None,
        return_resp_obj: bool = False,
    ) -> Union[requests.Response, None]:
        """Upload the Attributes to a dataset in a Fusion catalog. If no dataset is provided,
            attributes are registered to the catalog.

        Args:
            dataset (str): Dataset identifier.
            client (Union[Fusion, None], optional): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            catalog (Union[str, None], optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            Union[requests.Response, None]: The response object from the API call if return_resp_obj is True, 
            otherwise None.

        Examples:

            From scratch:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attributes = fusion.attributes(attributes=[attribute])
            >>> attributes.create(dataset="my_dataset", catalog="my_catalog")

            From a list of dictionaries:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> data = [
            ...     {
            ...         "identifier": "my_attribute",
            ...         "index": 0,
            ...         "data_type": "String",
            ...         "title": "My Attribute",
            ...         "description": "My attribute description"
            ...     }
            ... ]
            >>> attributes = fusion.attributes().from_dict_list(data)
            >>> attributes.create(dataset="my_dataset", catalog="my_catalog")

            From a pandas DataFrame:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> import pandas as pd
            >>> data = pd.DataFrame([
            ...     {
            ...         "identifier": "my_attribute",
            ...         "index": 0,
            ...         "data_type": "String",
            ...         "title": "My Attribute",
            ...         "description": "My attribute description"
            ...     }
            ... ])
            >>> attributes = fusion.attributes().from_dataframe(data)
            >>> attributes.create(dataset="my_dataset", catalog="my_catalog")

            From existing dataset's attributes in a Fusion catalog:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> attributes = fusion.attributes().from_catalog(dataset="my_dataset", catalog="my_catalog")
            >>> attributes.create(dataset="my_new_dataset", catalog="my_catalog")

            Register attributes to a catalog:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0, application_id="123", publisher="JPM")
            >>> attributes = fusion.attributes(attributes=[attribute])
            >>> attributes.create(catalog="my_catalog")

        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        data = self.to_dict()
        if dataset:
            url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes"
            resp = client.session.put(url, json=data)
            requests_raise_for_status(resp)
            return resp if return_resp_obj else None
        else:
            for attr in self.attributes:
                if attr.publisher is None:
                    raise ValueError("The 'publisher' attribute is required for catalog attributes.")
                if attr.application_id is None:
                    raise ValueError("The 'application_id' attribute is required for catalog attributes.")
            url = f"{client.root_url}catalogs/{catalog}/attributes"
            data_ = data.get("attributes", None)
            resp = client.session.post(url, json=data_)
            requests_raise_for_status(resp)
            return resp if return_resp_obj else None

    def delete(
        self: 'Attributes',
        dataset: str,
        catalog: Union[str, None] = None,
        client: Union["Fusion", None] = None,
        return_resp_obj: bool = False,
    ) -> Union[List[requests.Response], None]:
        """Delete the Attributes from a Fusion catalog.

        Args:
            dataset (str): Dataset identifier.
            client (Union[Fusion, None], optional): A Fusion client object. Defaults to the instance's _client.
            catalog (Union[str, None], optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            Union[List[requests.Response], None]: List of response objects from the API calls 
            if return_resp_obj is True, otherwise None.

        Examples:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> attributes = fusion.attributes().from_catalog(dataset="my_dataset", catalog="my_catalog")
            >>> attributes.delete(dataset="my_dataset", catalog="my_catalog")

        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        responses = []
        for attr in self.attributes:
            resp = client.session.delete(
                f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{attr.identifier}"
            )
            requests_raise_for_status(resp)
            responses.append(resp)

        return responses if return_resp_obj else None