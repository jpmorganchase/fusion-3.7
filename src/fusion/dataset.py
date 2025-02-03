"""Fusion Dataset class and functions."""

from __future__ import annotations

import json as js
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import pandas as pd

from .utils import (
    CamelCaseMeta,
    _is_json,
    camel_to_snake,
    convert_date_format,
    make_bool,
    make_list,
    requests_raise_for_status,
    snake_to_camel,
    tidy_string,
)

if TYPE_CHECKING:
    import requests

    from .fusion import Fusion

@dataclass
class Dataset(metaclass=CamelCaseMeta):
    """Fusion Dataset class for managing dataset metadata in a Fusion catalog."""

    identifier: str
    title: str = ""
    category: Union[str, List[str], None] = None
    description: str = ""
    frequency: str = "Once"
    is_internal_only_dataset: bool = False
    is_third_party_data: bool = True
    is_restricted: Optional[bool] = None
    is_raw_data: bool = True
    maintainer: Optional[str] = "J.P. Morgan Fusion"
    source: Union[str, List[str], None] = None
    region: Union[str, List[str], None] = None
    publisher: str = "J.P. Morgan"
    product: Union[str, List[str], None] = None
    sub_category: Union[str, List[str], None] = None
    tags: Union[str, List[str], None] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    delivery_channel: Union[str, List[str]] = field(default_factory=lambda: ["API"])
    language: str = "English"
    status: str = "Available"
    type_: Optional[str] = "Source"
    container_type: Optional[str] = "Snapshot-Full"
    snowflake: Optional[str] = None
    complexity: Optional[str] = None
    is_immutable: Optional[bool] = None
    is_mnpi: Optional[bool] = None
    is_pci: Optional[bool] = None
    is_pii: Optional[bool] = None
    is_client: Optional[bool] = None
    is_public: Optional[bool] = None
    is_internal: Optional[bool] = None
    is_confidential: Optional[bool] = None
    is_highly_confidential: Optional[bool] = None
    is_active: Optional[bool] = None
    owners: Optional[List[str]] = None
    application_id: Optional[Union[str, Dict[str, str]]] = None

    _client: Optional[Fusion] = field(init=False, repr=False, compare=False, default=None)

    def __repr__(self) -> str:
        """Return an object representation of the Dataset object."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Dataset(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"

    def __post_init__(self) -> None:
        """Format Dataset metadata fields after object initialization."""
        self.identifier = (
            tidy_string(self.identifier).upper().replace(" ", "_")
        )
        self.title = (
            tidy_string(self.title)
            if self.title != ""
            else self.identifier.replace("_", " ").title()
        )
        self.description = (
            tidy_string(self.description)
            if self.description != ""
            else self.title
        )
        self.category = (
            self.category if isinstance(self.category, list) or self.category is None else make_list(self.category)
        )
        self.delivery_channel = (
            self.delivery_channel if isinstance(self.delivery_channel, list) else make_list(self.delivery_channel)
        )
        self.source = (
            self.source if isinstance(self.source, list) or self.source is None else make_list(self.source)
        )
        self.region = (
            self.region if isinstance(self.region, list) or self.region is None else make_list(self.region)
        )
        self.product = (
            self.product
            if isinstance(self.product, list) or self.product is None
            else make_list(self.product)
        )
        self.sub_category = (
            self.sub_category
            if isinstance(self.sub_category, list) or self.sub_category is None
            else make_list(self.sub_category)
        )
        self.tags = (
            self.tags
            if isinstance(self.tags, list) or self.tags is None
            else make_list(self.tags)
        )
        self.is_internal_only_dataset = (
            self.is_internal_only_dataset
            if isinstance(self.is_internal_only_dataset, bool)
            else make_bool(self.is_internal_only_dataset)
        )
        self.created_date = (
            convert_date_format(self.created_date) if self.created_date else None
        )
        self.modified_date = (
            convert_date_format(self.modified_date) if self.modified_date else None
        )
        self.owners = self.owners if isinstance(self.owners, list) or self.owners is None else make_list(self.owners)
        self.application_id = (
            {"id": str(self.application_id), "type": "Application (SEAL)"}
            if isinstance(self.application_id, str)
            else self.application_id
        )

    def __getattr__(self, name: str) -> Any:
        # Redirect attribute access to the snake_case version
        snake_name = camel_to_snake(name)
        if snake_name in self.__dict__:
            return self.__dict__[snake_name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "client":
            # Use the property setter for client
            object.__setattr__(self, name, value)
        else:
            snake_name = camel_to_snake(name)
            self.__dict__[snake_name] = value

    @property
    def client(self) -> Optional[Fusion]:
        """Return the client."""
        return self._client

    @client.setter
    def client(self, client: Optional[Fusion]) -> None:
        """Set the client for the Dataset. Set automatically, if the Dataset is instantiated from a Fusion object.

        Args:
            client (Optional[Fusion]): Fusion client object.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset")
            >>> dataset.client = fusion

        """
        self._client = client

    def _use_client(self, client: Optional[Fusion]) -> Fusion:
        """Determine client."""
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    @classmethod
    def _from_series(cls: Type[Dataset], series: pd.Series) -> Dataset:
        """Instantiate a Dataset object from a pandas Series.

        Args:
            series (pd.Series): Dataset metadata as a pandas Series.

        Returns:
            Dataset: Dataset object.

        """
        # Renaming columns as per the provided logic
        series = series.rename(lambda x: x.replace(" ", "").replace("_", "").lower())
        series = series.rename({"tag": "tags"})
        series = series.rename({"type_": "type"})
        series = series.rename({"productId": "product"})

        # Converting specific boolean fields
        is_internal_only_dataset = series.get("isinternalonlydataset", None)
        is_internal_only_dataset = (
            make_bool(is_internal_only_dataset) if is_internal_only_dataset is not None else is_internal_only_dataset
        )
        is_restricted = series.get("isrestricted", None)
        is_restricted = make_bool(is_restricted) if is_restricted is not None else is_restricted
        is_immutable = series.get("isimmutable", None)
        is_immutable = make_bool(is_immutable) if is_immutable is not None else is_immutable
        is_mnpi = series.get("ismnpi", None)
        is_mnpi = make_bool(is_mnpi) if is_mnpi is not None else is_mnpi
        is_pci = series.get("ispci", None)
        is_pci = make_bool(is_pci) if is_pci is not None else is_pci
        is_pii = series.get("ispii", None)
        is_pii = make_bool(is_pii) if is_pii is not None else is_pii
        is_client = series.get("isclient", None)
        is_client = make_bool(is_client) if is_client is not None else is_client
        is_public = series.get("ispublic", None)
        is_public = make_bool(is_public) if is_public is not None else is_public
        is_internal = series.get("isinternal", None)
        is_internal = make_bool(is_internal) if is_internal is not None else is_internal
        is_confidential = series.get("isconfidential", None)
        is_confidential = make_bool(is_confidential) if is_confidential is not None else is_confidential
        is_highly_confidential = series.get("ishighlyconfidential", None)
        is_highly_confidential = (
            make_bool(is_highly_confidential) if is_highly_confidential is not None else is_highly_confidential
        )
        is_active = series.get("isactive", None)
        is_active = make_bool(is_active) if is_active is not None else is_active

        # Instantiating Dataset object with the processed data
        dataset = cls(
            identifier=series.get("identifier", ""),
            category=series.get("category", None),
            delivery_channel=series.get("deliverychannel", ["API"]),
            title=series.get("title", ""),
            description=series.get("description", ""),
            frequency=series.get("frequency", "Once"),
            is_internal_only_dataset=is_internal_only_dataset,  # type: ignore
            is_third_party_data=series.get("isthirdpartydata", True),
            is_restricted=is_restricted,
            is_raw_data=series.get("israwdata", True),
            maintainer=series.get("maintainer", "J.P. Morgan Fusion"),
            source=series.get("source", None),
            region=series.get("region", None),
            publisher=series.get("publisher", "J.P. Morgan"),
            product=series.get("product", None),
            sub_category=series.get("subcategory", None),
            tags=series.get("tags", None),
            container_type=series.get("containertype", "Snapshot-Full"),
            language=series.get("language", "English"),
            status=series.get("status", "Available"),
            type_=series.get("type", "Source"),
            created_date=series.get("createddate", None),
            modified_date=series.get("modifieddate", None),
            snowflake=series.get("snowflake", None),
            complexity=series.get("complexity", None),
            owners=series.get("owners", None),
            application_id=series.get("applicationid", None),
            is_immutable=is_immutable,
            is_mnpi=is_mnpi,
            is_pci=is_pci,
            is_pii=is_pii,
            is_client=is_client,
            is_public=is_public,
            is_internal=is_internal,
            is_confidential=is_confidential,
            is_highly_confidential=is_highly_confidential,
            is_active=is_active,
        )
        return dataset
    
    @classmethod
    def _from_dict(cls: Type[Dataset], data: dict) -> Dataset:
        """Instantiate a Dataset object from a dictionary.

        Args:
            data (dict): Dataset metadata as a dictionary.

        Returns:
            Dataset: Dataset object.
        """
        keys = [f.name for f in fields(cls)]
        keys = ["type" if key == "type_" else key for key in keys]
        data = {camel_to_snake(k): v for k, v in data.items()}
        data = {k: v for k, v in data.items() if k in keys}
        if "type" in data:
            data["type_"] = data.pop("type")
        return cls(**data)

    @classmethod
    def _from_csv(cls: Type[Dataset], file_path: str, identifier: Optional[str] = None) -> Dataset:
        """Instantiate a Dataset object from a CSV file.

        Args:
            file_path (str): Path to the CSV file.
            identifier (Optional[str], optional): Dataset identifier for filtering 
                if multiple datasets are defined in CSV. Defaults to None.

        Returns:
            Dataset: Dataset object.
        """
        data = pd.read_csv(file_path)

        return (
            cls._from_series(data[data["identifier"] == identifier].reset_index(drop=True).iloc[0])
            if identifier
            else cls._from_series(data.reset_index(drop=True).iloc[0])
        )

    @classmethod
    def from_object(
        cls,
        dataset_source: Union[Dataset, dict, str, pd.Series],
    ) -> Dataset:
        """Instantiate a Dataset object from a Dataset object, dictionary, JSON string, path to CSV, or pandas Series.

        Args:
            dataset_source (Union[Dataset, dict, str, pd.Series]): Dataset metadata source.

        Raises:
            TypeError: If the object provided is not a Dataset, dictionary, JSON string,
                path to CSV file, or pandas Series.

        Returns:
            Dataset: Dataset object.
        """
        if isinstance(dataset_source, Dataset):
            dataset = dataset_source
        elif isinstance(dataset_source, dict):
            dataset = cls._from_dict(dataset_source)
        elif isinstance(dataset_source, str):
            if _is_json(dataset_source):
                dataset = cls._from_dict(js.loads(dataset_source))
            else:
                dataset = cls._from_csv(dataset_source)
        elif isinstance(dataset_source, pd.Series):
            dataset = cls._from_series(dataset_source)
        else:
            raise TypeError(f"Could not resolve the object provided: {dataset_source}")

        dataset.client = cls._client

        return dataset

    def from_catalog(self, catalog: Optional[str] = None, client: Optional[Fusion] = None) -> Dataset:
        """Instantiate a Dataset object from a Fusion catalog.

        Args:
            catalog (Optional[str], optional): Catalog identifier. Defaults to None.
            client (Optional[Fusion], optional): Fusion session. Defaults to None.
                If instantiated from a Fusion object, then the client is set automatically.

        Returns:
            Dataset: Dataset object.
        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        dataset = self.identifier
        resp = client.session.get(f"{client.root_url}catalogs/{catalog}/datasets")
        requests_raise_for_status(resp)
        list_datasets = resp.json()["resources"]
        matching_datasets = [dict_ for dict_ in list_datasets if dict_["identifier"] == dataset]
        if not matching_datasets:
            raise ValueError(f"Dataset with identifier '{dataset}' not found in catalog '{catalog}'.")
        dict_ = matching_datasets[0]
        dataset_obj = self._from_dict(dict_)
        dataset_obj.client = client

        prod_df = client.list_product_dataset_mapping(catalog=catalog)

        if dataset.lower() in list(prod_df.dataset.str.lower()):
            product = [prod_df[prod_df["dataset"].str.lower() == dataset.lower()]["product"].iloc[0]]
            dataset_obj.product = product

        return dataset_obj
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the Dataset instance to a dictionary.

        Returns:
            Dict[str, Any]: Dataset metadata as a dictionary.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset")
            >>> dataset_dict = dataset.to_dict()
        """
        return {
            snake_to_camel(k): v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def create(
        self,
        catalog: Optional[str] = None,
        product: Optional[str] = None,
        client: Optional[Fusion] = None,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
        """Upload a new dataset to a Fusion catalog.

        Args:
            catalog (Optional[str]): A catalog identifier. Defaults to "common".
            product (Optional[str]): A product identifier to upload dataset to. If dataset object already has
                product attribute populated, the attribute will be overwritten by this value. Defaults to None.
            client (Optional[Fusion]): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            return_resp_obj (bool): If True, return the response object. Defaults to False.

        Returns:
            Optional[requests.Response]: The response object from the API call 
            if return_resp_obj is True, otherwise None.

        Raises:
            ValueError: If the report tier is blank in the dataset.

        Examples:
            From scratch:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset(
            ...     identifier="my_dataset",
            ...     title="My Dataset",
            ...     description="My dataset description",
            ...     category="Finance",
            ...     frequency="Daily",
            ...     is_restricted=False
            ... )
            >>> dataset.create(catalog="my_catalog")

            From a dictionary:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> dataset_dict = {
            ...     "identifier": "my_dataset",
            ...     "title": "My Dataset",
            ...     "description": "My dataset description",
            ...     "category": "Finance",
            ...     "frequency": "Daily",
            ...     "is_restricted": False,
            ... }
            >>> dataset = fusion.dataset("my_dataset").from_object(dataset_dict)
            >>> dataset.create(catalog="my_catalog")

            From existing dataset in a catalog:

            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset").from_catalog(catalog="my_catalog")
            >>> dataset.identifier = "my_new_dataset"
            >>> dataset.create(catalog="my_catalog")
        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)

        self.created_date = (
            self.created_date
            if self.created_date
            else pd.Timestamp("today").strftime("%Y-%m-%d")
        )
        self.modified_date = (
            self.modified_date
            if self.modified_date
            else pd.Timestamp("today").strftime("%Y-%m-%d")
        )

        self.product = [product] if product else self.product

        data = self.to_dict()

        if data.get("report", None) and data["report"]["tier"] == "":
            raise ValueError("Tier cannot be blank for reports.")

        url = f"{client.root_url}catalogs/{catalog}/datasets/{self.identifier}"
        resp: requests.Response = client.session.post(url, json=data)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None
    
    def update(
        self,
        catalog: Optional[str] = None,
        client: Optional[Fusion] = None,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
        """Updates a dataset via API from dataset object.

        Args:
            catalog (Optional[str]): A catalog identifier. Defaults to "common".
            client (Optional[Fusion]): A Fusion client object. Defaults to the instance's _client.
            return_resp_obj (bool): If True, return the response object. Defaults to False.

        Returns:
            Optional[requests.Response]: The response object if `return_resp_obj` is True.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset").from_catalog(catalog="my_catalog")
            >>> dataset.title = "My Updated Dataset"
            >>> dataset.update(catalog="my_catalog")
        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)

        self.created_date = (
            self.created_date
            if self.created_date
            else pd.Timestamp("today").strftime("%Y-%m-%d")
        )
        self.modified_date = (
            self.modified_date
            if self.modified_date
            else pd.Timestamp("today").strftime("%Y-%m-%d")
        )

        data = self.to_dict()
        url = f"{client.root_url}catalogs/{catalog}/datasets/{self.identifier}"
        resp: requests.Response = client.session.put(url, json=data)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None

    def delete(
        self,
        catalog: Optional[str] = None,
        client: Optional[Fusion] = None,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
        """Deletes a dataset via API from its dataset identifier.

        Args:
            catalog (Optional[str]): A catalog identifier. Defaults to "common".
            client (Optional[Fusion]): A Fusion client object. Defaults to the instance's _client.
            return_resp_obj (bool): If True, return the response object. Defaults to False.

        Returns:
            Optional[requests.Response]: The response object if `return_resp_obj` is True.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> fusion.dataset("my_dataset").delete(catalog="my_catalog")
        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)

        url = f"{client.root_url}catalogs/{catalog}/datasets/{self.identifier}"
        resp: requests.Response = client.session.delete(url)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None

    def copy(
        self,
        catalog_to: str,
        catalog_from: Optional[str] = None,
        client: Optional[Fusion] = None,
        client_to: Optional[Fusion] = None,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
        """Copies dataset from one catalog/environment to another.

        Args:
            catalog_to (str): A catalog identifier to copy dataset to.
            catalog_from (Optional[str]): A catalog identifier to copy dataset from. Defaults to "common".
            client (Optional[Fusion]): A Fusion client object. Defaults to the instance's _client.
            client_to (Optional[Fusion]): A Fusion client object for the target environment. Defaults to `client`.
            return_resp_obj (bool): If True, return the response object. Defaults to False.

        Returns:
            Optional[requests.Response]: The response object if `return_resp_obj` is True.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset").copy(
            ...     catalog_from="my_catalog", catalog_to="my_new_catalog"
            ... )
        """
        client = self._use_client(client)
        catalog_from = client._use_catalog(catalog_from)

        if client_to is None:
            client_to = client

        dataset_obj = self.from_catalog(catalog=catalog_from, client=client)
        dataset_obj.client = client_to
        resp = dataset_obj.create(client=client_to, catalog=catalog_to, return_resp_obj=True)

        return resp if return_resp_obj else None

    def activate(
        self,
        catalog: Optional[str] = None,
        client: Optional[Fusion] = None,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
        """Activates a dataset by setting the `isActive` flag to True.

        Args:
            catalog (Optional[str]): A catalog identifier. Defaults to "common".
            client (Optional[Fusion]): A Fusion client object. Defaults to the instance's _client.
            return_resp_obj (bool): If True, return the response object. Defaults to False.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> fusion.dataset("my_dataset").activate(catalog="my_catalog")
        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)

        dataset_obj = self.from_catalog(catalog=catalog, client=client)
        dataset_obj.is_active = True
        resp = dataset_obj.update(catalog=catalog, client=client, return_resp_obj=return_resp_obj)

        return resp if return_resp_obj else None

    def add_to_product(
        self,
        product: str,
        catalog: Optional[str] = None,
        client: Optional[Fusion] = None,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
        """Maps dataset to a product.

        Args:
            product (str): A product identifier.
            catalog (Optional[str]): A catalog identifier. Defaults to "common".
            client (Optional[Fusion]): A Fusion client object. Defaults to the instance's _client.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> fusion.dataset("my_dataset").add_to_product(
            ...     product="MY_PRODUCT", catalog="my_catalog"
            ... )
        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)

        url = f"{client.root_url}catalogs/{catalog}/productDatasets"
        data = {"product": product, "datasets": [self.identifier]}
        resp: requests.Response = client.session.put(url=url, json=data)

        requests_raise_for_status(resp)

        return resp if return_resp_obj else None

    def remove_from_product(
        self,
        product: str,
        catalog: Optional[str] = None,
        client: Optional[Fusion] = None,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
        """Deletes dataset-to-product mapping.

        Args:
            product (str): A product identifier.
            catalog (Optional[str]): A catalog identifier. Defaults to "common".
            client (Optional[Fusion]): A Fusion client object. Defaults to the instance's _client.

        Examples:
            >>> from fusion37 import Fusion
            >>> fusion = Fusion()
            >>> fusion.dataset("my_dataset").remove_from_product(
            ...     product="MY_PRODUCT", catalog="my_catalog"
            ... )
        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)

        url = f"{client.root_url}catalogs/{catalog}/productDatasets/{product}/{self.identifier}"
        resp: requests.Response = client.session.delete(url=url)

        requests_raise_for_status(resp)

        return resp if return_resp_obj else None
