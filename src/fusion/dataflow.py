from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from .dataset import Dataset
from .utils import requests_raise_for_status

if TYPE_CHECKING:
    import requests

    from fusion import Fusion


@dataclass
class DataFlow(Dataset):
    """Dataflow class for maintaining data flow metadata.

    Attributes:
        producer_application_id (Optional[Dict[str, str]]): The producer application ID (upstream application
            producing the flow).
        consumer_application_id (Union[List[Dict[str, str]], Dict[str, str], None]): The consumer application ID
            (downstream application, consuming the flow).
        flow_details (Optional[Dict[str, str]]): The flow details. Specifies input versus output flow.
        type_ (Optional[str]): The type of dataset. Defaults to "Flow".
    """
    producer_application_id: Optional[Dict[str, str]] = None
    consumer_application_id: Optional[Union[List[Dict[str, str]], Dict[str, str]]] = None
    flow_details: Optional[Dict[str, str]] = None
    type_: Optional[str] = "Flow"

    def __post_init__(self) -> None:
        """Format the Data Flow object."""
        if isinstance(self.consumer_application_id, dict):
            self.consumer_application_id = [self.consumer_application_id]
        super().__post_init__()

    def add_registered_attribute(
        self,
        attribute_identifier: str,
        catalog: Optional[str] = None,
        client: Optional[Fusion] = None,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
        """Add a registered attribute to the Data Flow.

        Args:
            attribute_identifier (str): Attribute identifier.
            catalog (Optional[str], optional): Catalog identifier. Defaults to 'common'.
            client (Optional[Fusion], optional): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            Optional[requests.Response]: The response object from the API call if return_resp_obj is True, 
            otherwise None.
        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        dataset = self.identifier

        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{attribute_identifier}/registration"

        data = {
            "isCriticalDataElement": False,
        }

        resp = client.session.post(url, json=data)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None


@dataclass
class InputDataFlow(DataFlow):
    """InputDataFlow class for maintaining input data flow metadata."""
    flow_details: Optional[Dict[str, str]] = field(default_factory=lambda: {"flowDirection": "Input"})

    def __repr__(self) -> str:
        """Return an object representation of the InputDataFlow object.

        Returns:
            str: Object representation of the dataset.
        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"InputDataFlow(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"


@dataclass
class OutputDataFlow(DataFlow):
    """OutputDataFlow class for maintaining output data flow metadata."""
    flow_details: Optional[Dict[str, str]] = field(default_factory=lambda: {"flowDirection": "Output"})

    def __repr__(self) -> str:
        """Return an object representation of the OutputDataFlow object.

        Returns:
            str: Object representation of the dataset.
        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"OutputDataFlow(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"
