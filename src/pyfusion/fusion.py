import json as js
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests
from tabulate import tabulate

logger = logging.getLogger(__name__)
VERBOSE_LVL = 25


class Fusion:
    """Core Fusion class for API access."""

    @staticmethod
    def _call_for_dataframe(url: str, session: requests.Session) -> pd.DataFrame:
        """Private function that calls an API endpoint and returns the data as a pandas dataframe.

        Args:
            url (str): URL for an API endpoint with valid parameters.
            session (requests.Session): Session object for making the API call.

        Returns:
            pandas.DataFrame: A dataframe containing the requested data.
        """
        response = session.get(url)
        response.raise_for_status()
        table = response.json()["resources"]
        ret_df = pd.DataFrame(table).reset_index(drop=True)
        return ret_df

    @staticmethod
    def _call_for_bytes_object(url: str, session: requests.Session) -> BytesIO:
        """Private function that calls an API endpoint and returns the data as a bytes object in memory.

        Args:
            url (str): URL for an API endpoint with valid parameters.
            session (requests.Session): Session object for making the API call.

        Returns:
            io.BytesIO: In-memory file content.
        """
        response = session.get(url)
        response.raise_for_status()
        return BytesIO(response.content)

    def __init__(
        self,
        credentials: Union[str, dict] = "config/client_credentials.json",
        root_url: str = "https://fusion.jpmorgan.com/api/v1/",
        download_folder: str = "downloads",
        log_level: int = logging.ERROR,
        log_path: str = ".",
    ) -> None:
        """Constructor to instantiate a new Fusion object.

        Args:
            credentials (Union[str, dict]): A path to a credentials file or a dictionary containing credentials.
                Defaults to 'config/client_credentials.json'.
            root_url (str): The API root URL. Defaults to "https://fusion.jpmorgan.com/api/v1/".
            download_folder (str): The folder path where downloaded data files are saved. Defaults to "downloads".
            log_level (int): Set the logging level. Defaults to logging.ERROR.
            log_path (str): The folder path where the log is stored. Defaults to the current directory.
        """
        self._default_catalog = "common"

        self.root_url = root_url
        self.download_folder = download_folder
        Path(download_folder).mkdir(parents=True, exist_ok=True)

        if logger.hasHandlers():
            logger.handlers.clear()
        file_handler = logging.FileHandler(filename=f"{log_path}/fusion_sdk.log")
        logging.addLevelName(VERBOSE_LVL, "VERBOSE")
        stdout_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(name)s:%(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)
        logger.setLevel(log_level)

        if isinstance(credentials, dict):
            self.credentials = credentials
        elif isinstance(credentials, str):
            self.credentials = self._load_credentials(Path(credentials))
        else:
            raise ValueError("credentials must be a path to a credentials file or a dictionary")

        self.session = self._create_session(self.credentials)

    def __repr__(self) -> str:
        """Object representation to list all available methods."""
        methods = [
            method_name
            for method_name in dir(Fusion)
            if callable(getattr(Fusion, method_name)) and not method_name.startswith("_")
        ]
        return "Fusion object \nAvailable methods:\n" + tabulate(
            [[method_name] for method_name in methods],
            headers=["Method Name"],
            tablefmt="psql",
        )

    @staticmethod
    def _load_credentials(credentials_path: Path) -> dict:
        """Load credentials from a JSON file.

        Args:
            credentials_path (Path): Path to the credentials file.

        Returns:
            dict: Loaded credentials.
        """
        with credentials_path.open() as file:
            return js.load(file)

    @staticmethod
    def _create_session(credentials: dict) -> requests.Session:
        """Create and return a session configured with the provided credentials.

        Args:
            credentials (dict): Dictionary containing session credentials.

        Returns:
            requests.Session: Configured requests session.
        """
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {credentials.get('token', '')}"})
        return session

    @property
    def default_catalog(self) -> str:
        """Returns the default catalog."""
        return self._default_catalog

    @default_catalog.setter
    def default_catalog(self, catalog: str) -> None:
        """Allow the default catalog, which is "common," to be overridden.

        Args:
            catalog (str): The catalog to use as the default.
        """
        self._default_catalog = catalog

    def _use_catalog(self, catalog: Optional[str]) -> str:
        """Determine which catalog to use in an API call.

        Args:
            catalog (Optional[str]): The catalog value passed as an argument to an API function wrapper.

        Returns:
            str: The catalog to use.
        """
        return catalog if catalog else self.default_catalog
