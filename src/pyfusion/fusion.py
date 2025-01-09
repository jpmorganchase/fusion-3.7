"""Main Fusion module."""

from __future__ import annotations

import json as js
import logging
import sys
from io import BytesIO
from typing import TYPE_CHECKING

import pandas as pd
import requests

if TYPE_CHECKING:
    from requests import Session

logger = logging.getLogger(__name__)


class Fusion:
    """Core Fusion class for API access."""

    def __init__(
        self,
        credentials: str,
        root_url: str = "https://fusion.jpmorgan.com/api/v1/",
        log_level: int = logging.ERROR,
        log_path: str = ".",
    ) -> None:
        """Constructor to instantiate a new Fusion object.

        Args:
            credentials (str): Path to a credentials file.
            root_url (str): The API root URL. Defaults to "https://fusion.jpmorgan.com/api/v1/".
            log_level (int): Set the logging level. Defaults to logging.ERROR.
            log_path (str): Path where logs will be stored. Defaults to current directory.
        """
        self.root_url = root_url

        if logger.hasHandlers():
            logger.handlers.clear()
        file_handler = logging.FileHandler(filename=f"{log_path}/fusion_sdk.log")
        stdout_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(name)s:%(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)
        logger.setLevel(log_level)

        # Load credentials
        self.credentials = self._load_credentials(credentials)
        self.session = self._create_session()

    def _load_credentials(self, credentials_path: str) -> dict:
        """Load credentials from a JSON file.

        Args:
            credentials_path (str): Path to the credentials file.

        Returns:
            dict: Credentials data.

        Raises:
            ValueError: If the credentials file is not found.
        """
        try:
            with open(credentials_path) as file:
                return js.load(file)
        except FileNotFoundError as err:
            raise ValueError(f"Credentials file not found at {credentials_path}") from err

    def _create_session(self) -> Session:
        """Create and configure a requests session.

        Returns:
            Session: Configured requests session.
        """
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {self.credentials['token']}"})
        return session

    @staticmethod
    def _call_for_dataframe(url: str, session: Session) -> pd.DataFrame:
        """Private function that calls an API endpoint and returns the data as a pandas DataFrame.

        Args:
            url (str): URL for an API endpoint with valid parameters.
            session (Session): Requests session with authentication.

        Returns:
            pd.DataFrame: A DataFrame containing the requested data.
        """
        response = session.get(url)
        response.raise_for_status()
        data = response.json().get("resources", [])
        return pd.DataFrame(data).reset_index(drop=True)

    @staticmethod
    def _call_for_bytes_object(url: str, session: Session) -> BytesIO:
        """Private function that calls an API endpoint and returns the data as a BytesIO object.

        Args:
            url (str): URL for an API endpoint with valid parameters.
            session (Session): Requests session with authentication.

        Returns:
            BytesIO: In-memory file content.
        """
        response = session.get(url)
        response.raise_for_status()
        return BytesIO(response.content)

    def list_catalogs(self) -> pd.DataFrame:
        """Lists the catalogs available to the API account.

        Returns:
            pd.DataFrame: A DataFrame with a row for each catalog.
        """
        url = f"{self.root_url}catalogs/"
        return self._call_for_dataframe(url, self.session)

    def catalog_resources(self, catalog: str) -> pd.DataFrame:
        """List the resources contained within a catalog.

        Args:
            catalog (str): Catalog identifier.

        Returns:
            pd.DataFrame: A DataFrame with a row for each resource in the catalog.
        """
        url = f"{self.root_url}catalogs/{catalog}"
        return self._call_for_dataframe(url, self.session)

    def download_as_dataframe(self, endpoint: str) -> pd.DataFrame:
        """Download data from a specified API endpoint and return it as a DataFrame.

        Args:
            endpoint (str): API endpoint relative to the root URL.

        Returns:
            pd.DataFrame: Data from the API as a DataFrame.
        """
        url = f"{self.root_url}{endpoint}"
        return self._call_for_dataframe(url, self.session)
