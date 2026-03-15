"""Base adapter interface for all data sources.

Every data adapter inherits from :class:`BaseAdapter` and implements the
four lifecycle methods: ``download``, ``load``, ``validate``, and
``get_metadata``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseAdapter(ABC):
    """Abstract base class for data-source adapters.

    Subclasses must implement all four abstract methods.  The expected
    workflow is:

    1. ``download()`` -- fetch raw data to local storage.
    2. ``load()`` -- read raw data into a :class:`~pandas.DataFrame`.
    3. ``validate()`` -- run quality checks on the loaded frame.
    4. ``get_metadata()`` -- return a dict describing the source.
    """

    @abstractmethod
    def download(self) -> None:
        """Download raw data from the upstream source."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load raw data into a DataFrame with a standardised schema."""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate a loaded DataFrame and return it (possibly cleaned)."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return a metadata dict describing the data source."""
