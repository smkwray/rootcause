"""County adjacency adapter using the public Census adjacency text file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from povcrime.config import ProjectConfig
from povcrime.data.base import BaseAdapter

logger = logging.getLogger(__name__)

_ADJACENCY_URL = "https://www2.census.gov/geo/docs/reference/county_adjacency.txt"
_RAW_NAME = "county_adjacency.txt"
_HEADERS = {"User-Agent": "Mozilla/5.0 (povcrime research)"}
EXPECTED_COLUMNS = [
    "county_fips",
    "neighbor_county_fips",
    "state_fips",
    "neighbor_state_fips",
    "county_name",
    "neighbor_county_name",
]


class CountyAdjacencyAdapter(BaseAdapter):
    """Adapter for public Census county adjacency pairs."""

    def __init__(self, config: ProjectConfig) -> None:
        self._cfg = config
        self._raw_dir: Path = config.raw_dir / "county_adjacency"

    def download(self) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._raw_dir / _RAW_NAME
        if out_path.exists():
            logger.info("County adjacency text already downloaded, skipping.")
            return

        resp = requests.get(_ADJACENCY_URL, timeout=120, headers=_HEADERS)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        logger.info("Saved county adjacency text -> %s", out_path)

    def load(self) -> pd.DataFrame:
        path = self._raw_dir / _RAW_NAME
        if not path.exists():
            raise FileNotFoundError(f"County adjacency file not found: {path}")

        try:
            raw = pd.read_csv(
                path,
                sep="\t",
                header=None,
                names=["county_name", "county_fips", "neighbor_county_name", "neighbor_county_fips"],
                dtype=str,
            )
        except UnicodeDecodeError:
            logger.warning(
                "County adjacency text is not clean UTF-8; retrying with latin-1 decoding."
            )
            raw = pd.read_csv(
                path,
                sep="\t",
                header=None,
                names=["county_name", "county_fips", "neighbor_county_name", "neighbor_county_fips"],
                dtype=str,
                encoding="latin-1",
            )
        return _reshape_adjacency(raw)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"County adjacency DataFrame is missing columns: {sorted(missing)}")

        dupes = df.duplicated(subset=["county_fips", "neighbor_county_fips"], keep="first")
        if dupes.any():
            logger.warning("Dropping %d duplicate county adjacency rows.", int(dupes.sum()))
            df = df.loc[~dupes].copy()

        self_links = df["county_fips"] == df["neighbor_county_fips"]
        if self_links.any():
            logger.warning("Dropping %d county self-links.", int(self_links.sum()))
            df = df.loc[~self_links].copy()

        return df.sort_values(["county_fips", "neighbor_county_fips"]).reset_index(drop=True)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "source_name": "Census County Adjacency",
            "url": _ADJACENCY_URL,
            "data_level": "county-pair",
            "update_frequency": "periodic",
            "expected_columns": EXPECTED_COLUMNS,
            "notes": "Public Census text file listing adjacent counties, including cross-state borders.",
        }


def _reshape_adjacency(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["county_name"] = df["county_name"].replace("", pd.NA).ffill()
    df["county_fips"] = df["county_fips"].replace("", pd.NA).ffill()
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["neighbor_county_fips"] = df["neighbor_county_fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["county_fips"].str[:2]
    df["neighbor_state_fips"] = df["neighbor_county_fips"].str[:2]
    return df[EXPECTED_COLUMNS].reset_index(drop=True)
