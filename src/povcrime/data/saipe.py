"""SAIPE (Small Area Income and Poverty Estimates) adapter.

Source
------
U.S. Census Bureau, Small Area Income and Poverty Estimates (SAIPE).
https://www.census.gov/programs-surveys/saipe.html

The SAIPE program provides annual estimates of income and poverty for all
U.S. counties and states.  Data are available via the Census API timeseries
endpoint:

    https://api.census.gov/data/timeseries/poverty/saipe

Update frequency: Annual (released ~December for prior calendar year).
Geographic coverage: All U.S. counties (FIPS).
Time coverage: 1989-present (county-level from 1995).

Output Schema
-------------
county_fips : str   -- 5-digit county FIPS code (zero-padded)
state_fips  : str   -- 2-digit state FIPS code (zero-padded)
year        : int   -- Calendar year
poverty_rate       : float -- Estimated poverty rate (percent)
median_hh_income   : float -- Estimated median household income (USD)
poverty_count      : int   -- Estimated number of persons in poverty
population         : int   -- Total population estimate
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from povcrime.config import ProjectConfig
from povcrime.data.base import BaseAdapter

logger = logging.getLogger(__name__)

# Expected output columns.
EXPECTED_COLUMNS: list[str] = [
    "county_fips",
    "state_fips",
    "year",
    "poverty_rate",
    "median_hh_income",
    "poverty_count",
    "population",
]

# Census SAIPE API timeseries endpoint.
# Variable names: SAEPOVRTALL_PT = poverty rate, SAEMHI_PT = median HH income,
# SAEPOVALL_PT = poverty count, SAEPOVU_ALL = poverty universe (total pop).
_SAIPE_URL = (
    "https://api.census.gov/data/timeseries/poverty/saipe"
    "?get=SAEPOVRTALL_PT,SAEMHI_PT,SAEPOVALL_PT,SAEPOVU_ALL"
    "&for=county:*&YEAR={year}&key={api_key}"
)


class SAIPEAdapter(BaseAdapter):
    """Adapter for Census SAIPE annual county poverty and income data."""

    def __init__(self, config: ProjectConfig) -> None:
        self._cfg = config
        self._raw_dir: Path = config.raw_dir / "saipe"

    # ------------------------------------------------------------------ #
    # download
    # ------------------------------------------------------------------ #
    def download(self) -> None:
        """Download SAIPE data from the Census API.

        Iterates over each year in the analysis window (start_year to
        end_year) and stores raw API responses as Parquet files in
        ``data/raw/saipe/saipe_{year}.parquet``.
        """
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        api_key = self._cfg.census_api_key
        if not api_key:
            raise RuntimeError(
                "CENSUS_API_KEY is not set. "
                "Set it in .env or as an environment variable."
            )

        for year in range(self._cfg.start_year, self._cfg.end_year + 1):
            out_path = self._raw_dir / f"saipe_{year}.parquet"
            if out_path.exists():
                logger.info("SAIPE %d already downloaded, skipping.", year)
                continue

            url = _SAIPE_URL.format(year=year, api_key=api_key)
            logger.info("Downloading SAIPE data for %d ...", year)

            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning(
                    "Failed to download SAIPE %d: %s. Skipping.", year, exc
                )
                continue

            try:
                payload = resp.json()
            except ValueError as exc:
                logger.warning(
                    "Invalid JSON for SAIPE %d: %s. Skipping.", year, exc
                )
                continue

            # Census API returns header row + data rows.
            header = payload[0]
            rows = payload[1:]

            df = pd.DataFrame(rows, columns=header)
            df["YEAR"] = year  # ensure year column is present
            df.to_parquet(out_path, index=False)
            logger.info(
                "Saved SAIPE %d -> %s (%d rows)", year, out_path, len(df)
            )

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    def load(self) -> pd.DataFrame:
        """Load downloaded SAIPE parquet files into a standardised DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: county_fips, state_fips, year,
            poverty_rate, median_hh_income, poverty_count, population.
        """
        parquet_files = sorted(self._raw_dir.glob("saipe_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No SAIPE parquet files found in {self._raw_dir}. "
                "Run download() first."
            )

        frames: list[pd.DataFrame] = []
        for fp in parquet_files:
            frames.append(pd.read_parquet(fp))

        raw = pd.concat(frames, ignore_index=True)

        # Build 5-digit county FIPS = state (2-digit) + county (3-digit).
        raw["state"] = raw["state"].astype(str).str.zfill(2)
        raw["county"] = raw["county"].astype(str).str.zfill(3)

        df = pd.DataFrame()
        df["county_fips"] = raw["state"] + raw["county"]
        df["state_fips"] = raw["state"]
        df["year"] = pd.to_numeric(raw["YEAR"], errors="coerce").astype(int)
        df["poverty_rate"] = pd.to_numeric(
            raw["SAEPOVRTALL_PT"], errors="coerce"
        )
        df["median_hh_income"] = pd.to_numeric(
            raw["SAEMHI_PT"], errors="coerce"
        )
        df["poverty_count"] = pd.to_numeric(
            raw["SAEPOVALL_PT"], errors="coerce"
        )
        df["population"] = pd.to_numeric(
            raw["SAEPOVU_ALL"], errors="coerce"
        )

        # Drop rows where critical values are missing (state-level totals
        # have county == "000" -- exclude those).
        df = df[df["county_fips"].str[-3:] != "000"].copy()

        # Drop rows with missing critical columns.
        df.dropna(
            subset=["poverty_rate", "population"], inplace=True
        )

        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # validate
    # ------------------------------------------------------------------ #
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate SAIPE DataFrame.

        Checks
        ------
        - All expected columns are present.
        - No duplicate (county_fips, year) rows.
        - Poverty rate is between 0 and 100.
        - Population and poverty_count are non-negative.

        Returns
        -------
        pd.DataFrame
            The validated (and possibly cleaned) DataFrame.
        """
        # --- column presence ---
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"SAIPE DataFrame is missing columns: {missing_cols}"
            )

        # --- duplicates ---
        dup_mask = df.duplicated(subset=["county_fips", "year"], keep="first")
        n_dups = dup_mask.sum()
        if n_dups > 0:
            logger.warning(
                "Dropping %d duplicate (county_fips, year) rows.", n_dups
            )
            df = df[~dup_mask].copy()

        # --- value ranges ---
        bad_rate = (df["poverty_rate"] < 0) | (df["poverty_rate"] > 100)
        if bad_rate.any():
            logger.warning(
                "Dropping %d rows with poverty_rate outside [0, 100].",
                bad_rate.sum(),
            )
            df = df[~bad_rate].copy()

        for col in ("poverty_count", "population"):
            neg = df[col] < 0
            if neg.any():
                logger.warning(
                    "Dropping %d rows with negative %s.", neg.sum(), col
                )
                df = df[~neg].copy()

        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # get_metadata
    # ------------------------------------------------------------------ #
    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for the SAIPE data source."""
        return {
            "source_name": "SAIPE",
            "url": "https://www.census.gov/programs-surveys/saipe.html",
            "api_endpoint": (
                "https://api.census.gov/data/timeseries/poverty/saipe"
            ),
            "update_frequency": "Annual",
            "geographic_coverage": "All U.S. counties",
            "time_coverage": f"{self._cfg.start_year}-{self._cfg.end_year}",
            "expected_columns": EXPECTED_COLUMNS,
        }
