"""BEA (Bureau of Economic Analysis) adapter.

Source
------
U.S. Bureau of Economic Analysis, Regional Economic Accounts.
https://www.bea.gov/data/economic-accounts/regional

BEA provides annual county-level personal income and per-capita income
data through the Regional Economic Accounts (CAINC1 table).

Update frequency: Annual (released ~November for prior year).
Geographic coverage: All U.S. counties.
Time coverage: 1969-present.

Output Schema
-------------
county_fips              : str   -- 5-digit county FIPS code (zero-padded)
state_fips               : str   -- 2-digit state FIPS code (zero-padded)
year                     : int   -- Calendar year
personal_income          : float -- Total personal income (thousands of USD)
per_capita_personal_income : float -- Per-capita personal income (USD)
population               : int   -- Mid-year population estimate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from povcrime.config import ProjectConfig
from povcrime.data.base import BaseAdapter
from povcrime.utils import standardize_fips

logger = logging.getLogger(__name__)

# Expected output columns.
EXPECTED_COLUMNS: list[str] = [
    "county_fips",
    "state_fips",
    "year",
    "personal_income",
    "per_capita_personal_income",
    "population",
]

# BEA API base URL.
_BEA_API_URL = "https://apps.bea.gov/api/data/"

# CAINC1 LineCode definitions:
#   1 = Personal income (thousands of dollars)
#   2 = Population (persons)
#   3 = Per capita personal income (dollars)
_LINE_CODES: dict[int, str] = {
    1: "personal_income",
    2: "population",
    3: "per_capita_personal_income",
}


class BEAAdapter(BaseAdapter):
    """Adapter for BEA Regional Economic Accounts (county income data)."""

    def __init__(self, config: ProjectConfig) -> None:
        self._cfg = config
        self._raw_dir: Path = config.raw_dir / "bea"

    # ------------------------------------------------------------------ #
    # download
    # ------------------------------------------------------------------ #
    def download(self) -> None:
        """Download BEA county income data via the BEA API.

        Makes one API call per LineCode (1, 2, 3) with ``Year=ALL`` and
        ``GeoFips=COUNTY``.  Stores the raw JSON responses in
        ``data/raw/bea/cainc1_lc{linecode}.json``.
        """
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        api_key = self._cfg.bea_api_key
        if not api_key:
            raise RuntimeError(
                "BEA_API_KEY is not set. "
                "Set it in .env or as an environment variable."
            )

        for linecode, col_name in _LINE_CODES.items():
            out_path = self._raw_dir / f"cainc1_lc{linecode}.json"
            if out_path.exists():
                logger.info(
                    "BEA LineCode=%d (%s) already downloaded, skipping.",
                    linecode,
                    col_name,
                )
                continue

            params = {
                "UserID": api_key,
                "method": "GetData",
                "DataSetName": "Regional",
                "TableName": "CAINC1",
                "LineCode": str(linecode),
                "GeoFips": "COUNTY",
                "Year": "ALL",
                "ResultFormat": "JSON",
            }

            logger.info(
                "Downloading BEA CAINC1 LineCode=%d (%s) ...",
                linecode,
                col_name,
            )

            try:
                resp = requests.get(
                    _BEA_API_URL, params=params, timeout=120
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning(
                    "Failed to download BEA LineCode=%d: %s. Skipping.",
                    linecode,
                    exc,
                )
                continue

            # Validate that the response contains expected data.
            try:
                payload = resp.json()
            except ValueError as exc:
                logger.warning(
                    "Invalid JSON for BEA LineCode=%d: %s. Skipping.",
                    linecode,
                    exc,
                )
                continue

            bea_api = payload.get("BEAAPI", {})
            results = bea_api.get("Results", {})

            if "Error" in results:
                logger.warning(
                    "BEA API error for LineCode=%d: %s. Skipping.",
                    linecode,
                    results["Error"],
                )
                continue

            if "Data" not in results:
                logger.warning(
                    "BEA response missing 'Data' key for LineCode=%d. "
                    "Keys present: %s. Skipping.",
                    linecode,
                    list(results.keys()),
                )
                continue

            n_records = len(results["Data"])
            out_path.write_text(resp.text, encoding="utf-8")
            logger.info(
                "Saved BEA LineCode=%d -> %s (%d records)",
                linecode,
                out_path,
                n_records,
            )

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    def load(self) -> pd.DataFrame:
        """Load downloaded BEA data into a standardised DataFrame.

        Reads the three raw JSON files (one per LineCode), pivots them
        into a single table with one row per (county, year), and
        standardises column names and types.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns defined in ``EXPECTED_COLUMNS``.
        """
        pieces: dict[str, pd.DataFrame] = {}
        for linecode, col_name in _LINE_CODES.items():
            fp = self._raw_dir / f"cainc1_lc{linecode}.json"
            if not fp.exists():
                raise FileNotFoundError(
                    f"BEA raw file not found: {fp}. Run download() first."
                )
            piece = self._parse_json(fp, col_name)
            pieces[col_name] = piece

        # Merge the three measures on (county_fips, year).
        col_names = list(_LINE_CODES.values())
        df = pieces[col_names[0]]
        for col_name in col_names[1:]:
            df = df.merge(
                pieces[col_name],
                on=["county_fips", "state_fips", "year"],
                how="outer",
            )

        # Filter to the analysis window.
        df = df[
            (df["year"] >= self._cfg.start_year)
            & (df["year"] <= self._cfg.end_year)
        ].copy()

        # Drop rows with all-NaN measure values.
        measure_cols = list(_LINE_CODES.values())
        df = df.dropna(subset=measure_cols, how="all").copy()

        # Cast population to nullable integer.
        df["population"] = df["population"].astype("Int64")

        df = df.reset_index(drop=True)
        return df

    def _parse_json(self, path: Path, value_col: str) -> pd.DataFrame:
        """Parse a single BEA raw JSON file into a long DataFrame."""
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)

        records = payload["BEAAPI"]["Results"]["Data"]
        raw = pd.DataFrame(records)

        # Standardise FIPS.
        raw["county_fips"] = raw["GeoFips"].apply(
            lambda x: standardize_fips(x, width=5)
        )
        raw["state_fips"] = raw["county_fips"].str[:2]
        raw["year"] = pd.to_numeric(raw["TimePeriod"], errors="coerce")

        # Parse data values -- BEA may return "(NA)", "(D)", or other
        # non-numeric markers.
        raw[value_col] = pd.to_numeric(
            raw["DataValue"].str.replace(",", "", regex=False),
            errors="coerce",
        )

        # Keep only county-level records (5-digit FIPS not ending in 000).
        county_mask = raw["county_fips"].str[-3:] != "000"
        raw = raw[county_mask].copy()

        # Drop rows where the year or value is missing.
        raw = raw.dropna(subset=["year", value_col]).copy()
        raw["year"] = raw["year"].astype(int)

        df = raw[["county_fips", "state_fips", "year", value_col]].copy()
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # validate
    # ------------------------------------------------------------------ #
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate BEA DataFrame.

        Checks
        ------
        - All expected columns are present.
        - No duplicate (county_fips, year) rows.
        - Income values are positive.
        - Population is positive.

        Returns
        -------
        pd.DataFrame
            The validated DataFrame.
        """
        # --- column presence ---
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"BEA DataFrame is missing columns: {missing_cols}"
            )

        # --- duplicates ---
        dup_mask = df.duplicated(
            subset=["county_fips", "year"], keep="first"
        )
        n_dups = dup_mask.sum()
        if n_dups > 0:
            logger.warning(
                "Dropping %d duplicate (county_fips, year) rows.", n_dups
            )
            df = df[~dup_mask].copy()

        # --- value ranges ---
        for col in ("personal_income", "per_capita_personal_income"):
            neg = df[col] < 0
            if neg.any():
                logger.warning(
                    "Dropping %d rows with negative %s.", neg.sum(), col
                )
                df = df[~neg].copy()

        neg_pop = df["population"] < 0
        if neg_pop.any():
            logger.warning(
                "Dropping %d rows with negative population.",
                neg_pop.sum(),
            )
            df = df[~neg_pop].copy()

        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # get_metadata
    # ------------------------------------------------------------------ #
    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for the BEA data source."""
        return {
            "source_name": "BEA Regional Economic Accounts",
            "url": (
                "https://www.bea.gov/data/economic-accounts/regional"
            ),
            "api_endpoint": _BEA_API_URL,
            "table": "CAINC1",
            "line_codes": _LINE_CODES,
            "update_frequency": "Annual",
            "geographic_coverage": "All U.S. counties",
            "time_coverage": (
                f"{self._cfg.start_year}-{self._cfg.end_year}"
            ),
            "expected_columns": EXPECTED_COLUMNS,
        }
