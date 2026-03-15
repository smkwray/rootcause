"""LAUS (Local Area Unemployment Statistics) adapter.

Source
------
U.S. Bureau of Labor Statistics, Local Area Unemployment Statistics.
https://www.bls.gov/lau/

LAUS provides monthly and annual labor-force estimates for all U.S.
counties (and county equivalents).

Update frequency: Monthly; annual averages released each spring.
Geographic coverage: All U.S. counties.
Time coverage: 1990-present.

Output Schema
-------------
county_fips      : str   -- 5-digit county FIPS code (zero-padded)
state_fips       : str   -- 2-digit state FIPS code (zero-padded)
year             : int   -- Calendar year
unemployment_rate : float -- Annual average unemployment rate (percent)
labor_force      : int   -- Annual average civilian labor force
employed         : int   -- Annual average number employed
unemployed       : int   -- Annual average number unemployed
"""

from __future__ import annotations

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
    "unemployment_rate",
    "labor_force",
    "employed",
    "unemployed",
]

# BLS flat-file URL pattern for county annual averages.
# Files are Excel workbooks named ``laucnty{yy}.xlsx`` where *yy* is the
# two-digit year.  Available for 1990-present (2025 not yet released at
# time of writing).
_LAUS_URL = "https://www.bls.gov/lau/laucnty{yy:02d}.xlsx"

# BLS blocks the default python-requests User-Agent.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Column names for the raw Excel files (row index 1 is the header).
_RAW_COLUMNS = [
    "laus_code",
    "state_fips_raw",
    "county_fips_raw",
    "county_name",
    "year_raw",
    "labor_force",
    "employed",
    "unemployed",
    "unemployment_rate",
]


class LAUSAdapter(BaseAdapter):
    """Adapter for BLS Local Area Unemployment Statistics (LAUS)."""

    def __init__(self, config: ProjectConfig) -> None:
        self._cfg = config
        self._raw_dir: Path = config.raw_dir / "laus"

    # ------------------------------------------------------------------ #
    # download
    # ------------------------------------------------------------------ #
    def download(self) -> None:
        """Download LAUS annual average data from BLS.

        For each year in the analysis window, downloads the Excel workbook
        ``laucnty{yy}.xlsx`` from BLS and saves it to
        ``data/raw/laus/laucnty{yy}.xlsx``.
        """
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        for year in range(self._cfg.start_year, self._cfg.end_year + 1):
            yy = year % 100
            filename = f"laucnty{yy:02d}.xlsx"
            out_path = self._raw_dir / filename

            if out_path.exists():
                logger.info("LAUS %d already downloaded, skipping.", year)
                continue

            url = _LAUS_URL.format(yy=yy)
            logger.info("Downloading LAUS %d from %s ...", year, url)

            try:
                resp = requests.get(url, timeout=60, headers=_HEADERS)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning(
                    "Failed to download LAUS %d: %s. Skipping.", year, exc
                )
                continue

            # Sanity-check: valid Excel files are at least a few KB.
            if len(resp.content) < 5000:
                logger.warning(
                    "LAUS %d response too small (%d bytes), skipping.",
                    year,
                    len(resp.content),
                )
                continue

            out_path.write_bytes(resp.content)
            logger.info(
                "Saved LAUS %d -> %s (%d bytes)",
                year,
                out_path,
                len(resp.content),
            )

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    def load(self) -> pd.DataFrame:
        """Load downloaded LAUS data into a standardised DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns defined in ``EXPECTED_COLUMNS``.
        """
        xlsx_files = sorted(self._raw_dir.glob("laucnty*.xlsx"))
        if not xlsx_files:
            raise FileNotFoundError(
                f"No LAUS Excel files found in {self._raw_dir}. "
                "Run download() first."
            )

        frames: list[pd.DataFrame] = []
        for fp in xlsx_files:
            df_year = self._parse_xlsx(fp)
            if df_year is not None:
                frames.append(df_year)

        if not frames:
            raise ValueError(
                "All LAUS files failed to parse. Check raw files in "
                f"{self._raw_dir}."
            )

        df = pd.concat(frames, ignore_index=True)
        return df

    def _parse_xlsx(self, path: Path) -> pd.DataFrame | None:
        """Parse a single BLS LAUS Excel file into a standardised frame."""
        try:
            raw = pd.read_excel(path, header=None, skiprows=2)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None

        if raw.shape[1] != 9:
            logger.warning(
                "Unexpected column count (%d) in %s, skipping.",
                raw.shape[1],
                path,
            )
            return None

        raw.columns = _RAW_COLUMNS

        # Drop footer rows and blanks (state_fips_raw is NaN for them).
        raw = raw.dropna(subset=["state_fips_raw"]).copy()

        # Build standardised columns.
        df = pd.DataFrame()
        df["state_fips"] = (
            raw["state_fips_raw"]
            .astype(float)
            .astype(int)
            .apply(lambda x: standardize_fips(x, width=2))
        )
        df["county_fips"] = df["state_fips"] + (
            raw["county_fips_raw"]
            .astype(float)
            .astype(int)
            .apply(lambda x: standardize_fips(x, width=3))
        )
        df["year"] = raw["year_raw"].astype(float).astype(int)
        df["labor_force"] = pd.to_numeric(
            raw["labor_force"], errors="coerce"
        )
        df["employed"] = pd.to_numeric(raw["employed"], errors="coerce")
        df["unemployed"] = pd.to_numeric(
            raw["unemployed"], errors="coerce"
        )
        df["unemployment_rate"] = pd.to_numeric(
            raw["unemployment_rate"], errors="coerce"
        )

        # Cast integer columns (allow NaN -> nullable int).
        for col in ("labor_force", "employed", "unemployed"):
            df[col] = df[col].astype("Int64")

        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # validate
    # ------------------------------------------------------------------ #
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate LAUS DataFrame.

        Checks
        ------
        - All expected columns are present.
        - No duplicate (county_fips, year) rows.
        - Unemployment rate is between 0 and 100.
        - Labor-force >= employed + unemployed (approx).

        Returns
        -------
        pd.DataFrame
            The validated DataFrame.
        """
        # --- column presence ---
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"LAUS DataFrame is missing columns: {missing_cols}"
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
        bad_rate = (df["unemployment_rate"] < 0) | (
            df["unemployment_rate"] > 100
        )
        if bad_rate.any():
            logger.warning(
                "Dropping %d rows with unemployment_rate outside [0, 100].",
                bad_rate.sum(),
            )
            df = df[~bad_rate].copy()

        # Labor force should be >= employed + unemployed (allow small
        # rounding tolerance).
        lf = df["labor_force"].astype(float)
        emp_unemp = df["employed"].astype(float) + df["unemployed"].astype(
            float
        )
        bad_lf = lf < (emp_unemp - 1)
        if bad_lf.any():
            logger.warning(
                "%d rows where labor_force < employed + unemployed.",
                bad_lf.sum(),
            )

        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # get_metadata
    # ------------------------------------------------------------------ #
    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for the LAUS data source."""
        return {
            "source_name": "LAUS",
            "url": "https://www.bls.gov/lau/",
            "file_pattern": _LAUS_URL,
            "update_frequency": "Annual averages released each spring",
            "geographic_coverage": "All U.S. counties",
            "time_coverage": (
                f"{self._cfg.start_year}-{self._cfg.end_year}"
            ),
            "expected_columns": EXPECTED_COLUMNS,
        }
