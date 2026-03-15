"""ACS (American Community Survey) adapter.

Source
------
U.S. Census Bureau, American Community Survey 5-Year Estimates.
https://www.census.gov/programs-surveys/acs

The ACS 5-year estimates provide detailed demographic and socioeconomic
data for all U.S. counties.  Each release is a *pooled* estimate covering
a 5-year collection period (e.g., 2015-2019).

**Caveat:** ACS 5-year estimates are NOT point-in-time annual snapshots.
They represent pooled data collected over the full 5-year window.  Care is
needed when aligning these with annual treatment and outcome data.

Update frequency: Annual release (each covers a rolling 5-year window).
Geographic coverage: All U.S. counties.
Time coverage: First county-level 5-year estimates cover 2005-2009.

Output Schema
-------------
county_fips : str   -- 5-digit county FIPS code (zero-padded)
state_fips  : str   -- 2-digit state FIPS code (zero-padded)
year        : int   -- End year of the 5-year ACS window
pct_male             : float -- Percent male
pct_white            : float -- Percent white alone
pct_black            : float -- Percent Black or African American alone
pct_hispanic         : float -- Percent Hispanic or Latino (any race)
pct_under_18         : float -- Percent under age 18
pct_over_65          : float -- Percent age 65 and over
pct_hs_or_higher     : float -- Percent with high-school diploma or higher
pct_bachelor_or_higher : float -- Percent with bachelor's degree or higher
pct_foreign_born     : float -- Percent foreign-born
median_age           : float -- Median age (years)
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
    "pct_male",
    "pct_white",
    "pct_black",
    "pct_hispanic",
    "pct_under_18",
    "pct_over_65",
    "pct_hs_or_higher",
    "pct_bachelor_or_higher",
    "pct_foreign_born",
    "median_age",
]

# First year that ACS 5-year county-level data is available (end-year of
# the 2005-2009 5-year window).
_ACS_FIRST_YEAR = 2009

# First year B15003 education table is available; earlier years use B15002.
_B15003_FIRST_YEAR = 2012

# Core B-table variables available across all ACS 5-year vintages (2009+).
_CORE_VARIABLES: list[str] = [
    "B01003_001E",  # Total population
    "B01002_001E",  # Median age
    "B01001_002E",  # Total male
    "B02001_002E",  # White alone
    "B02001_003E",  # Black alone
    "B03003_003E",  # Hispanic or Latino
    "B09001_001E",  # Under 18 population
    # Male 65+ age groups
    "B01001_020E",
    "B01001_021E",
    "B01001_022E",
    "B01001_023E",
    "B01001_024E",
    "B01001_025E",
    # Female 65+ age groups
    "B01001_044E",
    "B01001_045E",
    "B01001_046E",
    "B01001_047E",
    "B01001_048E",
    "B01001_049E",
    # Foreign-born
    "B05002_013E",
]

# Education variables for years 2012+ (B15003 table).
_EDU_B15003: list[str] = [
    "B15003_001E",  # Total 25+ (denominator)
    "B15003_017E",  # HS diploma
    "B15003_018E",  # GED / alternative
    "B15003_019E",  # Some college < 1 year
    "B15003_020E",  # Some college 1+ years
    "B15003_021E",  # Associate's degree
    "B15003_022E",  # Bachelor's degree
    "B15003_023E",  # Master's degree
    "B15003_024E",  # Professional school degree
    "B15003_025E",  # Doctorate degree
]

# Education variables for years 2009-2011 (B15002 table, by sex).
_EDU_B15002: list[str] = [
    "B15002_001E",  # Total 25+ (denominator)
    # Male HS or higher
    "B15002_011E",  # HS graduate / GED / alternative
    "B15002_012E",  # Some college < 1 year
    "B15002_013E",  # Some college 1+ years
    "B15002_014E",  # Associate's
    "B15002_015E",  # Bachelor's
    "B15002_016E",  # Master's
    "B15002_017E",  # Professional school
    "B15002_018E",  # Doctorate
    # Female HS or higher
    "B15002_028E",  # HS graduate / GED / alternative
    "B15002_029E",  # Some college < 1 year
    "B15002_030E",  # Some college 1+ years
    "B15002_031E",  # Associate's
    "B15002_032E",  # Bachelor's
    "B15002_033E",  # Master's
    "B15002_034E",  # Professional school
    "B15002_035E",  # Doctorate
]

_ACS_URL = (
    "https://api.census.gov/data/{year}/acs/acs5"
    "?get={variables}&for=county:*&key={api_key}"
)

# Male 65+ variable names.
_MALE_65_PLUS = [
    "B01001_020E",
    "B01001_021E",
    "B01001_022E",
    "B01001_023E",
    "B01001_024E",
    "B01001_025E",
]
# Female 65+ variable names.
_FEMALE_65_PLUS = [
    "B01001_044E",
    "B01001_045E",
    "B01001_046E",
    "B01001_047E",
    "B01001_048E",
    "B01001_049E",
]

# B15003 HS-or-higher = 017..025 (HS diploma through doctorate).
_HS_OR_HIGHER_B15003 = [
    "B15003_017E",
    "B15003_018E",
    "B15003_019E",
    "B15003_020E",
    "B15003_021E",
    "B15003_022E",
    "B15003_023E",
    "B15003_024E",
    "B15003_025E",
]
# B15003 bachelor's or higher = 022..025.
_BACHELOR_OR_HIGHER_B15003 = [
    "B15003_022E",
    "B15003_023E",
    "B15003_024E",
    "B15003_025E",
]

# B15002 HS-or-higher (male + female).
_HS_OR_HIGHER_B15002 = [
    "B15002_011E",
    "B15002_012E",
    "B15002_013E",
    "B15002_014E",
    "B15002_015E",
    "B15002_016E",
    "B15002_017E",
    "B15002_018E",
    "B15002_028E",
    "B15002_029E",
    "B15002_030E",
    "B15002_031E",
    "B15002_032E",
    "B15002_033E",
    "B15002_034E",
    "B15002_035E",
]
# B15002 bachelor's or higher (male + female).
_BACHELOR_OR_HIGHER_B15002 = [
    "B15002_015E",
    "B15002_016E",
    "B15002_017E",
    "B15002_018E",
    "B15002_032E",
    "B15002_033E",
    "B15002_034E",
    "B15002_035E",
]


def _variables_for_year(year: int) -> list[str]:
    """Return the full list of API variables to request for a given year."""
    edu = _EDU_B15003 if year >= _B15003_FIRST_YEAR else _EDU_B15002
    return _CORE_VARIABLES + edu


class ACSAdapter(BaseAdapter):
    """Adapter for Census ACS 5-year demographic control variables."""

    def __init__(self, config: ProjectConfig) -> None:
        self._cfg = config
        self._raw_dir: Path = config.raw_dir / "acs"

    # ------------------------------------------------------------------ #
    # download
    # ------------------------------------------------------------------ #
    def download(self) -> None:
        """Download ACS 5-year data from the Census API.

        Iterates over end-years from ``_ACS_FIRST_YEAR`` (2009) to
        ``end_year`` and stores raw responses as Parquet files in
        ``data/raw/acs/acs_{year}.parquet``.
        """
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        api_key = self._cfg.census_api_key
        if not api_key:
            raise RuntimeError(
                "CENSUS_API_KEY is not set. "
                "Set it in .env or as an environment variable."
            )

        start = max(self._cfg.start_year, _ACS_FIRST_YEAR)

        for year in range(start, self._cfg.end_year + 1):
            out_path = self._raw_dir / f"acs_{year}.parquet"
            if out_path.exists():
                logger.info("ACS %d already downloaded, skipping.", year)
                continue

            variables = _variables_for_year(year)
            variables_str = ",".join(variables)
            url = _ACS_URL.format(
                year=year, variables=variables_str, api_key=api_key
            )
            logger.info("Downloading ACS 5-year data for %d ...", year)

            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning(
                    "Failed to download ACS %d: %s. Skipping.", year, exc
                )
                continue

            try:
                payload = resp.json()
            except ValueError as exc:
                logger.warning(
                    "Invalid JSON for ACS %d: %s. Skipping.", year, exc
                )
                continue

            # Census API returns header row + data rows.
            header = payload[0]
            rows = payload[1:]

            df = pd.DataFrame(rows, columns=header)
            df["YEAR"] = year
            df.to_parquet(out_path, index=False)
            logger.info(
                "Saved ACS %d -> %s (%d rows)", year, out_path, len(df)
            )

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    def load(self) -> pd.DataFrame:
        """Load downloaded ACS parquet files and compute demographic percentages.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns defined in ``EXPECTED_COLUMNS``.
        """
        parquet_files = sorted(self._raw_dir.glob("acs_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No ACS parquet files found in {self._raw_dir}. "
                "Run download() first."
            )

        frames: list[pd.DataFrame] = []
        for fp in parquet_files:
            raw_year = pd.read_parquet(fp)
            year = int(raw_year["YEAR"].iloc[0])
            frames.append(self._transform_year(raw_year, year))

        df = pd.concat(frames, ignore_index=True)
        return df

    def _transform_year(
        self, raw: pd.DataFrame, year: int
    ) -> pd.DataFrame:
        """Transform one year of raw ACS data into the standard schema."""
        # Convert all variable columns to numeric.
        all_vars = _variables_for_year(year)
        for col in all_vars:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")

        # Build FIPS codes.
        raw["state"] = raw["state"].astype(str).str.zfill(2)
        raw["county"] = raw["county"].astype(str).str.zfill(3)

        df = pd.DataFrame()
        df["county_fips"] = raw["state"].values + raw["county"].values
        df["state_fips"] = raw["state"].values
        df["year"] = year

        total_pop = raw["B01003_001E"]
        df["median_age"] = raw["B01002_001E"].values
        df["pct_male"] = _safe_pct(raw["B01001_002E"], total_pop)
        df["pct_white"] = _safe_pct(raw["B02001_002E"], total_pop)
        df["pct_black"] = _safe_pct(raw["B02001_003E"], total_pop)
        df["pct_hispanic"] = _safe_pct(raw["B03003_003E"], total_pop)
        df["pct_under_18"] = _safe_pct(raw["B09001_001E"], total_pop)

        over_65 = (
            raw[_MALE_65_PLUS].sum(axis=1) + raw[_FEMALE_65_PLUS].sum(axis=1)
        )
        df["pct_over_65"] = _safe_pct(over_65, total_pop)

        df["pct_foreign_born"] = _safe_pct(raw["B05002_013E"], total_pop)

        # Education: use B15003 for 2012+, B15002 for 2009-2011.
        if year >= _B15003_FIRST_YEAR:
            edu_denom = raw["B15003_001E"]
            hs_or_higher = raw[_HS_OR_HIGHER_B15003].sum(axis=1)
            bachelor_or_higher = raw[_BACHELOR_OR_HIGHER_B15003].sum(axis=1)
        else:
            edu_denom = raw["B15002_001E"]
            hs_or_higher = raw[_HS_OR_HIGHER_B15002].sum(axis=1)
            bachelor_or_higher = raw[_BACHELOR_OR_HIGHER_B15002].sum(axis=1)

        df["pct_hs_or_higher"] = _safe_pct(hs_or_higher, edu_denom)
        df["pct_bachelor_or_higher"] = _safe_pct(bachelor_or_higher, edu_denom)

        # Exclude state-level totals (county == "000").
        df = df[df["county_fips"].str[-3:] != "000"].copy()

        # Drop rows with missing total population (pct_male will be NaN).
        df = df.dropna(subset=["pct_male"]).copy()

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # validate
    # ------------------------------------------------------------------ #
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate ACS DataFrame.

        Checks
        ------
        - All expected columns are present.
        - No duplicate (county_fips, year) rows.
        - Percentage columns are between 0 and 100.

        Returns
        -------
        pd.DataFrame
            The validated DataFrame.
        """
        # --- column presence ---
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"ACS DataFrame is missing columns: {missing_cols}"
            )

        # --- duplicates ---
        dup_mask = df.duplicated(subset=["county_fips", "year"], keep="first")
        n_dups = dup_mask.sum()
        if n_dups > 0:
            logger.warning(
                "Dropping %d duplicate (county_fips, year) rows.", n_dups
            )
            df = df[~dup_mask].copy()

        # --- percentage range checks ---
        pct_cols = [c for c in EXPECTED_COLUMNS if c.startswith("pct_")]
        for col in pct_cols:
            bad = (df[col] < 0) | (df[col] > 100)
            if bad.any():
                logger.warning(
                    "Dropping %d rows with %s outside [0, 100].",
                    bad.sum(),
                    col,
                )
                df = df[~bad].copy()

        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # get_metadata
    # ------------------------------------------------------------------ #
    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for the ACS data source."""
        return {
            "source_name": "ACS 5-Year",
            "url": "https://www.census.gov/programs-surveys/acs",
            "api_endpoint": "https://api.census.gov/data/{year}/acs/acs5",
            "update_frequency": "Annual (5-year rolling window)",
            "geographic_coverage": "All U.S. counties",
            "time_coverage": (
                f"{max(self._cfg.start_year, _ACS_FIRST_YEAR)}"
                f"-{self._cfg.end_year}"
            ),
            "expected_columns": EXPECTED_COLUMNS,
            "note": (
                "Education variables use B15003 for 2012+, "
                "B15002 for 2009-2011."
            ),
        }


def _safe_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute percentage, returning NaN where denominator is zero or NaN."""
    return (numerator / denominator.replace(0, float("nan"))) * 100
