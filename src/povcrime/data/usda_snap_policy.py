"""USDA SNAP Policy Database adapter.

Source
------
USDA Economic Research Service, SNAP Policy Database.
https://www.ers.usda.gov/data-products/snap-policy-data-sets/

The SNAP Policy Database tracks state-level policy choices for the
Supplemental Nutrition Assistance Program (SNAP), including eligibility
rules, benefit calculations, and administrative procedures.

Update frequency: Periodic (roughly annual updates).
Geographic coverage: All 50 states + DC (state-level).
Time coverage: 1996-2020 (as of the 2024 release).

Output Schema
-------------
state_fips                  : str  -- 2-digit state FIPS code (zero-padded)
year                        : int  -- Calendar year
broad_based_cat_elig        : int  -- 1 if state uses broad-based categorical
                                      eligibility, else 0
simplified_reporting        : int  -- 1 if state has simplified (semi-annual)
                                      reporting, else 0
vehicle_exemption           : int  -- 1 if state exempts vehicle value from
                                      asset test, else 0
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from povcrime.config import ProjectConfig
from povcrime.data.base import BaseAdapter

logger = logging.getLogger(__name__)

# Expected output columns.
EXPECTED_COLUMNS: list[str] = [
    "state_fips",
    "year",
    "broad_based_cat_elig",
    "simplified_reporting",
    "vehicle_exemption",
]

# Download URL for the SNAP Policy Database Excel workbook.
_SNAP_XLSX_URL = (
    "https://www.ers.usda.gov/media/6472/snap-policy-database.xlsx"
)

# The Excel sheet containing the policy data.
_SNAP_SHEET_NAME = "SNAP Policy Database"

# Mapping from SNAP database column names to our output column names.
# The source columns are monthly binary indicators (0/1).
_COLUMN_MAP: dict[str, str] = {
    "bbce": "broad_based_cat_elig",
    "reportsimple": "simplified_reporting",
    "vehexclall": "vehicle_exemption",
}


class USDASnapPolicyAdapter(BaseAdapter):
    """Adapter for USDA ERS SNAP Policy Database.

    The database is distributed as an Excel workbook with monthly
    observations for each state.  This adapter downloads the file,
    aggregates monthly data to annual level (a policy is coded as 1
    for a given year if it was active for the *majority* of months
    in that year), and outputs a clean state-year panel.

    If the download fails, the adapter can read a manually-placed
    Excel file at ``data/raw/usda_snap_policy/snap_policy_database.xlsx``.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self._config = config
        self._raw_dir = config.raw_dir / "usda_snap_policy"

    # ------------------------------------------------------------------ #
    # download
    # ------------------------------------------------------------------ #
    def download(self) -> None:
        """Download the SNAP Policy Database from USDA ERS.

        Downloads the Excel workbook and saves it to
        ``data/raw/usda_snap_policy/snap_policy_database.xlsx``.

        If the download fails, logs a warning with instructions for
        manual placement of the file.
        """
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._raw_dir / "snap_policy_database.xlsx"

        try:
            logger.info("Downloading SNAP Policy Database from %s", _SNAP_XLSX_URL)
            resp = requests.get(
                _SNAP_XLSX_URL,
                timeout=120,
                headers={"User-Agent": "Mozilla/5.0 (povcrime research)"},
            )
            resp.raise_for_status()

            out_path.write_bytes(resp.content)
            logger.info(
                "Saved SNAP Policy Database (%d bytes) to %s",
                len(resp.content),
                out_path,
            )
        except requests.RequestException as exc:
            logger.warning(
                "Failed to download SNAP Policy Database: %s. "
                "Manually download the file from %s and place it at %s",
                exc,
                "https://www.ers.usda.gov/data-products/snap-policy-data-sets/",
                out_path,
            )

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    def load(self) -> pd.DataFrame:
        """Load SNAP policy data into a standardised DataFrame.

        Reads the Excel workbook, extracts the monthly policy indicators,
        aggregates to annual level using a majority-of-months rule, and
        maps state FIPS codes.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns defined in ``EXPECTED_COLUMNS``.
            Policy feature columns are binary (0/1).

        Raises
        ------
        FileNotFoundError
            If the Excel workbook is not found.
        """
        xlsx_path = self._raw_dir / "snap_policy_database.xlsx"
        if not xlsx_path.exists():
            msg = (
                f"SNAP Policy Database not found at {xlsx_path}. "
                "Run download() first or manually place the Excel file. "
                "Download from: https://www.ers.usda.gov/data-products/"
                "snap-policy-data-sets/"
            )
            raise FileNotFoundError(msg)

        logger.info("Reading SNAP Policy Database from %s", xlsx_path)
        raw = pd.read_excel(xlsx_path, sheet_name=_SNAP_SHEET_NAME)

        # Extract year from yearmonth (format: YYYYMM, e.g. 200401).
        raw["year"] = raw["yearmonth"] // 100

        # Zero-pad state FIPS to 2 digits.
        raw["state_fips"] = raw["state_fips"].astype(str).str.zfill(2)

        # Filter to configured year range.
        raw = raw[
            (raw["year"] >= self._config.start_year)
            & (raw["year"] <= self._config.end_year)
        ].copy()

        # Select and rename policy columns.
        keep_cols = ["state_fips", "year"] + list(_COLUMN_MAP.keys())
        df = raw[keep_cols].rename(columns=_COLUMN_MAP)

        # Replace missing / sentinel values (-9 etc.) with NaN, then fill
        # with 0 (conservative: missing = policy not in effect).
        policy_cols = list(_COLUMN_MAP.values())
        for col in policy_cols:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = float("nan")

        # Aggregate monthly -> annual using majority-of-months rule:
        # policy = 1 for the year if it was active > 50% of observed months.
        annual = (
            df.groupby(["state_fips", "year"], as_index=False)[policy_cols]
            .mean()
        )
        for col in policy_cols:
            annual[col] = (annual[col] >= 0.5).astype(int)

        # Select and order columns.
        annual = (
            annual[EXPECTED_COLUMNS]
            .sort_values(["state_fips", "year"])
            .reset_index(drop=True)
        )

        logger.info("Loaded SNAP policy data: %d rows", len(annual))
        return annual

    # ------------------------------------------------------------------ #
    # validate
    # ------------------------------------------------------------------ #
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate SNAP policy DataFrame.

        Checks
        ------
        - All expected columns are present.
        - No duplicate (state_fips, year) rows.
        - Policy feature columns contain only 0 or 1.

        Returns
        -------
        pd.DataFrame
            The validated DataFrame.

        Raises
        ------
        ValueError
            If any validation check fails.
        """
        # Column check.
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            msg = f"Missing columns: {missing}"
            raise ValueError(msg)

        # Duplicate check.
        dupes = df.duplicated(subset=["state_fips", "year"], keep=False)
        if dupes.any():
            n_dupes = dupes.sum()
            msg = f"Found {n_dupes} duplicate (state_fips, year) rows"
            raise ValueError(msg)

        # Binary column check.
        policy_cols = list(_COLUMN_MAP.values())
        for col in policy_cols:
            bad = ~df[col].isin([0, 1])
            if bad.any():
                n_bad = bad.sum()
                msg = f"{n_bad} rows in '{col}' are not binary (0/1)"
                raise ValueError(msg)

        logger.info("Validation passed: %d rows", len(df))
        return df

    # ------------------------------------------------------------------ #
    # get_metadata
    # ------------------------------------------------------------------ #
    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for the SNAP Policy Database."""
        return {
            "source": "USDA Economic Research Service",
            "url": (
                "https://www.ers.usda.gov/data-products/"
                "snap-policy-data-sets/"
            ),
            "description": (
                "SNAP Policy Database tracking state-level SNAP policy "
                "choices including broad-based categorical eligibility, "
                "simplified reporting, and vehicle exemptions."
            ),
            "geographic_coverage": "All 50 states + DC (state-level)",
            "time_coverage": "1996-2020",
            "update_frequency": "Periodic (roughly annual)",
            "output_columns": EXPECTED_COLUMNS,
            "aggregation_method": (
                "Monthly observations aggregated to annual using "
                "majority-of-months rule (policy = 1 if active > 50% "
                "of observed months in a given year)."
            ),
        }
