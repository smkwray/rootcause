"""UKCPR National Welfare Data adapter.

Source
------
University of Kentucky Center for Poverty Research (UKCPR),
National Welfare Data, 1980-2023.
https://ukcpr.org/resources/national-welfare-data

The public workbook contains annual state-level economic, transfer, and
policy variables. This adapter extracts the EITC and TANF variables most
relevant to the current project.
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

EXPECTED_COLUMNS: list[str] = [
    "state_fips",
    "year",
    "state_eitc_rate",
    "state_eitc_refundable",
    "tanf_benefit_2_person",
    "tanf_benefit_3_person",
    "tanf_benefit_4_person",
]

_WORKBOOK_URL = (
    "https://ukcpr.uky.edu/sites/default/files/2025-09/"
    "ukcpr_national_welfare_data_1980_2023_july25update.xlsx"
)
_WORKBOOK_NAME = "ukcpr_national_welfare_data_1980_2023.xlsx"
_SHEET_NAME = "Data"
_HEADERS = {"User-Agent": "Mozilla/5.0 (povcrime research)"}
_COLUMN_MAP = {
    "State EITC Rate": "state_eitc_rate",
    "Refundable State EITC (1=Yes)": "state_eitc_refundable",
    "AFDC/TANF Benefit for 2-Person family": "tanf_benefit_2_person",
    "AFDC/TANF Benefit for 3-person family": "tanf_benefit_3_person",
    "AFDC/TANF benefit for 4-person family": "tanf_benefit_4_person",
}


class UKCPRWelfareAdapter(BaseAdapter):
    """Adapter for UKCPR National Welfare Data."""

    def __init__(self, config: ProjectConfig) -> None:
        self._cfg = config
        self._raw_dir: Path = config.raw_dir / "ukcpr_welfare"

    def download(self) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._raw_dir / _WORKBOOK_NAME
        if out_path.exists():
            logger.info("UKCPR welfare workbook already downloaded, skipping.")
            return

        resp = requests.get(_WORKBOOK_URL, timeout=120, headers=_HEADERS)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        logger.info("Saved UKCPR welfare workbook -> %s", out_path)

    def load(self) -> pd.DataFrame:
        xlsx_path = self._raw_dir / _WORKBOOK_NAME
        if not xlsx_path.exists():
            raise FileNotFoundError(
                f"UKCPR welfare workbook not found: {xlsx_path}. Run download() first."
            )

        raw = pd.read_excel(xlsx_path, sheet_name=_SHEET_NAME)
        df = _reshape_ukcpr_welfare(raw)
        df = df[
            (df["year"] >= self._cfg.start_year)
            & (df["year"] <= self._cfg.end_year)
        ].reset_index(drop=True)
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"UKCPR welfare DataFrame is missing columns: {missing}")

        dupes = df.duplicated(subset=["state_fips", "year"], keep=False)
        if dupes.any():
            raise ValueError(
                f"Found {int(dupes.sum())} duplicate UKCPR welfare state-year rows."
            )

        if ((df["state_eitc_rate"] < 0) | (df["state_eitc_rate"] > 5)).any():
            raise ValueError("state_eitc_rate must be non-negative and plausibly bounded.")

        bad_refundable = ~df["state_eitc_refundable"].isin([0, 1])
        if bad_refundable.any():
            raise ValueError("state_eitc_refundable must be binary (0/1).")

        for col in EXPECTED_COLUMNS[4:]:
            if (df[col] < 0).any():
                raise ValueError(f"{col} contains negative values.")

        return df.sort_values(["state_fips", "year"]).reset_index(drop=True)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "source_name": "UKCPR National Welfare Data",
            "url": "https://ukcpr.org/resources/national-welfare-data",
            "data_level": "state",
            "update_frequency": "annual",
            "time_coverage": "1980-2023",
            "expected_columns": EXPECTED_COLUMNS,
            "notes": (
                "Extracts state EITC generosity/refundability and AFDC/TANF "
                "cash benefit levels from the UKCPR state welfare panel."
            ),
        }


def _reshape_ukcpr_welfare(raw: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["state_fips", "year", *_COLUMN_MAP.keys()]
    df = raw[keep_cols].rename(columns=_COLUMN_MAP).copy()
    df["state_fips"] = pd.to_numeric(df["state_fips"], errors="coerce").astype("Int64")
    df = df.loc[df["state_fips"].notna()].copy()
    df["state_fips"] = df["state_fips"].astype(int).astype(str).str.zfill(2)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)

    for col in EXPECTED_COLUMNS[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["state_eitc_rate"] = df["state_eitc_rate"].fillna(0.0)
    df["state_eitc_refundable"] = df["state_eitc_refundable"].fillna(0).astype(int)

    return df[EXPECTED_COLUMNS].sort_values(["state_fips", "year"]).reset_index(drop=True)
