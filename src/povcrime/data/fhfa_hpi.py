"""FHFA county house price index adapter."""

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
    "county_fips",
    "state_fips",
    "year",
    "fhfa_hpi",
    "fhfa_hpi_1990_base",
    "fhfa_hpi_2000_base",
    "fhfa_annual_change_pct",
]

_HPI_URL = "https://www.fhfa.gov/hpi/download/annual/hpi_at_county.xlsx"
_WORKBOOK_NAME = "fhfa_hpi_at_county.xlsx"
_HEADERS = {"User-Agent": "Mozilla/5.0 (povcrime research)"}


class FHFAHPIAdapter(BaseAdapter):
    """Adapter for annual county-level FHFA house price indexes."""

    def __init__(self, config: ProjectConfig) -> None:
        self._cfg = config
        self._raw_dir: Path = config.raw_dir / "fhfa_hpi"

    def download(self) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._raw_dir / _WORKBOOK_NAME
        if out_path.exists():
            logger.info("FHFA county HPI workbook already downloaded, skipping.")
            return

        resp = requests.get(_HPI_URL, timeout=120, headers=_HEADERS)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        logger.info("Saved FHFA county HPI workbook -> %s", out_path)

    def load(self) -> pd.DataFrame:
        xlsx_path = self._raw_dir / _WORKBOOK_NAME
        if not xlsx_path.exists():
            raise FileNotFoundError(f"FHFA county HPI workbook not found: {xlsx_path}")

        raw = pd.read_excel(xlsx_path, sheet_name="county", skiprows=5)
        df = _reshape_fhfa_hpi(raw)
        df = df[
            (df["year"] >= self._cfg.start_year)
            & (df["year"] <= self._cfg.end_year)
        ].reset_index(drop=True)
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"FHFA HPI DataFrame is missing columns: {missing}")

        dup_mask = df.duplicated(subset=["county_fips", "year"], keep=False)
        if dup_mask.any():
            raise ValueError(
                f"Found {int(dup_mask.sum())} duplicate FHFA HPI county-year rows."
            )

        if ((df["fhfa_annual_change_pct"] < -100) | (df["fhfa_annual_change_pct"] > 1000)).any():
            raise ValueError("fhfa_annual_change_pct is outside a plausible range.")

        for col in ["fhfa_hpi", "fhfa_hpi_1990_base", "fhfa_hpi_2000_base"]:
            if (pd.to_numeric(df[col], errors="coerce").dropna() < 0).any():
                raise ValueError(f"FHFA HPI column {col} contains negative values.")

        return df.sort_values(["county_fips", "year"]).reset_index(drop=True)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "source_name": "FHFA County HPI",
            "url": "https://www.fhfa.gov/data/hpi/datasets",
            "data_level": "county",
            "update_frequency": "annual",
            "time_coverage": "1986-present",
            "expected_columns": EXPECTED_COLUMNS,
            "notes": (
                "Annual FHFA all-transactions county house price indexes. "
                "These are developmental county series and nominal, not inflation-adjusted."
            ),
        }


def _reshape_fhfa_hpi(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(
        columns={
            "FIPS code": "county_fips",
            "Year": "year",
            "Annual Change (%)": "fhfa_annual_change_pct",
            "HPI": "fhfa_hpi",
            "HPI with 1990 base": "fhfa_hpi_1990_base",
            "HPI with 2000 base": "fhfa_hpi_2000_base",
        }
    ).copy()

    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["county_fips"].str[:2]
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.loc[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    for col in EXPECTED_COLUMNS[3:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[EXPECTED_COLUMNS].reset_index(drop=True)
