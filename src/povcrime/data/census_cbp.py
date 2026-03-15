"""Census County Business Patterns adapter."""

from __future__ import annotations

import json
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
    "cbp_establishments",
    "cbp_employment",
    "cbp_annual_payroll",
]

_CBP_URL = "https://api.census.gov/data/{year}/cbp"
_INDUSTRY_PREDICATE_BY_YEAR = {
    **{year: ("NAICS1997", "00") for year in range(2000, 2003)},
    **{year: ("NAICS2002", "00") for year in range(2003, 2008)},
    **{year: ("NAICS2007", "00") for year in range(2008, 2012)},
    **{year: ("NAICS2012", "00") for year in range(2012, 2017)},
    **{year: ("NAICS2017", "00") for year in range(2017, 2024)},
}


class CensusCBPAdapter(BaseAdapter):
    """Adapter for county-level County Business Patterns totals."""

    def __init__(self, config: ProjectConfig) -> None:
        self._cfg = config
        self._raw_dir: Path = config.raw_dir / "census_cbp"

    def download(self) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        for year in range(self._cfg.start_year, self._cfg.end_year + 1):
            if year not in _INDUSTRY_PREDICATE_BY_YEAR:
                logger.warning("CBP year %d is not supported by the adapter, skipping.", year)
                continue

            out_path = self._raw_dir / f"cbp_{year}.json"
            if out_path.exists():
                logger.info("CBP %d already downloaded, skipping.", year)
                continue

            predicate_name, predicate_value = _INDUSTRY_PREDICATE_BY_YEAR[year]
            params = {
                "get": "ESTAB,EMP,PAYANN",
                "for": "county:*",
                "in": "state:*",
                predicate_name: predicate_value,
            }
            resp = requests.get(_CBP_URL.format(year=year), params=params, timeout=120)
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                if resp.status_code == 404:
                    logger.warning(
                        "CBP %d is not published at %s yet, skipping.",
                        year,
                        resp.url,
                    )
                    continue
                raise exc
            out_path.write_text(resp.text, encoding="utf-8")
            logger.info("Saved CBP %d -> %s", year, out_path)

    def load(self) -> pd.DataFrame:
        json_files = sorted(self._raw_dir.glob("cbp_*.json"))
        if not json_files:
            raise FileNotFoundError(f"No CBP JSON files found in {self._raw_dir}.")

        frames = [_parse_cbp_json(fp) for fp in json_files]
        df = pd.concat(frames, ignore_index=True)
        df = df[
            (df["year"] >= self._cfg.start_year)
            & (df["year"] <= self._cfg.end_year)
        ].reset_index(drop=True)
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"CBP DataFrame is missing columns: {missing}")

        dup_mask = df.duplicated(subset=["county_fips", "year"], keep="first")
        if dup_mask.any():
            logger.warning("Dropping %d duplicate CBP county-year rows.", int(dup_mask.sum()))
            df = df.loc[~dup_mask].copy()

        for col in EXPECTED_COLUMNS[3:]:
            negative = df[col].dropna() < 0
            if negative.any():
                raise ValueError(f"CBP column {col} contains negative values.")

        return df.reset_index(drop=True)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "source_name": "Census County Business Patterns",
            "url": "https://www.census.gov/programs-surveys/cbp.html",
            "data_level": "county",
            "update_frequency": "annual",
            "time_coverage": f"{self._cfg.start_year}-{self._cfg.end_year}",
            "expected_columns": EXPECTED_COLUMNS,
            "notes": (
                "County totals for all industries from the Census CBP API. "
                "Annual payroll is in thousands of dollars."
            ),
        }


def _parse_cbp_json(path: Path) -> pd.DataFrame:
    year = int(path.stem.split("_")[-1])
    rows = json.loads(path.read_text(encoding="utf-8"))
    raw = pd.DataFrame(rows[1:], columns=rows[0])
    df = pd.DataFrame(
        {
            "state_fips": raw["state"].astype(str).str.zfill(2),
            "county_fips": raw["state"].astype(str).str.zfill(2) + raw["county"].astype(str).str.zfill(3),
            "year": year,
            "cbp_establishments": pd.to_numeric(raw["ESTAB"], errors="coerce"),
            "cbp_employment": pd.to_numeric(raw["EMP"], errors="coerce"),
            "cbp_annual_payroll": pd.to_numeric(raw["PAYANN"], errors="coerce"),
        }
    )
    return df[EXPECTED_COLUMNS].reset_index(drop=True)
