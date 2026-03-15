"""HUD Fair Market Rent (FMR) adapter."""

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
    "fair_market_rent_0br",
    "fair_market_rent_1br",
    "fair_market_rent_2br",
    "fair_market_rent_3br",
    "fair_market_rent_4br",
]
_RENT_COLUMNS = EXPECTED_COLUMNS[3:]

_FMR_HISTORY_URL = "https://www.huduser.gov/portal/datasets/FMR/FMR_All_1983_2026.csv"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}


class HUDFMRAdapter(BaseAdapter):
    """Adapter for HUD county-level Fair Market Rent history."""

    def __init__(self, config: ProjectConfig) -> None:
        self._cfg = config
        self._raw_dir: Path = config.raw_dir / "hud_fmr"

    def download(self) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._raw_dir / "fmr_all_1983_2026.csv"
        if out_path.exists():
            logger.info("HUD FMR history already downloaded, skipping.")
            return

        resp = requests.get(_FMR_HISTORY_URL, timeout=120, headers=_HEADERS)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        logger.info("Saved HUD FMR history -> %s", out_path)

    def load(self) -> pd.DataFrame:
        csv_path = self._raw_dir / "fmr_all_1983_2026.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"HUD FMR raw file not found: {csv_path}")

        try:
            raw = pd.read_csv(csv_path, low_memory=False)
        except UnicodeDecodeError:
            logger.warning(
                "HUD FMR history is not clean UTF-8; retrying with latin-1 decoding."
            )
            raw = pd.read_csv(csv_path, low_memory=False, encoding="latin-1")
        df = _reshape_fmr_history(raw)
        df = df[
            (df["year"] >= self._cfg.start_year)
            & (df["year"] <= self._cfg.end_year)
        ].reset_index(drop=True)
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"HUD FMR DataFrame is missing columns: {missing}")

        dup_mask = df.duplicated(subset=["county_fips", "year"], keep="first")
        if dup_mask.any():
            logger.warning(
                "Dropping %d duplicate HUD FMR county-year rows.",
                int(dup_mask.sum()),
            )
            df = df.loc[~dup_mask].copy()

        for col in EXPECTED_COLUMNS[3:]:
            negative = df[col].dropna() < 0
            if negative.any():
                logger.warning("Coercing negative HUD FMR values in %s to missing.", col)
                bad_idx = negative[negative].index
                df.loc[bad_idx, col] = pd.NA

        return df.reset_index(drop=True)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "source_name": "HUD Fair Market Rent History",
            "url": _FMR_HISTORY_URL,
            "data_level": "county",
            "update_frequency": "annual",
            "time_coverage": "1983-present",
            "expected_columns": EXPECTED_COLUMNS,
            "notes": (
                "County-level FMR history from HUD User. Values reflect nominal "
                "fair market rents by bedroom size."
            ),
        }


def _reshape_fmr_history(raw: pd.DataFrame) -> pd.DataFrame:
    base = raw.copy()
    base["county_fips"] = pd.to_numeric(base["fips"] // 100000, errors="coerce").astype("Int64")
    base = base.loc[base["county_fips"].notna()].copy()
    base["county_fips"] = base["county_fips"].astype(int).astype(str).str.zfill(5)
    base["state_fips"] = base["county_fips"].str[:2]
    base["is_county_row"] = base["cousub"].astype(str).eq("99999") if "cousub" in base.columns else True

    rows: list[pd.DataFrame] = []
    for year in range(2000, 2027):
        yy = str(year)[2:]
        cols = {
            f"fmr{yy}_0": "fair_market_rent_0br",
            f"fmr{yy}_1": "fair_market_rent_1br",
            f"fmr{yy}_2": "fair_market_rent_2br",
            f"fmr{yy}_3": "fair_market_rent_3br",
            f"fmr{yy}_4": "fair_market_rent_4br",
        }
        if not set(cols).issubset(base.columns):
            continue
        piece = base[["county_fips", "state_fips", "is_county_row", *cols.keys()]].copy()
        piece = piece.rename(columns=cols)
        piece["year"] = year
        for col in cols.values():
            piece[col] = pd.to_numeric(
                piece[col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce",
            )
        rows.append(piece)

    if not rows:
        raise ValueError("No HUD FMR year columns could be parsed.")

    df = pd.concat(rows, ignore_index=True)
    county_level = df.loc[df["is_county_row"]].copy()
    county_level = county_level.drop_duplicates(subset=["county_fips", "year"], keep="first")

    subcounty = df.loc[~df["is_county_row"]].copy()
    if subcounty.empty:
        collapsed = county_level
    else:
        missing_keys = (
            subcounty[["county_fips", "year"]]
            .merge(
                county_level[["county_fips", "year"]],
                on=["county_fips", "year"],
                how="left",
                indicator=True,
            )
            .loc[lambda x: x["_merge"] == "left_only", ["county_fips", "year"]]
            .drop_duplicates()
        )
        subcounty = subcounty.merge(missing_keys, on=["county_fips", "year"], how="inner")
        subcounty = (
            subcounty.groupby(["county_fips", "state_fips", "year"], as_index=False)[_RENT_COLUMNS]
            .median()
        )
        collapsed = pd.concat(
            [county_level[["county_fips", "state_fips", "year", *_RENT_COLUMNS]], subcounty],
            ignore_index=True,
        )

    return collapsed[EXPECTED_COLUMNS].sort_values(["county_fips", "year"]).reset_index(drop=True)
