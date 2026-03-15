"""Helpers for robustness specifications built on top of the FE pipeline."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def build_placebo_treatment(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    entity_col: str,
    time_col: str,
    lead_periods: int,
    output_col: str,
) -> pd.DataFrame:
    """Add a future-treatment placebo variable within entity."""
    out = df.sort_values([entity_col, time_col]).copy()
    out[output_col] = out.groupby(entity_col)[treatment_col].shift(-lead_periods)
    return out


def detrend_variables_within_entity(
    df: pd.DataFrame,
    *,
    columns: Iterable[str],
    entity_col: str,
    time_col: str,
    prefix: str = "dt_",
) -> pd.DataFrame:
    """Residualize variables against an entity-specific linear time trend."""
    out = df.copy()
    years = out[time_col].astype(float)

    for col in columns:
        detrended = pd.Series(np.nan, index=out.index, dtype=float)
        for _, idx in out.groupby(entity_col).groups.items():
            sub_idx = pd.Index(idx)
            y = pd.to_numeric(out.loc[sub_idx, col], errors="coerce")
            x = years.loc[sub_idx]
            valid = y.notna() & x.notna()

            if valid.sum() == 0:
                continue

            if valid.sum() == 1:
                detrended.loc[sub_idx[valid]] = y.loc[valid] - y.loc[valid].mean()
                continue

            coeffs = np.polyfit(x.loc[valid], y.loc[valid], deg=1)
            fitted = coeffs[0] * x.loc[valid] + coeffs[1]
            detrended.loc[sub_idx[valid]] = y.loc[valid] - fitted

        out[f"{prefix}{col}"] = detrended

    return out


def extract_treatment_row(
    summary_df: pd.DataFrame,
    *,
    treatment: str,
) -> dict[str, float]:
    """Return the fitted coefficient row for the treatment variable."""
    row = summary_df.loc[summary_df["variable"] == treatment]
    if row.empty:
        raise ValueError(f"Treatment '{treatment}' not found in summary table.")
    return row.iloc[0].to_dict()
