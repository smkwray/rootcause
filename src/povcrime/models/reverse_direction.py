"""Helpers for reverse-direction and bidirectional exploratory specifications."""

from __future__ import annotations

import pandas as pd


def shift_variable_within_entity(
    panel: pd.DataFrame,
    *,
    variable_col: str,
    periods: int,
    entity_col: str = "county_fips",
    time_col: str = "year",
) -> pd.DataFrame:
    """Shift a variable within entity and write the shifted column."""
    out = panel.sort_values([entity_col, time_col]).copy()
    suffix = "lag" if periods > 0 else "lead"
    out[f"{variable_col}_{suffix}{abs(periods)}"] = (
        out.groupby(entity_col)[variable_col].shift(periods)
    )
    return out


def lag_treatment_within_county(
    panel: pd.DataFrame,
    *,
    treatment_col: str,
    entity_col: str = "county_fips",
    time_col: str = "year",
    periods: int = 1,
) -> pd.DataFrame:
    """Lag a treatment within county for exploratory reverse-direction specs."""
    return shift_variable_within_entity(
        panel,
        variable_col=treatment_col,
        periods=periods,
        entity_col=entity_col,
        time_col=time_col,
    )


def lead_treatment_within_county(
    panel: pd.DataFrame,
    *,
    treatment_col: str,
    entity_col: str = "county_fips",
    time_col: str = "year",
    periods: int = 1,
) -> pd.DataFrame:
    """Lead a treatment within county for placebo-style exploratory specs."""
    return shift_variable_within_entity(
        panel,
        variable_col=treatment_col,
        periods=-periods,
        entity_col=entity_col,
        time_col=time_col,
    )
