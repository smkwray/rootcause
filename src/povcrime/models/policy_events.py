"""Helpers for treatment event timing in policy panels."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_first_treatment_event_year(
    panel: pd.DataFrame,
    *,
    treatment_col: str,
    output_col: str,
    entity_col: str = "county_fips",
    time_col: str = "year",
    change_threshold: float = 0.01,
) -> pd.DataFrame:
    """Compute the first year a treatment rises above its lagged value."""
    panel = panel.sort_values([entity_col, time_col]).copy()
    lag_col = f"_{treatment_col}_lag"
    change_col = f"_{treatment_col}_change"
    panel[lag_col] = panel.groupby(entity_col)[treatment_col].shift(1)
    panel[change_col] = panel[treatment_col] - panel[lag_col]

    increases = panel.loc[panel[change_col] > change_threshold, [entity_col, time_col]]
    first_event = (
        increases.groupby(entity_col)[time_col]
        .min()
        .rename(output_col)
    )

    panel = panel.merge(first_event, on=entity_col, how="left")
    panel.drop(columns=[lag_col, change_col], inplace=True)

    logger.info(
        "Event timing for %s: %d entities with at least one qualifying increase.",
        treatment_col,
        first_event.shape[0],
    )
    return panel
