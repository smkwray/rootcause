"""Shared preprocessing helpers for panel-aware ML estimators."""

from __future__ import annotations

from typing import Literal

import pandas as pd

PanelMode = Literal["none", "two_way_within"]

_VALID_PANEL_MODES = {"none", "two_way_within"}


def validate_panel_mode(panel_mode: str) -> PanelMode:
    """Validate and normalize a panel-mode string."""
    if panel_mode not in _VALID_PANEL_MODES:
        raise ValueError(
            f"panel_mode must be one of {sorted(_VALID_PANEL_MODES)}; got {panel_mode!r}."
        )
    return panel_mode  # type: ignore[return-value]


def prepare_panel_ml_sample(
    df: pd.DataFrame,
    *,
    model_cols: list[str],
    keep_cols: list[str] | None = None,
    panel_mode: PanelMode = "none",
    entity_col: str = "county_fips",
    time_col: str = "year",
) -> pd.DataFrame:
    """Prepare a model sample with optional two-way panel residualization."""
    keep_cols = keep_cols or []
    required = list(dict.fromkeys([*model_cols, *keep_cols]))
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for panel ML sample: {missing}")

    sample = df[required].copy()
    for column in model_cols:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    sample = sample.dropna(subset=required).reset_index(drop=True)

    normalized_mode = validate_panel_mode(panel_mode)
    if normalized_mode == "two_way_within":
        sample = apply_two_way_within_transform(
            sample,
            columns=model_cols,
            entity_col=entity_col,
            time_col=time_col,
        )
    return sample


def apply_two_way_within_transform(
    df: pd.DataFrame,
    *,
    columns: list[str],
    entity_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Apply a two-way within transform to numeric columns."""
    missing = [col for col in [entity_col, time_col, *columns] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for two-way within transform: {missing}")

    transformed = df.copy()
    grand_mean = transformed[columns].mean()
    entity_mean = transformed.groupby(entity_col)[columns].transform("mean")
    time_mean = transformed.groupby(time_col)[columns].transform("mean")
    transformed.loc[:, columns] = transformed[columns] - entity_mean - time_mean + grand_mean
    return transformed
