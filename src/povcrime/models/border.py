"""Helpers for cross-state border-county designs."""

from __future__ import annotations

import pandas as pd


def canonical_cross_state_pairs(adjacency: pd.DataFrame) -> pd.DataFrame:
    """Return deduplicated adjacent county pairs that cross state lines."""
    pairs = adjacency.loc[
        adjacency["state_fips"] != adjacency["neighbor_state_fips"],
        ["county_fips", "neighbor_county_fips", "state_fips", "neighbor_state_fips"],
    ].copy()
    if pairs.empty:
        return pd.DataFrame(
            columns=["county_a", "county_b", "state_a", "state_b", "pair_id", "border_state_pair"]
        )

    pairs[["county_a", "county_b"]] = pd.DataFrame(
        pairs.apply(
            lambda row: sorted([row["county_fips"], row["neighbor_county_fips"]]),
            axis=1,
        ).tolist(),
        index=pairs.index,
    )
    pairs["state_a"] = pairs["county_a"].str[:2]
    pairs["state_b"] = pairs["county_b"].str[:2]
    pairs["pair_id"] = pairs["county_a"] + "__" + pairs["county_b"]
    pairs["border_state_pair"] = pairs["state_a"] + "__" + pairs["state_b"]
    return pairs[
        ["county_a", "county_b", "state_a", "state_b", "pair_id", "border_state_pair"]
    ].drop_duplicates().reset_index(drop=True)


def build_border_pair_panel(
    *,
    panel: pd.DataFrame,
    pairs: pd.DataFrame,
    treatment: str,
    outcome: str,
    controls: list[str],
) -> pd.DataFrame:
    """Construct a border-pair/year differenced panel."""
    left_cols = ["county_fips", "year", treatment, outcome, *controls]
    right_cols = ["county_fips", "year", treatment, outcome, *controls]

    left = pairs.merge(
        panel[left_cols],
        left_on="county_a",
        right_on="county_fips",
        how="inner",
    ).drop(columns=["county_fips"])
    left = left.rename(
        columns={
            treatment: f"{treatment}_a",
            outcome: f"{outcome}_a",
            **{col: f"{col}_a" for col in controls},
        }
    )

    right = left.merge(
        panel[right_cols],
        left_on=["county_b", "year"],
        right_on=["county_fips", "year"],
        how="inner",
    ).drop(columns=["county_fips"])
    right = right.rename(
        columns={
            treatment: f"{treatment}_b",
            outcome: f"{outcome}_b",
            **{col: f"{col}_b" for col in controls},
        }
    )

    out = right.copy()
    out[f"diff_{treatment}"] = out[f"{treatment}_a"] - out[f"{treatment}_b"]
    out[f"diff_{outcome}"] = out[f"{outcome}_a"] - out[f"{outcome}_b"]
    for col in controls:
        out[f"diff_{col}"] = out[f"{col}_a"] - out[f"{col}_b"]
    out["state_fips"] = out["border_state_pair"]
    return out


def build_first_difference_border_panel(
    *,
    pair_panel: pd.DataFrame,
    treatment: str,
    outcome: str,
    controls: list[str],
    entity_col: str = "pair_id",
    time_col: str = "year",
    shock_threshold: float = 0.01,
) -> pd.DataFrame:
    """Construct a within-pair first-difference panel for border identification."""
    out = pair_panel.sort_values([entity_col, time_col]).copy()

    diff_treatment = f"diff_{treatment}"
    diff_outcome = f"diff_{outcome}"
    diff_controls = [f"diff_{col}" for col in controls]

    out[f"delta_{diff_treatment}"] = out.groupby(entity_col)[diff_treatment].diff()
    out[f"delta_{diff_outcome}"] = out.groupby(entity_col)[diff_outcome].diff()
    for col in diff_controls:
        out[f"delta_{col}"] = out.groupby(entity_col)[col].diff()

    a_change = out.groupby(entity_col)[f"{treatment}_a"].diff()
    b_change = out.groupby(entity_col)[f"{treatment}_b"].diff()
    a_changed = a_change.abs() > shock_threshold
    b_changed = b_change.abs() > shock_threshold
    out["one_sided_treatment_shock"] = (a_changed ^ b_changed) & (
        out[f"delta_{diff_treatment}"].abs() > shock_threshold
    )
    return out
