"""Tests for overlap/support diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from povcrime.models import overlap as overlap_module
from povcrime.models.overlap import (
    build_continuous_treatment_support_diagnostics,
    compute_out_of_fold_predictions,
)


def test_build_continuous_treatment_support_diagnostics_writes_outputs(tmp_path: Path):
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "effective_min_wage": rng.normal(10, 1.5, 200),
            "poverty_rate": rng.normal(15, 3, 200),
            "unemployment_rate": rng.normal(6, 1, 200),
            "log_population": rng.normal(10, 0.8, 200),
        }
    )

    out_dir = tmp_path / "overlap"
    summary = build_continuous_treatment_support_diagnostics(
        df=df,
        treatment="effective_min_wage",
        controls=["poverty_rate", "unemployment_rate", "log_population"],
        output_dir=out_dir,
    )

    assert summary["n_obs"] == 200
    assert (out_dir / "support_summary.json").exists()
    assert (out_dir / "control_balance.csv").exists()
    assert (out_dir / "support_bins.csv").exists()
    saved = json.loads((out_dir / "support_summary.json").read_text(encoding="utf-8"))
    assert saved["treatment"] == "effective_min_wage"


def test_compute_out_of_fold_predictions_groups_by_county(monkeypatch):
    df = pd.DataFrame(
        {
            "county_fips": ["01001", "01001", "01003", "01003", "01005", "01005"],
            "poverty_rate": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "unemployment_rate": [4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
        }
    )
    y = pd.Series([1.0, 1.3, 2.0, 2.2, 3.1, 3.4])
    seen: dict[str, list[str] | None] = {"groups": None}

    class FakeGroupKFold:
        def __init__(self, n_splits: int) -> None:
            self.n_splits = n_splits

        def split(self, X, y=None, groups=None):
            seen["groups"] = None if groups is None else list(groups)
            groups_array = np.asarray(groups)
            for group in pd.Index(groups_array).unique():
                test_idx = np.flatnonzero(groups_array == group)
                train_idx = np.flatnonzero(groups_array != group)
                yield train_idx, test_idx

    monkeypatch.setattr(overlap_module, "GroupKFold", FakeGroupKFold)

    preds = compute_out_of_fold_predictions(
        df,
        y,
        n_splits=3,
        group_col="county_fips",
    )

    assert preds.shape == (len(df),)
    assert seen["groups"] == df["county_fips"].tolist()


def test_build_continuous_treatment_support_diagnostics_supports_grouped_oof(tmp_path: Path):
    df = pd.DataFrame(
        {
            "county_fips": ["01001"] * 20 + ["01003"] * 20 + ["01005"] * 20,
            "effective_min_wage": np.linspace(5.0, 10.0, 60),
            "poverty_rate": np.linspace(10.0, 20.0, 60),
            "unemployment_rate": np.linspace(4.0, 7.0, 60),
            "log_population": np.linspace(8.0, 10.0, 60),
        }
    )

    out_dir = tmp_path / "overlap_grouped"
    summary = build_continuous_treatment_support_diagnostics(
        df=df,
        treatment="effective_min_wage",
        controls=["poverty_rate", "unemployment_rate", "log_population"],
        output_dir=out_dir,
        group_col="county_fips",
        n_splits=3,
    )

    assert summary["n_obs"] == 60
    assert (out_dir / "support_summary.json").exists()


def test_build_continuous_treatment_support_diagnostics_supports_panel_mode(tmp_path: Path):
    df = pd.DataFrame(
        {
            "county_fips": ["01001"] * 20 + ["01003"] * 20 + ["01005"] * 20,
            "year": list(range(2000, 2020)) * 3,
            "effective_min_wage": np.linspace(5.0, 10.0, 60),
            "poverty_rate": np.linspace(10.0, 20.0, 60),
            "unemployment_rate": np.linspace(4.0, 7.0, 60),
            "log_population": np.linspace(8.0, 10.0, 60),
        }
    )

    out_dir = tmp_path / "overlap_panel_mode"
    summary = build_continuous_treatment_support_diagnostics(
        df=df,
        treatment="effective_min_wage",
        controls=["poverty_rate", "unemployment_rate", "log_population"],
        output_dir=out_dir,
        group_col="county_fips",
        panel_mode="two_way_within",
        entity_col="county_fips",
        time_col="year",
        n_splits=3,
    )

    assert summary["panel_mode"] == "two_way_within"
    saved = json.loads((out_dir / "support_summary.json").read_text(encoding="utf-8"))
    assert saved["panel_mode"] == "two_way_within"
