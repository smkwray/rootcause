"""Tests for DML estimator reproducibility and grouped splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from povcrime.models.dml import DMLEstimator
from povcrime.models.panel_ml import prepare_panel_ml_sample


def _make_panel_like_frame(n_counties: int = 3, n_years: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for county_idx in range(n_counties):
        county_fips = f"01{county_idx + 1:03d}"
        for year in range(n_years):
            x1 = float(rng.normal())
            x2 = float(rng.normal())
            treatment = 0.5 * x1 - 0.3 * x2 + float(rng.normal(scale=0.5))
            outcome = 2.0 * treatment + 0.7 * x1 + float(rng.normal(scale=0.5))
            rows.append(
                {
                    "county_fips": county_fips,
                    "year": 2000 + year,
                    "x1": x1,
                    "x2": x2,
                    "d": treatment,
                    "y": outcome,
                }
            )
    return pd.DataFrame(rows)


def test_dml_estimator_is_reproducible_with_default_seed():
    rng = np.random.default_rng(42)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + 0.7 * x1 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})

    est1 = DMLEstimator(
        df=df,
        outcome="y",
        treatment="d",
        controls=["x1", "x2"],
        n_folds=3,
    )
    est1.fit()
    est2 = DMLEstimator(
        df=df,
        outcome="y",
        treatment="d",
        controls=["x1", "x2"],
        n_folds=3,
    )
    est2.fit()

    assert est1.summary()["theta"] == est2.summary()["theta"]
    assert est1.summary()["p_value"] == est2.summary()["p_value"]


def test_dml_estimator_groups_panel_units_in_cross_fitting():
    df = _make_panel_like_frame()

    est = DMLEstimator(
        df=df,
        outcome="y",
        treatment="d",
        controls=["x1", "x2"],
        group_col="county_fips",
        n_folds=3,
    )

    assert "county_fips" in est._df.columns

    folds = est._build_sample_splitting()[0]
    for train_idx, test_idx in folds:
        train_counties = set(est._df.loc[train_idx, "county_fips"])
        test_counties = set(est._df.loc[test_idx, "county_fips"])
        assert train_counties.isdisjoint(test_counties)


def test_dml_estimator_rejects_too_few_unique_groups():
    df = _make_panel_like_frame(n_counties=2, n_years=4)

    with pytest.raises(ValueError, match="unique groups"):
        DMLEstimator(
            df=df,
            outcome="y",
            treatment="d",
            controls=["x1", "x2"],
            group_col="county_fips",
            n_folds=3,
        )


def test_dml_estimator_still_supports_ungrouped_panel_data():
    df = _make_panel_like_frame()

    est = DMLEstimator(
        df=df,
        outcome="y",
        treatment="d",
        controls=["x1", "x2"],
        group_col=None,
        n_folds=3,
    )
    est.fit()

    summary = est.summary()
    assert summary["n_obs"] == len(df)
    assert summary["group_col"] is None


def test_prepare_panel_ml_sample_two_way_within_preserves_rows_and_keys():
    df = _make_panel_like_frame(n_counties=4, n_years=5)

    sample = prepare_panel_ml_sample(
        df,
        model_cols=["y", "d", "x1", "x2"],
        keep_cols=["county_fips", "year"],
        panel_mode="two_way_within",
        entity_col="county_fips",
        time_col="year",
    )

    assert len(sample) == len(df)
    assert sample[["county_fips", "year"]].equals(df[["county_fips", "year"]])
    assert sample.groupby("county_fips")["d"].mean().abs().max() < 1e-10
    assert sample.groupby("year")["d"].mean().abs().max() < 1e-10


def test_dml_estimator_supports_panel_mode_summary_fields():
    df = _make_panel_like_frame(n_counties=6, n_years=5)

    est = DMLEstimator(
        df=df,
        outcome="y",
        treatment="d",
        controls=["x1", "x2"],
        group_col="county_fips",
        panel_mode="two_way_within",
        entity_col="county_fips",
        time_col="year",
        n_folds=3,
    )
    est.fit()

    summary = est.summary()
    assert summary["n_obs"] == len(df)
    assert summary["group_col"] == "county_fips"
    assert summary["panel_mode"] == "two_way_within"
    assert summary["entity_col"] == "county_fips"
    assert summary["time_col"] == "year"
