"""Tests for DML estimator reproducibility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from povcrime.models.dml import DMLEstimator


def test_dml_estimator_is_reproducible_with_default_seed():
    rng = np.random.default_rng(42)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + 0.7 * x1 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})

    est1 = DMLEstimator(df=df, outcome="y", treatment="d", controls=["x1", "x2"], n_folds=3)
    est1.fit()
    est2 = DMLEstimator(df=df, outcome="y", treatment="d", controls=["x1", "x2"], n_folds=3)
    est2.fit()

    assert est1.summary()["theta"] == est2.summary()["theta"]
    assert est1.summary()["p_value"] == est2.summary()["p_value"]
