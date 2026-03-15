"""Tests for overlap/support diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from povcrime.models.overlap import build_continuous_treatment_support_diagnostics


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
