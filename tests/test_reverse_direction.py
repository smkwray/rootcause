"""Tests for the reverse-direction exploratory scaffold."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from povcrime.reports.reverse_direction import build_reverse_direction_scaffold


def test_build_reverse_direction_scaffold_writes_outputs(tmp_path: Path):
    panel = pd.DataFrame(
        {
            "violent_crime_rate": [100.0, 120.0],
            "property_crime_rate": [1000.0, 900.0],
            "poverty_rate": [10.0, 12.0],
            "unemployment_rate": [5.0, 6.0],
        }
    )

    md_path, json_path = build_reverse_direction_scaffold(
        panel=panel,
        output_dir=tmp_path / "exploratory",
    )

    assert md_path.exists()
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "exploratory_only"
    assert payload["specs"][0]["usable_rows"] == 2
