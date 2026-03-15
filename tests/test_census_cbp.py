"""Tests for Census CBP parsing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import requests

from povcrime.config import ProjectConfig
from povcrime.data.census_cbp import CensusCBPAdapter, _parse_cbp_json


def test_parse_cbp_json_builds_expected_schema(tmp_path: Path):
    path = tmp_path / "cbp_2023.json"
    payload = [
        ["ESTAB", "EMP", "PAYANN", "state", "county"],
        ["968", "12749", "527009", "01", "001"],
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    df = _parse_cbp_json(path)

    assert df.loc[0, "county_fips"] == "01001"
    assert df.loc[0, "state_fips"] == "01"
    assert df.loc[0, "year"] == 2023
    assert df.loc[0, "cbp_establishments"] == 968


def test_download_skips_unpublished_404_year(monkeypatch, tmp_path: Path):
    cfg = ProjectConfig(start_year=2023, end_year=2024)
    cfg.raw_dir = tmp_path

    response = Mock()
    response.status_code = 404
    response.url = "https://api.census.gov/data/2024/cbp"
    response.raise_for_status.side_effect = requests.HTTPError(response=response)

    def fake_get(url, params, timeout):  # noqa: ANN001
        if "2024" in url:
            return response
        ok = Mock()
        ok.status_code = 200
        ok.text = json.dumps([["ESTAB", "EMP", "PAYANN", "state", "county"], ["1", "2", "3", "01", "001"]])
        ok.raise_for_status.return_value = None
        return ok

    monkeypatch.setattr("povcrime.data.census_cbp.requests.get", fake_get)

    adapter = CensusCBPAdapter(cfg)
    adapter.download()

    assert (tmp_path / "census_cbp" / "cbp_2023.json").exists()
    assert not (tmp_path / "census_cbp" / "cbp_2024.json").exists()
