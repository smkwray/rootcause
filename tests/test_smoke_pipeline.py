"""Fixture-backed script smoke test for the backend pipeline."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd

from povcrime.config import load_project_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "smoke"


def _load_script_module(name: str):
    script_path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_panel = _load_script_module("build_panel")
build_qa_report = _load_script_module("build_qa_report")
build_app_artifacts = _load_script_module("build_app_artifacts")
build_final_report = _load_script_module("build_final_report")
refresh_public_data = _load_script_module("refresh_public_data")


def test_fixture_backed_script_smoke_pipeline(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "smoke_project"
    _prepare_smoke_project(project_root)
    config = load_project_config(project_root)

    monkeypatch.setattr(build_panel, "get_config", lambda: config)
    monkeypatch.setattr(build_qa_report, "get_config", lambda: config)
    monkeypatch.setattr(build_app_artifacts, "get_config", lambda: config)
    monkeypatch.setattr(build_final_report, "get_config", lambda: config)
    monkeypatch.setattr(refresh_public_data, "ROOT", project_root)
    monkeypatch.setattr(refresh_public_data, "OUTPUTS", config.output_dir / "app")
    monkeypatch.setattr(refresh_public_data, "DEST", project_root / "docs" / "assets" / "data")

    build_panel.main([])
    build_qa_report.main([])
    build_app_artifacts.main([])
    build_final_report.main([])
    refresh_public_data.build_site_data()

    panel_path = config.processed_dir / "panel.parquet"
    qa_path = config.output_dir / "qa" / "data_quality_report.md"
    results_path = config.output_dir / "app" / "results_summary.json"
    report_path = config.output_dir / "report" / "final_report.md"
    site_data_path = project_root / "docs" / "assets" / "data" / "site_data.json"

    assert panel_path.exists()
    assert qa_path.exists()
    assert results_path.exists()
    assert report_path.exists()
    assert site_data_path.exists()

    results = json.loads(results_path.read_text(encoding="utf-8"))
    site_data = json.loads(site_data_path.read_text(encoding="utf-8"))
    panel = pd.read_parquet(panel_path)

    assert list(panel["county_fips"]) == ["01001", "01003"]
    assert results["panel"]["crime_data_level"] == "county_fallback"
    assert "min_wage_violent" in results["research_lanes"]["primary"]
    assert any(row["slug"] == "eitc_violent" for row in results["estimands"])
    assert any(row["name"] == "fbi_crime" for row in results["panel"]["available_sources"])
    assert any(lane["slug"] == "eitc_violent" and lane["tier"] == "secondary" for lane in site_data["lanes"])


def _prepare_smoke_project(project_root: Path) -> None:
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "configs" / "project.yaml", project_root / "configs" / "project.yaml")

    _materialize_saipe_parquet(project_root)
    _copy_fixture_file(
        FIXTURE_ROOT / "raw" / "fbi_county.csv",
        project_root / "data" / "raw" / "fbi_crime" / "county_crime.csv",
    )
    shutil.copytree(FIXTURE_ROOT / "outputs", project_root / "outputs", dirs_exist_ok=True)
    _write_placeholder_png(project_root / "outputs" / "baseline" / "min_wage_violent" / "event_study_plot.png")
    _write_placeholder_png(project_root / "outputs" / "baseline" / "min_wage_property" / "event_study_plot.png")
    (project_root / "docs").mkdir(parents=True, exist_ok=True)


def _materialize_saipe_parquet(project_root: Path) -> None:
    raw_csv = FIXTURE_ROOT / "raw" / "saipe.csv"
    out_path = project_root / "data" / "raw" / "saipe" / "saipe_2020.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(raw_csv).to_parquet(out_path, index=False)


def _copy_fixture_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_placeholder_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
