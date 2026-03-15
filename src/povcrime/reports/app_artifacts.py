"""Machine-readable artifact bundle for frontend consumption."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_PANEL_SOURCE_COLUMNS: dict[str, list[str]] = {
    "saipe": ["poverty_rate", "median_hh_income", "population"],
    "laus": ["unemployment_rate"],
    "bea": ["per_capita_personal_income"],
    "acs": [
        "pct_white",
        "pct_black",
        "pct_hispanic",
        "pct_under_18",
        "pct_over_65",
        "pct_hs_or_higher",
        "median_age",
    ],
    "census_cbp": [
        "cbp_establishments",
        "cbp_employment",
        "cbp_annual_payroll",
        "cbp_establishments_per_1k",
        "cbp_employment_per_capita",
        "cbp_payroll_per_employee",
    ],
    "fhfa_hpi": [
        "fhfa_hpi",
        "fhfa_hpi_1990_base",
        "fhfa_hpi_2000_base",
        "fhfa_annual_change_pct",
        "log_fhfa_hpi_2000_base",
    ],
    "hud_fmr": [
        "fair_market_rent_0br",
        "fair_market_rent_1br",
        "fair_market_rent_2br",
        "fair_market_rent_3br",
        "fair_market_rent_4br",
        "log_fair_market_rent_2br",
        "rent_to_income_ratio_2br",
    ],
    "ukcpr_welfare": [
        "state_eitc_rate",
        "state_eitc_refundable",
        "tanf_benefit_2_person",
        "tanf_benefit_3_person",
        "tanf_benefit_4_person",
    ],
    "fbi_crime": [
        "violent_crime_count",
        "property_crime_count",
        "violent_crime_rate",
        "property_crime_rate",
    ],
    "policy": ["effective_min_wage", "broad_based_cat_elig"],
}

_LANE_FAMILIES: dict[str, dict[str, object]] = {
    "min_wage": {
        "treatment_family": "minimum_wage",
        "display_priority": "primary",
        "headline_eligible": True,
    },
    "eitc": {
        "treatment_family": "state_eitc",
        "display_priority": "secondary",
        "headline_eligible": False,
    },
    "tanf": {
        "treatment_family": "tanf",
        "display_priority": "exploratory",
        "headline_eligible": False,
    },
    "snap_bbce": {
        "treatment_family": "snap_bbce",
        "display_priority": "exploratory",
        "headline_eligible": False,
    },
}


def build_app_artifacts(
    *,
    project_root: Path,
    panel: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write a stable artifact manifest and results summary for the app."""
    app_dir = output_dir / "app"
    app_dir.mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(project_root=project_root, output_dir=output_dir)
    summary = _build_results_summary(project_root=project_root, panel=panel, output_dir=output_dir)
    credibility = _build_credibility_summary(summary=summary, output_dir=output_dir)

    manifest_path = app_dir / "artifact_manifest.json"
    summary_path = app_dir / "results_summary.json"
    credibility_path = app_dir / "credibility_summary.json"
    summary["artifacts"]["credibility_summary"] = str(credibility_path.relative_to(project_root))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    credibility_path.write_text(json.dumps(credibility, indent=2), encoding="utf-8")
    return manifest_path, summary_path


def _build_manifest(*, project_root: Path, output_dir: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "category": path.relative_to(output_dir).parts[0],
                "path": str(path.relative_to(project_root)),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "generated_date": date.today().isoformat(),
        "output_root": str(output_dir.relative_to(project_root)),
        "files": files,
    }


def _build_results_summary(
    *,
    project_root: Path,
    panel: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    estimand_slugs = _discover_estimand_slugs(output_dir)
    estimands = [
        _build_estimand_summary(slug=slug, project_root=project_root, output_dir=output_dir)
        for slug in estimand_slugs
    ]
    return {
        "generated_date": date.today().isoformat(),
        "panel": {
            "rows": int(len(panel)),
            "counties": int(panel["county_fips"].nunique()),
            "year_min": int(panel["year"].min()),
            "year_max": int(panel["year"].max()),
            "violent_rows": int(panel["violent_crime_rate"].notna().sum())
            if "violent_crime_rate" in panel.columns
            else 0,
            "property_rows": int(panel["property_crime_rate"].notna().sum())
            if "property_crime_rate" in panel.columns
            else 0,
            "low_coverage_rows": int(panel["low_coverage"].sum())
            if "low_coverage" in panel.columns
            else 0,
            "available_sources": _panel_source_coverage(panel),
        },
        "artifacts": {
            "qa_report": _relpath_if_exists(output_dir / "qa" / "data_quality_report.md", project_root),
            "final_report": _relpath_if_exists(output_dir / "report" / "final_report.md", project_root),
            "robustness_summary": _relpath_if_exists(output_dir / "robustness" / "robustness_summary.csv", project_root),
            "border_summary": _relpath_if_exists(output_dir / "border" / "border_summary.csv", project_root),
            "min_wage_identification_summary": _relpath_if_exists(
                output_dir / "min_wage_identification" / "min_wage_identification_summary.csv",
                project_root,
            ),
            "min_wage_event_study_summary": _relpath_if_exists(
                output_dir / "min_wage_identification" / "min_wage_event_study_summary.csv",
                project_root,
            ),
            "min_wage_dose_bucket_summary": _relpath_if_exists(
                output_dir / "min_wage_identification" / "min_wage_dose_bucket_summary.csv",
                project_root,
            ),
            "min_wage_negative_control_treatment_summary": _relpath_if_exists(
                output_dir / "min_wage_identification" / "min_wage_negative_control_treatment_summary.csv",
                project_root,
            ),
            "support_trim_summary": _relpath_if_exists(output_dir / "support_trim" / "support_trim_summary.csv", project_root),
            "negative_control_summary": _relpath_if_exists(output_dir / "falsification" / "negative_control_summary.csv", project_root),
            "staggered_summary": _relpath_if_exists(output_dir / "staggered" / "staggered_summary.csv", project_root),
            "crime_measurement_validation": _relpath_if_exists(
                output_dir / "crime_validation" / "crime_measurement_validation.json",
                project_root,
            ),
            "credibility_summary": _relpath_if_exists(
                output_dir / "app" / "credibility_summary.json",
                project_root,
            ),
            "bidirectional_poverty_crime_summary": _relpath_if_exists(
                output_dir / "exploratory" / "bidirectional_poverty_crime" / "bidirectional_summary.json",
                project_root,
            ),
        },
        "exploratory": {
            "bidirectional_poverty_crime": _read_json_relaxed(
                output_dir / "exploratory" / "bidirectional_poverty_crime" / "bidirectional_summary.json"
            ),
        },
        "research_lanes": _build_research_lanes(estimands),
        "estimands": estimands,
    }


def _discover_estimand_slugs(output_dir: Path) -> list[str]:
    slugs = {
        path.name
        for path in [
            *(output_dir / "baseline").glob("*"),
            *(output_dir / "dml").glob("*"),
            *(output_dir / "overlap").glob("*"),
            *(output_dir / "robustness").glob("*"),
        ]
        if path.is_dir()
    }
    return sorted(slugs)


def _build_estimand_summary(
    *,
    slug: str,
    project_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    baseline_dir = output_dir / "baseline" / slug
    dml_dir = output_dir / "dml" / slug
    overlap_dir = output_dir / "overlap" / slug

    baseline = _baseline_summary(baseline_dir=baseline_dir, project_root=project_root)
    dml = _dml_summary(dml_dir=dml_dir, project_root=project_root)
    causal_forest = _causal_forest_summary(dml_dir=dml_dir, project_root=project_root)
    robustness = _robustness_summary(slug=slug, output_dir=output_dir, project_root=project_root)
    overlap = _overlap_summary(overlap_dir=overlap_dir, project_root=project_root)

    result: dict[str, object] = {
        "slug": slug,
        "title": _slug_to_title(slug),
        "baseline": baseline,
        "dml": dml,
        "causal_forest": causal_forest,
        "robustness": robustness,
        "overlap": overlap,
        "frontend": _build_frontend_metadata(
            slug=slug,
            baseline=baseline,
            dml=dml,
            overlap=overlap,
        ),
    }
    return result


def _build_research_lanes(estimands: list[dict[str, object]]) -> dict[str, list[str]]:
    lanes: dict[str, list[str]] = {
        "primary": [],
        "secondary": [],
        "exploratory": [],
    }
    for estimand in estimands:
        frontend = estimand.get("frontend") or {}
        priority = frontend.get("display_priority")
        if priority in lanes:
            lanes[str(priority)].append(str(estimand["slug"]))
    return lanes


def _baseline_summary(*, baseline_dir: Path, project_root: Path) -> dict[str, object] | None:
    summary_path = baseline_dir / "baseline_fe_summary.csv"
    if not summary_path.exists():
        return None
    summary = pd.read_csv(summary_path)
    treatment_row = summary.iloc[0].to_dict() if not summary.empty else {}
    pretrend_path = baseline_dir / "event_study_pretrend.json"
    pretrend = json.loads(pretrend_path.read_text()) if pretrend_path.exists() else None
    return {
        "summary_path": str(summary_path.relative_to(project_root)),
        "event_study_path": _relpath_if_exists(baseline_dir / "event_study_coefs.csv", project_root),
        "event_study_plot_path": _relpath_if_exists(baseline_dir / "event_study_plot.png", project_root),
        "treatment_row": treatment_row,
        "pretrend": pretrend,
    }


def _dml_summary(*, dml_dir: Path, project_root: Path) -> dict[str, object] | None:
    json_files = sorted(dml_dir.glob("dml_*.json"))
    if not json_files:
        return None
    path = json_files[0]
    summary = json.loads(path.read_text())
    summary["path"] = str(path.relative_to(project_root))
    return summary


def _causal_forest_summary(*, dml_dir: Path, project_root: Path) -> dict[str, object] | None:
    ate_files = sorted(dml_dir.glob("causal_forest_ate_*.json"))
    if not ate_files:
        return None
    ate = json.loads(ate_files[0].read_text())
    ate["path"] = str(ate_files[0].relative_to(project_root))
    importance_files = sorted(dml_dir.glob("causal_forest_importance_*.csv"))
    if importance_files:
        fi = pd.read_csv(importance_files[0]).head(5)
        ate["top_features"] = fi.to_dict(orient="records")
        ate["importance_path"] = str(importance_files[0].relative_to(project_root))
    return ate


def _robustness_summary(
    *,
    slug: str,
    output_dir: Path,
    project_root: Path,
) -> dict[str, object] | None:
    summary_path = output_dir / "robustness" / "robustness_summary.csv"
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    sub = df.loc[df["label"] == slug].copy()
    if sub.empty:
        return None
    return {
        "path": str(summary_path.relative_to(project_root)),
        "rows": sub.to_dict(orient="records"),
    }


def _overlap_summary(*, overlap_dir: Path, project_root: Path) -> dict[str, object] | None:
    summary_path = overlap_dir / "support_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    summary["path"] = str(summary_path.relative_to(project_root))
    summary["control_balance_path"] = _relpath_if_exists(overlap_dir / "control_balance.csv", project_root)
    summary["support_bins_path"] = _relpath_if_exists(overlap_dir / "support_bins.csv", project_root)
    return summary


def _relpath_if_exists(path: Path, project_root: Path) -> str | None:
    if not path.exists():
        return None
    return str(path.relative_to(project_root))


def _read_json_relaxed(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _slug_to_title(slug: str) -> str:
    mapping = {
        "min_wage_violent": "Minimum Wage -> Violent Crime",
        "min_wage_property": "Minimum Wage -> Property Crime",
        "snap_bbce_violent": "SNAP BBCE -> Violent Crime",
        "snap_bbce_property": "SNAP BBCE -> Property Crime",
        "eitc_violent": "State EITC Rate -> Violent Crime",
        "eitc_property": "State EITC Rate -> Property Crime",
        "tanf_violent": "TANF Benefit -> Violent Crime",
        "tanf_property": "TANF Benefit -> Property Crime",
    }
    return mapping.get(slug, slug.replace("_", " ").title())


def _build_frontend_metadata(
    *,
    slug: str,
    baseline: dict[str, object] | None,
    dml: dict[str, object] | None,
    overlap: dict[str, object] | None,
) -> dict[str, object]:
    family_key = _family_key_from_slug(slug)
    family = _LANE_FAMILIES[family_key]
    pretrend = baseline.get("pretrend") if baseline else None
    overlap_max_abs_smd = _as_float((overlap or {}).get("max_abs_smd"))
    pretrend_pass = None if pretrend is None else bool(pretrend.get("pass"))
    summary = _frontend_summary(
        family_key=family_key,
        baseline=baseline,
        dml=dml,
        overlap=overlap,
        pretrend_pass=pretrend_pass,
    )
    caveats = _frontend_caveats(
        baseline=baseline,
        dml=dml,
        overlap=overlap,
        pretrend_pass=pretrend_pass,
    )
    return {
        "treatment_family": family["treatment_family"],
        "outcome_family": _outcome_family_from_slug(slug),
        "display_priority": family["display_priority"],
        "headline_eligible": family["headline_eligible"],
        "status": _frontend_status(
            family_key=family_key,
            pretrend_pass=pretrend_pass,
            overlap_max_abs_smd=overlap_max_abs_smd,
        ),
        "summary": summary,
        "caveats": caveats,
    }


def _family_key_from_slug(slug: str) -> str:
    if slug.startswith("snap_bbce_"):
        return "snap_bbce"
    if slug.startswith("min_wage_"):
        return "min_wage"
    if slug.startswith("eitc_"):
        return "eitc"
    if slug.startswith("tanf_"):
        return "tanf"
    raise KeyError(f"Unrecognized estimand slug: {slug}")


def _outcome_family_from_slug(slug: str) -> str:
    if slug.endswith("_violent"):
        return "violent_crime"
    if slug.endswith("_property"):
        return "property_crime"
    return "unknown"


def _frontend_status(
    *,
    family_key: str,
    pretrend_pass: bool | None,
    overlap_max_abs_smd: float | None,
) -> str:
    if family_key == "snap_bbce":
        return "exploratory_failed_pretrends"
    if family_key == "tanf":
        return "exploratory_low_signal"
    if family_key == "eitc":
        return "secondary_method_sensitive"
    if pretrend_pass is False:
        return "primary_pretrend_warning"
    if overlap_max_abs_smd is not None and overlap_max_abs_smd >= 0.7:
        return "primary_overlap_caution"
    return "primary_mixed_signal"


def _frontend_summary(
    *,
    family_key: str,
    baseline: dict[str, object] | None,
    dml: dict[str, object] | None,
    overlap: dict[str, object] | None,
    pretrend_pass: bool | None,
) -> str:
    fe_row = (baseline or {}).get("treatment_row") or {}
    fe_sig = _is_significant(fe_row.get("p_value"))
    dml_sig = _is_significant((dml or {}).get("p_value"))
    dml_direction = _direction_label((dml or {}).get("theta"))
    overlap_label = _overlap_label((overlap or {}).get("max_abs_smd"))

    if family_key == "min_wage":
        if dml_sig:
            return f"FE is weak; DML is {dml_direction}; overlap caution is {overlap_label}."
        return f"FE and DML are both weak or mixed; overlap caution is {overlap_label}."
    if family_key == "eitc":
        fe_clause = "FE pretrends pass but FE is weak" if not fe_sig and pretrend_pass is True else "FE remains mixed"
        dml_clause = f"DML is {dml_direction}" if dml_sig else "DML is weak"
        return f"{fe_clause}; {dml_clause}; treat this lane as method-sensitive."
    if family_key == "tanf":
        return f"FE and DML stay near zero; overlap is {overlap_label}; keep this lane exploratory."
    return "FE is weak and pretrend tests fail; keep this lane exploratory."


def _frontend_caveats(
    *,
    baseline: dict[str, object] | None,
    dml: dict[str, object] | None,
    overlap: dict[str, object] | None,
    pretrend_pass: bool | None,
) -> list[str]:
    caveats: list[str] = []
    fe_row = (baseline or {}).get("treatment_row") or {}
    fe_sig = _is_significant(fe_row.get("p_value"))
    dml_sig = _is_significant((dml or {}).get("p_value"))
    max_abs_smd = _as_float((overlap or {}).get("max_abs_smd"))
    if pretrend_pass is False:
        caveats.append("Event-study pretrend tests fail.")
    elif pretrend_pass is None and baseline is not None:
        caveats.append("No event-study pretrend test is available for this lane.")
    if max_abs_smd is not None:
        if max_abs_smd >= 1.0:
            caveats.append("Overlap is poor across treatment tails.")
        elif max_abs_smd >= 0.7:
            caveats.append("Overlap is only moderate across treatment tails.")
    if baseline is not None and dml is not None and fe_sig != dml_sig:
        caveats.append("FE and DML disagree on statistical detectability.")
    if dml is None:
        caveats.append("No ML estimate was generated for this lane.")
    return caveats


def _direction_label(value: object) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "mixed"
    if numeric > 0:
        return "positive"
    if numeric < 0:
        return "negative"
    return "near zero"


def _overlap_label(value: object) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "unavailable"
    if numeric >= 1.0:
        return "poor"
    if numeric >= 0.7:
        return "moderate"
    return "usable"


def _is_significant(value: object) -> bool:
    numeric = _as_float(value)
    return numeric is not None and numeric < 0.05


def _as_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _panel_source_coverage(panel: pd.DataFrame) -> list[dict[str, object]]:
    total_rows = max(len(panel), 1)
    summaries: list[dict[str, object]] = []
    for name, columns in _PANEL_SOURCE_COLUMNS.items():
        present_columns = [col for col in columns if col in panel.columns]
        if not present_columns:
            continue
        key_series = panel[present_columns].notna().any(axis=1)
        summaries.append(
            {
                "name": name,
                "columns": present_columns,
                "non_null_rows": int(key_series.sum()),
                "share": round(float(key_series.mean()), 4),
                "min_year_share": round(
                    float(key_series.groupby(panel["year"]).mean().min()),
                    4,
                ),
                "max_year_share": round(
                    float(key_series.groupby(panel["year"]).mean().max()),
                    4,
                ),
                "missing_rows": int(total_rows - key_series.sum()),
            }
        )
    return summaries


def _build_credibility_summary(
    *,
    summary: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    border = _read_csv_if_exists(output_dir / "border" / "border_summary.csv")
    support_trim = _read_csv_if_exists(output_dir / "support_trim" / "support_trim_summary.csv")
    falsification = _read_csv_if_exists(output_dir / "falsification" / "negative_control_summary.csv")
    staggered = _read_csv_if_exists(output_dir / "staggered" / "staggered_summary.csv")
    min_wage_id = _read_csv_if_exists(output_dir / "min_wage_identification" / "min_wage_identification_summary.csv")
    min_wage_event = _read_csv_if_exists(output_dir / "min_wage_identification" / "min_wage_event_study_summary.csv")
    min_wage_negative = _read_csv_if_exists(
        output_dir / "min_wage_identification" / "min_wage_negative_control_treatment_summary.csv"
    )
    crime_validation = _read_json_relaxed(output_dir / "crime_validation" / "crime_measurement_validation.json")

    lanes: list[dict[str, object]] = []
    for estimand in summary.get("estimands", []):
        slug = str(estimand["slug"])
        frontend = estimand.get("frontend") or {}
        checks = [
            _credibility_pretrend(estimand),
            _credibility_overlap(estimand),
            _credibility_support_trim(slug=slug, support_trim=support_trim),
            _credibility_border(slug=slug, border=border),
            _credibility_negative_outcomes(slug=slug, falsification=falsification),
            _credibility_staggered(slug=slug, staggered=staggered),
            _credibility_min_wage_redesign(
                slug=slug,
                min_wage_id=min_wage_id,
                min_wage_event=min_wage_event,
                min_wage_negative=min_wage_negative,
                crime_validation=crime_validation,
            ),
        ]
        checks = [check for check in checks if check is not None]
        statuses = {str(check["status"]) for check in checks}
        verdict = "caution" if "fail" in statuses else "mixed" if "warn" in statuses else "clean"
        lanes.append(
            {
                "slug": slug,
                "title": estimand.get("title"),
                "headline_eligible": frontend.get("headline_eligible"),
                "frontend_status": frontend.get("status"),
                "verdict": verdict,
                "checks": checks,
            }
        )
    return {
        "generated_date": summary.get("generated_date"),
        "lanes": lanes,
    }


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _credibility_pretrend(estimand: dict[str, object]) -> dict[str, object]:
    baseline = estimand.get("baseline") or {}
    pretrend = baseline.get("pretrend")
    if pretrend is None:
        return {"name": "event_pretrend", "status": "not_available", "detail": "No event-study pretrend test."}
    p_value = _as_float(pretrend.get("p_value"))
    if p_value is None:
        return {"name": "event_pretrend", "status": "warn", "detail": "p=n/a"}
    return {
        "name": "event_pretrend",
        "status": "pass" if bool(pretrend.get("pass")) else "fail",
        "detail": f"p={_fmt_optional(p_value)}",
    }


def _credibility_overlap(estimand: dict[str, object]) -> dict[str, object]:
    overlap = estimand.get("overlap") or {}
    max_abs_smd = _as_float(overlap.get("max_abs_smd"))
    if max_abs_smd is None:
        return {"name": "overlap", "status": "not_available", "detail": "No overlap diagnostics."}
    status = "pass" if max_abs_smd < 0.7 else "warn" if max_abs_smd < 1.0 else "fail"
    return {"name": "overlap", "status": status, "detail": f"max_abs_smd={max_abs_smd:.3f}"}


def _credibility_support_trim(*, slug: str, support_trim: pd.DataFrame | None) -> dict[str, object]:
    if support_trim is None or support_trim.empty:
        return {"name": "support_trim", "status": "not_available", "detail": "No support-trim summary."}
    required = {"label", "theta_base", "theta_trimmed", "p_value_trimmed"}
    if not required.issubset(support_trim.columns):
        return {"name": "support_trim", "status": "not_available", "detail": "Support-trim columns are incomplete."}
    row = support_trim.loc[support_trim["label"] == slug]
    if row.empty:
        return {"name": "support_trim", "status": "not_available", "detail": "No support-trim row for this lane."}
    row0 = row.iloc[0]
    theta_base = _as_float(row0.get("theta_base"))
    theta_trimmed = _as_float(row0.get("theta_trimmed"))
    p_trimmed = _as_float(row0.get("p_value_trimmed"))
    if theta_base is None or theta_trimmed is None:
        status = "not_available"
    elif np.sign(theta_base) != np.sign(theta_trimmed) and theta_base != 0 and theta_trimmed != 0:
        status = "fail"
    elif p_trimmed is not None and p_trimmed < 0.05:
        status = "pass"
    else:
        status = "warn"
    return {
        "name": "support_trim",
        "status": status,
        "detail": f"base={_fmt_optional(theta_base)}, trimmed={_fmt_optional(theta_trimmed)}",
    }


def _credibility_border(*, slug: str, border: pd.DataFrame | None) -> dict[str, object]:
    if border is None or border.empty:
        return {"name": "border_design", "status": "not_available", "detail": "No border summary."}
    required = {"label", "spec", "p_value"}
    if not required.issubset(border.columns):
        return {"name": "border_design", "status": "not_available", "detail": "Border summary columns are incomplete."}
    border_label = f"border_{slug}"
    rows = border.loc[border["label"] == border_label]
    if rows.empty:
        return {"name": "border_design", "status": "not_available", "detail": "No border row for this lane."}
    baseline = rows.loc[rows["spec"] == "baseline"]
    placebo = rows.loc[rows["spec"] == "placebo_lead"]
    p_base = _as_float(baseline["p_value"].iloc[0]) if not baseline.empty else None
    p_placebo = _as_float(placebo["p_value"].iloc[0]) if not placebo.empty else None
    if p_placebo is not None and p_placebo < 0.05:
        status = "fail"
    elif p_base is not None and p_base < 0.05:
        status = "pass"
    else:
        status = "warn"
    return {
        "name": "border_design",
        "status": status,
        "detail": f"baseline_p={_fmt_optional(p_base)}, placebo_p={_fmt_optional(p_placebo)}",
    }


def _credibility_negative_outcomes(*, slug: str, falsification: pd.DataFrame | None) -> dict[str, object]:
    if falsification is None or falsification.empty:
        return {"name": "negative_control_outcomes", "status": "not_available", "detail": "No falsification summary."}
    required = {"label", "p_value"}
    if not required.issubset(falsification.columns):
        return {"name": "negative_control_outcomes", "status": "not_available", "detail": "Falsification columns are incomplete."}
    prefix = "min_wage__" if slug.startswith("min_wage_") else "eitc__" if slug.startswith("eitc_") else None
    if prefix is None:
        return {"name": "negative_control_outcomes", "status": "not_available", "detail": "No negative-control outcome lane."}
    rows = falsification.loc[falsification["label"].astype(str).str.startswith(prefix)]
    if rows.empty:
        return {"name": "negative_control_outcomes", "status": "not_available", "detail": "No falsification rows for this lane."}
    n_sig = int((rows["p_value"] < 0.05).sum())
    return {
        "name": "negative_control_outcomes",
        "status": "fail" if n_sig > 0 else "pass",
        "detail": f"significant_negative_controls={n_sig}",
    }


def _credibility_staggered(*, slug: str, staggered: pd.DataFrame | None) -> dict[str, object]:
    if staggered is None or staggered.empty:
        return {"name": "staggered_att", "status": "not_available", "detail": "No staggered summary."}
    required = {"label", "interpretable", "pretrend_pass"}
    if not required.issubset(staggered.columns):
        return {"name": "staggered_att", "status": "not_available", "detail": "Staggered summary columns are incomplete."}
    row = staggered.loc[staggered["label"] == slug]
    if row.empty:
        return {"name": "staggered_att", "status": "not_available", "detail": "No staggered row for this lane."}
    row0 = row.iloc[0]
    interpretable = bool(row0.get("interpretable"))
    pretrend_pass = bool(row0.get("pretrend_pass"))
    if not interpretable:
        status = "warn"
    elif pretrend_pass:
        status = "pass"
    else:
        status = "fail"
    return {
        "name": "staggered_att",
        "status": status,
        "detail": f"interpretable={interpretable}, pretrend_pass={pretrend_pass}",
    }


def _credibility_min_wage_redesign(
    *,
    slug: str,
    min_wage_id: pd.DataFrame | None,
    min_wage_event: pd.DataFrame | None,
    min_wage_negative: pd.DataFrame | None,
    crime_validation: dict[str, object] | None,
) -> dict[str, object] | None:
    if not slug.startswith("min_wage_"):
        return None
    if min_wage_id is None or min_wage_id.empty:
        return {"name": "min_wage_redesign", "status": "not_available", "detail": "No minimum-wage redesign summary."}
    required = {"outcome_label", "spec", "p_value"}
    if not required.issubset(min_wage_id.columns):
        return {"name": "min_wage_redesign", "status": "not_available", "detail": "Minimum-wage redesign columns are incomplete."}
    outcome_label = "violent" if slug.endswith("_violent") else "property"
    fd = min_wage_id.loc[
        (min_wage_id["outcome_label"] == outcome_label)
        & (min_wage_id["spec"] == "border_pair_first_difference")
    ]
    placebo = min_wage_id.loc[
        (min_wage_id["outcome_label"] == outcome_label)
        & (min_wage_id["spec"] == "border_pair_first_difference_placebo")
    ]
    event = (
        min_wage_event.loc[min_wage_event["outcome_label"] == outcome_label]
        if min_wage_event is not None and {"outcome_label", "pretrend_pass"}.issubset(min_wage_event.columns)
        else None
    )
    neg = (
        min_wage_negative.loc[min_wage_negative["outcome_label"] == outcome_label]
        if min_wage_negative is not None and {"outcome_label", "p_value"}.issubset(min_wage_negative.columns)
        else None
    )
    p_fd = _as_float(fd["p_value"].iloc[0]) if not fd.empty else None
    p_placebo = _as_float(placebo["p_value"].iloc[0]) if not placebo.empty else None
    p_neg = _as_float(neg["p_value"].iloc[0]) if neg is not None and not neg.empty else None
    event_pass = bool(event["pretrend_pass"].iloc[0]) if event is not None and not event.empty else None
    if p_placebo is not None and p_placebo < 0.05:
        status = "fail"
    elif p_neg is not None and p_neg < 0.05:
        status = "fail"
    elif p_fd is not None and p_fd < 0.05 and event_pass is not False:
        status = "pass"
    else:
        status = "warn"
    detail = (
        f"fd_p={_fmt_optional(p_fd)}, placebo_p={_fmt_optional(p_placebo)}, "
        f"neg_treat_p={_fmt_optional(p_neg)}, event_pretrend={event_pass}"
    )
    sens_rows = ((crime_validation or {}).get("robustness_sensitivity") or []) if crime_validation else []
    sens = next((row for row in sens_rows if row.get("label") == slug), None)
    if sens is not None:
        detail += f", coverage_sign_flip={sens.get('sign_flip')}"
    return {"name": "min_wage_redesign", "status": status, "detail": detail}


def _fmt_optional(value: object) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.4f}"
