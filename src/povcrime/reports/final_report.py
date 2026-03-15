"""Assemble a final markdown report from generated backend artifacts."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd


_OUTCOME_SPECS = [
    {
        "label": "Minimum Wage -> Violent Crime",
        "slug": "min_wage_violent",
        "outcome": "violent_crime_rate",
    },
    {
        "label": "Minimum Wage -> Property Crime",
        "slug": "min_wage_property",
        "outcome": "property_crime_rate",
    },
]

_SOURCE_LABELS = {
    "census_cbp": "County Business Patterns",
    "hud_fmr": "HUD Fair Market Rents",
    "fhfa_hpi": "FHFA County HPI",
    "ukcpr_welfare": "UKCPR Welfare / EITC / TANF",
}


def build_final_report(
    *,
    panel: pd.DataFrame,
    panel_path: Path,
    qa_report_path: Path,
    baseline_dir: Path,
    dml_dir: Path,
    robustness_summary_path: Path,
    output_dir: Path,
    overlap_dir: Path | None = None,
    app_dir: Path | None = None,
) -> Path:
    """Build the final markdown report and return the written path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "final_report.md"

    lines = [
        "# Final Backend Report",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Backend Status",
        "",
        "- Status: backend-complete for the implemented county-year pipeline.",
        "- Core data products: FBI county fallback, panel, QA report, baseline FE/event-study, DML, causal forest, robustness checks.",
        "- Remaining work outside backend scope: frontend/UI presentation and any future expansion tracks.",
        "",
        "## Data And Sample",
        "",
        f"- Panel rows: {len(panel):,}",
        f"- Counties: {panel['county_fips'].nunique():,}",
        f"- Years: {panel['year'].min()}-{panel['year'].max()} ({panel['year'].nunique()} years)",
        f"- Violent-crime rate non-null rows: {int(panel['violent_crime_rate'].notna().sum()):,}",
        f"- Property-crime rate non-null rows: {int(panel['property_crime_rate'].notna().sum()):,}",
        f"- Mean source share: {panel['source_share'].mean():.3f}" if "source_share" in panel.columns else "- Mean source share: n/a",
        f"- Low-coverage rows: {int(panel['low_coverage'].sum()):,}" if "low_coverage" in panel.columns else "- Low-coverage rows: n/a",
        f"- QA report: [{qa_report_path.name}]({_relative_link(qa_report_path, output_dir)})",
        "",
        "## Main Results",
        "",
    ]

    for spec in _OUTCOME_SPECS:
        lines.extend(
            _build_outcome_section(
                spec=spec,
                baseline_dir=baseline_dir,
                dml_dir=dml_dir,
                report_dir=output_dir,
            )
        )

    lines.extend(
        _build_interpretation_section(
            baseline_dir=baseline_dir,
            dml_dir=dml_dir,
        )
    )
    lines.extend(
        _build_data_expansion_section(
            app_dir=app_dir,
            report_dir=output_dir,
        )
    )
    lines.extend(
        _build_policy_lane_section(
            app_dir=app_dir,
            report_dir=output_dir,
        )
    )
    lines.extend(
        _build_border_section(
            report_dir=output_dir,
        )
    )
    lines.extend(
        _build_min_wage_identification_section(
            report_dir=output_dir,
        )
    )
    lines.extend(
        _build_crime_validation_section(
            report_dir=output_dir,
        )
    )
    lines.extend(
        _build_support_trim_section(
            report_dir=output_dir,
        )
    )
    lines.extend(
        _build_falsification_section(
            report_dir=output_dir,
        )
    )
    lines.extend(
        _build_staggered_section(
            report_dir=output_dir,
        )
    )
    lines.extend(
        _build_bidirectional_section(
            app_dir=app_dir,
            report_dir=output_dir,
        )
    )
    lines.extend(
        _build_overlap_section(
            overlap_dir=overlap_dir,
            report_dir=output_dir,
        )
    )
    lines.extend(
        _build_snap_section(
            baseline_dir=baseline_dir,
            report_dir=output_dir,
        )
    )
    lines.extend(
        _build_robustness_section(
            robustness_summary_path=robustness_summary_path,
            report_dir=output_dir,
        )
    )

    lines.extend(
        [
            "## Caveats",
            "",
            "- Crime outcomes are recorded crimes, not total victimization.",
            "- The county-level FBI fallback comes from official Return A master files plus identifier-first county matching; covered population is approximate and month-reporting share is not recoverable from the annual master format.",
            "- ACS controls are pooled 5-year estimates and should be interpreted as slow-moving context, not precise annual shocks.",
            "- The DML and causal-forest stages are complementary to the FE/event-study design, not substitutes for identification assumptions.",
            "",
            "## Artifact Map",
            "",
            f"- Panel: [{panel_path.name}]({_relative_link(panel_path, output_dir)})",
            f"- QA: [{qa_report_path.name}]({_relative_link(qa_report_path, output_dir)})",
            f"- Baseline outputs: [{baseline_dir.name}]({_relative_link(baseline_dir, output_dir)})",
            f"- DML outputs: [{dml_dir.name}]({_relative_link(dml_dir, output_dir)})",
            f"- Robustness outputs: [{robustness_summary_path.parent.name}]({_relative_link(robustness_summary_path.parent, output_dir)})",
            f"- Overlap diagnostics: [{overlap_dir.name}]({_relative_link(overlap_dir, output_dir)})" if overlap_dir and overlap_dir.exists() else "- Overlap diagnostics: n/a",
            f"- App artifacts: [{app_dir.name}]({_relative_link(app_dir, output_dir)})" if app_dir and app_dir.exists() else "- App artifacts: n/a",
            "",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _build_outcome_section(
    *,
    spec: dict[str, str],
    baseline_dir: Path,
    dml_dir: Path,
    report_dir: Path,
) -> list[str]:
    slug = spec["slug"]
    base_dir = baseline_dir / slug
    dml_spec_dir = dml_dir / slug

    fe_summary = pd.read_csv(base_dir / "baseline_fe_summary.csv")
    fe_row = fe_summary.loc[fe_summary["variable"] == "effective_min_wage"].iloc[0]
    pretrend = json.loads((base_dir / "event_study_pretrend.json").read_text())

    dml_summary_path = next(dml_spec_dir.glob("dml_*.json"))
    dml_summary = json.loads(dml_summary_path.read_text())
    forest_ate_path = next(dml_spec_dir.glob("causal_forest_ate_*.json"))
    forest_ate = json.loads(forest_ate_path.read_text())
    forest_importance_path = next(dml_spec_dir.glob("causal_forest_importance_*.csv"))
    forest_importance = pd.read_csv(forest_importance_path).head(3)

    lines = [
        f"### {spec['label']}",
        "",
        f"- TWFE coefficient: {fe_row['coefficient']:.4f} (SE {fe_row['std_error']:.4f}, p={fe_row['p_value']:.4f})",
        f"- TWFE 95% CI: [{fe_row['ci_lower']:.4f}, {fe_row['ci_upper']:.4f}]",
        f"- Event-study pretrend p-value: {_fmt_stat(pretrend.get('p_value'))}",
        f"- Event-study plot: [{(base_dir / 'event_study_plot.png').name}]({_relative_link(base_dir / 'event_study_plot.png', report_dir)})",
        f"- DML theta: {dml_summary['theta']:.4f} (SE {dml_summary['se']:.4f}, p={dml_summary['p_value']:.4f})",
        f"- DML 95% CI: [{dml_summary['ci_lower']:.4f}, {dml_summary['ci_upper']:.4f}]",
        f"- Causal-forest ATE: {forest_ate['ate']:.4f} [{forest_ate['ci_lower']:.4f}, {forest_ate['ci_upper']:.4f}]",
        "- Top heterogeneity features:",
    ]
    for _, row in forest_importance.iterrows():
        lines.append(f"  - {row['feature']}: {row['importance']:.4f}")
    lines.extend(
        [
            f"- FE table: [{(base_dir / 'baseline_fe_summary.csv').name}]({_relative_link(base_dir / 'baseline_fe_summary.csv', report_dir)})",
            f"- Event-study coefficients: [{(base_dir / 'event_study_coefs.csv').name}]({_relative_link(base_dir / 'event_study_coefs.csv', report_dir)})",
            f"- DML summary: [{dml_summary_path.name}]({_relative_link(dml_summary_path, report_dir)})",
            f"- Causal-forest importance: [{forest_importance_path.name}]({_relative_link(forest_importance_path, report_dir)})",
            "",
        ]
    )
    return lines


def _build_interpretation_section(
    *,
    baseline_dir: Path,
    dml_dir: Path,
) -> list[str]:
    violent_fe = _read_treatment_row(
        baseline_dir / "min_wage_violent" / "baseline_fe_summary.csv",
        variable="effective_min_wage",
    )
    property_fe = _read_treatment_row(
        baseline_dir / "min_wage_property" / "baseline_fe_summary.csv",
        variable="effective_min_wage",
    )
    violent_dml = _read_first_json(dml_dir / "min_wage_violent", "dml_*.json")
    property_dml = _read_first_json(dml_dir / "min_wage_property", "dml_*.json")

    lines = [
        "## Interpretation",
        "",
        f"- Minimum-wage TWFE estimates are directionally mixed and statistically weak: violent `{violent_fe['coefficient']:.4f}` (p={violent_fe['p_value']:.4f}) and property `{property_fe['coefficient']:.4f}` (p={property_fe['p_value']:.4f}).",
        f"- The DML stage does not line up cleanly with TWFE: violent `{violent_dml['theta']:.4f}` is positive while property `{property_dml['theta']:.4f}` is much closer to zero after the expanded housing/business control set.",
        "- The practical reading is not “all methods agree”; it is that the main minimum-wage lane remains mixed and should be presented with estimator sensitivity, not as a single settled effect size.",
        "",
    ]
    return lines


def _build_data_expansion_section(
    *,
    app_dir: Path | None,
    report_dir: Path,
) -> list[str]:
    summary = _read_app_summary(app_dir)
    lines = [
        "## Public Data Expansion",
        "",
    ]
    if summary is None:
        lines.extend(["- App summary not available; source-coverage expansion is not summarized here.", ""])
        return lines

    available_sources = {
        row["name"]: row
        for row in summary.get("panel", {}).get("available_sources", [])
    }
    lines.extend(
        [
            "| Source | Non-null Rows | Share | Notes |",
            "|--------|---------------|-------|-------|",
        ]
    )
    for source_name in ("census_cbp", "hud_fmr", "fhfa_hpi", "ukcpr_welfare"):
        row = available_sources.get(source_name)
        if row is None:
            continue
        lines.append(
            f"| {_SOURCE_LABELS[source_name]} | {int(row['non_null_rows']):,} | "
            f"{float(row['share']):.1%} | {', '.join(row['columns'][:3])}{'...' if len(row['columns']) > 3 else ''} |"
        )
    lines.extend(
        [
            "",
            "- The backend is no longer just minimum wage plus SNAP; the panel now includes county business activity, county housing-price pressure, local fair-market rents, and state EITC/TANF policy histories.",
            "",
        ]
    )
    return lines


def _build_policy_lane_section(
    *,
    app_dir: Path | None,
    report_dir: Path,
) -> list[str]:
    summary = _read_app_summary(app_dir)
    lines = [
        "## Additional Policy Lanes",
        "",
    ]
    if summary is None:
        lines.extend(["- App summary not available; additional policy lanes are not summarized here.", ""])
        return lines

    estimands = {row["slug"]: row for row in summary.get("estimands", [])}
    specs = [
        ("eitc_violent", "State EITC Rate -> Violent Crime"),
        ("eitc_property", "State EITC Rate -> Property Crime"),
        ("tanf_violent", "TANF Benefit -> Violent Crime"),
        ("tanf_property", "TANF Benefit -> Property Crime"),
        ("snap_bbce_violent", "SNAP BBCE -> Violent Crime"),
        ("snap_bbce_property", "SNAP BBCE -> Property Crime"),
    ]
    lines.extend(
        [
            "| Lane | FE Coef | FE p-value | Pretrend p-value | DML Theta | DML p-value | Max Abs SMD | Frontend Status |",
            "|------|---------|------------|------------------|-----------|-------------|-------------|-----------------|",
        ]
    )
    for slug, label in specs:
        estimand = estimands.get(slug)
        if estimand is None:
            continue
        baseline = estimand.get("baseline") or {}
        treatment_row = baseline.get("treatment_row") or {}
        pretrend = baseline.get("pretrend") or {}
        dml = estimand.get("dml") or {}
        overlap = estimand.get("overlap") or {}
        frontend = estimand.get("frontend") or {}
        lines.append(
            f"| {label} | {_fmt_stat(treatment_row.get('coefficient'))} | {_fmt_stat(treatment_row.get('p_value'))} | "
            f"{_fmt_stat(pretrend.get('p_value'))} | {_fmt_stat(dml.get('theta'))} | {_fmt_stat(dml.get('p_value'))} | "
            f"{_fmt_stat(overlap.get('max_abs_smd'))} | {frontend.get('status', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "- EITC is the only serious non-minimum-wage lane right now: FE pretrends pass, but FE remains weak while DML is large and positive, so it should be treated as secondary and method-sensitive rather than headline-clean.",
            "- TANF is low-signal and has the weakest overlap/support diagnostics in the project, so it should remain exploratory.",
            "- SNAP BBCE remains exploratory because both event-study pretrend tests fail.",
            f"- Frontend summary contract: [{(app_dir / 'results_summary.json').name}]({_relative_link(app_dir / 'results_summary.json', report_dir)})" if app_dir and (app_dir / "results_summary.json").exists() else "- Frontend summary contract: n/a",
            "",
        ]
    )
    return lines


def _build_bidirectional_section(
    *,
    app_dir: Path | None,
    report_dir: Path,
) -> list[str]:
    summary = _read_bidirectional_summary(app_dir)
    lines = [
        "## Bidirectional Poverty-Crime Check",
        "",
    ]
    if summary is None:
        lines.extend(["- Bidirectional poverty/crime summary not available.", ""])
        return lines

    lines.extend(
        [
            "| Lane | FE Coef | FE p-value | DML Theta | DML p-value | Max Abs SMD | Headline |",
            "|------|---------|------------|-----------|-------------|-------------|----------|",
        ]
    )
    for row in summary.get("estimands", []):
        dml = row.get("dml") or {}
        overlap = row.get("overlap") or {}
        fe = row.get("baseline_fe") or {}
        lines.append(
            f"| {row.get('title', row.get('label', 'n/a'))} | {_fmt_stat(fe.get('coefficient'))} | "
            f"{_fmt_stat(fe.get('p_value'))} | {_fmt_stat(dml.get('theta'))} | {_fmt_stat(dml.get('p_value'))} | "
            f"{_fmt_stat(overlap.get('max_abs_smd'))} | {row.get('headline', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "- This section is still exploratory, but it puts `poverty -> crime` and `crime -> poverty` on much closer methodological footing than the earlier one-off reverse-direction FE check.",
            f"- Bidirectional summary: [{'bidirectional_summary.json'}]({_relative_link(app_dir.parent / 'exploratory' / 'bidirectional_poverty_crime' / 'bidirectional_summary.json', report_dir)})" if app_dir and (app_dir.parent / "exploratory" / "bidirectional_poverty_crime" / "bidirectional_summary.json").exists() else "- Bidirectional summary: n/a",
            "",
        ]
    )
    return lines


def _build_border_section(*, report_dir: Path) -> list[str]:
    summary_path = report_dir.parent / "border" / "border_summary.csv"
    lines = [
        "## Border-County Design",
        "",
    ]
    if not summary_path.exists():
        lines.extend(["- Border-county design outputs were not generated.", ""])
        return lines

    summary = pd.read_csv(summary_path)
    lines.extend(
        [
            "| Label | Spec | Coef | p-value | Pairs |",
            "|-------|------|------|---------|-------|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['spec']} | {row['coefficient']:.4f} | {row['p_value']:.4f} | {int(row['n_entities']):,} |"
        )
    lines.extend(
        [
            "",
            "- This is the strongest new identification check in the backend: adjacent cross-state counties are compared within border pairs over time.",
            "- For minimum wage and EITC, the border estimates are less supportive of strong headline effects than the broader national ML results.",
            f"- Border summary: [{summary_path.name}]({_relative_link(summary_path, report_dir)})",
            "",
        ]
    )
    return lines


def _build_min_wage_identification_section(*, report_dir: Path) -> list[str]:
    summary_path = report_dir.parent / "min_wage_identification" / "min_wage_identification_summary.csv"
    event_path = report_dir.parent / "min_wage_identification" / "min_wage_event_study_summary.csv"
    dose_path = report_dir.parent / "min_wage_identification" / "min_wage_dose_bucket_summary.csv"
    neg_path = report_dir.parent / "min_wage_identification" / "min_wage_negative_control_treatment_summary.csv"
    lines = [
        "## Minimum Wage Identification Redesign",
        "",
    ]
    if not summary_path.exists():
        lines.extend(["- Minimum-wage redesign outputs were not generated.", ""])
        return lines

    summary = pd.read_csv(summary_path)
    lines.extend(
        [
            "| Outcome | Spec | Coef | p-value | Pairs |",
            "|---------|------|------|---------|-------|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['outcome_label']} | {row['spec']} | {row['coefficient']:.4f} | "
            f"{row['p_value']:.4f} | {int(row['n_entities']):,} |"
        )
    lines.extend(
        [
            "",
            "- This redesign tightens identification for minimum wage by using only adjacent cross-state county pairs and then identifying from within-pair changes, not level differences.",
            "- The key stronger spec is the border-pair first-difference model, with placebo and one-sided shock restrictions to isolate years when only one side of the border changes minimum wage.",
            f"- Border-pair event-study summary: [{event_path.name}]({_relative_link(event_path, report_dir)})" if event_path.exists() else "- Border-pair event-study summary: n/a",
            f"- Dose-bucket summary: [{dose_path.name}]({_relative_link(dose_path, report_dir)})" if dose_path.exists() else "- Dose-bucket summary: n/a",
            f"- Negative-control treatment summary: [{neg_path.name}]({_relative_link(neg_path, report_dir)})" if neg_path.exists() else "- Negative-control treatment summary: n/a",
            f"- Minimum-wage redesign summary: [{summary_path.name}]({_relative_link(summary_path, report_dir)})",
            "",
        ]
    )
    return lines


def _build_crime_validation_section(*, report_dir: Path) -> list[str]:
    summary_path = report_dir.parent / "crime_validation" / "crime_measurement_validation.json"
    md_path = report_dir.parent / "crime_validation" / "crime_measurement_validation.md"
    lines = [
        "## Crime Measurement Validation",
        "",
    ]
    if not summary_path.exists():
        lines.extend(["- Crime measurement validation outputs were not generated.", ""])
        return lines

    summary = json.loads(summary_path.read_text())
    lines.extend(
        [
            f"- External benchmark available: {summary.get('external_benchmark_available')}",
            f"- Note: {summary.get('note', 'n/a')}",
        ]
    )
    for row in summary.get("robustness_sensitivity", []):
        lines.append(
            f"- {row.get('label')}: coefficient range across coverage rules is {_fmt_stat(row.get('coef_range'))}; "
            f"sign flip = {row.get('sign_flip')}."
        )
    lines.extend(
        [
            f"- Crime measurement validation: [{md_path.name}]({_relative_link(md_path, report_dir)})" if md_path.exists() else "- Crime measurement validation: n/a",
            "",
        ]
    )
    return lines


def _build_support_trim_section(*, report_dir: Path) -> list[str]:
    summary_path = report_dir.parent / "support_trim" / "support_trim_summary.csv"
    lines = [
        "## Support-Trimmed DML",
        "",
    ]
    if not summary_path.exists():
        lines.extend(["- Support-trimmed DML outputs were not generated.", ""])
        return lines

    summary = pd.read_csv(summary_path)
    lines.extend(
        [
            "| Label | Theta Base | Theta Trimmed | p Base | p Trimmed |",
            "|-------|------------|---------------|--------|-----------|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['theta_base']:.4f} | {row['theta_trimmed']:.4f} | "
            f"{row['p_value_base']:.4f} | {row['p_value_trimmed']:.4f} |"
        )
    lines.extend(
        [
            "",
            "- Trimming to the central predicted-treatment support region materially shrinks the EITC DML signal and leaves the minimum-wage DML signal directionally intact.",
            f"- Support-trim summary: [{summary_path.name}]({_relative_link(summary_path, report_dir)})",
            "",
        ]
    )
    return lines


def _build_falsification_section(*, report_dir: Path) -> list[str]:
    summary_path = report_dir.parent / "falsification" / "negative_control_summary.csv"
    lines = [
        "## Negative-Control Outcomes",
        "",
    ]
    if not summary_path.exists():
        lines.extend(["- Negative-control outcome outputs were not generated.", ""])
        return lines

    summary = pd.read_csv(summary_path)
    lines.extend(
        [
            "| Label | Coef | p-value |",
            "|-------|------|---------|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['coefficient']:.4f} | {row['p_value']:.4f} |"
        )
    lines.extend(
        [
            "",
            "- These falsification checks are cautionary rather than comforting: several policy lanes also move slow-moving demographic outcomes.",
            f"- Negative-control summary: [{summary_path.name}]({_relative_link(summary_path, report_dir)})",
            "",
        ]
    )
    return lines


def _build_staggered_section(*, report_dir: Path) -> list[str]:
    summary_path = report_dir.parent / "staggered" / "staggered_summary.csv"
    lines = [
        "## Staggered-Adoption ATT",
        "",
    ]
    if not summary_path.exists():
        lines.extend(["- Staggered-adoption ATT outputs were not generated.", ""])
        return lines

    summary = pd.read_csv(summary_path)
    lines.extend(
        [
            "| Label | Interpretable | Pretrend p-value | Pretrend Pass | Pre Coefs |",
            "|-------|---------------|------------------|---------------|-----------|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {bool(row.get('interpretable', False))} | {_fmt_stat(row.get('pretrend_p_value'))} | "
            f"{bool(row['pretrend_pass'])} | {int(row.get('n_pre_coefs', 0))} |"
        )
    lines.extend(
        [
            "",
            "- This stacked not-yet-treated event-study is a defensibility supplement to the generic TWFE/event-study lane.",
            "- In the current data, the SNAP violent lane looks usable, SNAP property still fails pretrends, and the minimum-wage stacked specification is not interpretable because pre-period coefficients are absorbed in this sample.",
            f"- Staggered summary: [{summary_path.name}]({_relative_link(summary_path, report_dir)})",
            "",
        ]
    )
    return lines


def _build_overlap_section(
    *,
    overlap_dir: Path | None,
    report_dir: Path,
) -> list[str]:
    lines = [
        "## ML Support Diagnostics",
        "",
    ]
    if overlap_dir is None or not overlap_dir.exists():
        lines.extend(["- Overlap diagnostics were not generated.", ""])
        return lines

    lines.extend(
        [
            "| Outcome | OOF R2 | Residual/Treatment SD | Max Abs SMD |",
            "|---------|--------|------------------------|-------------|",
        ]
    )
    for slug, label in (
        ("min_wage_violent", "Minimum Wage -> Violent Crime"),
        ("min_wage_property", "Minimum Wage -> Property Crime"),
        ("eitc_violent", "State EITC Rate -> Violent Crime"),
        ("eitc_property", "State EITC Rate -> Property Crime"),
        ("tanf_violent", "TANF Benefit -> Violent Crime"),
        ("tanf_property", "TANF Benefit -> Property Crime"),
    ):
        summary_path = overlap_dir / slug / "support_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        lines.append(
            f"| {label} | {summary['oof_r2']:.3f} | "
            f"{summary['residual_to_treatment_std']:.3f} | {summary['max_abs_smd']:.3f} |"
        )
    lines.extend(
        [
            "",
            "- Higher OOF R2 means treatment is more predictable from observables, which reduces practical support for purely model-free extrapolation.",
            "- The max absolute SMD values indicate meaningful imbalance across treatment tails. TANF is the weakest lane on this metric, while minimum wage and EITC are usable but still cautionary.",
            f"- Diagnostic files: [{overlap_dir.name}]({_relative_link(overlap_dir, report_dir)})",
            "",
        ]
    )
    return lines


def _build_snap_section(
    *,
    baseline_dir: Path,
    report_dir: Path,
) -> list[str]:
    snap_specs = [
        ("snap_bbce_violent", "SNAP BBCE -> Violent Crime"),
        ("snap_bbce_property", "SNAP BBCE -> Property Crime"),
    ]
    available = [(slug, label) for slug, label in snap_specs if (baseline_dir / slug / "baseline_fe_summary.csv").exists()]
    lines = [
        "## SNAP Exploratory Lane",
        "",
    ]
    if not available:
        lines.extend(["- SNAP BBCE baseline/event-study outputs were not generated.", ""])
        return lines

    lines.extend(
        [
            "| Outcome | FE Coef | FE p-value | Pretrend p-value |",
            "|---------|---------|------------|------------------|",
        ]
    )
    for slug, label in available:
        row = _read_treatment_row(
            baseline_dir / slug / "baseline_fe_summary.csv",
            variable="broad_based_cat_elig",
        )
        pretrend = _read_json_if_exists(baseline_dir / slug / "event_study_pretrend.json") or {}
        lines.append(
            f"| {label} | {row['coefficient']:.4f} | {row['p_value']:.4f} | {_fmt_stat(pretrend.get('p_value'))} |"
        )
    lines.extend(
        [
            "",
            "- These SNAP results are exploratory only.",
            "- Both SNAP pretrend tests fail, so the event-study lane is useful for context and future work, not for headline causal claims.",
            f"- SNAP baseline outputs: [{baseline_dir.name}]({_relative_link(baseline_dir, report_dir)})",
            "",
        ]
    )
    return lines


def _build_robustness_section(
    *,
    robustness_summary_path: Path,
    report_dir: Path,
) -> list[str]:
    lines = [
        "## Robustness",
        "",
    ]
    if not robustness_summary_path.exists():
        lines.extend(["- Robustness summary not found.", ""])
        return lines

    summary = pd.read_csv(robustness_summary_path)
    lines.extend(
        [
            f"- Summary CSV: [{robustness_summary_path.name}]({_relative_link(robustness_summary_path, report_dir)})",
            "",
            "| Outcome | Spec | Coef | SE | p-value | N used |",
            "|---------|------|------|----|---------|--------|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['spec']} | {row['coefficient']:.4f} | "
            f"{row['std_error']:.4f} | {row['p_value']:.4f} | {int(row['n_obs_used']):,} |"
        )
    lines.extend(
        [
            "",
            "- Coverage sensitivity is represented by `all_rows`, `baseline_high_coverage`, and `strict_coverage`.",
            "- The placebo spec leads the treatment by two years; material effects there would weaken a causal interpretation.",
            "- The county-detrended spec is an approximate county-trend robustness check based on within-county linear detrending.",
            "",
        ]
    )
    return lines


def _fmt_stat(value: float | None) -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(val):
        return "n/a"
    return f"{val:.4f}"


def _relative_link(path: Path, report_dir: Path) -> str:
    return os.path.relpath(Path(path).resolve(), start=Path(report_dir).resolve())


def _read_first_json(directory: Path, pattern: str) -> dict[str, object]:
    path = next(directory.glob(pattern))
    return json.loads(path.read_text())


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_app_summary(app_dir: Path | None) -> dict[str, object] | None:
    if app_dir is None:
        return None
    path = app_dir / "results_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_bidirectional_summary(app_dir: Path | None) -> dict[str, object] | None:
    if app_dir is None:
        return None
    path = app_dir.parent / "exploratory" / "bidirectional_poverty_crime" / "bidirectional_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_treatment_row(path: Path, *, variable: str) -> dict[str, float]:
    summary = pd.read_csv(path)
    row = summary.loc[summary["variable"] == variable]
    if row.empty:
        return summary.iloc[0].to_dict()
    return row.iloc[0].to_dict()
