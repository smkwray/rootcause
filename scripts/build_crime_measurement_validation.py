"""Build a crime-measurement validation appendix from existing artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from povcrime.config import get_config
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_OUTCOMES = [
    ("violent_crime_rate", "violent"),
    ("property_crime_rate", "property"),
]


def _mean_county_autocorr(df: pd.DataFrame, outcome: str) -> float | None:
    vals: list[float] = []
    for _, sub in df.groupby("county_fips"):
        series = sub.sort_values("year")[outcome].dropna()
        if len(series) < 3:
            continue
        corr = series.autocorr(lag=1)
        if pd.notna(corr):
            vals.append(float(corr))
    if not vals:
        return None
    return float(np.mean(vals))


def _coverage_summary(panel: pd.DataFrame) -> list[dict[str, object]]:
    if "low_coverage" not in panel.columns:
        panel = panel.copy()
        panel["low_coverage"] = False
    rows: list[dict[str, object]] = []
    groups = [
        ("high_coverage", panel.loc[~panel["low_coverage"]].copy()),
        ("low_coverage", panel.loc[panel["low_coverage"]].copy()),
    ]
    for group_name, sub in groups:
        if sub.empty:
            continue
        row: dict[str, object] = {
            "coverage_group": group_name,
            "rows": int(len(sub)),
            "counties": int(sub["county_fips"].nunique()),
            "mean_source_share": float(sub["source_share"].mean()) if "source_share" in sub.columns else None,
        }
        for outcome, label in _OUTCOMES:
            if outcome not in sub.columns:
                continue
            row[f"{label}_mean"] = float(sub[outcome].mean())
            row[f"{label}_std"] = float(sub[outcome].std())
            row[f"{label}_lag1_autocorr"] = _mean_county_autocorr(sub, outcome)
        rows.append(row)
    return rows


def _robustness_sensitivity(output_dir: Path) -> list[dict[str, object]]:
    path = output_dir / "robustness" / "robustness_summary.csv"
    if not path.exists():
        return []
    summary = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for label in ("min_wage_violent", "min_wage_property"):
        sub = summary.loc[
            (summary["label"] == label)
            & (summary["spec"].isin(["baseline_high_coverage", "all_rows", "strict_coverage"]))
        ].copy()
        if sub.empty:
            continue
        coef_range = float(sub["coefficient"].max() - sub["coefficient"].min())
        signs = {int(np.sign(val)) for val in sub["coefficient"].dropna()}
        rows.append(
            {
                "label": label,
                "n_specs": int(len(sub)),
                "coef_range": coef_range,
                "sign_flip": bool(len(signs - {0}) > 1),
                "baseline_high_coverage_coef": _lookup_coef(sub, "baseline_high_coverage"),
                "all_rows_coef": _lookup_coef(sub, "all_rows"),
                "strict_coverage_coef": _lookup_coef(sub, "strict_coverage"),
            }
        )
    return rows


def _lookup_coef(summary: pd.DataFrame, spec: str) -> float | None:
    row = summary.loc[summary["spec"] == spec, "coefficient"]
    if row.empty:
        return None
    return float(row.iloc[0])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a crime-measurement validation appendix.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)

    output_dir = config.output_dir / "crime_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows = _coverage_summary(panel)
    sensitivity_rows = _robustness_sensitivity(config.output_dir)
    payload = {
        "generated_date": pd.Timestamp.now().date().isoformat(),
        "external_benchmark_available": False,
        "note": (
            "No second national county-year public crime benchmark is wired into this repo. "
            "This appendix validates the current FBI fallback through coverage sensitivity and "
            "outcome-stability checks instead."
        ),
        "coverage_groups": coverage_rows,
        "robustness_sensitivity": sensitivity_rows,
    }

    with open(output_dir / "crime_measurement_validation.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    pd.DataFrame(coverage_rows).to_csv(output_dir / "crime_measurement_coverage_summary.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(output_dir / "crime_measurement_robustness_summary.csv", index=False)

    lines = [
        "# Crime Measurement Validation",
        "",
        f"- External benchmark available: {payload['external_benchmark_available']}",
        f"- Note: {payload['note']}",
        "",
        "## Coverage Groups",
        "",
        "| Group | Rows | Counties | Mean Source Share | Violent Mean | Violent Lag1 AC | Property Mean | Property Lag1 AC |",
        "|-------|------|----------|-------------------|--------------|-----------------|---------------|------------------|",
    ]
    for row in coverage_rows:
        lines.append(
            f"| {row['coverage_group']} | {row['rows']:,} | {row['counties']:,} | {_fmt(row.get('mean_source_share'))} | "
            f"{_fmt(row.get('violent_mean'))} | {_fmt(row.get('violent_lag1_autocorr'))} | "
            f"{_fmt(row.get('property_mean'))} | {_fmt(row.get('property_lag1_autocorr'))} |"
        )
    lines.extend(
        [
            "",
            "## Minimum Wage Robustness Across Coverage Rules",
            "",
            "| Label | Coef Range | Sign Flip | High Coverage | All Rows | Strict Coverage |",
            "|-------|------------|-----------|---------------|----------|-----------------|",
        ]
    )
    for row in sensitivity_rows:
        lines.append(
            f"| {row['label']} | {_fmt(row['coef_range'])} | {row['sign_flip']} | "
            f"{_fmt(row['baseline_high_coverage_coef'])} | {_fmt(row['all_rows_coef'])} | {_fmt(row['strict_coverage_coef'])} |"
        )
    (output_dir / "crime_measurement_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Crime measurement validation written to %s", output_dir)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.4f}"


if __name__ == "__main__":
    main()
