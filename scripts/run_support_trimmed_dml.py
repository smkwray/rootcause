"""Run support-trimmed DML comparisons for key treatment lanes."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from povcrime.analysis import get_analysis_lanes
from povcrime.config import get_config
from povcrime.models.dml import DMLEstimator
from povcrime.models.overlap import compute_out_of_fold_predictions
from povcrime.models.panel_ml import prepare_panel_ml_sample
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_CONTROLS = [
    "unemployment_rate",
    "poverty_rate",
    "per_capita_personal_income",
    "cbp_employment_per_capita",
    "cbp_establishments_per_1k",
    "rent_to_income_ratio_2br",
    "log_fhfa_hpi_2000_base",
    "log_population",
    "pct_white",
    "pct_black",
    "pct_hispanic",
    "pct_under_18",
    "pct_over_65",
    "pct_hs_or_higher",
    "median_age",
]


def _write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    summary = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    summary.to_csv(output_dir / "support_trim_summary.csv", index=False)
    (output_dir / "support_trim_summary.json").write_text(
        json.dumps({"estimands": rows}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Support-Trimmed DML Summary",
        "",
        "- Design: re-estimate DML after trimming rows outside the central predicted-treatment support region.",
        "",
        "| Label | N base | N trimmed | Theta base | Theta trimmed | p base | p trimmed |",
        "|-------|--------|-----------|------------|---------------|--------|-----------|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {int(row['n_base']):,} | {int(row['n_trimmed']):,} | "
            f"{row['theta_base']:.4f} | {row['theta_trimmed']:.4f} | "
            f"{row['p_value_base']:.4f} | {row['p_value_trimmed']:.4f} |"
        )
    (output_dir / "support_trim_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run support-trimmed DML comparisons.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    parser.add_argument("--n-folds", type=int, default=3, help="Cross-fitting folds.")
    parser.add_argument("--trim-quantile", type=float, default=0.10, help="Lower/upper predicted-treatment trim quantile.")
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    if "low_coverage" in panel.columns:
        panel = panel.loc[~panel["low_coverage"]].copy()

    output_dir = config.output_dir / "support_trim"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for lane in get_analysis_lanes(config=config, method="support_trim"):
        controls = [col for col in _CONTROLS if col in panel.columns]
        cols = ["county_fips", "year", lane.treatment, lane.outcome, *controls]
        sub = panel[cols].dropna().reset_index(drop=True)
        if len(sub) < 200:
            logger.warning("Skipping %s: only %d usable rows.", lane.slug, len(sub))
            continue

        base = DMLEstimator(
            df=sub,
            outcome=lane.outcome,
            treatment=lane.treatment,
            controls=controls,
            group_col="county_fips",
            panel_mode="two_way_within",
            entity_col="county_fips",
            time_col="year",
            n_folds=args.n_folds,
        )
        base.fit()
        base_summary = base.summary()

        overlap_sample = prepare_panel_ml_sample(
            sub,
            model_cols=[lane.treatment, *controls],
            keep_cols=["county_fips", "year"],
            panel_mode="two_way_within",
            entity_col="county_fips",
            time_col="year",
        )
        preds = compute_out_of_fold_predictions(
            overlap_sample[[*controls, "county_fips"]],
            pd.to_numeric(overlap_sample[lane.treatment], errors="coerce"),
            n_splits=args.n_folds,
            group_col="county_fips",
        )
        lower = float(pd.Series(preds).quantile(args.trim_quantile))
        upper = float(pd.Series(preds).quantile(1 - args.trim_quantile))
        trimmed = sub.loc[(preds >= lower) & (preds <= upper)].reset_index(drop=True)
        if len(trimmed) < 200:
            logger.warning("Skipping trimmed DML for %s: only %d rows after trim.", lane.slug, len(trimmed))
            continue

        trimmed_est = DMLEstimator(
            df=trimmed,
            outcome=lane.outcome,
            treatment=lane.treatment,
            controls=controls,
            group_col="county_fips",
            panel_mode="two_way_within",
            entity_col="county_fips",
            time_col="year",
            n_folds=args.n_folds,
        )
        trimmed_est.fit()
        trimmed_summary = trimmed_est.summary()

        spec_dir = output_dir / lane.slug
        spec_dir.mkdir(parents=True, exist_ok=True)
        (spec_dir / "support_trim_result.json").write_text(
            json.dumps(
                {
                    "label": lane.slug,
                    "trim_quantile": args.trim_quantile,
                    "predicted_treatment_lower": lower,
                    "predicted_treatment_upper": upper,
                    "base": base_summary,
                    "trimmed": trimmed_summary,
                    "n_trimmed": int(len(trimmed)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "label": lane.slug,
                "trim_quantile": args.trim_quantile,
                "n_base": int(base_summary["n_obs"]),
                "n_trimmed": int(len(trimmed)),
                "theta_base": float(base_summary["theta"]),
                "theta_trimmed": float(trimmed_summary["theta"]),
                "p_value_base": float(base_summary["p_value"]),
                "p_value_trimmed": float(trimmed_summary["p_value"]),
            }
        )
        logger.info(
            "Support trim %s: theta %.4f -> %.4f; n %d -> %d",
            lane.slug,
            base_summary["theta"],
            trimmed_summary["theta"],
            base_summary["n_obs"],
            len(trimmed),
        )

    if not rows:
        raise SystemExit("No support-trimmed DML specifications were estimated.")

    _write_summary(rows, output_dir)


if __name__ == "__main__":
    main()
