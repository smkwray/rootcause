"""Run stronger border-based identification checks for minimum wage."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from povcrime.config import get_config
from povcrime.data.county_adjacency import CountyAdjacencyAdapter
from povcrime.models.baseline_fe import BaselineFE
from povcrime.models.border import (
    build_border_pair_panel,
    build_first_difference_border_panel,
    canonical_cross_state_pairs,
)
from povcrime.models.event_study import EventStudy
from povcrime.models.robustness import build_placebo_treatment, extract_treatment_row
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_OUTCOMES = [
    {"label": "violent", "outcome": "violent_crime_rate"},
    {"label": "property", "outcome": "property_crime_rate"},
]

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

_DOSE_BUCKETS = [
    ("small", 0.01, 0.5),
    ("medium", 0.5, 1.0),
    ("large", 1.0, None),
]


def _fit_fe(
    *,
    df: pd.DataFrame,
    entity_col: str,
    outcome_col: str,
    treatment_cols: str | list[str],
    controls: list[str],
    output_dir: Path,
    spec_name: str,
) -> tuple[BaselineFE, pd.DataFrame, pd.DataFrame]:
    treatment_list = [treatment_cols] if isinstance(treatment_cols, str) else list(treatment_cols)
    cols_needed = [entity_col, "year", "border_state_pair", outcome_col, *treatment_list, *controls]
    sub = df[cols_needed].dropna(subset=[outcome_col, *treatment_list]).copy()
    if len(sub) < 150:
        raise ValueError(f"Only {len(sub)} usable rows remain for {spec_name}.")

    usable_treatments = [col for col in treatment_list if sub[col].nunique() >= 2]
    if not usable_treatments:
        raise ValueError(f"No treatment variation remains for {spec_name}.")

    model = BaselineFE(
        df=sub.rename(columns={"border_state_pair": "state_fips"}),
        outcome=outcome_col,
        treatment=usable_treatments,
        controls=controls,
        entity_col=entity_col,
        cluster_col="state_fips",
    )
    result = model.fit()
    spec_dir = output_dir / spec_name
    model.save_results(spec_dir)
    summary = model.summary_table()
    return model, result, summary


def _single_treatment_row(
    *,
    summary: pd.DataFrame,
    treatment_col: str,
    spec_name: str,
    outcome_label: str,
    result,
) -> dict[str, object]:
    coef = extract_treatment_row(summary, treatment=treatment_col)
    return {
        "spec": spec_name,
        "sample_rows": int(result.nobs),
        "n_obs_used": int(result.nobs),
        "n_entities": int(result.entity_info["total"]),
        "coefficient": float(coef["coefficient"]),
        "std_error": float(coef["std_error"]),
        "p_value": float(coef["p_value"]),
        "ci_lower": float(coef["ci_lower"]),
        "ci_upper": float(coef["ci_upper"]),
        "outcome_label": outcome_label,
    }


def _add_pair_event_year(
    df: pd.DataFrame,
    *,
    entity_col: str,
    time_col: str,
    shock_col: str,
    output_col: str,
) -> pd.DataFrame:
    out = df.copy()
    first_event = (
        out.loc[out[shock_col], [entity_col, time_col]]
        .groupby(entity_col)[time_col]
        .min()
        .rename(output_col)
    )
    return out.merge(first_event, on=entity_col, how="left")


def _add_dose_buckets(
    df: pd.DataFrame,
    *,
    treatment_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    abs_change = out[treatment_col].abs()
    bucket_cols: list[str] = []
    for label, lower, upper in _DOSE_BUCKETS:
        col = f"{treatment_col}_{label}"
        cond = abs_change >= lower
        if upper is not None:
            cond &= abs_change < upper
        out[col] = cond.astype(float)
        bucket_cols.append(col)
    return out, bucket_cols


def _add_permuted_treatment(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    time_col: str,
    output_col: str,
    seed: int = 42,
) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    out[output_col] = np.nan
    for _, idx in out.groupby(time_col).groups.items():
        group_idx = pd.Index(idx)
        values = out.loc[group_idx, treatment_col].to_numpy(copy=True)
        out.loc[group_idx, output_col] = rng.permutation(values)
    return out


def _fit_event_study(
    *,
    df: pd.DataFrame,
    outcome_col: str,
    event_col: str,
    controls: list[str],
    output_dir: Path,
    spec_name: str,
    outcome_label: str,
) -> dict[str, object]:
    cols_needed = ["pair_id", "year", "border_state_pair", outcome_col, event_col, *controls]
    sub = df[cols_needed].dropna(subset=[outcome_col]).copy()
    if sub[event_col].notna().sum() < 50:
        raise ValueError(f"Too few treated pair-years for {spec_name}.")
    model = EventStudy(
        df=sub.rename(columns={"border_state_pair": "state_fips"}),
        outcome=outcome_col,
        event_col=event_col,
        controls=controls,
        entity_col="pair_id",
        cluster_col="state_fips",
        leads=3,
        lags=4,
    )
    model.fit()
    spec_dir = output_dir / spec_name
    model.save_results(spec_dir)
    pretrend = model.pretrend_test()
    coefs = model.coef_table()
    coef_0 = coefs.loc[coefs["relative_time"] == 0, "coefficient"]
    coef_1 = coefs.loc[coefs["relative_time"] == 1, "coefficient"]
    return {
        "outcome_label": outcome_label,
        "spec": spec_name,
        "n_coefs": int(len(coefs)),
        "n_pre_coefs": int(pretrend["n_pre_coefs"]),
        "pretrend_p_value": float(pretrend["p_value"]),
        "pretrend_pass": bool(pretrend["pass"]),
        "coef_at_0": float(coef_0.iloc[0]) if not coef_0.empty else float("nan"),
        "coef_at_1": float(coef_1.iloc[0]) if not coef_1.empty else float("nan"),
    }


def _write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    summary = pd.DataFrame(rows).sort_values(["outcome_label", "spec"]).reset_index(drop=True)
    summary.to_csv(output_dir / "min_wage_identification_summary.csv", index=False)

    lines = [
        "# Minimum Wage Identification Summary",
        "",
        "- Scope: adjacent cross-state county pairs only.",
        "- Goal: identify minimum-wage effects from within-pair policy divergences, not broad national cross-sectional differences.",
        "- Stronger specs: pair first-differences, placebo leads, and one-sided border shocks where only one side changes minimum wage.",
        "",
        "| Outcome | Spec | Coef | SE | p-value | N used | Pairs |",
        "|---------|------|------|----|---------|--------|-------|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['outcome_label']} | {row['spec']} | {row['coefficient']:.4f} | {row['std_error']:.4f} | "
            f"{row['p_value']:.4f} | {int(row['n_obs_used']):,} | {int(row['n_entities']):,} |"
        )
    (output_dir / "min_wage_identification_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_event_study_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    summary = pd.DataFrame(rows).sort_values("outcome_label").reset_index(drop=True)
    summary.to_csv(output_dir / "min_wage_event_study_summary.csv", index=False)


def _write_dose_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    summary = pd.DataFrame(rows).sort_values(["outcome_label", "dose_bucket"]).reset_index(drop=True)
    summary.to_csv(output_dir / "min_wage_dose_bucket_summary.csv", index=False)


def _write_negative_control_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    summary = pd.DataFrame(rows).sort_values(["outcome_label", "spec"]).reset_index(drop=True)
    summary.to_csv(output_dir / "min_wage_negative_control_treatment_summary.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run stronger border-based minimum-wage identification checks.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    parser.add_argument(
        "--placebo-lead",
        type=int,
        default=1,
        help="Years to lead first-difference treatment for placebo tests.",
    )
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    if "low_coverage" in panel.columns:
        panel = panel.loc[~panel["low_coverage"]].copy()

    adjacency_adapter = CountyAdjacencyAdapter(config)
    if not (config.raw_dir / "county_adjacency" / "county_adjacency.txt").exists():
        adjacency_adapter.download()
    adjacency = adjacency_adapter.validate(adjacency_adapter.load())
    pairs = canonical_cross_state_pairs(adjacency)

    output_dir = config.output_dir / "min_wage_identification"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    dose_rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []

    for outcome_spec in _OUTCOMES:
        controls = [
            col
            for col in _CONTROLS
            if col in panel.columns and col not in {"effective_min_wage", outcome_spec["outcome"]}
        ]
        pair_panel = build_border_pair_panel(
            panel=panel,
            pairs=pairs,
            treatment="effective_min_wage",
            outcome=outcome_spec["outcome"],
            controls=controls,
        )
        if pair_panel.empty:
            logger.warning("Skipping min_wage_%s: no usable border-pair rows.", outcome_spec["label"])
            continue

        pair_outcome = f"diff_{outcome_spec['outcome']}"
        pair_treatment = "diff_effective_min_wage"
        pair_controls = [f"diff_{col}" for col in controls]
        spec_dir = output_dir / outcome_spec["label"]
        spec_dir.mkdir(parents=True, exist_ok=True)

        _, pair_result, pair_summary = _fit_fe(
            df=pair_panel,
            entity_col="pair_id",
            outcome_col=pair_outcome,
            treatment_cols=pair_treatment,
            controls=pair_controls,
            output_dir=spec_dir,
            spec_name="border_pair_fe",
        )
        rows.append(
            _single_treatment_row(
                summary=pair_summary,
                treatment_col=pair_treatment,
                spec_name="border_pair_fe",
                outcome_label=outcome_spec["label"],
                result=pair_result,
            )
        )

        fd_panel = build_first_difference_border_panel(
            pair_panel=pair_panel,
            treatment="effective_min_wage",
            outcome=outcome_spec["outcome"],
            controls=controls,
        )
        fd_outcome = f"delta_{pair_outcome}"
        fd_treatment = f"delta_{pair_treatment}"
        fd_controls = [f"delta_diff_{col}" for col in controls]

        _, fd_result, fd_summary = _fit_fe(
            df=fd_panel,
            entity_col="pair_id",
            outcome_col=fd_outcome,
            treatment_cols=fd_treatment,
            controls=fd_controls,
            output_dir=spec_dir,
            spec_name="border_pair_first_difference",
        )
        rows.append(
            _single_treatment_row(
                summary=fd_summary,
                treatment_col=fd_treatment,
                spec_name="border_pair_first_difference",
                outcome_label=outcome_spec["label"],
                result=fd_result,
            )
        )

        placebo_col = f"{fd_treatment}_placebo_lead_{args.placebo_lead}"
        placebo_panel = build_placebo_treatment(
            fd_panel,
            treatment_col=fd_treatment,
            entity_col="pair_id",
            time_col="year",
            lead_periods=args.placebo_lead,
            output_col=placebo_col,
        )
        _, placebo_result, placebo_summary = _fit_fe(
            df=placebo_panel,
            entity_col="pair_id",
            outcome_col=fd_outcome,
            treatment_cols=placebo_col,
            controls=fd_controls,
            output_dir=spec_dir,
            spec_name="border_pair_first_difference_placebo",
        )
        rows.append(
            _single_treatment_row(
                summary=placebo_summary,
                treatment_col=placebo_col,
                spec_name="border_pair_first_difference_placebo",
                outcome_label=outcome_spec["label"],
                result=placebo_result,
            )
        )

        one_sided = fd_panel.loc[fd_panel["one_sided_treatment_shock"]].copy()
        _, one_sided_result, one_sided_summary = _fit_fe(
            df=one_sided,
            entity_col="pair_id",
            outcome_col=fd_outcome,
            treatment_cols=fd_treatment,
            controls=fd_controls,
            output_dir=spec_dir,
            spec_name="border_pair_first_difference_one_sided",
        )
        rows.append(
            _single_treatment_row(
                summary=one_sided_summary,
                treatment_col=fd_treatment,
                spec_name="border_pair_first_difference_one_sided",
                outcome_label=outcome_spec["label"],
                result=one_sided_result,
            )
        )

        try:
            event_panel = _add_pair_event_year(
                fd_panel,
                entity_col="pair_id",
                time_col="year",
                shock_col="one_sided_treatment_shock",
                output_col="pair_min_wage_event_year",
            )
            event_rows.append(
                _fit_event_study(
                    df=event_panel,
                    outcome_col=fd_outcome,
                    event_col="pair_min_wage_event_year",
                    controls=fd_controls,
                    output_dir=spec_dir,
                    spec_name="border_pair_first_difference_event_study",
                    outcome_label=outcome_spec["label"],
                )
            )
        except Exception:
            logger.exception("Border pair event-study failed for %s.", outcome_spec["label"])

        try:
            dose_panel, dose_treatments = _add_dose_buckets(
                fd_panel,
                treatment_col=fd_treatment,
            )
            _, dose_result, dose_summary = _fit_fe(
                df=dose_panel,
                entity_col="pair_id",
                outcome_col=fd_outcome,
                treatment_cols=dose_treatments,
                controls=fd_controls,
                output_dir=spec_dir,
                spec_name="border_pair_first_difference_dose_buckets",
            )
            treatment_rows = dose_summary.loc[dose_summary["variable"].isin(dose_treatments)].copy()
            for _, row in treatment_rows.iterrows():
                dose_rows.append(
                    {
                        "outcome_label": outcome_spec["label"],
                        "dose_bucket": str(row["variable"]).replace(f"{fd_treatment}_", ""),
                        "coefficient": float(row["coefficient"]),
                        "std_error": float(row["std_error"]),
                        "p_value": float(row["p_value"]),
                        "ci_lower": float(row["ci_lower"]),
                        "ci_upper": float(row["ci_upper"]),
                        "n_obs_used": int(dose_result.nobs),
                        "n_entities": int(dose_result.entity_info["total"]),
                    }
                )
        except Exception:
            logger.exception("Dose bucket spec failed for %s.", outcome_spec["label"])

        try:
            permuted_col = f"{fd_treatment}_permuted"
            permuted_panel = _add_permuted_treatment(
                fd_panel,
                treatment_col=fd_treatment,
                time_col="year",
                output_col=permuted_col,
            )
            _, perm_result, perm_summary = _fit_fe(
                df=permuted_panel,
                entity_col="pair_id",
                outcome_col=fd_outcome,
                treatment_cols=permuted_col,
                controls=fd_controls,
                output_dir=spec_dir,
                spec_name="border_pair_first_difference_negative_control_treatment",
            )
            negative_rows.append(
                _single_treatment_row(
                    summary=perm_summary,
                    treatment_col=permuted_col,
                    spec_name="border_pair_first_difference_negative_control_treatment",
                    outcome_label=outcome_spec["label"],
                    result=perm_result,
                )
            )
        except Exception:
            logger.exception("Negative-control treatment spec failed for %s.", outcome_spec["label"])

        logger.info(
            "Minimum-wage ID %s: pair FE p=%.4f, FD p=%.4f, placebo p=%.4f, one-sided p=%.4f",
            outcome_spec["label"],
            rows[-4]["p_value"],
            rows[-3]["p_value"],
            rows[-2]["p_value"],
            rows[-1]["p_value"],
        )

    if not rows:
        raise SystemExit("No minimum-wage identification specifications could be estimated.")

    _write_summary(rows, output_dir)
    _write_event_study_summary(event_rows, output_dir)
    _write_dose_summary(dose_rows, output_dir)
    _write_negative_control_summary(negative_rows, output_dir)


if __name__ == "__main__":
    main()
