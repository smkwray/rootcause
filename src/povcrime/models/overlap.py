"""Support diagnostics for continuous-treatment ML analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

from povcrime.models.panel_ml import PanelMode, prepare_panel_ml_sample, validate_panel_mode


def build_continuous_treatment_support_diagnostics(
    *,
    df: pd.DataFrame,
    treatment: str,
    controls: list[str],
    output_dir: str | Path,
    n_splits: int = 5,
    group_col: str | None = None,
    panel_mode: PanelMode = "none",
    entity_col: str = "county_fips",
    time_col: str = "year",
) -> dict[str, float | int | str]:
    """Build lightweight overlap/support diagnostics for a continuous treatment."""
    normalized_mode = validate_panel_mode(panel_mode)
    keep_cols: list[str] = []
    if group_col is not None:
        keep_cols.append(group_col)
    if normalized_mode != "none":
        keep_cols.extend([entity_col, time_col])

    sample = prepare_panel_ml_sample(
        df,
        model_cols=[treatment, *controls],
        keep_cols=keep_cols,
        panel_mode=normalized_mode,
        entity_col=entity_col,
        time_col=time_col,
    )
    if len(sample) < max(50, n_splits * 10):
        raise ValueError(
            f"Only {len(sample)} complete observations available for diagnostics."
        )

    X = sample[controls]
    y = pd.to_numeric(sample[treatment], errors="coerce")

    oof_pred = compute_out_of_fold_predictions(
        sample[[*controls, group_col]] if group_col is not None else X,
        y,
        n_splits=n_splits,
        group_col=group_col,
    )
    residual = y - oof_pred

    balance = _control_balance_by_treatment_tails(sample, treatment=treatment, controls=controls)
    support_bins = _support_bins(sample.assign(predicted_treatment=oof_pred), treatment=treatment)

    treatment_std = float(y.std(ddof=0))
    residual_std = float(residual.std(ddof=0))
    summary: dict[str, float | int | str] = {
        "treatment": treatment,
        "n_obs": int(len(sample)),
        "n_controls": int(len(controls)),
        "treatment_min": float(y.min()),
        "treatment_p05": float(y.quantile(0.05)),
        "treatment_median": float(y.median()),
        "treatment_p95": float(y.quantile(0.95)),
        "treatment_max": float(y.max()),
        "treatment_std": treatment_std,
        "oof_r2": float(r2_score(y, oof_pred)),
        "residual_std": residual_std,
        "residual_to_treatment_std": float(residual_std / treatment_std)
        if treatment_std > 0
        else float("nan"),
        "tail_low_n": int(balance["group_low"].iloc[0]) if not balance.empty else 0,
        "tail_high_n": int(balance["group_high"].iloc[0]) if not balance.empty else 0,
        "max_abs_smd": float(balance["abs_smd"].max()) if not balance.empty else float("nan"),
        "n_controls_abs_smd_gt_0_10": int((balance["abs_smd"] > 0.10).sum()) if not balance.empty else 0,
        "n_controls_abs_smd_gt_0_25": int((balance["abs_smd"] > 0.25).sum()) if not balance.empty else 0,
        "panel_mode": normalized_mode,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "support_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    balance.to_csv(output_dir / "control_balance.csv", index=False)
    support_bins.to_csv(output_dir / "support_bins.csv", index=False)
    return summary


def compute_out_of_fold_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    group_col: str | None = None,
) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        max_iter=150,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=50,
        early_stopping=True,
        random_state=42,
    )
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    feature_cols = [col for col in X.columns if col != group_col]
    if not feature_cols:
        raise ValueError("At least one feature column is required for OOF predictions.")

    if group_col is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X[feature_cols])
    else:
        if group_col not in X.columns:
            raise ValueError(f"Missing grouping column for OOF predictions: {group_col}")
        groups = X[group_col]
        if int(groups.nunique(dropna=False)) < n_splits:
            raise ValueError(
                f"Only {int(groups.nunique(dropna=False))} unique groups available in {group_col}, "
                f"need at least {n_splits} for grouped OOF predictions."
            )
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X[feature_cols], y, groups=groups)

    preds = np.empty(len(y), dtype=float)
    for train_idx, test_idx in split_iter:
        fitted = clone(model).fit(X.iloc[train_idx][feature_cols], y.iloc[train_idx])
        preds[test_idx] = fitted.predict(X.iloc[test_idx][feature_cols])
    return preds


def _control_balance_by_treatment_tails(
    df: pd.DataFrame,
    *,
    treatment: str,
    controls: list[str],
) -> pd.DataFrame:
    low_cut = df[treatment].quantile(0.25)
    high_cut = df[treatment].quantile(0.75)
    low = df.loc[df[treatment] <= low_cut, controls]
    high = df.loc[df[treatment] >= high_cut, controls]

    rows: list[dict[str, float | str | int]] = []
    for control in controls:
        low_mean = float(low[control].mean())
        high_mean = float(high[control].mean())
        low_var = float(low[control].var(ddof=0))
        high_var = float(high[control].var(ddof=0))
        pooled_sd = np.sqrt((low_var + high_var) / 2)
        smd = (high_mean - low_mean) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append(
            {
                "control": control,
                "low_mean": low_mean,
                "high_mean": high_mean,
                "smd": float(smd),
                "abs_smd": float(abs(smd)),
                "group_low": int(len(low)),
                "group_high": int(len(high)),
            }
        )

    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)


def _support_bins(
    df: pd.DataFrame,
    *,
    treatment: str,
) -> pd.DataFrame:
    ranked = df.copy()
    ranked["predicted_bin"] = pd.qcut(
        ranked["predicted_treatment"],
        q=min(10, ranked["predicted_treatment"].nunique()),
        duplicates="drop",
    )
    summary = (
        ranked.groupby("predicted_bin", observed=False)[treatment]
        .agg(["count", "min", "median", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n_obs",
                "min": "actual_min",
                "median": "actual_median",
                "max": "actual_max",
            }
        )
    )
    summary["predicted_bin"] = summary["predicted_bin"].astype(str)
    return summary
