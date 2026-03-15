"""Stacked not-yet-treated event study for staggered adoption."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

logger = logging.getLogger(__name__)


def build_stacked_event_panel(
    df: pd.DataFrame,
    *,
    event_col: str,
    entity_col: str = "county_fips",
    time_col: str = "year",
    leads: int = 4,
    lags: int = 6,
) -> pd.DataFrame:
    """Construct a stacked cohort-window panel with not-yet-treated controls."""
    working = df.copy()
    treated_events = (
        working[[entity_col, event_col]]
        .drop_duplicates()
        .dropna(subset=[event_col])
        .rename(columns={event_col: "cohort_year"})
    )
    treated_events["cohort_year"] = treated_events["cohort_year"].astype(int)
    cohort_years = sorted(treated_events["cohort_year"].unique())

    stacks: list[pd.DataFrame] = []
    for cohort in cohort_years:
        window_start = cohort - leads
        window_end = cohort + lags
        eligible = working.loc[
            (
                (working[event_col] == cohort)
                | working[event_col].isna()
                | (working[event_col] > window_end)
            )
            & (working[time_col].between(window_start, window_end)),
        ].copy()
        if eligible.empty:
            continue
        treated_units = set(
            treated_events.loc[treated_events["cohort_year"] == cohort, entity_col]
        )
        eligible["stack_id"] = cohort
        eligible["is_cohort_treated"] = eligible[entity_col].isin(treated_units).astype(int)
        eligible["relative_time"] = np.where(
            eligible["is_cohort_treated"] == 1,
            eligible[time_col] - cohort,
            np.nan,
        )
        eligible["stack_entity"] = eligible["stack_id"].astype(str) + "__" + eligible[entity_col].astype(str)
        eligible["stack_time"] = eligible["stack_id"].astype(str) + "__" + eligible[time_col].astype(str)
        stacks.append(eligible)

    if not stacks:
        raise ValueError("No stacked cohort windows could be constructed.")
    out = pd.concat(stacks, ignore_index=True)
    logger.info(
        "Stacked event panel built from %d cohorts: %d rows.",
        len(cohort_years),
        len(out),
    )
    return out


def _build_event_time_dummies(
    df: pd.DataFrame,
    *,
    leads: int,
    lags: int,
    ref: int,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    dummy_cols: list[str] = []
    for k in range(-leads, lags + 1):
        if k == ref:
            continue
        col = f"et_{k}" if k < 0 else f"et_plus_{k}"
        out[col] = ((out["relative_time"] == k) & (out["is_cohort_treated"] == 1)).astype(float)
        dummy_cols.append(col)
    return out, dummy_cols


class StaggeredEventStudy:
    """Stacked not-yet-treated event study estimator."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        outcome: str,
        event_col: str,
        controls: list[str] | None = None,
        entity_col: str = "county_fips",
        time_col: str = "year",
        cluster_col: str = "state_fips",
        leads: int = 4,
        lags: int = 6,
        ref: int = -1,
    ) -> None:
        self.outcome = outcome
        self.event_col = event_col
        self.controls = list(controls) if controls else []
        self.entity_col = entity_col
        self.time_col = time_col
        self.cluster_col = cluster_col
        self.leads = leads
        self.lags = lags
        self.ref = ref

        stacked = build_stacked_event_panel(
            df,
            event_col=event_col,
            entity_col=entity_col,
            time_col=time_col,
            leads=leads,
            lags=lags,
        )
        self._df, self._et_cols = _build_event_time_dummies(
            stacked,
            leads=leads,
            lags=lags,
            ref=ref,
        )
        self._result = None

    def fit(self):
        regressors = self._et_cols + self.controls
        sub = self._df.dropna(subset=[self.outcome, *regressors]).copy()
        if sub.empty:
            raise ValueError("No observations remain after dropping missing values.")
        sub["stack_time_fe"] = sub["stack_id"].astype(str) + "__" + sub[self.time_col].astype(int).astype(str)
        sub = sub.set_index(["stack_entity", self.time_col])
        usable_events = [col for col in self._et_cols if sub[col].nunique() >= 2]
        usable_controls = [col for col in self.controls if sub[col].nunique() >= 2]
        time_fe = pd.get_dummies(sub["stack_time_fe"], prefix="stack_time_fe", dtype=float)
        if time_fe.shape[1] > 1:
            time_fe = time_fe.iloc[:, 1:]
        else:
            time_fe = time_fe.iloc[:, 0:0]
        x = pd.concat([sub[usable_events + usable_controls], time_fe], axis=1)
        model = PanelOLS(
            sub[self.outcome],
            x,
            entity_effects=True,
            drop_absorbed=True,
            check_rank=False,
        )
        self._et_cols = usable_events
        self._result = model.fit(
            cov_type="clustered",
            clusters=sub[[self.cluster_col]],
        )
        return self._result

    def coef_table(self) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("Call fit() before coef_table().")
        ci = self._result.conf_int()
        rows: list[dict[str, float]] = []
        for col in self._et_cols:
            rel = int(col.replace("et_plus_", "")) if col.startswith("et_plus_") else int(col.replace("et_", ""))
            rows.append(
                {
                    "relative_time": rel,
                    "coefficient": float(self._result.params[col]),
                    "std_error": float(self._result.std_errors[col]),
                    "ci_lower": float(ci.loc[col].iloc[0]),
                    "ci_upper": float(ci.loc[col].iloc[1]),
                    "p_value": float(self._result.pvalues[col]),
                }
            )
        rows.append(
            {
                "relative_time": self.ref,
                "coefficient": 0.0,
                "std_error": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "p_value": float("nan"),
            }
        )
        return pd.DataFrame(rows).sort_values("relative_time").reset_index(drop=True)

    def pretrend_test(self) -> dict[str, Any]:
        if self._result is None:
            raise RuntimeError("Call fit() before pretrend_test().")
        param_names = list(self._result.params.index)
        pre_cols = [
            c
            for c in self._et_cols
            if c.startswith("et_") and not c.startswith("et_plus_") and c in param_names
        ]
        if not pre_cols:
            return {
                "f_stat": float("nan"),
                "p_value": float("nan"),
                "n_pre_coefs": 0,
                "pass": False,
                "interpretable": False,
            }
        n_params = len(param_names)
        n_pre = len(pre_cols)
        r = np.zeros((n_pre, n_params))
        for i, col in enumerate(pre_cols):
            r[i, param_names.index(col)] = 1.0
        beta = self._result.params.values
        v = np.array(self._result.cov)
        rvrt = r @ v @ r.T
        try:
            inv = np.linalg.inv(rvrt)
            f_stat = float((r @ beta) @ inv @ (r @ beta) / n_pre)
        except np.linalg.LinAlgError:
            f_stat = float("nan")
        from scipy import stats

        if np.isnan(f_stat):
            p_value = float("nan")
        else:
            dof2 = int(self._result.nobs - n_params)
            p_value = float(1.0 - stats.f.cdf(f_stat, n_pre, max(dof2, 1)))
        interpretable = not np.isnan(p_value)
        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "n_pre_coefs": n_pre,
            "pass": bool(p_value > 0.05) if interpretable else False,
            "interpretable": interpretable,
        }

    def plot(self, output_path: str | Path | None = None) -> plt.Figure:
        coefs = self.coef_table()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(
            coefs["relative_time"],
            coefs["coefficient"],
            yerr=[coefs["coefficient"] - coefs["ci_lower"], coefs["ci_upper"] - coefs["coefficient"]],
            fmt="o-",
            capsize=3,
            color="darkgreen",
            markerfacecolor="white",
            markeredgecolor="darkgreen",
            linewidth=1.5,
        )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(self.ref + 0.5, color="red", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Relative time")
        ax.set_ylabel(f"Coefficient ({self.outcome})")
        ax.set_title("Stacked Not-Yet-Treated Event Study")
        fig.tight_layout()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    def save_results(self, output_dir: str | Path) -> None:
        if self._result is None:
            raise RuntimeError("Call fit() before save_results().")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.coef_table().to_csv(output_dir / "staggered_event_study_coefs.csv", index=False)
        with open(output_dir / "staggered_event_study_pretrend.json", "w") as fh:
            json.dump(self.pretrend_test(), fh, indent=2)
        fig = self.plot(output_dir / "staggered_event_study_plot.png")
        plt.close(fig)
