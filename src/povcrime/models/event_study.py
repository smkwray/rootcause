"""Event-study estimator for dynamic treatment effects.

Estimates lead-lag coefficients around treatment adoption to visualise
pre-trends and dynamic treatment effects using the two-way fixed effects
framework.  Wraps :class:`linearmodels.panel.PanelOLS`.
"""

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


def _build_event_time_dummies(
    df: pd.DataFrame,
    event_col: str,
    time_col: str,
    *,
    leads: int = 4,
    lags: int = 6,
    ref: int = -1,
) -> tuple[pd.DataFrame, list[str]]:
    """Create event-time dummy columns.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame with *event_col* (year treatment first adopted,
        ``NaN`` for never-treated) and *time_col* (calendar year).
    event_col : str
        Column containing the year of treatment adoption (or NaN).
    time_col : str
        Calendar time column.
    leads : int
        Number of pre-treatment periods to include (positive integer).
    lags : int
        Number of post-treatment periods to include (positive integer).
    ref : int
        Reference (omitted) relative-time period (default ``-1``).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        DataFrame augmented with event-time dummies and a list of the
        dummy column names (excluding the reference period).
    """
    df = df.copy()
    df["_event_time"] = df[time_col] - df[event_col]

    dummy_cols: list[str] = []
    for k in range(-leads, lags + 1):
        if k == ref:
            continue
        col_name = f"et_{k}" if k < 0 else f"et_plus_{k}"
        df[col_name] = (df["_event_time"] == k).astype(float)
        # For never-treated units, event_time is NaN -> dummies are 0
        df.loc[df[event_col].isna(), col_name] = 0.0
        dummy_cols.append(col_name)

    return df, dummy_cols


class EventStudy:
    """Event-study estimator around treatment adoption.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame.
    outcome : str
        Dependent variable column.
    event_col : str
        Column containing the year the unit first received treatment
        (``NaN`` for never-treated units).
    controls : list[str] | None
        Additional control variables.
    entity_col : str
        Panel entity identifier (default ``"county_fips"``).
    time_col : str
        Time period identifier (default ``"year"``).
    cluster_col : str
        Clustering variable for standard errors (default ``"state_fips"``).
    leads : int
        Number of pre-treatment periods (default 4).
    lags : int
        Number of post-treatment periods (default 6).
    ref : int
        Omitted reference period (default ``-1``).
    """

    def __init__(
        self,
        df: pd.DataFrame,
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

        # Build event-time dummies
        self._df, self._et_cols = _build_event_time_dummies(
            df,
            event_col=event_col,
            time_col=time_col,
            leads=leads,
            lags=lags,
            ref=ref,
        )
        self._regressors = self._et_cols + self.controls
        self._result = None

        logger.info(
            "EventStudy initialised: outcome=%s, event_col=%s, "
            "%d event-time dummies (leads=%d, lags=%d, ref=%d), n_obs=%d.",
            outcome,
            event_col,
            len(self._et_cols),
            leads,
            lags,
            ref,
            len(self._df),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        """Fit the event-study model via PanelOLS.

        Returns
        -------
        linearmodels.panel.results.PanelEffectsResults
            Fitted model result.
        """
        df = self._df.dropna(
            subset=[self.outcome] + self._regressors
        ).copy()

        if df.empty:
            raise ValueError("No observations remain after dropping NaN rows.")

        df = df.set_index([self.entity_col, self.time_col])

        y = df[self.outcome]
        usable_event_cols = [col for col in self._et_cols if df[col].nunique() >= 2]
        dropped_event_cols = [col for col in self._et_cols if col not in usable_event_cols]
        if dropped_event_cols:
            logger.warning(
                "Dropping unsupported event-time dummies with no variation: %s",
                dropped_event_cols,
            )
        self._et_cols = usable_event_cols

        usable_controls = [col for col in self.controls if df[col].nunique() >= 2]
        dropped_controls = [col for col in self.controls if col not in usable_controls]
        if dropped_controls:
            logger.warning(
                "Dropping zero-variance event-study controls: %s",
                dropped_controls,
            )

        regressors = self._et_cols + usable_controls
        x = df[regressors]
        cluster = df[[self.cluster_col]]

        model = PanelOLS(
            y,
            x,
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True,
            check_rank=False,
        )
        self._result = model.fit(
            cov_type="clustered",
            clusters=cluster,
        )

        logger.info(
            "EventStudy fitted: R-sq (within)=%.4f, n_obs=%d.",
            self._result.rsquared_within,
            self._result.nobs,
        )
        return self._result

    def coef_table(self) -> pd.DataFrame:
        """Return event-time coefficients as a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``relative_time``, ``coefficient``, ``std_error``,
            ``ci_lower``, ``ci_upper``, ``p_value``.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before coef_table().")

        res = self._result
        ci = res.conf_int()
        rows = []
        for col in self._et_cols:
            if col not in res.params.index:
                continue
            if col.startswith("et_plus_"):
                k = int(col.replace("et_plus_", ""))
            else:
                k = int(col.replace("et_", ""))
            rows.append(
                {
                    "relative_time": k,
                    "coefficient": res.params[col],
                    "std_error": res.std_errors[col],
                    "ci_lower": ci.loc[col].iloc[0],
                    "ci_upper": ci.loc[col].iloc[1],
                    "p_value": res.pvalues[col],
                }
            )
        # Add the reference period as zero
        rows.append(
            {
                "relative_time": self.ref,
                "coefficient": 0.0,
                "std_error": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "p_value": np.nan,
            }
        )
        table = pd.DataFrame(rows).sort_values("relative_time").reset_index(drop=True)
        return table

    def pretrend_test(self) -> dict[str, Any]:
        """Joint F-test that all pre-treatment coefficients equal zero.

        Returns
        -------
        dict
            Keys: ``f_stat``, ``p_value``, ``n_pre_coefs``, ``pass``
            (True if p > 0.05, i.e. we fail to reject the null of no
            pre-trends at the 5% level).
        """
        if self._result is None:
            raise RuntimeError("Call fit() before pretrend_test().")

        res = self._result
        # Identify pre-treatment dummies (negative relative time)
        pre_cols = [
            c for c in self._et_cols
            if c.startswith("et_")
            and not c.startswith("et_plus_")
            and c in res.params.index
        ]
        if not pre_cols:
            return {
                "f_stat": float("nan"),
                "p_value": float("nan"),
                "n_pre_coefs": 0,
                "pass": True,
            }

        # Build the restriction matrix R such that R @ beta = 0
        param_names = list(res.params.index)
        n_params = len(param_names)
        n_pre = len(pre_cols)
        R = np.zeros((n_pre, n_params))
        for i, col in enumerate(pre_cols):
            j = param_names.index(col)
            R[i, j] = 1.0

        beta = res.params.values
        V = np.array(res.cov)

        Rb = R @ beta
        RVR = R @ V @ R.T

        try:
            inv_RVR = np.linalg.inv(RVR)
            f_stat = float((Rb @ inv_RVR @ Rb) / n_pre)
        except np.linalg.LinAlgError:
            f_stat = float("nan")

        # Approximate p-value from F distribution
        from scipy import stats

        if np.isnan(f_stat):
            p_value = float("nan")
        else:
            dof2 = int(res.nobs - n_params)
            p_value = float(1.0 - stats.f.cdf(f_stat, n_pre, max(dof2, 1)))

        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "n_pre_coefs": n_pre,
            "pass": bool(p_value > 0.05) if not np.isnan(p_value) else True,
        }

    def plot(self, output_path: str | Path | None = None) -> plt.Figure:
        """Plot event-study coefficients with 95% CI.

        Parameters
        ----------
        output_path : str | Path | None
            If provided, saves the plot to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        coefs = self.coef_table()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(
            coefs["relative_time"],
            coefs["coefficient"],
            yerr=[
                coefs["coefficient"] - coefs["ci_lower"],
                coefs["ci_upper"] - coefs["coefficient"],
            ],
            fmt="o-",
            capsize=3,
            color="steelblue",
            markerfacecolor="white",
            markeredgecolor="steelblue",
            linewidth=1.5,
        )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(
            self.ref + 0.5, color="red", linestyle=":", linewidth=0.8, alpha=0.7
        )
        ax.set_xlabel("Relative time (periods since treatment)")
        ax.set_ylabel(f"Coefficient ({self.outcome})")
        ax.set_title("Event-Study Estimates")
        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Event-study plot saved to %s.", output_path)

        return fig

    def save_results(self, output_dir: str | Path) -> None:
        """Save coefficient table, pre-trend test, and plot.

        Parameters
        ----------
        output_dir : str | Path
            Directory to write outputs into.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before save_results().")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Coefficient table
        csv_path = output_dir / "event_study_coefs.csv"
        self.coef_table().to_csv(csv_path, index=False)
        logger.info("Event-study coefficients saved to %s.", csv_path)

        # Pre-trend test
        pretrend = self.pretrend_test()
        json_path = output_dir / "event_study_pretrend.json"
        with open(json_path, "w") as fh:
            json.dump(pretrend, fh, indent=2)
        logger.info("Pre-trend test saved to %s.", json_path)

        # Plot
        plot_path = output_dir / "event_study_plot.png"
        fig = self.plot(output_path=plot_path)
        plt.close(fig)
