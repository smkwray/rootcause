"""Two-way fixed effects estimator for panel data.

Wraps :class:`linearmodels.panel.PanelOLS` to provide a streamlined
interface for county + year fixed effects regressions with clustered
standard errors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from linearmodels.panel import PanelOLS

logger = logging.getLogger(__name__)


class BaselineFE:
    """Two-way fixed effects estimator for panel data.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame in long format (one row per entity-time).
    outcome : str
        Name of the dependent variable column.
    treatment : str | list[str]
        Treatment variable(s) to include on the right-hand side.
    controls : list[str] | None
        Additional control variables.  May be empty or ``None``.
    entity_col : str
        Column identifying the panel entity (default ``"county_fips"``).
    time_col : str
        Column identifying the time period (default ``"year"``).
    cluster_col : str
        Column on which to cluster standard errors (default ``"state_fips"``).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        outcome: str,
        treatment: str | list[str],
        controls: list[str] | None = None,
        entity_col: str = "county_fips",
        time_col: str = "year",
        cluster_col: str = "state_fips",
        weight_col: str | None = None,
    ) -> None:
        self.outcome = outcome
        self.treatment = [treatment] if isinstance(treatment, str) else list(treatment)
        self.controls = list(controls) if controls else []
        self.entity_col = entity_col
        self.time_col = time_col
        self.cluster_col = cluster_col
        self.weight_col = weight_col

        self._regressors = self.treatment + self.controls

        # Validate columns exist
        required = {outcome, entity_col, time_col, cluster_col} | set(self._regressors)
        if weight_col is not None:
            required.add(weight_col)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Columns missing from DataFrame: {sorted(missing)}")

        # Store a copy with only the columns we need
        self._df = df[list(required)].copy()
        self._result = None

        logger.info(
            "BaselineFE initialised: outcome=%s, treatments=%s, controls=%s, "
            "n_obs=%d.",
            self.outcome,
            self.treatment,
            self.controls,
            len(self._df),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        """Fit the two-way fixed effects model.

        Sets a MultiIndex of (entity, time), estimates via
        :class:`~linearmodels.panel.PanelOLS` with ``entity_effects=True``
        and ``time_effects=True``, and clusters standard errors at the
        level specified by ``cluster_col``.

        Returns
        -------
        linearmodels.panel.results.PanelEffectsResults
            The fitted model result object.

        Raises
        ------
        ValueError
            If the treatment column(s) have zero variance.
        """
        df = self._df.dropna(subset=[self.outcome] + self._regressors).copy()

        if df.empty:
            raise ValueError(
                "No observations remain after dropping rows with missing values."
            )

        # Guard against all-zero / zero-variance treatments
        for col in self.treatment:
            if df[col].nunique() < 2:
                raise ValueError(
                    f"Treatment variable '{col}' has no variation "
                    f"(unique values: {df[col].unique().tolist()})."
                )

        # Set the panel MultiIndex expected by linearmodels
        df = df.set_index([self.entity_col, self.time_col])

        y = df[self.outcome]
        regressors = list(self._regressors)
        zero_var = [col for col in self.controls if df[col].nunique() < 2]
        if zero_var:
            regressors = [col for col in regressors if col not in zero_var]
            logger.warning(
                "Dropping zero-variance controls from BaselineFE: %s",
                zero_var,
            )

        x = df[regressors]
        cluster = df[[self.cluster_col]]
        weights = None
        if self.weight_col is not None:
            weights = df[[self.weight_col]]

        model = PanelOLS(
            y,
            x,
            weights=weights,
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True,
            check_rank=False,
        )
        self._result = model.fit(cov_type="clustered", clusters=cluster)

        logger.info(
            "Model fitted: R-sq (within)=%.4f, n_obs=%d, n_entities=%d.",
            self._result.rsquared_within,
            self._result.nobs,
            self._result.entity_info["total"],
        )
        return self._result

    def summary_table(self) -> pd.DataFrame:
        """Return a tidy DataFrame of regression coefficients.

        Returns
        -------
        pd.DataFrame
            Columns: ``variable``, ``coefficient``, ``std_error``,
            ``t_stat``, ``p_value``, ``ci_lower``, ``ci_upper``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self._result is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        res = self._result
        ci = res.conf_int()

        table = pd.DataFrame(
            {
                "variable": res.params.index,
                "coefficient": res.params.values,
                "std_error": res.std_errors.values,
                "t_stat": res.tstats.values,
                "p_value": res.pvalues.values,
                "ci_lower": ci.iloc[:, 0].values,
                "ci_upper": ci.iloc[:, 1].values,
            }
        )
        return table.reset_index(drop=True)

    def save_results(self, output_dir: str | Path) -> None:
        """Save summary CSV and model metadata JSON.

        Parameters
        ----------
        output_dir : str | Path
            Directory to write outputs into.  Created if it does not exist.

        Outputs
        -------
        ``baseline_fe_summary.csv``
            Tidy coefficient table.
        ``baseline_fe_metadata.json``
            Model metadata (specification, fit statistics).
        """
        if self._result is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        csv_path = output_dir / "baseline_fe_summary.csv"
        self.summary_table().to_csv(csv_path, index=False)
        logger.info("Summary table saved to %s.", csv_path)

        # Metadata JSON
        meta = self._build_metadata()
        json_path = output_dir / "baseline_fe_metadata.json"
        with open(json_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Metadata saved to %s.", json_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_metadata(self) -> dict[str, Any]:
        """Assemble a JSON-serialisable metadata dictionary."""
        res = self._result
        return {
            "model": "TwoWayFixedEffects",
            "estimator": "PanelOLS",
            "outcome": self.outcome,
            "treatments": self.treatment,
            "controls": self.controls,
            "entity_col": self.entity_col,
            "time_col": self.time_col,
            "cluster_col": self.cluster_col,
            "weight_col": self.weight_col,
            "n_obs": int(res.nobs),
            "n_entities": int(res.entity_info["total"]),
            "rsquared_within": float(res.rsquared_within),
            "rsquared_between": float(res.rsquared_between),
            "rsquared_overall": float(res.rsquared_overall),
            "f_statistic": float(res.f_statistic.stat),
            "f_p_value": float(res.f_statistic.pval),
        }
