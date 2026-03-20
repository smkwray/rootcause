"""Double/Debiased Machine Learning (DML) estimator.

Uses the :mod:`doubleml` package to estimate average treatment effects
with ML-based nuisance-parameter estimation, providing robustness
against model mis-specification in the first stage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

from povcrime.models.panel_ml import PanelMode, prepare_panel_ml_sample, validate_panel_mode

logger = logging.getLogger(__name__)


class DMLEstimator:
    """Double Machine Learning estimator for average treatment effects.

    Implements the Partially Linear Regression (PLR) model from
    Chernozhukov et al. (2018) via the ``doubleml`` package.

    The model is::

        Y = theta * D + g(X) + epsilon
        D = m(X) + v

    where ``Y`` is the outcome, ``D`` is the treatment, ``X`` are
    controls, ``g`` and ``m`` are estimated with ML (gradient boosting
    by default), and ``theta`` is the causal parameter of interest.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame (should already be filtered for quality).
    outcome : str
        Name of the dependent variable column.
    treatment : str
        Name of the treatment variable column.
    controls : list[str]
        Names of control/confounding variable columns.
    group_col : str or None
        Optional grouping column used to keep repeated observations from
        the same panel unit together in cross-fitting.
    panel_mode : {"none", "two_way_within"}
        Optional panel preprocessing mode applied before nuisance fitting.
    n_folds : int
        Number of cross-fitting folds (default 5).
    n_rep : int
        Number of repeated cross-fitting rounds (default 1).
    ml_g : sklearn estimator or None
        Learner for the outcome nuisance function ``g(X)``.
        Defaults to ``HistGradientBoostingRegressor``.
    ml_m : sklearn estimator or None
        Learner for the treatment nuisance function ``m(X)``.
        Defaults to ``HistGradientBoostingRegressor``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        outcome: str,
        treatment: str,
        controls: list[str],
        group_col: str | None = None,
        panel_mode: PanelMode = "none",
        entity_col: str = "county_fips",
        time_col: str = "year",
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 42,
        ml_g: Any | None = None,
        ml_m: Any | None = None,
    ) -> None:
        self.outcome = outcome
        self.treatment = treatment
        self.controls = list(controls)
        self.group_col = group_col
        self.panel_mode = validate_panel_mode(panel_mode)
        self.entity_col = entity_col
        self.time_col = time_col
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

        if self.group_col is not None and self.group_col in self.controls:
            raise ValueError("group_col must not also appear in controls.")

        # Validate columns
        all_cols = [outcome, treatment, *self.controls]
        if self.group_col is not None:
            all_cols.append(self.group_col)
        if self.panel_mode != "none":
            all_cols.extend([self.entity_col, self.time_col])
        all_cols = list(dict.fromkeys(all_cols))
        missing = set(all_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Columns missing from DataFrame: {sorted(missing)}")

        self._df = prepare_panel_ml_sample(
            df,
            model_cols=[self.outcome, self.treatment, *self.controls],
            keep_cols=[col for col in all_cols if col not in {self.outcome, self.treatment, *self.controls}],
            panel_mode=self.panel_mode,
            entity_col=self.entity_col,
            time_col=self.time_col,
        )

        if len(self._df) < 2 * n_folds:
            raise ValueError(
                f"Only {len(self._df)} complete observations available, "
                f"need at least {2 * n_folds} for {n_folds}-fold cross-fitting."
            )
        if self.group_col is not None:
            n_groups = int(self._df[self.group_col].nunique(dropna=False))
            if n_groups < self.n_folds:
                raise ValueError(
                    f"Only {n_groups} unique groups available in {self.group_col}, "
                    f"need at least {self.n_folds} for grouped cross-fitting."
                )

        # Default ML learners
        self._ml_g = ml_g or HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=50,
            early_stopping=True,
            random_state=42,
        )
        self._ml_m = ml_m or HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=50,
            early_stopping=True,
            random_state=42,
        )

        self._dml_data = None
        self._dml_model = None
        self._is_fitted = False

        logger.info(
            "DMLEstimator initialised: outcome=%s, treatment=%s, "
            "%d controls, n_obs=%d, n_folds=%d.",
            outcome,
            treatment,
            len(self.controls),
            len(self._df),
            n_folds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> DoubleMLPLR:
        """Fit the DML model.

        Returns
        -------
        DoubleMLPLR
            The fitted DoubleML model object.
        """
        self._dml_data = DoubleMLData(
            self._df[[self.outcome, self.treatment, *self.controls]],
            y_col=self.outcome,
            d_cols=self.treatment,
            x_cols=self.controls,
        )

        self._dml_model = DoubleMLPLR(
            self._dml_data,
            ml_l=self._ml_g,
            ml_m=self._ml_m,
            n_folds=self.n_folds,
            n_rep=self.n_rep,
            draw_sample_splitting=False,
        )
        self._dml_model.set_sample_splitting(self._build_sample_splitting())
        self._dml_model.fit()
        self._is_fitted = True

        theta = self._dml_model.coef[0]
        se = self._dml_model.se[0]
        logger.info(
            "DML fitted: theta=%.6f, SE=%.6f, t=%.3f.",
            theta,
            se,
            theta / se if se > 0 else float("nan"),
        )
        return self._dml_model

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary of DML results.

        Returns
        -------
        dict
            Keys: ``treatment``, ``outcome``, ``theta`` (ATE estimate),
            ``se``, ``t_stat``, ``p_value``, ``ci_lower``, ``ci_upper``,
            ``n_obs``, ``n_folds``, ``n_rep``.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before summary().")

        model = self._dml_model
        theta = float(model.coef[0])
        se = float(model.se[0])
        t_stat = float(model.t_stat[0])
        p_value = float(model.pval[0])
        ci = model.confint()

        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "theta": theta,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_lower": float(ci.iloc[0, 0]),
            "ci_upper": float(ci.iloc[0, 1]),
            "n_obs": len(self._df),
            "n_folds": self.n_folds,
            "n_rep": self.n_rep,
            "random_state": self.random_state,
            "group_col": self.group_col,
            "panel_mode": self.panel_mode,
            "entity_col": self.entity_col if self.panel_mode != "none" else None,
            "time_col": self.time_col if self.panel_mode != "none" else None,
        }

    def save_results(self, output_dir: str | Path) -> None:
        """Save DML results to JSON.

        Parameters
        ----------
        output_dir : str | Path
            Directory to write outputs into.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before save_results().")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = self.summary()
        label = f"{self.treatment}__{self.outcome}"

        json_path = output_dir / f"dml_{label}.json"
        with open(json_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("DML results saved to %s.", json_path)

        # Also save the DoubleML summary table as CSV
        try:
            summary_df = self._dml_model.summary
            csv_path = output_dir / f"dml_{label}_summary.csv"
            summary_df.to_csv(csv_path)
            logger.info("DML summary table saved to %s.", csv_path)
        except Exception:
            logger.debug("Could not save DoubleML summary table.", exc_info=True)

    def _build_sample_splitting(self) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        if self.group_col is None:
            splitter = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            folds = list(splitter.split(np.arange(len(self._df))))
        else:
            splitter = GroupKFold(n_splits=self.n_folds)
            folds = list(
                splitter.split(
                    np.arange(len(self._df)),
                    groups=self._df[self.group_col].to_numpy(),
                )
            )
        return [list(folds) for _ in range(self.n_rep)]
