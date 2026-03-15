"""Causal Forest estimator for heterogeneous treatment effects.

Wraps :class:`econml.dml.CausalForestDML` to estimate conditional
average treatment effects (CATEs) and feature importance for
treatment-effect heterogeneity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import HistGradientBoostingRegressor

logger = logging.getLogger(__name__)


class CausalForestEstimator:
    """Causal Forest estimator for CATE estimation.

    Uses the CausalForestDML from the ``econml`` package, which combines
    the causal forest methodology of Athey & Imbens (2018) with the
    double machine learning framework of Chernozhukov et al. (2018).

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame.
    outcome : str
        Name of the dependent variable column.
    treatment : str
        Name of the treatment variable column.
    controls : list[str]
        Names of control/confounding variable columns (used in
        first-stage nuisance estimation).
    effect_modifiers : list[str] | None
        Variables along which treatment effects may vary.  If ``None``,
        defaults to using the same columns as ``controls``.
    n_estimators : int
        Number of trees in the causal forest (default 500).
    min_samples_leaf : int
        Minimum samples per leaf in each tree (default 10).
    model_y : sklearn estimator or None
        First-stage model for the outcome.
    model_t : sklearn estimator or None
        First-stage model for the treatment.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        outcome: str,
        treatment: str,
        controls: list[str],
        effect_modifiers: list[str] | None = None,
        n_estimators: int = 500,
        min_samples_leaf: int = 10,
        model_y: Any | None = None,
        model_t: Any | None = None,
    ) -> None:
        self.outcome = outcome
        self.treatment = treatment
        self.controls = list(controls)
        self.effect_modifiers = (
            list(effect_modifiers) if effect_modifiers else list(controls)
        )
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf

        # Validate columns
        all_cols = (
            {outcome, treatment}
            | set(self.controls)
            | set(self.effect_modifiers)
        )
        missing = all_cols - set(df.columns)
        if missing:
            raise ValueError(f"Columns missing from DataFrame: {sorted(missing)}")

        # Drop rows with NaNs
        self._df = df[list(all_cols)].dropna().reset_index(drop=True)

        if len(self._df) < 50:
            raise ValueError(
                f"Only {len(self._df)} observations after dropping NaN; "
                "need at least 50 for meaningful CATE estimation."
            )

        self._model_y = model_y or HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=50,
            early_stopping=True,
            random_state=42,
        )
        self._model_t = model_t or HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=50,
            early_stopping=True,
            random_state=42,
        )

        self._cf = None
        self._is_fitted = False

        logger.info(
            "CausalForestEstimator initialised: outcome=%s, treatment=%s, "
            "%d controls, %d effect modifiers, n_obs=%d.",
            outcome,
            treatment,
            len(self.controls),
            len(self.effect_modifiers),
            len(self._df),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> CausalForestDML:
        """Fit the causal forest model.

        Returns
        -------
        CausalForestDML
            The fitted causal forest object.
        """
        Y = self._df[self.outcome].values
        T = self._df[self.treatment].values
        X = self._df[self.effect_modifiers].values
        W = self._df[self.controls].values

        self._cf = CausalForestDML(
            model_y=self._model_y,
            model_t=self._model_t,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self._cf.fit(Y=Y, T=T, X=X, W=W)
        self._is_fitted = True

        ate = float(self._cf.ate(X))
        logger.info("CausalForest fitted: ATE=%.6f, n_obs=%d.", ate, len(self._df))
        return self._cf

    def ate_summary(self) -> dict[str, Any]:
        """Compute the average treatment effect with inference.

        Returns
        -------
        dict
            Keys: ``treatment``, ``outcome``, ``ate``, ``ate_se``,
            ``ci_lower``, ``ci_upper``, ``n_obs``.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before ate_summary().")

        X = self._df[self.effect_modifiers].values
        ate = float(self._cf.ate(X))
        ate_interval = self._cf.ate_interval(X)

        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "ate": ate,
            "ci_lower": float(ate_interval[0]),
            "ci_upper": float(ate_interval[1]),
            "n_obs": len(self._df),
        }

    def cate_predictions(self) -> pd.DataFrame:
        """Return individual-level CATE predictions.

        Returns
        -------
        pd.DataFrame
            DataFrame with effect modifier columns and ``cate``,
            ``cate_lower``, ``cate_upper`` columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before cate_predictions().")

        X = self._df[self.effect_modifiers].values
        cate = self._cf.effect(X).flatten()
        cate_interval = self._cf.effect_interval(X)

        result = self._df[self.effect_modifiers].copy()
        result["cate"] = cate
        result["cate_lower"] = cate_interval[0].flatten()
        result["cate_upper"] = cate_interval[1].flatten()
        return result

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance for treatment-effect heterogeneity.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``.  Sorted descending
            by importance.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before feature_importance().")

        importances = self._cf.feature_importances_
        fi_df = pd.DataFrame(
            {
                "feature": self.effect_modifiers,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        return fi_df

    def save_results(self, output_dir: str | Path) -> None:
        """Save CATE results, ATE summary, and feature importance.

        Parameters
        ----------
        output_dir : str | Path
            Directory to write outputs into.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before save_results().")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        label = f"{self.treatment}__{self.outcome}"

        # ATE summary
        ate = self.ate_summary()
        json_path = output_dir / f"causal_forest_ate_{label}.json"
        with open(json_path, "w") as fh:
            json.dump(ate, fh, indent=2)
        logger.info("Causal forest ATE saved to %s.", json_path)

        # Feature importance
        fi = self.feature_importance()
        fi_path = output_dir / f"causal_forest_importance_{label}.csv"
        fi.to_csv(fi_path, index=False)
        logger.info("Feature importance saved to %s.", fi_path)

        # CATE summary statistics (not full predictions, to save space)
        cates = self.cate_predictions()
        cate_stats = {
            "cate_mean": float(cates["cate"].mean()),
            "cate_median": float(cates["cate"].median()),
            "cate_std": float(cates["cate"].std()),
            "cate_min": float(cates["cate"].min()),
            "cate_max": float(cates["cate"].max()),
            "cate_q10": float(cates["cate"].quantile(0.10)),
            "cate_q25": float(cates["cate"].quantile(0.25)),
            "cate_q75": float(cates["cate"].quantile(0.75)),
            "cate_q90": float(cates["cate"].quantile(0.90)),
        }
        cate_path = output_dir / f"causal_forest_cate_stats_{label}.json"
        with open(cate_path, "w") as fh:
            json.dump(cate_stats, fh, indent=2)
        logger.info("CATE statistics saved to %s.", cate_path)
