"""Markdown report generator for causal poverty-crime analysis.

Assembles a structured research report from model outputs, panel
summaries, and diagnostic results.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class ReportBuilder:
    """Build a Markdown research report from model outputs.

    Parameters
    ----------
    output_dir : Path
        Directory where the report and referenced artefacts will be
        written.
    """

    def __init__(self, output_dir: Path) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sections: list[str] = []

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def add_panel_summary(self, panel: pd.DataFrame) -> None:
        """Add sections 1 (Data Sources) and 2 (Sample Description).

        Parameters
        ----------
        panel : pd.DataFrame
            The merged county-year panel with coverage metrics.
        """
        n_rows = len(panel)
        n_counties = panel["county_fips"].nunique() if "county_fips" in panel.columns else 0
        n_years = panel["year"].nunique() if "year" in panel.columns else 0
        year_min = int(panel["year"].min()) if "year" in panel.columns else 0
        year_max = int(panel["year"].max()) if "year" in panel.columns else 0

        # Source coverage
        source_indicators = {
            "SAIPE (poverty, population)": "poverty_rate",
            "LAUS (unemployment)": "unemployment_rate",
            "BEA (per-capita income)": "per_capita_personal_income",
            "ACS (demographics)": "pct_male",
            "FBI Crime Data": "violent_crime_count",
            "DOL Minimum Wage": "effective_min_wage",
            "USDA SNAP Policy": "broad_based_cat_elig",
        }

        source_lines = []
        for desc, col in source_indicators.items():
            if col in panel.columns:
                non_null = panel[col].notna().sum()
                pct = 100 * non_null / max(n_rows, 1)
                source_lines.append(f"- **{desc}**: {non_null:,} non-null rows ({pct:.1f}%)")
            else:
                source_lines.append(f"- **{desc}**: not available")

        sources_section = (
            "## 1. Data Sources\n\n"
            + "\n".join(source_lines)
            + f"\n\n- **Year range**: {year_min}-{year_max}\n"
        )
        self._sections.append(sources_section)

        # Missingness
        total_cells = n_rows * len(panel.columns)
        missing_cells = int(panel.isna().sum().sum())
        missing_pct = 100 * missing_cells / max(total_cells, 1)

        coverage_info = ""
        if "source_share" in panel.columns:
            mean_share = panel["source_share"].mean()
            coverage_info += f"- Mean source coverage: {mean_share:.2f}\n"
        if "low_coverage" in panel.columns:
            n_low = int(panel["low_coverage"].sum())
            pct_low = 100 * n_low / max(n_rows, 1)
            coverage_info += f"- Low-coverage rows: {n_low:,} ({pct_low:.1f}%)\n"

        sample_section = (
            "## 2. Sample Description\n\n"
            f"- **Panel dimensions**: {n_counties:,} counties x {n_years} years "
            f"= {n_rows:,} observations\n"
            f"- **Year range**: {year_min}-{year_max}\n"
            + coverage_info
            + f"- **Overall missingness**: {missing_cells:,} cells "
            f"({missing_pct:.1f}% of {total_cells:,} total)\n"
        )
        self._sections.append(sample_section)

    def add_treatment_description(self, panel: pd.DataFrame) -> None:
        """Add section 3 (Treatment Definition).

        Parameters
        ----------
        panel : pd.DataFrame
            The merged panel.
        """
        lines = ["## 3. Treatment Definition\n"]

        # Minimum wage treatment
        if "effective_min_wage" in panel.columns:
            mw = panel["effective_min_wage"].dropna()
            lines.append("### Minimum Wage\n")
            lines.append(
                "Treatment is defined as changes in the effective (max of state "
                "and federal) minimum wage. The effective minimum wage captures "
                "policy variation driven by state-level decisions to set wages "
                "above the federal floor.\n"
            )
            lines.append(f"- Observations with min-wage data: {len(mw):,}")
            lines.append(f"- Range: ${mw.min():.2f} - ${mw.max():.2f}")
            lines.append(f"- Mean: ${mw.mean():.2f}\n")

            # Count state-level wage change events
            if "state_fips" in panel.columns and "year" in panel.columns:
                state_yr = (
                    panel.groupby(["state_fips", "year"])["effective_min_wage"]
                    .first()
                    .reset_index()
                    .sort_values(["state_fips", "year"])
                )
                state_yr["wage_change"] = state_yr.groupby("state_fips")[
                    "effective_min_wage"
                ].diff()
                n_events = int((state_yr["wage_change"].abs() > 0).sum())
                lines.append(f"- State-year wage-change events: {n_events}\n")

        # SNAP policy treatment
        if "broad_based_cat_elig" in panel.columns:
            lines.append("### SNAP Policy (Broad-Based Categorical Eligibility)\n")
            lines.append(
                "Treatment is defined as state adoption of broad-based "
                "categorical eligibility (BBCE), which expands SNAP "
                "eligibility beyond traditional income/asset limits.\n"
            )
            snap = panel["broad_based_cat_elig"].dropna()
            n_treated = int((snap == 1).sum())
            n_total = len(snap)
            lines.append(f"- Observations with SNAP data: {n_total:,}")
            lines.append(
                f"- Treated (BBCE=1): {n_treated:,} ({100 * n_treated / max(n_total, 1):.1f}%)\n"
            )

        self._sections.append("\n".join(lines))

    def add_outcome_description(self, panel: pd.DataFrame) -> None:
        """Add section 4 (Outcomes).

        Parameters
        ----------
        panel : pd.DataFrame
            The merged panel.
        """
        lines = ["## 4. Outcomes\n"]

        for col, label in [
            ("violent_crime_rate", "Violent Crime Rate (per 100k)"),
            ("property_crime_rate", "Property Crime Rate (per 100k)"),
        ]:
            if col in panel.columns:
                s = panel[col].dropna()
                lines.append(f"### {label}\n")
                lines.append("| Statistic | Value |")
                lines.append("|-----------|-------|")
                lines.append(f"| N         | {len(s):,} |")
                lines.append(f"| Mean      | {s.mean():.2f} |")
                lines.append(f"| Std Dev   | {s.std():.2f} |")
                lines.append(f"| Min       | {s.min():.2f} |")
                lines.append(f"| 25th pctl | {s.quantile(0.25):.2f} |")
                lines.append(f"| Median    | {s.median():.2f} |")
                lines.append(f"| 75th pctl | {s.quantile(0.75):.2f} |")
                lines.append(f"| Max       | {s.max():.2f} |")
                lines.append("")
            else:
                lines.append(f"### {label}\n")
                lines.append("*Column not available in the panel.*\n")

        self._sections.append("\n".join(lines))

    def add_baseline_results(self, results_csv: Path) -> None:
        """Add section 5 (Baseline Results - TWFE).

        Parameters
        ----------
        results_csv : Path
            Path to a CSV file with columns: ``variable``,
            ``coefficient``, ``std_error``, ``t_stat``, ``p_value``,
            ``ci_lower``, ``ci_upper``.
        """
        results_csv = Path(results_csv)
        if not results_csv.exists():
            self._sections.append(
                "## 5. Baseline Results (TWFE)\n\n"
                "*Results file not found.*\n"
            )
            return

        df = pd.read_csv(results_csv)

        lines = [
            "## 5. Baseline Results (TWFE)\n",
            "Two-way fixed effects (county + year) with clustered standard "
            "errors at the state level.\n",
            "| Variable | Coefficient | Std Error | t-stat | p-value | 95% CI |",
            "|----------|-------------|-----------|--------|---------|--------|",
        ]

        for _, row in df.iterrows():
            ci = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
            sig = ""
            if row["p_value"] < 0.01:
                sig = "***"
            elif row["p_value"] < 0.05:
                sig = "**"
            elif row["p_value"] < 0.10:
                sig = "*"
            lines.append(
                f"| {row['variable']} | {row['coefficient']:.4f}{sig} | "
                f"{row['std_error']:.4f} | {row['t_stat']:.3f} | "
                f"{row['p_value']:.4f} | {ci} |"
            )

        lines.append("")
        lines.append(
            "Significance: *** p<0.01, ** p<0.05, * p<0.10\n"
        )

        self._sections.append("\n".join(lines))

    def add_event_study(
        self,
        coefs_csv: Path,
        plot_png: Path,
        pretrend: dict[str, Any],
    ) -> None:
        """Add section 6 (Event Study).

        Parameters
        ----------
        coefs_csv : Path
            Path to event-study coefficient CSV.
        plot_png : Path
            Path to the event-study coefficient plot.
        pretrend : dict
            Pre-trend test results with keys ``f_stat``, ``p_value``,
            ``n_pre_coefs``, ``pass``.
        """
        lines = ["## 6. Event Study\n"]

        # Pre-trend test
        f_stat = pretrend.get("f_stat", float("nan"))
        p_val = pretrend.get("p_value", float("nan"))
        n_pre = pretrend.get("n_pre_coefs", 0)
        passed = pretrend.get("pass", True)
        verdict = "PASS (fail to reject no pre-trends)" if passed else "FAIL (pre-trends detected)"

        lines.append("### Pre-Trend Test\n")
        lines.append(f"- Joint F-test of {n_pre} pre-treatment coefficients = 0")
        lines.append(f"- F-statistic: {f_stat:.3f}")
        lines.append(f"- p-value: {p_val:.4f}")
        lines.append(f"- **Result: {verdict}**\n")

        # Coefficient table
        coefs_csv = Path(coefs_csv)
        if coefs_csv.exists():
            df = pd.read_csv(coefs_csv)
            lines.append("### Event-Time Coefficients\n")
            lines.append("| Relative Time | Coefficient | Std Error | 95% CI |")
            lines.append("|---------------|-------------|-----------|--------|")
            for _, row in df.iterrows():
                rt = int(row["relative_time"])
                ci = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                lines.append(
                    f"| {rt:+d} | {row['coefficient']:.4f} | "
                    f"{row['std_error']:.4f} | {ci} |"
                )
            lines.append("")

        # Plot reference
        plot_png = Path(plot_png)
        if plot_png.exists():
            rel_path = plot_png.name
            lines.append(f"![Event-Study Plot]({rel_path})\n")

        self._sections.append("\n".join(lines))

    def add_dml_results(self, summary: dict[str, Any]) -> None:
        """Add section 7 (ML-Enhanced Results - DML).

        Parameters
        ----------
        summary : dict
            DML summary with keys: ``treatment``, ``outcome``,
            ``theta``, ``se``, ``t_stat``, ``p_value``,
            ``ci_lower``, ``ci_upper``, ``n_obs``.
        """
        lines = [
            "## 7. ML-Enhanced Results (DML)\n",
            "Double/Debiased Machine Learning with histogram gradient boosting for "
            "nuisance-parameter estimation.\n",
        ]

        theta = summary.get("theta", float("nan"))
        se = summary.get("se", float("nan"))
        p = summary.get("p_value", float("nan"))
        ci_lo = summary.get("ci_lower", float("nan"))
        ci_hi = summary.get("ci_upper", float("nan"))
        treatment = summary.get("treatment", "?")
        outcome = summary.get("outcome", "?")
        n_obs = summary.get("n_obs", 0)

        sig = ""
        if p < 0.01:
            sig = "***"
        elif p < 0.05:
            sig = "**"
        elif p < 0.10:
            sig = "*"

        lines.append(f"- **Treatment**: {treatment}")
        lines.append(f"- **Outcome**: {outcome}")
        lines.append(f"- **ATE estimate (theta)**: {theta:.6f}{sig}")
        lines.append(f"- **Standard error**: {se:.6f}")
        lines.append(f"- **t-statistic**: {summary.get('t_stat', float('nan')):.3f}")
        lines.append(f"- **p-value**: {p:.4f}")
        lines.append(f"- **95% CI**: [{ci_lo:.6f}, {ci_hi:.6f}]")
        lines.append(f"- **N**: {n_obs:,}\n")

        self._sections.append("\n".join(lines))

    def add_cate_results(
        self,
        ate: dict[str, Any],
        importance_df: pd.DataFrame,
    ) -> None:
        """Add section 8 (Heterogeneous Effects).

        Parameters
        ----------
        ate : dict
            ATE summary from CausalForestEstimator.ate_summary().
        importance_df : pd.DataFrame
            Feature importance table with ``feature`` and ``importance``
            columns.
        """
        lines = [
            "## 8. Heterogeneous Effects\n",
            "Causal Forest (CausalForestDML) estimates of conditional "
            "average treatment effects.\n",
        ]

        # ATE from the forest
        lines.append("### Average Treatment Effect (Forest)\n")
        lines.append(f"- **ATE**: {ate.get('ate', float('nan')):.6f}")
        ci_lo = ate.get("ci_lower", float("nan"))
        ci_hi = ate.get("ci_upper", float("nan"))
        lines.append(f"- **95% CI**: [{ci_lo:.6f}, {ci_hi:.6f}]")
        lines.append(f"- **N**: {ate.get('n_obs', 0):,}\n")

        # Feature importance
        lines.append("### Feature Importance for Treatment Effect Heterogeneity\n")
        lines.append("| Rank | Feature | Importance |")
        lines.append("|------|---------|------------|")
        for rank, (_, row) in enumerate(importance_df.iterrows(), start=1):
            lines.append(
                f"| {rank} | {row['feature']} | {row['importance']:.4f} |"
            )
        lines.append("")

        self._sections.append("\n".join(lines))

    def add_sensitivity(
        self,
        coverage_note: str = "",
        weighting_note: str = "",
        alt_sample_note: str = "",
    ) -> None:
        """Add section 9 (Sensitivity & Robustness).

        Parameters
        ----------
        coverage_note : str
            Description of coverage sensitivity checks.
        weighting_note : str
            Description of population-weighting checks.
        alt_sample_note : str
            Description of alternative sample checks.
        """
        lines = ["## 9. Sensitivity & Robustness\n"]

        if coverage_note:
            lines.append(f"### Coverage Sensitivity\n\n{coverage_note}\n")
        else:
            lines.append(
                "### Coverage Sensitivity\n\n"
                "Baseline results use rows with source coverage >= 75%. "
                "Alternative thresholds (60%, 90%) can be tested to verify "
                "robustness to sample selection.\n"
            )

        if weighting_note:
            lines.append(f"### Population Weighting\n\n{weighting_note}\n")
        else:
            lines.append(
                "### Population Weighting\n\n"
                "Results should be compared with and without population "
                "weighting to assess whether effects are driven by large "
                "vs. small counties.\n"
            )

        if alt_sample_note:
            lines.append(f"### Alternative Samples\n\n{alt_sample_note}\n")
        else:
            lines.append(
                "### Alternative Samples\n\n"
                "Consider excluding counties with imputed crime data and "
                "testing urban-only vs. rural-only subsamples.\n"
            )

        self._sections.append("\n".join(lines))

    def add_limitations(self) -> None:
        """Add section 10 (Limitations)."""
        text = (
            "## 10. Limitations\n\n"
            "### Crime Data Caveats\n\n"
            "- FBI UCR data reflect *recorded* crimes (offenses known to law "
            "enforcement), not total victimization. Under-reporting varies "
            "by crime type, jurisdiction, and year.\n"
            "- The UCR-to-NIBRS transition (completed 2021) creates "
            "comparability challenges across years. Agencies transitioning "
            "to NIBRS may show apparent changes in crime counts that reflect "
            "measurement rather than real changes.\n"
            "- The county-level FBI fallback built from official Return A "
            "master files depends on heuristic agency-to-county matching, "
            "and its coverage metrics are approximate rather than official "
            "covered-population shares.\n\n"
            "### ACS Pooling Issues\n\n"
            "- ACS 5-year estimates pool data over a 5-year window (e.g., "
            "the 2019 release covers 2015-2019). This smoothing means "
            "demographic controls may not reflect year-specific conditions "
            "and may attenuate estimated effects.\n"
            "- Margins of error for small counties can be substantial, "
            "introducing measurement error in control variables.\n\n"
            "### Area-Level Interpretation\n\n"
            "- All estimates are area-level (ecological) associations. "
            "County-level effects cannot be directly attributed to "
            "individual-level causal mechanisms (ecological fallacy).\n"
            "- TWFE estimates may be biased in settings with staggered "
            "adoption and heterogeneous treatment effects "
            "(Goodman-Bacon 2021, de Chaisemartin & D'Haultfoeuille 2020). "
            "The DML and causal forest approaches provide complementary "
            "evidence but rest on different identifying assumptions.\n"
        )
        self._sections.append(text)

    def add_next_steps(self, items: list[str] | None = None) -> None:
        """Add section 11 (Next Steps).

        Parameters
        ----------
        items : list[str] | None
            Specific next-step items.  If ``None``, uses defaults.
        """
        if items is None:
            items = [
                "Apply staggered difference-in-differences estimators "
                "(Callaway-Sant'Anna, Sun-Abraham) to address TWFE bias.",
                "Add a deeper overlap/support diagnostic layer for the ML "
                "estimators.",
                "Add instrumental-variable analysis for minimum wage using "
                "federal wage floor changes.",
                "Extend panel to include additional safety-net programmes "
                "(EITC, Medicaid expansion).",
                "Conduct spatial analysis to account for cross-county "
                "spillovers in crime.",
                "Incorporate cost-of-living adjustments for real minimum "
                "wage comparisons.",
            ]

        lines = ["## 11. Next Steps\n"]
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
        self._sections.append("\n".join(lines))

    # ------------------------------------------------------------------
    # Build & save
    # ------------------------------------------------------------------

    def build(self) -> str:
        """Return the full Markdown report string.

        Returns
        -------
        str
            Complete Markdown document.
        """
        header = (
            f"# Causal Poverty-Crime Analysis Report\n\n"
            f"Generated: {date.today().isoformat()}\n\n"
            f"---\n"
        )
        body = "\n\n".join(self._sections)
        return header + "\n" + body

    def save(self, filename: str = "report.md") -> Path:
        """Write report to output directory.

        Parameters
        ----------
        filename : str
            Name of the output file (default ``"report.md"``).

        Returns
        -------
        Path
            Path to the written file.
        """
        report_text = self.build()
        out_path = self._dir / filename
        out_path.write_text(report_text, encoding="utf-8")
        logger.info("Report saved to %s (%d characters).", out_path, len(report_text))
        return out_path
