"""Coverage diagnostics for data-quality assessment.

Functions for computing, flagging, and summarising coverage across the
county-year panel.  Coverage metrics help identify counties or years
with insufficient data for reliable estimation.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns whose non-null presence signals source coverage.
_SOURCE_INDICATORS: dict[str, list[str]] = {
    "saipe": ["poverty_rate"],
    "laus": ["unemployment_rate"],
    "bea": ["per_capita_personal_income"],
    "acs": ["pct_male"],
    "census_cbp": ["cbp_employment"],
    "fhfa_hpi": ["fhfa_hpi_2000_base"],
    "hud_fmr": ["fair_market_rent_2br"],
    "fbi_crime": ["violent_crime_count"],
}


def compute_coverage_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute coverage metrics for each county-year.

    Metrics added
    -------------
    ``source_count`` : int
        Number of data sources with non-null key variable for this row.
    ``source_share`` : float
        ``source_count / total_sources`` (0-1 scale).
    ``control_completeness`` : float
        Fraction of core control columns that are non-null.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame with county_fips, year, and source columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with coverage metric columns.
    """
    df = df.copy()

    # Count how many sources have data for this row.
    total_sources = 0
    source_present = pd.Series(0, index=df.index, dtype=int)
    for _source, cols in _SOURCE_INDICATORS.items():
        # A source is "present" if all its indicator columns exist and
        # at least one is non-null.
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        total_sources += 1
        has_data = df[available].notna().any(axis=1)
        source_present += has_data.astype(int)

    df["source_count"] = source_present
    df["source_share"] = (
        source_present / max(total_sources, 1)
    )

    # Control completeness: fraction of expected control columns that
    # are non-null.
    control_cols = [
        "poverty_rate",
        "unemployment_rate",
        "per_capita_personal_income",
        "median_hh_income",
        "population",
        "pct_male",
        "pct_white",
        "pct_black",
        "pct_hispanic",
        "pct_under_18",
        "pct_over_65",
        "pct_hs_or_higher",
        "pct_bachelor_or_higher",
        "median_age",
        "cbp_employment_per_capita",
        "cbp_establishments_per_1k",
        "log_fhfa_hpi_2000_base",
        "rent_to_income_ratio_2br",
    ]
    present_controls = [c for c in control_cols if c in df.columns]
    if present_controls:
        df["control_completeness"] = (
            df[present_controls].notna().sum(axis=1) / len(present_controls)
        )
    else:
        df["control_completeness"] = 0.0

    return df


def flag_low_coverage(
    df: pd.DataFrame,
    threshold: float = 0.75,
) -> pd.DataFrame:
    """Flag county-years with coverage below a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``source_share`` column (from
        :func:`compute_coverage_metrics`).
    threshold : float
        Minimum acceptable source share (default 0.75).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``low_coverage`` boolean column.
    """
    df = df.copy()
    if "source_share" not in df.columns:
        raise ValueError(
            "DataFrame must have a 'source_share' column. "
            "Run compute_coverage_metrics() first."
        )
    df["low_coverage"] = df["source_share"] < threshold
    n_low = df["low_coverage"].sum()
    logger.info(
        "Coverage flagging: %d / %d rows (%.1f%%) below threshold %.2f.",
        n_low,
        len(df),
        100 * n_low / max(len(df), 1),
        threshold,
    )
    return df


def coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a summary table of coverage by year and state.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame with ``state_fips``, ``year``, ``source_share``,
        and ``low_coverage`` columns.

    Returns
    -------
    pd.DataFrame
        Aggregated coverage statistics (mean source_share,
        mean control_completeness, count of low-coverage units)
        grouped by year and state.
    """
    required = {"state_fips", "year", "source_share"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns for coverage_summary: {missing}. "
            "Run compute_coverage_metrics() and flag_low_coverage() first."
        )

    agg_cols: dict[str, tuple[str, str]] = {
        "n_counties": ("source_share", "count"),
        "mean_source_share": ("source_share", "mean"),
        "min_source_share": ("source_share", "min"),
    }
    if "control_completeness" in df.columns:
        agg_cols["mean_control_completeness"] = (
            "control_completeness",
            "mean",
        )
    if "low_coverage" in df.columns:
        agg_cols["n_low_coverage"] = ("low_coverage", "sum")

    summary = df.groupby(["state_fips", "year"]).agg(**agg_cols)
    return summary.reset_index()
