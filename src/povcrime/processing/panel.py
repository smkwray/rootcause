"""Panel builder for constructing the county-year analysis dataset.

Functions for merging multiple data sources into a single balanced (or
semi-balanced) county-year panel, validating primary keys, and computing
derived rate variables.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from povcrime.config import ProjectConfig
from povcrime.processing.geography import (
    merge_state_to_county,
    standardize_fips_format,
    validate_county_fips,
)

logger = logging.getLogger(__name__)


def build_county_year_panel(
    sources: dict[str, pd.DataFrame],
    config: ProjectConfig,
) -> pd.DataFrame:
    """Build the merged county-year panel from individual data sources.

    Merge strategy
    --------------
    1. Start with the SAIPE county-year spine (poverty + population).
    2. Left-join county-level sources on (county_fips, year):
       LAUS, BEA, ACS.
    3. Left-join state-level policy sources on (state_fips, year), broadcast
       to counties: DOL min wage, USDA SNAP.
    4. Left-join county-level FBI crime data on (county_fips, year) when a
       county-level crime file is available.
    5. Compute derived rate columns (crime rates per 100k).
    6. Add coverage metrics.

    Parameters
    ----------
    sources : dict[str, pd.DataFrame]
        Mapping of source name to its loaded DataFrame.  Expected keys:
        ``"saipe"``, ``"laus"``, ``"bea"``, ``"acs"``,
        ``"fbi_crime"``, ``"dol_min_wage"``, ``"usda_snap_policy"``.
    config : ProjectConfig
        Project configuration object.

    Returns
    -------
    pd.DataFrame
        Merged panel with one row per (county_fips, year).
    """
    # ----- Spine: SAIPE (county-year) ----- #
    if "saipe" not in sources:
        raise ValueError("SAIPE data is required as the panel spine.")

    spine = sources["saipe"].copy()
    spine = standardize_fips_format(spine, "county_fips")
    spine = standardize_fips_format(spine, "state_fips")
    spine = validate_county_fips(spine)

    # Ensure state_fips is derived from county_fips for consistency.
    spine["state_fips"] = spine["county_fips"].str[:2]

    logger.info(
        "Panel spine (SAIPE): %d rows, %d counties, years %d-%d.",
        len(spine),
        spine["county_fips"].nunique(),
        spine["year"].min(),
        spine["year"].max(),
    )

    panel = spine

    # ----- County-level merges ----- #
    county_sources = {
        "laus": ["county_fips", "year"],
        "bea": ["county_fips", "year"],
        "acs": ["county_fips", "year"],
        "census_cbp": ["county_fips", "year"],
        "fhfa_hpi": ["county_fips", "year"],
        "hud_fmr": ["county_fips", "year"],
    }
    for name, keys in county_sources.items():
        if name not in sources:
            logger.warning("Source '%s' not provided, skipping.", name)
            continue
        src = sources[name].copy()
        src = standardize_fips_format(src, "county_fips")
        # Drop state_fips from source to avoid collision — we keep spine's.
        src_cols = [c for c in src.columns if c != "state_fips"]
        panel = panel.merge(
            src[src_cols],
            on=keys,
            how="left",
            suffixes=("", f"_{name}"),
        )
        logger.info(
            "Merged '%s': panel now has %d columns.", name, len(panel.columns)
        )

    # ----- State-level merges (broadcast to counties) ----- #
    state_sources = ["dol_min_wage", "ukcpr_welfare", "usda_snap_policy"]
    for name in state_sources:
        if name not in sources:
            logger.warning("Source '%s' not provided, skipping.", name)
            continue
        src = sources[name].copy()
        src = standardize_fips_format(src, "state_fips")
        panel = merge_state_to_county(src, panel)
        logger.info(
            "Merged state-level '%s': panel now has %d columns.",
            name,
            len(panel.columns),
        )

    # ----- County-level crime merge ----- #
    if "fbi_crime" in sources:
        src = sources["fbi_crime"].copy()
        src = standardize_fips_format(src, "county_fips")
        src = standardize_fips_format(src, "state_fips")
        has_county_level = (src["county_fips"].str[-3:] != "000").any()
        if has_county_level:
            src_cols = [c for c in src.columns if c != "state_fips"]
            panel = panel.merge(
                src[src_cols],
                on=["county_fips", "year"],
                how="left",
                suffixes=("", "_fbi"),
            )
            logger.info(
                "Merged county-level 'fbi_crime': panel now has %d columns.",
                len(panel.columns),
            )
        else:
            logger.warning(
                "FBI source contains state-level rows only; skipping merge into "
                "county-year panel because state outcomes cannot be treated as "
                "county outcomes."
            )

    # ----- Derived columns ----- #
    # Crime rates per 100k using SAIPE population.
    pop_col = "population"
    if "population_covered" in panel.columns and pop_col in panel.columns:
        covered = pd.to_numeric(panel["population_covered"], errors="coerce")
        population = pd.to_numeric(panel[pop_col], errors="coerce")
        over_covered = covered.notna() & population.notna() & (covered > population)
        if over_covered.any():
            logger.info(
                "Clipping %d FBI coverage rows where covered population exceeds "
                "county population.",
                int(over_covered.sum()),
            )
            panel.loc[over_covered, "population_covered"] = population.loc[over_covered]
    if "violent_crime_count" in panel.columns and pop_col in panel.columns:
        panel["violent_crime_rate"] = compute_rates(
            panel, "violent_crime_count", pop_col
        )
    if "property_crime_count" in panel.columns and pop_col in panel.columns:
        panel["property_crime_rate"] = compute_rates(
            panel, "property_crime_count", pop_col
        )

    # Derived economic controls from newly added county sources.
    if "cbp_establishments" in panel.columns and pop_col in panel.columns:
        panel["cbp_establishments_per_1k"] = compute_rates(
            panel,
            "cbp_establishments",
            pop_col,
            per=1_000,
        )
    if "cbp_employment" in panel.columns and pop_col in panel.columns:
        panel["cbp_employment_per_capita"] = compute_rates(
            panel,
            "cbp_employment",
            pop_col,
            per=1,
        )
    if {"cbp_annual_payroll", "cbp_employment"}.issubset(panel.columns):
        employment = pd.to_numeric(panel["cbp_employment"], errors="coerce").replace(0, float("nan"))
        payroll = pd.to_numeric(panel["cbp_annual_payroll"], errors="coerce")
        panel["cbp_payroll_per_employee"] = (payroll * 1_000.0) / employment
    if {"fair_market_rent_2br", "median_hh_income"}.issubset(panel.columns):
        income = pd.to_numeric(panel["median_hh_income"], errors="coerce").replace(0, float("nan"))
        rent = pd.to_numeric(panel["fair_market_rent_2br"], errors="coerce")
        panel["rent_to_income_ratio_2br"] = (rent * 12.0) / income
    if "fhfa_hpi_2000_base" in panel.columns:
        panel["log_fhfa_hpi_2000_base"] = np.log(
            pd.to_numeric(panel["fhfa_hpi_2000_base"], errors="coerce").clip(lower=1)
        )

    # Log population and log housing cost are useful scale controls.
    if pop_col in panel.columns:
        panel["log_population"] = np.log(
            panel[pop_col].clip(lower=1).astype(float)
        )
    if "fair_market_rent_2br" in panel.columns:
        panel["log_fair_market_rent_2br"] = np.log(
            pd.to_numeric(panel["fair_market_rent_2br"], errors="coerce").clip(lower=1)
        )

    # ----- Final ordering ----- #
    panel = (
        panel.sort_values(["county_fips", "year"])
        .reset_index(drop=True)
    )

    logger.info(
        "Panel built: %d rows, %d columns, %d counties, %d years.",
        len(panel),
        len(panel.columns),
        panel["county_fips"].nunique(),
        panel["year"].nunique(),
    )
    return panel


def validate_panel_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the panel has unique (county_fips, year) keys.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame.

    Returns
    -------
    pd.DataFrame
        The same DataFrame if keys are unique.

    Raises
    ------
    ValueError
        If duplicate (county_fips, year) rows are found.
    """
    dup_mask = df.duplicated(subset=["county_fips", "year"], keep=False)
    n_dups = dup_mask.sum()
    if n_dups > 0:
        sample = df[dup_mask].head(10)[["county_fips", "year"]]
        raise ValueError(
            f"Panel has {n_dups} duplicate (county_fips, year) rows. "
            f"Examples:\n{sample}"
        )
    logger.info("Panel key validation passed: %d unique rows.", len(df))
    return df


def compute_rates(
    df: pd.DataFrame,
    count_col: str,
    pop_col: str,
    per: int = 100_000,
) -> pd.Series:
    """Compute a per-population rate from a count column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the count and population columns.
    count_col : str
        Name of the column with raw counts.
    pop_col : str
        Name of the column with population denominators.
    per : int
        Rate denominator (default 100,000).

    Returns
    -------
    pd.Series
        The computed rate: ``(count / population) * per``.
        Returns NaN where population is zero or null.
    """
    pop = df[pop_col].astype(float).replace(0, float("nan"))
    return (df[count_col].astype(float) / pop) * per
