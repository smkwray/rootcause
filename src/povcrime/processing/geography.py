"""Geography utilities for FIPS code standardisation and merging.

Functions for ensuring consistent FIPS formatting, validating county
FIPS codes, and merging state-level data onto county-level frames.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Valid 2-digit state FIPS codes (50 states + DC).
VALID_STATE_FIPS: set[str] = {
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56",
}


def standardize_fips_format(df: pd.DataFrame, fips_col: str) -> pd.DataFrame:
    """Zero-pad a FIPS column to the appropriate width.

    County FIPS codes (columns containing "county") are padded to 5 digits;
    state FIPS codes are padded to 2 digits.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    fips_col : str
        Name of the column containing FIPS codes.

    Returns
    -------
    pd.DataFrame
        DataFrame with the FIPS column standardised to zero-padded strings.
    """
    width = 5 if "county" in fips_col.lower() else 2
    df = df.copy()
    df[fips_col] = df[fips_col].astype(str).str.strip().str.zfill(width)
    return df


def validate_county_fips(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that county FIPS codes are well-formed.

    Checks
    ------
    - ``county_fips`` column exists and contains 5-character strings.
    - State prefix (first 2 digits) is a valid state FIPS code.
    - No pseudo-FIPS or statewide aggregates (county part != '000').

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``county_fips`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame filtered to valid county FIPS rows, with a log
        of removed records.
    """
    if "county_fips" not in df.columns:
        raise ValueError("DataFrame must have a 'county_fips' column.")

    n_start = len(df)
    df = df.copy()

    # Ensure string type and 5 characters.
    df["county_fips"] = df["county_fips"].astype(str).str.strip().str.zfill(5)
    bad_len = df["county_fips"].str.len() != 5
    if bad_len.any():
        logger.warning(
            "Dropping %d rows with non-5-digit county_fips.", bad_len.sum()
        )
        df = df[~bad_len].copy()

    # Valid state prefix.
    state_prefix = df["county_fips"].str[:2]
    bad_state = ~state_prefix.isin(VALID_STATE_FIPS)
    if bad_state.any():
        logger.warning(
            "Dropping %d rows with invalid state FIPS prefix.",
            bad_state.sum(),
        )
        df = df[~bad_state].copy()

    # Exclude statewide aggregates (county portion == "000").
    is_state_total = df["county_fips"].str[-3:] == "000"
    if is_state_total.any():
        logger.warning(
            "Dropping %d statewide aggregate rows (county='000').",
            is_state_total.sum(),
        )
        df = df[~is_state_total].copy()

    n_end = len(df)
    if n_end < n_start:
        logger.info(
            "validate_county_fips: %d -> %d rows (%d removed).",
            n_start, n_end, n_start - n_end,
        )

    return df.reset_index(drop=True)


def merge_state_to_county(
    state_df: pd.DataFrame,
    county_df: pd.DataFrame,
    *,
    suffixes: tuple[str, str] = ("", "_state"),
) -> pd.DataFrame:
    """Merge state-level data onto a county-level DataFrame.

    Joins on ``state_fips`` and ``year``.  State-level values are
    broadcast to every county within the state.

    Parameters
    ----------
    state_df : pd.DataFrame
        State-level DataFrame with ``state_fips`` and ``year`` columns.
    county_df : pd.DataFrame
        County-level DataFrame with ``state_fips`` and ``year`` columns.
    suffixes : tuple[str, str]
        Suffixes for overlapping columns (default: no suffix for county,
        ``_state`` for state).

    Returns
    -------
    pd.DataFrame
        The county DataFrame with state-level columns merged on.
    """
    # Drop county_fips from state_df if present (it's a state-total code).
    state_cols = [
        c for c in state_df.columns
        if c not in ("county_fips",)
    ]
    state_clean = state_df[state_cols].copy()

    merged = county_df.merge(
        state_clean,
        on=["state_fips", "year"],
        how="left",
        suffixes=suffixes,
    )
    return merged
