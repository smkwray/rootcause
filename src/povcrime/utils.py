"""Utility helpers for the povcrime package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from povcrime.config import ProjectConfig


def ensure_dirs(config: ProjectConfig) -> None:
    """Create the standard data and output directories if they don't exist."""
    for directory in (
        config.raw_dir,
        config.interim_dir,
        config.processed_dir,
        config.output_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def standardize_fips(fips: str | int, width: int = 5) -> str:
    """Zero-pad a FIPS code to *width* characters.

    Parameters
    ----------
    fips : str | int
        Raw FIPS value (e.g. ``1001`` or ``"1001"``).
    width : int, optional
        Target string width (default ``5`` for county FIPS).

    Returns
    -------
    str
        Zero-padded FIPS string, e.g. ``"01001"``.
    """
    return str(fips).zfill(width)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame.

    Thin wrapper around :func:`pandas.read_parquet` to keep IO calls
    consistent across the project.
    """
    return pd.read_parquet(Path(path))


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to Parquet with the pyarrow engine.

    Parent directories are created automatically.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
