"""Build the county-year analysis panel from raw data sources.

Loads all adapter outputs, merges them into a single county-year panel,
computes derived variables and coverage metrics, validates the result,
and saves to ``data/processed/panel.parquet``.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from povcrime.config import get_config
from povcrime.utils import ensure_dirs, save_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the build-panel script."""
    parser = argparse.ArgumentParser(
        description="Build the county-year analysis panel.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip per-source validation before merging.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.75,
        help="Minimum source coverage share for quality flagging (default: 0.75).",
    )
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    # Import adapters.
    from povcrime.data.acs import ACSAdapter
    from povcrime.data.bea import BEAAdapter
    from povcrime.data.census_cbp import CensusCBPAdapter
    from povcrime.data.dol_min_wage import DOLMinWageAdapter
    from povcrime.data.fbi_crime import FBICrimeAdapter
    from povcrime.data.fhfa_hpi import FHFAHPIAdapter
    from povcrime.data.hud_fmr import HUDFMRAdapter
    from povcrime.data.laus import LAUSAdapter
    from povcrime.data.saipe import SAIPEAdapter
    from povcrime.data.ukcpr_welfare import UKCPRWelfareAdapter
    from povcrime.data.usda_snap_policy import USDASnapPolicyAdapter
    from povcrime.processing.coverage import (
        compute_coverage_metrics,
        flag_low_coverage,
    )
    from povcrime.processing.panel import (
        build_county_year_panel,
        validate_panel_keys,
    )

    # Load and optionally validate each source.
    adapters = {
        "saipe": SAIPEAdapter(config),
        "acs": ACSAdapter(config),
        "laus": LAUSAdapter(config),
        "bea": BEAAdapter(config),
        "census_cbp": CensusCBPAdapter(config),
        "fhfa_hpi": FHFAHPIAdapter(config),
        "hud_fmr": HUDFMRAdapter(config),
        "dol_min_wage": DOLMinWageAdapter(config),
        "ukcpr_welfare": UKCPRWelfareAdapter(config),
        "usda_snap_policy": USDASnapPolicyAdapter(config),
        "fbi_crime": FBICrimeAdapter(config),
    }

    sources: dict[str, pd.DataFrame] = {}
    for name, adapter in adapters.items():
        logger.info("Loading source: %s", name)
        try:
            df = adapter.load()
            if not args.skip_validation:
                df = adapter.validate(df)
            sources[name] = df
            logger.info(
                "Loaded %s: %d rows, %d columns.",
                name,
                len(df),
                len(df.columns),
            )
        except FileNotFoundError:
            logger.warning(
                "Raw data for '%s' not found. Run download_data.py first or "
                "provide the documented fallback file when applicable. "
                "Skipping.",
                name,
            )
        except Exception:
            logger.exception("Failed to load '%s'. Skipping.", name)

    if "saipe" not in sources:
        logger.error(
            "SAIPE data is required as the panel spine. "
            "Run download_data.py --sources saipe first."
        )
        raise SystemExit(1)

    # Build the panel.
    panel = build_county_year_panel(sources, config)

    # Validate keys.
    panel = validate_panel_keys(panel)

    # Add coverage metrics.
    panel = compute_coverage_metrics(panel)
    panel = flag_low_coverage(panel, threshold=args.coverage_threshold)

    # Save.
    out_path = config.processed_dir / "panel.parquet"
    save_parquet(panel, out_path)
    logger.info("Panel saved to %s", out_path)

    # Print summary.
    n_counties = panel["county_fips"].nunique()
    n_years = panel["year"].nunique()
    n_low = panel["low_coverage"].sum() if "low_coverage" in panel.columns else 0
    print(
        f"\nPanel summary:\n"
        f"  Rows:           {len(panel):,}\n"
        f"  Counties:       {n_counties:,}\n"
        f"  Years:          {n_years}\n"
        f"  Columns:        {len(panel.columns)}\n"
        f"  Low coverage:   {n_low:,} ({100 * n_low / max(len(panel), 1):.1f}%)\n"
        f"  Output:         {out_path}\n"
    )


if __name__ == "__main__":
    main()
