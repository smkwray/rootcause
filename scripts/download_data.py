"""Download raw data sources for the poverty-crime research project."""

from __future__ import annotations

import argparse
import logging
import sys

from povcrime.config import get_config
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the download-data script."""
    parser = argparse.ArgumentParser(
        description="Download raw data for the poverty-crime project.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only create directory structure; do not download data.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Download only these sources (e.g., saipe laus bea).",
    )
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    if args.init_only:
        print("Directory structure initialised. Exiting (--init-only).")
        sys.exit(0)

    # Import adapters lazily so --init-only works without all deps.
    from povcrime.data.acs import ACSAdapter
    from povcrime.data.bea import BEAAdapter
    from povcrime.data.census_cbp import CensusCBPAdapter
    from povcrime.data.county_adjacency import CountyAdjacencyAdapter
    from povcrime.data.dol_min_wage import DOLMinWageAdapter
    from povcrime.data.fbi_crime import FBICrimeAdapter
    from povcrime.data.fhfa_hpi import FHFAHPIAdapter
    from povcrime.data.hud_fmr import HUDFMRAdapter
    from povcrime.data.laus import LAUSAdapter
    from povcrime.data.saipe import SAIPEAdapter
    from povcrime.data.ukcpr_welfare import UKCPRWelfareAdapter
    from povcrime.data.usda_snap_policy import USDASnapPolicyAdapter

    adapters: dict[str, object] = {
        "saipe": SAIPEAdapter(config),
        "acs": ACSAdapter(config),
        "laus": LAUSAdapter(config),
        "bea": BEAAdapter(config),
        "census_cbp": CensusCBPAdapter(config),
        "county_adjacency": CountyAdjacencyAdapter(config),
        "fhfa_hpi": FHFAHPIAdapter(config),
        "hud_fmr": HUDFMRAdapter(config),
        "dol_min_wage": DOLMinWageAdapter(config),
        "ukcpr_welfare": UKCPRWelfareAdapter(config),
        "usda_snap_policy": USDASnapPolicyAdapter(config),
        "fbi_crime": FBICrimeAdapter(config),
    }

    targets = args.sources if args.sources else list(adapters.keys())

    for name in targets:
        if name not in adapters:
            logger.warning("Unknown source '%s', skipping.", name)
            continue
        adapter = adapters[name]
        logger.info("=== Downloading: %s ===", name)
        try:
            adapter.download()
            logger.info("=== %s download complete ===", name)
        except Exception:
            logger.exception("Failed to download %s.", name)


if __name__ == "__main__":
    main()
