"""Build the markdown data-quality report for the analysis panel."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.config import get_config
from povcrime.processing.coverage import compute_coverage_metrics, flag_low_coverage
from povcrime.reports.qa import build_data_quality_report
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Entry point for QA report generation."""
    parser = argparse.ArgumentParser(
        description="Build the panel data-quality markdown report.",
    )
    parser.add_argument(
        "--panel",
        type=str,
        default=None,
        help="Path to the panel parquet file.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.75,
        help="Coverage threshold used when low_coverage is missing.",
    )
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = (
        Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    )
    if not panel_path.exists():
        logger.error("Panel not found at %s. Run build_panel.py first.", panel_path)
        raise SystemExit(1)

    panel = pd.read_parquet(panel_path)

    if "source_share" not in panel.columns:
        panel = compute_coverage_metrics(panel)
    if "low_coverage" not in panel.columns:
        panel = flag_low_coverage(panel, threshold=args.coverage_threshold)

    out_path = config.output_dir / "qa" / "data_quality_report.md"
    build_data_quality_report(panel, out_path)
    logger.info("QA report written to %s", out_path)


if __name__ == "__main__":
    main()
