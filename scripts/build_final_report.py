"""Build the final markdown report from generated backend artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.config import get_config
from povcrime.reports.final_report import build_final_report
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the final backend report.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = (
        Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    )
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}")

    panel = pd.read_parquet(panel_path)
    out_path = build_final_report(
        panel=panel,
        panel_path=panel_path,
        qa_report_path=config.output_dir / "qa" / "data_quality_report.md",
        baseline_dir=config.output_dir / "baseline",
        dml_dir=config.output_dir / "dml",
        robustness_summary_path=config.output_dir / "robustness" / "robustness_summary.csv",
        output_dir=config.output_dir / "report",
        overlap_dir=config.output_dir / "overlap",
        app_dir=config.output_dir / "app",
        config=config,
    )
    logger.info("Final report written to %s", out_path)


if __name__ == "__main__":
    main()
