"""Build the exploratory reverse-direction scaffold."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.config import get_config
from povcrime.reports.reverse_direction import build_reverse_direction_scaffold
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build reverse-direction exploratory scaffold.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)

    md_path, json_path = build_reverse_direction_scaffold(
        panel=panel,
        output_dir=config.output_dir / "exploratory" / "reverse_direction",
    )
    logger.info("Reverse-direction scaffold written to %s", md_path)
    logger.info("Reverse-direction specs written to %s", json_path)


if __name__ == "__main__":
    main()
