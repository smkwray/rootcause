"""Build frontend-facing JSON artifacts under outputs/app."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.config import get_config
from povcrime.reports.app_artifacts import build_app_artifacts
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build app artifact JSON files.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)

    manifest_path, summary_path = build_app_artifacts(
        project_root=config.project_root,
        panel=panel,
        output_dir=config.output_dir,
    )
    logger.info("App artifact manifest written to %s", manifest_path)
    logger.info("App results summary written to %s", summary_path)


if __name__ == "__main__":
    main()
