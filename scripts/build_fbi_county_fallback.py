"""Build a county-level FBI crime fallback file from CDE master files."""

from __future__ import annotations

import logging

from povcrime.config import get_config
from povcrime.data.fbi_reta_master import build_county_fallback


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    cfg = get_config()
    output_path = cfg.raw_dir / "fbi_crime" / "county_crime.parquet"
    df = build_county_fallback(
        start_year=cfg.start_year,
        end_year=cfg.end_year,
        raw_dir=cfg.raw_dir / "fbi_crime",
        output_path=output_path,
    )
    logging.getLogger(__name__).info("Wrote %s rows to %s", len(df), output_path)


if __name__ == "__main__":
    main()
