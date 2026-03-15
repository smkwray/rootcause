"""Project configuration loader.

Reads ``configs/project.yaml`` and ``.env``, then exposes a
:class:`ProjectConfig` dataclass via the :func:`get_config` singleton.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ProjectConfig:
    """Centralised project configuration."""

    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    interim_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    start_year: int = 2000
    end_year: int = 2024

    census_api_key: str = ""
    bls_api_key: str = ""
    bea_api_key: str = ""
    fred_api_key: str = ""
    ipums_api_key: str = ""
    noaa_api_key: str = ""
    usda_quickstats_api_key: str = ""

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        self.output_dir = self.project_root / "outputs"


_config: ProjectConfig | None = None


def get_config() -> ProjectConfig:
    """Return a singleton :class:`ProjectConfig`.

    On the first call the function loads ``configs/project.yaml`` and
    ``.env``, merges them, and caches the result.
    """
    global _config  # noqa: PLW0603
    if _config is not None:
        return _config

    # Load .env into os.environ
    env_path = _PROJECT_ROOT / ".env"
    load_dotenv(env_path)

    # Load YAML config
    yaml_path = _PROJECT_ROOT / "configs" / "project.yaml"
    yaml_data: dict = {}
    if yaml_path.exists():
        with open(yaml_path) as fh:
            yaml_data = yaml.safe_load(fh) or {}

    _config = ProjectConfig(
        start_year=int(yaml_data.get("start_year", 2000)),
        end_year=int(yaml_data.get("end_year", 2024)),
        census_api_key=os.getenv("CENSUS_API_KEY", ""),
        bls_api_key=os.getenv("BLS_API_KEY", ""),
        bea_api_key=os.getenv("BEA_API_KEY", ""),
        fred_api_key=os.getenv("FRED_API_KEY", ""),
        ipums_api_key=os.getenv("IPUMS_API_KEY", ""),
        noaa_api_key=os.getenv("NOAA_API_KEY", ""),
        usda_quickstats_api_key=os.getenv("USDA_QUICKSTATS_API_KEY", ""),
    )
    return _config
