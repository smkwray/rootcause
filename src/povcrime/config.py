"""Project configuration loader and typed analysis definitions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_LANE_TIERS = {"primary", "secondary", "exploratory"}
_ALLOWED_LANE_METHODS = {
    "baseline",
    "border",
    "dml",
    "falsification",
    "overlap",
    "robustness",
    "staggered",
    "support_trim",
}
_EVENT_TIMING_REQUIRED_METHODS = {"staggered"}

LaneTier = Literal["primary", "secondary", "exploratory"]
LaneMethod = Literal[
    "baseline",
    "border",
    "dml",
    "falsification",
    "overlap",
    "robustness",
    "staggered",
    "support_trim",
]


@dataclass(frozen=True)
class PanelConfig:
    """Typed panel metadata loaded from config."""

    unit: str = "county_fips x year"
    cluster_var: str = "state_fips"


@dataclass(frozen=True)
class AnalysisLane:
    """One canonical treatment/outcome lane."""

    slug: str
    title: str
    family: str
    treatment: str
    outcome: str
    tier: LaneTier
    methods: tuple[LaneMethod, ...] = ()
    event_col: str | None = None
    event_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.slug:
            raise ValueError("analysis lane slug must be non-empty.")
        if not self.title:
            raise ValueError(f"analysis lane {self.slug!r} must define a title.")
        if not self.family:
            raise ValueError(f"analysis lane {self.slug!r} must define a family.")
        if not self.treatment:
            raise ValueError(f"analysis lane {self.slug!r} must define a treatment.")
        if not self.outcome:
            raise ValueError(f"analysis lane {self.slug!r} must define an outcome.")
        if self.tier not in _ALLOWED_LANE_TIERS:
            raise ValueError(
                f"analysis lane {self.slug!r} has invalid tier {self.tier!r}."
            )
        invalid_methods = sorted(set(self.methods) - _ALLOWED_LANE_METHODS)
        if invalid_methods:
            raise ValueError(
                f"analysis lane {self.slug!r} has invalid methods: {invalid_methods}."
            )
        if not self.methods:
            raise ValueError(f"analysis lane {self.slug!r} must define at least one method.")
        if self.event_threshold is not None and self.event_col is None:
            raise ValueError(
                f"analysis lane {self.slug!r} defines event_threshold without event_col."
            )
        if self.event_col is not None and self.event_threshold is None:
            raise ValueError(
                f"analysis lane {self.slug!r} defines event_col without event_threshold."
            )
        if set(self.methods) & _EVENT_TIMING_REQUIRED_METHODS and self.event_col is None:
            required_for = ", ".join(sorted(set(self.methods) & _EVENT_TIMING_REQUIRED_METHODS))
            raise ValueError(
                f"analysis lane {self.slug!r} must define event_col and event_threshold "
                f"for methods requiring event timing: {required_for}."
            )

    @property
    def display_priority(self) -> LaneTier:
        return self.tier

    @property
    def headline_eligible(self) -> bool:
        return self.tier == "primary"

    @property
    def outcome_family(self) -> str:
        if self.outcome == "violent_crime_rate":
            return "violent_crime"
        if self.outcome == "property_crime_rate":
            return "property_crime"
        return self.outcome

    @property
    def falsification_label(self) -> str:
        if "_" not in self.slug:
            return self.slug
        return self.slug.rsplit("_", 1)[0]


@dataclass
class ProjectConfig:
    """Centralised project configuration."""

    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    study_name: str = "Causal Poverty-Crime Analysis"
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    interim_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    start_year: int = 2000
    end_year: int = 2024
    panel: PanelConfig = field(default_factory=PanelConfig)
    analysis_lanes: tuple[AnalysisLane, ...] = field(default_factory=tuple)
    bidirectional_lanes: tuple[AnalysisLane, ...] = field(default_factory=tuple)

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
        self._validate_analysis_lanes()

    @property
    def treatments(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(lane.treatment for lane in self.analysis_lanes))

    @property
    def outcomes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(lane.outcome for lane in self.analysis_lanes))

    def _validate_analysis_lanes(self) -> None:
        policy_seen = self._validate_lane_set(self.analysis_lanes, "analysis")
        bidirectional_seen = self._validate_lane_set(
            self.bidirectional_lanes,
            "bidirectional",
        )
        overlap = sorted(policy_seen & bidirectional_seen)
        if overlap:
            raise ValueError(
                "Policy and bidirectional lane slugs must be disjoint; "
                f"overlap={overlap}"
            )

    @staticmethod
    def _validate_lane_set(
        lanes: tuple[AnalysisLane, ...],
        label: str,
    ) -> set[str]:
        seen: set[str] = set()
        for lane in lanes:
            if lane.slug in seen:
                raise ValueError(f"Duplicate {label} lane slug: {lane.slug}")
            seen.add(lane.slug)
        return seen


_config: ProjectConfig | None = None


def reset_config_cache() -> None:
    """Reset the cached default configuration."""
    global _config  # noqa: PLW0603
    _config = None


def load_project_config(project_root: str | Path | None = None) -> ProjectConfig:
    """Load project configuration from disk without caching."""
    resolved_root = Path(project_root) if project_root is not None else _PROJECT_ROOT

    env_path = resolved_root / ".env"
    load_dotenv(env_path)

    yaml_path = resolved_root / "configs" / "project.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing project configuration file: {yaml_path}")
    with open(yaml_path, encoding="utf-8") as fh:
        yaml_data = yaml.safe_load(fh) or {}

    panel_data = _require_mapping(yaml_data.get("panel"), "panel")
    analysis_lanes_raw = _require_list(yaml_data.get("analysis_lanes"), "analysis_lanes")
    if not analysis_lanes_raw:
        raise ValueError("analysis_lanes must define at least one lane.")
    analysis_lanes = tuple(
        _parse_analysis_lane(item, index)
        for index, item in enumerate(analysis_lanes_raw)
    )
    bidirectional_lanes = tuple(
        _parse_analysis_lane(item, index, section="bidirectional_lanes")
        for index, item in enumerate(
            _require_list(yaml_data.get("bidirectional_lanes", []), "bidirectional_lanes")
        )
    )

    return ProjectConfig(
        project_root=resolved_root,
        study_name=str(yaml_data.get("study_name", "Causal Poverty-Crime Analysis")),
        start_year=int(yaml_data.get("start_year", 2000)),
        end_year=int(yaml_data.get("end_year", 2024)),
        panel=PanelConfig(
            unit=_require_str(panel_data, "unit", "panel"),
            cluster_var=_require_str(panel_data, "cluster_var", "panel"),
        ),
        analysis_lanes=analysis_lanes,
        bidirectional_lanes=bidirectional_lanes,
        census_api_key=os.getenv("CENSUS_API_KEY", ""),
        bls_api_key=os.getenv("BLS_API_KEY", ""),
        bea_api_key=os.getenv("BEA_API_KEY", ""),
        fred_api_key=os.getenv("FRED_API_KEY", ""),
        ipums_api_key=os.getenv("IPUMS_API_KEY", ""),
        noaa_api_key=os.getenv("NOAA_API_KEY", ""),
        usda_quickstats_api_key=os.getenv("USDA_QUICKSTATS_API_KEY", ""),
    )


def get_config() -> ProjectConfig:
    """Return a cached :class:`ProjectConfig` for the default project root."""
    global _config  # noqa: PLW0603
    if _config is None:
        _config = load_project_config()
    return _config


def _parse_analysis_lane(raw: object, index: int, *, section: str = "analysis_lanes") -> AnalysisLane:
    mapping = _require_mapping(raw, f"{section}[{index}]")
    methods = tuple(
        _coerce_method(value, f"{section}[{index}].methods[{method_index}]")
        for method_index, value in enumerate(
            _require_list(mapping.get("methods", []), f"{section}[{index}].methods")
        )
    )
    return AnalysisLane(
        slug=_require_str(mapping, "slug", f"{section}[{index}]"),
        title=_require_str(mapping, "title", f"{section}[{index}]"),
        family=_require_str(mapping, "family", f"{section}[{index}]"),
        treatment=_require_str(mapping, "treatment", f"{section}[{index}]"),
        outcome=_require_str(mapping, "outcome", f"{section}[{index}]"),
        tier=_coerce_tier(
            _require_str(mapping, "tier", f"{section}[{index}]"),
            f"{section}[{index}].tier",
        ),
        methods=methods,
        event_col=_optional_str(mapping.get("event_col")),
        event_threshold=_optional_float(mapping.get("event_threshold")),
    )


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping.")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list.")
    return value


def _require_str(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must define non-empty string field {key!r}.")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("Optional string field must be a non-empty string when provided.")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError("Optional float field must be numeric when provided.")


def _coerce_tier(value: str, label: str) -> LaneTier:
    if value not in _ALLOWED_LANE_TIERS:
        raise ValueError(f"{label} must be one of {sorted(_ALLOWED_LANE_TIERS)}.")
    return value  # type: ignore[return-value]


def _coerce_method(value: object, label: str) -> LaneMethod:
    if not isinstance(value, str) or value not in _ALLOWED_LANE_METHODS:
        raise ValueError(f"{label} must be one of {sorted(_ALLOWED_LANE_METHODS)}.")
    return value  # type: ignore[return-value]
