"""Shared validation helpers for backend report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping, NotRequired, TypedDict, cast

import pandas as pd

CrimeDataLevel = Literal["county_fallback", "state_estimate", "missing"]

_VALID_CRIME_DATA_LEVELS = {"county_fallback", "state_estimate", "missing"}
_REQUIRED_RESULTS_ARTIFACT_KEYS = {"final_report", "credibility_summary"}


class PanelSourceSummary(TypedDict):
    """Coverage summary for one panel source."""

    name: str
    columns: list[str]
    non_null_rows: int
    share: float
    min_year_share: float
    max_year_share: float
    missing_rows: int


class PanelSummary(TypedDict):
    """High-level panel statistics embedded in results_summary.json."""

    rows: int
    counties: int
    year_min: int
    year_max: int
    violent_rows: int
    property_rows: int
    low_coverage_rows: int
    crime_data_level: CrimeDataLevel
    available_sources: list[PanelSourceSummary]


class EstimandSummary(TypedDict, total=False):
    """One policy estimand entry in results_summary.json."""

    slug: str
    title: str
    baseline: dict[str, object] | None
    dml: dict[str, object] | None
    causal_forest: dict[str, object] | None
    robustness: dict[str, object] | None
    overlap: dict[str, object] | None
    frontend: dict[str, object] | None


class BidirectionalEstimandSummary(TypedDict, total=False):
    """One exploratory bidirectional estimand entry."""

    label: str
    title: str
    treatment: str
    outcome: str
    baseline_fe: dict[str, object]
    dml: dict[str, object] | None
    overlap: dict[str, object] | None
    robustness: list[dict[str, object]]
    headline: str


class BidirectionalSummary(TypedDict, total=False):
    """Typed contract for exploratory bidirectional summary JSON."""

    generated_date: str
    design: dict[str, object]
    estimands: list[BidirectionalEstimandSummary]


class ResultsSummary(TypedDict):
    """Typed contract for outputs/app/results_summary.json."""

    generated_date: str
    panel: PanelSummary
    artifacts: dict[str, str | None]
    research_lanes: dict[str, list[str]]
    estimands: list[EstimandSummary]
    exploratory: dict[str, object]


class CredibilityCheck(TypedDict):
    """One credibility check entry."""

    name: str
    status: str
    detail: str


class CredibilityLane(TypedDict):
    """One lane entry inside credibility_summary.json."""

    slug: str
    title: NotRequired[str]
    headline_eligible: NotRequired[bool]
    frontend_status: NotRequired[str]
    verdict: NotRequired[str]
    checks: NotRequired[list[CredibilityCheck]]


class CredibilitySummary(TypedDict):
    """Typed contract for outputs/app/credibility_summary.json."""

    generated_date: str
    lanes: list[CredibilityLane]


def load_results_summary(path: Path) -> ResultsSummary:
    """Load and validate the frontend-facing results summary JSON."""
    return validate_results_summary(json.loads(path.read_text(encoding="utf-8")))


def load_credibility_summary(path: Path) -> CredibilitySummary:
    """Load and validate the credibility summary JSON."""
    return validate_credibility_summary(json.loads(path.read_text(encoding="utf-8")))


def load_bidirectional_summary(path: Path) -> BidirectionalSummary:
    """Load and validate the exploratory bidirectional summary JSON."""
    return validate_bidirectional_summary(json.loads(path.read_text(encoding="utf-8")))


def validate_results_summary(raw: object) -> ResultsSummary:
    """Validate the structure of the machine-readable results summary."""
    data = _require_mapping(raw, "results summary")

    generated_date = _require_str(data, "generated_date", "results summary")
    panel = _require_mapping_field(data, "panel", "results summary")
    artifacts = _require_mapping_field(data, "artifacts", "results summary")
    research_lanes = _require_mapping_field(data, "research_lanes", "results summary")
    estimands = _require_list(data, "estimands", "results summary")
    exploratory = _require_mapping_field(data, "exploratory", "results summary")

    normalized_panel = _validate_panel_summary(panel)
    normalized_artifacts = _validate_artifact_map(artifacts)
    normalized_research_lanes = _validate_research_lanes(research_lanes)
    normalized_estimands = _validate_estimands(estimands)
    normalized_exploratory = _validate_exploratory(exploratory)

    return cast(
        ResultsSummary,
        {
            "generated_date": generated_date,
            "panel": normalized_panel,
            "artifacts": normalized_artifacts,
            "research_lanes": normalized_research_lanes,
            "estimands": normalized_estimands,
            "exploratory": normalized_exploratory,
        },
    )


def validate_credibility_summary(raw: object) -> CredibilitySummary:
    """Validate the credibility summary JSON."""
    data = _require_mapping(raw, "credibility summary")
    generated_date = _require_str(data, "generated_date", "credibility summary")
    lanes = _require_list(data, "lanes", "credibility summary")
    normalized_lanes = [_validate_credibility_lane(lane, idx) for idx, lane in enumerate(lanes)]

    return cast(
        CredibilitySummary,
        {
            "generated_date": generated_date,
            "lanes": normalized_lanes,
        },
    )


def validate_bidirectional_summary(raw: object) -> BidirectionalSummary:
    """Validate the exploratory bidirectional summary JSON."""
    data = _require_mapping(raw, "bidirectional summary")
    normalized: dict[str, object] = {}
    if "generated_date" in data and data["generated_date"] is not None:
        normalized["generated_date"] = _require_str(data, "generated_date", "bidirectional summary")
    if "design" in data and data["design"] is not None:
        normalized["design"] = dict(_require_mapping(data["design"], "bidirectional summary['design']"))
    estimands = _require_list(data, "estimands", "bidirectional summary")
    normalized["estimands"] = [
        _validate_bidirectional_estimand(estimand, index)
        for index, estimand in enumerate(estimands)
    ]
    return cast(BidirectionalSummary, normalized)


def infer_crime_data_level(panel: pd.DataFrame) -> CrimeDataLevel:
    """Infer which crime data tier populated a county-year panel."""
    if "county_fips" not in panel.columns:
        return "missing"

    crime_columns = [
        column
        for column in (
            "violent_crime_rate",
            "property_crime_rate",
            "violent_crime_count",
            "property_crime_count",
        )
        if column in panel.columns
    ]
    if not crime_columns:
        return "missing"

    crime_mask = pd.Series(False, index=panel.index)
    for column in crime_columns:
        crime_mask |= panel[column].notna()

    if not crime_mask.any():
        return "missing"

    county_fips = panel.loc[crime_mask, "county_fips"].astype(str).str.zfill(5)
    if not len(county_fips):
        return "missing"
    if county_fips.str.endswith("000").all():
        return "state_estimate"
    return "county_fallback"


def _validate_panel_summary(panel: Mapping[str, Any]) -> PanelSummary:
    rows = _require_int(panel, "rows", "panel summary")
    counties = _require_int(panel, "counties", "panel summary")
    year_min = _require_int(panel, "year_min", "panel summary")
    year_max = _require_int(panel, "year_max", "panel summary")
    violent_rows = _require_int(panel, "violent_rows", "panel summary")
    property_rows = _require_int(panel, "property_rows", "panel summary")
    low_coverage_rows = _require_int(panel, "low_coverage_rows", "panel summary")
    crime_data_level = _require_str(panel, "crime_data_level", "panel summary")
    if crime_data_level not in _VALID_CRIME_DATA_LEVELS:
        raise ValueError(
            "panel summary has invalid crime_data_level "
            f"{crime_data_level!r}; expected one of {sorted(_VALID_CRIME_DATA_LEVELS)}."
        )
    available_sources_raw = _require_list(panel, "available_sources", "panel summary")
    available_sources = [
        _validate_panel_source_summary(source, idx)
        for idx, source in enumerate(available_sources_raw)
    ]
    return cast(
        PanelSummary,
        {
            "rows": rows,
            "counties": counties,
            "year_min": year_min,
            "year_max": year_max,
            "violent_rows": violent_rows,
            "property_rows": property_rows,
            "low_coverage_rows": low_coverage_rows,
            "crime_data_level": cast(CrimeDataLevel, crime_data_level),
            "available_sources": available_sources,
        },
    )


def _validate_panel_source_summary(source: object, index: int) -> PanelSourceSummary:
    mapping = _require_mapping(source, f"panel available_sources[{index}]")
    name = _require_str(mapping, "name", f"panel available_sources[{index}]")
    columns_raw = _require_list(mapping, "columns", f"panel available_sources[{index}]")
    columns = _require_str_list(columns_raw, f"panel available_sources[{index}].columns")
    non_null_rows = _require_int(mapping, "non_null_rows", f"panel available_sources[{index}]")
    share = _require_number(mapping, "share", f"panel available_sources[{index}]")
    min_year_share = _require_number(mapping, "min_year_share", f"panel available_sources[{index}]")
    max_year_share = _require_number(mapping, "max_year_share", f"panel available_sources[{index}]")
    missing_rows = _require_int(mapping, "missing_rows", f"panel available_sources[{index}]")
    return cast(
        PanelSourceSummary,
        {
            "name": name,
            "columns": columns,
            "non_null_rows": non_null_rows,
            "share": share,
            "min_year_share": min_year_share,
            "max_year_share": max_year_share,
            "missing_rows": missing_rows,
        },
    )


def _validate_artifact_map(artifacts: Mapping[str, Any]) -> dict[str, str | None]:
    normalized: dict[str, str | None] = {}
    missing = sorted(_REQUIRED_RESULTS_ARTIFACT_KEYS - set(artifacts))
    if missing:
        raise ValueError(
            "results summary artifacts is missing required keys: "
            f"{missing}"
        )
    for key, value in artifacts.items():
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"results summary artifacts[{key!r}] must be a string or null; "
                f"got {type(value).__name__}."
            )
        normalized[str(key)] = value
    return normalized


def _validate_research_lanes(research_lanes: Mapping[str, Any]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for key, value in research_lanes.items():
        if not isinstance(value, list):
            raise TypeError(
                f"results summary research_lanes[{key!r}] must be a list of strings; "
                f"got {type(value).__name__}."
            )
        normalized[str(key)] = _require_str_list(value, f"research_lanes[{key!r}]")
    return normalized


def _validate_estimands(estimands: list[Any]) -> list[EstimandSummary]:
    normalized: list[EstimandSummary] = []
    for index, estimand in enumerate(estimands):
        mapping = _require_mapping(estimand, f"estimands[{index}]")
        slug = _require_str(mapping, "slug", f"estimands[{index}]")
        title = _require_str(mapping, "title", f"estimands[{index}]")
        item: dict[str, object] = {"slug": slug, "title": title}
        if "baseline" in mapping:
            item["baseline"] = _validate_baseline_summary(mapping["baseline"], index)
        if "dml" in mapping:
            item["dml"] = _validate_dml_summary(mapping["dml"], f"estimands[{index}].dml")
        if "causal_forest" in mapping:
            item["causal_forest"] = _validate_causal_forest_summary(
                mapping["causal_forest"],
                f"estimands[{index}].causal_forest",
            )
        if "robustness" in mapping:
            item["robustness"] = _validate_robustness_summary(
                mapping["robustness"],
                f"estimands[{index}].robustness",
            )
        if "overlap" in mapping:
            item["overlap"] = _validate_overlap_summary(mapping["overlap"], f"estimands[{index}].overlap")
        if "frontend" in mapping:
            item["frontend"] = _validate_frontend_summary(
                mapping["frontend"],
                f"estimands[{index}].frontend",
            )
        normalized.append(cast(EstimandSummary, item))
    return normalized


def _validate_exploratory(exploratory: Mapping[str, Any]) -> dict[str, object]:
    normalized = dict(exploratory)
    bidirectional = exploratory.get("bidirectional_poverty_crime")
    if bidirectional is not None:
        normalized["bidirectional_poverty_crime"] = validate_bidirectional_summary(bidirectional)
    return normalized


def _validate_baseline_summary(value: object, index: int) -> dict[str, object] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, f"estimands[{index}].baseline")
    normalized: dict[str, object] = dict(mapping)
    for key in ("summary_path", "event_study_path", "event_study_plot_path"):
        if key in mapping and mapping[key] is not None:
            normalized[key] = _require_str(mapping, key, f"estimands[{index}].baseline")
    if "treatment_row" in mapping and mapping["treatment_row"] is not None:
        normalized["treatment_row"] = _validate_metric_mapping(
            mapping["treatment_row"],
            f"estimands[{index}].baseline.treatment_row",
        )
    if "pretrend" in mapping:
        normalized["pretrend"] = _validate_pretrend_summary(
            mapping["pretrend"],
            f"estimands[{index}].baseline.pretrend",
        )
    return normalized


def _validate_pretrend_summary(value: object, label: str) -> dict[str, object] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, label)
    normalized: dict[str, object] = dict(mapping)
    for key in ("p_value", "f_stat"):
        if key in mapping and mapping[key] is not None:
            normalized[key] = _require_number(mapping, key, label)
    if "n_pre_coefs" in mapping and mapping["n_pre_coefs"] is not None:
        normalized["n_pre_coefs"] = _require_int(mapping, "n_pre_coefs", label)
    if "pass" in mapping and mapping["pass"] is not None:
        normalized["pass"] = _require_bool(mapping, "pass", label)
    return normalized


def _validate_dml_summary(value: object, label: str) -> dict[str, object] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, label)
    normalized: dict[str, object] = dict(mapping)
    for key in ("theta", "se", "t_stat", "p_value", "ci_lower", "ci_upper"):
        if key in mapping and mapping[key] is not None:
            normalized[key] = _require_number(mapping, key, label)
    if "n_obs" in mapping and mapping["n_obs"] is not None:
        normalized["n_obs"] = _require_int(mapping, "n_obs", label)
    if "n_folds" in mapping and mapping["n_folds"] is not None:
        normalized["n_folds"] = _require_int(mapping, "n_folds", label)
    if "group_col" in mapping and mapping["group_col"] is not None:
        normalized["group_col"] = _require_str(mapping, "group_col", label)
    if "panel_mode" in mapping and mapping["panel_mode"] is not None:
        normalized["panel_mode"] = _require_str(mapping, "panel_mode", label)
    for key in ("treatment", "outcome", "path", "entity_col", "time_col"):
        if key in mapping and mapping[key] is not None:
            normalized[key] = _require_str(mapping, key, label)
    return normalized


def _validate_causal_forest_summary(value: object, label: str) -> dict[str, object] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, label)
    normalized: dict[str, object] = dict(mapping)
    for key in ("ate", "ci_lower", "ci_upper"):
        if key in mapping and mapping[key] is not None:
            normalized[key] = _require_number(mapping, key, label)
    for key in ("path", "importance_path"):
        if key in mapping and mapping[key] is not None:
            normalized[key] = _require_str(mapping, key, label)
    if "top_features" in mapping and mapping["top_features"] is not None:
        features = _require_list(mapping, "top_features", label)
        normalized["top_features"] = [
            _validate_feature_importance(row, f"{label}.top_features[{index}]")
            for index, row in enumerate(features)
        ]
    return normalized


def _validate_feature_importance(value: object, label: str) -> dict[str, object]:
    mapping = _require_mapping(value, label)
    normalized = dict(mapping)
    if "feature" in mapping:
        normalized["feature"] = _require_str(mapping, "feature", label)
    if "importance" in mapping:
        normalized["importance"] = _require_number(mapping, "importance", label)
    return normalized


def _validate_robustness_summary(value: object, label: str) -> dict[str, object] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, label)
    normalized = dict(mapping)
    if "path" in mapping and mapping["path"] is not None:
        normalized["path"] = _require_str(mapping, "path", label)
    if "rows" in mapping and mapping["rows"] is not None:
        rows = _require_list(mapping, "rows", label)
        normalized["rows"] = [
            _validate_metric_mapping(row, f"{label}.rows[{index}]")
            for index, row in enumerate(rows)
        ]
    return normalized


def _validate_overlap_summary(value: object, label: str) -> dict[str, object] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, label)
    normalized = dict(mapping)
    for key in (
        "treatment_min",
        "treatment_p05",
        "treatment_median",
        "treatment_p95",
        "treatment_max",
        "treatment_std",
        "oof_r2",
        "residual_std",
        "residual_to_treatment_std",
        "max_abs_smd",
    ):
        if key in mapping and mapping[key] is not None:
            normalized[key] = _require_number(mapping, key, label)
    for key in ("n_obs", "n_controls", "tail_low_n", "tail_high_n"):
        if key in mapping and mapping[key] is not None:
            normalized[key] = _require_int(mapping, key, label)
    for key in ("treatment", "path", "control_balance_path", "support_bins_path", "panel_mode"):
        if key in mapping and mapping[key] is not None:
            normalized[key] = _require_str(mapping, key, label)
    return normalized


def _validate_frontend_summary(value: object, label: str) -> dict[str, object] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, label)
    normalized = dict(mapping)
    for key in ("treatment_family", "outcome_family", "display_priority", "status", "summary"):
        if key in mapping and mapping[key] is not None:
            normalized[key] = _require_str(mapping, key, label)
    if "headline_eligible" in mapping and mapping["headline_eligible"] is not None:
        normalized["headline_eligible"] = _require_bool(mapping, "headline_eligible", label)
    if "caveats" in mapping and mapping["caveats"] is not None:
        normalized["caveats"] = _require_str_list(
            _require_list(mapping, "caveats", label),
            f"{label}.caveats",
        )
    return normalized


def _validate_bidirectional_estimand(value: object, index: int) -> BidirectionalEstimandSummary:
    mapping = _require_mapping(value, f"bidirectional summary estimands[{index}]")
    normalized: dict[str, object] = {
        "label": _require_str(mapping, "label", f"bidirectional summary estimands[{index}]"),
        "title": _require_str(mapping, "title", f"bidirectional summary estimands[{index}]"),
        "treatment": _require_str(mapping, "treatment", f"bidirectional summary estimands[{index}]"),
        "outcome": _require_str(mapping, "outcome", f"bidirectional summary estimands[{index}]"),
    }
    if "baseline_fe" in mapping:
        normalized["baseline_fe"] = _validate_metric_mapping(
            mapping["baseline_fe"],
            f"bidirectional summary estimands[{index}].baseline_fe",
        )
    if "dml" in mapping:
        normalized["dml"] = _validate_dml_summary(
            mapping["dml"],
            f"bidirectional summary estimands[{index}].dml",
        )
    if "overlap" in mapping:
        normalized["overlap"] = _validate_overlap_summary(
            mapping["overlap"],
            f"bidirectional summary estimands[{index}].overlap",
        )
    if "robustness" in mapping and mapping["robustness"] is not None:
        rows = _require_list(mapping, "robustness", f"bidirectional summary estimands[{index}]")
        normalized["robustness"] = [
            _validate_metric_mapping(row, f"bidirectional summary estimands[{index}].robustness[{row_index}]")
            for row_index, row in enumerate(rows)
        ]
    if "headline" in mapping and mapping["headline"] is not None:
        normalized["headline"] = _require_str(mapping, "headline", f"bidirectional summary estimands[{index}]")
    return cast(BidirectionalEstimandSummary, normalized)


def _validate_metric_mapping(value: object, label: str) -> dict[str, object]:
    mapping = _require_mapping(value, label)
    normalized = dict(mapping)
    for key, item in mapping.items():
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float, str)) or item is None:
            continue
        raise TypeError(
            f"{label}[{key!r}] must be a scalar JSON value; got {type(item).__name__}."
        )
    return normalized


def _validate_credibility_lane(lane: object, index: int) -> CredibilityLane:
    mapping = _require_mapping(lane, f"credibility lanes[{index}]")
    slug = _require_str(mapping, "slug", f"credibility lanes[{index}]")
    normalized: dict[str, object] = {"slug": slug}
    for key in ("title", "frontend_status", "verdict"):
        if key in mapping and mapping[key] is not None:
            normalized[key] = _require_str(mapping, key, f"credibility lanes[{index}]")
    if "headline_eligible" in mapping and mapping["headline_eligible"] is not None:
        normalized["headline_eligible"] = _require_bool(
            mapping, "headline_eligible", f"credibility lanes[{index}]"
        )
    if "checks" in mapping and mapping["checks"] is not None:
        checks_raw = _require_list(mapping, "checks", f"credibility lanes[{index}]")
        normalized["checks"] = [
            _validate_credibility_check(check, check_index, index)
            for check_index, check in enumerate(checks_raw)
        ]
    return cast(CredibilityLane, normalized)


def _validate_credibility_check(check: object, check_index: int, lane_index: int) -> CredibilityCheck:
    mapping = _require_mapping(check, f"credibility lanes[{lane_index}].checks[{check_index}]")
    return cast(
        CredibilityCheck,
        {
            "name": _require_str(mapping, "name", f"credibility lanes[{lane_index}].checks[{check_index}]"),
            "status": _require_str(mapping, "status", f"credibility lanes[{lane_index}].checks[{check_index}]"),
            "detail": _require_str(mapping, "detail", f"credibility lanes[{lane_index}].checks[{check_index}]"),
        },
    )


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping; got {type(value).__name__}.")
    return value


def _require_mapping_field(data: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    if key not in data:
        raise ValueError(f"{label} is missing required key: {key!r}.")
    return _require_mapping(data[key], f"{label}[{key!r}]")


def _require_list(data: Mapping[str, Any], key: str, label: str) -> list[Any]:
    if key not in data:
        raise ValueError(f"{label} is missing required key: {key!r}.")
    value = data[key]
    if not isinstance(value, list):
        raise TypeError(
            f"{label}[{key!r}] must be a list; got {type(value).__name__}."
        )
    return value


def _require_str(data: Mapping[str, Any], key: str, label: str) -> str:
    if key not in data:
        raise ValueError(f"{label} is missing required key: {key!r}.")
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(
            f"{label}[{key!r}] must be a string; got {type(value).__name__}."
        )
    return value


def _require_int(data: Mapping[str, Any], key: str, label: str) -> int:
    if key not in data:
        raise ValueError(f"{label} is missing required key: {key!r}.")
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            f"{label}[{key!r}] must be an integer; got {type(value).__name__}."
        )
    return value


def _require_number(data: Mapping[str, Any], key: str, label: str) -> float:
    if key not in data:
        raise ValueError(f"{label} is missing required key: {key!r}.")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{label}[{key!r}] must be a number; got {type(value).__name__}."
        )
    return float(value)


def _require_bool(data: Mapping[str, Any], key: str, label: str) -> bool:
    if key not in data:
        raise ValueError(f"{label} is missing required key: {key!r}.")
    value = data[key]
    if not isinstance(value, bool):
        raise TypeError(
            f"{label}[{key!r}] must be a boolean; got {type(value).__name__}."
        )
    return value


def _require_str_list(values: list[Any], label: str) -> list[str]:
    result: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(
                f"{label}[{index}] must be a string; got {type(value).__name__}."
            )
        result.append(value)
    return result
