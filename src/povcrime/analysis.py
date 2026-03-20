"""Shared analysis-lane registry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from povcrime.config import AnalysisLane, LaneMethod, ProjectConfig, get_config


@dataclass(frozen=True)
class PolicyEventDefinition:
    """Canonical event-timing definition for a treatment family."""

    treatment_col: str
    output_col: str
    change_threshold: float


def get_analysis_lanes(
    *,
    config: ProjectConfig | None = None,
    method: LaneMethod | None = None,
    tiers: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
) -> list[AnalysisLane]:
    """Return canonical analysis lanes filtered by method, tier, or family."""
    cfg = config or get_config()
    lanes = list(cfg.analysis_lanes)
    if method is not None:
        lanes = [lane for lane in lanes if method in lane.methods]
    if tiers is not None:
        tier_set = set(tiers)
        lanes = [lane for lane in lanes if lane.tier in tier_set]
    if families is not None:
        family_set = set(families)
        lanes = [lane for lane in lanes if lane.family in family_set]
    return lanes


def get_analysis_lane(
    slug: str,
    *,
    config: ProjectConfig | None = None,
) -> AnalysisLane | None:
    """Return one lane by slug, if configured."""
    for lane in get_analysis_lanes(config=config):
        if lane.slug == slug:
            return lane
    return None


def get_event_definitions(
    *,
    config: ProjectConfig | None = None,
    method: LaneMethod | None = None,
) -> list[PolicyEventDefinition]:
    """Return unique event definitions for lanes that declare event metadata."""
    seen: set[tuple[str, str, float]] = set()
    defs: list[PolicyEventDefinition] = []
    for lane in get_analysis_lanes(config=config, method=method):
        if lane.event_col is None or lane.event_threshold is None:
            continue
        key = (lane.treatment, lane.event_col, lane.event_threshold)
        if key in seen:
            continue
        seen.add(key)
        defs.append(
            PolicyEventDefinition(
                treatment_col=lane.treatment,
                output_col=lane.event_col,
                change_threshold=lane.event_threshold,
            )
        )
    return defs


def get_falsification_treatment_specs(
    *,
    config: ProjectConfig | None = None,
) -> list[dict[str, str]]:
    """Return unique treatment specs for falsification regressions."""
    specs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for lane in get_analysis_lanes(config=config, method="falsification"):
        key = (lane.falsification_label, lane.treatment)
        if key in seen:
            continue
        seen.add(key)
        specs.append({"label": lane.falsification_label, "treatment": lane.treatment})
    return specs


def get_bidirectional_lanes(
    *,
    config: ProjectConfig | None = None,
    method: LaneMethod | None = None,
    tiers: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
) -> list[AnalysisLane]:
    """Return canonical bidirectional lanes filtered by method, tier, or family."""
    cfg = config or get_config()
    lanes = list(cfg.bidirectional_lanes)
    if method is not None:
        lanes = [lane for lane in lanes if method in lane.methods]
    if tiers is not None:
        tier_set = set(tiers)
        lanes = [lane for lane in lanes if lane.tier in tier_set]
    if families is not None:
        family_set = set(families)
        lanes = [lane for lane in lanes if lane.family in family_set]
    return lanes


def get_bidirectional_lane(
    slug: str,
    *,
    config: ProjectConfig | None = None,
) -> AnalysisLane | None:
    """Return one bidirectional lane by slug, if configured."""
    for lane in get_bidirectional_lanes(config=config):
        if lane.slug == slug:
            return lane
    return None
