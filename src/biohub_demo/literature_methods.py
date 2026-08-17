"""Executable coarse-screening implementations for literature recipes."""

from __future__ import annotations

from typing import Any

from .hyper_search import _signature
from .literature_registry import LiteratureMethod
from .proposal_postprocess import (
    division_specialist,
    division_value_gate,
    ensemble_graphs,
    expected_jaccard_prune,
    future_supported_divisions,
    metric_aware_track_prune,
    motion_refine,
    node_count_calibrate,
)
from .tracking_variants import build_variant


def _uot(**changes: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "tracker": "unbalanced_ot",
        "ot_mode": "hybrid",
        "ot_device": "auto",
        "detection_threshold": 0.95,
        "edge_threshold": 0.01,
        "max_distance_um": 10.0,
        "distance_weight": 1.0,
        "transformer_weight": 1.0,
        "detection_weight": 0.25,
        "entropy_epsilon": 0.05,
        "source_mass_penalty": 0.5,
        "target_mass_penalty": 0.5,
        "sinkhorn_iterations": 100,
        "sinkhorn_tolerance": 1e-4,
        "ot_threshold": 0.05,
        "top_k_source": 2,
        "top_k_target": 3,
        "minimum_track_nodes": 4,
        "smoothing_weight": 0.15,
        "max_smoothing_shift_um": 1.5,
        "linked_nodes_only": True,
        "ot_projection": "mutual_top",
    }
    values.update(changes)
    return values


PROFILE_SETTINGS: dict[str, dict[str, Any]] = {
    "uot_base": _uot(),
    "uot_cycle": _uot(cycle_weight=1.0),
    "uot_structured": _uot(structure_weight=0.35, structure_neighbors=5),
    "uot_division": _uot(
        ot_mode="hybrid_division", safe_divisions=True, top_k_source=3,
        division_min_probability=0.25,
    ),
    "uot_structured_division": _uot(
        ot_mode="hybrid_division", safe_divisions=True, top_k_source=3,
        division_min_probability=0.25, structure_weight=0.35,
        structure_neighbors=5,
    ),
    "uot_flow": _uot(
        ot_mode="consensus_ilp", top_k_source=3, ot_projection="hungarian",
        ilp_appearance_weight=0.1, ilp_disappearance_weight=0.1,
        ilp_division_weight=0.8,
    ),
    "uot_permissive": _uot(detection_threshold=0.93, ot_threshold=0.04),
    "uot_strict": _uot(
        detection_threshold=0.97, edge_threshold=0.05, ot_threshold=0.09,
        minimum_track_nodes=4,
    ),
    "distance_filtered": {
        "tracker": "distance_hungarian",
        "detection_threshold": 0.95,
        "edge_threshold": 0.0,
        "strong_edge_threshold": 0.0,
        "top_k_parents": 0,
        "tight_gate_um": 6.0,
        "max_distance_um": 10.0,
        "minimum_track_nodes": 4,
        "smoothing_weight": 0.15,
        "max_smoothing_shift_um": 1.5,
        "linked_nodes_only": True,
    },
    "hybrid_division": {
        "tracker": "hybrid_hungarian",
        "detection_threshold": 0.95,
        "edge_threshold": 0.03,
        "strong_edge_threshold": 0.30,
        "top_k_parents": 3,
        "tight_gate_um": 6.0,
        "max_distance_um": 10.0,
        "probability_weight": 4.0,
        "safe_divisions": True,
        "division_min_probability": 0.25,
        "minimum_track_nodes": 4,
        "smoothing_weight": 0.15,
        "max_smoothing_shift_um": 1.5,
        "linked_nodes_only": True,
    },
}


def _profile_graph(
    profile: str,
    raw_graph,
    raw_candidates,
    cache: dict[str, tuple[Any, dict[str, Any]]],
) -> tuple[Any, dict[str, Any]]:
    settings = PROFILE_SETTINGS[profile]
    key = _signature({"profile": profile, "settings": settings, "dataset": raw_graph.dataset})
    if key not in cache:
        graph, diagnostic = build_variant(raw_graph, raw_candidates, settings)
        cache[key] = (graph.copy(), dict(diagnostic))
    graph, diagnostic = cache[key]
    return graph.copy(), {**diagnostic, "shared_profile": profile, "profile_cache_key": key}


def _combine(method: LiteratureMethod, raw_graph, members: list[Any]):
    if method.combine == "single":
        return members[0].copy(), {"combination": "single"}
    if method.combine in {"weighted", "union_ilp", "agreement", "abstention"}:
        return ensemble_graphs(raw_graph, members, mode=method.combine)
    if method.combine == "division":
        return division_specialist(raw_graph, members[0], members[1:])
    raise ValueError(f"unsupported literature combination: {method.combine}")


def _apply_operator(name: str, graph, raw_candidates):
    if name == "motion":
        result, value = motion_refine(graph, raw_candidates, residual_gate_um=7.0)
        return result, {"motion_edge_delta": value}
    if name == "motion_strict":
        result, value = motion_refine(graph, raw_candidates, residual_gate_um=5.5)
        return result, {"strict_motion_edge_delta": value}
    if name == "future_divisions":
        result, value = future_supported_divisions(graph)
        return result, {"unsupported_divisions_removed": value}
    if name == "division_value":
        result, value = division_value_gate(graph, minimum_probability=0.35)
        return result, {"negative_value_divisions_removed": value}
    if name == "node_count":
        result, value = node_count_calibrate(graph, target_quantile=0.90)
        return result, {"node_count_removed": value}
    if name == "metric_track":
        result, value = metric_aware_track_prune(graph)
        return result, {"components_removed": value}
    if name == "expected_jaccard":
        result, value = expected_jaccard_prune(graph, minimum_probability=0.10)
        return result, {"expected_jaccard_edges_removed": value}
    raise ValueError(f"unsupported literature operator: {name}")


def build_literature_method(
    method: LiteratureMethod,
    raw_graph,
    raw_candidates,
    cache: dict[str, tuple[Any, dict[str, Any]]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build one screening graph while sharing expensive profile computations."""
    cache = cache if cache is not None else {}
    members: list[Any] = []
    member_diagnostics: dict[str, Any] = {}
    for profile in method.profiles:
        graph, diagnostic = _profile_graph(profile, raw_graph, raw_candidates, cache)
        members.append(graph)
        member_diagnostics[profile] = diagnostic
    graph, diagnostic = _combine(method, raw_graph, members)
    operator_diagnostics: dict[str, Any] = {}
    for operator in method.operators:
        graph, values = _apply_operator(operator, graph, raw_candidates)
        operator_diagnostics[operator] = values
    return graph, {
        "fidelity": method.fidelity,
        "paper_faithful": False,
        "profiles": list(method.profiles),
        "member_diagnostics": member_diagnostics,
        "combination_diagnostic": diagnostic,
        "operator_diagnostics": operator_diagnostics,
        "output_nodes": len(graph.nodes),
        "output_edges": len(graph.edges),
        "output_divisions": sum(len(graph.outgoing(node_id)) >= 2 for node_id in graph.nodes),
    }


def build_current_baseline(raw_graph, raw_candidates, cache=None):
    """Return the exact current uot_mutual_top anchor used for comparisons."""
    cache = cache if cache is not None else {}
    return _profile_graph("uot_base", raw_graph, raw_candidates, cache)
