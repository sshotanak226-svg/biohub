"""Coarse tracking-method variants inspired by the public top notebooks."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .candidates import Candidate
from .graph import Edge, Node, TrackGraph, physical_distance
from .optimal_transport import solve_ot_tracking
from .optimizer import solve_forest
from .postprocess import add_safe_divisions, smooth_non_branching


def graph_from_raw_prediction(
    dataset: str,
    coords: np.ndarray,
    detection_scores: np.ndarray,
    raw_edges: list[tuple[int, int, float, float]],
    scale: tuple[float, float, float],
) -> tuple[TrackGraph, list[Candidate]]:
    if len(coords) != len(detection_scores):
        raise ValueError("coordinate/detection-score count mismatch")
    graph = TrackGraph(dataset)
    for node_id, ((t, z, y, x), score) in enumerate(zip(coords, detection_scores)):
        graph.add_node(Node(node_id, int(t), float(z), float(y), float(x), float(score)))
    candidates: list[Candidate] = []
    for source, target, probability, _ in raw_edges:
        distance = physical_distance(graph.nodes[source], graph.nodes[target], scale)
        candidates.append(Candidate(int(source), int(target), float(probability), distance))
    return graph, candidates


def _candidate_subset(
    graph: TrackGraph,
    candidates: list[Candidate],
    *,
    detection_threshold: float,
    edge_threshold: float,
    strong_edge_threshold: float,
    top_k_parents: int,
    max_distance_um: float,
    distance_only: bool,
) -> tuple[TrackGraph, list[Candidate]]:
    selected_nodes = {
        node_id: node for node_id, node in graph.nodes.items()
        if node.confidence >= detection_threshold
    }
    output = TrackGraph(graph.dataset, selected_nodes, {})
    if distance_only:
        pool: list[Candidate] = []
        max_t = max((node.t for node in selected_nodes.values()), default=-1)
        scale = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
        for t in range(max_t):
            sources, targets = output.nodes_at(t), output.nodes_at(t + 1)
            if not sources or not targets:
                continue
            source_xyz = np.stack([node.zyx for node in sources]) * scale
            target_xyz = np.stack([node.zyx for node in targets]) * scale
            target_tree = cKDTree(target_xyz)
            neighbours = target_tree.query_ball_point(source_xyz, r=max_distance_um)
            for source_index, target_indexes in enumerate(neighbours):
                for target_index in target_indexes:
                    distance = float(np.linalg.norm(
                        source_xyz[source_index] - target_xyz[target_index]
                    ))
                    probability = float(np.exp(-0.5 * (distance / 3.5) ** 2))
                    pool.append(Candidate(
                        sources[source_index].node_id,
                        targets[target_index].node_id,
                        probability,
                        distance,
                    ))
        return output, pool

    eligible = [
        candidate for candidate in candidates
        if candidate.source_id in selected_nodes
        and candidate.target_id in selected_nodes
        and candidate.probability >= edge_threshold
        and candidate.distance_um <= max_distance_um
    ]
    strong = [candidate for candidate in eligible if candidate.probability >= strong_edge_threshold]
    keep = {(item.source_id, item.target_id): item for item in strong}
    by_target: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in eligible:
        by_target[candidate.target_id].append(candidate)
    for values in by_target.values():
        values.sort(key=lambda item: (-item.probability, item.distance_um))
        for candidate in values[:top_k_parents]:
            keep[(candidate.source_id, candidate.target_id)] = candidate
    return output, list(keep.values())


def _greedy(graph: TrackGraph, candidates: list[Candidate]) -> TrackGraph:
    output = graph.copy()
    indegree: dict[int, int] = defaultdict(int)
    outdegree: dict[int, int] = defaultdict(int)
    for item in sorted(candidates, key=lambda value: (-value.probability, value.distance_um)):
        if indegree[item.target_id] >= 1 or outdegree[item.source_id] >= 2:
            continue
        output.add_edge(Edge(item.source_id, item.target_id, item.probability, item.distance_um))
        indegree[item.target_id] += 1
        outdegree[item.source_id] += 1
    return output


def _hungarian_pass(
    sources: list[Node],
    targets: list[Node],
    candidates: dict[tuple[int, int], Candidate],
    *,
    gate_um: float,
    probability_weight: float,
) -> list[Candidate]:
    if not sources or not targets:
        return []
    large = 1e6
    cost = np.full((len(sources), len(targets)), large, dtype=np.float64)
    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            candidate = candidates.get((source.node_id, target.node_id))
            if candidate is None or candidate.distance_um > gate_um:
                continue
            cost[i, j] = candidate.distance_um - probability_weight * candidate.probability
    rows, cols = linear_sum_assignment(cost)
    return [
        candidates[(sources[i].node_id, targets[j].node_id)]
        for i, j in zip(rows, cols) if cost[i, j] < large
    ]


def _two_pass_hungarian(
    graph: TrackGraph,
    candidates: list[Candidate],
    *,
    tight_gate_um: float,
    max_distance_um: float,
    probability_weight: float,
) -> TrackGraph:
    output = graph.copy()
    by_pair = {(item.source_id, item.target_id): item for item in candidates}
    max_t = max((node.t for node in graph.nodes.values()), default=-1)
    for t in range(max_t):
        sources, targets = graph.nodes_at(t), graph.nodes_at(t + 1)
        first = _hungarian_pass(
            sources, targets, by_pair,
            gate_um=min(tight_gate_um, max_distance_um),
            probability_weight=probability_weight,
        )
        used_sources = {item.source_id for item in first}
        used_targets = {item.target_id for item in first}
        second = _hungarian_pass(
            [node for node in sources if node.node_id not in used_sources],
            [node for node in targets if node.node_id not in used_targets],
            by_pair,
            gate_um=max_distance_um,
            probability_weight=probability_weight,
        )
        for item in first + second:
            output.add_edge(Edge(item.source_id, item.target_id, item.probability, item.distance_um))
    return output


def _remove_short_components(graph: TrackGraph, minimum_nodes: int) -> TrackGraph:
    if minimum_nodes <= 1:
        return graph.copy()
    neighbours: dict[int, set[int]] = defaultdict(set)
    outdegree: dict[int, int] = defaultdict(int)
    for edge in graph.edges.values():
        neighbours[edge.source_id].add(edge.target_id)
        neighbours[edge.target_id].add(edge.source_id)
        outdegree[edge.source_id] += 1
    keep: set[int] = set()
    seen: set[int] = set()
    for start in graph.nodes:
        if start in seen:
            continue
        queue = deque([start])
        component: set[int] = set()
        while queue:
            node_id = queue.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            component.add(node_id)
            queue.extend(neighbours[node_id] - seen)
        if len(component) >= minimum_nodes or any(outdegree[node_id] >= 2 for node_id in component):
            keep.update(component)
    return TrackGraph(
        graph.dataset,
        {node_id: node for node_id, node in graph.nodes.items() if node_id in keep},
        {key: edge for key, edge in graph.edges.items()
         if edge.source_id in keep and edge.target_id in keep},
    )


def _linked_nodes_only(graph: TrackGraph) -> TrackGraph:
    used = {node_id for edge in graph.edges.values() for node_id in (edge.source_id, edge.target_id)}
    return TrackGraph(
        graph.dataset,
        {node_id: node for node_id, node in graph.nodes.items() if node_id in used},
        dict(graph.edges),
    )


def build_variant(
    raw_graph: TrackGraph,
    raw_candidates: list[Candidate],
    settings: dict[str, Any],
) -> tuple[TrackGraph, dict[str, Any]]:
    """Build one deliberately coarse method combination from shared raw inference."""
    tracker = str(settings["tracker"])
    solver_diagnostic: dict[str, Any] = {}
    if tracker == "unbalanced_ot":
        selected_nodes = {
            node_id: node for node_id, node in raw_graph.nodes.items()
            if node.confidence >= float(settings["detection_threshold"])
        }
        graph = TrackGraph(raw_graph.dataset, selected_nodes, {})
        candidates = [
            item for item in raw_candidates
            if item.source_id in selected_nodes
            and item.target_id in selected_nodes
            and item.distance_um <= float(settings.get("max_distance_um", 10.0))
        ]
        result, candidates, solver_diagnostic = solve_ot_tracking(
            graph, candidates, settings, (1.625, 0.40625, 0.40625)
        )
        solver = f"unbalanced-ot:{settings.get('ot_mode', 'hybrid')}"
    else:
        graph, candidates = _candidate_subset(
            raw_graph,
            raw_candidates,
            detection_threshold=float(settings["detection_threshold"]),
            edge_threshold=float(settings.get("edge_threshold", 0.01)),
            strong_edge_threshold=float(settings.get("strong_edge_threshold", 0.30)),
            top_k_parents=int(settings.get("top_k_parents", 3)),
            max_distance_um=float(settings.get("max_distance_um", 10.0)),
            distance_only=tracker == "distance_hungarian",
        )
    if tracker == "greedy":
        result = _greedy(graph, candidates)
        solver = "greedy-probability"
    elif tracker == "ilp":
        result, solver = solve_forest(
            graph, candidates,
            edge_weight=float(settings.get("ilp_edge_weight", -1.0)),
            appearance_weight=float(settings.get("ilp_appearance_weight", 0.1)),
            disappearance_weight=float(settings.get("ilp_disappearance_weight", 0.1)),
            division_weight=float(settings.get("ilp_division_weight", 1.0)),
        )
    elif tracker in {"distance_hungarian", "hybrid_hungarian"}:
        result = _two_pass_hungarian(
            graph,
            candidates,
            tight_gate_um=float(settings.get("tight_gate_um", 6.0)),
            max_distance_um=float(settings.get("max_distance_um", 10.0)),
            probability_weight=(
                0.0 if tracker == "distance_hungarian"
                else float(settings.get("probability_weight", 4.0))
            ),
        )
        solver = tracker
    elif tracker != "unbalanced_ot":
        raise ValueError(f"unknown tracker: {tracker}")

    divisions_added = 0
    if bool(settings.get("safe_divisions", False)):
        result, divisions_added = add_safe_divisions(
            result, candidates, (1.625, 0.40625, 0.40625),
            parent_child_um=float(settings.get("division_parent_child_um", 7.5)),
            child_child_um=float(settings.get("division_child_child_um", 8.5)),
            min_probability=float(settings.get("division_min_probability", 0.20)),
        )
    result = _remove_short_components(result, int(settings.get("minimum_track_nodes", 1)))
    smoothed = 0
    if float(settings.get("smoothing_weight", 0.0)) > 0:
        result, smoothed = smooth_non_branching(
            result,
            (1.625, 0.40625, 0.40625),
            weight=float(settings["smoothing_weight"]),
            max_shift_um=float(settings.get("max_smoothing_shift_um", 1.5)),
        )
    if bool(settings.get("linked_nodes_only", True)):
        result = _linked_nodes_only(result)
    return result, {
        "solver": solver,
        "eligible_nodes": len(graph.nodes),
        "candidate_edges": len(candidates),
        "output_nodes": len(result.nodes),
        "output_edges": len(result.edges),
        "divisions_added": divisions_added,
        "smoothed_nodes": smoothed,
        **solver_diagnostic,
    }
