"""Graph-level operators used by the full proposal benchmark."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Iterable

import numpy as np

from .candidates import Candidate
from .graph import Edge, Node, TrackGraph, physical_distance
from .optimizer import solve_forest

SCALE = (1.625, 0.40625, 0.40625)


def _edge_graph(raw_graph: TrackGraph, edges: Iterable[Edge]) -> TrackGraph:
    edge_values = list(edges)
    used = {value for edge in edge_values for value in (edge.source_id, edge.target_id)}
    graph = TrackGraph(
        raw_graph.dataset,
        {node_id: raw_graph.nodes[node_id] for node_id in used if node_id in raw_graph.nodes},
        {},
    )
    for edge in edge_values:
        if edge.source_id in graph.nodes and edge.target_id in graph.nodes:
            graph.add_edge(edge)
    return graph


def motion_refine(
    graph: TrackGraph,
    candidates: list[Candidate],
    *,
    residual_gate_um: float = 7.0,
) -> tuple[TrackGraph, int]:
    """Use a three-frame constant-velocity residual to replace weak links."""
    result = graph.copy()
    predecessor = {
        edge.target_id: edge.source_id for edge in result.edges.values()
        if len(result.incoming(edge.target_id)) == 1
    }
    incoming = {edge.target_id for edge in result.edges.values()}
    outgoing = {edge.source_id for edge in result.edges.values()}
    removed = 0
    for edge in list(result.edges.values()):
        previous_id = predecessor.get(edge.source_id)
        if previous_id is None:
            continue
        previous = result.nodes[previous_id]
        source = result.nodes[edge.source_id]
        target = result.nodes[edge.target_id]
        predicted = source.zyx + (source.zyx - previous.zyx)
        residual = physical_distance(predicted, target, SCALE)
        if residual > residual_gate_um and edge.probability < 0.8:
            result.remove_edge(edge.source_id, edge.target_id)
            incoming.discard(edge.target_id)
            outgoing.discard(edge.source_id)
            removed += 1

    by_source: dict[int, list[Candidate]] = defaultdict(list)
    for item in candidates:
        if item.source_id in result.nodes and item.target_id in result.nodes:
            by_source[item.source_id].append(item)
    added = 0
    for source_id, values in by_source.items():
        if source_id in outgoing:
            continue
        previous_id = predecessor.get(source_id)
        source = result.nodes[source_id]
        previous = result.nodes.get(previous_id) if previous_id is not None else None
        scored: list[tuple[float, Candidate]] = []
        for item in values:
            if item.target_id in incoming:
                continue
            residual = item.distance_um
            if previous is not None:
                predicted = source.zyx + (source.zyx - previous.zyx)
                residual = physical_distance(predicted, result.nodes[item.target_id], SCALE)
            score = residual - 2.0 * item.probability
            scored.append((score, item))
        if scored:
            _, best = min(scored, key=lambda pair: (pair[0], pair[1].target_id))
            if best.distance_um <= residual_gate_um:
                result.add_edge(Edge(
                    best.source_id, best.target_id, best.probability, best.distance_um
                ))
                incoming.add(best.target_id)
                outgoing.add(best.source_id)
                added += 1
    return _edge_graph(result, result.edges.values()), added - removed


def future_supported_divisions(graph: TrackGraph) -> tuple[TrackGraph, int]:
    """Remove a weaker second-child edge when the two branches lack future support."""
    result = graph.copy()
    removed = 0
    max_t = max((node.t for node in result.nodes.values()), default=-1)
    for parent in list(result.nodes):
        children = sorted(result.outgoing(parent), key=lambda edge: -edge.probability)
        if len(children) < 2:
            continue
        supported = [
            edge for edge in children
            if result.nodes[edge.target_id].t >= max_t - 1 or bool(result.outgoing(edge.target_id))
        ]
        if len(supported) < 2:
            for edge in children[1:]:
                result.remove_edge(edge.source_id, edge.target_id)
                removed += 1
    return _edge_graph(result, result.edges.values()), removed


def expected_jaccard_prune(graph: TrackGraph, minimum_probability: float = 0.0) -> tuple[TrackGraph, int]:
    """Apply the proposal's local expected-Jaccard inclusion threshold."""
    edges = list(graph.edges.values())
    if not edges:
        return graph.copy(), 0
    expected_tp = sum(edge.probability for edge in edges)
    denominator = len(edges)
    threshold = expected_tp / max(denominator + 1.0 + expected_tp, 1.0)
    threshold = max(threshold, minimum_probability)
    kept = [edge for edge in edges if edge.probability > threshold]
    return _edge_graph(graph, kept), len(edges) - len(kept)


def division_value_gate(graph: TrackGraph, minimum_probability: float = 0.35) -> tuple[TrackGraph, int]:
    """Keep a second daughter only when its estimated score value is positive."""
    result = graph.copy()
    removed = 0
    for parent in list(result.nodes):
        children = sorted(result.outgoing(parent), key=lambda edge: -edge.probability)
        for edge in children[1:]:
            expected_division_gain = 0.1 * edge.probability
            expected_edge_risk = 1.0 - edge.probability
            if edge.probability < minimum_probability or expected_division_gain <= 0.1 * expected_edge_risk:
                result.remove_edge(edge.source_id, edge.target_id)
                removed += 1
    return _edge_graph(result, result.edges.values()), removed


def metric_aware_track_prune(
    graph: TrackGraph,
    *,
    minimum_nodes: int = 3,
    minimum_mean_probability: float = 0.35,
) -> tuple[TrackGraph, int]:
    """Prune components by expected contribution instead of length alone."""
    neighbours: dict[int, set[int]] = defaultdict(set)
    for edge in graph.edges.values():
        neighbours[edge.source_id].add(edge.target_id)
        neighbours[edge.target_id].add(edge.source_id)
    seen: set[int] = set()
    keep: set[int] = set()
    removed_components = 0
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
        edges = [
            edge
            for node_id in component
            for edge in graph.outgoing(node_id)
        ]
        mean_probability = float(np.mean([edge.probability for edge in edges])) if edges else 0.0
        has_division = any(len(graph.outgoing(node_id)) >= 2 for node_id in component)
        if has_division or (
            len(component) >= minimum_nodes and mean_probability >= minimum_mean_probability
        ):
            keep.update(component)
        else:
            removed_components += 1
    result = TrackGraph(
        graph.dataset,
        {node_id: node for node_id, node in graph.nodes.items() if node_id in keep},
        {key: edge for key, edge in graph.edges.items()
         if edge.source_id in keep and edge.target_id in keep},
    )
    return result, removed_components


def node_count_calibrate(
    graph: TrackGraph,
    *,
    target_quantile: float = 0.90,
) -> tuple[TrackGraph, int]:
    """Unsupervised per-movie count calibration using confidence quantiles."""
    if not graph.nodes:
        return graph.copy(), 0
    by_time: dict[int, list[Node]] = defaultdict(list)
    for node in graph.nodes.values():
        by_time[node.t].append(node)
    counts = np.asarray([len(nodes) for nodes in by_time.values()], dtype=np.float64)
    target = int(max(1, np.quantile(counts, target_quantile)))
    keep: set[int] = set()
    for nodes in by_time.values():
        keep.update(
            node.node_id for node in sorted(nodes, key=lambda item: -item.confidence)[:target]
        )
    result = TrackGraph(
        graph.dataset,
        {node_id: node for node_id, node in graph.nodes.items() if node_id in keep},
        {key: edge for key, edge in graph.edges.items()
         if edge.source_id in keep and edge.target_id in keep},
    )
    return result, len(graph.nodes) - len(result.nodes)


def ensemble_graphs(
    raw_graph: TrackGraph,
    members: list[TrackGraph],
    *,
    mode: str,
    weights: list[float] | None = None,
) -> tuple[TrackGraph, dict[str, int | float | str]]:
    """Combine dependency graphs while enforcing a valid lineage forest."""
    if not members:
        raise ValueError("ensemble requires at least one member")
    weights = weights or [1.0] * len(members)
    if len(weights) != len(members):
        raise ValueError("ensemble member/weight count mismatch")
    edge_votes: Counter[tuple[int, int]] = Counter()
    edge_weight: dict[tuple[int, int], float] = defaultdict(float)
    edge_probability: dict[tuple[int, int], list[float]] = defaultdict(list)
    edge_distance: dict[tuple[int, int], float] = {}
    for member, weight in zip(members, weights):
        for key, edge in member.edges.items():
            edge_votes[key] += 1
            edge_weight[key] += float(weight)
            edge_probability[key].append(edge.probability)
            edge_distance[key] = edge.distance_um
    total_weight = max(sum(weights), 1e-8)
    candidates = [
        Candidate(
            source,
            target,
            min(1.0, 0.5 * edge_weight[(source, target)] / total_weight
                + 0.5 * float(np.mean(edge_probability[(source, target)]))),
            edge_distance[(source, target)],
        )
        for source, target in edge_votes
    ]
    if mode == "hard":
        minimum = (len(members) + 1) // 2
        candidates = [item for item in candidates if edge_votes[(item.source_id, item.target_id)] >= minimum]
        result, solver = solve_forest(raw_graph, candidates, appearance_weight=0.0, disappearance_weight=0.2)
    elif mode == "weighted":
        candidates = [item for item in candidates if edge_weight[(item.source_id, item.target_id)] >= 0.5 * total_weight]
        result, solver = solve_forest(raw_graph, candidates, appearance_weight=0.0, disappearance_weight=0.2)
    elif mode == "union_ilp":
        result, solver = solve_forest(raw_graph, candidates, appearance_weight=0.1, disappearance_weight=0.1)
    elif mode == "agreement":
        candidates = [item for item in candidates if edge_votes[(item.source_id, item.target_id)] == len(members)]
        result, solver = solve_forest(raw_graph, candidates, appearance_weight=0.0, disappearance_weight=0.3)
    elif mode == "abstention":
        candidates = [item for item in candidates if edge_votes[(item.source_id, item.target_id)] / len(members) >= 0.75]
        result, solver = solve_forest(raw_graph, candidates, appearance_weight=0.0, disappearance_weight=0.2)
    else:
        raise ValueError(f"unknown ensemble mode: {mode}")
    return _edge_graph(result, result.edges.values()), {
        "ensemble_mode": mode,
        "ensemble_members": len(members),
        "union_edges": len(edge_votes),
        "selected_edges": len(result.edges),
        "projection": solver,
    }


def cascade_graphs(raw_graph: TrackGraph, members: list[TrackGraph]) -> tuple[TrackGraph, dict[str, int]]:
    """Lock high-precision edges, then fill only currently orphaned nodes."""
    selected: list[Edge] = []
    incoming: set[int] = set()
    outgoing: set[int] = set()
    for member in members:
        for edge in sorted(member.edges.values(), key=lambda item: -item.probability):
            if edge.target_id in incoming or edge.source_id in outgoing:
                continue
            selected.append(edge)
            incoming.add(edge.target_id)
            outgoing.add(edge.source_id)
    result = _edge_graph(raw_graph, selected)
    return result, {"cascade_members": len(members), "selected_edges": len(selected)}


def division_specialist(
    raw_graph: TrackGraph,
    base: TrackGraph,
    specialists: list[TrackGraph],
) -> tuple[TrackGraph, dict[str, int]]:
    """Keep ordinary edges from the best base and vote only on second daughters."""
    result = base.copy()
    votes: Counter[tuple[int, int]] = Counter()
    examples: dict[tuple[int, int], Edge] = {}
    for specialist in specialists:
        for parent in specialist.nodes:
            children = sorted(specialist.outgoing(parent), key=lambda edge: -edge.probability)
            for edge in children[1:]:
                votes[(edge.source_id, edge.target_id)] += 1
                examples[(edge.source_id, edge.target_id)] = edge
    added = 0
    for key, count in votes.most_common():
        source, target = key
        if count < max(1, (len(specialists) + 1) // 2):
            continue
        if target not in result.nodes and target in raw_graph.nodes:
            result.add_node(raw_graph.nodes[target])
        if source not in result.nodes and source in raw_graph.nodes:
            result.add_node(raw_graph.nodes[source])
        if len(result.incoming(target)) == 0 and len(result.outgoing(source)) == 1:
            result.add_edge(examples[key])
            added += 1
    return _edge_graph(result, result.edges.values()), {"division_edges_added": added}
