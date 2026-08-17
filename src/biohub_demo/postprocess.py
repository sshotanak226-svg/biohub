"""Bounded graph repair: one-frame gaps, safe divisions and non-branch smoothing."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .candidates import Candidate
from .graph import Edge, Node, TrackGraph, physical_distance


def close_one_frame_gaps(
    graph: TrackGraph,
    image: np.ndarray | None,
    scale: tuple[float, float, float],
    max_added_fraction: float = 0.005,
    max_added_absolute: int = 180,
    max_link_um: float = 6.0,
) -> tuple[TrackGraph, int]:
    output = graph.copy()
    budget = min(max_added_absolute, max(1, int(len(output.nodes) * max_added_fraction)))
    added = 0
    max_t = max((n.t for n in output.nodes.values()), default=-1)
    for t in range(max_t - 1):
        if added >= budget:
            break
        ends = [node for node in output.nodes_at(t) if not output.outgoing(node.node_id)]
        starts = [node for node in output.nodes_at(t + 2) if not output.incoming(node.node_id)]
        proposals: list[tuple[float, Node, Node]] = []
        for source in ends:
            for target in starts:
                distance = physical_distance(source, target, scale)
                if distance <= 2 * max_link_um:
                    proposals.append((distance, source, target))
        used_sources: set[int] = set()
        used_targets: set[int] = set()
        for _, source, target in sorted(proposals, key=lambda item: item[0]):
            if added >= budget or source.node_id in used_sources or target.node_id in used_targets:
                continue
            midpoint = (source.zyx + target.zyx) / 2.0
            existing = [
                node for node in output.nodes_at(t + 1)
                if physical_distance(node, midpoint, scale) <= 2.5 and not output.incoming(node.node_id)
            ]
            if existing:
                middle = min(existing, key=lambda node: physical_distance(node, midpoint, scale))
            else:
                if image is None:
                    continue
                refined = _refine_local_peak(np.asarray(image[t + 1]), midpoint)
                if physical_distance(refined, midpoint, scale) > 2.5:
                    continue
                middle = Node(output.next_node_id(), t + 1, *refined, confidence=0.5)
            first = physical_distance(source, middle, scale)
            second = physical_distance(middle, target, scale)
            if first <= max_link_um and second <= max_link_um:
                if middle.node_id not in output.nodes:
                    output.add_node(middle)
                    added += 1
                output.add_edge(Edge(source.node_id, middle.node_id, 0.5, first))
                output.add_edge(Edge(middle.node_id, target.node_id, 0.5, second))
                used_sources.add(source.node_id)
                used_targets.add(target.node_id)
    return output, added


def _refine_local_peak(frame: np.ndarray, midpoint: np.ndarray) -> np.ndarray:
    center = np.rint(midpoint).astype(int)
    radii = np.asarray((1, 4, 4))
    lower = np.maximum(0, center - radii)
    upper = np.minimum(np.asarray(frame.shape), center + radii + 1)
    patch = frame[tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))]
    if patch.size == 0 or not np.isfinite(patch).all():
        return midpoint
    return lower + np.asarray(np.unravel_index(int(np.argmax(patch)), patch.shape))


def add_safe_divisions(
    graph: TrackGraph,
    candidates: list[Candidate],
    scale: tuple[float, float, float],
    parent_child_um: float = 7.5,
    child_child_um: float = 8.5,
    min_probability: float = 0.20,
) -> tuple[TrackGraph, int]:
    output = graph.copy()
    by_source: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_source[candidate.source_id].append(candidate)
    added = 0
    for source_id in sorted(by_source):
        selected = output.outgoing(source_id)
        if len(selected) != 1:
            continue
        first_id = selected[0].target_id
        first = output.nodes[first_id]
        # Both daughters need a following-frame edge; near the final boundary we do not invent divisions.
        if not output.outgoing(first_id):
            continue
        alternatives = sorted(by_source[source_id], key=lambda c: (-c.probability, c.distance_um))
        for candidate in alternatives:
            if candidate.target_id == first_id or candidate.probability < min_probability:
                continue
            second = output.nodes[candidate.target_id]
            if output.incoming(second.node_id) or not output.outgoing(second.node_id):
                continue
            if candidate.distance_um > parent_child_um:
                continue
            if physical_distance(first, second, scale) > child_child_um:
                continue
            output.add_edge(Edge(source_id, second.node_id, candidate.probability, candidate.distance_um))
            added += 1
            break
    return output, added


def smooth_non_branching(
    graph: TrackGraph,
    scale: tuple[float, float, float],
    weight: float = 0.15,
    max_shift_um: float = 1.5,
) -> tuple[TrackGraph, int]:
    output = graph.copy()
    changed = 0
    for node_id in sorted(graph.nodes):
        incoming, outgoing = graph.incoming(node_id), graph.outgoing(node_id)
        if len(incoming) != 1 or len(outgoing) != 1:
            continue
        previous = graph.nodes[incoming[0].source_id]
        current = graph.nodes[node_id]
        following = graph.nodes[outgoing[0].target_id]
        predicted = (previous.zyx + following.zyx) / 2.0
        proposal = (1.0 - weight) * current.zyx + weight * predicted
        shift_um = physical_distance(current, proposal, scale)
        if shift_um > max_shift_um:
            proposal = current.zyx + (proposal - current.zyx) * (max_shift_um / shift_um)
        if not np.allclose(proposal, current.zyx):
            output.replace_node(node_id, z=float(proposal[0]), y=float(proposal[1]), x=float(proposal[2]))
            changed += 1
    # Refresh edge distances after coordinates move.
    for key, edge in list(output.edges.items()):
        distance = physical_distance(output.nodes[edge.source_id], output.nodes[edge.target_id], scale)
        output.replace_edge(Edge(edge.source_id, edge.target_id, edge.probability, distance))
    return output, changed
