"""Sparse edge candidate construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import Node, TrackGraph, physical_distance


@dataclass(frozen=True, slots=True)
class Candidate:
    source_id: int
    target_id: int
    probability: float
    distance_um: float


def distance_probabilities(
    sources: list[Node], targets: list[Node], scale: tuple[float, float, float], sigma_um: float = 3.5
) -> np.ndarray:
    if not sources or not targets:
        return np.zeros((len(sources), len(targets)), dtype=np.float64)
    source_coords = np.stack([n.zyx for n in sources])
    target_coords = np.stack([n.zyx for n in targets])
    delta = (source_coords[:, None] - target_coords[None]) * np.asarray(scale)
    distance = np.linalg.norm(delta, axis=-1)
    scores = np.exp(-0.5 * (distance / sigma_um) ** 2)
    # Parent-wise softmax is what the reference transformer inference uses.
    scores /= np.maximum(scores.sum(axis=0, keepdims=True), 1e-12)
    return scores


def sparse_candidates(
    sources: list[Node],
    targets: list[Node],
    probabilities: np.ndarray,
    scale: tuple[float, float, float],
    strong_threshold: float = 0.50,
    min_probability: float = 0.20,
    top_k_parents: int = 3,
    max_distance_um: float = 10.0,
) -> list[Candidate]:
    expected = (len(sources), len(targets))
    if probabilities.shape != expected:
        raise ValueError(f"probability shape {probabilities.shape} != {expected}")
    pairs = {tuple(map(int, pair)) for pair in np.argwhere(probabilities >= strong_threshold)}
    for target_index in range(len(targets)):
        k = min(top_k_parents, len(sources))
        if k:
            top = np.argpartition(probabilities[:, target_index], -k)[-k:]
            pairs.update((int(source_index), target_index) for source_index in top)
    result: list[Candidate] = []
    for source_index, target_index in sorted(pairs):
        probability = float(probabilities[source_index, target_index])
        if probability < min_probability:
            continue
        distance = physical_distance(sources[source_index], targets[target_index], scale)
        if distance <= max_distance_um:
            result.append(Candidate(
                sources[source_index].node_id, targets[target_index].node_id, probability, distance
            ))
    return result


def build_distance_candidates(graph: TrackGraph, scale: tuple[float, float, float], **kwargs: float | int) -> list[Candidate]:
    all_candidates: list[Candidate] = []
    max_t = max((n.t for n in graph.nodes.values()), default=-1)
    for t in range(max_t):
        sources, targets = graph.nodes_at(t), graph.nodes_at(t + 1)
        probabilities = distance_probabilities(sources, targets, scale)
        all_candidates.extend(sparse_candidates(sources, targets, probabilities, scale, **kwargs))
    return all_candidates
