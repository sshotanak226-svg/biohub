"""Global binary ILP forest selection with a deterministic greedy fallback."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .candidates import Candidate
from .graph import Edge, TrackGraph


def solve_forest(
    graph: TrackGraph,
    candidates: list[Candidate],
    edge_weight: float = -1.0,
    appearance_weight: float = 0.0,
    disappearance_weight: float = 1.4,
    division_weight: float = 1.0,
) -> tuple[TrackGraph, str]:
    if not candidates:
        return graph.copy(), "empty"
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix

        source_ids = sorted({c.source_id for c in candidates})
        target_ids = sorted({c.target_id for c in candidates})
        n_edges, n_sources, n_targets = len(candidates), len(source_ids), len(target_ids)
        source_var = {node_id: n_edges + i for i, node_id in enumerate(source_ids)}
        division_var = {node_id: n_edges + n_sources + i for i, node_id in enumerate(source_ids)}
        appearance_var = {node_id: n_edges + 2 * n_sources + i for i, node_id in enumerate(target_ids)}
        size = n_edges + 2 * n_sources + n_targets
        objective = np.zeros(size)
        objective[:n_edges] = [edge_weight * c.probability for c in candidates]
        objective[n_edges:n_edges + n_sources] = disappearance_weight
        objective[n_edges + n_sources:n_edges + 2 * n_sources] = division_weight
        objective[n_edges + 2 * n_sources:] = appearance_weight
        rows: list[tuple[dict[int, float], float, float]] = []

        incoming: dict[int, list[int]] = defaultdict(list)
        outgoing: dict[int, list[int]] = defaultdict(list)
        for i, candidate in enumerate(candidates):
            outgoing[candidate.source_id].append(i)
            incoming[candidate.target_id].append(i)
        for target_id, edge_indices in incoming.items():
            coeffs = {i: 1.0 for i in edge_indices}
            coeffs[appearance_var[target_id]] = 1.0
            rows.append((coeffs, 1.0, 1.0))
        for source_id, edge_indices in outgoing.items():
            # outdegree <= 2
            rows.append(({i: 1.0 for i in edge_indices}, -np.inf, 2.0))
            # disappearance is one exactly when outdegree is zero.
            coeffs = {i: 1.0 for i in edge_indices}
            coeffs[source_var[source_id]] = 1.0
            rows.append((coeffs, 1.0, np.inf))
            rows.append(({**{i: 1.0 for i in edge_indices}, source_var[source_id]: 2.0}, -np.inf, 2.0))
            # division is one exactly when outdegree is two.
            rows.append(({**{i: 1.0 for i in edge_indices}, division_var[source_id]: -1.0}, -np.inf, 1.0))
            rows.append(({**{i: 1.0 for i in edge_indices}, division_var[source_id]: -2.0}, 0.0, np.inf))
        matrix = lil_matrix((len(rows), size), dtype=np.float64)
        lower, upper = np.empty(len(rows)), np.empty(len(rows))
        for row_index, (coeffs, lo, hi) in enumerate(rows):
            for column, value in coeffs.items():
                matrix[row_index, column] = value
            lower[row_index], upper[row_index] = lo, hi
        result = milp(
            objective, integrality=np.ones(size), bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix.tocsr(), lower, upper),
            options={"time_limit": 120.0, "presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(result.message)
        selected = [candidate for i, candidate in enumerate(candidates) if result.x[i] > 0.5]
        method = "scipy.milp"
    except Exception as exc:  # pragma: no cover - exercised only without a MILP backend
        selected = _greedy(candidates)
        method = f"greedy-fallback:{type(exc).__name__}"

    output = graph.copy()
    output.clear_edges()
    for candidate in selected:
        output.add_edge(Edge(
            candidate.source_id, candidate.target_id, candidate.probability, candidate.distance_um
        ))
    return output, method


def _greedy(candidates: list[Candidate]) -> list[Candidate]:
    selected: list[Candidate] = []
    indegree: dict[int, int] = defaultdict(int)
    outdegree: dict[int, int] = defaultdict(int)
    for candidate in sorted(candidates, key=lambda c: (-c.probability, c.distance_um, c.source_id, c.target_id)):
        if indegree[candidate.target_id] < 1 and outdegree[candidate.source_id] < 2:
            selected.append(candidate)
            indegree[candidate.target_id] += 1
            outdegree[candidate.source_id] += 1
    return selected
