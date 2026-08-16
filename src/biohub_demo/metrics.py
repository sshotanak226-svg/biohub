"""Transparent synthetic regression metrics matching the published score structure."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .graph import TrackGraph


def evaluate_graph(
    prediction: TrackGraph,
    truth: TrackGraph,
    scale: tuple[float, float, float],
    max_distance_um: float = 7.0,
) -> dict[str, Any]:
    matched: dict[int, int] = {}
    distances: list[float] = []
    times = sorted({node.t for node in truth.nodes.values()} | {node.t for node in prediction.nodes.values()})
    scale_array = np.asarray(scale)
    for t in times:
        pred_nodes, true_nodes = prediction.nodes_at(t), truth.nodes_at(t)
        if not pred_nodes or not true_nodes:
            continue
        pred_coords = np.stack([node.zyx for node in pred_nodes])
        true_coords = np.stack([node.zyx for node in true_nodes])
        costs = np.linalg.norm((pred_coords[:, None] - true_coords[None]) * scale_array, axis=-1)
        pred_index, true_index = linear_sum_assignment(costs)
        for pi, ti in zip(pred_index, true_index):
            if costs[pi, ti] <= max_distance_um:
                matched[pred_nodes[pi].node_id] = true_nodes[ti].node_id
                distances.append(float(costs[pi, ti]))

    true_edges = set(truth.edges)
    mapped_edges: set[tuple[int, int]] = set()
    invalid_pred_edges = 0
    for edge in prediction.edges.values():
        if edge.source_id in matched and edge.target_id in matched:
            mapped_edges.add((matched[edge.source_id], matched[edge.target_id]))
        else:
            invalid_pred_edges += 1
    edge_tp = len(mapped_edges & true_edges)
    edge_fp = len(mapped_edges - true_edges) + invalid_pred_edges
    edge_fn = len(true_edges - mapped_edges)
    edge_jaccard = _ratio(edge_tp, edge_tp + edge_fp + edge_fn)

    true_divisions = {node_id for node_id in truth.nodes if len(truth.outgoing(node_id)) == 2}
    pred_divisions = {
        matched[node_id] for node_id in prediction.nodes
        if node_id in matched and len(prediction.outgoing(node_id)) == 2
    }
    div_tp = len(true_divisions & pred_divisions)
    div_fp = len(pred_divisions - true_divisions)
    div_fn = len(true_divisions - pred_divisions)
    division_jaccard = _ratio(div_tp, div_tp + div_fp + div_fn)
    node_recall = len(set(matched.values())) / max(1, len(truth.nodes))
    node_ratio = (len(prediction.nodes) - len(truth.nodes)) / max(1, len(truth.nodes))
    edge_for_score = edge_jaccard if not math.isnan(edge_jaccard) else 0.0
    adjusted = max(0.0, edge_for_score * (1.0 - 0.1 * node_ratio))
    score = adjusted + (0.1 * division_jaccard if not math.isnan(division_jaccard) else 0.0)
    return {
        "score": score,
        "edge_jaccard": edge_jaccard if not math.isnan(edge_jaccard) else None,
        "adjusted_edge_jaccard": adjusted,
        "division_jaccard": division_jaccard if not math.isnan(division_jaccard) else None,
        "node_recall": node_recall,
        "predicted_node_ratio": 1.0 + node_ratio,
        "mean_position_error_um": float(np.mean(distances)) if distances else None,
        "edge_tp": edge_tp, "edge_fp": edge_fp, "edge_fn": edge_fn,
        "division_tp": div_tp, "division_fp": div_fp, "division_fn": div_fn,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")
