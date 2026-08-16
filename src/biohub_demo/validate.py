"""Strict graph and submission validation; never mutates predictions."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .graph import TrackGraph
from .io import SUBMISSION_COLUMNS


def validate_graph(graph: TrackGraph, image_shape: tuple[int, int, int, int]) -> dict[str, Any]:
    errors: list[str] = []
    t_size, z_size, y_size, x_size = image_shape
    for node in graph.nodes.values():
        values = np.asarray((node.z, node.y, node.x), dtype=float)
        if node.t < 0 or node.t >= t_size:
            errors.append(f"node {node.node_id}: t outside image")
        if not np.isfinite(values).all():
            errors.append(f"node {node.node_id}: non-finite coordinate")
        if not (0 <= node.z < z_size and 0 <= node.y < y_size and 0 <= node.x < x_size):
            errors.append(f"node {node.node_id}: coordinate outside image")
    indegree: Counter[int] = Counter()
    outdegree: Counter[int] = Counter()
    for source_id, target_id in graph.edges:
        if source_id not in graph.nodes or target_id not in graph.nodes:
            errors.append(f"dangling edge {source_id}->{target_id}")
            continue
        source, target = graph.nodes[source_id], graph.nodes[target_id]
        if source_id == target_id:
            errors.append(f"self edge {source_id}")
        if target.t != source.t + 1:
            errors.append(f"non-consecutive edge {source_id}->{target_id}")
        indegree[target_id] += 1
        outdegree[source_id] += 1
    errors.extend(f"node {node_id}: indegree {degree}>1" for node_id, degree in indegree.items() if degree > 1)
    errors.extend(f"node {node_id}: outdegree {degree}>2" for node_id, degree in outdegree.items() if degree > 2)
    return {"valid": not errors, "error_count": len(errors), "errors": errors[:100]}


def validate_submission(table: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    missing = [column for column in SUBMISSION_COLUMNS if column not in table.columns]
    if missing:
        return {"valid": False, "error_count": 1, "errors": [f"missing columns: {missing}"]}
    if list(table.columns) != list(SUBMISSION_COLUMNS):
        errors.append("column order differs from contract")
    if table.isna().any().any():
        errors.append("submission contains null/NaN")
    numeric = table.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        errors.append("submission contains infinity")
    if table["id"].tolist() != list(range(len(table))):
        errors.append("id is not a contiguous zero-based sequence")
    if table.duplicated().any():
        errors.append("duplicate rows")
    invalid_types = set(table.row_type.astype(str)) - {"node", "edge"}
    if invalid_types:
        errors.append(f"invalid row_type values: {sorted(invalid_types)}")
    for dataset, group in table.groupby("dataset", sort=False):
        nodes = group[group.row_type == "node"]
        edges = group[group.row_type == "edge"]
        if nodes.node_id.duplicated().any():
            errors.append(f"{dataset}: duplicate node_id")
        ids = set(nodes.node_id.astype(int))
        dangling = edges[~edges.source_id.isin(ids) | ~edges.target_id.isin(ids)]
        if len(dangling):
            errors.append(f"{dataset}: {len(dangling)} dangling edges")
        if len(edges):
            if (edges.source_id == edges.target_id).any():
                errors.append(f"{dataset}: self edge")
            node_times = nodes.set_index("node_id")["t"].to_dict()
            valid_edges = edges[edges.source_id.isin(ids) & edges.target_id.isin(ids)]
            nonconsecutive = sum(
                int(node_times[int(row.target_id)]) != int(node_times[int(row.source_id)]) + 1
                for row in valid_edges.itertuples()
            )
            if nonconsecutive:
                errors.append(f"{dataset}: {nonconsecutive} non-consecutive edges")
            if len(valid_edges) and valid_edges.target_id.value_counts().max() > 1:
                errors.append(f"{dataset}: indegree > 1")
            if len(valid_edges) and valid_edges.source_id.value_counts().max() > 2:
                errors.append(f"{dataset}: outdegree > 2")
        if len(nodes) and (nodes.t < 0).any():
            errors.append(f"{dataset}: negative time")
    return {"valid": not errors, "error_count": len(errors), "errors": errors[:100]}
