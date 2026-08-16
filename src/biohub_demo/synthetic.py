"""Deterministic synthetic 3-D lineage with one division and one injected miss."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .graph import Edge, Node, TrackGraph


@dataclass(frozen=True)
class SyntheticData:
    image: np.ndarray
    truth: TrackGraph
    scale: tuple[float, float, float]
    missing_truth_node: int


def generate_synthetic(
    seed: int = 20260809,
    shape: tuple[int, int, int, int] = (12, 16, 128, 128),
    cell_count: int = 24,
    scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
) -> SyntheticData:
    if shape[0] < 9 or cell_count < 8:
        raise ValueError("synthetic demo needs at least 9 frames and 8 cells")
    rng = np.random.default_rng(seed)
    t_count, z_size, y_size, x_size = shape
    graph = TrackGraph("synthetic_demo")

    # A jittered grid keeps identities separable while preserving realistic local motion.
    grid_y = np.linspace(14, y_size - 15, int(np.ceil(np.sqrt(cell_count))))
    grid_x = np.linspace(14, x_size - 15, int(np.ceil(np.sqrt(cell_count))))
    starts = np.asarray([(z_size / 2, y, x) for y in grid_y for x in grid_x][:cell_count], dtype=float)
    starts[:, 0] += rng.uniform(-3.0, 3.0, cell_count)
    starts[:, 1:] += rng.uniform(-2.0, 2.0, (cell_count, 2))
    velocities = rng.normal(0.0, (0.08, 0.32, 0.32), size=(cell_count, 3))

    active: dict[int, tuple[np.ndarray, np.ndarray]] = {
        lineage: (starts[lineage], velocities[lineage]) for lineage in range(cell_count)
    }
    previous: dict[int, int] = {}
    next_node_id = 0
    division_parent = 0
    division_t = t_count // 2
    missing_lineage = 3
    missing_t = max(3, division_t - 2)
    missing_truth_node = -1

    for t in range(t_count):
        if t == division_t:
            parent_pos, parent_velocity = active.pop(division_parent)
            active[1000] = (parent_pos + np.asarray((0.0, -4.0, -4.0)), parent_velocity + (0.0, -0.12, -0.10))
            active[1001] = (parent_pos + np.asarray((0.0, 4.0, 4.0)), parent_velocity + (0.0, 0.12, 0.10))

        current: dict[int, int] = {}
        for lineage in sorted(active):
            pos, velocity = active[lineage]
            if t > 0:
                pos = pos + velocity + rng.normal(0.0, (0.025, 0.06, 0.06), 3)
                pos = np.clip(pos, (2, 6, 6), (z_size - 3, y_size - 7, x_size - 7))
                active[lineage] = (pos, velocity)
            node = Node(next_node_id, t, *pos, confidence=1.0)
            graph.add_node(node)
            current[lineage] = next_node_id
            if lineage in previous:
                graph.add_edge(Edge(previous[lineage], next_node_id))
            elif t == division_t and lineage in (1000, 1001):
                graph.add_edge(Edge(previous[division_parent], next_node_id))
            if lineage == missing_lineage and t == missing_t:
                missing_truth_node = next_node_id
            next_node_id += 1
        previous = current | ({division_parent: previous[division_parent]} if t == division_t else {})

    if missing_truth_node < 0:
        raise RuntimeError("failed to choose synthetic missing node")

    image = rng.normal(0.025, 0.008, shape).astype(np.float32)
    for node in graph.nodes.values():
        zi, yi, xi = (int(round(v)) for v in node.zyx)
        image[node.t, zi, yi, xi] += 1.0
    for t in range(t_count):
        image[t] = gaussian_filter(image[t], sigma=(0.65, 1.0, 1.0), mode="nearest")
    image -= image.min()
    image /= max(float(image.max()), 1e-6)
    return SyntheticData(image, graph, scale, missing_truth_node)
