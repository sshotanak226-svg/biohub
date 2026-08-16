"""Small dependency-light graph representation shared by both runners."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True, slots=True)
class Node:
    node_id: int
    t: int
    z: float
    y: float
    x: float
    confidence: float = 1.0

    @property
    def zyx(self) -> np.ndarray:
        return np.asarray((self.z, self.y, self.x), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Edge:
    source_id: int
    target_id: int
    probability: float = 1.0
    distance_um: float = 0.0


@dataclass
class TrackGraph:
    dataset: str
    nodes: dict[int, Node] = field(default_factory=dict)
    edges: dict[tuple[int, int], Edge] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node_id: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(f"dangling edge: {edge.source_id}->{edge.target_id}")
        self.edges[(edge.source_id, edge.target_id)] = edge

    def remove_edge(self, source_id: int, target_id: int) -> None:
        self.edges.pop((source_id, target_id), None)

    def replace_node(self, node_id: int, **changes: float | int) -> None:
        self.nodes[node_id] = replace(self.nodes[node_id], **changes)

    def nodes_at(self, t: int) -> list[Node]:
        return sorted((n for n in self.nodes.values() if n.t == t), key=lambda n: n.node_id)

    def incoming(self, node_id: int) -> list[Edge]:
        return [e for e in self.edges.values() if e.target_id == node_id]

    def outgoing(self, node_id: int) -> list[Edge]:
        return [e for e in self.edges.values() if e.source_id == node_id]

    def next_node_id(self) -> int:
        return max(self.nodes, default=-1) + 1

    def copy(self) -> "TrackGraph":
        return TrackGraph(self.dataset, dict(self.nodes), dict(self.edges))


def physical_distance(a: Node | np.ndarray, b: Node | np.ndarray, scale: tuple[float, float, float]) -> float:
    aa = a.zyx if isinstance(a, Node) else np.asarray(a, dtype=np.float64)
    bb = b.zyx if isinstance(b, Node) else np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm((aa - bb) * np.asarray(scale, dtype=np.float64)))
