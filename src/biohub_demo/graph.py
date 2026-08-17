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
    _nodes_by_time: dict[int, tuple[Node, ...]] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _incoming_index: dict[int, list[Edge]] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _outgoing_index: dict[int, list[Edge]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def _invalidate_nodes(self) -> None:
        self._nodes_by_time = None

    def _invalidate_edges(self) -> None:
        self._incoming_index = None
        self._outgoing_index = None

    def _ensure_edge_indexes(self) -> None:
        if self._incoming_index is not None and self._outgoing_index is not None:
            return
        incoming: dict[int, list[Edge]] = {}
        outgoing: dict[int, list[Edge]] = {}
        for edge in self.edges.values():
            incoming.setdefault(edge.target_id, []).append(edge)
            outgoing.setdefault(edge.source_id, []).append(edge)
        self._incoming_index = incoming
        self._outgoing_index = outgoing

    def add_node(self, node: Node) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node_id: {node.node_id}")
        self.nodes[node.node_id] = node
        self._invalidate_nodes()

    def add_edge(self, edge: Edge) -> None:
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(f"dangling edge: {edge.source_id}->{edge.target_id}")
        key = (edge.source_id, edge.target_id)
        previous = self.edges.get(key)
        self.edges[key] = edge
        if self._incoming_index is not None and self._outgoing_index is not None:
            if previous is not None:
                self._incoming_index[previous.target_id].remove(previous)
                self._outgoing_index[previous.source_id].remove(previous)
            self._incoming_index.setdefault(edge.target_id, []).append(edge)
            self._outgoing_index.setdefault(edge.source_id, []).append(edge)

    def remove_edge(self, source_id: int, target_id: int) -> None:
        previous = self.edges.pop((source_id, target_id), None)
        if (
            previous is not None
            and self._incoming_index is not None
            and self._outgoing_index is not None
        ):
            self._incoming_index[previous.target_id].remove(previous)
            self._outgoing_index[previous.source_id].remove(previous)

    def clear_edges(self) -> None:
        self.edges.clear()
        self._invalidate_edges()

    def replace_edge(self, edge: Edge) -> None:
        key = (edge.source_id, edge.target_id)
        if key not in self.edges:
            raise KeyError(key)
        self.add_edge(edge)

    def replace_node(self, node_id: int, **changes: float | int) -> None:
        self.nodes[node_id] = replace(self.nodes[node_id], **changes)
        self._invalidate_nodes()

    def nodes_at(self, t: int) -> list[Node]:
        if self._nodes_by_time is None:
            grouped: dict[int, list[Node]] = {}
            for node in self.nodes.values():
                grouped.setdefault(node.t, []).append(node)
            self._nodes_by_time = {
                timepoint: tuple(sorted(values, key=lambda node: node.node_id))
                for timepoint, values in grouped.items()
            }
        return list(self._nodes_by_time.get(t, ()))

    def incoming(self, node_id: int) -> list[Edge]:
        self._ensure_edge_indexes()
        assert self._incoming_index is not None
        return list(self._incoming_index.get(node_id, ()))

    def outgoing(self, node_id: int) -> list[Edge]:
        self._ensure_edge_indexes()
        assert self._outgoing_index is not None
        return list(self._outgoing_index.get(node_id, ()))

    def next_node_id(self) -> int:
        return max(self.nodes, default=-1) + 1

    def copy(self) -> "TrackGraph":
        return TrackGraph(self.dataset, dict(self.nodes), dict(self.edges))


def physical_distance(a: Node | np.ndarray, b: Node | np.ndarray, scale: tuple[float, float, float]) -> float:
    aa = a.zyx if isinstance(a, Node) else np.asarray(a, dtype=np.float64)
    bb = b.zyx if isinstance(b, Node) else np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm((aa - bb) * np.asarray(scale, dtype=np.float64)))
