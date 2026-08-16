"""Classical detector for the local demo and shared physical NMS utilities."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from .graph import Node, TrackGraph
from .io import quantile_normalize


def physical_nms(
    coords: np.ndarray,
    scores: np.ndarray,
    min_distance_um: float,
    scale: tuple[float, float, float],
) -> np.ndarray:
    if len(coords) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(scores)[::-1]
    physical = np.asarray(coords, dtype=np.float64) * np.asarray(scale)
    kept: list[int] = []
    for index in order:
        if all(np.linalg.norm(physical[index] - physical[other]) >= min_distance_um for other in kept):
            kept.append(int(index))
    return np.asarray(kept, dtype=np.int64)


def detect_classical(
    image: np.ndarray,
    dataset: str,
    scale: tuple[float, float, float],
    threshold: float = 0.96875,
    nms_um: float = 3.0,
    inject_missing_zyxt: tuple[int, np.ndarray] | None = None,
) -> TrackGraph:
    normalized = quantile_normalize(image)
    graph = TrackGraph(dataset)
    node_id = 0
    footprint = tuple(max(3, int(round(nms_um / s)) | 1) for s in scale)
    for t, frame in enumerate(normalized):
        response = gaussian_filter(frame, sigma=(0.35, 0.55, 0.55), mode="nearest")
        positive = response[response > 0]
        reference = float(np.quantile(positive, 0.9995)) if positive.size else 1.0
        confidence = 1.0 - np.exp(-7.0 * response / max(reference, 1e-6))
        maxima = maximum_filter(confidence, size=footprint, mode="nearest")
        coords = np.argwhere((confidence == maxima) & (confidence >= threshold))
        scores = confidence[tuple(coords.T)] if len(coords) else np.empty(0)
        keep = physical_nms(coords, scores, nms_um, scale)
        for index in keep:
            coord = coords[index].astype(float)
            if inject_missing_zyxt is not None and t == inject_missing_zyxt[0]:
                missing_coord = inject_missing_zyxt[1]
                if np.linalg.norm((coord - missing_coord) * np.asarray(scale)) < 1.5:
                    continue
            graph.add_node(Node(node_id, t, *coord, confidence=float(scores[index])))
            node_id += 1
    return graph


def decode_probability(
    probability: np.ndarray,
    threshold: float,
    scale: tuple[float, float, float],
    nms_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    footprint = tuple(max(1, int(round(nms_um / s)) | 1) for s in scale)
    maxima = maximum_filter(probability, size=footprint, mode="nearest")
    coords = np.argwhere((probability == maxima) & (probability >= threshold))
    scores = probability[tuple(coords.T)] if len(coords) else np.empty(0)
    keep = physical_nms(coords, scores, nms_um, scale)
    return coords[keep].astype(np.float32), scores[keep].astype(np.float32)
