"""Image, artifact and submission I/O."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .graph import Edge, Node, TrackGraph

SUBMISSION_COLUMNS = (
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
)
DEFAULT_SCALE = (1.625, 0.40625, 0.40625)


def quantile_normalize(image: np.ndarray, q_min: float = 0.001, q_max: float = 0.999) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("image contains NaN or infinity")
    sample = array.ravel()[:: max(1, array.size // 1_000_000)]
    lo, hi = np.quantile(sample, (q_min, q_max))
    if not np.isfinite((lo, hi)).all() or hi <= lo:
        raise ValueError(f"invalid image quantiles: q_low={lo}, q_high={hi}")
    return np.clip((array - lo) / (hi - lo), 0.0, 4.0).astype(np.float32, copy=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_rows(graph: TrackGraph) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in sorted(graph.nodes.values(), key=lambda n: (n.t, n.node_id)):
        rows.append({
            "dataset": graph.dataset, "row_type": "node", "node_id": node.node_id,
            "t": node.t, "z": int(round(node.z)), "y": int(round(node.y)), "x": int(round(node.x)),
            "source_id": -1, "target_id": -1,
        })
    for edge in sorted(graph.edges.values(), key=lambda e: (e.source_id, e.target_id)):
        rows.append({
            "dataset": graph.dataset, "row_type": "edge", "node_id": -1,
            "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": edge.source_id, "target_id": edge.target_id,
        })
    return rows


def write_submission(graphs: Iterable[TrackGraph], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for graph in sorted(graphs, key=lambda g: g.dataset):
        rows.extend(graph_rows(graph))
    for row_id, row in enumerate(rows):
        row["id"] = row_id
    pd.DataFrame(rows, columns=SUBMISSION_COLUMNS).to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    return output


def write_graph_json(graph: TrackGraph, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": graph.dataset,
        "nodes": [vars_without_slots(n) for n in sorted(graph.nodes.values(), key=lambda x: x.node_id)],
        "edges": [vars_without_slots(e) for e in sorted(graph.edges.values(), key=lambda x: (x.source_id, x.target_id))],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def read_graph_json(path: str | Path) -> TrackGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = TrackGraph(str(payload["dataset"]))
    for values in payload["nodes"]:
        graph.add_node(Node(**values))
    for values in payload["edges"]:
        graph.add_edge(Edge(**values))
    return graph


def is_complete_geff(path: str | Path) -> bool:
    """Return true only after GEFF's final root metadata has been published."""
    output = Path(path)
    try:
        metadata = json.loads((output / "zarr.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    geff = metadata.get("attributes", {}).get("geff")
    return output.is_dir() and isinstance(geff, dict) and bool(geff.get("geff_version"))


def _discard_staging(path: Path) -> None:
    """Best-effort cleanup for a uniquely named directory created by this process."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _replace_with_retry(source: Path, destination: Path, attempts: int = 5) -> None:
    last_error: PermissionError | None = None
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.2 * (2 ** attempt))
    assert last_error is not None
    raise last_error


def write_geff(graph: TrackGraph, path: str | Path, attempts: int = 5) -> Path:
    """Write GEFF to staging, validate it, then atomically publish the directory.

    Zarr updates ``zarr.json`` through an atomic file replacement.  Windows can
    transiently reject that replacement when another process (commonly a file
    indexer or antivirus scanner) has the metadata open.  A failed direct write
    also leaves a directory that merely checking ``Path.exists`` would mistake
    for a completed artifact.  Unique staging directories plus a final metadata
    marker make interrupted method searches safely resumable.
    """
    import polars as pl
    import tracksdata as td

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if is_complete_geff(output):
        return output
    target = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        target.add_node_attr_key(key, pl.Float64, -999999.0)
    ordered = sorted(graph.nodes.values(), key=lambda node: node.node_id)
    internal_ids = target.bulk_add_nodes([
        {"t": int(node.t), "z": float(node.z), "y": float(node.y), "x": float(node.x)} for node in ordered
    ])
    id_map = {node.node_id: internal for node, internal in zip(ordered, internal_ids)}
    if graph.edges:
        target.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        target.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        target.bulk_add_edges([
            {
                "source_id": id_map[edge.source_id], "target_id": id_map[edge.target_id],
                "edge_prob": float(edge.probability), "edge_dist": float(edge.distance_um),
            }
            for edge in graph.edges.values()
        ])
    staging: Path | None = None
    last_error: PermissionError | None = None
    for attempt in range(attempts):
        staging = output.with_name(
            f".{output.name}.{uuid.uuid4().hex}.partial"
        )
        try:
            target.to_geff(staging)
            if not is_complete_geff(staging):
                raise RuntimeError(f"GEFF write completed without final metadata: {staging}")
            break
        except PermissionError as exc:
            last_error = exc
            _discard_staging(staging)
            staging = None
            if attempt + 1 < attempts:
                time.sleep(0.2 * (2 ** attempt))
    if staging is None:
        assert last_error is not None
        raise last_error

    # Preserve an interrupted artifact for diagnosis instead of deleting it.
    if output.exists():
        if is_complete_geff(output):
            _discard_staging(staging)
            return output
        quarantine = output.with_name(
            f"{output.name}.incomplete-{uuid.uuid4().hex[:12]}"
        )
        _replace_with_retry(output, quarantine, attempts=attempts)

    try:
        _replace_with_retry(staging, output, attempts=attempts)
    except Exception:
        # Keep the complete staging artifact so it can be inspected or recovered.
        raise
    if not is_complete_geff(output):
        raise RuntimeError(f"published GEFF failed completion validation: {output}")
    return output


def vars_without_slots(value: Any) -> dict[str, Any]:
    return {name: getattr(value, name) for name in value.__dataclass_fields__}


def environment_manifest() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=5
        )
        commit = result.stdout.strip() if result.returncode == 0 else "uncommitted"
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "source_commit": commit,
        "numpy": np.__version__, "pandas": pd.__version__,
    }


def peak_rss_bytes() -> int | None:
    """Return OS-reported process peak RSS without adding a runtime dependency."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak is not None:
            return int(peak)
    except (ImportError, OSError):
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
        except (AttributeError, OSError):
            return None
    else:
        try:
            import resource
            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value if sys.platform == "darwin" else value * 1024
        except (ImportError, OSError):
            return None
    return None


def write_json(payload: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def parse_zarr_scale(attrs: dict[str, Any]) -> tuple[float, float, float]:
    try:
        transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if transform["type"] == "scale":
            return tuple(float(v) for v in transform["scale"][-3:])
    except (KeyError, IndexError, TypeError):
        pass
    return DEFAULT_SCALE
