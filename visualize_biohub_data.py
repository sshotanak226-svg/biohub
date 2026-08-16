"""Visualize the Biohub cell-tracking competition data structure.

The script follows the official baseline representation:

* image: OME-Zarr array with axes (T, Z, Y, X)
* tracks: GEFF graph with node attributes (t, z, y, x)
* edges: temporal links; a parent with two outgoing edges is a division

If competition data is not present, a clearly labelled synthetic example is
rendered so the notebook remains executable and explains the exact schema.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tracksdata as td
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SCALE_UM = (1.625, 0.40625, 0.40625)
BLUE = "#2869B0"
ORANGE = "#D97706"
PINK = "#C24177"
INK = "#1F2937"
GRID = "#D1D5DB"


@dataclass
class DatasetEntry:
    split: str
    name: str
    zarr_path: Path
    geff_path: Path | None
    shape: tuple[int, int, int, int]
    dtype: str
    scale_um: tuple[float, float, float]


@dataclass
class VisualizationBundle:
    label: str
    source_kind: str
    frame: np.ndarray
    image_shape: tuple[int, int, int, int]
    dtype: str
    scale_um: tuple[float, float, float]
    timepoint: int
    nodes: pd.DataFrame
    edges: pd.DataFrame


def parse_scale(attrs: dict) -> tuple[float, float, float]:
    """Extract spatial scale (Z, Y, X) in micrometres from OME-NGFF attrs."""
    try:
        transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if transform["type"] == "scale":
            return tuple(float(value) for value in transform["scale"][-3:])
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return DEFAULT_SCALE_UM


def resolve_data_root(requested: Path | None) -> Path | None:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    env_path = os.environ.get("BIOHUB_DATA_DIR") or os.environ.get("CELLMOT_DATA_DIR")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("data/biohub-cell-tracking-during-development"),
            Path("data"),
            Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists() and any((candidate / split).exists() for split in ("train", "test")):
            return candidate
    return None


def scan_datasets(data_root: Path | None) -> list[DatasetEntry]:
    entries: list[DatasetEntry] = []
    if data_root is None:
        return entries
    for split in ("train", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for zarr_path in sorted(split_dir.glob("*.zarr")):
            group = zarr.open_group(zarr_path, mode="r")
            array = group["0"]
            if array.ndim != 4:
                raise ValueError(f"Expected (T,Z,Y,X), got {array.shape} for {zarr_path}")
            geff_candidate = split_dir / f"{zarr_path.stem}.geff"
            entries.append(
                DatasetEntry(
                    split=split,
                    name=zarr_path.stem,
                    zarr_path=zarr_path,
                    geff_path=geff_candidate if geff_candidate.exists() else None,
                    shape=tuple(int(v) for v in array.shape),
                    dtype=str(array.dtype),
                    scale_um=parse_scale(dict(group.attrs)),
                )
            )
    return entries


def inventory_frame(entries: list[DatasetEntry]) -> pd.DataFrame:
    columns = [
        "split",
        "dataset",
        "shape_TZYX",
        "frames_T",
        "depth_Z",
        "height_Y",
        "width_X",
        "dtype",
        "scale_ZYX_um",
        "has_geff",
    ]
    rows = [
        {
            "split": entry.split,
            "dataset": entry.name,
            "shape_TZYX": " x ".join(map(str, entry.shape)),
            "frames_T": entry.shape[0],
            "depth_Z": entry.shape[1],
            "height_Y": entry.shape[2],
            "width_X": entry.shape[3],
            "dtype": entry.dtype,
            "scale_ZYX_um": ", ".join(f"{v:g}" for v in entry.scale_um),
            "has_geff": entry.geff_path is not None,
        }
        for entry in entries
    ]
    return pd.DataFrame(rows, columns=columns)


def load_geff_tables(geff_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_columns = ["node_id", "t", "z", "y", "x"]
    edge_columns = ["source_id", "target_id"]
    if geff_path is None:
        return pd.DataFrame(columns=node_columns), pd.DataFrame(columns=edge_columns)
    result = td.graph.IndexedRXGraph.from_geff(geff_path)
    graph = result[0] if isinstance(result, tuple) else result
    nodes = graph.node_attrs(attr_keys=node_columns).to_pandas()
    edges = graph.edge_attrs(attr_keys=edge_columns).to_pandas()
    return nodes, edges


def load_real_bundle(entry: DatasetEntry, requested_timepoint: int | None) -> VisualizationBundle:
    group = zarr.open_group(entry.zarr_path, mode="r")
    array = group["0"]
    timepoint = entry.shape[0] // 2 if requested_timepoint is None else requested_timepoint
    timepoint = max(0, min(int(timepoint), entry.shape[0] - 1))
    frame = np.asarray(array[timepoint])
    nodes, edges = load_geff_tables(entry.geff_path)
    return VisualizationBundle(
        label=f"{entry.split}/{entry.name}",
        source_kind="competition data",
        frame=frame,
        image_shape=entry.shape,
        dtype=entry.dtype,
        scale_um=entry.scale_um,
        timepoint=timepoint,
        nodes=nodes,
        edges=edges,
    )


def synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes: list[dict[str, float | int]] = []
    edges: list[dict[str, int]] = []

    # One ordinary trajectory.
    for t in range(6):
        nodes.append({"node_id": t, "t": t, "z": 12 + 0.8 * t, "y": 25 + 5 * t, "x": 20 + 7 * t})
        if t:
            edges.append({"source_id": t - 1, "target_id": t})

    # A second trajectory that divides after t=2.
    for t in range(3):
        node_id = 100 + t
        nodes.append({"node_id": node_id, "t": t, "z": 23 - 0.5 * t, "y": 70 - 4 * t, "x": 68 - 3 * t})
        if t:
            edges.append({"source_id": node_id - 1, "target_id": node_id})
    for branch, sign in ((200, -1), (300, 1)):
        for t in range(3, 6):
            node_id = branch + t
            nodes.append(
                {
                    "node_id": node_id,
                    "t": t,
                    "z": 21.5 + sign * 0.6 * (t - 2),
                    "y": 58 + sign * 7 * (t - 2),
                    "x": 59 - 3 * (t - 2),
                }
            )
            if t == 3:
                edges.append({"source_id": 102, "target_id": node_id})
            else:
                edges.append({"source_id": node_id - 1, "target_id": node_id})
    return pd.DataFrame(nodes), pd.DataFrame(edges)


def make_synthetic_bundle(requested_timepoint: int | None) -> VisualizationBundle:
    shape = (6, 36, 96, 96)
    # t=2 is the division frame in this schema demo.
    timepoint = 2 if requested_timepoint is None else requested_timepoint
    timepoint = max(0, min(int(timepoint), shape[0] - 1))
    nodes, edges = synthetic_tables()
    frame = np.random.default_rng(7).normal(70, 8, size=shape[1:]).astype(np.float32)
    zz, yy, xx = np.indices(shape[1:], dtype=np.float32)
    current_nodes = nodes[nodes["t"] == timepoint]
    for row in current_nodes.itertuples(index=False):
        squared_distance = ((zz - row.z) / 1.8) ** 2 + ((yy - row.y) / 3.2) ** 2 + ((xx - row.x) / 3.2) ** 2
        frame += 900 * np.exp(-0.5 * squared_distance)
    return VisualizationBundle(
        label="schema_demo",
        source_kind="synthetic schema demo (competition files not found)",
        frame=frame,
        image_shape=shape,
        dtype="synthetic float32",
        scale_um=DEFAULT_SCALE_UM,
        timepoint=timepoint,
        nodes=nodes,
        edges=edges,
    )


def robust_limits(frame: np.ndarray) -> tuple[float, float]:
    sampled = np.asarray(frame).ravel()[:: max(1, frame.size // 500_000)]
    low, high = np.quantile(sampled, [0.01, 0.997])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def edge_coordinates(bundle: VisualizationBundle) -> pd.DataFrame:
    if bundle.nodes.empty or bundle.edges.empty:
        return pd.DataFrame()
    source = bundle.nodes.add_prefix("source_")
    target = bundle.nodes.add_prefix("target_")
    joined = bundle.edges.merge(source, left_on="source_id", right_on="source_node_id", how="left")
    joined = joined.merge(target, left_on="target_id", right_on="target_node_id", how="left")
    return joined.dropna(subset=["source_t", "target_t"])


def division_node_ids(edges: pd.DataFrame) -> set[int]:
    if edges.empty:
        return set()
    counts = edges.groupby("source_id").size()
    return set(int(v) for v in counts[counts >= 2].index)


def overlay_projection(
    axis: plt.Axes,
    bundle: VisualizationBundle,
    horizontal: str,
    vertical: str,
) -> None:
    if bundle.nodes.empty:
        return
    current = bundle.nodes[bundle.nodes["t"] == bundle.timepoint]
    axis.scatter(
        current[horizontal],
        current[vertical],
        s=48,
        facecolors="none",
        edgecolors=BLUE,
        linewidths=1.5,
        label=f"nodes at t={bundle.timepoint}",
    )
    joined = edge_coordinates(bundle)
    outgoing = joined[joined["source_t"] == bundle.timepoint]
    for row in outgoing.itertuples(index=False):
        axis.plot(
            [getattr(row, f"source_{horizontal}"), getattr(row, f"target_{horizontal}")],
            [getattr(row, f"source_{vertical}"), getattr(row, f"target_{vertical}")],
            color=ORANGE,
            linewidth=1.2,
            alpha=0.85,
        )
    divisions = current[current["node_id"].isin(division_node_ids(bundle.edges))]
    if not divisions.empty:
        axis.scatter(
            divisions[horizontal],
            divisions[vertical],
            marker="*",
            s=120,
            color=PINK,
            edgecolors=INK,
            linewidths=0.5,
            label="division parent",
        )


def style_image_axis(axis: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=11, color=INK, weight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.tick_params(colors="#4B5563", labelsize=8)


def render_overview(bundle: VisualizationBundle, output_path: Path) -> None:
    frame = np.asarray(bundle.frame)
    if frame.ndim != 3:
        raise ValueError(f"A selected timepoint must be (Z,Y,X), got {frame.shape}")
    vmin, vmax = robust_limits(frame)
    xy = frame.max(axis=0)
    xz = frame.max(axis=1)
    yz = frame.max(axis=2)

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    grid = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)
    ax_xy = fig.add_subplot(grid[0, 0])
    ax_xz = fig.add_subplot(grid[0, 1])
    ax_yz = fig.add_subplot(grid[1, 0])
    ax_3d = fig.add_subplot(grid[1, 1], projection="3d")

    ax_xy.imshow(xy, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    overlay_projection(ax_xy, bundle, "x", "y")
    style_image_axis(ax_xy, "XY maximum-intensity projection", "X voxel", "Y voxel")

    ax_xz.imshow(xz, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
    overlay_projection(ax_xz, bundle, "x", "z")
    style_image_axis(ax_xz, "XZ maximum-intensity projection", "X voxel", "Z voxel")

    ax_yz.imshow(yz, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
    overlay_projection(ax_yz, bundle, "y", "z")
    style_image_axis(ax_yz, "YZ maximum-intensity projection", "Y voxel", "Z voxel")

    nodes = bundle.nodes
    if not nodes.empty:
        plotted = nodes if len(nodes) <= 8_000 else nodes.sample(8_000, random_state=7)
        scatter = ax_3d.scatter(
            plotted["x"], plotted["y"], plotted["z"],
            c=plotted["t"], cmap="viridis", s=13, alpha=0.75,
        )
        joined = edge_coordinates(bundle)
        for row in joined.head(3_000).itertuples(index=False):
            ax_3d.plot(
                [row.source_x, row.target_x],
                [row.source_y, row.target_y],
                [row.source_z, row.target_z],
                color=ORANGE,
                linewidth=0.65,
                alpha=0.45,
            )
        colorbar = fig.colorbar(scatter, ax=ax_3d, shrink=0.62, pad=0.08)
        colorbar.set_label("time t")
    else:
        ax_3d.text2D(0.08, 0.5, "No GEFF ground truth for this split", transform=ax_3d.transAxes, color="#6B7280")
    ax_3d.set_title("Sparse GEFF track graph", loc="left", fontsize=11, color=INK, weight="bold")
    ax_3d.set_xlabel("X voxel")
    ax_3d.set_ylabel("Y voxel")
    ax_3d.set_zlabel("Z voxel")
    ax_3d.view_init(elev=24, azim=-55)

    handles, labels = ax_xy.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.955), frameon=False)
    z_um, y_um, x_um = bundle.scale_um
    title = f"Biohub cell-tracking data | {bundle.label} | t={bundle.timepoint}"
    subtitle = (
        f"Source: {bundle.source_kind}   |   image shape (T,Z,Y,X): {bundle.image_shape}   |   "
        f"dtype: {bundle.dtype}   |   voxel scale (Z,Y,X): {z_um:g}, {y_um:g}, {x_um:g} um"
    )
    fig.suptitle(title, x=0.06, y=0.985, ha="left", fontsize=17, color=INK, weight="bold")
    fig.text(0.06, 0.952, subtitle, ha="left", fontsize=9.5, color="#4B5563")
    fig.text(
        0.06,
        0.015,
        "Blue rings: annotated nodes at the selected frame. Orange links: temporal edges. Pink star: parent of a division.",
        fontsize=9,
        color="#4B5563",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_structure(output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(13, 6.5), facecolor="white")
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 7)
    axis.axis("off")

    def box(x: float, y: float, width: float, height: float, text: str, color: str, size: int = 10) -> None:
        axis.add_patch(
            plt.Rectangle((x, y), width, height, facecolor="white", edgecolor=color, linewidth=1.8)
        )
        axis.text(x + 0.25, y + height / 2, text, va="center", ha="left", fontsize=size, color=INK)

    axis.text(0.3, 6.55, "Biohub competition data layout", fontsize=18, color=INK, weight="bold")
    axis.text(
        0.3,
        6.12,
        "Images are dense 4D volumes; annotations are sparse temporal graphs.",
        fontsize=10.5,
        color="#4B5563",
    )
    box(0.5, 3.2, 2.5, 1.6, "competition root\ntrain/\ntest/", BLUE, 11)
    box(4.0, 4.5, 3.2, 1.35, "train/{name}.zarr\nOME-Zarr image", BLUE, 10.5)
    box(4.0, 2.65, 3.2, 1.35, "train/{name}.geff\nSparse ground truth", ORANGE, 10.5)
    box(4.0, 0.8, 3.2, 1.35, "test/{name}.zarr\nImage only", BLUE, 10.5)
    box(8.3, 4.5, 4.1, 1.35, "array['0']\nshape = (T, Z, Y, X)", BLUE, 10.5)
    box(8.3, 2.65, 4.1, 1.35, "nodes: (node_id, t, z, y, x)\nedges: (source_id, target_id)", ORANGE, 10.5)
    box(8.3, 0.8, 4.1, 1.35, "submission.csv\nnode rows + edge rows", PINK, 10.5)

    arrows = [
        ((3.0, 4.25), (4.0, 5.15)),
        ((3.0, 4.0), (4.0, 3.35)),
        ((3.0, 3.7), (4.0, 1.5)),
        ((7.2, 5.15), (8.3, 5.15)),
        ((7.2, 3.35), (8.3, 3.35)),
        ((10.35, 2.65), (10.35, 2.15)),
    ]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#6B7280", "lw": 1.4})
    axis.text(0.5, 0.2, "Official voxel scale (Z,Y,X): 1.625, 0.40625, 0.40625 um", fontsize=9.5, color="#4B5563")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def choose_entry(entries: list[DatasetEntry], requested_name: str | None) -> DatasetEntry:
    if requested_name is None:
        train_with_tracks = [entry for entry in entries if entry.split == "train" and entry.geff_path is not None]
        return (train_with_tracks or entries)[0]
    for entry in entries:
        if entry.name == requested_name or f"{entry.split}/{entry.name}" == requested_name:
            return entry
    available = ", ".join(entry.name for entry in entries[:12])
    raise ValueError(f"Dataset '{requested_name}' not found. First available names: {available}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Directory containing train/ and test/")
    parser.add_argument("--dataset", default=None, help="Dataset stem or split/stem; defaults to first train item")
    parser.add_argument("--timepoint", type=int, default=None, help="Frame index; defaults to the middle frame")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/biohub_data_visualization"))
    parser.add_argument("--no-synthetic", action="store_true", help="Fail instead of rendering a schema demo")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_data_root(args.data_root)
    entries = scan_datasets(data_root)
    inventory = inventory_frame(entries)
    inventory.to_csv(output_dir / "dataset_inventory.csv", index=False)
    render_structure(output_dir / "data_structure.png")

    if entries:
        entry = choose_entry(entries, args.dataset)
        bundle = load_real_bundle(entry, args.timepoint)
    elif args.no_synthetic:
        raise FileNotFoundError(
            "Competition files were not found. Put train/*.zarr + train/*.geff and test/*.zarr "
            "under --data-root, or omit --no-synthetic to render the schema demo."
        )
    else:
        bundle = make_synthetic_bundle(args.timepoint)

    overview_path = output_dir / "latest_overview.png"
    render_overview(bundle, overview_path)

    print(f"data_root: {data_root if data_root else 'not found'}")
    print(f"datasets discovered: {len(entries)}")
    if not inventory.empty:
        print(inventory.head(10).to_string(index=False))
    print(f"visualized source: {bundle.source_kind}")
    print(f"image shape (T,Z,Y,X): {bundle.image_shape}")
    print(f"nodes: {len(bundle.nodes):,} | edges: {len(bundle.edges):,}")
    print(f"wrote: {overview_path}")
    print(f"wrote: {output_dir / 'data_structure.png'}")
    print(f"wrote: {output_dir / 'dataset_inventory.csv'}")


if __name__ == "__main__":
    main()
