"""GPU-accelerated optimal-transport post-processing for cell linking.

The neural network inference is intentionally kept outside this module.  The
functions here consume the shared node/edge cache used by method_search and
move the dense numerical work (distance matrices, cost construction,
log-domain Sinkhorn, and top-k filtering) to CUDA when it is available.
Graph construction and the optional MILP projection remain on the CPU.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .candidates import Candidate
from .graph import Edge, Node, TrackGraph
from .optimizer import solve_forest


@dataclass(slots=True)
class _FrameResult:
    projected: list[Candidate]
    retained: list[Candidate]
    diagnostics: dict[str, float | int]


def resolve_ot_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when possible, otherwise use CPU."""
    value = requested.lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA OT post-processing was requested but CUDA is unavailable")
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported OT device: {requested}")
    return device


def _robust_scale(term: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = term[mask]
    if values.numel() == 0:
        return term
    scale = torch.median(values).clamp_min(1e-6)
    return term / scale


def _log_unbalanced_sinkhorn(
    cost: torch.Tensor,
    allowed: torch.Tensor,
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
    *,
    epsilon: float,
    source_penalty: float,
    target_penalty: float,
    iterations: int,
    tolerance: float,
) -> tuple[torch.Tensor, int, float]:
    """Compute a KL-unbalanced transport plan in the log domain."""
    if epsilon <= 0 or source_penalty <= 0 or target_penalty <= 0:
        raise ValueError("OT epsilon and marginal penalties must be positive")
    active_rows = allowed.any(dim=1)
    active_cols = allowed.any(dim=0)
    plan = torch.zeros_like(cost)
    if not bool(active_rows.any()) or not bool(active_cols.any()):
        return plan, 0, 0.0

    sub_cost = cost[active_rows][:, active_cols]
    sub_allowed = allowed[active_rows][:, active_cols]
    log_kernel = (-sub_cost / epsilon).masked_fill(~sub_allowed, -torch.inf)
    log_a = source_mass[active_rows].clamp_min(1e-8).log()
    log_b = target_mass[active_cols].clamp_min(1e-8).log()
    rho_source = source_penalty / (source_penalty + epsilon)
    rho_target = target_penalty / (target_penalty + epsilon)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    residual = float("inf")
    completed = 0
    for step in range(iterations):
        previous = log_u
        log_u = rho_source * (
            log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        )
        log_v = rho_target * (
            log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
        )
        completed = step + 1
        if tolerance > 0 and (step + 1) % 5 == 0:
            residual = float(torch.max(torch.abs(log_u - previous)).item())
            if residual <= tolerance:
                break

    log_plan = (log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0)).clamp(max=50.0)
    sub_plan = torch.exp(log_plan).masked_fill(~sub_allowed, 0.0)
    plan[active_rows.unsqueeze(1) & active_cols.unsqueeze(0)] = sub_plan.flatten()
    if residual == float("inf"):
        residual = 0.0
    return plan, completed, residual


def _topk_mask(
    plan: torch.Tensor,
    allowed: torch.Tensor,
    *,
    threshold: float,
    top_k_source: int,
    top_k_target: int,
) -> torch.Tensor:
    keep = allowed & (plan >= threshold)
    masked = plan.masked_fill(~allowed, -torch.inf)
    if top_k_source > 0 and masked.shape[1] > 0:
        k = min(top_k_source, masked.shape[1])
        values, indices = torch.topk(masked, k, dim=1)
        rows = torch.arange(masked.shape[0], device=plan.device).unsqueeze(1).expand_as(indices)
        keep[rows[torch.isfinite(values)], indices[torch.isfinite(values)]] = True
    if top_k_target > 0 and masked.shape[0] > 0:
        k = min(top_k_target, masked.shape[0])
        values, indices = torch.topk(masked, k, dim=0)
        cols = torch.arange(masked.shape[1], device=plan.device).unsqueeze(0).expand_as(indices)
        keep[indices[torch.isfinite(values)], cols[torch.isfinite(values)]] = True
    return keep & allowed


def _assignment_pairs(score: np.ndarray, allowed: np.ndarray) -> set[tuple[int, int]]:
    if score.size == 0 or not allowed.any():
        return set()
    cost = np.where(allowed, -score, 1e9)
    rows, cols = linear_sum_assignment(cost)
    return {(int(i), int(j)) for i, j in zip(rows, cols) if allowed[i, j]}


def _mutual_top_pairs(score: np.ndarray, allowed: np.ndarray, second_pass: bool) -> set[tuple[int, int]]:
    """Return mutual row/column maxima, optionally filling orphans by assignment."""
    if score.size == 0 or not allowed.any():
        return set()
    masked = np.where(allowed, score, -np.inf)
    row_best = np.argmax(masked, axis=1)
    col_best = np.argmax(masked, axis=0)
    pairs = {
        (i, int(j)) for i, j in enumerate(row_best)
        if allowed[i, j] and col_best[j] == i
    }
    if not second_pass:
        return pairs
    used_rows = {i for i, _ in pairs}
    used_cols = {j for _, j in pairs}
    remaining_rows = [i for i in range(score.shape[0]) if i not in used_rows]
    remaining_cols = [j for j in range(score.shape[1]) if j not in used_cols]
    if remaining_rows and remaining_cols:
        sub_allowed = allowed[np.ix_(remaining_rows, remaining_cols)]
        sub_score = score[np.ix_(remaining_rows, remaining_cols)]
        pairs.update(
            (remaining_rows[i], remaining_cols[j])
            for i, j in _assignment_pairs(sub_score, sub_allowed)
        )
    return pairs


def _partial_mass_mask(plan: torch.Tensor, allowed: torch.Tensor, fraction: float) -> torch.Tensor:
    """Keep the highest-mass entries until the requested transported mass is covered."""
    if not 0 < fraction <= 1:
        raise ValueError("partial_mass_fraction must be in (0, 1]")
    flat = plan.masked_fill(~allowed, 0).flatten()
    positive = flat > 0
    if not bool(positive.any()):
        return torch.zeros_like(allowed)
    indices = torch.nonzero(positive, as_tuple=False).flatten()
    values = flat[indices]
    order = torch.argsort(values, descending=True)
    ordered_values = values[order]
    cutoff = fraction * ordered_values.sum()
    count = int(torch.searchsorted(torch.cumsum(ordered_values, dim=0), cutoff).item()) + 1
    selected = indices[order[:count]]
    keep = torch.zeros_like(flat, dtype=torch.bool)
    keep[selected] = True
    return keep.reshape_as(allowed)


def _frame_transport(
    sources: list[Node],
    targets: list[Node],
    raw_candidates: list[Candidate],
    settings: dict[str, Any],
    device: torch.device,
    scale: tuple[float, float, float],
    *,
    use_transformer: bool,
    consensus: bool,
) -> _FrameResult:
    if not sources or not targets:
        return _FrameResult([], [], {"gated_edges": 0, "retained_edges": 0})
    dtype = torch.float32
    source_xyz = torch.as_tensor(
        [[node.z, node.y, node.x] for node in sources], dtype=dtype, device=device
    )
    target_xyz = torch.as_tensor(
        [[node.z, node.y, node.x] for node in targets], dtype=dtype, device=device
    )
    voxel_scale = torch.as_tensor(scale, dtype=dtype, device=device)
    distance = torch.cdist(source_xyz * voxel_scale, target_xyz * voxel_scale)
    max_distance = float(settings.get("max_distance_um", 10.0))
    allowed = distance <= max_distance

    source_index = {node.node_id: index for index, node in enumerate(sources)}
    target_index = {node.node_id: index for index, node in enumerate(targets)}
    transformer = torch.zeros_like(distance)
    if use_transformer:
        triples = [
            (source_index[item.source_id], target_index[item.target_id], item.probability)
            for item in raw_candidates
            if item.source_id in source_index and item.target_id in target_index
        ]
        if triples:
            row, col, probability = zip(*triples)
            transformer[
                torch.as_tensor(row, device=device), torch.as_tensor(col, device=device)
            ] = torch.as_tensor(probability, dtype=dtype, device=device)
        allowed &= transformer >= float(settings.get("edge_threshold", 0.01))

    source_confidence = torch.as_tensor(
        [node.confidence for node in sources], dtype=dtype, device=device
    ).clamp(float(settings.get("minimum_mass", 0.05)), 1.0)
    target_confidence = torch.as_tensor(
        [node.confidence for node in targets], dtype=dtype, device=device
    ).clamp(float(settings.get("minimum_mass", 0.05)), 1.0)
    distance_term = (distance / max_distance).square()
    detection_term = -0.5 * (
        source_confidence.log().unsqueeze(1) + target_confidence.log().unsqueeze(0)
    )
    probability_term = -torch.log(transformer.clamp_min(1e-6))
    if bool(settings.get("normalize_costs", True)):
        distance_term = _robust_scale(distance_term, allowed)
        detection_term = _robust_scale(detection_term, allowed)
        if use_transformer:
            probability_term = _robust_scale(probability_term, allowed)
    cost = (
        float(settings.get("distance_weight", 1.0)) * distance_term
        + float(settings.get("detection_weight", 0.25)) * detection_term
    )
    if use_transformer:
        cost = cost + float(settings.get("transformer_weight", 1.0)) * probability_term
    structure_weight = float(settings.get("structure_weight", 0.0))
    if structure_weight > 0 and len(sources) > 1 and len(targets) > 1:
        # A cheap fused-GW proxy: compare each node's local k-NN distance
        # signature.  It retains the GPU-friendly pairwise matrix form while
        # adding within-frame neighbourhood structure to the cross-frame cost.
        k_source = min(int(settings.get("structure_neighbors", 5)) + 1, len(sources))
        k_target = min(int(settings.get("structure_neighbors", 5)) + 1, len(targets))
        source_local = torch.cdist(source_xyz * voxel_scale, source_xyz * voxel_scale)
        target_local = torch.cdist(target_xyz * voxel_scale, target_xyz * voxel_scale)
        source_signature = torch.topk(source_local, k_source, largest=False, dim=1).values[:, 1:].mean(dim=1)
        target_signature = torch.topk(target_local, k_target, largest=False, dim=1).values[:, 1:].mean(dim=1)
        structure_term = torch.abs(
            source_signature.unsqueeze(1) - target_signature.unsqueeze(0)
        ) / max(max_distance, 1e-6)
        if bool(settings.get("normalize_costs", True)):
            structure_term = _robust_scale(structure_term, allowed)
        cost = cost + structure_weight * structure_term
    cost = cost.masked_fill(~allowed, 0.0)

    sinkhorn_kwargs = {
        "epsilon": float(settings.get("entropy_epsilon", 0.05)),
        "source_penalty": float(settings.get("source_mass_penalty", 0.5)),
        "target_penalty": float(settings.get("target_mass_penalty", 0.5)),
        "iterations": int(settings.get("sinkhorn_iterations", 100)),
        "tolerance": float(settings.get("sinkhorn_tolerance", 1e-4)),
    }
    mesh_rounds = max(1, int(settings.get("mesh_rounds", 1)))
    working_cost = cost
    plan = torch.zeros_like(cost)
    completed = 0
    residual = 0.0
    for mesh_round in range(mesh_rounds):
        plan, round_completed, residual = _log_unbalanced_sinkhorn(
            working_cost, allowed, source_confidence, target_confidence, **sinkhorn_kwargs
        )
        completed += round_completed
        if mesh_round + 1 < mesh_rounds:
            concentration = -torch.log(plan.clamp_min(1e-8))
            concentration = _robust_scale(concentration, allowed)
            working_cost = cost + float(settings.get("mesh_strength", 0.15)) * concentration

    cycle_weight = float(settings.get("cycle_weight", 0.0))
    if cycle_weight > 0:
        reverse, reverse_completed, _ = _log_unbalanced_sinkhorn(
            cost.T, allowed.T, target_confidence, source_confidence,
            epsilon=sinkhorn_kwargs["epsilon"],
            source_penalty=sinkhorn_kwargs["target_penalty"],
            target_penalty=sinkhorn_kwargs["source_penalty"],
            iterations=sinkhorn_kwargs["iterations"],
            tolerance=sinkhorn_kwargs["tolerance"],
        )
        completed += reverse_completed
        geometric = torch.sqrt(plan.clamp_min(0) * reverse.T.clamp_min(0))
        plan = (1.0 - cycle_weight) * plan + cycle_weight * geometric
    row_mass = plan.sum(dim=1).clamp_min(1e-8)
    col_mass = plan.sum(dim=0).clamp_min(1e-8)
    affinity = (plan / torch.sqrt(row_mass.unsqueeze(1) * col_mass.unsqueeze(0))).clamp(0, 1)
    retained_mask = _topk_mask(
        plan,
        allowed,
        threshold=float(settings.get("ot_threshold", 0.05)),
        top_k_source=int(settings.get("top_k_source", 2)),
        top_k_target=int(settings.get("top_k_target", 3)),
    )
    if settings.get("partial_mass_fraction") is not None:
        retained_mask &= _partial_mass_mask(
            plan, allowed, float(settings["partial_mass_fraction"])
        )

    distance_cpu = distance.detach().cpu().numpy()
    affinity_cpu = affinity.detach().cpu().numpy()
    transformer_cpu = transformer.detach().cpu().numpy()
    allowed_cpu = retained_mask.detach().cpu().numpy()
    projection_mode = str(settings.get("ot_projection", "hungarian"))
    if projection_mode == "hungarian":
        projected_pairs = _assignment_pairs(affinity_cpu, allowed_cpu)
    elif projection_mode == "mutual_top":
        projected_pairs = _mutual_top_pairs(affinity_cpu, allowed_cpu, second_pass=True)
    elif projection_mode == "mutual_only":
        projected_pairs = _mutual_top_pairs(affinity_cpu, allowed_cpu, second_pass=False)
    else:
        raise ValueError(f"unknown OT projection: {projection_mode}")
    vote_counts: dict[tuple[int, int], int] = defaultdict(int)
    if consensus:
        for pair in projected_pairs:
            vote_counts[pair] += 1
        for pair in _assignment_pairs(-distance_cpu, allowed_cpu):
            vote_counts[pair] += 1
        if use_transformer:
            for pair in _assignment_pairs(transformer_cpu, allowed_cpu):
                vote_counts[pair] += 1

    retained: list[Candidate] = []
    retained_pairs = np.argwhere(allowed_cpu)
    for i, j in retained_pairs:
        probability = float(affinity_cpu[i, j])
        if consensus:
            vote = vote_counts[(int(i), int(j))] / (3.0 if use_transformer else 2.0)
            weights = (
                float(settings.get("ot_score_weight", 0.55)),
                float(settings.get("transformer_score_weight", 0.20)) if use_transformer else 0.0,
                float(settings.get("distance_score_weight", 0.10)),
                float(settings.get("vote_score_weight", 0.15)),
            )
            distance_score = float(np.exp(-0.5 * (distance_cpu[i, j] / 3.5) ** 2))
            numerator = (
                weights[0] * probability
                + weights[1] * float(transformer_cpu[i, j])
                + weights[2] * distance_score
                + weights[3] * vote
            )
            probability = numerator / max(sum(weights), 1e-8)
        retained.append(Candidate(
            sources[int(i)].node_id,
            targets[int(j)].node_id,
            probability,
            float(distance_cpu[i, j]),
        ))
    by_pair = {(item.source_id, item.target_id): item for item in retained}
    projected = [
        by_pair[(sources[i].node_id, targets[j].node_id)]
        for i, j in sorted(projected_pairs)
    ]
    positive = plan[plan > 0]
    entropy = float((-(positive * positive.clamp_min(1e-12).log()).sum()).item()) if positive.numel() else 0.0
    return _FrameResult(projected, retained, {
        "gated_edges": int(allowed.sum().item()),
        "retained_edges": len(retained),
        "sinkhorn_iterations": completed,
        "sinkhorn_residual": residual,
        "transport_entropy": entropy,
        "projection_mode": projection_mode,
    })


def solve_ot_tracking(
    graph: TrackGraph,
    raw_candidates: list[Candidate],
    settings: dict[str, Any],
    scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
) -> tuple[TrackGraph, list[Candidate], dict[str, Any]]:
    """Run one of the four OT comparison modes from the proposal."""
    mode = str(settings.get("ot_mode", "hybrid"))
    if mode not in {"distance", "hybrid", "hybrid_division", "consensus_ilp"}:
        raise ValueError(f"unknown OT mode: {mode}")
    device = resolve_ot_device(str(settings.get("ot_device", "auto")))
    use_transformer = mode != "distance"
    consensus = mode == "consensus_ilp"
    by_time: dict[int, list[Candidate]] = defaultdict(list)
    for item in raw_candidates:
        source = graph.nodes.get(item.source_id)
        target = graph.nodes.get(item.target_id)
        if source is not None and target is not None and target.t == source.t + 1:
            by_time[source.t].append(item)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    projected: list[Candidate] = []
    retained: list[Candidate] = []
    frame_diagnostics: list[dict[str, float | int]] = []
    max_t = max((node.t for node in graph.nodes.values()), default=-1)
    for t in range(max_t):
        frame = _frame_transport(
            graph.nodes_at(t), graph.nodes_at(t + 1), by_time[t], settings,
            device, scale, use_transformer=use_transformer, consensus=consensus,
        )
        projected.extend(frame.projected)
        retained.extend(frame.retained)
        frame_diagnostics.append(frame.diagnostics)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tensor_seconds = perf_counter() - started

    if consensus:
        result, projection = solve_forest(
            graph,
            retained,
            edge_weight=float(settings.get("ilp_edge_weight", -1.0)),
            appearance_weight=float(settings.get("ilp_appearance_weight", 0.1)),
            disappearance_weight=float(settings.get("ilp_disappearance_weight", 0.1)),
            division_weight=float(settings.get("ilp_division_weight", 1.0)),
        )
    else:
        result = graph.copy()
        result.clear_edges()
        for item in projected:
            result.add_edge(Edge(
                item.source_id, item.target_id, item.probability, item.distance_um
            ))
        projection = "hungarian-from-uot"

    diagnostics = {
        "ot_mode": mode,
        "ot_device": str(device),
        "gpu_accelerated": device.type == "cuda",
        "tensor_seconds": tensor_seconds,
        "frames": len(frame_diagnostics),
        "gated_edges": sum(int(item.get("gated_edges", 0)) for item in frame_diagnostics),
        "retained_edges": len(retained),
        "projected_edges": len(result.edges),
        "mean_sinkhorn_iterations": (
            sum(int(item.get("sinkhorn_iterations", 0)) for item in frame_diagnostics)
            / max(len(frame_diagnostics), 1)
        ),
        "transport_entropy": sum(
            float(item.get("transport_entropy", 0.0)) for item in frame_diagnostics
        ),
        "projection": projection,
        "consensus_vote_members": 3 if consensus else 0,
    }
    return result, retained, diagnostics
