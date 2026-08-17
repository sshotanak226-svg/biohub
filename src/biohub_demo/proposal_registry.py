"""Registry for every unique method named in the OT proposal.

``fidelity`` is intentionally explicit.  ``exact`` means the implementation
matches the proposal at the level needed for this repository.  ``proxy`` is an
executable coarse ablation of a research idea, not a claim of paper-faithful
reproduction.  ``requires_training`` is never silently replaced by another
method and is reported as unavailable until its declared artifacts exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProposalMethod:
    name: str
    group: str
    fidelity: str
    builder: str
    settings: dict[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    note: str = ""
    checkpoint_reuse: str = "direct_raw_prediction"


def _uot(**changes: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "tracker": "unbalanced_ot",
        "ot_mode": "hybrid",
        "ot_device": "auto",
        "detection_threshold": 0.95,
        "edge_threshold": 0.01,
        "max_distance_um": 10.0,
        "distance_weight": 1.0,
        "transformer_weight": 1.0,
        "detection_weight": 0.25,
        "entropy_epsilon": 0.05,
        "source_mass_penalty": 0.5,
        "target_mass_penalty": 0.5,
        "sinkhorn_iterations": 100,
        "sinkhorn_tolerance": 1e-4,
        "ot_threshold": 0.05,
        "top_k_source": 2,
        "top_k_target": 3,
        "minimum_track_nodes": 4,
        "smoothing_weight": 0.15,
        "max_smoothing_shift_um": 1.5,
        "linked_nodes_only": True,
    }
    values.update(changes)
    return values


def proposal_methods() -> tuple[ProposalMethod, ...]:
    methods = [
        ProposalMethod("uot_distance", "simple", "exact", "single", _uot(ot_mode="distance", edge_threshold=0.0, transformer_weight=0.0)),
        ProposalMethod("uot_hybrid", "simple", "exact", "single", _uot()),
        ProposalMethod("uot_hybrid_division", "complex", "exact", "single", _uot(ot_mode="hybrid_division", safe_divisions=True, top_k_source=3, division_min_probability=0.20)),
        ProposalMethod("uot_consensus_ilp", "complex", "exact", "single", _uot(ot_mode="consensus_ilp", top_k_source=3)),
        ProposalMethod("transformer_greedy", "baseline", "exact", "configured"),
        ProposalMethod("transformer_ilp", "baseline", "exact", "configured"),
        ProposalMethod("distance_hungarian_top_style", "baseline", "exact", "configured"),
        ProposalMethod("distance_hungarian_filtered", "baseline", "exact", "configured"),
        ProposalMethod("hybrid_hungarian", "baseline", "exact", "configured"),
        ProposalMethod("hybrid_repair_division", "baseline", "exact", "configured"),
        ProposalMethod("balanced_sinkhorn_dustbin", "simple", "proxy", "single", _uot(source_mass_penalty=25.0, target_mass_penalty=25.0, entropy_epsilon=0.08, ot_projection="mutual_top"), note="High-penalty balanced-limit proxy with orphan-preserving projection."),
        ProposalMethod("partial_ot_hybrid", "simple", "proxy", "single", _uot(partial_mass_fraction=0.80, ot_threshold=0.08), note="Partial transported-mass truncation on the UOT plan."),
        ProposalMethod("uot_mutual_top", "simple", "exact", "single", _uot(ot_projection="mutual_top")),
        ProposalMethod("fused_unbalanced_gw", "complex", "proxy", "single", _uot(structure_weight=0.40, structure_neighbors=5), note="GPU local-neighbour signature proxy for full fused UGW."),
        ProposalMethod("three_frame_mmot", "complex", "proxy", "single_post", _uot(), note="Pairwise UOT followed by three-frame constant-velocity repair."),
        ProposalMethod("windowed_dynamic_ot", "complex", "proxy", "single_post", _uot(cycle_weight=0.50), note="Forward/backward UOT plus windowed motion pruning."),
        ProposalMethod("optimal_flow_transport", "complex", "proxy", "single", _uot(ot_mode="consensus_ilp", ilp_appearance_weight=0.15, ilp_disappearance_weight=0.15, ilp_division_weight=0.8), note="Global forest-flow ILP over GPU UOT candidates."),
        ProposalMethod("mesh_uot_hybrid", "latest", "proxy", "single", _uot(mesh_rounds=3, mesh_strength=0.15), note="Iterative entropy-concentration cost refinement."),
        ProposalMethod("oft_sinkhorn_lineage", "latest", "proxy", "single", _uot(ot_mode="consensus_ilp", cycle_weight=0.5, safe_divisions=True, top_k_source=3, ilp_division_weight=0.6), note="GPU Sinkhorn plus lineage-flow ILP proxy."),
        ProposalMethod("ot_localization_loss", "latest", "requires_training", "external", requirements=("ot_localization_checkpoint",), note="Fine-tune the existing detection head with an OT set loss.", checkpoint_reuse="fine_tune_local_checkpoint"),
        ProposalMethod("ot_triplet_edge_embedding", "latest", "requires_training", "external", requirements=("ot_triplet_checkpoint",), note="Reuse the backbone and fine-tune/add an appearance embedding head.", checkpoint_reuse="frozen_backbone_or_fine_tune_head"),
        ProposalMethod("structurally_constrained_dynamic_ot", "latest", "proxy", "single_post", _uot(structure_weight=0.35, cycle_weight=0.5), note="Local structure and temporal-consistency proxy."),
        ProposalMethod("parallel_time_sinkhorn", "latest", "proxy", "single", _uot(structure_weight=0.35), note="Same objective as structural UOT; records GPU timing as the speed-only comparison."),
        ProposalMethod("metric_aware_ot", "original", "proxy", "single_post", _uot(partial_mass_fraction=0.88, ot_threshold=0.08), note="Conservative expected-Jaccard pruning after UOT."),
        ProposalMethod("reaction_division_ot", "original", "proxy", "single_post", _uot(ot_mode="hybrid_division", safe_divisions=True, top_k_source=3, division_min_probability=0.25), note="Division reaction represented by gated second-child mass."),
        ProposalMethod("forward_backward_cycle_ot", "original", "exact", "single", _uot(cycle_weight=1.0, ot_projection="mutual_top")),
        ProposalMethod("family_conditioned_ot", "original", "exact", "family_single", _uot()),
        ProposalMethod("uncertainty_barycenter_ensemble", "original", "proxy", "ensemble", dependencies=("distance_hungarian_filtered", "hybrid_hungarian", "uot_hybrid"), note="Weighted edge-plan barycenter proxy from graph supports."),
        ProposalMethod("counterfactual_division_ot", "original", "proxy", "single_post", _uot(ot_mode="hybrid_division", safe_divisions=True, top_k_source=3), note="Division candidates are retained only with future branch support."),
        ProposalMethod("current6_hard_edge_vote", "ensemble", "exact", "ensemble", dependencies=("transformer_greedy", "transformer_ilp", "distance_hungarian_top_style", "distance_hungarian_filtered", "hybrid_hungarian", "hybrid_repair_division")),
        ProposalMethod("current6_weighted_edge_vote", "ensemble", "proxy", "ensemble", dependencies=("transformer_greedy", "transformer_ilp", "distance_hungarian_top_style", "distance_hungarian_filtered", "hybrid_hungarian", "hybrid_repair_division"), requirements=("inner_cv_method_weights",), note="Uses configured fallback weights until inner-CV weights are supplied."),
        ProposalMethod("current6_union_ilp", "ensemble", "exact", "ensemble_ilp", dependencies=("transformer_greedy", "transformer_ilp", "distance_hungarian_top_style", "distance_hungarian_filtered", "hybrid_hungarian", "hybrid_repair_division")),
        ProposalMethod("hungarian_uot_consensus", "ensemble", "exact", "ensemble", dependencies=("hybrid_hungarian", "uot_hybrid")),
        ProposalMethod("precision_recall_cascade", "ensemble", "exact", "cascade", dependencies=("transformer_ilp", "uot_hybrid", "distance_hungarian_filtered")),
        ProposalMethod("division_specialist_ensemble", "ensemble", "exact", "division_ensemble", dependencies=("distance_hungarian_filtered", "hybrid_repair_division", "uot_hybrid_division", "counterfactual_division_ot")),
        ProposalMethod("multi_seed_probability_ensemble", "ensemble", "requires_training", "external", requirements=("at_least_two_seed_raw_caches",), checkpoint_reuse="base_member_plus_additional_seed_checkpoints"),
        ProposalMethod("checkpoint_tracker_cross_ensemble", "ensemble", "requires_training", "external", requirements=("multiple_checkpoint_raw_caches",), checkpoint_reuse="base_member_plus_additional_checkpoints"),
        ProposalMethod("family_conditioned_mixture_of_experts", "ensemble", "requires_training", "external", requirements=("family_router_oof_model",), checkpoint_reuse="reuse_predictions_train_router_only"),
        ProposalMethod("stacked_edge_meta_model", "ensemble", "requires_training", "external", requirements=("stacked_edge_oof_model",), checkpoint_reuse="reuse_predictions_train_meta_model_only"),
        ProposalMethod("uncertainty_abstention_ensemble", "ensemble", "exact", "abstention", dependencies=("transformer_greedy", "distance_hungarian_filtered", "hybrid_hungarian", "uot_hybrid")),
        ProposalMethod("annotation_propensity_pruning", "competition", "requires_training", "external", requirements=("annotation_propensity_model",), checkpoint_reuse="frozen_backbone_train_propensity_head"),
        ProposalMethod("family_expert_44b6_6bba", "competition", "exact", "family_single", _uot()),
        ProposalMethod("development_phase_conditioned_ot", "competition", "proxy", "phase_single", _uot()),
        ProposalMethod("family_time_node_count_prior", "competition", "proxy", "single_post", _uot(), requirements=("family_time_count_prior",)),
        ProposalMethod("expected_jaccard_edge_selection", "competition", "proxy", "single_post", _uot(ot_threshold=0.03), note="Uses calibrated edge confidence as the expected-Jaccard local surrogate."),
        ProposalMethod("division_value_gate", "competition", "proxy", "single_post", _uot(ot_mode="hybrid_division", safe_divisions=True, top_k_source=3)),
        ProposalMethod("full_movie_transductive_ot", "competition", "proxy", "single_post", _uot(cycle_weight=1.0, structure_weight=0.25), note="Uses full-movie node statistics and future-consistency pruning without labels."),
        ProposalMethod("per_sequence_policy_router", "competition", "requires_training", "external", requirements=("policy_router_oof_model",), checkpoint_reuse="reuse_method_outputs_train_router_only"),
        ProposalMethod("test_distribution_calibrated_detection", "competition", "proxy", "calibrated_single", _uot()),
        ProposalMethod("sparse_gt_positive_unlabeled", "competition", "requires_training", "external", requirements=("nnpu_checkpoint",), checkpoint_reuse="fine_tune_local_checkpoint"),
        ProposalMethod("score_frontier_ensemble", "competition", "exact", "ensemble", dependencies=("transformer_ilp", "distance_hungarian_filtered", "uot_hybrid_division")),
        ProposalMethod("metric_aware_track_pruning", "competition", "proxy", "single_post", _uot()),
        ProposalMethod("four_test_movie_specialist_router", "competition", "requires_training", "external", requirements=("four_movie_policy_manifest",), checkpoint_reuse="reuse_method_outputs_train_router_only"),
    ]
    names = [method.name for method in methods]
    if len(names) != len(set(names)):
        raise RuntimeError("proposal method registry contains duplicate names")
    return tuple(methods)


PROPOSAL_METHODS = proposal_methods()
PROPOSAL_METHOD_BY_NAME = {method.name: method for method in PROPOSAL_METHODS}
