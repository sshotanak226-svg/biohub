"""Registry for the twenty literature-backed screening recipes.

The recipes deliberately reuse the existing local checkpoint and raw prediction
cache.  They are executable ablations of the cited ideas, not paper-faithful
reimplementations that would require new annotations, architectures, and weeks
of training.  This distinction is written into every result artifact.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LiteratureMethod:
    name: str
    paper: str
    paper_url: str
    focus: str
    profiles: tuple[str, ...]
    combine: str = "single"
    operators: tuple[str, ...] = ()
    note: str = ""
    fidelity: str = "screening_proxy"


LITERATURE_METHODS = (
    LiteratureMethod(
        "hoct_edge_attention_proxy",
        "Higher-Order Cell Tracking Transformer",
        "https://arxiv.org/abs/2607.11754",
        "edges+division",
        ("uot_structured_division",),
        operators=("future_divisions",),
        note="Edge-centric 3D-structure and division proxy; no learned HOCT encoder.",
    ),
    LiteratureMethod(
        "trackastra_window_proxy",
        "Trackastra: Transformer-based cell tracking for live-cell microscopy",
        "https://arxiv.org/abs/2405.15700",
        "temporal-links+division",
        ("uot_cycle",),
        operators=("motion", "future_divisions"),
        note="Cycle and constant-velocity window proxy; no Trackastra weights.",
    ),
    LiteratureMethod(
        "calibrated_probabilistic_flow_proxy",
        "Cell tracking with accurate error prediction",
        "https://www.nature.com/articles/s41592-025-02845-6",
        "nodes+links+division",
        ("uot_flow",),
        operators=("node_count", "division_value"),
        note="Probability-cost global-flow proxy using existing neural scores.",
    ),
    LiteratureMethod(
        "mitosis_mht_uncertainty_proxy",
        "Cell Tracking according to Biological Needs",
        "https://arxiv.org/abs/2403.15011",
        "long-term+division",
        ("uot_base", "uot_cycle", "uot_division"),
        combine="division",
        operators=("future_divisions",),
        note="Multi-hypothesis consensus and mitosis-aware branch proxy.",
    ),
    LiteratureMethod(
        "ultrack_multi_hypothesis_proxy",
        "Ultrack: pushing the limits of cell tracking across biological scales",
        "https://www.nature.com/articles/s41592-025-02778-0",
        "nodes+links",
        ("uot_permissive", "uot_base", "uot_strict"),
        combine="union_ilp",
        operators=("node_count",),
        note="Multiple confidence-threshold hypotheses with temporal graph selection.",
    ),
    LiteratureMethod(
        "seven_frame_4d_motion_proxy",
        "Automated reconstruction of whole-embryo cell lineages by learning from sparse annotations",
        "https://www.nature.com/articles/s41587-022-01427-7",
        "nodes+motion",
        ("uot_cycle",),
        operators=("motion", "motion_strict"),
        note="Repeated temporal-consistency refinement; no trained seven-frame 4D U-Net.",
    ),
    LiteratureMethod(
        "full_sequence_gnn_proxy",
        "Graph Neural Network for Cell Tracking in Microscopy Videos",
        "https://arxiv.org/abs/2202.04731",
        "global-links+mitosis",
        ("uot_flow",),
        operators=("future_divisions",),
        note="Global structured-flow proxy; no learned message-passing network.",
    ),
    LiteratureMethod(
        "embedtrack_offset_proxy",
        "EmbedTrack -- Simultaneous Cell Segmentation and Tracking",
        "https://arxiv.org/abs/2204.10713",
        "nodes+motion",
        ("hybrid_division",),
        operators=("motion",),
        note="Hybrid score and motion-offset refinement proxy.",
    ),
    LiteratureMethod(
        "adaptive_distance_map_proxy",
        "Cell tracking with accurate error prediction",
        "https://www.nature.com/articles/s41592-025-02845-6",
        "nodes",
        ("uot_permissive",),
        operators=("node_count", "metric_track"),
        note="Confidence/count separation proxy; no adaptive-distance-map retraining.",
    ),
    LiteratureMethod(
        "neighbor_distance_graph_relink_proxy",
        "Cell Segmentation and Tracking using CNN-Based Distance Predictions",
        "https://arxiv.org/abs/2004.01486",
        "nodes+gap-links",
        ("distance_filtered",),
        operators=("motion",),
        note="Distance-neighbour graph with short temporal relinking.",
    ),
    LiteratureMethod(
        "stardist3d_uot_consensus_proxy",
        "Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy",
        "https://arxiv.org/abs/1908.03636",
        "nodes",
        ("distance_filtered", "uot_base"),
        combine="agreement",
        note="Shape-independent distance detector and learned detector consensus; no StarDist masks.",
    ),
    LiteratureMethod(
        "nnunet_deep_supervision_proxy",
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation",
        "https://www.nature.com/articles/s41592-020-01008-z",
        "nodes",
        ("uot_permissive", "uot_base", "uot_strict"),
        combine="weighted",
        operators=("node_count",),
        note="Multi-operating-point deep-supervision proxy; architecture is unchanged.",
    ),
    LiteratureMethod(
        "swin_unetr_context_proxy",
        "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors",
        "https://arxiv.org/abs/2201.01266",
        "nodes+context",
        ("uot_structured",),
        operators=("metric_track",),
        note="Local-structure context proxy; no Swin UNETR encoder.",
    ),
    LiteratureMethod(
        "self_supervised_3d_context_proxy",
        "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis",
        "https://arxiv.org/abs/2111.14791",
        "nodes+context",
        ("uot_structured", "uot_cycle"),
        combine="weighted",
        operators=("metric_track",),
        note="Context/motion ensemble proxy; no self-supervised pretraining is claimed.",
    ),
    LiteratureMethod(
        "cellpose_finetune_consensus_proxy",
        "Cellpose 2.0: how to train your own model",
        "https://www.nature.com/articles/s41592-022-01663-4",
        "nodes",
        ("distance_filtered", "hybrid_division"),
        combine="agreement",
        operators=("node_count",),
        note="Independent-detector agreement proxy; no Cellpose instance-mask checkpoint.",
    ),
    LiteratureMethod(
        "cell_tractr_query_split_proxy",
        "Cell-TRACTR: A transformer-based model for end-to-end segmentation and tracking of cells",
        "https://pubmed.ncbi.nlm.nih.gov/40408631/",
        "nodes+division",
        ("uot_structured_division",),
        operators=("future_divisions", "division_value"),
        note="Query-split proxy using two-child candidates and future support.",
    ),
    LiteratureMethod(
        "four_frame_division_detector_proxy",
        "DeepKymoTracker",
        "https://pubmed.ncbi.nlm.nih.gov/39928591/",
        "division",
        ("uot_division",),
        operators=("future_divisions", "division_value"),
        note="Four-frame support proxy; no dedicated crop CNN is trained.",
    ),
    LiteratureMethod(
        "conservation_tracking_ilp_proxy",
        "Conservation Tracking",
        "https://openaccess.thecvf.com/content_iccv_2013/papers/Schiegg_Conservation_Tracking_2013_ICCV_paper.pdf",
        "global-links+division",
        ("uot_flow",),
        operators=("division_value",),
        note="Global birth/death/division forest optimization over UOT candidates.",
    ),
    LiteratureMethod(
        "focal_hard_negative_proxy",
        "Focal Loss for Dense Object Detection",
        "https://arxiv.org/abs/1708.02002",
        "node-precision+division-precision",
        ("uot_strict",),
        operators=("metric_track", "expected_jaccard"),
        note="Hard-negative operating-point proxy; no focal-loss fine-tuning is claimed.",
    ),
    LiteratureMethod(
        "deep_ensemble_tta_consensus_proxy",
        "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles",
        "https://arxiv.org/abs/1612.01474",
        "nodes+links+division",
        ("uot_base", "uot_cycle", "uot_structured_division", "hybrid_division"),
        combine="abstention",
        operators=("node_count", "future_divisions"),
        note="Existing TTA raw inference plus multi-recipe uncertainty abstention.",
    ),
)


LITERATURE_METHOD_BY_NAME = {method.name: method for method in LITERATURE_METHODS}
if len(LITERATURE_METHOD_BY_NAME) != 20:
    raise RuntimeError("literature registry must contain exactly twenty unique methods")
