from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineConfig:
    input_path: str
    cellmarker_db: str
    output_dir: str
    panglaodb_tsv: Optional[str] = None

    # metadata
    label_col: Optional[str] = None
    batch_col: Optional[str] = None
    barcode_col: Optional[str] = None
    csv_expr_start_col: int = 0
    species: str = "Human"
    tissue_keywords: List[str] = field(default_factory=lambda: ["Pancreas"])

    # preprocessing
    min_genes_per_cell: int = 200
    min_cells_per_gene: int = 3
    max_mt_pct: Optional[float] = 20.0
    target_sum: float = 1e4
    n_top_hvg: int = 2000

    # CellMarker CSV parsing and specificity selection
    max_markers_per_type: int = 50
    panglaodb_max_markers_per_type: int = 50
    panglaodb_canonical_only: bool = True
    panglaodb_max_ubiquitousness: float = 0.08
    panglaodb_source_weight: float = 1.0
    normalize_marker_source_support: bool = True
    min_marker_genes_present: int = 3
    n_specific_markers_per_program: int = 15
    min_specific_markers_floor: int = 3
    min_marker_mean: float = 0.05
    min_marker_detect_rate: float = 0.05
    marker_overlap_penalty: float = 0.5
    marker_gene_quota: float = 0.30

    # automatic marker quality control
    marker_auto_filter_enabled: bool = True
    marker_filter_remove_blacklist: bool = False
    marker_filter_remove_high_overlap: bool = False
    marker_filter_overlap_count_threshold: int = 4
    marker_filter_remove_ubiquitous: bool = True
    marker_filter_ubiquitous_detect_rate: float = 0.80
    marker_filter_min_markers_after_filter: int = 3
    marker_filter_min_specificity_quantile: float = 0.05
    extra_public_markers_enabled: bool = False
    extra_public_marker_weight: float = 1.5

    # high-confidence pseudo labels (per program)
    high_conf_top_quantile: float = 0.85
    high_conf_min_margin: float = 0.10
    high_conf_min_cells_per_program: int = 10
    high_conf_max_cells_per_program: Optional[int] = None

    # latent dimension estimation + scVI
    latent_dim_min: int = 10
    latent_dim_max: int = 30
    latent_dim_cap_by_hvg: bool = True
    scvi_max_epochs: int = 200
    scvi_gene_likelihood: str = "nb"
    scvi_n_layers: int = 2
    prototype_refine_strength: float = 0.10

    # adaptive clustering
    neighbors_grid: List[int] = field(default_factory=lambda: [10, 15, 20])
    resolution_grid: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    clustering_repeats: int = 5
    score_alpha: float = 0.55
    score_beta: float = 0.35
    score_gamma: float = 0.10
    consensus_top_k: int = 5

    # merge guard
    merge_guard_enabled: bool = False
    merge_small_cluster_threshold: float = 0.02
    merge_marker_gap_threshold: float = 0.10

    # secondary refinement for mixed clusters
    secondary_refine_enabled: bool = True
    secondary_min_cluster_size: int = 60
    secondary_entropy_threshold: float = 0.90
    secondary_top_gap_threshold: float = 0.05
    secondary_resolution_grid: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6])
    secondary_max_subclusters: int = 3
    secondary_min_improvement: float = 0.05
    secondary_local_repeats: int = 4
    secondary_min_local_stability: float = 0.85
    secondary_child_min_cells: int = 30
    secondary_child_min_fraction: float = 0.03
    secondary_child_min_dominance: float = 0.40
    secondary_child_min_top_gap: float = 0.02
    secondary_protect_dominant_share: float = 0.70

    # runtime
    random_seed: int = 42

    # output control
    save_full_outputs: bool = True
    save_report_json: bool = True
    save_cluster_plot: bool = True
    save_true_label_plot: bool = False
    save_heatmap_plot: bool = False
