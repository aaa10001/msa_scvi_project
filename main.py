import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc

from config import PipelineConfig
from io_utils import ensure_dir, load_input_data, save_json, save_selected_markers
from preprocessing import get_dense_matrix, preprocess_adata
from marker_selection import (
    PANCREAS_ALLOWED_TYPES,
    PANCREAS_EXCLUDED_TYPES,
    compute_specific_markers_unsupervised,
    detect_and_filter_unsuitable_markers,
    filter_support_table_to_data,
    load_cellmarker_csv,
    load_panglaodb_tsv,
    merge_marker_sources,
    add_extra_public_markers,
    map_markers_to_data,
    score_marker_programs,
    select_high_confidence_cells,
)
from latent_dim import estimate_latent_dim
from scvi_model import refine_latent_with_marker_prototypes, train_scvi
from clustering import (
    build_consensus_from_top_results,
    refine_mixed_clusters,
    search_multiscale_stable_clusters,
)
from merge_guard import maybe_merge_oversplit_clusters
from evaluation import compute_supervised_metrics, plot_embeddings, save_heatmap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--cellmarker_db", required=True)
    p.add_argument("--panglaodb_tsv", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--label_col", default=None)
    p.add_argument("--batch_col", default=None)
    p.add_argument("--species", default="Human")
    p.add_argument("--tissue", default="Pancreas")
    p.add_argument("--csv_expr_start_col", type=int, default=0)
    p.add_argument("--n_top_hvg", type=int, default=2000)
    p.add_argument("--enable_merge_guard", action="store_true")
    return p.parse_args()


def run_pipeline(cfg: PipelineConfig):
    ensure_dir(cfg.output_dir)

    adata = load_input_data(
        cfg.input_path,
        csv_expr_start_col=cfg.csv_expr_start_col,
        label_col=cfg.label_col,
        batch_col=cfg.batch_col,
    )
    adata = preprocess_adata(
        adata,
        min_genes_per_cell=cfg.min_genes_per_cell,
        min_cells_per_gene=cfg.min_cells_per_gene,
        max_mt_pct=cfg.max_mt_pct,
        target_sum=cfg.target_sum,
        n_top_hvg=cfg.n_top_hvg,
    )

    raw_markers_cellmarker, raw_support_cellmarker = load_cellmarker_csv(
        cfg.cellmarker_db,
        species=cfg.species,
        tissue_keywords=cfg.tissue_keywords,
        max_markers_per_type=cfg.max_markers_per_type,
        allowed_cell_types=PANCREAS_ALLOWED_TYPES if any("pancre" in x.lower() for x in cfg.tissue_keywords) else None,
        excluded_cell_types=PANCREAS_EXCLUDED_TYPES if any("pancre" in x.lower() for x in cfg.tissue_keywords) else None,
        return_support=True,
    )

    marker_sources = [raw_markers_cellmarker]
    support_sources = [raw_support_cellmarker]
    panglao_loaded = False
    if cfg.panglaodb_tsv:
        raw_markers_panglao, raw_support_panglao = load_panglaodb_tsv(
            cfg.panglaodb_tsv,
            species=cfg.species,
            tissue_keywords=cfg.tissue_keywords,
            max_markers_per_type=cfg.panglaodb_max_markers_per_type,
            allowed_cell_types=PANCREAS_ALLOWED_TYPES if any("pancre" in x.lower() for x in cfg.tissue_keywords) else None,
            excluded_cell_types=PANCREAS_EXCLUDED_TYPES if any("pancre" in x.lower() for x in cfg.tissue_keywords) else None,
            canonical_only=cfg.panglaodb_canonical_only,
            max_ubiquitousness=cfg.panglaodb_max_ubiquitousness,
            source_weight=cfg.panglaodb_source_weight,
            return_support=True,
        )
        marker_sources.append(raw_markers_panglao)
        support_sources.append(raw_support_panglao)
        panglao_loaded = True

    raw_markers, raw_support = merge_marker_sources(
        marker_sources,
        support_sources,
        max_markers_per_type=max(cfg.max_markers_per_type, cfg.panglaodb_max_markers_per_type),
        normalize_source_support=cfg.normalize_marker_source_support,
    )

    raw_markers, raw_support = add_extra_public_markers(
        raw_markers,
        raw_support,
        var_names=list(adata.var_names),
        enabled=cfg.extra_public_markers_enabled,
        extra_weight=cfg.extra_public_marker_weight,
    )
    mapped_markers = map_markers_to_data(list(adata.var_names), raw_markers)
    mapped_support = filter_support_table_to_data(raw_support, mapped_markers)

    selected_markers, marker_score_table = compute_specific_markers_unsupervised(
        adata,
        mapped_markers,
        mapped_support,
        n_per_program=cfg.n_specific_markers_per_program,
        min_per_program_floor=cfg.min_specific_markers_floor,
        min_marker_mean=cfg.min_marker_mean,
        min_marker_detect_rate=cfg.min_marker_detect_rate,
        overlap_weight=cfg.marker_overlap_penalty,
    )

    unsuitable_marker_report = pd.DataFrame()
    if cfg.marker_auto_filter_enabled:
        selected_markers, unsuitable_marker_report = detect_and_filter_unsuitable_markers(
            selected_markers,
            marker_score_table,
            mapped_markers,
            remove_blacklist=cfg.marker_filter_remove_blacklist,
            remove_high_overlap=cfg.marker_filter_remove_high_overlap,
            overlap_count_threshold=cfg.marker_filter_overlap_count_threshold,
            remove_ubiquitous=cfg.marker_filter_remove_ubiquitous,
            ubiquitous_detect_rate=cfg.marker_filter_ubiquitous_detect_rate,
            min_markers_after_filter=cfg.marker_filter_min_markers_after_filter,
            min_specificity_quantile=cfg.marker_filter_min_specificity_quantile,
        )

    initial_program_scores = score_marker_programs(adata, mapped_markers)
    confidence_labels = select_high_confidence_cells(
        initial_program_scores,
        top_quantile=cfg.high_conf_top_quantile,
        min_margin=cfg.high_conf_min_margin,
        min_cells_per_program=cfg.high_conf_min_cells_per_program,
        max_cells_per_program=cfg.high_conf_max_cells_per_program,
    )

    hvg = adata.var_names[adata.var["highly_variable"]].tolist() if "highly_variable" in adata.var.columns else list(adata.var_names)
    specific_marker_union = sorted({g for gl in selected_markers.values() for g in gl})
    marker_budget = int(round(cfg.n_top_hvg * cfg.marker_gene_quota))
    specific_marker_union = specific_marker_union[:marker_budget] if marker_budget > 0 else []
    final_genes = sorted(set(hvg).union(set(specific_marker_union)))
    adata = adata[:, final_genes].copy()

    final_marker_sets = {k: [g for g in v if g in adata.var_names] for k, v in selected_markers.items() if len(v) > 0}
    final_program_scores = score_marker_programs(adata, final_marker_sets)
    if final_program_scores.shape[1] == 0:
        final_program_scores = initial_program_scores.reindex(adata.obs_names).fillna(0.0)

    X = get_dense_matrix(adata)
    latent_info = estimate_latent_dim(X, min_dim=cfg.latent_dim_min, max_dim=cfg.latent_dim_max)
    n_latent = latent_info["recommended_latent_dim"]

    _, _, latent = train_scvi(
        adata,
        batch_col=cfg.batch_col,
        n_latent=n_latent,
        max_epochs=cfg.scvi_max_epochs,
        gene_likelihood=cfg.scvi_gene_likelihood,
        n_layers=cfg.scvi_n_layers,
        seed=cfg.random_seed,
    )
    refined_latent, prototypes = refine_latent_with_marker_prototypes(
        latent,
        final_program_scores,
        confidence_labels.reindex(adata.obs_names).fillna("UNSURE"),
        strength=cfg.prototype_refine_strength,
    )

    batch_series = adata.obs[cfg.batch_col] if cfg.batch_col and cfg.batch_col in adata.obs.columns else (adata.obs["batch"] if "batch" in adata.obs.columns else None)
    search_df, run_cache = search_multiscale_stable_clusters(
        refined_latent,
        final_program_scores.reindex(adata.obs_names),
        batch_series,
        cfg.neighbors_grid,
        cfg.resolution_grid,
        cfg.clustering_repeats,
        cfg.score_alpha,
        cfg.score_beta,
        cfg.score_gamma,
        base_seed=cfg.random_seed,
    )

    consensus, final_labels = build_consensus_from_top_results(run_cache, adata.n_obs, top_k=cfg.consensus_top_k)

    secondary_refine_df = pd.DataFrame()
    if cfg.secondary_refine_enabled:
        final_labels, secondary_refine_df = refine_mixed_clusters(
            final_labels,
            refined_latent,
            final_program_scores.reindex(adata.obs_names),
            min_cluster_size=cfg.secondary_min_cluster_size,
            entropy_threshold=cfg.secondary_entropy_threshold,
            top_gap_threshold=cfg.secondary_top_gap_threshold,
            resolution_grid=cfg.secondary_resolution_grid,
            max_subclusters=cfg.secondary_max_subclusters,
            min_improvement=cfg.secondary_min_improvement,
            base_seed=cfg.random_seed,
            local_repeats=cfg.secondary_local_repeats,
            min_local_stability=cfg.secondary_min_local_stability,
            child_min_cells=cfg.secondary_child_min_cells,
            child_min_fraction=cfg.secondary_child_min_fraction,
            child_min_dominance=cfg.secondary_child_min_dominance,
            child_min_top_gap=cfg.secondary_child_min_top_gap,
            protect_dominant_share=cfg.secondary_protect_dominant_share,
        )

    merge_info = None
    if cfg.merge_guard_enabled:
        final_labels, merge_info = maybe_merge_oversplit_clusters(
            final_labels,
            refined_latent,
            final_program_scores.reindex(adata.obs_names),
            small_cluster_threshold=cfg.merge_small_cluster_threshold,
            marker_gap_threshold=cfg.merge_marker_gap_threshold,
        )

    adata.obs["msa_cluster"] = pd.Categorical(final_labels)
    adata.obsm["X_emb"] = refined_latent

    # 只有需要保存图片时才计算 UMAP，减少不必要开销。
    need_plot = (
        cfg.save_full_outputs
        or cfg.save_cluster_plot
        or cfg.save_true_label_plot
        or cfg.save_heatmap_plot
    )
    if need_plot:
        sc.pp.neighbors(adata, use_rep="X_emb")
        sc.tl.umap(adata)

    ann_df = final_program_scores.copy()
    ann_df["cluster"] = final_labels
    annotations = []
    for cl, sub in ann_df.groupby("cluster"):
        mean_scores = sub.drop(columns=["cluster"]).mean(axis=0).sort_values(ascending=False)
        if len(mean_scores) == 0:
            top_program, gap = "unknown", 0.0
        else:
            top_program = mean_scores.index[0]
            gap = float(mean_scores.iloc[0] - mean_scores.iloc[1]) if len(mean_scores) > 1 else 0.0
        annotations.append({"cluster": cl, "annotation": top_program, "annotation_gap": gap, "n_cells": int(len(sub))})
    cluster_annotations = pd.DataFrame(annotations).sort_values("cluster")

    supervised = None
    if cfg.label_col and "true_label" in adata.obs.columns:
        supervised = compute_supervised_metrics(
            adata.obs["true_label"].astype(str).values,
            final_labels.astype(str),
        )

        if cfg.save_full_outputs or cfg.save_heatmap_plot:
            save_heatmap(
                adata.obs["true_label"].astype(str).values,
                final_labels.astype(str),
                os.path.join(cfg.output_dir, "true_vs_cluster_heatmap.png"),
            )

        if cfg.save_full_outputs or cfg.save_true_label_plot:
            plot_embeddings(
                adata,
                "true_label",
                os.path.join(cfg.output_dir, "true_label_plot.png"),
                "True labels",
            )

    if cfg.save_full_outputs or cfg.save_cluster_plot:
        plot_embeddings(
            adata,
            "msa_cluster",
            os.path.join(cfg.output_dir, "embedding_plot.png"),
            "MSA-scVI clusters",
        )

    pseudo_label_counts = confidence_labels.value_counts(dropna=False).to_dict()
    pseudo_labeled_n = int((confidence_labels != "UNSURE").sum())

    report = {
        "method": {
            "name": "MSA-scVI",
            "description": "CellMarker-CSV unsupervised specific marker selection + per-program pseudo labels + data-driven latent dimension + multi-scale stability consensus clustering + mixed-cluster secondary refinement",
        },
        "data": {
            "input_path": cfg.input_path,
            "n_cells_after_qc": int(adata.n_obs),
            "n_genes_after_feature_selection": int(adata.n_vars),
            "batch_col": cfg.batch_col,
            "label_col": cfg.label_col,
            "species": cfg.species,
            "tissue_keywords": cfg.tissue_keywords,
        },
        "marker_sources": {
            "cellmarker_db_path": cfg.cellmarker_db,
            "panglaodb_tsv_path": cfg.panglaodb_tsv,
            "panglaodb_loaded": bool(panglao_loaded),
            "n_programs_raw_merged": int(len(raw_markers)),
            "n_programs_mapped": int(len(mapped_markers)),
        },
        "marker_selection": {
            "n_hvg": int(len(hvg)),
            "n_specific_markers": int(len(specific_marker_union)),
            "selected_markers": selected_markers,
        },
        "pseudo_labels": {
            "top_quantile": cfg.high_conf_top_quantile,
            "min_margin": cfg.high_conf_min_margin,
            "min_cells_per_program": cfg.high_conf_min_cells_per_program,
            "pseudo_labeled_n": pseudo_labeled_n,
            "pseudo_labeled_fraction": float(pseudo_labeled_n / max(len(confidence_labels), 1)),
            "counts": pseudo_label_counts,
        },
        "latent_dimension": latent_info,
        "prototype_refinement": {
            "strength": cfg.prototype_refine_strength,
            "n_prototypes": int(len(prototypes)),
        },
        "clustering_search_top10": search_df.head(10).to_dict(orient="records"),
        "secondary_refinement": {
            "enabled": bool(cfg.secondary_refine_enabled),
            "n_refined_parents": int(len(secondary_refine_df)) if secondary_refine_df is not None else 0,
        },
        "final": {
            "n_clusters": int(len(np.unique(final_labels))),
            "merge_guard": merge_info,
        },
        "supervised_metrics": supervised,
    }

    # 输出控制：
    # 1. save_full_outputs=True：保存所有中间文件；
    # 2. 参数实验中 save_full_outputs=False，只保存 report.json 和聚类 png。
    if cfg.save_full_outputs:
        cell_df = adata.obs.copy()
        cell_df["barcode"] = adata.obs_names.astype(str)
        cell_df["pred_cluster"] = final_labels.astype(str)
        top_program = (
            final_program_scores.idxmax(axis=1)
            if final_program_scores.shape[1] > 0
            else pd.Series(index=adata.obs_names, data="unknown")
        )
        cell_df["top_marker_program"] = top_program.reindex(adata.obs_names).astype(str).values
        cell_df["confidence_label"] = confidence_labels.reindex(adata.obs_names).fillna("UNSURE").astype(str).values

        final_program_scores.to_csv(os.path.join(cfg.output_dir, "marker_program_scores.csv"))
        marker_score_table.to_csv(os.path.join(cfg.output_dir, "specific_marker_scores.csv"), index=False)
        if unsuitable_marker_report is not None and not unsuitable_marker_report.empty:
            unsuitable_marker_report.to_csv(os.path.join(cfg.output_dir, "unsuitable_marker_report.csv"), index=False)
        save_selected_markers(os.path.join(cfg.output_dir, "selected_markers.json"), selected_markers)
        np.save(os.path.join(cfg.output_dir, "embeddings_scvi.npy"), latent)
        np.save(os.path.join(cfg.output_dir, "embeddings_refined.npy"), refined_latent)
        np.save(os.path.join(cfg.output_dir, "cluster_labels.npy"), final_labels)
        np.save(os.path.join(cfg.output_dir, "consensus_matrix.npy"), consensus)

        if cfg.label_col and "true_label" in adata.obs.columns:
            true_labels = adata.obs["true_label"].astype(str).values
            np.save(os.path.join(cfg.output_dir, "true_labels.npy"), true_labels)

            cell_ids = adata.obs_names.astype(str).values
            np.save(os.path.join(cfg.output_dir, "cell_ids.npy"), cell_ids)

        np.save(os.path.join(cfg.output_dir, "consensus_matrix.npy"), consensus)

        search_df.to_csv(os.path.join(cfg.output_dir, "clustering_search_results.csv"), index=False)
        cluster_annotations.to_csv(os.path.join(cfg.output_dir, "cluster_annotations.csv"), index=False)
        cell_df.to_csv(os.path.join(cfg.output_dir, "cell_metadata_with_results.csv"), index=False, encoding="utf-8-sig")
        if secondary_refine_df is not None and not secondary_refine_df.empty:
            secondary_refine_df.to_csv(os.path.join(cfg.output_dir, "secondary_refinement_report.csv"), index=False)

    if cfg.save_full_outputs or cfg.save_report_json:
        save_json(os.path.join(cfg.output_dir, "report.json"), report)

    return report


def main():
    args = parse_args()
    cfg = PipelineConfig(
        input_path=args.input,
        cellmarker_db=args.cellmarker_db,
        panglaodb_tsv=args.panglaodb_tsv,
        output_dir=args.output_dir,
        label_col=args.label_col,
        batch_col=args.batch_col,
        csv_expr_start_col=args.csv_expr_start_col,
        species=args.species,
        tissue_keywords=[x.strip() for x in str(args.tissue).split(",") if x.strip()],
        n_top_hvg=args.n_top_hvg,
        merge_guard_enabled=args.enable_merge_guard,
    )
    report = run_pipeline(cfg)
    print(report)


if __name__ == "__main__":
    main()
