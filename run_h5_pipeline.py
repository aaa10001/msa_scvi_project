#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
import traceback

import pandas as pd
import scanpy as sc

from config import PipelineConfig
from main import run_pipeline


# =========================================================
# 1. 路径设置
# =========================================================

H5AD_DIR = r"C:\Users\26852\Desktop\paper\tabula_muris"
OUTPUT_ROOT = r"C:\Users\26852\Desktop\paper\result\msa_scvi_all_optimized3"

CELLMARKER_DB = r"C:\Users\26852\Desktop\paper\cell_marker\Cell_marker_Mouse.csv"
PANGLAODB_TSV = None


# =========================================================
# 2. 数据集列表
# =========================================================

DATASETS = [
    {"name": "Heart", "input_path": os.path.join(H5AD_DIR, "Heart.h5ad"), "tissue_keywords": ["Heart"]},
    # {"name": "Kidney", "input_path": os.path.join(H5AD_DIR, "Kidney.h5ad"), "tissue_keywords": ["Kidney"]},
    #{"name": "Liver", "input_path": os.path.join(H5AD_DIR, "Liver.h5ad"), "tissue_keywords": ["Liver"]},
    #{"name": "Lung", "input_path": os.path.join(H5AD_DIR, "Lung.h5ad"), "tissue_keywords": ["Lung"]},
    #{"name": "Pancreas", "input_path": os.path.join(H5AD_DIR, "Pancreas.h5ad"), "tissue_keywords": ["Pancreas"]},
    #{"name": "Trachea", "input_path": os.path.join(H5AD_DIR, "Trachea.h5ad"), "tissue_keywords": ["Trachea"]},
]


# =========================================================
# 3. 通用配置
# =========================================================

LABEL_COL = "cell_ontology_class"
BATCH_COL = "mouse.id"
SPECIES = "Mouse"

# 小类过滤只是构建 benchmark，不参与训练调参。
# 如果你想保留全部类别，可改成 None。
MIN_CELLS_PER_LABEL = None

# 不下采样
MAX_CELLS_PER_LABEL = None

RANDOM_SEED = 42


# =========================================================
# 4. 数据集专属参数
# =========================================================

# 不使用真实标签，只根据数据集规模/组织复杂度预设。
DATASET_LATENT_DIM = {
    "Heart": 30,
    "Kidney": 30,
    "Liver": 30,
    "Lung": 30,
    "Pancreas": 30,
    "Trachea": 30,
}

# 对复杂组织提高 resolution 上限
DATASET_RESOLUTION_GRID = {
    "Heart": [0.15, 0.2, 0.3, 0.4, 0.5, 0.6],
    "Kidney": [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
    "Liver": [0.15, 0.2, 0.3, 0.4, 0.5, 0.6],
    "Lung": [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
    "Pancreas": [0.15, 0.2, 0.3, 0.4, 0.5, 0.6],
    "Spleen": [0.15, 0.2, 0.3, 0.4, 0.5, 0.6],
    "Trachea": [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
}

DATASET_SECONDARY_REFINE = {
    "Heart": True,
    "Kidney": True,
    "Liver": True,
    "Lung": True,
    "Pancreas": True,
    "Trachea": True,
}


# =========================================================
# 5. 数据准备
# =========================================================

def prepare_h5ad_for_pipeline(
    input_h5ad: str,
    output_dir: str,
    label_col: str,
    batch_col: str | None,
    min_cells_per_label: int | None = 50,
    max_cells_per_label: int | None = None,
    seed: int = 42,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_h5ad):
        raise FileNotFoundError(f"找不到 h5ad 文件：{input_h5ad}")

    adata = sc.read_h5ad(input_h5ad)

    if label_col not in adata.obs.columns:
        raise ValueError(f"没有找到标签列 {label_col}，当前 obs 列为：{list(adata.obs.columns)}")

    if batch_col is not None and batch_col not in adata.obs.columns:
        print(f"警告：没有找到 batch_col={batch_col}，后续会自动使用 batch1")
        batch_col = None

    adata.obs[label_col] = adata.obs[label_col].astype(str)

    if batch_col is not None:
        adata.obs[batch_col] = adata.obs[batch_col].astype(str)

    print("\n=== 原始标签分布 ===")
    print(adata.obs[label_col].value_counts())

    if min_cells_per_label is not None and min_cells_per_label > 1:
        vc = adata.obs[label_col].value_counts()
        keep_labels = vc[vc >= min_cells_per_label].index
        adata = adata[adata.obs[label_col].isin(keep_labels)].copy()

    if max_cells_per_label is not None and max_cells_per_label > 0:
        import numpy as np

        rng = np.random.default_rng(seed)
        keep_idx = []

        for _, sub in adata.obs.groupby(label_col):
            idx = sub.index.to_numpy()
            if len(idx) > max_cells_per_label:
                idx = rng.choice(idx, size=max_cells_per_label, replace=False)
            keep_idx.extend(idx.tolist())

        adata = adata[keep_idx].copy()

    print("\n=== 过滤后标签分布 ===")
    print(adata.obs[label_col].value_counts())
    print("过滤后数据：", adata)

    prepared_path = os.path.join(output_dir, Path(input_h5ad).stem + "_prepared.h5ad")
    adata.write_h5ad(prepared_path)

    return prepared_path


# =========================================================
# 6. 单个数据集运行
# =========================================================

def build_config(dataset_name: str, input_h5ad: str, prepared_h5ad: str, output_dir: str, tissue_keywords: list[str]) -> PipelineConfig:
    latent_dim = DATASET_LATENT_DIM.get(dataset_name, 15)
    resolution_grid = DATASET_RESOLUTION_GRID.get(dataset_name, [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
    secondary_enabled = DATASET_SECONDARY_REFINE.get(dataset_name, True)

    return PipelineConfig(
        input_path=prepared_h5ad,
        cellmarker_db=CELLMARKER_DB,
        panglaodb_tsv=PANGLAODB_TSV,
        output_dir=output_dir,
        label_col=LABEL_COL,
        batch_col=BATCH_COL,
        species=SPECIES,
        tissue_keywords=tissue_keywords,

        # 预处理
        n_top_hvg=2000,
        min_genes_per_cell=200,
        min_cells_per_gene=3,
        max_mt_pct=None,

        # Marker 来源
        max_markers_per_type=80,
        panglaodb_max_markers_per_type=80,
        panglaodb_canonical_only=True,
        panglaodb_max_ubiquitousness=0.15,
        panglaodb_source_weight=1.0,
        normalize_marker_source_support=True,

        # Marker 筛选：针对 processed h5ad 放宽，避免 selected_markers 全空
        n_specific_markers_per_program=20,
        min_specific_markers_floor=8,
        min_marker_mean=0.0,
        min_marker_detect_rate=0.0,
        marker_auto_filter_enabled=False,
        marker_filter_remove_ubiquitous=False,
        marker_filter_min_specificity_quantile=0.0,
        marker_gene_quota=0.50,

        # 伪标签 + prototype
        high_conf_top_quantile=0.80,
        high_conf_min_margin=0.05,
        high_conf_min_cells_per_program=20,
        prototype_refine_strength=0.08,

        # scVI
        latent_dim_min=latent_dim,
        latent_dim_max=latent_dim,
        scvi_max_epochs=150,
        scvi_gene_likelihood="normal",
        scvi_n_layers=2,

        # 无监督多尺度聚类
        neighbors_grid=[8,10, 12,15,18,20,22, 25],
        resolution_grid=resolution_grid,
        clustering_repeats=3,
        score_alpha=0.45,
        score_beta=0.45,
        score_gamma=0.10,
        consensus_top_k=5,

        # merge guard
        merge_guard_enabled=True,
        merge_small_cluster_threshold=0.015,
        merge_marker_gap_threshold=0.08,

        # 二次细分
        secondary_refine_enabled=secondary_enabled,
        secondary_min_cluster_size=100,
        secondary_entropy_threshold=0.88,
        secondary_top_gap_threshold=0.06,
        secondary_resolution_grid=[0.2, 0.3, 0.4, 0.5, 0.6],
        secondary_max_subclusters=3,
        secondary_min_improvement=0.03,
        secondary_local_repeats=3,
        secondary_min_local_stability=0.80,
        secondary_child_min_cells=30,
        secondary_child_min_fraction=0.03,
        secondary_child_min_dominance=0.35,
        secondary_child_min_top_gap=0.015,
        secondary_protect_dominant_share=0.75,

        random_seed=RANDOM_SEED,
    )


def run_one_h5_dataset(dataset_cfg: dict) -> dict:
    dataset_name = dataset_cfg["name"]
    input_h5ad = dataset_cfg["input_path"]
    tissue_keywords = dataset_cfg["tissue_keywords"]

    output_dir = os.path.join(OUTPUT_ROOT, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"开始运行数据集：{dataset_name}")
    print("=" * 90)

    prepared_h5ad = prepare_h5ad_for_pipeline(
        input_h5ad=input_h5ad,
        output_dir=output_dir,
        label_col=LABEL_COL,
        batch_col=BATCH_COL,
        min_cells_per_label=MIN_CELLS_PER_LABEL,
        max_cells_per_label=MAX_CELLS_PER_LABEL,
        seed=RANDOM_SEED,
    )

    cfg = build_config(
        dataset_name=dataset_name,
        input_h5ad=input_h5ad,
        prepared_h5ad=prepared_h5ad,
        output_dir=output_dir,
        tissue_keywords=tissue_keywords,
    )

    report = run_pipeline(cfg)

    supervised = report.get("supervised_metrics", {}) or {}
    final_info = report.get("final", {}) or {}
    data_info = report.get("data", {}) or {}
    marker_info = report.get("marker_selection", {}) or {}
    marker_source_info = report.get("marker_sources", {}) or {}

    summary = {
        "dataset": dataset_name,
        "input_path": input_h5ad,
        "prepared_h5ad": prepared_h5ad,
        "output_dir": output_dir,
        "n_cells_after_qc": data_info.get("n_cells_after_qc"),
        "n_genes_after_feature_selection": data_info.get("n_genes_after_feature_selection"),
        "n_specific_markers": marker_info.get("n_specific_markers"),
        "n_programs_mapped": marker_source_info.get("n_programs_mapped"),
        "n_clusters": final_info.get("n_clusters"),
        "ARI": supervised.get("ARI"),
        "NMI": supervised.get("NMI"),
        "FMI": supervised.get("FMI"),
        "ACC": supervised.get("ACC"),
        "status": "success",
        "error": "",
    }

    with open(os.path.join(output_dir, "short_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n数据集 {dataset_name} 运行完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


# =========================================================
# 7. 主函数
# =========================================================

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_summaries = []

    for dataset_cfg in DATASETS:
        dataset_name = dataset_cfg["name"]

        try:
            one_summary = run_one_h5_dataset(dataset_cfg)
            all_summaries.append(one_summary)

        except Exception as e:
            print("\n" + "!" * 90)
            print(f"数据集 {dataset_name} 运行失败")
            print(str(e))
            traceback.print_exc()
            print("!" * 90)

            all_summaries.append({
                "dataset": dataset_name,
                "input_path": dataset_cfg.get("input_path"),
                "prepared_h5ad": "",
                "output_dir": os.path.join(OUTPUT_ROOT, dataset_name),
                "n_cells_after_qc": None,
                "n_genes_after_feature_selection": None,
                "n_specific_markers": None,
                "n_programs_mapped": None,
                "n_clusters": None,
                "ARI": None,
                "NMI": None,
                "FMI": None,
                "ACC": None,
                "status": "failed",
                "error": str(e),
            })

        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(
            os.path.join(OUTPUT_ROOT, "msa_scvi_all_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    print("\n" + "=" * 90)
    print("全部数据集运行结束")
    print("=" * 90)

    summary_df = pd.DataFrame(all_summaries)
    print(summary_df)

    out_path = os.path.join(OUTPUT_ROOT, "msa_scvi_all_summary.csv")
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n总表已保存到：")
    print(out_path)


if __name__ == "__main__":
    main()
