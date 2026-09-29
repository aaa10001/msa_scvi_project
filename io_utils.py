import json
import os
from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


DEFAULT_BATCH_CANDIDATES = [
    "batch", "donor", "sample", "sample_id", "patient", "patient_id", "donor_id",
    "orig.ident", "tech", "method", "mouse.id", "FACS.selection", "subtissue"
]

DEFAULT_LABEL_CANDIDATES = [
    "cell_ontology_class", "celltype", "cell_type", "celltypes", "assigned_cluster",
    "free_annotation", "annotation", "annot", "label", "labels", "cluster"
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _auto_find_col(df: pd.DataFrame, candidates) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _to_dense(x):
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def load_input_data(
    input_path: str,
    csv_expr_start_col: int = 0,
    label_col: str | None = None,
    batch_col: str | None = None,
) -> ad.AnnData:
    """
    统一读取H5AD。

    对 H5AD：
    - true label 统一复制到 adata.obs["true_label"]；
    - batch 统一复制到 adata.obs["batch"]；
    - 如果存在 layers["counts"]，使用它作为 scVI counts；
    - 如果不存在 counts，则把 adata.X 复制为 pseudo-counts，并在 uns 中记录。
      Tabula Muris Senis 的 processed h5ad 通常没有 counts 层，此时属于近似处理。
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".h5ad":
        adata = sc.read_h5ad(input_path).copy()
        adata.var_names = pd.Index([str(x).upper() for x in adata.var_names])
        adata.var_names_make_unique()
        adata.obs_names = pd.Index(adata.obs_names.astype(str))
        adata.obs_names_make_unique()

        if label_col is None:
            label_col = _auto_find_col(adata.obs, DEFAULT_LABEL_CANDIDATES)
        if batch_col is None:
            batch_col = _auto_find_col(adata.obs, DEFAULT_BATCH_CANDIDATES)

        if label_col and label_col in adata.obs.columns:
            adata.obs["true_label"] = adata.obs[label_col].astype(str).values
        if batch_col and batch_col in adata.obs.columns:
            adata.obs["batch"] = adata.obs[batch_col].astype(str).values
        else:
            adata.obs["batch"] = "batch1"

        adata.obs["barcode"] = adata.obs_names.astype(str)

        has_counts = "counts" in adata.layers.keys()
        adata.uns["_input_ext"] = ".h5ad"
        adata.uns["_input_h5ad_has_counts_layer"] = bool(has_counts)
        adata.uns["_original_label_col"] = label_col if label_col else ""
        adata.uns["_original_batch_col"] = batch_col if batch_col else ""

        counts = _to_dense(adata.layers["counts"] if has_counts else adata.X)
        adata.X = counts.copy()
        adata.layers["counts"] = counts.copy()
        return adata

    if ext == ".csv":
        df = pd.read_csv(input_path)
        if csv_expr_start_col > 0:
            meta_df = df.iloc[:, :csv_expr_start_col].copy()
            expr_df = df.iloc[:, csv_expr_start_col:].copy()
        else:
            if label_col is None:
                label_col = _auto_find_col(df, DEFAULT_LABEL_CANDIDATES)
            if batch_col is None:
                batch_col = _auto_find_col(df, DEFAULT_BATCH_CANDIDATES)
            barcode_col = _auto_find_col(df, ["barcode", "cell", "cell_id", "barcode_id"]) or df.columns[0]
            excluded = [barcode_col]
            if label_col and label_col in df.columns:
                excluded.append(label_col)
            if batch_col and batch_col in df.columns and batch_col not in excluded:
                excluded.append(batch_col)
            expr_df = df[[c for c in df.columns if c not in excluded]].copy()
            meta_df = df[excluded].copy()

        expr_df.columns = [str(x).upper() for x in expr_df.columns]
        expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        adata = ad.AnnData(X=expr_df.to_numpy(dtype=np.float32))
        adata.obs_names = pd.Index(
            df.iloc[:, 0].astype(str) if "barcode" not in meta_df.columns else meta_df["barcode"].astype(str)
        )
        adata.var_names = pd.Index(expr_df.columns.astype(str))
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        adata.obs = pd.DataFrame(index=adata.obs_names)
        for c in meta_df.columns:
            adata.obs[c] = meta_df[c].astype(str).to_numpy()
        if label_col and label_col in df.columns:
            adata.obs["true_label"] = df[label_col].astype(str).to_numpy()
        if batch_col and batch_col in df.columns:
            adata.obs["batch"] = df[batch_col].astype(str).to_numpy()
        else:
            adata.obs["batch"] = "batch1"
        if "barcode" not in adata.obs.columns:
            adata.obs["barcode"] = adata.obs_names.astype(str)
        adata.layers["counts"] = _to_dense(adata.X)
        adata.uns["_input_ext"] = ".csv"
        adata.uns["_input_h5ad_has_counts_layer"] = True
        return adata

    raise ValueError(f"Unsupported input format: {ext}")


def save_selected_markers(path: str, selected_markers: dict) -> None:
    save_json(path, selected_markers)


def infer_meta_columns(adata: ad.AnnData) -> Tuple[list, list]:
    cat_cols, num_cols = [], []
    for c in adata.obs.columns:
        if pd.api.types.is_numeric_dtype(adata.obs[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return cat_cols, num_cols
