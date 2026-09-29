import numpy as np
import scanpy as sc
from scipy import sparse


def _to_dense(x):
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def preprocess_adata(
    adata,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    max_mt_pct: float | None = 20.0,
    target_sum: float = 1e4,
    n_top_hvg: int = 2000,
):
    """
    统一预处理。

    CSV / 有 counts 层的 h5ad：
        counts -> QC -> normalize_total -> log1p -> HVG

    Tabula Muris Senis processed h5ad 通常没有 counts 层：
        adata.X 已经是官方 processed 表达矩阵。
        为避免二次 normalize/log，这里保留 X，只做必要过滤和 HVG 标记。
        同时把 X 复制到 layers["counts"]，供后续 scVI 接口使用。
    """
    adata = adata.copy()
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    input_ext = adata.uns.get("_input_ext", "")
    h5_has_counts = bool(adata.uns.get("_input_h5ad_has_counts_layer", True))
    processed_h5_without_counts = (input_ext == ".h5ad" and not h5_has_counts)

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

    # 对 processed h5ad，obs 中通常已经有 n_genes/n_counts。
    # 如果没有，则用当前 X 估算。
    try:
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    except Exception:
        pass

    if min_genes_per_cell is not None:
        if "n_genes_by_counts" in adata.obs.columns:
            adata = adata[adata.obs["n_genes_by_counts"] >= min_genes_per_cell].copy()
        elif "n_genes" in adata.obs.columns:
            adata = adata[adata.obs["n_genes"] >= min_genes_per_cell].copy()

    if max_mt_pct is not None and "pct_counts_mt" in adata.obs.columns:
        adata = adata[adata.obs["pct_counts_mt"] <= max_mt_pct].copy()

    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    adata.layers["counts"] = _to_dense(adata.layers.get("counts", adata.X)).copy()

    if processed_h5_without_counts:
        # 官方 processed h5ad 已经处理过，不再 normalize/log。
        if "highly_variable" not in adata.var.columns or int(np.sum(adata.var["highly_variable"].values)) < 100:
            try:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=min(n_top_hvg, adata.n_vars),
                    flavor="cell_ranger",
                    batch_key="batch" if "batch" in adata.obs.columns and adata.obs["batch"].astype(str).nunique() > 1 else None,
                )
            except Exception:
                adata.var["highly_variable"] = False
                # 兜底：按方差取前 n_top_hvg
                X = get_dense_matrix(adata)
                var = np.var(X, axis=0)
                idx = np.argsort(-var)[: min(n_top_hvg, adata.n_vars)]
                adata.var.iloc[idx, adata.var.columns.get_loc("highly_variable")] = True
        else:
            # 如果官方已经给了 HVG，但数量过多，则按 dispersions_norm 截断到 n_top_hvg。
            hv = adata.var["highly_variable"].astype(bool).values
            if hv.sum() > n_top_hvg and "dispersions_norm" in adata.var.columns:
                adata.var["highly_variable"] = False
                cand = adata.var.sort_values("dispersions_norm", ascending=False).head(min(n_top_hvg, adata.n_vars)).index
                adata.var.loc[cand, "highly_variable"] = True
        adata.raw = adata
        return adata

    # 原始 counts 数据：标准 Scanpy 预处理。
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(n_top_hvg, adata.n_vars),
            flavor="seurat_v3",
            layer="counts",
            batch_key="batch" if "batch" in adata.obs.columns and adata.obs["batch"].astype(str).nunique() > 1 else None,
        )
    except Exception:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(n_top_hvg, adata.n_vars),
            flavor="cell_ranger",
            batch_key="batch" if "batch" in adata.obs.columns and adata.obs["batch"].astype(str).nunique() > 1 else None,
        )
    adata.raw = adata
    return adata


def get_dense_matrix(adata):
    X = adata.X
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)
