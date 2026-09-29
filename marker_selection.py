import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


PANCREAS_ALLOWED_TYPES = {
    "acinar", "activated_stellate", "alpha", "beta", "delta", "ductal", "endothelial",
    "epsilon", "gamma_pp", "macrophage", "mast", "quiescent_stellate", "schwann", "t_cell",
}
PANCREAS_EXCLUDED_TYPES = {
    "b_cell", "fibroblast", "leukocyte", "mesenchymal_cell", "monocyte", "stellate", "endocrine_cell",
}

# Generalizable version: no cell-type-specific blacklists.
# 不针对某个数据集或某个 cell type 手工指定黑名单。
PROGRAM_MARKER_BLACKLIST = {}



# Generalizable version: no manually supplied extra marker whitelist.
EXTRA_PUBLIC_MARKERS = {}



def _dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def normalize_cell_type_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("β", "beta").replace("α", "alpha").replace("δ", "delta").replace("γ", "gamma")
    s = re.sub(r"\([^)]*\)", "", s)
    s = s.replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if "activated stellate" in s:
        return "activated_stellate"
    if "quiescent stellate" in s:
        return "quiescent_stellate"
    if "pancreatic stellate" in s or s == "stellate cells" or s == "stellate cell":
        return "quiescent_stellate"
    if s == "schwann cell" or "schwann" in s:
        return "schwann"
    if "acinar" in s:
        return "acinar"
    if "duct" in s:
        return "ductal"
    if "beta" in s:
        return "beta"
    if "alpha" in s:
        return "alpha"
    if "delta" in s:
        return "delta"
    if "epsilon" in s:
        return "epsilon"
    if "gamma" in s or "pancreatic polypeptide" in s or s == "pp":
        return "gamma_pp"
    if "endothelial" in s:
        return "endothelial"
    if "macrophage" in s:
        return "macrophage"
    if "mast" in s:
        return "mast"
    if "t cell" in s:
        return "t_cell"
    if "b cell" in s:
        return "b_cell"
    return s.replace(" ", "_")



def load_panglaodb_tsv(
    tsv_path: str,
    species: str = "Human",
    tissue_keywords: List[str] | None = None,
    max_markers_per_type: int = 50,
    allowed_cell_types: set | None = None,
    excluded_cell_types: set | None = None,
    canonical_only: bool = True,
    max_ubiquitousness: float = 0.08,
    source_weight: float = 1.0,
    return_support: bool = False,
):
    """
    读取 PanglaoDB marker TSV，并转换成与 CellMarker 相同的 marker_sets/support_tables 格式。

    不使用真实标签。主要字段：
    - official gene symbol
    - cell type
    - organ
    - species
    - canonical marker
    - ubiquitousness index
    - specificity_human / specificity_mouse
    - sensitivity_human / sensitivity_mouse
    """
    df = pd.read_csv(tsv_path, sep="\t")
    required_cols = ["species", "official gene symbol", "cell type", "organ"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"PanglaoDB TSV 缺少必要列: {missing}")

    df = df.copy()
    df = df.dropna(subset=["official gene symbol", "cell type"])

    # species 字段常见形式：'Mm Hs', 'Hs', 'Mm'
    if species.lower().startswith("human"):
        df = df[df["species"].astype(str).str.contains("Hs", case=False, na=False)]
        spec_col = "specificity_human"
        sens_col = "sensitivity_human"
    elif species.lower().startswith("mouse"):
        df = df[df["species"].astype(str).str.contains("Mm", case=False, na=False)]
        spec_col = "specificity_mouse"
        sens_col = "sensitivity_mouse"
    else:
        spec_col = "specificity_human"
        sens_col = "sensitivity_human"

    if tissue_keywords:
        tissue_keywords = {str(x).strip().lower() for x in tissue_keywords}
        organ_series = df["organ"].astype(str).str.strip().str.lower()
        exact = df[organ_series.isin(tissue_keywords)]
        if exact.empty:
            pat = "|".join(re.escape(str(k)) for k in tissue_keywords)
            df = df[df["organ"].astype(str).str.contains(pat, case=False, na=False)]
        else:
            df = exact

    if canonical_only and "canonical marker" in df.columns:
        df = df[df["canonical marker"].fillna(0).astype(float) >= 1]

    if "ubiquitousness index" in df.columns and max_ubiquitousness is not None:
        ubi = pd.to_numeric(df["ubiquitousness index"], errors="coerce")
        df = df[(ubi.isna()) | (ubi <= max_ubiquitousness)].copy()

    df["Symbol"] = (
        df["official gene symbol"]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
        .str.strip()
    )
    df["canonical_cell_type"] = df["cell type"].map(normalize_cell_type_name)

    if allowed_cell_types is not None:
        df = df[df["canonical_cell_type"].isin(allowed_cell_types)]
    if excluded_cell_types:
        df = df[~df["canonical_cell_type"].isin(excluded_cell_types)]

    # PanglaoDB 中 specificity 数值越大通常越特异；sensitivity 表示覆盖度。
    # 这里作为外部数据库支持强度，不与真实标签相关。
    spec = pd.to_numeric(df[spec_col], errors="coerce") if spec_col in df.columns else pd.Series(0.0, index=df.index)
    sens = pd.to_numeric(df[sens_col], errors="coerce") if sens_col in df.columns else pd.Series(0.0, index=df.index)
    canonical = pd.to_numeric(df.get("canonical marker", pd.Series(1, index=df.index)), errors="coerce").fillna(1.0)
    ubi = pd.to_numeric(df.get("ubiquitousness index", pd.Series(0, index=df.index)), errors="coerce").fillna(0.0)

    df["_panglao_support"] = (
        source_weight
        * (1.0 + canonical)
        * (1.0 + spec.fillna(0.0))
        * (1.0 + 0.25 * sens.fillna(0.0))
        * (1.0 / (1.0 + ubi))
    )

    marker_sets = {}
    support_tables = {}
    for cell_type, grp in df.groupby("canonical_cell_type"):
        # 同一 gene 多行时取最大支持度
        score = grp.groupby("Symbol")["_panglao_support"].max().sort_values(ascending=False)
        keep = score.index.tolist()[:max_markers_per_type]
        if keep:
            marker_sets[cell_type] = keep
            support_tables[cell_type] = {g: float(score.loc[g]) for g in keep}

    if return_support:
        return marker_sets, support_tables
    return marker_sets


def merge_marker_sources(
    marker_sources: List[Dict[str, List[str]]],
    support_sources: List[Dict[str, Dict[str, float]]] | None = None,
    max_markers_per_type: int = 80,
    normalize_source_support: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]]]:
    """
    合并多个 marker 数据库来源，例如 CellMarker + PanglaoDB。

    泛化版规则：
    1. 每个数据库来源的总权重相同；
    2. 不让某个数据库因为原始 support 数值范围更大而主导；
    3. 每个来源内部先按 program 做 0-1 归一化；
    4. 同时被多个数据库支持的 marker 会因为支持分数相加而自然排前。

    这满足“两个 marker 集合权重要相同”的要求。
    """
    support_sources = support_sources or [{} for _ in marker_sources]
    merged_support: Dict[str, Dict[str, float]] = {}

    n_sources = max(len(marker_sources), 1)
    source_base_weight = 1.0 / n_sources

    for marker_dict, support_dict in zip(marker_sources, support_sources):
        for program, genes in marker_dict.items():
            merged_support.setdefault(program, {})
            prog_support = support_dict.get(program, {})

            raw_scores = {}
            for rank, g in enumerate(genes):
                g = str(g).upper()
                # 如果数据库没有支持度，则用 rank-based 分数；越靠前越高。
                default_score = max(1.0, float(len(genes) - rank) / max(len(genes), 1))
                raw_scores[g] = float(prog_support.get(g, default_score))

            if not raw_scores:
                continue

            if normalize_source_support:
                vals = np.array(list(raw_scores.values()), dtype=float)
                min_v, max_v = float(vals.min()), float(vals.max())
                if max_v > min_v:
                    norm_scores = {g: (s - min_v) / (max_v - min_v) for g, s in raw_scores.items()}
                else:
                    norm_scores = {g: 1.0 for g in raw_scores}
            else:
                norm_scores = raw_scores

            for g, s in norm_scores.items():
                # 加一个很小的 rank/support 底分，防止归一化最小值变成 0 后完全消失。
                contribution = source_base_weight * (0.1 + 0.9 * float(s))
                merged_support[program][g] = merged_support[program].get(g, 0.0) + contribution

    merged_markers: Dict[str, List[str]] = {}
    for program, score_dict in merged_support.items():
        ordered = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        merged_markers[program] = [g for g, _ in ordered[:max_markers_per_type]]

    return merged_markers, merged_support



def load_cellmarker_csv(
    db_path: str,
    species: str = "Human",
    tissue_keywords: List[str] | None = None,
    max_markers_per_type: int = 50,
    allowed_cell_types: set | None = None,
    excluded_cell_types: set | None = None,
    return_support: bool = False,
):
    df = pd.read_csv(db_path)
    required_cols = ["species", "tissue_class", "cell_name", "Symbol"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CellMarker 缺少必要列: {missing}")

    df = df[required_cols].copy()
    df = df.dropna(subset=required_cols)
    df = df[df["species"].astype(str).str.lower() == species.lower()]

    if tissue_keywords:
        tissue_keywords = {str(x).strip().lower() for x in tissue_keywords}
        tissue_series = df["tissue_class"].astype(str).str.strip().str.lower()
        exact = df[tissue_series.isin(tissue_keywords)]
        if exact.empty:
            pat = "|".join(re.escape(str(k)) for k in tissue_keywords)
            df = df[df["tissue_class"].astype(str).str.contains(pat, case=False, na=False)]
        else:
            df = exact

    df["Symbol"] = df["Symbol"].astype(str).str.replace(r"\s+", "", regex=True).str.upper().str.strip()
    df["canonical_cell_type"] = df["cell_name"].map(normalize_cell_type_name)

    if allowed_cell_types is not None:
        df = df[df["canonical_cell_type"].isin(allowed_cell_types)]
    if excluded_cell_types:
        df = df[~df["canonical_cell_type"].isin(excluded_cell_types)]

    marker_sets = {}
    support_tables = {}
    for cell_type, grp in df.groupby("canonical_cell_type"):
        counts = grp["Symbol"].value_counts()
        keep = counts.index.tolist()[:max_markers_per_type]
        if keep:
            marker_sets[cell_type] = keep
            support_tables[cell_type] = counts.loc[keep].astype(int).to_dict()
    if not marker_sets:
        raise ValueError("CellMarker 过滤后没有得到有效 marker。")
    if return_support:
        return marker_sets, support_tables
    return marker_sets


def map_markers_to_data(var_names: List[str], marker_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    gene_set = {str(g).upper() for g in var_names}
    mapped = {}
    for program, genes in marker_dict.items():
        keep = [str(g).upper() for g in genes if str(g).upper() in gene_set]
        if keep:
            mapped[program] = keep
    return mapped


def filter_support_table_to_data(
    support_table: Dict[str, Dict[str, int]],
    mapped_markers: Dict[str, List[str]],
) -> Dict[str, Dict[str, int]]:
    out = {}
    for program, genes in mapped_markers.items():
        prog_support = support_table.get(program, {})
        out[program] = {g: int(prog_support.get(g, 1)) for g in genes}
    return out



def add_extra_public_markers(
    marker_sets: Dict[str, List[str]],
    support_tables: Dict[str, Dict[str, float]],
    var_names: List[str] | None = None,
    enabled: bool = False,
    extra_weight: float = 1.0,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]]]:
    """
    泛化版本默认不人工补充特定细胞类型 marker。
    该函数保留只是为了兼容 main.py，默认不做任何修改。
    """
    return marker_sets, support_tables


def compute_marker_program_counts(marker_dict: Dict[str, List[str]]) -> Dict[str, int]:
    counts = {}
    for genes in marker_dict.values():
        for g in genes:
            g = str(g).upper()
            counts[g] = counts.get(g, 0) + 1
    return counts


def compute_marker_overlap_penalty(marker_dict: Dict[str, List[str]]) -> Dict[str, float]:
    counts = {}
    for genes in marker_dict.values():
        for g in genes:
            counts[g] = counts.get(g, 0) + 1
    penalty = {}
    for g, c in counts.items():
        penalty[g] = 0.0 if c <= 1 else 1.0 - (1.0 / float(c))
    return penalty


def score_marker_programs(adata, marker_dict: Dict[str, List[str]]) -> pd.DataFrame:
    X = _dense(adata.X)
    genes = [str(g).upper() for g in adata.var_names]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    scores = {}
    for program, gene_list in marker_dict.items():
        idx = [gene_to_idx[g] for g in gene_list if g in gene_to_idx]
        if len(idx) == 0:
            continue
        scores[program] = X[:, idx].mean(axis=1)
    return pd.DataFrame(scores, index=adata.obs_names)


def select_high_confidence_cells(
    program_scores: pd.DataFrame,
    top_quantile: float = 0.85,
    min_margin: float = 0.10,
    min_cells_per_program: int = 8,
    max_cells_per_program: int | None = None,
) -> pd.Series:
    """
    按 program 单独筛高置信细胞，而不是全局统一前 15%。
    这样小类不会被大类分数整体挤掉。
    """
    if program_scores.shape[1] == 0:
        return pd.Series(index=program_scores.index, data="UNSURE")

    vals = program_scores.values
    order = np.argsort(-vals, axis=1)
    best_idx = order[:, 0]
    second_idx = order[:, 1] if vals.shape[1] > 1 else order[:, 0]
    best_score = vals[np.arange(vals.shape[0]), best_idx]
    second_score = vals[np.arange(vals.shape[0]), second_idx]
    margins = best_score - second_score
    best_program = np.array(program_scores.columns)[best_idx]

    out = pd.Series(index=program_scores.index, data="UNSURE", dtype=object)
    for program in program_scores.columns:
        mask = best_program == program
        if mask.sum() == 0:
            continue
        scores_p = best_score[mask]
        margins_p = margins[mask]
        idx_p = program_scores.index[mask]
        cutoff = np.quantile(scores_p, top_quantile) if len(scores_p) > 0 else np.inf
        keep = (scores_p >= cutoff) & (margins_p >= min_margin)

        selected_idx = list(idx_p[keep])
        if len(selected_idx) < min_cells_per_program:
            rank_df = pd.DataFrame({
                "cell": idx_p,
                "score": scores_p,
                "margin": margins_p,
            }).sort_values(["score", "margin"], ascending=False)
            fallback = rank_df[rank_df["margin"] >= min(0.0, min_margin / 2.0)]["cell"].tolist()
            need = min_cells_per_program - len(selected_idx)
            for c in fallback:
                if c not in selected_idx:
                    selected_idx.append(c)
                if len(selected_idx) >= min_cells_per_program:
                    break

        if max_cells_per_program is not None and len(selected_idx) > max_cells_per_program:
            rank_df = pd.DataFrame({
                "cell": idx_p,
                "score": scores_p,
                "margin": margins_p,
            }).set_index("cell").loc[selected_idx].reset_index().sort_values(["score", "margin"], ascending=False)
            selected_idx = rank_df["cell"].head(max_cells_per_program).tolist()

        out.loc[selected_idx] = program
    return out




def detect_and_filter_unsuitable_markers(
    selected_markers: Dict[str, List[str]],
    marker_score_table: pd.DataFrame,
    mapped_markers: Dict[str, List[str]],
    remove_blacklist: bool = False,
    remove_high_overlap: bool = False,
    overlap_count_threshold: int = 4,
    remove_ubiquitous: bool = True,
    ubiquitous_detect_rate: float = 0.80,
    min_markers_after_filter: int = 3,
    min_specificity_quantile: float = 0.05,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    泛化 marker 质控：不使用真实标签，也不使用细胞类型特异黑名单/白名单。
    判定依据只包括：
    1. 当前数据中的全局检出率是否过高，过高说明可能是泛表达基因；
    2. 当前数据中的 specificity_score 是否极低；
    3. marker 是否在过多 program 中重复出现。默认只记录，不直接删除；
    4. 过滤后每个 program 至少保留 min_markers_after_filter 个 marker。
    """
    if marker_score_table is None or marker_score_table.empty:
        return selected_markers, pd.DataFrame()

    overlap_count = compute_marker_program_counts(mapped_markers)
    score_df = marker_score_table.copy()
    score_df["gene"] = score_df["gene"].astype(str).str.upper()
    score_df["program"] = score_df["program"].astype(str)

    if "specificity_score" in score_df.columns:
        spec_cutoff = float(score_df["specificity_score"].quantile(min_specificity_quantile))
    else:
        spec_cutoff = -np.inf

    cleaned = {}
    records = []

    for program, genes in selected_markers.items():
        program = str(program)
        genes_upper = [str(g).upper() for g in genes]
        kept = []

        for g in genes_upper:
            sub = score_df[(score_df["program"] == program) & (score_df["gene"] == g)]
            detect_rate = float(sub["global_detect_rate"].iloc[0]) if (not sub.empty and "global_detect_rate" in sub.columns) else np.nan
            specificity_score = float(sub["specificity_score"].iloc[0]) if (not sub.empty and "specificity_score" in sub.columns) else np.nan
            overlap = int(overlap_count.get(g, 1))

            reasons = []

            # 不使用 program-specific blacklist，仅保留接口兼容。
            if remove_blacklist and g in PROGRAM_MARKER_BLACKLIST.get(program, set()):
                reasons.append("program_blacklist")

            # 默认不删除高 overlap marker，只记录；共享 marker 在打分阶段已自动降权。
            if overlap >= overlap_count_threshold:
                reasons.append(f"high_overlap_{overlap}_flag")

            if remove_high_overlap and overlap >= overlap_count_threshold:
                reasons.append(f"high_overlap_{overlap}_remove")

            if remove_ubiquitous and not np.isnan(detect_rate) and detect_rate >= ubiquitous_detect_rate:
                reasons.append("ubiquitous_detect_rate")

            if not np.isnan(specificity_score) and specificity_score <= spec_cutoff:
                reasons.append("very_low_specificity_score")

            should_remove = (
                "ubiquitous_detect_rate" in reasons
                or "very_low_specificity_score" in reasons
                or any(r.endswith("_remove") for r in reasons)
                or "program_blacklist" in reasons
            )

            action = "remove" if should_remove else "keep"
            if not should_remove:
                kept.append(g)

            records.append({
                "program": program,
                "gene": g,
                "action": action,
                "reason": ";".join(reasons),
                "overlap_count": overlap,
                "global_detect_rate": detect_rate,
                "specificity_score": specificity_score,
            })

        # 保底补足：从同一 program 的候选中按 specificity_score 补足。
        # 仍然只依赖表达统计和数据库支持，不依赖真实标签。
        if len(kept) < min_markers_after_filter:
            candidates = score_df[score_df["program"] == program].sort_values(
                ["specificity_score", "db_support", "global_detect_rate"],
                ascending=False
            )
            for _, row in candidates.iterrows():
                g = str(row["gene"]).upper()
                if g in kept:
                    continue
                detect_rate = float(row.get("global_detect_rate", np.nan))
                specificity_score = float(row.get("specificity_score", np.nan))
                overlap = int(overlap_count.get(g, 1))

                if remove_ubiquitous and not np.isnan(detect_rate) and detect_rate >= ubiquitous_detect_rate:
                    continue
                if not np.isnan(specificity_score) and specificity_score <= spec_cutoff:
                    continue
                if remove_high_overlap and overlap >= overlap_count_threshold:
                    continue

                kept.append(g)
                records.append({
                    "program": program,
                    "gene": g,
                    "action": "fallback_keep",
                    "reason": "min_marker_floor",
                    "overlap_count": overlap,
                    "global_detect_rate": detect_rate,
                    "specificity_score": specificity_score,
                })
                if len(kept) >= min_markers_after_filter:
                    break

        cleaned[program] = kept

    report = pd.DataFrame(records)
    return cleaned, report


def compute_specific_markers_unsupervised(
    adata,
    mapped_markers: Dict[str, List[str]],
    support_table: Dict[str, Dict[str, int]],
    n_per_program: int = 15,
    min_per_program_floor: int = 3,
    min_marker_mean: float = 0.05,
    min_marker_detect_rate: float = 0.05,
    overlap_weight: float = 1.0,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    完全不使用真实标签。
    新增每个 program 的 marker 保底保留，避免小类 program 直接被筛成空。
    """
    X = _dense(adata.X)
    genes = np.array([str(g).upper() for g in adata.var_names])
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    overlap_penalty = compute_marker_overlap_penalty(mapped_markers)

    selected: Dict[str, List[str]] = {}
    rows = []

    global_mean = X.mean(axis=0)
    global_detect = (X > 0).mean(axis=0)
    global_std = X.std(axis=0)
    cv = global_std / np.maximum(global_mean, 1e-6)

    for program, genes_prog in mapped_markers.items():
        idx = [gene_to_idx[g] for g in genes_prog if g in gene_to_idx]
        if not idx:
            selected[program] = []
            continue

        support_dict = support_table.get(program, {})
        support = np.array([float(support_dict.get(genes[i], 1.0)) for i in idx], dtype=float)
        support_norm = support / np.maximum(support.max(), 1.0)
        mean_v = global_mean[idx]
        detect_v = global_detect[idx]
        cv_v = np.clip(cv[idx], 0.0, 5.0) / 5.0
        penalties = np.array([overlap_penalty.get(genes[i], 0.0) for i in idx], dtype=float)

        sub = pd.DataFrame({
            "program": program,
            "gene": genes[idx],
            "db_support": support,
            "db_support_norm": support_norm,
            "global_mean": mean_v,
            "global_detect_rate": detect_v,
            "global_cv_norm": cv_v,
            "overlap_penalty": penalties,
        })

        sub["passes_base_filter"] = (
            (sub["global_mean"] >= min_marker_mean)
            & (sub["global_detect_rate"] >= min_marker_detect_rate)
        )

        # 共享 marker 不直接删除，而是软降权，减少内分泌共享 marker 对区分边界的干扰。
        overlap_count = 1.0 / np.maximum(1.0 - sub["overlap_penalty"], 1e-6)
        shared_weight = np.where(overlap_count >= 3, 0.30, np.where(overlap_count >= 2, 0.60, 1.0))
        sub["specificity_score"] = (
            sub["db_support_norm"]
            * np.log1p(sub["global_mean"])
            * sub["global_detect_rate"]
            * (0.5 + 0.5 * sub["global_cv_norm"])
            * shared_weight
        )

        sub = sub.sort_values(["specificity_score", "db_support", "global_detect_rate"], ascending=False)

        normal_keep = sub[sub["passes_base_filter"]]["gene"].head(n_per_program).tolist()
        if len(normal_keep) < min_per_program_floor:
            fallback_pool = sub["gene"].tolist()
            for g in fallback_pool:
                if g not in normal_keep:
                    normal_keep.append(g)
                if len(normal_keep) >= min_per_program_floor:
                    break

        selected[program] = normal_keep[:max(n_per_program, min_per_program_floor)]
        rows.append(sub)

    score_df = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    return selected, score_df
