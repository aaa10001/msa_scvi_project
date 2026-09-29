from itertools import combinations
from typing import List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score


def _run_leiden(embedding: np.ndarray, n_neighbors: int, resolution: float, seed: int) -> np.ndarray:
    adata_tmp = ad.AnnData(X=embedding.copy())
    adata_tmp.obsm["X_emb"] = embedding.copy()
    sc.pp.neighbors(
        adata_tmp,
        n_neighbors=min(max(5, n_neighbors), max(5, embedding.shape[0] - 1)),
        use_rep="X_emb",
        random_state=seed,
    )
    sc.tl.leiden(adata_tmp, resolution=resolution, random_state=seed, key_added="leiden")
    return adata_tmp.obs["leiden"].astype(str).values


def average_pairwise_ari(clusterings: List[np.ndarray]) -> float:
    if len(clusterings) < 2:
        return 0.0
    vals = []
    for a, b in combinations(clusterings, 2):
        vals.append(adjusted_rand_score(a, b))
    return float(np.mean(vals)) if vals else 0.0


def marker_specificity_score(cluster_labels: np.ndarray, program_scores: pd.DataFrame) -> float:
    df = program_scores.copy()
    df["cluster"] = cluster_labels
    cluster_scores = []
    for _, sub in df.groupby("cluster"):
        mean_scores = sub.drop(columns=["cluster"]).mean(axis=0).sort_values(ascending=False)
        if len(mean_scores) < 2:
            cluster_scores.append(0.0)
            continue
        gap = float(mean_scores.iloc[0] - mean_scores.iloc[1])
        dominance = float(mean_scores.iloc[0] / (mean_scores.sum() + 1e-8))
        cluster_scores.append(0.5 * gap + 0.5 * dominance)
    return float(np.mean(cluster_scores)) if cluster_scores else 0.0


def cluster_interpretability_stats(cluster_labels: np.ndarray, program_scores: pd.DataFrame) -> dict:
    """
    计算一个聚类结果的可解释性。
    不使用真实标签，只看每个预测簇是否存在明显占优的 marker program。
    """
    df = program_scores.copy()
    df["cluster"] = cluster_labels

    dominance_vals = []
    gap_vals = []
    min_child_frac = 1.0

    for _, sub in df.groupby("cluster"):
        frac = len(sub) / max(len(df), 1)
        min_child_frac = min(min_child_frac, frac)

        means = sub.drop(columns=["cluster"]).mean(axis=0).sort_values(ascending=False)
        if len(means) < 2:
            dominance_vals.append(0.0)
            gap_vals.append(0.0)
            continue

        dominance_vals.append(float(means.iloc[0] / (means.sum() + 1e-8)))
        gap_vals.append(float(means.iloc[0] - means.iloc[1]))

    return {
        "mean_dominance": float(np.mean(dominance_vals)) if dominance_vals else 0.0,
        "mean_top_gap": float(np.mean(gap_vals)) if gap_vals else 0.0,
        "min_child_fraction": float(min_child_frac),
    }


def batch_mixing_score(cluster_labels: np.ndarray, batch_series: pd.Series) -> float:
    if batch_series is None:
        return 0.0
    df = pd.DataFrame({"cluster": cluster_labels, "batch": batch_series.values})
    vals = []
    for _, sub in df.groupby("cluster"):
        props = sub["batch"].value_counts(normalize=True)
        entropy = -(props * np.log(props + 1e-12)).sum()
        norm = np.log(max(len(props), 2))
        vals.append(float(entropy / norm) if norm > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def oversplit_penalty(cluster_labels: np.ndarray, min_frac: float = 0.02) -> float:
    vc = pd.Series(cluster_labels).value_counts(normalize=True)
    tiny = (vc < min_frac).sum()
    return float(tiny / max(len(vc), 1))


def search_multiscale_stable_clusters(
    embedding: np.ndarray,
    program_scores: pd.DataFrame,
    batch_series: pd.Series,
    neighbors_grid: List[int],
    resolution_grid: List[float],
    repeats: int,
    alpha: float,
    beta: float,
    gamma: float,
    base_seed: int = 42,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    多尺度聚类搜索。
    重要：这里不使用真实标签，也不使用固定 target 聚类数。
    评分只来自内部无标签指标：稳定性、marker 可解释性、batch mixing、过分裂惩罚。
    """
    rows = []
    run_cache = []
    for nn in neighbors_grid:
        for res in resolution_grid:
            clusterings = []
            for r in range(repeats):
                seed = base_seed + 100 * r + int(res * 100) + nn
                labels = _run_leiden(embedding, nn, res, seed)
                clusterings.append(labels)

            stability = average_pairwise_ari(clusterings)
            rep = clusterings[0]
            marker_score = marker_specificity_score(rep, program_scores)
            batch_score = batch_mixing_score(rep, batch_series) if batch_series is not None else 0.0
            penalty = oversplit_penalty(rep)

            total = alpha * stability + beta * marker_score + 0.10 * batch_score - gamma * penalty

            rows.append({
                "n_neighbors": nn,
                "resolution": res,
                "stability_ari": stability,
                "marker_specificity": marker_score,
                "batch_mixing": batch_score,
                "oversplit_penalty": penalty,
                "total_score": total,
                "n_clusters": int(len(np.unique(rep))),
            })
            run_cache.append({
                "n_neighbors": nn,
                "resolution": res,
                "clusterings": clusterings,
                "representative": rep,
                "total_score": total,
            })
    return pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True), run_cache


def build_consensus_from_top_results(run_cache: List[dict], n_cells: int, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    top = sorted(run_cache, key=lambda x: x["total_score"], reverse=True)[:top_k]
    consensus = np.zeros((n_cells, n_cells), dtype=float)
    count = 0
    for item in top:
        for labels in item["clusterings"]:
            same = (labels[:, None] == labels[None, :]).astype(float)
            consensus += same
            count += 1
    consensus = consensus / max(count, 1)
    dist = 1.0 - consensus
    Z = linkage(squareform(dist, checks=False), method="average")
    final_labels = fcluster(Z, t=0.5, criterion="distance").astype(str)
    return consensus, final_labels


def cluster_program_entropy(cluster_labels: np.ndarray, program_scores: pd.DataFrame) -> pd.DataFrame:
    df = program_scores.copy()
    df["cluster"] = cluster_labels
    rows = []
    for cl, sub in df.groupby("cluster"):
        means = sub.drop(columns=["cluster"]).mean(axis=0)
        probs = means / max(float(means.sum()), 1e-8)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        max_entropy = float(np.log(max(len(probs), 2)))
        norm_entropy = entropy / max(max_entropy, 1e-8)
        sorted_means = means.sort_values(ascending=False)
        top_gap = float(sorted_means.iloc[0] - sorted_means.iloc[1]) if len(sorted_means) > 1 else 0.0
        dominant_program = str(sorted_means.index[0]) if len(sorted_means) else "unknown"
        dominant_share = float(sorted_means.iloc[0] / (sorted_means.sum() + 1e-8)) if len(sorted_means) else 0.0
        rows.append({
            "cluster": str(cl),
            "n_cells": int(len(sub)),
            "program_entropy": norm_entropy,
            "top_gap": top_gap,
            "dominant_program": dominant_program,
            "dominant_share": dominant_share,
        })
    return pd.DataFrame(rows)


def refine_mixed_clusters(
    labels: np.ndarray,
    embedding: np.ndarray,
    program_scores: pd.DataFrame,
    min_cluster_size: int = 60,
    entropy_threshold: float = 0.85,
    top_gap_threshold: float = 0.08,
    resolution_grid: List[float] | None = None,
    max_subclusters: int = 4,
    min_improvement: float = 0.03,
    base_seed: int = 42,
    local_repeats: int = 4,
    min_local_stability: float = 0.85,
    child_min_cells: int = 30,
    child_min_fraction: float = 0.03,
    child_min_dominance: float = 0.40,
    child_min_top_gap: float = 0.02,
    protect_dominant_share: float = 0.70,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    泛化版二次细分：
    - 不使用真实标签；
    - 不使用细胞类型黑白名单；
    - 不强行追求更多簇；
    - 只有当局部细分同时满足“稳定、可解释、子簇不太小、marker specificity 有提升”时才接受。

    这样做的目的：
    1. 防止把 beta/acinar/ductal 这类大纯簇切碎；
    2. 对真正混杂且局部结构稳定的簇进行二次细分；
    3. 保持跨数据集泛化能力。
    """
    resolution_grid = resolution_grid or [0.2, 0.3, 0.4, 0.5, 0.6]
    out = pd.Series(labels.astype(str)).copy()
    out.index = np.arange(len(out))
    stats = cluster_program_entropy(labels, program_scores)
    records = []

    for _, row in stats.iterrows():
        cl = str(row["cluster"])
        n_cells = int(row["n_cells"])
        ent = float(row["program_entropy"])
        top_gap = float(row["top_gap"])
        dominant_share = float(row.get("dominant_share", 0.0))
        dominant_program = str(row.get("dominant_program", "unknown"))

        mixed_trigger = (ent >= entropy_threshold) or (top_gap <= top_gap_threshold)
        if n_cells < min_cluster_size or not mixed_trigger:
            continue

        # 如果父簇已经被单个 marker program 明显主导，默认不拆。
        # 这是通用规则，不针对具体细胞类型。
        if dominant_share >= protect_dominant_share:
            records.append({
                "parent_cluster": cl,
                "n_cells": n_cells,
                "decision": "skip_protected_dominant_parent",
                "parent_program_entropy": ent,
                "parent_top_gap": top_gap,
                "parent_dominant_program": dominant_program,
                "parent_dominant_share": dominant_share,
            })
            continue

        idx = np.where(labels.astype(str) == cl)[0]
        emb_sub = embedding[idx]
        score_sub = program_scores.iloc[idx].copy()
        parent_labels = np.array([cl] * len(idx), dtype=str)
        parent_score = marker_specificity_score(parent_labels, score_sub)
        local_neighbors = min(15, max(5, len(idx) - 1))

        best = None
        for res in resolution_grid:
            local_clusterings = []
            for r in range(local_repeats):
                seed = base_seed + 1000 + 101 * r + int(float(res) * 100) + len(idx)
                pred = _run_leiden(emb_sub, local_neighbors, res, seed)
                local_clusterings.append(pred)

            stability = average_pairwise_ari(local_clusterings)
            pred = local_clusterings[0]
            n_sub = len(np.unique(pred))
            if n_sub <= 1 or n_sub > max_subclusters:
                continue

            vc = pd.Series(pred).value_counts()
            min_child_n = int(vc.min())
            min_child_frac = float(vc.min() / max(len(pred), 1))
            if min_child_n < child_min_cells or min_child_frac < child_min_fraction:
                continue

            child_marker_score = marker_specificity_score(pred, score_sub)
            interp = cluster_interpretability_stats(pred, score_sub)
            penalty = oversplit_penalty(pred, min_frac=max(0.08, child_min_fraction))
            local_score = child_marker_score + 0.15 * stability - 0.10 * penalty

            if best is None or local_score > best["local_score"]:
                best = {
                    "resolution": float(res),
                    "pred": pred,
                    "n_sub": int(n_sub),
                    "stability": float(stability),
                    "child_marker_score": float(child_marker_score),
                    "local_score": float(local_score),
                    "penalty": float(penalty),
                    "min_child_n": min_child_n,
                    "min_child_frac": min_child_frac,
                    "mean_dominance": float(interp["mean_dominance"]),
                    "mean_top_gap": float(interp["mean_top_gap"]),
                }

        if best is None:
            records.append({
                "parent_cluster": cl,
                "n_cells": n_cells,
                "decision": "skip_no_valid_local_split",
                "parent_program_entropy": ent,
                "parent_top_gap": top_gap,
                "parent_dominant_program": dominant_program,
                "parent_dominant_share": dominant_share,
                "parent_marker_specificity": float(parent_score),
            })
            continue

        improvement = best["child_marker_score"] - parent_score
        accept = (
            best["stability"] >= min_local_stability
            and best["mean_dominance"] >= child_min_dominance
            and best["mean_top_gap"] >= child_min_top_gap
            and improvement >= min_improvement
        )

        record = {
            "parent_cluster": cl,
            "n_cells": n_cells,
            "decision": "accept" if accept else "reject",
            "parent_program_entropy": ent,
            "parent_top_gap": top_gap,
            "parent_dominant_program": dominant_program,
            "parent_dominant_share": dominant_share,
            "parent_marker_specificity": float(parent_score),
            "chosen_resolution": float(best["resolution"]),
            "child_clusters": int(best["n_sub"]),
            "child_marker_specificity": float(best["child_marker_score"]),
            "improvement": float(improvement),
            "local_stability": float(best["stability"]),
            "child_mean_dominance": float(best["mean_dominance"]),
            "child_mean_top_gap": float(best["mean_top_gap"]),
            "min_child_cells": int(best["min_child_n"]),
            "min_child_fraction": float(best["min_child_frac"]),
            "oversplit_penalty": float(best["penalty"]),
        }
        records.append(record)

        if not accept:
            continue

        pred = pd.Series(best["pred"], index=idx)
        for sub_id, sub_idx in pred.groupby(pred).groups.items():
            out.loc[list(sub_idx)] = f"{cl}.{sub_id}"

    uniq = pd.unique(out)
    mapping = {old: str(i) for i, old in enumerate(uniq)}
    relabeled = out.map(mapping).values.astype(str)
    return relabeled, pd.DataFrame(records)
