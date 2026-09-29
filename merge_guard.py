import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def maybe_merge_oversplit_clusters(
    labels: np.ndarray,
    embedding: np.ndarray,
    program_scores: pd.DataFrame,
    small_cluster_threshold: float = 0.02,
    marker_gap_threshold: float = 0.10,
):
    ser = pd.Series(labels, name="cluster")
    props = ser.value_counts(normalize=True)
    unique = list(props.index)
    cluster_centers = {}
    top_program = {}
    top_gap = {}

    tmp = program_scores.copy()
    tmp["cluster"] = labels
    for cl, sub in tmp.groupby("cluster"):
        mean_scores = sub.drop(columns=["cluster"]).mean(axis=0).sort_values(ascending=False)
        cluster_centers[cl] = embedding[ser.values == cl].mean(axis=0)
        top_program[cl] = mean_scores.index[0]
        top_gap[cl] = float(mean_scores.iloc[0] - mean_scores.iloc[1]) if len(mean_scores) > 1 else 0.0

    new_labels = ser.copy()
    changed = 0
    for cl in unique:
        if props[cl] >= small_cluster_threshold:
            continue
        center = cluster_centers[cl]
        candidates = [c for c in unique if c != cl and top_program.get(c) == top_program.get(cl)]
        if not candidates:
            continue
        dists = {c: np.linalg.norm(center - cluster_centers[c]) for c in candidates}
        target = min(dists, key=dists.get)
        if min(top_gap.get(cl, 0.0), top_gap.get(target, 0.0)) < marker_gap_threshold:
            new_labels[new_labels == cl] = target
            changed += 1

    # relabel contiguous strings
    uniq = pd.unique(new_labels)
    mapping = {old: str(i) for i, old in enumerate(uniq)}
    relabeled = new_labels.map(mapping).values
    return relabeled, {"merge_operations": int(changed), "final_n_clusters": int(len(np.unique(relabeled)))}
