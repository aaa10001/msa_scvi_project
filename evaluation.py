import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score


def clustering_accuracy(y_true, y_pred):
    true_labels = pd.Categorical(y_true).codes
    pred_labels = pd.Categorical(y_pred).codes

    D = max(pred_labels.max(), true_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(len(pred_labels)):
        w[pred_labels[i], true_labels[i]] += 1

    r, c = linear_sum_assignment(w.max() - w)
    return float(w[r, c].sum() / len(pred_labels))


def compute_supervised_metrics(y_true, y_pred):
    """
    外部评价指标。
    """
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "FMI": float(fowlkes_mallows_score(y_true, y_pred)),
        "ACC": float(clustering_accuracy(y_true, y_pred)),
    }


def plot_embeddings(adata, color_col: str, path: str, title: str):
    adata_plot = adata.copy()

    if "X_umap" not in adata_plot.obsm:
        sc.pp.neighbors(adata_plot, use_rep="X_emb")
        sc.tl.umap(adata_plot)

    sc.pl.umap(adata_plot, color=color_col, show=False, save=None)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_heatmap(true_labels, pred_labels, path: str):
    tab = pd.crosstab(
        pd.Series(true_labels, name="true"),
        pd.Series(pred_labels, name="pred"),
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(tab.values, aspect="auto")
    plt.xticks(range(tab.shape[1]), tab.columns, rotation=90)
    plt.yticks(range(tab.shape[0]), tab.index)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
