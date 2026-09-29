import numpy as np
import scvi
import torch


def set_all_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_scvi(adata, batch_col=None, n_latent: int = 20, max_epochs: int = 200, gene_likelihood: str = "nb", n_layers: int = 2, seed: int = 42):
    set_all_seeds(seed)
    adata_scvi = adata.copy()
    if batch_col and batch_col in adata_scvi.obs.columns and adata_scvi.obs[batch_col].astype(str).nunique() > 1:
        scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts", batch_key=batch_col)
    else:
        scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts")
    model = scvi.model.SCVI(
        adata_scvi,
        n_latent=n_latent,
        gene_likelihood=gene_likelihood,
        n_layers=n_layers,
    )
    model.train(max_epochs=max_epochs, early_stopping=True, accelerator="auto", devices="auto")
    latent = model.get_latent_representation()
    return model, adata_scvi, latent


def refine_latent_with_marker_prototypes(latent, program_scores_df, confidence_labels, strength: float = 0.15):
    Z = latent.copy().astype(float)
    programs = [c for c in program_scores_df.columns]
    prototypes = {}
    aligned_conf = confidence_labels.reindex(program_scores_df.index).fillna("UNSURE")

    for p in programs:
        mask = aligned_conf.values == p
        if mask.sum() < 3:
            continue
        prototypes[p] = Z[mask].mean(axis=0)

    for i, cell in enumerate(program_scores_df.index):
        label = aligned_conf.loc[cell]
        if label == "UNSURE" or label not in prototypes:
            continue
        score_vec = program_scores_df.loc[cell].values.astype(float)
        denom = np.sum(score_vec) + 1e-8
        label_score = float(program_scores_df.loc[cell, label]) if label in program_scores_df.columns else 0.0
        weight = label_score / denom if denom > 0 else 0.0
        alpha = strength * np.clip(weight, 0.0, 1.0)
        Z[i] = (1 - alpha) * Z[i] + alpha * prototypes[label]
    return Z, {k: v.tolist() for k, v in prototypes.items()}
