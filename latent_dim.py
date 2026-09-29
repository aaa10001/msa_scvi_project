import numpy as np
from sklearn.decomposition import PCA


def estimate_latent_dim(X, min_dim: int = 10, max_dim: int = 30) -> dict:
    n_features = X.shape[1]
    max_dim = min(max_dim, n_features - 1 if n_features > 1 else 1)
    pca = PCA(n_components=min(max(50, max_dim), n_features))
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    knee = int(np.argmax(cum >= 0.85) + 1)
    # 简化版 noise floor：看相邻 explained variance 的变化
    diffs = np.abs(np.diff(evr, prepend=evr[0]))
    flat_idx = int(np.argmax(diffs < np.median(diffs) if np.median(diffs) > 0 else np.zeros_like(diffs, dtype=bool)) + 1)
    rec = max(min_dim, min(max_dim, max(knee, flat_idx)))

    return {
        "recommended_latent_dim": int(rec),
        "knee_dim_85pct": int(knee),
        "flat_signal_dim": int(flat_idx),
        "cum_explained_variance": cum[:max_dim].tolist(),
        "explained_variance_ratio": evr[:max_dim].tolist(),
    }



