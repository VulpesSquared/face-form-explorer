from __future__ import annotations

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from .similarity import standardized_features


def cluster_people(person_df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    if len(person_df) < 2:
        out = person_df.copy()
        out["cluster"] = 1
        out["pca_x"] = 0.0
        out["pca_y"] = 0.0
        return out

    n_clusters = max(2, min(n_clusters, len(person_df)))
    Z, _ = standardized_features(person_df)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(Z.to_numpy()) + 1

    if Z.shape[1] >= 2 and len(person_df) >= 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(Z.to_numpy())
    else:
        coords = [[0.0, 0.0] for _ in range(len(person_df))]

    out = person_df[["person", "accepted_images"]].copy()
    out["cluster"] = labels
    out["pca_x"] = [float(x[0]) for x in coords]
    out["pca_y"] = [float(x[1]) for x in coords]
    return out
