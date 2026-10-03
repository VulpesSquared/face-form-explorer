from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .geometry import FEATURE_GROUPS, FEATURE_LABELS

META = {"person", "accepted_images", "cluster", "pca_x", "pca_y"}


def feature_columns(person_df: pd.DataFrame) -> list[str]:
    return [c for c in person_df.columns if c not in META and pd.api.types.is_numeric_dtype(person_df[c])]


def standardized_features(person_df: pd.DataFrame, selected: Iterable[str] | None = None) -> tuple[pd.DataFrame, RobustScaler]:
    cols = list(selected) if selected else feature_columns(person_df)
    X = person_df[cols].astype(float).copy()
    X = X.fillna(X.median(numeric_only=True))
    # Constant columns are okay; RobustScaler maps them to zero.
    scaler = RobustScaler(quantile_range=(20, 80))
    Z = scaler.fit_transform(X)
    return pd.DataFrame(Z, columns=cols, index=person_df.index), scaler


def build_similarity_matrix(person_df: pd.DataFrame, selected: Iterable[str] | None = None) -> pd.DataFrame:
    if len(person_df) < 2:
        names = person_df.get("person", pd.Series(dtype=str)).tolist()
        return pd.DataFrame(np.eye(len(names)) * 100.0, index=names, columns=names)

    Z, _ = standardized_features(person_df, selected)
    arr = Z.to_numpy(float)
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.mean(diff ** 2, axis=2))
    # Relative morphology index, not an identity probability.
    sim = np.exp(-0.70 * dist) * 100.0
    np.fill_diagonal(sim, 100.0)
    names = person_df["person"].tolist()
    return pd.DataFrame(sim, index=names, columns=names)


def group_similarity(person_df: pd.DataFrame, person_a: str, person_b: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for group, features in FEATURE_GROUPS.items():
        available = [f for f in features if f in person_df.columns]
        if not available:
            continue
        sim = build_similarity_matrix(person_df, available)
        out[group] = float(sim.loc[person_a, person_b])
    return out


def compare_people(person_df: pd.DataFrame, person_a: str, person_b: str) -> dict:
    Z, _ = standardized_features(person_df)
    names = person_df["person"].tolist()
    ia, ib = names.index(person_a), names.index(person_b)
    diffs = (Z.iloc[ia] - Z.iloc[ib]).abs().sort_values()
    sim = build_similarity_matrix(person_df)

    def rows(series: pd.Series) -> list[dict]:
        return [
            {
                "feature": f,
                "label": FEATURE_LABELS.get(f, f.replace("_", " ").title()),
                "standardized_difference": float(v),
            }
            for f, v in series.items()
        ]

    return {
        "overall": float(sim.loc[person_a, person_b]),
        "groups": group_similarity(person_df, person_a, person_b),
        "most_similar": rows(diffs.head(6)),
        "most_different": rows(diffs.tail(6).sort_values(ascending=False)),
    }
