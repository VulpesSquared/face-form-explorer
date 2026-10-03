from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .geometry import calculate_features, quality_check
from .landmarks import FaceLandmarkerEngine


def _load_rgb(image_bytes: bytes) -> tuple[np.ndarray, int, int]:
    img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    # Avoid pathological uploads while retaining enough detail for landmarks.
    img.thumbnail((2200, 2200))
    arr = np.asarray(img)
    return arr, img.width, img.height


def analyze_dataset(
    records: Iterable[dict],
    model_path: str | Path = "models/face_landmarker.task",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze user-labeled image records.

    Each record must contain: person, filename, bytes.
    Returns image-level results, person-level median features, and QC failures.
    """
    rows: list[dict] = []
    failures: list[dict] = []

    with FaceLandmarkerEngine(model_path) as engine:
        for rec in records:
            person = str(rec.get("person", "")).strip()
            filename = str(rec.get("filename", "image"))
            if not person:
                failures.append({"person": "", "filename": filename, "reason": "Missing person label"})
                continue

            try:
                rgb, width, height = _load_rgb(rec["bytes"])
            except Exception as exc:
                failures.append({"person": person, "filename": filename, "reason": f"Could not read image: {exc}"})
                continue

            detection = engine.detect_rgb(rgb)
            if detection.landmarks is None:
                failures.append({"person": person, "filename": filename, "reason": detection.error or "Detection failed"})
                continue

            try:
                quality = quality_check(detection.landmarks, width, height)
                features = calculate_features(detection.landmarks, width, height)
            except ValueError as exc:
                failures.append({"person": person, "filename": filename, "reason": str(exc)})
                continue
            if not quality.passed:
                failures.append({
                    "person": person,
                    "filename": filename,
                    "reason": "; ".join(quality.reasons),
                })
                continue

            if not all(np.isfinite(value) for value in features.values()):
                failures.append({"person": person, "filename": filename, "reason": "Invalid landmark geometry"})
                continue
            rows.append({
                "person": person,
                "filename": filename,
                "image_width": width,
                "image_height": height,
                **{f"qc_{k}": v for k, v in asdict(quality).items() if k != "reasons"},
                **features,
            })

    image_df = pd.DataFrame(rows)
    failure_df = pd.DataFrame(failures, columns=["person", "filename", "reason"])

    if image_df.empty:
        return image_df, pd.DataFrame(), failure_df

    feature_cols = [
        c for c in image_df.columns
        if c not in {"person", "filename", "image_width", "image_height"}
        and not c.startswith("qc_")
    ]
    person_df = image_df.groupby("person", as_index=False)[feature_cols].median(numeric_only=True)
    counts = image_df.groupby("person").size().rename("accepted_images").reset_index()
    person_df = counts.merge(person_df, on="person", how="left")
    return image_df, person_df, failure_df
