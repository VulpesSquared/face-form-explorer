from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile
from typing import Optional
from urllib.request import urlopen

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


@dataclass
class LandmarkDetection:
    landmarks: Optional[np.ndarray]
    face_count: int
    error: Optional[str] = None


def ensure_model(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.is_file() and sha256(path.read_bytes()).hexdigest() == MODEL_SHA256:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        # Separate temporary files allow concurrent Streamlit sessions to start.
        with urlopen(MODEL_URL, timeout=60) as response, NamedTemporaryFile(
            dir=path.parent, suffix=".download", delete=False,
        ) as target:
            tmp = Path(target.name)
            data = response.read()
            if sha256(data).hexdigest() != MODEL_SHA256:
                raise ValueError("Downloaded model checksum does not match")
            target.write(data)
        tmp.replace(path)
    except Exception as exc:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the MediaPipe Face Landmarker model. "
            f"Download it manually from {MODEL_URL} and save it as {path}."
        ) from exc
    return path


class FaceLandmarkerEngine:
    """Thin wrapper around MediaPipe Tasks Face Landmarker.

    The model is configured to detect up to two faces so we can reject group
    photos rather than accidentally analyze the wrong person.
    """

    def __init__(self, model_path: str | Path):
        import mediapipe as mp

        if platform.system() == "Darwin" and mp.__version__ in {"1.0.0", "1.0.1"}:
            raise RuntimeError(
                "MediaPipe 1.0.x Face Landmarker can abort the native process on macOS. "
                "Install this project's requirements.txt to use MediaPipe 0.10.35 on macOS."
            )
        self.mp = mp
        model = ensure_model(model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def detect_rgb(self, rgb: np.ndarray) -> LandmarkDetection:
        try:
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(image)
        except Exception as exc:
            return LandmarkDetection(None, 0, f"MediaPipe error: {exc}")

        count = len(result.face_landmarks)
        if count == 0:
            return LandmarkDetection(None, 0, "No face detected")
        if count > 1:
            return LandmarkDetection(None, count, "More than one face detected")

        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]],
            dtype=np.float64,
        )
        return LandmarkDetection(pts, 1, None)
