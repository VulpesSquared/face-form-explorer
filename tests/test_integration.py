from pathlib import Path
import platform

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from face_analysis.landmarks import FaceLandmarkerEngine
from face_analysis.pipeline import _load_rgb, analyze_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
MODEL = ROOT / "models" / "face_landmarker.task"
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    import mediapipe as mp

    assert mp.__version__ == ("0.10.35" if platform.system() == "Darwin" else "1.0.1")
    for path in [MODEL, *(DATA / name for name in ["portrait.jpg", "portrait_small.jpg", "grace_hopper.jpg"])]:
        assert path.exists(), "Run python -m scripts.download_test_assets first"
    with FaceLandmarkerEngine(MODEL) as instance:
        yield instance


@pytest.mark.parametrize("filename", ["portrait.jpg", "portrait_small.jpg", "grace_hopper.jpg"])
def test_real_portraits_have_478_landmarks(engine, filename):
    rgb, _, _ = _load_rgb((DATA / filename).read_bytes())
    result = engine.detect_rgb(rgb)
    assert result.face_count == 1
    assert result.error is None
    assert result.landmarks.shape == (478, 3)
    assert np.isfinite(result.landmarks).all()


def test_blank_image_and_multiple_faces_are_rejected(engine):
    result = engine.detect_rgb(np.zeros((800, 800, 3), dtype=np.uint8))
    assert result.face_count == 0
    assert result.landmarks is None
    assert result.error == "No face detected"
    # Crop two copies of the same fixture to make a controlled group photo.
    with Image.open(DATA / "portrait.jpg") as original:
        portrait = original.crop((200, 0, 650, 600))
        group = Image.new("RGB", (900, 600))
        group.paste(portrait, (0, 0))
        group.paste(portrait, (450, 0))
    result = engine.detect_rgb(np.asarray(group))
    assert result.face_count == 2
    assert result.landmarks is None
    assert result.error == "More than one face detected"


def test_real_pipeline_aggregates_user_labels_and_reports_failures(engine):
    records = [
        {"person": label, "filename": filename, "bytes": (DATA / filename).read_bytes()}
        for label, filename in [("Sample A", "portrait.jpg"), ("Sample B", "grace_hopper.jpg"), ("Sample A", "portrait_small.jpg")]
    ]
    records.append({**records[0], "filename": "another.jpg"})
    records.append({**records[0], "person": "", "filename": "unlabeled.jpg"})
    records.append({"person": "Sample C", "filename": "broken.jpg", "bytes": b"broken"})
    images, people, failures = analyze_dataset(records, MODEL)
    assert len(images) == 3
    assert people.set_index("person")["accepted_images"].to_dict() == {"Sample A": 2, "Sample B": 1}
    reasons = failures.set_index("filename")["reason"].to_dict()
    assert "too small" in reasons["portrait_small.jpg"]
    assert reasons["unlabeled.jpg"] == "Missing person label"
    assert "Could not read image" in reasons["broken.jpg"]
    assert np.isfinite(people.select_dtypes("number").to_numpy()).all()


def test_streamlit_real_upload_analysis_and_exports(engine):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.file_uploader[0].set_value([
        (f"{label}__{filename}", (DATA / filename).read_bytes(), "image/jpeg")
        for label, filename in [("Sample A", "portrait.jpg"), ("Sample B", "grace_hopper.jpg"), ("Sample A", "portrait_small.jpg")]
    ]).run()
    at.button[0].click().run()
    assert not at.exception
    assert not at.error
    assert len(at.tabs) == 6
    assert len(at.get("download_button")) == 2
    assert [m.value for m in at.metric if m.label == "Images accepted"] == ["2"]
    assert [m.value for m in at.metric if m.label == "Images rejected"] == ["1"]
