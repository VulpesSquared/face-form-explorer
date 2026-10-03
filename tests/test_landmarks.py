from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
import sys

import pytest

from face_analysis import landmarks


def test_cached_model_checksum_is_verified(tmp_path, monkeypatch):
    content = b"verified model"
    path = tmp_path / "face.task"
    path.write_bytes(content)
    monkeypatch.setattr(landmarks, "MODEL_SHA256", sha256(content).hexdigest())
    monkeypatch.setattr(landmarks, "urlopen", lambda *a, **k: pytest.fail("Unexpected network request"))
    assert landmarks.ensure_model(path) == path


def test_corrupt_download_is_not_installed(tmp_path, monkeypatch):
    path = tmp_path / "face.task"
    monkeypatch.setattr(landmarks, "urlopen", lambda *a, **k: BytesIO(b"invalid model"))
    with pytest.raises(RuntimeError, match="Could not download"):
        landmarks.ensure_model(path)
    assert not path.exists()
    assert not list(tmp_path.glob("*.download"))


def test_corrupt_cached_model_is_replaced(tmp_path, monkeypatch):
    content = b"verified model"
    path = tmp_path / "face.task"
    path.write_bytes(b"corrupt cached data")
    monkeypatch.setattr(landmarks, "MODEL_SHA256", sha256(content).hexdigest())
    monkeypatch.setattr(landmarks, "urlopen", lambda *a, **k: BytesIO(content))
    assert landmarks.ensure_model(path).read_bytes() == content
    assert not list(tmp_path.glob("*.download"))


def test_macos_101_guard_prevents_native_abort(monkeypatch):
    monkeypatch.setattr(landmarks.platform, "system", lambda: "Darwin")
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(__version__="1.0.1"))
    with pytest.raises(RuntimeError, match="0.10.35"):
        landmarks.FaceLandmarkerEngine("unused.task")
