from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def results(n):
    people = pd.DataFrame({
        "person": [f"Sample {i}" for i in range(n)],
        "accepted_images": [1] * n,
        "face_width_height": [0.7 + i * 0.02 for i in range(n)],
        "nose_width_face": [0.2 + i * 0.01 for i in range(n)],
    })
    return people.copy(), people, pd.DataFrame(columns=["person", "filename", "reason"])


def uploaded_app(n=2):
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.file_uploader[0].set_value([
        (f"Sample {i}__photo.jpg", b"image bytes handled by mock", "image/jpeg")
        for i in range(n)
    ]).run()
    return at


def test_empty_app_starts():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Face Form Explorer"


@pytest.mark.parametrize("count", [1, 2, 3])
def test_analysis_renders_every_tab_and_export(count):
    at = uploaded_app(count)
    with patch("face_analysis.pipeline.analyze_dataset", return_value=results(count)):
        at.button[0].click().run()
    assert not at.exception
    assert len(at.tabs) == 6
    assert len(at.get("download_button")) == 2
    if count == 2:
        assert not at.slider
    if count == 3:
        assert at.slider[0].value == 3


@pytest.mark.parametrize("change", ["clear", "replace", "relabel"])
def test_changed_input_clears_old_results(change):
    at = uploaded_app()
    with patch("face_analysis.pipeline.analyze_dataset", return_value=results(2)):
        at.button[0].click().run()
    assert len(at.tabs) == 6
    if change == "clear":
        at.file_uploader[0].clear().run()
    elif change == "replace":
        at.file_uploader[0].set_value(("Sample 0__photo.jpg", b"new image", "image/jpeg")).run()
    else:
        with patch("streamlit.data_editor", return_value=pd.DataFrame({"person": ["Edited", "Sample 1"]})):
            at.run()
    assert not at.exception
    assert not at.tabs
    assert "analysis" not in at.session_state


def test_duplicate_filenames_preserve_separate_user_labels():
    at = uploaded_app()
    at.file_uploader[0].set_value([
        ("same.jpg", b"first", "image/jpeg"), ("same.jpg", b"second", "image/jpeg"),
    ]).run()
    with patch("streamlit.data_editor", return_value=pd.DataFrame({"person": ["First", "Second"]})), patch(
        "face_analysis.pipeline.analyze_dataset", return_value=results(2),
    ) as analyze:
        at.button[0].click().run()
    assert not at.exception
    assert [r["person"] for r in analyze.call_args.args[0]] == ["First", "Second"]


def test_model_failure_is_actionable_without_traceback():
    at = uploaded_app()
    with patch("face_analysis.pipeline.analyze_dataset", side_effect=RuntimeError("Model unavailable")):
        at.button[0].click().run()
    assert not at.exception
    assert "Model unavailable" in at.error[0].value


def test_bad_zip_shows_upload_error():
    at = AppTest.from_file(str(APP)).run()
    at.file_uploader[0].set_value(("broken.zip", b"not a zip", "application/zip")).run()
    assert not at.exception
    assert "Could not read ZIP" in at.error[0].value
