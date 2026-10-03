import numpy as np
import pytest

from face_analysis.geometry import calculate_features, quality_check, pixel_landmarks


def test_calculate_features_returns_finite_core_values():
    pts = np.zeros((478, 3), dtype=float)
    # Generic nondegenerate scaffold; exact anatomy is irrelevant for unit coverage.
    for i in range(478):
        pts[i, 0] = 0.2 + (i % 17) * 0.03
        pts[i, 1] = 0.1 + (i % 19) * 0.03
    # Important face anchors must be deliberately non-collinear/nonzero.
    pts[234, :2] = [0.2, 0.5]
    pts[454, :2] = [0.8, 0.5]
    pts[10, :2] = [0.5, 0.1]
    pts[152, :2] = [0.5, 0.9]
    features = calculate_features(pts)
    assert "face_width_height" in features
    assert np.isfinite(features["face_width_height"])


def test_features_and_pose_do_not_change_with_canvas_aspect_ratio():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(200, 400, (478, 3))
    pixels[234, :2], pixels[454, :2] = [200, 300], [400, 300]
    pixels[10, :2], pixels[152, :2] = [300, 100], [300, 500]
    pixels[33, :2], pixels[133, :2] = [230, 240], [270, 240]
    pixels[362, :2], pixels[263, :2] = [330, 250], [370, 250]
    pixels[1, :2] = [300, 300]
    measurements = []
    quality = []
    for width, height in [(600, 600), (800, 1200), (1600, 700)]:
        normalized = pixels / [width, height, width]
        measurements.append(calculate_features(normalized, width, height))
        quality.append(quality_check(normalized, width, height))
    for measured in measurements:
        assert measured["face_width_height"] == pytest.approx(0.5)
        assert measured == pytest.approx(measurements[0])
    for qc in quality:
        assert qc.passed
        assert qc.roll_deg == pytest.approx(5.710593, abs=1e-5)
        assert qc.face_width_px == pytest.approx(200)


@pytest.mark.parametrize("points", [np.zeros((10, 3)), np.full((478, 3), np.nan)])
def test_invalid_landmarks_are_rejected(points):
    with pytest.raises(ValueError):
        pixel_landmarks(points, 640, 480)
