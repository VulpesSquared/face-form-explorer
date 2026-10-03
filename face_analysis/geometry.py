from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

# MediaPipe Face Landmarker landmark indices. The analysis intentionally uses
# stable geometric anchors rather than expression/blendshape coefficients.
IDX = {
    "forehead_top": 10,
    "chin": 152,
    "face_left": 234,
    "face_right": 454,
    "right_eye_outer": 33,
    "right_eye_inner": 133,
    "left_eye_inner": 362,
    "left_eye_outer": 263,
    "right_eye_upper": 159,
    "right_eye_lower": 145,
    "left_eye_upper": 386,
    "left_eye_lower": 374,
    "right_brow_outer": 70,
    "right_brow_inner": 107,
    "right_brow_peak": 105,
    "left_brow_inner": 336,
    "left_brow_outer": 300,
    "left_brow_peak": 334,
    "nose_bridge": 168,
    "nose_tip": 1,
    "nose_bottom": 2,
    "nose_left": 98,
    "nose_right": 327,
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip": 13,
    "lower_lip": 14,
    "upper_lip_top": 0,
    "right_cheek": 205,
    "left_cheek": 425,
    "right_jaw": 172,
    "left_jaw": 397,
    "right_temple": 127,
    "left_temple": 356,
}

FEATURE_GROUPS = {
    "Overall proportions": [
        "face_width_height",
        "upper_face_share",
        "midface_share",
        "lower_face_share",
        "temple_to_face_width",
        "cheek_to_face_width",
    ],
    "Eyes": [
        "eye_spacing_face",
        "right_eye_width_face",
        "left_eye_width_face",
        "mean_eye_aspect",
        "eye_line_tilt_deg",
    ],
    "Brows": [
        "mean_brow_eye_gap",
        "brow_span_face",
        "brow_peak_height",
    ],
    "Nose": [
        "nose_width_face",
        "nose_length_height",
        "nose_width_length",
        "nose_to_mouth_width",
    ],
    "Mouth & philtrum": [
        "mouth_width_face",
        "lip_opening_height",
        "philtrum_share",
        "mouth_chin_share",
    ],
    "Cheeks & jaw": [
        "cheek_to_face_width",
        "jaw_to_face_width",
        "jaw_to_cheek_width",
        "lower_face_taper",
    ],
    "Chin": [
        "chin_length_height",
        "mouth_chin_share",
        "lower_face_taper",
    ],
}

FEATURE_LABELS = {
    "face_width_height": "Face width : height",
    "upper_face_share": "Upper-face proportion",
    "midface_share": "Midface proportion",
    "lower_face_share": "Lower-face proportion",
    "temple_to_face_width": "Temple width",
    "cheek_to_face_width": "Cheek width",
    "eye_spacing_face": "Eye spacing",
    "right_eye_width_face": "Right eye width",
    "left_eye_width_face": "Left eye width",
    "mean_eye_aspect": "Eye openness/aspect",
    "eye_line_tilt_deg": "Eye-line tilt",
    "mean_brow_eye_gap": "Brow-to-eye spacing",
    "brow_span_face": "Brow span",
    "brow_peak_height": "Brow arch height",
    "nose_width_face": "Nose width",
    "nose_length_height": "Nose length",
    "nose_width_length": "Nose width : length",
    "nose_to_mouth_width": "Nose : mouth width",
    "mouth_width_face": "Mouth width",
    "lip_opening_height": "Lip opening/height",
    "philtrum_share": "Philtrum proportion",
    "mouth_chin_share": "Mouth-to-chin proportion",
    "jaw_to_face_width": "Jaw width",
    "jaw_to_cheek_width": "Jaw : cheek width",
    "lower_face_taper": "Lower-face taper",
    "chin_length_height": "Chin length",
}


@dataclass
class QualityResult:
    passed: bool
    roll_deg: float
    yaw_proxy: float
    face_width_px: float
    reasons: list[str]


def _p(landmarks: np.ndarray, name: str) -> np.ndarray:
    return landmarks[IDX[name], :2]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-9 else np.nan


def _vertical(a: np.ndarray, b: np.ndarray) -> float:
    return abs(float(a[1] - b[1]))


def pixel_landmarks(landmarks: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Put MediaPipe's separately normalized x/y values in the same unit."""
    points = np.asarray(landmarks, dtype=float)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if points.ndim != 2 or points.shape[0] <= max(IDX.values()) or points.shape[1] < 2:
        raise ValueError("Incomplete face landmarks")
    if not np.isfinite(points).all():
        raise ValueError("Non-finite face landmarks")
    points = points.copy()
    points[:, 0] *= image_width
    points[:, 1] *= image_height
    return points


def quality_check(landmarks: np.ndarray, image_width: int, image_height: int) -> QualityResult:
    landmarks = pixel_landmarks(landmarks, image_width, image_height)
    left = _p(landmarks, "face_left")
    right = _p(landmarks, "face_right")
    width = _dist(left, right)

    eye_r = (_p(landmarks, "right_eye_outer") + _p(landmarks, "right_eye_inner")) / 2
    eye_l = (_p(landmarks, "left_eye_outer") + _p(landmarks, "left_eye_inner")) / 2
    dx, dy = eye_l - eye_r
    roll = math.degrees(math.atan2(float(dy), float(dx)))

    nose = _p(landmarks, "nose_tip")
    center_x = (left[0] + right[0]) / 2
    yaw_proxy = abs(float(nose[0] - center_x)) / max(width, 1e-9)

    face_width_px = width
    reasons: list[str] = []
    if abs(roll) > 12:
        reasons.append(f"head roll {roll:.1f}°")
    if yaw_proxy > 0.12:
        reasons.append("face is turned too far from frontal")
    if face_width_px < 120:
        reasons.append("face is too small in the image")

    return QualityResult(not reasons, roll, yaw_proxy, face_width_px, reasons)


def calculate_features(
    landmarks: np.ndarray, image_width: int = 1, image_height: int = 1,
) -> Dict[str, float]:
    """Measure ratios in pixels; pass actual dimensions for normalized landmarks."""
    landmarks = pixel_landmarks(landmarks, image_width, image_height)
    P = lambda name: _p(landmarks, name)

    face_width = _dist(P("face_left"), P("face_right"))
    face_height = _dist(P("forehead_top"), P("chin"))
    eye_r_center = (P("right_eye_outer") + P("right_eye_inner")) / 2
    eye_l_center = (P("left_eye_outer") + P("left_eye_inner")) / 2
    eye_mid_y = float((eye_r_center[1] + eye_l_center[1]) / 2)
    forehead_y = float(P("forehead_top")[1])
    nose_y = float(P("nose_bottom")[1])
    chin_y = float(P("chin")[1])

    right_eye_width = _dist(P("right_eye_outer"), P("right_eye_inner"))
    left_eye_width = _dist(P("left_eye_outer"), P("left_eye_inner"))
    right_eye_height = _dist(P("right_eye_upper"), P("right_eye_lower"))
    left_eye_height = _dist(P("left_eye_upper"), P("left_eye_lower"))

    eye_line_dx = float(eye_l_center[0] - eye_r_center[0])
    eye_line_dy = float(eye_l_center[1] - eye_r_center[1])
    eye_line_tilt = math.degrees(math.atan2(eye_line_dy, eye_line_dx))

    nose_width = _dist(P("nose_left"), P("nose_right"))
    nose_length = _dist(P("nose_bridge"), P("nose_bottom"))
    mouth_width = _dist(P("mouth_left"), P("mouth_right"))

    cheek_width = _dist(P("right_cheek"), P("left_cheek"))
    jaw_width = _dist(P("right_jaw"), P("left_jaw"))
    temple_width = _dist(P("right_temple"), P("left_temple"))

    right_brow_center = (P("right_brow_outer") + P("right_brow_inner")) / 2
    left_brow_center = (P("left_brow_outer") + P("left_brow_inner")) / 2
    right_brow_gap = _vertical(right_brow_center, eye_r_center)
    left_brow_gap = _vertical(left_brow_center, eye_l_center)
    brow_span = _dist(P("right_brow_outer"), P("left_brow_outer"))
    brow_peak_height = (
        _vertical(P("right_brow_peak"), right_brow_center)
        + _vertical(P("left_brow_peak"), left_brow_center)
    ) / 2

    upper_face = abs(eye_mid_y - forehead_y)
    midface = abs(nose_y - eye_mid_y)
    lower_face = abs(chin_y - nose_y)
    philtrum = _vertical(P("nose_bottom"), P("upper_lip_top"))
    mouth_chin = _vertical(P("lower_lip"), P("chin"))
    chin_length = _vertical(P("lower_lip"), P("chin"))

    return {
        "face_width_height": _safe_div(face_width, face_height),
        "upper_face_share": _safe_div(upper_face, face_height),
        "midface_share": _safe_div(midface, face_height),
        "lower_face_share": _safe_div(lower_face, face_height),
        "temple_to_face_width": _safe_div(temple_width, face_width),
        "cheek_to_face_width": _safe_div(cheek_width, face_width),
        "eye_spacing_face": _safe_div(_dist(P("right_eye_inner"), P("left_eye_inner")), face_width),
        "right_eye_width_face": _safe_div(right_eye_width, face_width),
        "left_eye_width_face": _safe_div(left_eye_width, face_width),
        "mean_eye_aspect": np.nanmean([
            _safe_div(right_eye_height, right_eye_width),
            _safe_div(left_eye_height, left_eye_width),
        ]),
        "eye_line_tilt_deg": eye_line_tilt,
        "mean_brow_eye_gap": _safe_div((right_brow_gap + left_brow_gap) / 2, face_height),
        "brow_span_face": _safe_div(brow_span, face_width),
        "brow_peak_height": _safe_div(brow_peak_height, face_height),
        "nose_width_face": _safe_div(nose_width, face_width),
        "nose_length_height": _safe_div(nose_length, face_height),
        "nose_width_length": _safe_div(nose_width, nose_length),
        "nose_to_mouth_width": _safe_div(nose_width, mouth_width),
        "mouth_width_face": _safe_div(mouth_width, face_width),
        "lip_opening_height": _safe_div(_dist(P("upper_lip"), P("lower_lip")), face_height),
        "philtrum_share": _safe_div(philtrum, face_height),
        "mouth_chin_share": _safe_div(mouth_chin, face_height),
        "jaw_to_face_width": _safe_div(jaw_width, face_width),
        "jaw_to_cheek_width": _safe_div(jaw_width, cheek_width),
        "lower_face_taper": 1.0 - _safe_div(jaw_width, cheek_width),
        "chin_length_height": _safe_div(chin_length, face_height),
    }
