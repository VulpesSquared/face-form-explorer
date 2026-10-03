# Validation — October 2, 2026

## Local environment

Apple M1, macOS arm64, Python 3.12.14, Streamlit 1.65.0, pandas 2.3.3,
NumPy 2.5.3, scikit-learn 1.9.1. Full installed versions are in
`requirements-lock-macos-arm64.txt`.

## Results

- Original supplied test: **1 passed**.
- Expanded offline suite: **22 passed**, six integration tests intentionally skipped.
- Full suite, `python -m pytest -q --run-integration`: **28 passed**.
- `python -m pip check`: **No broken requirements found**.
- MediaPipe **1.0.1**: import and API inspection succeeded, but native model
  creation aborted on macOS, with both default and explicit CPU options.
  Failure: `graph_service.h:139`, `Service is unavailable`,
  `DrishtiMetalHelper`, `TensorsToDetectionsCalculator::Open()`.
- MediaPipe **0.10.35**: official Face Landmarker model loaded, inferred and
  closed successfully on macOS. This is the platform-specific fallback.
- The Linux workflow tests **1.0.1** separately; its GitHub Actions result is the
  evidence for that platform, not the macOS fallback test.
- A clean Linux runner exposed a missing `libEGL.so.1` dependency. Native EGL,
  GLES, GL and GLib packages are now included in CI and `packages.txt` for
  Streamlit Community Cloud; CPU mode still requires these shared libraries.

## Real sample images

| Fixture | Detected landmarks | Pipeline result |
| --- | ---: | --- |
| MediaPipe `portrait.jpg` (820 × 1024) | 478 | Accepted |
| Matplotlib `grace_hopper.jpg` (512 × 600) | 478 | Accepted |
| MediaPipe `portrait_small.jpg` (205 × 256) | 478 | Rejected: face width below 120 pixels |

Additional cases: a blank image returns no face; a controlled two-face composite
is rejected; corrupt bytes and missing labels produce per-image failure reasons.
A duplicate accepted photo is aggregated under its user-provided label.
The real Streamlit upload test renders all six tabs and both export controls.

## Corrections covered by regression tests

- Replaced the unsupported pandas `Series.reset_index(names=...)` call.
- Avoided an invalid equal-min/max cluster slider for two people.
- Converted normalized landmarks to pixels before distances and pose checks;
  tests verify invariance to canvas aspect ratio.
- Kept separate labels when files share a filename.
- Cleared stale results when images or user labels change, including removal.
- Corrected single-person self-similarity to 100 on the 0–100 scale.
- Added actionable model/upload errors and a guard against the macOS 1.0.x abort.
- Validated model checksums and used separate temporary files for downloads.

Labels always come from user edits, filenames or ZIP folders. No identity
recognition, identity embeddings, unknown-person lookup, or sensitive-attribute
inference has been added. This validates runtime behavior, not measurement
accuracy across populations or camera conditions.
