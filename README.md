# Face Form Explorer

A browser-based, interpretable facial-morphology explorer built with **Streamlit** and **Google MediaPipe Face Landmarker**.

The app accepts user-labeled photographs, extracts facial landmarks, calculates normalized geometric ratios, aggregates multiple photos per person, and provides:

- pose and image-quality screening
- per-person median morphology features
- configurable similarity matrices
- pairwise feature-family comparisons
- feature exploration within the current dataset
- hierarchical clustering + PCA visualization
- Excel/CSV export

## Important scope

This is **not face identification**. The user supplies the person's label. The app does not attempt to recognize an unknown person from an image.

It also does not infer ethnicity/ancestry, personality, health, intelligence, attractiveness, or other sensitive/non-observable traits. Similarity scores are descriptive indices within the current uploaded dataset, not identity or kinship probabilities.

## Fastest deployment: Streamlit Community Cloud

1. Create a GitHub repository and put these files in it.
2. Go to Streamlit Community Cloud and sign in with GitHub.
3. Create a new app from the repository.
4. Set the entrypoint to `app.py`.
5. Use Python **3.12** in Advanced settings.
6. Deploy.

The first analysis downloads Google's official `face_landmarker.task` model into `models/`. If your deployment environment cannot access Google Cloud Storage, manually download the model from:

`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`

and save it locally as (downloaded models are excluded from Git):

`models/face_landmarker.task`

## Upload formats

### Option A — ZIP folders (recommended)

```text
faces.zip
├── Rose Byrne/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
├── Emmy Rossum/
│   ├── 01.jpg
│   └── 02.jpg
└── Jenna Coleman/
    ├── 01.jpg
    └── 02.jpg
```

Folder name becomes the initial person label. You can edit labels in the app before analysis.

### Option B — loose files

Use a double underscore between the person's name and the rest of the filename:

```text
Rose Byrne__01.jpg
Rose Byrne__02.jpg
Emmy Rossum__01.jpg
```

Again, labels remain editable before analysis.

## Photo guidance

Best results come from 3–6 photos per person with:

- one face in the image
- near-frontal head pose
- neutral or mild expression
- good lighting
- face at least ~120 pixels wide
- minimal lens distortion / extreme selfie perspective

The app rejects large head roll, strong yaw, very small faces, multiple faces, or undetected faces.

## Run locally

Use Python **3.12**. On macOS, use an Apple Silicon Python runtime; the selected
MediaPipe wheels do not support Intel Macs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # macOS/Linux
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Open http://localhost:8501. On Windows, activate with `.venv/Scripts/activate`.
The model location is anchored to `app.py`, so starting from another working
directory does not create a second model cache.

### MediaPipe compatibility

`requirements.txt` selects **MediaPipe 0.10.35 on macOS** and **1.0.1 elsewhere**.
On the tested Apple M1 Mac, 1.0.1 imports successfully but constructing Face
Landmarker aborts the native process with `DrishtiMetalHelper` /
`Service is unavailable`, including when CPU is explicitly selected. A Python
exception handler cannot catch that abort. The app detects the known affected
macOS versions before creating the native model and gives installation guidance.

The 0.10.35 macOS fallback uses the same Tasks Face Landmarker API and official
478-landmark model. The Linux GitHub Actions workflow exercises 1.0.1 with real
images. See [VALIDATION.md](VALIDATION.md) for recorded local results.

`requirements-lock-macos-arm64.txt` records the exact environment tested locally;
use it only for reproducing that macOS environment. The regular requirements
file selects platform-compatible packages on other systems.

### Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
# Optional real-model tests (downloads public test images and Google's model):
python -m scripts.download_test_assets
python -m pytest -q --run-integration
```

Ordinary tests are offline. Integration tests cover real landmark loading,
three portrait images, no-face and multiple-face cases, quality rejection,
explicit user labels, aggregation, and the full Streamlit upload/results/export
flow. Model and image downloads are checksum-verified and excluded from Git.

## How similarity works

1. Normalized landmarks are converted to pixel coordinates using both image dimensions before calculating ratios relative to face width or height. This prevents portrait/landscape canvas shape from distorting measurements.
2. Multiple accepted images per person are aggregated by the median.
3. Features are robust-scaled across the current dataset.
4. Pairwise root-mean-square standardized distance is transformed into a 0–100 **relative morphology index**.

The value is deliberately presented as an index rather than a probability. Changing the people in the dataset can change the scaling and therefore the scores.

## Feature families

The current build includes interpretable ratios for:

- overall facial proportions
- eyes
- brows
- nose
- mouth / philtrum
- cheeks / jaw
- chin / lower face

The feature list lives in `face_analysis/geometry.py`, so it is straightforward to add or remove measurements later.

## Privacy

MediaPipe Tasks performs the landmark model computation in the environment running the app. The application code does not intentionally persist uploaded photos. However, when deployed to Streamlit Community Cloud, browser uploads are necessarily transmitted to that hosted server for processing. For sensitive/private photos, run the same repository locally.

## Project structure

```text
app.py
face_analysis/
  landmarks.py      # MediaPipe model + detections
  geometry.py       # landmark ratios + QC
  pipeline.py       # per-image → per-person aggregation
  similarity.py     # scaling, matching, explanations
  clustering.py     # exploratory grouping + PCA
  exporting.py      # Excel output
  uploads.py        # ZIP / upload parsing
models/
.streamlit/
requirements.txt
```

## Measurement limits

These are 2D image measurements. Pose, expression, camera perspective and landmark
error can still affect comparisons after quality screening. Eye-line tilt is a
pose-related feature; turn off the Eyes family if it is not useful for your
comparison. The measurements and clusters are exploratory, not validated
biometric or clinical assessments.
