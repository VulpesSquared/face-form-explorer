# Optional public test images

Run `python -m scripts.download_test_assets` from the repository root.
Images and the model are downloaded locally and excluded from Git.

- `portrait.jpg` and `portrait_small.jpg`: Google's public [MediaPipe test assets](https://storage.googleapis.com/mediapipe-assets/portrait.jpg), also used in [upstream face-landmarker tests](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/test/vision/face_landmarker_test.py).
- `grace_hopper.jpg`: [Matplotlib's sample data](https://github.com/matplotlib/matplotlib/blob/v3.10.7/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg).

The fixtures are assigned explicit test labels. Their filenames are not identity predictions.
The small portrait should be detected but rejected for insufficient face width.
