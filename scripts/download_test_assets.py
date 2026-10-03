"""Download public test fixtures explicitly; ordinary unit tests stay offline."""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

from face_analysis.landmarks import ensure_model

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "portrait.jpg": (
        "https://storage.googleapis.com/mediapipe-assets/portrait.jpg",
        "a6f11efaa834706db23f275b6115058fa87fc7f14362681e6abe14e82749de3e",
    ),
    "portrait_small.jpg": (
        "https://storage.googleapis.com/mediapipe-assets/portrait_small.jpg",
        "873a1a5e4cc86c040101362c5dea6a71cf524563b0700640175e5c3763a4246a",
    ),
    "grace_hopper.jpg": (
        "https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.7/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg",
        "a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130",
    ),
}


def main():
    folder = ROOT / "tests" / "data"
    folder.mkdir(exist_ok=True)
    for name, (url, checksum) in ASSETS.items():
        path = folder / name
        if not path.exists() or sha256(path.read_bytes()).hexdigest() != checksum:
            with urlopen(url, timeout=60) as response:
                data = response.read()
            if sha256(data).hexdigest() != checksum:
                raise RuntimeError(f"Checksum mismatch for {name}")
            path.write_bytes(data)
        print(f"Ready: {name}")
    print(f"Ready: {ensure_model(ROOT / 'models' / 'face_landmarker.task')}")


if __name__ == "__main__":
    main()
