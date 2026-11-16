"""Detector model files: downloaded once into the cache, checksum-verified."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from .config import cache_root

MODELS = {
    # OpenCV YuNet face detector (works down to small faces, 230 kB)
    "yunet": {
        "file": "face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "urls": [
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
    },
    # MediaPipe face landmarker (eye blink blendshapes + head pose)
    "landmarker": {
        "file": "face_landmarker.task",
        "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        "urls": ["https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"],
    },
    # MediaPipe EfficientDet-Lite0 (person boxes when no face is visible)
    "persons": {
        "file": "efficientdet_lite0.tflite",
        "sha256": "4b59100025bea1235a84c1038879a6cccc9f6c49f5e41144e91e74d99e780993",
        "urls": ["https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"],
    },
}


def models_dir() -> Path:
    d = cache_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_path(name: str) -> Path | None:
    """Path of an already-downloaded, verified model, else None."""
    p = models_dir() / MODELS[name]["file"]
    return p if p.exists() else None


def ensure(name: str, log=print) -> Path | None:
    """Download a model if needed. Returns None (and logs why) on failure:
    the pipeline degrades instead of crashing when offline."""
    spec = MODELS[name]
    dest = models_dir() / spec["file"]
    if dest.exists():
        return dest
    for url in spec["urls"]:
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            log(f"downloading {spec['file']} ...")
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            if _sha256(tmp) != spec["sha256"]:
                raise ValueError("checksum mismatch")
            os.replace(tmp, dest)
            return dest
        except Exception as e:
            log(f"  could not fetch {url}: {e}")
            tmp.unlink(missing_ok=True)
    return None


def ensure_all(log=print) -> dict[str, bool]:
    return {name: ensure(name, log) is not None for name in MODELS}
