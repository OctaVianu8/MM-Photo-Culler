"""Subject detection: faces (OpenCV YuNet), eye state + head pose (MediaPipe
face landmarker on each face crop), person boxes (MediaPipe EfficientDet).

Every detector is optional. Missing model / missing mediapipe / headless box
without libEGL -> that capability is reported as off and scoring falls back.
"""
from __future__ import annotations

import contextlib
import math
import os

import cv2
import numpy as np

from . import models

_yunet: dict[tuple[int, int], object] = {}
_landmarker = None
_persons = None
_tried = {"landmarker": False, "persons": False}
_errors: dict[str, str] = {}


@contextlib.contextmanager
def _quiet_stderr():
    """MediaPipe logs C++ warnings straight to fd 2 while building a graph."""
    try:
        saved = os.dup(2)
        null = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(null)


def capabilities() -> dict:
    """What this process can actually run (initialises the detectors)."""
    return {"faces": models.model_path("yunet") is not None,
            "eyes": _get_landmarker() is not None,
            "persons": _get_persons() is not None,
            "errors": dict(_errors)}


def _get_landmarker():
    global _landmarker
    if _tried["landmarker"]:
        return _landmarker
    _tried["landmarker"] = True
    path = models.model_path("landmarker")
    if path is None:
        _errors["eyes"] = "face_landmarker model not downloaded"
        return None
    try:
        with _quiet_stderr():
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            _landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(path)), num_faces=1,
                output_face_blendshapes=True, output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))
            _landmarker.detect(_mp_image(np.zeros((64, 64, 3), np.uint8)))   # warm-up logs
    except Exception as e:
        _errors["eyes"] = f"{type(e).__name__}: {e}"
    return _landmarker


def _get_persons():
    global _persons
    if _tried["persons"]:
        return _persons
    _tried["persons"] = True
    path = models.model_path("persons")
    if path is None:
        _errors["persons"] = "person detector model not downloaded"
        return None
    try:
        with _quiet_stderr():
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            _persons = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(path)), score_threshold=0.35,
                category_allowlist=["person"], max_results=5))
            _persons.detect(_mp_image(np.zeros((64, 64, 3), np.uint8)))
    except Exception as e:
        _errors["persons"] = f"{type(e).__name__}: {e}"
    return _persons


def close() -> None:
    """Release MediaPipe graphs explicitly (their __del__ is noisy at exit)."""
    global _landmarker, _persons
    for obj in (_landmarker, _persons):
        try:
            if obj is not None:
                obj.close()
        except Exception:
            pass
    _landmarker = _persons = None
    _tried.update(landmarker=False, persons=False)


def _mp_image(rgb: np.ndarray):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def detect_faces(rgb: np.ndarray, min_score: float) -> list[dict]:
    """Faces as dicts with normalised box [x, y, w, h], score, and the five
    YuNet landmarks (normalised). Runs at two scales: YuNet at one scale
    misses either the frame-filling face or the small one at the back."""
    path = models.model_path("yunet")
    if path is None:
        return []
    H, W = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    found: list[dict] = []
    for edge in (1280, 480):
        sc = min(1.0, edge / max(H, W))
        img = bgr if sc == 1.0 else cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        det = _yunet.get((w, h))
        if det is None:
            det = _yunet[(w, h)] = cv2.FaceDetectorYN.create(str(path), "", (w, h), min_score, 0.3, 200)
        _, faces = det.detect(img)
        for f in (faces if faces is not None else []):
            x, y, fw, fh = (float(v) for v in f[:4])
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(w), x + fw), min(float(h), y + fh)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            pts = [(float(f[4 + 2 * i]) / w, float(f[5 + 2 * i]) / h) for i in range(5)]
            found.append({"box": [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h],
                          "score": float(f[14]), "pts": pts})
    found.sort(key=lambda d: -d["score"])
    kept: list[dict] = []
    for f in found:
        if all(_iou(f["box"], k["box"]) < 0.35 for k in kept):
            kept.append(f)
    for f in kept:
        (rx, ry), (lx, ly), (nx, ny) = f["pts"][:3]
        eye_d = math.hypot((lx - rx) * W, (ly - ry) * H) + 1e-6
        # crude yaw from where the nose sits between the eyes (used when the
        # landmark model is unavailable or fails on this face)
        off = ((nx - (lx + rx) / 2) * W) / eye_d
        f["yaw"] = float(max(-90.0, min(90.0, off * 100.0)))
        f["yaw_src"] = "5pt"
    return kept


def face_state(rgb: np.ndarray, face: dict) -> dict:
    """Eye-closure probability + head pose for one face, or {} if unknown.

    The landmark model happily hallucinates a face onto a microphone or a
    hand, so its answer is only trusted when its eyes land where YuNet's
    independent eye points are."""
    lm = _get_landmarker()
    if lm is None:
        return {}
    H, W = rgb.shape[:2]
    x, y, w, h = face["box"]
    cx, cy = (x + w / 2) * W, (y + h / 2) * H
    side = max(w * W, h * H)
    for pad in (1.5, 1.0):
        half = side * pad
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
        crop = rgb[y0:y1, x0:x1]
        if min(crop.shape[:2]) < 24:
            return {}
        sc = min(1.0, 384 / max(crop.shape[:2]))
        if sc < 1.0:
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        try:
            res = lm.detect(_mp_image(crop))
        except Exception:
            return {}
        if not res.face_landmarks:
            continue
        pts = res.face_landmarks[0]
        ch, cw = crop.shape[:2]

        def to_img(i):
            return (x0 + pts[i].x * cw / sc, y0 + pts[i].y * ch / sc)
        # eye centres: mean of the eye-corner landmarks
        e1 = np.mean([to_img(33), to_img(133)], axis=0)
        e2 = np.mean([to_img(362), to_img(263)], axis=0)
        (rx, ry), (lx, ly) = face["pts"][:2]
        y1_, y2_ = np.array([rx * W, ry * H]), np.array([lx * W, ly * H])
        eye_d = np.linalg.norm(y1_ - y2_) + 1e-6
        err = min(np.linalg.norm(e1 - y1_) + np.linalg.norm(e2 - y2_),
                  np.linalg.norm(e1 - y2_) + np.linalg.norm(e2 - y1_)) / (2 * eye_d)
        if err > 0.35:
            continue
        out: dict = {}
        if res.face_blendshapes:
            bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
            if "eyeBlinkLeft" in bs and "eyeBlinkRight" in bs:
                # min(): closed only if *both* eyes say so. A wink, or the
                # hidden far eye of a 3/4 profile, must not count as closed.
                out["blink"] = float(min(bs["eyeBlinkLeft"], bs["eyeBlinkRight"]))
        if res.facial_transformation_matrixes:
            R = np.array(res.facial_transformation_matrixes[0])[:3, :3]
            out["yaw"] = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
            out["pitch"] = float(np.degrees(np.arcsin(np.clip(-R[1, 2], -1, 1))))
            out["yaw_src"] = "mesh"
        return out
    return {}


def detect_person(rgb: np.ndarray) -> list[float] | None:
    """Normalised box of the most prominent person, or None."""
    det = _get_persons()
    if det is None:
        return None
    H, W = rgb.shape[:2]
    sc = min(1.0, 960 / max(H, W))
    img = rgb if sc == 1.0 else cv2.resize(rgb, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    try:
        res = det.detect(_mp_image(img))
    except Exception:
        return None
    h, w = img.shape[:2]
    best, best_v = None, 0.0
    for d in res.detections:
        b = d.bounding_box
        box = [max(0, b.origin_x) / w, max(0, b.origin_y) / h,
               min(b.width, w - b.origin_x) / w, min(b.height, h - b.origin_y) / h]
        v = d.categories[0].score * box[2] * box[3]
        if v > best_v:
            best, best_v = box, v
    return best
