"""Milestone 3a: per-frame measurements.

The unit of work is a *moment*, not a frame: when the face detector loses the
face in one frame of a burst (blur, hair, a hand), that frame is measured on
the box borrowed from its nearest sibling, so frames of one burst are always
compared on the same region.

Sharpness, in one paragraph. For a region we take the gradient magnitude after
a light blur and compute

    raw = sqrt(P99.5(|grad|)^2 - (2.8 * median(|grad|))^2) / (P98 - P2 of luminance)

* P99.5 looks only at the strongest edges (eyelashes, hair, strings), which is
  where focus shows; the mean would be dominated by smooth skin.
* The median is the noise floor: edges are sparse, noise is everywhere. For
  pure noise P99.5/median is ~2.8, so the numerator is ~0 - high ISO alone
  cannot fake sharpness (variance-of-Laplacian fails exactly here).
* Dividing by local contrast makes it ~1/edge-width: a dim singer and a
  spot-lit one with equal focus get equal scores.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import detect, raw

NOISE_K = 2.8


class Frame:
    """Decoded analysis image + its gradient field (computed once)."""

    def __init__(self, rgb: np.ndarray):
        self.rgb = rgb
        self.h, self.w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        self.lum = cv2.GaussianBlur(gray, (0, 0), 0.8)
        self.gx = cv2.Sobel(self.lum, cv2.CV_32F, 1, 0, scale=1 / 8)
        self.gy = cv2.Sobel(self.lum, cv2.CV_32F, 0, 1, scale=1 / 8)
        self.mag = cv2.magnitude(self.gx, self.gy)

    def px(self, box, min_px: int = 0) -> tuple[int, int, int, int]:
        """Normalised [x, y, w, h] -> pixel slice bounds, optionally grown to min_px."""
        x, y, w, h = box
        cx, cy = (x + w / 2) * self.w, (y + h / 2) * self.h
        bw, bh = max(w * self.w, min_px), max(h * self.h, min_px)
        x0, x1 = int(max(0, cx - bw / 2)), int(min(self.w, cx + bw / 2))
        y0, y1 = int(max(0, cy - bh / 2)), int(min(self.h, cy + bh / 2))
        return x0, y0, x1, y1


def region_sharpness(f: Frame, bounds) -> dict | None:
    x0, y0, x1, y1 = bounds
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    mag = f.mag[y0:y1, x0:x1]
    lum = f.lum[y0:y1, x0:x1]
    lit = lum > 6.0                      # crushed black carries no information
    if lit.sum() < 100:
        return {"raw": 0.0, "contrast": 0.0, "noise": 0.0, "conf": 0.0}
    floor = float(np.median(mag[lit]))
    edge = float(np.percentile(mag, 99.5))
    p2, p98 = np.percentile(lum, (2, 98))
    contrast = float(p98 - p2)
    # signal and noise gradients add in quadrature at an edge
    raw_v = math.sqrt(max(0.0, edge * edge - (NOISE_K * floor) ** 2)) / max(contrast, 12.0)
    # trust drops when there is barely any signal above the noise floor
    conf = min(1.0, contrast / 40.0) * min(1.0, edge / (4.0 * floor + 1e-6))
    return {"raw": raw_v, "contrast": contrast, "noise": floor, "conf": float(conf)}


def region_motion(f: Frame, bounds) -> dict:
    """Directional blur cue: structure-tensor coherence of the strong
    gradients. Motion blur wipes out gradients *along* the motion, so what is
    left points one way (coherence -> 1). Defocus is isotropic (-> 0).
    Scene structure is also directional, so scoring only uses this relative
    to the rest of the burst."""
    x0, y0, x1, y1 = bounds
    gx, gy, mag = f.gx[y0:y1, x0:x1], f.gy[y0:y1, x0:x1], f.mag[y0:y1, x0:x1]
    if mag.size < 400:
        return {"coherence": 0.0, "blur_angle": 0.0}
    sel = mag >= np.percentile(mag, 90)
    a, b, c = float((gx[sel] ** 2).sum()), float((gx[sel] * gy[sel]).sum()), float((gy[sel] ** 2).sum())
    root = math.sqrt((a - c) ** 2 + 4 * b * b)
    coh = root / (a + c + 1e-9)
    grad_angle = 0.5 * math.atan2(2 * b, a - c)          # dominant gradient direction
    return {"coherence": float(coh), "blur_angle": float((math.degrees(grad_angle) + 90) % 180)}


def region_exposure(f: Frame, bounds) -> dict:
    x0, y0, x1, y1 = bounds
    rgb = f.rgb[y0:y1:2, x0:x1:2]
    if rgb.size == 0:
        return {"clip": 0.0, "crush": 0.0}
    mx = rgb.max(axis=2)
    return {"clip": float((mx >= 250).mean()), "crush": float((mx <= 5).mean())}


def tile_sharpness(f: Frame, nx: int = 8, ny: int = 6) -> list[float]:
    out = []
    for j in range(ny):
        for i in range(nx):
            b = (i * f.w // nx, j * f.h // ny, (i + 1) * f.w // nx, (j + 1) * f.h // ny)
            s = region_sharpness(f, b)
            out.append(s["raw"] * (1.0 if s["conf"] > 0.3 else 0.0) if s else 0.0)
    return out


def _load(folder: Path, p: dict, edge: int) -> np.ndarray:
    try:
        im = raw.load_image(folder, p["rel_path"], p.get("companion"), p.get("preview_src"),
                            p.get("orientation"), edge)
    except Exception:
        im = Image.open(p["preview_path"]).convert("RGB")   # original gone/unreadable
    return np.asarray(im)


def _detect(rgb: np.ndarray, cfg: dict, preview_edge: int) -> dict:
    """Faces (+ eye state), else a person box."""
    H, W = rgb.shape[:2]
    to_preview = preview_edge / max(H, W) if max(H, W) > preview_edge else 1.0
    faces = []
    for fc in detect.detect_faces(rgb, cfg["face_min_score"]):
        px = max(fc["box"][2] * W, fc["box"][3] * H) * to_preview
        if px >= cfg["face_min_px"]:
            faces.append({**fc, "px": px})
    faces.sort(key=lambda d: -(d["box"][2] * d["box"][3]))
    faces = faces[:cfg["max_faces"]]
    for fc in faces:
        # eye state only for faces that are big and confidently faces
        if fc["px"] >= cfg["eye_min_face_px"] and fc["score"] >= 0.7:
            fc.update(detect.face_state(rgb, fc))
    person = None if faces else detect.detect_person(rgb)
    return {"faces": faces, "person": person}


def _person_subject(box: list[float]) -> list[float]:
    """Head-and-shoulders part of a person box: that is where focus should be."""
    x, y, w, h = box
    return [x + 0.15 * w, y, 0.7 * w, 0.4 * h]


def _measure(f: Frame, det: dict, borrowed: list[float] | None) -> dict:
    out: dict = {"faces": [], "analysis_px": [f.w, f.h]}
    tiles = tile_sharpness(f)
    out["global_raw"] = float(np.percentile(tiles, 90))
    out["tile_max"] = float(max(tiles))
    whole = (0, 0, f.w, f.h)
    out["frame_exposure"] = region_exposure(f, whole)

    regions: list[tuple[str, tuple, float]] = []      # (source, bounds, weight)
    for fc in det["faces"]:
        b = f.px(fc["box"], min_px=96)
        s = region_sharpness(f, b) or {"raw": 0.0, "contrast": 0.0, "noise": 0.0, "conf": 0.0}
        rec = {k: fc[k] for k in ("box", "score", "px", "yaw", "yaw_src") if k in fc}
        for k in ("blink", "pitch"):
            if k in fc:
                rec[k] = fc[k]
        rec["sharp_raw"] = s["raw"]
        out["faces"].append(rec)
        regions.append(("face", b, fc["box"][2] * fc["box"][3]))
    if regions:
        # the subject is the largest face, unless a face of comparable size is
        # sharper (two people on stage: either may be the one in focus)
        big = max(w for _, _, w in regions)
        cands = [i for i, (_, _, w) in enumerate(regions) if w >= 0.4 * big]
        best = max(cands, key=lambda i: out["faces"][i]["sharp_raw"])
        src, bounds, box = "face", regions[best][1], det["faces"][best]["box"]
        out["subject_face"] = best
    elif borrowed is not None:
        src, box = "borrowed", borrowed
        bounds = f.px(box, min_px=96)
    elif det["person"] is not None:
        src, box = "person", _person_subject(det["person"])
        bounds = f.px(box, min_px=96)
    else:
        src, box, bounds = "frame", [0.0, 0.0, 1.0, 1.0], whole

    out["subject"] = {"src": src, "box": [round(float(v), 5) for v in box]}
    if src == "frame":
        # no subject found: "how sharp is the sharpest part of the frame"
        s = {"raw": out["global_raw"], "contrast": 0.0, "noise": 0.0, "conf": 0.5}
        out.update(region_motion(f, whole))
        out["subject_exposure"] = out["frame_exposure"]
    else:
        s = region_sharpness(f, bounds) or {"raw": 0.0, "contrast": 0.0, "noise": 0.0, "conf": 0.0}
        out.update(region_motion(f, bounds))
        out["subject_exposure"] = region_exposure(f, bounds)
    out.update(sharp_raw=s["raw"], contrast=s["contrast"], noise=s["noise"], sharp_conf=s["conf"])
    return out


def analyse_moment(job: dict) -> list[dict]:
    """Worker entry point. job: {folder, cfg, preview_edge, photos:[...]}.
    Returns [{id, data}|{id, error}]."""
    cv2.setNumThreads(1)
    folder, cfg = Path(job["folder"]), job["cfg"]
    photos = job["photos"]
    results: dict[int, dict] = {}
    dets: dict[int, dict] = {}
    for p in photos:
        try:
            rgb = _load(folder, p, cfg["analysis_edge"])
            det = dets[p["id"]] = _detect(rgb, cfg, cfg["preview_edge"])
            results[p["id"]] = _measure(Frame(rgb), det, None)
        except Exception as e:
            results[p["id"]] = {"error": f"{type(e).__name__}: {e}"}
    # Detectors flicker inside a burst (blur, hair, a hand over the face).
    # Frames whose subject is weaker evidence than the best in the moment
    # (face > person > nothing) are re-measured on the nearest good sibling's
    # box, so one burst is always compared on the same region. Only these
    # frames are decoded twice.
    rank = {"face": 3, "person": 2, "frame": 1}
    ok = [p for p in photos if "subject" in results.get(p["id"], {})]
    top = max((rank[results[p["id"]]["subject"]["src"]] for p in ok), default=1)
    donors = [p for p in ok if rank[results[p["id"]]["subject"]["src"]] == top]
    for p in ok if top > 1 else []:
        if rank[results[p["id"]]["subject"]["src"]] == top:
            continue
        try:
            near = min(donors, key=lambda q: abs((q["capture_ts"] or 0) - (p["capture_ts"] or 0)))
            if abs((near["capture_ts"] or 0) - (p["capture_ts"] or 0)) > 2.0:
                continue
            donor = results[near["id"]]["subject"]
            x, y, w, h = donor["box"]
            borrowed = [max(0.0, x - 0.05 * w), max(0.0, y - 0.05 * h), 1.1 * w, 1.1 * h]
            rgb = _load(folder, p, cfg["analysis_edge"])
            results[p["id"]] = _measure(Frame(rgb), {**dets[p["id"]], "faces": []}, borrowed)
            results[p["id"]]["subject"]["from"] = donor["src"]
        except Exception as e:
            results[p["id"]] = {"error": f"{type(e).__name__}: {e}"}
    return [{"id": pid, **({"error": r["error"]} if "error" in r else {"data": r})}
            for pid, r in results.items()]
