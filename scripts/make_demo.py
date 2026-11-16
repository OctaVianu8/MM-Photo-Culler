#!/usr/bin/env python3
"""Build a synthetic "event" folder from a handful of ordinary photos, for
trying the app and for regression-testing the scorer without real RAWs.

Each source photo becomes a burst: slightly re-framed copies with a known
degradation (defocus, motion blur) at a shared high-ISO noise level and
burst-like EXIF timestamps. ``truth.json`` records which frames are sharp.

    python scripts/make_demo.py SOURCE_DIR OUT_DIR [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    for t in np.linspace(-c, c, length * 4):
        k[int(round(c + t * dy)), int(round(c + t * dx))] = 1
    return k / k.sum()


def save(img: np.ndarray, path: Path, when: datetime, iso: int) -> None:
    im = Image.fromarray(img)
    exif = Image.Exif()
    exif[0x0110] = "DemoCam"
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")
    sub[0x9291] = f"{when.microsecond // 10000:02d}"
    sub[0x8827] = iso
    sub[0x829A] = (1, 200)
    sub[0x920A] = (85, 1)
    im.save(path, "JPEG", quality=92, exif=exif)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-burst", type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    when = datetime(2026, 10, 3, 21, 14, 5)
    truth = {}
    n = 0
    for src in sorted(Path(a.src).glob("*.jpg")):
        base = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)
        h, w = base.shape[:2]
        when += timedelta(seconds=float(rng.uniform(25, 120)))
        iso = int(rng.choice([1600, 3200, 6400, 12800]))
        sigma = {1600: 3, 3200: 5, 6400: 8, 12800: 12}[iso]
        size = int(rng.integers(1, a.max_burst + 1))
        kinds = ["sharp"] + [str(rng.choice(["sharp", "soft", "defocus", "motion", "motion"]))
                             for _ in range(size - 1)]
        rng.shuffle(kinds)
        for kind in kinds:
            n += 1
            # hand-held jitter: small shift + zoom
            z = 1 + rng.uniform(0, 0.02)
            M = np.float32([[z, 0, rng.uniform(-12, 12)], [0, z, rng.uniform(-12, 12)]])
            img = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            if kind == "soft":
                img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(1.0, 1.6)))
            elif kind == "defocus":
                img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(2.5, 5)))
            elif kind == "motion":
                img = cv2.filter2D(img, -1, motion_kernel(int(rng.integers(9, 26)), float(rng.uniform(0, 180))))
            noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
            name = f"DEMO{n:04d}.JPG"
            save(np.clip(noisy, 0, 255).astype(np.uint8), out / name, when, iso)
            truth[name] = {"source": src.name, "kind": kind}
            when += timedelta(seconds=float(rng.uniform(0.08, 0.2)))
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {n} frames from {len(set(t['source'] for t in truth.values()))} sources to {out}")


if __name__ == "__main__":
    main()
