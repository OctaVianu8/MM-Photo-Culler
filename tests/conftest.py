import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image


def scene(seed: int, size=(900, 600)) -> np.ndarray:
    """A textured synthetic 'photo': random blobs, lines and text-like detail."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.full((h, w, 3), rng.integers(20, 60), np.uint8)
    for _ in range(40):
        c = tuple(int(v) for v in rng.integers(40, 255, 3))
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if rng.random() < 0.5:
            cv2.circle(img, p1, int(rng.integers(8, 70)), c, -1, cv2.LINE_AA)
        else:
            p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(img, p1, p2, c, int(rng.integers(1, 5)), cv2.LINE_AA)
    return img


def save_jpeg(arr: np.ndarray, path: Path, when: datetime, iso=3200) -> None:
    exif = Image.Exif()
    exif[0x0110] = "TestCam"
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")
    sub[0x9291] = f"{when.microsecond // 10000:02d}"
    sub[0x8827] = iso
    Image.fromarray(arr).save(path, "JPEG", quality=92, exif=exif)


def tree_hash(folder: Path, suffixes=(".jpg", ".JPG")) -> dict:
    return {str(p.relative_to(folder)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*")) if p.suffix in suffixes}


@pytest.fixture()
def event(tmp_path, monkeypatch):
    """3 bursts: (sharp, blurred, blurred), (blurred, sharp), single. 30 s apart."""
    monkeypatch.setenv("MMCULL_CACHE", str(tmp_path / "cache"))
    folder = tmp_path / "card" / "DCIM"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    t = datetime(2026, 5, 1, 22, 0, 0)
    layout = [["sharp", "blur", "blur"], ["blur", "sharp"], ["sharp"]]
    truth, n = {}, 0
    for b, kinds in enumerate(layout):
        base = scene(b + 1)
        for kind in kinds:
            n += 1
            img = cv2.GaussianBlur(base, (0, 0), 7.0) if kind == "blur" else base
            img = np.clip(img.astype(np.float32) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
            name = f"IMG_{n:04d}.JPG"
            save_jpeg(img, folder / name, t)
            truth[name] = (b, kind)
            t += timedelta(milliseconds=150)
        t += timedelta(seconds=30)
    return folder.parent, truth
