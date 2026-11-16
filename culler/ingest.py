"""Milestone 1: scan a folder and build the thumbnail / preview cache.

Runs in worker processes; each worker only *reads* originals and writes into
the cache directory.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import raw
from .config import INGEST_VERSION, JPEG_EXTS, RAW_EXTS, Config
from .db import DB


def scan(folder: Path) -> list[dict]:
    """Find photos (recursively). A RAW and a JPEG with the same stem in the
    same directory are one shot: the RAW is primary, the JPEG its companion."""
    found: dict[tuple[str, str], dict[str, str]] = {}
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in files:
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in RAW_EXTS and ext not in JPEG_EXTS:
                continue
            rel = os.path.relpath(os.path.join(root, name), folder)
            stem = os.path.splitext(rel)[0]
            slot = found.setdefault((os.path.dirname(rel), os.path.basename(stem).lower()), {})
            slot.setdefault("raw" if ext in RAW_EXTS else "jpeg", rel)
    out = []
    for slot in found.values():
        primary = slot.get("raw") or slot["jpeg"]
        try:
            size, mtime_ns = raw.stat_key(folder / primary)
        except OSError:
            continue
        out.append({"rel_path": primary, "kind": "raw" if "raw" in slot else "jpeg",
                    "companion": slot.get("jpeg") if "raw" in slot else None,
                    "size": size, "mtime_ns": mtime_ns})
    out.sort(key=lambda d: d["rel_path"])
    return out


def sync_files(db: DB, files: list[dict]) -> list[dict]:
    """Reconcile the scan with the cache; return rows that need (re)ingesting."""
    known = {r["rel_path"]: r for r in db.q(
        "SELECT id, rel_path, size, mtime_ns, ingest_version, companion, missing FROM photos")}
    seen = set()
    todo = []
    with db.lock, db.conn:
        for f in files:
            seen.add(f["rel_path"])
            k = known.get(f["rel_path"])
            if k is None:
                cur = db.conn.execute(
                    "INSERT INTO photos(rel_path, companion, kind, size, mtime_ns) VALUES (?,?,?,?,?)",
                    (f["rel_path"], f["companion"], f["kind"], f["size"], f["mtime_ns"]))
                todo.append({**f, "id": cur.lastrowid})
                continue
            changed = (k["size"], k["mtime_ns"]) != (f["size"], f["mtime_ns"])
            if changed or k["companion"] != f["companion"] or k["missing"]:
                db.conn.execute(
                    "UPDATE photos SET size=?, mtime_ns=?, companion=?, missing=0 WHERE id=?",
                    (f["size"], f["mtime_ns"], f["companion"], k["id"]))
            if changed:
                db.conn.execute("DELETE FROM features WHERE photo_id=?", (k["id"],))
            if changed or k["ingest_version"] != INGEST_VERSION:
                todo.append({**f, "id": k["id"]})
        for rel, k in known.items():
            if rel not in seen and not k["missing"]:
                db.conn.execute("UPDATE photos SET missing=1 WHERE id=?", (k["id"],))
    return todo


# -- visual descriptors used for grouping ------------------------------------

def phash(gray: np.ndarray) -> str:
    """Classic 64-bit DCT perceptual hash."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:8, :8].ravel()
    bits = d > np.median(d[1:])
    return f"{int(np.packbits(bits).view('>u8')[0]):016x}"


def layout_descriptor(gray: np.ndarray) -> np.ndarray:
    """Edge-orientation layout (a tiny HOG). Much less sensitive than a
    brightness hash to stage lights changing colour/intensity mid-burst."""
    g = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * 6
    bins = np.minimum(ang.astype(np.int32), 5)
    desc = np.zeros((8, 8, 6), np.float32)
    cell = np.arange(64) // 8
    np.add.at(desc, (cell[:, None].repeat(64, 1), cell[None, :].repeat(64, 0), bins), mag)
    desc = np.sqrt(desc).ravel()
    desc -= desc.mean()
    n = np.linalg.norm(desc)
    return (desc / n if n > 1e-6 else desc).astype(np.float16)


def ingest_one(job: dict) -> dict:
    """Worker: build thumb + preview + metadata for one photo."""
    folder = Path(job["folder"])
    res: dict = {"id": job["id"]}
    try:
        path = folder / job["rel_path"]
        jpeg, arr, meta = raw.read_embedded(path)
        orientation = meta.get("orientation", 1)
        full_w, full_h = raw.embedded_size(jpeg, arr, orientation)
        src = meta["preview_src"]
        # Small embedded preview (e.g. Sony's 1616 px) but a full JPEG was shot
        # alongside: the JPEG is the better thing to look at and measure.
        if job.get("companion") and max(full_w, full_h) < 2400:
            try:
                cdata = (folder / job["companion"]).read_bytes()
                with Image.open(io.BytesIO(cdata)) as cim:
                    cori = int(cim.getexif().get(0x0112, 1) or 1)
                    cw, ch = cim.size
                if max(cw, ch) > max(full_w, full_h):
                    jpeg, arr, orientation, src = cdata, None, cori, "companion"
                    full_w, full_h = (ch, cw) if cori in (5, 6, 7, 8) else (cw, ch)
            except OSError:
                pass
        im = raw.decode(jpeg, arr, orientation, job["preview_edge"])
        im.thumbnail((job["preview_edge"], job["preview_edge"]), Image.Resampling.LANCZOS)
        tmp = Path(job["preview_path"] + ".tmp")
        im.save(tmp, "JPEG", quality=88)
        os.replace(tmp, job["preview_path"])
        th = im.copy()
        th.thumbnail((job["thumb_edge"], job["thumb_edge"]), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        th.save(buf, "JPEG", quality=80)
        gray = np.asarray(th.convert("L"))
        if "capture_ts" not in meta:
            meta["capture_ts"] = job["mtime_ns"] / 1e9
            meta["ts_from_mtime"] = True
        res.update(meta, orientation=orientation, preview_src=src, full_w=full_w, full_h=full_h,
                   preview_w=im.width, preview_h=im.height, thumb=buf.getvalue(),
                   phash=phash(gray), descr=layout_descriptor(gray).tobytes(), error=None)
    except Exception as e:  # a corrupt file must not stop the run
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def store_ingest(db: DB, res: dict) -> None:
    if res.get("error"):
        db.x("UPDATE photos SET error=?, ingest_version=0 WHERE id=?", (res["error"], res["id"]))
        return
    db.x("""UPDATE photos SET error=NULL, ingest_version=?, capture_ts=?, camera=?, lens=?, iso=?,
            shutter=?, aperture=?, focal=?, orientation=?, full_w=?, full_h=?, preview_w=?,
            preview_h=?, preview_src=?, thumb=?, phash=?, descr=? WHERE id=?""",
         (INGEST_VERSION, res.get("capture_ts"), res.get("camera"), res.get("lens"), res.get("iso"),
          res.get("shutter"), res.get("aperture"), res.get("focal"), res.get("orientation"),
          res["full_w"], res["full_h"], res["preview_w"], res["preview_h"], res["preview_src"],
          res["thumb"], res["phash"], res["descr"], res["id"]))
    db.x("DELETE FROM features WHERE photo_id=?", (res["id"],))


def make_jobs(db: DB, todo: list[dict], cfg: Config) -> list[dict]:
    return [{**t, "folder": str(db.folder), "preview_path": str(db.preview_path(t["id"])),
             "preview_edge": cfg.preview_edge, "thumb_edge": cfg.thumb_edge} for t in todo]
