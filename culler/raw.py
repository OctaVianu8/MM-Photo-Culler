"""Read-only access to the original files.

Every function here opens originals in read mode only. Nothing in this module
(or anywhere outside ``export.py``) writes next to the originals.
"""
from __future__ import annotations

import calendar
import io
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .config import JPEG_EXTS, RAW_EXTS

Image.MAX_IMAGE_PIXELS = None  # our own files, not untrusted uploads

# LibRaw "flip" -> EXIF orientation
_FLIP_TO_EXIF = {0: 1, 3: 3, 5: 8, 6: 6}
_T = Image.Transpose
_EXIF_TRANSPOSE = {2: _T.FLIP_LEFT_RIGHT, 3: _T.ROTATE_180, 4: _T.FLIP_TOP_BOTTOM, 5: _T.TRANSPOSE,
                   6: _T.ROTATE_270, 7: _T.TRANSVERSE, 8: _T.ROTATE_90}


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTS


def is_jpeg(path: Path) -> bool:
    return path.suffix.lower() in JPEG_EXTS


def _wall_ts(dt: datetime) -> float:
    """Naive camera wall-clock time -> float, without any timezone games."""
    return calendar.timegm(dt.timetuple()) + dt.microsecond / 1e6


def _parse_exif_dt(s: str) -> datetime | None:
    m = re.match(r"(\d{4}):(\d\d):(\d\d)[ T](\d\d):(\d\d):(\d\d)", str(s))
    if not m:
        return None
    try:
        return datetime(*map(int, m.groups()))
    except ValueError:
        return None


def _subsec(s) -> float:
    s = str(s).strip()
    return float("0." + s) if s.isdigit() else 0.0


def _ratio(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _jpeg_meta(im: Image.Image) -> dict:
    meta: dict = {}
    try:
        exif = im.getexif()
        sub = exif.get_ifd(0x8769)
    except Exception:
        return meta
    meta["orientation"] = int(exif.get(0x0112, 1) or 1)
    meta["camera"] = str(exif.get(0x0110, "") or "").strip() or None
    dt = _parse_exif_dt(sub.get(0x9003) or exif.get(0x0132) or "")
    if dt:
        meta["capture_ts"] = _wall_ts(dt) + _subsec(sub.get(0x9291, ""))
    iso = sub.get(0x8827)
    if isinstance(iso, (tuple, list)):
        iso = iso[0] if iso else None
    meta["iso"] = _ratio(iso)
    meta["shutter"] = _ratio(sub.get(0x829A))
    meta["aperture"] = _ratio(sub.get(0x829D))
    meta["focal"] = _ratio(sub.get(0x920A))
    meta["lens"] = str(sub.get(0xA434, "") or "").strip() or None
    return meta


def _raw_subsec(path: Path) -> tuple[float, str | None]:
    """Sub-second capture time + body serial from the RAW's EXIF, when the
    container is TIFF-like (ARW, NEF, CR2, DNG...). CR3 is not, so bursts
    from Canon bodies are ordered by file name within the same second."""
    if path.suffix.lower() == ".cr3":
        return 0.0, None
    try:
        import exifread
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False, extract_thumbnail=False)
        return (_subsec(tags.get("EXIF SubSecTimeOriginal", "")),
                str(tags.get("EXIF BodySerialNumber", "") or "") or None)
    except Exception:
        return 0.0, None


def read_embedded(path: Path, want_meta: bool = True) -> tuple[bytes | None, np.ndarray | None, dict]:
    """Return (jpeg_bytes, rgb_array, meta) for an original.

    For RAW files this is the camera's embedded JPEG preview - the RAW data
    itself is not demosaiced. Exactly one of jpeg_bytes / rgb_array is set.
    """
    meta: dict = {"preview_src": "file"}
    if is_jpeg(path):
        data = path.read_bytes()
        if want_meta:
            with Image.open(io.BytesIO(data)) as im:
                meta.update(_jpeg_meta(im))
        return data, None, meta

    import rawpy
    with rawpy.imread(str(path)) as raw:
        if want_meta:
            try:
                o = raw.other
                if o.timestamp and o.timestamp.year > 1990:
                    meta["capture_ts"] = _wall_ts(o.timestamp)
                meta["iso"] = float(o.iso_speed) or None
                meta["shutter"] = float(o.shutter_speed) or None
                meta["aperture"] = float(o.aperture) or None
                meta["focal"] = float(o.focal_length) or None
            except Exception:
                pass
            try:
                meta["lens"] = (raw.lens.model or "").strip() or None
            except Exception:
                pass
        meta["orientation"] = _FLIP_TO_EXIF.get(raw.sizes.flip, 1)
        try:
            thumb = raw.extract_thumb()
        except Exception:  # no / unsupported embedded preview
            thumb = None
        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
            meta["preview_src"] = "embedded"
            jpeg, arr = bytes(thumb.data), None
        elif thumb is not None:
            meta["preview_src"] = "embedded"
            jpeg, arr = None, np.asarray(thumb.data)
        else:
            # Last resort: a half-size demosaic (slow, ~1 s). Reported in the UI.
            meta["preview_src"] = "raw_decode"
            meta["orientation"] = 1  # postprocess already rotates
            jpeg, arr = None, raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
    if want_meta:
        sub, serial = _raw_subsec(path)
        if "capture_ts" in meta:
            meta["capture_ts"] += sub
        if jpeg is not None and not meta.get("camera"):
            try:
                with Image.open(io.BytesIO(jpeg)) as im:
                    meta["camera"] = str(im.getexif().get(0x0110, "") or "").strip() or None
            except Exception:
                pass
        if serial:
            meta["camera"] = f"{meta.get('camera') or ''} #{serial}".strip()
    return jpeg, arr, meta


def decode(jpeg: bytes | None, arr: np.ndarray | None, orientation: int, min_edge: int) -> Image.Image:
    """Decode to an upright RGB PIL image whose long edge is >= min_edge when
    the source allows. Uses JPEG DCT scaling, so big previews decode quickly."""
    if jpeg is not None:
        im = Image.open(io.BytesIO(jpeg))
        im.draft("RGB", (min_edge, min_edge))
        im = im.convert("RGB")
    else:
        im = Image.fromarray(arr).convert("RGB")
    op = _EXIF_TRANSPOSE.get(orientation or 1)
    if op is not None:
        im = im.transpose(op)
    return im


def embedded_size(jpeg: bytes | None, arr: np.ndarray | None, orientation: int) -> tuple[int, int]:
    if jpeg is not None:
        with Image.open(io.BytesIO(jpeg)) as im:
            w, h = im.size
    else:
        h, w = arr.shape[:2]
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


def load_image(folder: Path, rel_path: str, companion: str | None, preview_src: str | None,
               orientation: int | None, min_edge: int) -> Image.Image:
    """Re-open the best available preview source for a photo (used by the
    analysis stage and the 100 % zoom)."""
    if preview_src == "companion" and companion:
        data = (folder / companion).read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            ori = int(im.getexif().get(0x0112, 1) or 1)
        return decode(data, None, ori, min_edge)
    jpeg, arr, meta = read_embedded(folder / rel_path, want_meta=False)
    ori = orientation if orientation else meta.get("orientation", 1)
    if is_jpeg(folder / rel_path):
        with Image.open(io.BytesIO(jpeg)) as im:
            ori = int(im.getexif().get(0x0112, 1) or 1)
    return decode(jpeg, arr, ori, min_edge)


def raw_render(path: Path) -> Image.Image:
    """Real demosaic for the 'true pixels' zoom. On demand only."""
    import rawpy
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=False)
    return Image.fromarray(rgb)


def stat_key(path: Path) -> tuple[int, int]:
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns
