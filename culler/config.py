"""All tunables in one place.

Anything here can be overridden with a JSON file passed as ``--config``; the
evaluation script (``culler eval --sweep``) is the intended way to decide
whether a change is an improvement on *your* photos.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# Bump when the corresponding stage's output changes, so cached rows are redone.
INGEST_VERSION = 1
FEATURE_VERSION = 1

RAW_EXTS = {".cr3", ".cr2", ".arw", ".nef", ".nrw", ".raf", ".rw2", ".orf", ".dng", ".pef", ".srw"}
JPEG_EXTS = {".jpg", ".jpeg"}


@dataclass
class Config:
    # --- ingest ---------------------------------------------------------
    preview_edge: int = 2048      # long edge of the cached review preview
    thumb_edge: int = 360
    # Sharpness is measured on the embedded JPEG decoded at >= this long edge
    # (JPEG DCT scaling only, never resampled), so full-size Canon/Nikon
    # previews are analysed at 1/2 scale and Sony's 1616 px one natively.
    analysis_edge: int = 2700
    workers: int = 0              # 0 = number of CPU cores

    # --- grouping -------------------------------------------------------
    burst_gap_s: float = 1.5      # frames closer than this are one burst...
    burst_min_sim: float = 0.10   # ...unless they look nothing alike
    moment_gap_s: float = 20.0    # bursts this close are merged...
    moment_min_sim: float = 0.55  # ...if they look alike
    moment_max_s: float = 60.0    # hard cap on a moment's duration

    # --- detection ------------------------------------------------------
    face_min_score: float = 0.6
    face_min_px: int = 28         # in preview pixels; smaller faces are ignored
    max_faces: int = 4            # faces analysed per frame (largest first)
    eye_min_face_px: int = 70     # below this eye state is "unknown"

    # --- scoring --------------------------------------------------------
    w_eyes: float = 0.18          # penalty for confidently closed eyes
    w_pose: float = 0.08          # penalty for face turned away
    w_motion: float = 0.12        # penalty for motion blur
    w_exposure: float = 0.08      # max penalty for blown / crushed subject
    picks_small: int = 1          # picks for moments with < second_pick_min frames
    second_pick_min: int = 5      # moments this large may get a 2nd pick
    second_pick_margin: float = 0.08
    reject_abs: float = 0.20      # absolute floor (only with a trusted subject)
    reject_rel: float = 0.28      # ...or this far below the best in the moment
    reject_rel_ceiling: float = 0.5  # never reject a frame scoring above this
    reject_closed_eyes: bool = True  # only when a sibling frame has them open
    sharp_lo: float = 0.02        # raw sharpness mapped to 0 ...
    sharp_hi: float = 0.50        # ... and to 1 (log scale in between)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        cfg = cls()
        if path:
            data = json.loads(Path(path).read_text())
            known = {f.name for f in fields(cls)}
            unknown = set(data) - known
            if unknown:
                raise ValueError(f"unknown config keys: {sorted(unknown)}")
            for k, v in data.items():
                setattr(cfg, k, v)
        return cfg

    def n_workers(self) -> int:
        return self.workers or max(1, os.cpu_count() or 1)


def cache_root() -> Path:
    """Where thumbnails, previews, models and databases live.

    Deliberately *outside* the photo folder: the SD card / import folder is
    never written to except by an explicit export.
    """
    env = os.environ.get("MMCULL_CACHE")
    if env:
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "mm-photo-culler"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "mm-photo-culler"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mm-photo-culler"
