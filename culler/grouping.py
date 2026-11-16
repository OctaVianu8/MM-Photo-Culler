"""Milestone 2: group frames into "moments".

Time is the primary signal (a burst is frames fired within ~1 s of each
other); visual similarity decides whether neighbouring bursts are the same
moment and stops a moment from swallowing a re-framed shot.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .db import DB


def similarity(a: dict, b: dict) -> float:
    """0 (unrelated) .. 1 (identical). Max of two cues, because either can be
    knocked out on its own: the brightness hash by a strobe, the edge layout
    by heavy blur."""
    ham = bin(int(a["phash"], 16) ^ int(b["phash"], 16)).count("1")
    s_hash = max(0.0, 1.0 - ham / 26.0)
    s_layout = float(np.dot(a["descr"], b["descr"]))
    return max(s_hash, max(0.0, s_layout))


def group(photos: list[dict], cfg: Config) -> list[list[dict]]:
    """photos: dicts with capture_ts, rel_path, camera, phash, descr (np array).
    Returns moments (lists of photos in shooting order), ordered by start time."""
    by_cam: dict[str, list[dict]] = {}
    for p in photos:
        by_cam.setdefault(p.get("camera") or "", []).append(p)
    moments: list[list[dict]] = []
    for seq in by_cam.values():
        seq.sort(key=lambda p: (p["capture_ts"] or 0.0, p["rel_path"]))
        cur = [seq[0]]
        for prev, p in zip(seq, seq[1:]):
            dt = (p["capture_ts"] or 0.0) - (prev["capture_ts"] or 0.0)
            span = (p["capture_ts"] or 0.0) - (cur[0]["capture_ts"] or 0.0)
            same = False
            if span <= cfg.moment_max_s:
                if dt <= cfg.burst_gap_s:
                    same = similarity(prev, p) >= cfg.burst_min_sim
                elif dt <= cfg.moment_gap_s:
                    # compare with the last frame and the moment's first frame,
                    # so slow drift cannot chain unrelated shots together
                    same = (similarity(prev, p) >= cfg.moment_min_sim
                            and similarity(cur[0], p) >= cfg.moment_min_sim * 0.6)
            if same:
                cur.append(p)
            else:
                moments.append(cur)
                cur = [p]
        moments.append(cur)
    moments.sort(key=lambda m: (m[0]["capture_ts"] or 0.0, m[0]["rel_path"]))
    return moments


def regroup(db: DB, cfg: Config) -> int:
    rows = db.q("""SELECT id, rel_path, capture_ts, camera, phash, descr FROM photos
                   WHERE missing=0 AND ingest_version>0""")
    photos = [{**dict(r), "descr": np.frombuffer(r["descr"], np.float16).astype(np.float32)}
              for r in rows]
    if not photos:
        with db.lock, db.conn:
            db.conn.execute("DELETE FROM moments")
        return 0
    moments = group(photos, cfg)
    with db.lock, db.conn:
        db.conn.execute("DELETE FROM moments")
        db.conn.execute("UPDATE photos SET moment_id=NULL, seq=NULL")
        for mid, m in enumerate(moments, 1):
            db.conn.execute("INSERT INTO moments VALUES (?,?,?,?)",
                            (mid, m[0]["capture_ts"], m[-1]["capture_ts"], len(m)))
            db.conn.executemany("UPDATE photos SET moment_id=?, seq=? WHERE id=?",
                                [(mid, i, p["id"]) for i, p in enumerate(m)])
    return len(moments)
