"""Milestone 3b: turn measurements into a score, picks and rejects.

Pure functions of (features, config): re-scoring after changing a weight is
instant and never touches an image. That is what ``culler eval --sweep`` uses.

The score is sharpness-on-subject (0..1) minus soft penalties. Everything is
judged *within a moment*: frames of one burst share ISO, light and framing,
so their differences are meaningful even when the absolute numbers are not.
"""
from __future__ import annotations

import json
import math
from statistics import median

from .config import Config
from .db import DB


def ramp(v: float, lo: float, hi: float) -> float:
    return 0.0 if v <= lo else 1.0 if v >= hi else (v - lo) / (hi - lo)


def sharp01(raw: float, cfg: Config) -> float:
    if raw <= 0:
        return 0.0
    return max(0.0, min(1.0, math.log(raw / cfg.sharp_lo) / math.log(cfg.sharp_hi / cfg.sharp_lo)))


def _subject_face(ft: dict) -> dict | None:
    i = ft.get("subject_face")
    return ft["faces"][i] if i is not None and i < len(ft.get("faces", [])) else None


def score_moment(feats: list[dict | None], cfg: Config) -> list[dict]:
    """feats: one features dict per frame of a moment (None = analysis failed).
    Returns, per frame: score, components, auto_flag, auto_rank, reasons."""
    n = len(feats)
    comps: list[dict | None] = []
    for ft in feats:
        if not ft:
            comps.append(None)
            continue
        src = ft["subject"]["src"]
        face = _subject_face(ft)
        c = {"sharp": sharp01(ft["sharp_raw"], cfg), "src": src,
             "trusted": src != "frame" and ft.get("sharp_conf", 0) >= 0.5,
             "eyes": 0.0, "pose": 0.0, "motion": 0.0, "exposure": 0.0,
             "blink": face.get("blink") if face else None}
        if face:
            if face.get("blink") is not None:
                c["eyes"] = ramp(face["blink"], 0.45, 0.75)
            c["pose"] = ramp(abs(face.get("yaw", 0.0)), 40, 75)
        elif src == "borrowed" and ft["subject"].get("from") == "face":
            c["pose"] = 0.6      # siblings show a face, this frame lost it
        ex = ft.get("subject_exposure") or {}
        c["exposure"] = (0.7 * ramp(ex.get("clip", 0), 0.2, 0.7) + 0.3 * ramp(ex.get("crush", 0), 0.7, 0.98))
        if src == "frame":
            c["exposure"] *= 0.5
        comps.append(c)

    ok = [c for c in comps if c]
    best_sharp = max((c["sharp"] for c in ok), default=0.0)
    cohs = [ft["coherence"] for ft in feats if ft]
    coh_ref = median(cohs) if len(cohs) >= 3 else 0.45
    for ft, c in zip(feats, comps):
        if not c:
            continue
        # directional blur only counts if the frame is also soft
        soft = ramp(best_sharp - c["sharp"], 0.05, 0.2) if n >= 2 else ramp(0.5 - c["sharp"], 0.0, 0.25)
        c["motion"] = ramp(ft["coherence"] - coh_ref, 0.12, 0.35) * soft
        c["score"] = max(0.0, min(1.0, c["sharp"] - cfg.w_eyes * c["eyes"] - cfg.w_pose * c["pose"]
                                  - cfg.w_motion * c["motion"] - cfg.w_exposure * c["exposure"]))

    order = sorted((i for i, c in enumerate(comps) if c), key=lambda i: -comps[i]["score"])
    best = comps[order[0]]["score"] if order else 0.0
    eyes_open_somewhere = any(c["blink"] is not None and c["blink"] < 0.3 for c in ok)

    out: list[dict] = [{"score": None, "components": None, "auto_flag": None, "auto_rank": None,
                        "reasons": ["analysis failed"]} for _ in feats]
    rejected = set()
    for rank, i in enumerate(order, 1):
        c = comps[i]
        reasons = []
        flag = None
        if c["trusted"] and c["score"] < cfg.reject_abs:
            flag = "reject"
            reasons.append("subject out of focus")
        elif n >= 2 and c["score"] < best - cfg.reject_rel and c["score"] < cfg.reject_rel_ceiling:
            flag = "reject"
            reasons.append(f"much softer than the best frame ({c['score']:.2f} vs {best:.2f})")
        elif cfg.reject_closed_eyes and c["eyes"] >= 0.99 and eyes_open_somewhere and rank > 1:
            flag = "reject"
        if c["eyes"] >= 0.5:
            reasons.append("eyes closed")
        if c["motion"] >= 0.5:
            reasons.append("motion blur")
        if c["pose"] >= 0.5:
            reasons.append("face not found in this frame" if c["src"] == "borrowed" else "face turned away")
        if c["exposure"] >= 0.5:
            reasons.append("subject exposure clipped")
        if c["src"] == "frame":
            reasons.append("no subject found - judged on whole frame")
        if flag == "reject":
            rejected.add(i)
        out[i] = {"score": round(c["score"], 4), "auto_flag": flag, "auto_rank": rank,
                  "reasons": reasons,
                  "components": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()
                                 if k != "score"}}

    # picks: the best frame, plus a second for big moments if it is nearly as good
    cands = [i for i in order if i not in rejected]
    if cands:
        first = cands[0]
        picks = [first]
        if n >= cfg.second_pick_min:
            near = [i for i in cands[1:] if comps[i]["score"] >= comps[first]["score"] - cfg.second_pick_margin]
            # among practically-equal runners-up, prefer one that is not the
            # neighbouring (near-identical) frame
            apart = [i for i in near if abs(i - first) >= 2
                     and comps[i]["score"] >= comps[near[0]]["score"] - 0.02]
            if near:
                picks.append((apart or near)[0])
        picks = picks[:max(cfg.picks_small, 1) if n < cfg.second_pick_min else 2]
        for k, i in enumerate(picks):
            out[i]["auto_flag"] = "pick"
            out[i]["reasons"].insert(0, (f"best of {n}" if k == 0 else f"runner-up of {n}") if n > 1 else "single frame")
    return out


def rescore(db: DB, cfg: Config) -> None:
    rows = db.q("""SELECT p.id, p.moment_id, f.data FROM photos p
                   LEFT JOIN features f ON f.photo_id = p.id
                   WHERE p.missing=0 AND p.ingest_version>0 ORDER BY p.moment_id, p.seq""")
    moments: dict[int, list] = {}
    for r in rows:
        moments.setdefault(r["moment_id"], []).append(r)
    out = []
    for frames in moments.values():
        res = score_moment([json.loads(r["data"]) if r["data"] else None for r in frames], cfg)
        for r, s in zip(frames, res):
            out.append((r["id"], s["score"], json.dumps(s["components"]), s["auto_flag"],
                        s["auto_rank"], json.dumps(s["reasons"])))
    with db.lock, db.conn:
        db.conn.execute("DELETE FROM scores")
        db.conn.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?)", out)
