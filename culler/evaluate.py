"""How often does the automatic selection agree with you?

Workflow: open a folder in the UI, press B for *blind mode* (hides every
automatic suggestion so it cannot anchor you), cull a few hundred frames the
way you normally would, then run ``culler eval FOLDER``.

A moment counts as labelled once you have flagged at least one of its frames;
its frames you left unflagged count as "not a keeper".
"""
from __future__ import annotations

import csv
import itertools
from dataclasses import replace

from . import scoring
from .config import Config
from .db import DB


def _moments(db: DB) -> list[list[dict]]:
    by: dict[int, list[dict]] = {}
    for r in db.photo_rows():
        by.setdefault(r["moment_id"], []).append(r)
    return [m for m in by.values() if any(r["user_flag"] in ("pick", "reject") for r in m)]


def _top1(moments: list[list[dict]], cfg: Config, key=None) -> tuple[int, int, list]:
    """(#agree, #moments, disagreements) over multi-frame moments with a pick."""
    agree = total = 0
    misses = []
    for m in moments:
        picks = {r["id"] for r in m if r["user_flag"] == "pick"}
        if len(m) < 2 or not picks or any(r["features"] is None for r in m):
            continue
        total += 1
        if key is None:
            res = scoring.score_moment([r["features"] for r in m], cfg)
            best = min(range(len(m)), key=lambda i: res[i]["auto_rank"] or 1e9)
        else:
            best = max(range(len(m)), key=lambda i: key(m[i]))
        if m[best]["id"] in picks:
            agree += 1
        else:
            misses.append((m[best], [r for r in m if r["id"] in picks]))
    return agree, total, misses


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def report(db: DB, cfg: Config, sweep: bool = False, labels_csv: str | None = None) -> str:
    moments = _moments(db)
    frames = [r for m in moments for r in m]
    if not frames:
        return ("No labelled moments yet. Open the folder in the UI, press B (blind mode), "
                "pick/reject as you normally would, then run this again.")
    out = [f"Labelled: {len(moments)} moments, {len(frames)} frames "
           f"({sum(r['user_flag'] == 'pick' for r in frames)} picks, "
           f"{sum(r['user_flag'] == 'reject' for r in frames)} rejects by you)", ""]

    # live re-score with the given config, so --config experiments are comparable
    auto: dict[int, dict] = {}
    for m in moments:
        if all(r["features"] is not None for r in m):
            for r, s in zip(m, scoring.score_moment([r["features"] for r in m], cfg)):
                auto[r["id"]] = s

    agree, total, misses = _top1(moments, cfg)
    multi = [m for m in moments if len(m) >= 2 and any(r["user_flag"] == "pick" for r in m)]
    chance = sum(sum(r["user_flag"] == "pick" for r in m) / len(m) for m in multi) / max(1, len(multi))
    out.append("Bursts (moments with 2+ frames where you picked something)")
    out.append(f"  auto's #1 frame is one of your picks:   {_pct(agree, total)}"
               f"   [picking at random would score {100 * chance:.0f}%]")
    any_hit = sum(1 for m in multi if any(auto.get(r["id"], {}).get("auto_flag") == "pick"
                                          and r["user_flag"] == "pick" for r in m))
    out.append(f"  at least one auto pick is yours:        {_pct(any_hit, len(multi))}")
    yours = [r for m in multi for r in m if r["user_flag"] == "pick"]
    covered = sum(auto.get(r["id"], {}).get("auto_flag") == "pick" for r in yours)
    out.append(f"  your picks that were auto-picked:       {_pct(covered, len(yours))}")
    s_agree, s_total, _ = _top1(moments, cfg, key=lambda r: r["features"]["sharp_raw"])
    out.append(f"  (sharpness alone, no penalties:         {_pct(s_agree, s_total)})")

    out += ["", "Frames: automatic flag (rows) vs yours (columns)",
            f"  {'':10s}{'pick':>8s}{'none':>8s}{'reject':>8s}"]
    conf = {(a, u): 0 for a in ("pick", None, "reject") for u in ("pick", None, "reject")}
    for r in frames:
        u = r["user_flag"] if r["user_flag"] in ("pick", "reject") else None
        a = auto.get(r["id"], {}).get("auto_flag")
        conf[(a, u)] += 1
    for a in ("pick", None, "reject"):
        out.append(f"  {str(a or 'none'):10s}" + "".join(f"{conf[(a, u)]:8d}" for u in ("pick", None, "reject")))
    n_rej = sum(conf[("reject", u)] for u in ("pick", None, "reject"))
    out.append(f"  auto-rejects you actually KEPT (the costly error): {_pct(conf[('reject', 'pick')], n_rej)}")
    your_rej = sum(conf[(a, "reject")] for a in ("pick", None, "reject"))
    out.append(f"  your rejects found automatically:                  {_pct(conf[('reject', 'reject')], your_rej)}")

    if misses:
        out += ["", "Disagreements (auto's #1 -> what you picked) - worth a look in the UI:"]
        for best, picks in misses[:25]:
            out.append(f"  moment {best['moment_id']}: {best['rel_path']} -> "
                       + ", ".join(p["rel_path"] for p in picks[:3]))
        if len(misses) > 25:
            out.append(f"  ... and {len(misses) - 25} more")

    if sweep:
        out += ["", "Weight sweep (top-1 agreement). With a few hundred labels, differences of",
                "a couple of points are noise - only adopt a change that is clearly ahead."]
        grid = itertools.product((0.0, 0.09, 0.18, 0.30), (0.0, 0.08, 0.16), (0.0, 0.12, 0.25))
        rows = []
        for we, wp, wm in grid:
            c = replace(cfg, w_eyes=we, w_pose=wp, w_motion=wm)
            a, t, _ = _top1(moments, c)
            rows.append((a, we, wp, wm))
        rows.sort(reverse=True)
        for a, we, wp, wm in rows[:8]:
            cur = "  <- current" if (we, wp, wm) == (cfg.w_eyes, cfg.w_pose, cfg.w_motion) else ""
            out.append(f"  w_eyes={we:<5} w_pose={wp:<5} w_motion={wm:<5} {_pct(a, total)}{cur}")

    if labels_csv:
        with open(labels_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["moment", "file", "your_flag", "your_rating", "auto_flag", "auto_rank", "score",
                        "sharp", "eyes", "pose", "motion", "exposure", "subject"])
            for r in frames:
                s = auto.get(r["id"], {})
                c = s.get("components") or {}
                w.writerow([r["moment_id"], r["rel_path"], r["user_flag"] or "", r["rating"] or "",
                            s.get("auto_flag") or "", s.get("auto_rank") or "", s.get("score"),
                            c.get("sharp"), c.get("eyes"), c.get("pose"), c.get("motion"),
                            c.get("exposure"), c.get("src")])
        out += ["", f"wrote {labels_csv}"]
    return "\n".join(out)
