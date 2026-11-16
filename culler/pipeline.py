"""Orchestration: scan -> ingest -> group -> analyse -> score.

Each stage is incremental (keyed on file size + mtime and a stage version),
so re-opening a processed folder does no image work at all. CPU-heavy stages
fan out over a process pool; only the main process writes to SQLite.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from . import grouping, ingest, models, scoring
from .config import FEATURE_VERSION, Config
from .db import DB


class Cancelled(Exception):
    pass


def _caps(_=None) -> dict:
    from . import detect
    c = detect.capabilities()
    detect.close()
    return c


def _worker_exit_cleanup() -> None:
    import atexit

    from . import detect
    atexit.register(detect.close)


class Progress:
    def __init__(self, cb: Callable[[dict], None] | None = None):
        self.cb = cb
        self.state = {"stage": "idle", "done": 0, "total": 0, "running": False, "error": None,
                      "message": "", "eta_s": None}
        self._t0 = time.time()

    def stage(self, name: str, total: int, message: str = "") -> None:
        self._t0 = time.time()
        self.state.update(stage=name, done=0, total=total, message=message, eta_s=None)
        self._emit()

    def tick(self, n: int = 1) -> None:
        s = self.state
        s["done"] += n
        el = time.time() - self._t0
        s["eta_s"] = round(el / s["done"] * (s["total"] - s["done"]), 1) if s["done"] else None
        self._emit()

    def _emit(self) -> None:
        if self.cb:
            self.cb(dict(self.state))


def run(db: DB, cfg: Config, progress: Progress | None = None,
        cancel: threading.Event | None = None, log=print) -> dict:
    """Process a folder end to end. Returns a small summary."""
    pr = progress or Progress()
    pr.state.update(running=True, error=None)
    t_start = time.time()

    def check():
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    summary: dict = {}
    pool: ProcessPoolExecutor | None = None
    try:
        pr.stage("scan", 0, "scanning folder")
        files = ingest.scan(db.folder)
        todo = ingest.sync_files(db, files)
        summary.update(files=len(files), ingested=len(todo))
        check()

        def get_pool() -> ProcessPoolExecutor:
            nonlocal pool
            if pool is None:
                # spawn: MediaPipe / OpenCV state must not be inherited by fork
                pool = ProcessPoolExecutor(cfg.n_workers(), mp_context=mp.get_context("spawn"),
                                           initializer=_worker_exit_cleanup)
            return pool

        # -- 1. ingest ---------------------------------------------------
        if todo:
            pr.stage("ingest", len(todo), "extracting embedded previews")
            futs = [get_pool().submit(ingest.ingest_one, j) for j in ingest.make_jobs(db, todo, cfg)]
            for fut in as_completed(futs):
                check()
                ingest.store_ingest(db, fut.result())
                pr.tick()

        # -- 2. group ----------------------------------------------------
        pr.stage("group", 0, "grouping into moments")
        cfg_key = json.dumps([cfg.burst_gap_s, cfg.burst_min_sim, cfg.moment_gap_s,
                              cfg.moment_min_sim, cfg.moment_max_s])
        ungrouped = db.q("SELECT COUNT(*) FROM photos WHERE missing=0 AND ingest_version>0 AND moment_id IS NULL")[0][0]
        if todo or ungrouped or db.get_meta("group_cfg") != cfg_key or summary["files"] != db.get_meta("n_files"):
            grouping.regroup(db, cfg)
            db.set_meta("group_cfg", cfg_key)
            db.set_meta("n_files", summary["files"])
        summary["moments"] = db.q("SELECT COUNT(*) FROM moments")[0][0]
        check()

        # -- 3. analyse ----------------------------------------------------
        rows = db.q("""SELECT p.id, p.rel_path, p.companion, p.preview_src, p.orientation,
                              p.capture_ts, p.moment_id
                       FROM photos p LEFT JOIN features f ON f.photo_id = p.id
                       WHERE p.missing=0 AND p.ingest_version>0
                         AND p.moment_id IN (
                            SELECT p2.moment_id FROM photos p2 LEFT JOIN features f2 ON f2.photo_id=p2.id
                            WHERE p2.missing=0 AND p2.ingest_version>0
                              AND (f2.photo_id IS NULL OR f2.version != ?))
                       ORDER BY p.moment_id, p.seq""", (FEATURE_VERSION,))
        summary["analysed"] = len(rows)
        if rows:
            pr.stage("models", 0, "checking detector models")
            have = models.ensure_all(log)
            caps = get_pool().submit(_caps).result()
            caps["models"] = have
            db.set_meta("capabilities", caps)
            by_moment: dict[int, list[dict]] = {}
            for r in rows:
                by_moment.setdefault(r["moment_id"], []).append(
                    {**dict(r), "preview_path": str(db.preview_path(r["id"]))})
            pr.stage("analyse", len(rows), "detecting subjects, measuring focus")
            cfgd = asdict(cfg)
            jobs = sorted(by_moment.values(), key=len, reverse=True)   # big bursts first
            futs = [get_pool().submit(_analyse, {"folder": str(db.folder), "cfg": cfgd, "photos": ph})
                    for ph in jobs]
            for fut in as_completed(futs):
                check()
                res = fut.result()
                db.xmany("INSERT OR REPLACE INTO features VALUES (?,?,?)",
                         [(r["id"], FEATURE_VERSION, json.dumps(r["data"])) for r in res if "data" in r])
                for r in res:
                    if "error" in r:
                        log(f"analysis failed for photo {r['id']}: {r['error']}")
                pr.tick(len(res))

        # -- 4. score ----------------------------------------------------
        pr.stage("score", 0, "selecting picks")
        scoring.rescore(db, cfg)
        db.set_meta("score_cfg", cfg.to_json())
        summary["seconds"] = round(time.time() - t_start, 1)
        pr.state.update(stage="done", running=False, message="", eta_s=None)
        pr._emit()
        return summary
    except Cancelled:
        pr.state.update(stage="cancelled", running=False)
        pr._emit()
        return summary
    except Exception as e:
        pr.state.update(stage="error", running=False, error=f"{type(e).__name__}: {e}")
        pr._emit()
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


def _analyse(job: dict) -> list[dict]:
    from . import features
    return features.analyse_moment(job)


def open_folder(folder: str | Path) -> DB:
    p = Path(folder).expanduser()
    if not p.is_dir():
        raise NotADirectoryError(str(p))
    return DB(p)
