"""Local web server: JSON API + static review UI. Binds to 127.0.0.1 only."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import export, pipeline, raw
from .config import JPEG_EXTS, RAW_EXTS, Config, cache_root
from .db import DB

STATIC = Path(__file__).parent / "static"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "testserver"}


class Session:
    """The one open folder and its background job."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db: DB | None = None
        self.thread: threading.Thread | None = None
        self.cancel = threading.Event()
        self.progress = {"stage": "idle", "done": 0, "total": 0, "running": False, "error": None,
                         "message": "", "eta_s": None}
        self.version = 0          # bumped whenever photo data changes
        self.lock = threading.Lock()

    def open(self, folder: str) -> None:
        with self.lock:
            self.stop()
            self.db = pipeline.open_folder(folder)
            self.cancel = threading.Event()
            self.progress = {**self.progress, "stage": "scan", "running": True, "error": None}
            self.thread = threading.Thread(target=self._run, args=(self.db, self.cancel), daemon=True)
            self.thread.start()

    def stop(self) -> None:
        if self.thread and self.thread.is_alive():
            self.cancel.set()
            self.thread.join(timeout=60)
        self.thread = None

    def _run(self, db: DB, cancel: threading.Event) -> None:
        def cb(state: dict) -> None:
            if state["stage"] != self.progress.get("stage") or state["done"] % 16 == 0:
                self.version += 1
            self.progress = state
        try:
            pipeline.run(db, self.cfg, pipeline.Progress(cb), cancel, log=lambda m: None)
        except Exception as e:  # surfaced through /api/status
            self.progress = {**self.progress, "stage": "error", "running": False,
                             "error": f"{type(e).__name__}: {e}"}
        self.version += 1

    def need_db(self) -> DB:
        if self.db is None:
            raise HTTPException(409, "no folder open")
        return self.db


def _slim(r: dict) -> dict:
    """Photo row -> what the UI needs."""
    ft = r["features"] or {}
    return {
        "id": r["id"], "name": r["rel_path"], "companion": r["companion"], "kind": r["kind"],
        "ts": r["capture_ts"], "camera": r["camera"], "lens": r["lens"], "iso": r["iso"],
        "shutter": r["shutter"], "aperture": r["aperture"], "focal": r["focal"],
        "w": r["preview_w"], "h": r["preview_h"], "full_w": r["full_w"], "full_h": r["full_h"],
        "src": r["preview_src"], "moment": r["moment_id"], "seq": r["seq"],
        "score": r["score"], "comp": r["components"], "auto": r["auto_flag"], "rank": r["auto_rank"],
        "reasons": r["reasons"] or [], "user": r["user_flag"], "rating": r["rating"],
        "subject": ft.get("subject"),
        "faces": [{"box": f["box"], "blink": f.get("blink"), "yaw": f.get("yaw")} for f in ft.get("faces", [])],
        "analysed": bool(ft),
    }


def create_app(cfg: Config | None = None, folder: str | None = None) -> FastAPI:
    app = FastAPI(title="MM Photo Culler", docs_url=None, redoc_url=None)
    sess = Session(cfg or Config())
    app.state.session = sess

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # The API can list directories and read images: refuse anything that is
        # not addressed to localhost (blocks DNS-rebinding from a web page).
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        if host not in ALLOWED_HOSTS:
            return JSONResponse({"detail": "local access only"}, status_code=403)
        return await call_next(request)

    @app.on_event("shutdown")
    def _shutdown():
        sess.stop()

    @app.get("/api/status")
    def status():
        db = sess.db
        out = {"folder": str(db.folder) if db else None, "progress": sess.progress,
               "version": sess.version}
        if db:
            out["capabilities"] = db.get_meta("capabilities")
            out["errors"] = [dict(r) for r in db.q(
                "SELECT rel_path, error FROM photos WHERE error IS NOT NULL AND missing=0 LIMIT 50")]
        return out

    @app.post("/api/open")
    def open_folder(body: dict):
        try:
            sess.open(body.get("folder", ""))
        except (NotADirectoryError, FileNotFoundError):
            raise HTTPException(400, "not a folder")
        return {"ok": True, "folder": str(sess.db.folder)}

    @app.post("/api/cancel")
    def cancel():
        sess.cancel.set()
        return {"ok": True}

    @app.get("/api/photos")
    def photos():
        db = sess.need_db()
        return {"version": sess.version, "photos": [_slim(r) for r in db.photo_rows()]}

    @app.get("/api/thumb/{pid}")
    def thumb(pid: int):
        r = sess.need_db().q("SELECT thumb FROM photos WHERE id=?", (pid,))
        if not r or r[0][0] is None:
            raise HTTPException(404)
        return Response(r[0][0], media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/preview/{pid}")
    def preview(pid: int):
        p = sess.need_db().preview_path(pid)
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/full/{pid}")
    def full(pid: int, raw_decode: int = 0):
        """Full-resolution view for the 100 % zoom: the embedded preview at its
        native size, or (raw_decode=1) an actual demosaic of the RAW."""
        db = sess.need_db()
        rows = db.q("SELECT rel_path, companion, preview_src, orientation, kind FROM photos WHERE id=?", (pid,))
        if not rows:
            raise HTTPException(404)
        r = rows[0]
        out = db.dir / "full" / (f"{pid}_raw.jpg" if raw_decode else f"{pid}.jpg")
        if not out.exists():
            try:
                if raw_decode and r["kind"] == "raw":
                    im = raw.raw_render(db.folder / r["rel_path"])
                else:
                    im = raw.load_image(db.folder, r["rel_path"], r["companion"], r["preview_src"],
                                        r["orientation"], 1 << 16)
            except Exception as e:
                raise HTTPException(500, f"cannot read original: {e}")
            tmp = out.with_suffix(".tmp")
            im.save(tmp, "JPEG", quality=92)
            os.replace(tmp, out)
            _trim(db.dir / "full", keep=300)
        return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/decisions")
    def decisions(body: dict):
        try:
            sess.need_db().set_decisions(body.get("items", []))
        except (ValueError, KeyError, sqlite3.IntegrityError) as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.post("/api/export")
    def do_export(body: dict):
        db = sess.need_db()
        allowed = set(export.Options.__dataclass_fields__)
        opts = export.Options(**{k: v for k, v in body.items() if k in allowed})
        rep = export.run(db, opts)
        rep["text"] = export.format_report(rep)
        rep["files"] = rep["files"][:200]
        return rep

    @app.get("/api/browse")
    def browse(path: str = ""):
        """Folder picker: sub-directories of `path` (default: home)."""
        p = Path(path).expanduser() if path else Path.home()
        if not p.is_dir():
            raise HTTPException(400, "not a folder")
        p = p.resolve()
        dirs, n_photos = [], 0
        try:
            for e in sorted(os.scandir(p), key=lambda e: e.name.lower()):
                if e.name.startswith("."):
                    continue
                if e.is_dir(follow_symlinks=True):
                    dirs.append(e.name)
                elif os.path.splitext(e.name)[1].lower() in RAW_EXTS | JPEG_EXTS:
                    n_photos += 1
        except PermissionError:
            pass
        return {"path": str(p), "parent": str(p.parent) if p.parent != p else None,
                "dirs": dirs, "photos": n_photos, "recent": _recent()}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    if folder:
        sess.open(folder)
    return app


def _recent() -> list[str]:
    out = []
    root = cache_root() / "folders"
    if root.is_dir():
        for d in sorted(root.iterdir(), key=lambda d: -d.stat().st_mtime)[:8]:
            try:
                con = sqlite3.connect(f"file:{d / 'cache.sqlite'}?mode=ro", uri=True)
                row = con.execute("SELECT value FROM meta WHERE key='folder'").fetchone()
                con.close()
                if row and Path(json.loads(row[0])).is_dir():
                    out.append(json.loads(row[0]))
            except (sqlite3.Error, OSError):
                pass
    return out


def _trim(d: Path, keep: int) -> None:
    files = sorted(d.glob("*.jpg"), key=lambda f: f.stat().st_mtime)
    for f in files[:-keep]:
        f.unlink(missing_ok=True)
