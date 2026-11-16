"""SQLite cache: one database per photo folder, stored under the cache root.

Three kinds of data live here, kept apart on purpose:

* derived, disposable data (``photos`` metadata/thumbs, ``features``,
  ``scores``, ``moments``) - safe to delete, rebuilt from the originals;
* the machine's opinion (``scores.auto_flag``);
* *your* opinion (``decisions``) - never overwritten by any pipeline stage.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from .config import cache_root

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS photos (
    id            INTEGER PRIMARY KEY,
    rel_path      TEXT NOT NULL UNIQUE,   -- primary file (RAW if there is one)
    companion     TEXT,                   -- paired JPEG of a RAW+JPEG shot
    kind          TEXT NOT NULL,          -- 'raw' | 'jpeg'
    size          INTEGER NOT NULL,       -- size + mtime_ns = cache key
    mtime_ns      INTEGER NOT NULL,
    missing       INTEGER NOT NULL DEFAULT 0,
    ingest_version INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    capture_ts    REAL,                   -- wall-clock seconds, sub-second if known
    camera        TEXT,
    lens          TEXT,
    iso           REAL,
    shutter       REAL,                   -- seconds
    aperture      REAL,
    focal         REAL,
    orientation   INTEGER,                -- EXIF orientation applied to previews
    full_w        INTEGER,                -- embedded preview size (after rotation)
    full_h        INTEGER,
    preview_w     INTEGER,                -- cached preview size
    preview_h     INTEGER,
    preview_src   TEXT,                   -- 'embedded' | 'companion' | 'file' | 'raw_decode'
    thumb         BLOB,
    phash         TEXT,                   -- 64-bit DCT hash, hex
    descr         BLOB,                   -- float16 gradient-layout descriptor
    moment_id     INTEGER,
    seq           INTEGER                 -- position inside the moment
);
CREATE INDEX IF NOT EXISTS photos_moment ON photos(moment_id, seq);

CREATE TABLE IF NOT EXISTS moments (
    id INTEGER PRIMARY KEY, start_ts REAL, end_ts REAL, n INTEGER
);

CREATE TABLE IF NOT EXISTS features (
    photo_id INTEGER PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
    version  INTEGER NOT NULL,
    data     TEXT NOT NULL                -- JSON, see features.analyse_frame
);

CREATE TABLE IF NOT EXISTS scores (
    photo_id   INTEGER PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
    score      REAL,
    components TEXT,                      -- JSON
    auto_flag  TEXT,                      -- 'pick' | 'reject' | NULL
    auto_rank  INTEGER,                   -- 1 = best in its moment
    reasons    TEXT                       -- JSON list of short strings
);

CREATE TABLE IF NOT EXISTS decisions (
    photo_id   INTEGER PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
    flag       TEXT,     -- 'pick' | 'reject' | 'none' (explicitly unflagged) | NULL (no opinion)
    rating     INTEGER,  -- 0..5 or NULL
    updated_at REAL
);
"""


def folder_cache_dir(folder: Path) -> Path:
    key = hashlib.sha1(str(folder.resolve()).encode()).hexdigest()[:16]
    return cache_root() / "folders" / f"{folder.resolve().name[:40]}-{key}"


class DB:
    def __init__(self, folder: Path):
        self.folder = folder.resolve()
        self.dir = folder_cache_dir(self.folder)
        (self.dir / "previews").mkdir(parents=True, exist_ok=True)
        (self.dir / "full").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.dir / "cache.sqlite", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.set_meta("folder", str(self.folder))

    # -- helpers -----------------------------------------------------------
    def q(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def x(self, sql: str, args=()) -> None:
        with self.lock, self.conn:
            self.conn.execute(sql, args)

    def xmany(self, sql: str, rows) -> None:
        with self.lock, self.conn:
            self.conn.executemany(sql, rows)

    def set_meta(self, key: str, value) -> None:
        self.x("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def get_meta(self, key: str, default=None):
        r = self.q("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(r[0][0]) if r else default

    def preview_path(self, photo_id: int) -> Path:
        return self.dir / "previews" / f"{photo_id}.jpg"

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    # -- decisions ---------------------------------------------------------
    def set_decisions(self, items: list[dict]) -> None:
        """items: {id, flag?, rating?}; a key that is absent is left unchanged."""
        now = time.time()
        with self.lock, self.conn:
            for it in items:
                pid = int(it["id"])
                self.conn.execute(
                    "INSERT OR IGNORE INTO decisions(photo_id, updated_at) VALUES (?, ?)", (pid, now))
                if "flag" in it:
                    if it["flag"] not in ("pick", "reject", "none", None):
                        raise ValueError(f"bad flag {it['flag']!r}")
                    self.conn.execute("UPDATE decisions SET flag=?, updated_at=? WHERE photo_id=?",
                                      (it["flag"], now, pid))
                if "rating" in it:
                    r = it["rating"]
                    if r is not None and not 0 <= int(r) <= 5:
                        raise ValueError(f"bad rating {r!r}")
                    self.conn.execute("UPDATE decisions SET rating=?, updated_at=? WHERE photo_id=?",
                                      (r, now, pid))

    def photo_rows(self) -> list[dict]:
        """Everything the UI / export / eval need, one dict per photo, in shooting order."""
        rows = self.q("""
            SELECT p.id, p.rel_path, p.companion, p.kind, p.capture_ts, p.camera, p.lens, p.iso,
                   p.shutter, p.aperture, p.focal, p.full_w, p.full_h, p.preview_w, p.preview_h,
                   p.preview_src, p.moment_id, p.seq, p.error,
                   f.data AS features, s.score, s.components, s.auto_flag, s.auto_rank, s.reasons,
                   d.flag AS user_flag, d.rating
            FROM photos p
            LEFT JOIN features f ON f.photo_id = p.id
            LEFT JOIN scores s ON s.photo_id = p.id
            LEFT JOIN decisions d ON d.photo_id = p.id
            LEFT JOIN moments m ON m.id = p.moment_id
            WHERE p.missing = 0 AND p.ingest_version > 0
            ORDER BY m.start_ts, p.moment_id, p.seq, p.capture_ts, p.rel_path""")
        out = []
        for r in rows:
            d = dict(r)
            for k in ("features", "components", "reasons"):
                d[k] = json.loads(d[k]) if d[k] else None
            d["flag"] = effective_flag(d["user_flag"], d["auto_flag"])
            out.append(d)
        return out


def effective_flag(user_flag: str | None, auto_flag: str | None) -> str | None:
    """Your decision always wins; 'none' means you explicitly cleared the flag."""
    if user_flag is not None:
        return None if user_flag == "none" else user_flag
    return auto_flag
