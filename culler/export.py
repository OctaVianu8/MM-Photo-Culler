"""Milestone 5: the only module that writes outside the cache.

Guarantees:
* original image files are never opened for writing, renamed or deleted
  (``move`` mode relocates keepers only when you explicitly ask for it);
* sidecars are written atomically (temp file + rename) and an existing
  sidecar is backed up into the cache before its first modification;
* nothing at a destination is ever overwritten.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from . import xmp
from .db import DB


@dataclass
class Options:
    pick_label: str = "Green"
    reject_label: str = "Red"
    min_pick_rating: int = 0        # 0 = do not invent stars
    reject_rating: bool = False     # also xmp:Rating=-1 on rejects (Bridge)
    include_auto: bool = True       # export automatic flags you did not override
    keepers_dir: str | None = None
    keepers_mode: str = "copy"      # 'copy' | 'move'
    dry_run: bool = False


def sidecar_path(original: Path) -> Path:
    """Lightroom / Camera Raw convention: IMG_1234.CR3 -> IMG_1234.xmp"""
    return original.with_suffix(".xmp")


def plan(db: DB, opts: Options) -> list[dict]:
    items = []
    for r in db.photo_rows():
        flag = r["flag"] if opts.include_auto else (None if r["user_flag"] in (None, "none") else r["user_flag"])
        rating = r["rating"] if r["rating"] else None
        label = None
        if flag == "pick":
            label = opts.pick_label or None
            if opts.min_pick_rating and (rating or 0) < opts.min_pick_rating:
                rating = opts.min_pick_rating
        elif flag == "reject":
            label = opts.reject_label or None
            if opts.reject_rating:
                rating = -1
        items.append({"id": r["id"], "rel_path": r["rel_path"], "companion": r["companion"],
                      "kind": r["kind"], "flag": flag, "rating": rating, "label": label})
    return items


def _backup(db: DB, sidecar: Path, rel: str) -> None:
    dest = db.dir / "xmp_backup" / (rel + time.strftime(".%Y%m%d-%H%M%S.xmp"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sidecar, dest)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".culler-tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _transfer(src: Path, dst: Path, mode: str, dry: bool) -> str:
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "exists"
        raise FileExistsError(f"{dst} exists with different content - not overwriting")
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "move":
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)
    return "moved" if mode == "move" else "copied"


def run(db: DB, opts: Options) -> dict:
    rep = {"dry_run": opts.dry_run, "written": 0, "updated": 0, "unchanged": 0, "picks": 0,
           "rejects": 0, "rated": 0, "jpeg_only_skipped": 0, "keepers": 0, "keepers_existing": 0,
           "errors": [], "files": []}
    owned = tuple(l for l in (opts.pick_label, opts.reject_label) if l)
    keepers = None
    if opts.keepers_dir:
        keepers = Path(opts.keepers_dir).expanduser().resolve()
        if keepers == db.folder or db.folder in keepers.parents:
            rep["errors"].append("the keepers folder must be outside the photo folder "
                                 "(it would be re-imported as new photos)")
            return rep
        if opts.keepers_mode not in ("copy", "move"):
            rep["errors"].append(f"bad keepers mode {opts.keepers_mode!r}")
            return rep

    for it in plan(db, opts):
        rep["picks"] += it["flag"] == "pick"
        rep["rejects"] += it["flag"] == "reject"
        rep["rated"] += bool(it["rating"] and it["rating"] > 0)
        original = db.folder / it["rel_path"]
        side = sidecar_path(original)
        try:
            if not original.exists():
                raise FileNotFoundError(f"{it['rel_path']} is no longer there")
            if it["kind"] != "raw":
                # Lightroom ignores sidecars for JPEG/TIFF (it reads metadata
                # embedded in the file) and we do not modify originals.
                if it["flag"] or it["rating"]:
                    rep["jpeg_only_skipped"] += 1
            else:
                existing = side.read_text(encoding="utf-8-sig") if side.exists() else None
                if existing is not None or it["rating"] is not None or it["label"]:
                    text = xmp.build(existing, it["rating"], it["label"], owned)
                    if existing is not None and _same(existing, text):
                        rep["unchanged"] += 1
                    else:
                        rep["updated" if existing is not None else "written"] += 1
                        rep["files"].append(str(side.relative_to(db.folder)))
                        if not opts.dry_run:
                            if existing is not None:
                                _backup(db, side, it["rel_path"])
                            _write_atomic(side, text)
            if keepers is not None and it["flag"] == "pick":
                for rel in filter(None, (it["rel_path"], it["companion"])):
                    res = _transfer(db.folder / rel, keepers / rel, opts.keepers_mode, opts.dry_run)
                    if rel == it["rel_path"]:
                        rep["keepers" if res != "exists" else "keepers_existing"] += 1
                if it["kind"] == "raw" and side.exists() and not opts.dry_run:
                    # the sidecar travels with its RAW; a sidecar already in the
                    # keepers folder (maybe edited in Lightroom since) is merged
                    dst = keepers / side.relative_to(db.folder)
                    if dst.exists():
                        old = dst.read_text(encoding="utf-8-sig")
                        new = xmp.build(old, it["rating"], it["label"], owned)
                        if not _same(old, new):
                            _backup(db, dst, "keepers/" + it["rel_path"])
                            _write_atomic(dst, new)
                    else:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(side, dst)
                    if opts.keepers_mode == "move":
                        side.unlink()
        except Exception as e:
            rep["errors"].append(f"{it['rel_path']}: {type(e).__name__}: {e}")
    if not opts.dry_run:
        db.set_meta("last_export", {"at": time.time(), "written": rep["written"], "updated": rep["updated"]})
    return rep


def _same(a: str, b: str) -> bool:
    try:
        return xmp.read(a) == xmp.read(b)
    except xmp.XmpError:
        return False


def format_report(rep: dict) -> str:
    lines = [("DRY RUN - nothing was written\n" if rep["dry_run"] else "") +
             f"{rep['picks']} picks, {rep['rejects']} rejects, {rep['rated']} rated",
             f"sidecars: {rep['written']} new, {rep['updated']} merged into existing, {rep['unchanged']} unchanged"]
    if rep["jpeg_only_skipped"]:
        lines.append(f"{rep['jpeg_only_skipped']} JPEG-only photos have decisions but no sidecar "
                     "(Lightroom reads sidecars only for RAW; originals are never modified)")
    if rep["keepers"] or rep["keepers_existing"]:
        lines.append(f"keepers: {rep['keepers']} transferred, {rep['keepers_existing']} already there")
    for e in rep["errors"]:
        lines.append(f"ERROR {e}")
    return "\n".join(lines)
