"""Command line entry point: ``culler <command>``."""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import Config


def _progress_printer():
    last = {"stage": None, "t": 0.0}

    def cb(s: dict) -> None:
        now = time.time()
        new_stage = s["stage"] != last["stage"]
        if new_stage and last["stage"] in ("ingest", "analyse"):
            sys.stderr.write("\n")
        if s["total"]:
            if new_stage or now - last["t"] > 0.2 or s["done"] == s["total"]:
                eta = f" eta {s['eta_s']:.0f}s" if s.get("eta_s") else ""
                sys.stderr.write(f"\r{s['stage']:8s} {s['done']}/{s['total']}{eta}   ")
                last["t"] = now
        elif new_stage and s["stage"] not in ("done",):
            sys.stderr.write(f"{s['stage']}: {s.get('message', '')}\n")
        sys.stderr.flush()
        last["stage"] = s["stage"]
    return cb


def cmd_process(a) -> int:
    from . import pipeline
    cfg = Config.load(a.config)
    if a.workers:
        cfg.workers = a.workers
    db = pipeline.open_folder(a.folder)
    summary = pipeline.run(db, cfg, pipeline.Progress(_progress_printer()),
                           log=lambda m: print(m, file=sys.stderr))
    sys.stderr.write("\n")
    caps = db.get_meta("capabilities") or {}
    counts = {r[0]: r[1] for r in db.q("SELECT auto_flag, COUNT(*) FROM scores GROUP BY auto_flag")}
    print(f"{summary['files']} photos in {summary['moments']} moments "
          f"({summary['ingested']} ingested, {summary['analysed']} analysed, {summary['seconds']} s)")
    print(f"auto: {counts.get('pick', 0)} picks, {counts.get('reject', 0)} rejects, "
          f"{counts.get(None, 0)} undecided")
    for cap, label in (("faces", "face detection"), ("eyes", "eye state / head pose"),
                       ("persons", "person detection")):
        if caps and not caps.get(cap):
            why = (caps.get("errors") or {}).get(cap, "model missing")
            print(f"note: {label} is OFF ({why})")
    print(f"cache: {db.dir}")
    return 0


def cmd_serve(a) -> int:
    import uvicorn

    from . import server
    cfg = Config.load(a.config)
    if a.workers:
        cfg.workers = a.workers
    app = server.create_app(cfg, a.folder)
    url = f"http://127.0.0.1:{a.port}"
    print(f"MM Photo Culler: {url}")
    if not a.no_browser:
        import threading
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
    return 0


def cmd_export(a) -> int:
    from . import export, pipeline
    db = pipeline.open_folder(a.folder)
    opts = export.Options(pick_label=a.pick_label, reject_label=a.reject_label,
                          min_pick_rating=a.min_pick_rating, include_auto=not a.only_my_decisions,
                          reject_rating=a.reject_rating, keepers_dir=a.keepers, keepers_mode=a.mode,
                          dry_run=a.dry_run)
    rep = export.run(db, opts)
    print(json.dumps(rep, indent=1) if a.json else export.format_report(rep))
    return 1 if rep["errors"] else 0


def cmd_eval(a) -> int:
    from . import evaluate, pipeline
    db = pipeline.open_folder(a.folder)
    cfg = Config.load(a.config)
    print(evaluate.report(db, cfg, sweep=a.sweep, labels_csv=a.labels))
    return 0


def cmd_models(a) -> int:
    from . import models
    ok = models.ensure_all()
    for k, v in ok.items():
        print(f"{k:11s} {'ok' if v else 'MISSING'}")
    print(f"in {models.models_dir()}")
    return 0 if all(ok.values()) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="culler", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", help="JSON file overriding defaults in culler/config.py")
        p.add_argument("--workers", type=int, default=0, help="worker processes (default: CPU cores)")

    p = sub.add_parser("serve", help="start the review UI")
    p.add_argument("folder", nargs="?", help="folder to open right away")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    common(p)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("process", help="ingest + group + score a folder without the UI")
    p.add_argument("folder")
    common(p)
    p.set_defaults(fn=cmd_process)

    p = sub.add_parser("export", help="write XMP sidecars (and optionally copy keepers)")
    p.add_argument("folder")
    p.add_argument("--dry-run", action="store_true", help="show what would be written")
    p.add_argument("--pick-label", default="Green", help="xmp:Label for picks ('' = none)")
    p.add_argument("--reject-label", default="Red", help="xmp:Label for rejects ('' = none)")
    p.add_argument("--min-pick-rating", type=int, default=0,
                   help="give picks at least this many stars (0 = leave unrated picks unrated)")
    p.add_argument("--reject-rating", action="store_true",
                   help="also write xmp:Rating=-1 for rejects (Bridge convention; Lightroom ignores it)")
    p.add_argument("--only-my-decisions", action="store_true",
                   help="ignore automatic picks/rejects you have not confirmed")
    p.add_argument("--keepers", help="folder to copy/move picks into")
    p.add_argument("--mode", choices=["copy", "move"], default="copy")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("eval", help="compare automatic picks with your decisions")
    p.add_argument("folder")
    p.add_argument("--sweep", action="store_true", help="also try alternative weights")
    p.add_argument("--labels", help="write per-frame labels + scores to this CSV")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("models", help="download the detector models")
    p.set_defaults(fn=cmd_models)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
