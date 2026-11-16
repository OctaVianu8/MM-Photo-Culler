"""End-to-end on synthetic JPEG bursts (no detector models needed: with no
subject found, frames are judged on their sharpest region)."""
import shutil

import pytest
from fastapi.testclient import TestClient

from culler import evaluate, export, pipeline
from culler.config import Config

from conftest import tree_hash


@pytest.fixture()
def processed(event, monkeypatch):
    folder, truth = event
    # keep the test offline and fast: no model downloads
    monkeypatch.setattr("culler.models.ensure", lambda name, log=print: None)
    before = tree_hash(folder)
    db = pipeline.open_folder(folder)
    cfg = Config(workers=2)
    summary = pipeline.run(db, cfg, log=lambda m: None)
    yield folder, truth, db, cfg, summary, before
    db.close()


def test_grouping_scoring_and_cache(processed):
    folder, truth, db, cfg, summary, before = processed
    assert summary["files"] == 6 and summary["moments"] == 3
    rows = db.photo_rows()
    by_moment = {}
    for r in rows:
        by_moment.setdefault(r["moment_id"], []).append(r)
    assert sorted(len(m) for m in by_moment.values()) == [1, 2, 3]
    for m in by_moment.values():
        assert len({truth[r["rel_path"].split("/")[-1]][0] for r in m}) == 1     # one burst per moment
        best = min(m, key=lambda r: r["auto_rank"])
        assert truth[best["rel_path"].split("/")[-1]][1] == "sharp"
        assert best["auto_flag"] == "pick"
    for r in rows:
        if r["auto_flag"] == "reject":
            assert truth[r["rel_path"].split("/")[-1]][1] == "blur"
    assert rows[0]["capture_ts"] % 1 == 0 and rows[1]["capture_ts"] % 1 > 0    # sub-second EXIF read
    # originals untouched, nothing written next to them
    assert tree_hash(folder) == before
    assert {p.suffix for p in folder.rglob("*") if p.is_file()} == {".JPG"}
    # second run does no work
    again = pipeline.run(db, cfg, log=lambda m: None)
    assert again["ingested"] == 0 and again["analysed"] == 0


def test_new_and_removed_files(processed):
    folder, truth, db, cfg, *_ = processed
    d = folder / "DCIM"
    shutil.copy2(d / "IMG_0006.JPG", d / "IMG_0007.JPG")
    (d / "IMG_0001.JPG").unlink()
    s = pipeline.run(db, cfg, log=lambda m: None)
    assert s["files"] == 6 and s["ingested"] == 1
    names = {r["rel_path"].split("/")[-1] for r in db.photo_rows()}
    assert "IMG_0007.JPG" in names and "IMG_0001.JPG" not in names


def test_user_decisions_win_and_survive_rescoring(processed):
    folder, truth, db, cfg, *_ = processed
    rows = db.photo_rows()
    rej = next(r for r in rows if r["auto_flag"] == "reject")
    pick = next(r for r in rows if r["auto_flag"] == "pick")
    db.set_decisions([{"id": rej["id"], "flag": "pick", "rating": 4}, {"id": pick["id"], "flag": "none"}])
    pipeline.run(db, cfg, log=lambda m: None)
    rows = {r["id"]: r for r in db.photo_rows()}
    assert rows[rej["id"]]["flag"] == "pick" and rows[rej["id"]]["rating"] == 4
    assert rows[pick["id"]]["flag"] is None and rows[pick["id"]]["auto_flag"] == "pick"


def test_export_sidecars_and_keepers(processed, tmp_path):
    folder, truth, db, cfg, summary, before = processed
    # pretend the files are RAW so sidecars are written
    db.x("UPDATE photos SET kind='raw'")
    first = db.photo_rows()[0]
    db.set_decisions([{"id": first["id"], "flag": "pick", "rating": 5}])

    dry = export.run(db, export.Options(dry_run=True))
    assert dry["written"] > 0 and not list(folder.rglob("*.xmp"))

    rep = export.run(db, export.Options(keepers_dir=str(tmp_path / "keepers")))
    assert not rep["errors"], rep["errors"]
    side = (folder / first["rel_path"]).with_suffix(".xmp")
    text = side.read_text()
    assert 'xmp:Rating="5"' in text and 'xmp:Label="Green"' in text
    assert tree_hash(folder) == before                       # originals byte-identical, mtime too
    assert not list(folder.rglob("*.culler-tmp"))
    kept = tree_hash(tmp_path / "keepers")
    assert len(kept) == rep["picks"] and all(kept[k][0] == before[k][0] for k in kept)
    assert (tmp_path / "keepers" / first["rel_path"]).with_suffix(".xmp").exists()

    # idempotent; and a sidecar edited elsewhere is merged + backed up
    assert export.run(db, export.Options())["written"] == 0
    side.write_text(text.replace('xmp:Rating="5"', 'xmp:Rating="5" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" crs:Exposure2012="+1.00"'))
    db.set_decisions([{"id": first["id"], "rating": 3}])
    rep = export.run(db, export.Options())
    assert rep["updated"] == 1
    assert 'crs:Exposure2012="+1.00"' in side.read_text() and 'xmp:Rating="3"' in side.read_text()
    assert list((db.dir / "xmp_backup").rglob("*.xmp"))

    # keepers inside the photo folder would be re-imported: refused
    assert export.run(db, export.Options(keepers_dir=str(folder / "keepers")))["errors"]


def test_jpeg_only_gets_no_sidecar(processed):
    folder, truth, db, *_ = processed
    rep = export.run(db, export.Options())
    assert rep["written"] == 0 and rep["jpeg_only_skipped"] > 0
    assert not list(folder.rglob("*.xmp"))


def test_eval_report(processed):
    folder, truth, db, cfg, *_ = processed
    assert "No labelled moments" in evaluate.report(db, cfg)
    rows = db.photo_rows()
    labels = [{"id": r["id"], "flag": "pick" if truth[r["rel_path"].split("/")[-1]][1] == "sharp" else "reject"}
              for r in rows]
    db.set_decisions(labels)
    text = evaluate.report(db, cfg, sweep=True)
    assert "auto's #1 frame is one of your picks:   2/2 (100%)" in text
    assert "auto-rejects you actually KEPT (the costly error): 0/" in text


def test_api(processed):
    from culler import server
    folder, truth, db, cfg, *_ = processed
    app = server.create_app(cfg)
    app.state.session.db = db
    c = TestClient(app)
    photos = c.get("/api/photos").json()["photos"]
    assert len(photos) == 6 and photos[0]["moment"] is not None
    pid = photos[0]["id"]
    for url in (f"/api/thumb/{pid}", f"/api/preview/{pid}", f"/api/full/{pid}"):
        r = c.get(url)
        assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    assert c.post("/api/decisions", json={"items": [{"id": pid, "flag": "reject", "rating": 2}]}).status_code == 200
    assert c.post("/api/decisions", json={"items": [{"id": pid, "flag": "bogus"}]}).status_code == 400
    p = next(p for p in c.get("/api/photos").json()["photos"] if p["id"] == pid)
    assert p["user"] == "reject" and p["rating"] == 2
    assert c.get("/api/status", headers={"host": "evil.example"}).status_code == 403
    assert c.get("/").status_code == 200
