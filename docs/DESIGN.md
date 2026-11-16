# Design

## Architecture

```
 originals (read-only)          cache (~/.cache/mm-photo-culler, never the card)
 ┌──────────────┐  embedded     ┌─────────────────────────────────────────────┐
 │ CR3 ARW NEF  │──JPEG──────▶  │ folders/<name>-<hash>/cache.sqlite          │
 │ JPG          │  (rawpy)      │   photos · moments · features · scores      │
 └──────────────┘               │   decisions (yours, never overwritten)      │
        ▲                       │ previews/<id>.jpg (2048 px) · full/ (zoom)  │
        │ sidecars only,        │ models/ (detector weights)                  │
        │ on explicit export    └─────────────────────────────────────────────┘
        │                                  ▲                 ▲
 ┌──────┴───────┐   ┌──────────────────────┴───┐   ┌─────────┴──────────┐
 │  export.py   │   │ pipeline.py              │   │ server.py (FastAPI)│
 │  xmp.py      │   │ scan→ingest→group→       │   │ 127.0.0.1 only     │
 └──────────────┘   │ analyse→score            │   │ static/ vanilla JS │
                    │ ProcessPool (spawn)      │   └────────────────────┘
                    └──────────────────────────┘
```

* **Python + FastAPI + a no-build vanilla-JS page.** I considered a native UI
  (Qt) for raw image-display speed, but the browser already decodes and scales
  JPEGs on the GPU, and the review loop only needs "show cached JPEG, toggle a
  class". No Node toolchain, nothing to compile.
* **Stages are incremental and versioned.** A photo is identified by relative
  path, cached on `(size, mtime_ns)`. Re-opening a processed folder runs no
  image code (≈0.2 s for the demo set).
* **Workers only read originals and return results; the main process is the
  only SQLite writer.** Pool uses `spawn` so OpenCV/MediaPipe state is never
  forked.
* **Scoring is a pure function of stored features + config**, so changing a
  weight re-scores instantly, and `culler eval --sweep` can try alternatives
  without touching an image.

## Scoring pipeline

1. **Ingest** – `rawpy.extract_thumb()` pulls the camera's embedded JPEG; EXIF
   comes from LibRaw (+ sub-second time via exifread where the container
   allows). Cached: 2048 px preview, 360 px thumbnail, 64-bit DCT pHash, and a
   small edge-orientation layout descriptor. No demosaic.
2. **Moments** – per camera, sorted by capture time. Frames ≤ 1.5 s apart are
   one burst unless they look nothing alike; bursts ≤ 20 s apart merge if they
   look alike (to the previous frame *and* the moment's first frame, so drift
   cannot chain); hard cap 60 s. Similarity = max(pHash, edge layout): stage
   lights changing colour wreck a brightness hash but not the edge layout.
3. **Subject** (per moment, not per frame) – YuNet faces at two scales →
   MediaPipe FaceLandmarker on each face crop (blink blendshapes, head pose) →
   else MediaPipe person box (head-and-shoulders part) → else whole frame.
   Frames where the detector lost the subject are measured on the box borrowed
   from the nearest sibling, so one burst is always compared on the same region.
4. **Measurements** on the embedded JPEG at ≥ 2700 px (DCT-scaled, not
   resampled):
   * sharpness `sqrt(P99.5(|∇|)² − (2.8·median(|∇|))²) / (P98−P2 luminance)`
   * motion: structure-tensor coherence of the strongest gradients
   * exposure: clipped / crushed fraction **of the subject box**
5. **Score** = sharpness(0..1, log scale) − w·eyes − w·pose − w·motion −
   w·exposure, all penalties soft (≤ 0.18). Ranked within the moment: best
   frame is the pick, a second pick for moments of ≥ 5 frames if within 0.08.
   Reject only when: subject clearly out of focus in absolute terms (trusted
   subject only), or ≥ 0.28 below the burst's best and below 0.5, or eyes
   confidently closed while a sibling has them open.

## Data model

| table | what | owner |
|---|---|---|
| `photos` | path, companion JPEG, size/mtime, EXIF, thumb BLOB, pHash, descriptor, `moment_id`, `seq` | ingest / grouping |
| `moments` | id, start/end time, frame count | grouping |
| `features` | per-photo JSON: subject box + source, faces (box, blink, yaw), raw sharpness, contrast, noise floor, coherence, exposure | analysis |
| `scores` | score, components JSON, `auto_flag`, `auto_rank`, reasons | scoring (rebuilt freely) |
| `decisions` | `flag` (`pick`/`reject`/`none`/NULL), `rating` | **you** – no pipeline stage writes here |

Effective flag = your decision if you made one (`none` = "I cleared it"),
otherwise the automatic one.

## Risks and what was done about them

**Embedded previews differ a lot between brands** (measured on real files):

| camera | embedded preview | consequence |
|---|---|---|
| Canon R6 (CR3) | 5472×3648, full size | fine |
| Nikon Z6 (NEF) | 6048×4024, full size | fine |
| Sony A7 III (ARW) | **1616×1080** | a slight focus miss is invisible at this size |

For Sony: shoot RAW+JPEG (the paired JPEG is used automatically when the
embedded preview is small), or use Shift+Z for a real RAW decode on the frames
that matter. Previews are also sharpened and noise-reduced in camera; that is
common to all frames of a burst, which is why ranking is relative. A RAW with
no usable preview falls back to a half-size demosaic and says so.

**Noise vs sharpness.** Variance-of-Laplacian rises with ISO. The metric above
subtracts a noise floor in quadrature and looks only at the strongest edges.
On the synthetic sweep, adding σ=15 noise changes it by ~5 % while Laplacian
variance grows 6×; blur ordering is preserved at every noise level
(`tests/test_metrics.py`). Residual risk: real sensor noise after in-camera NR
is not white, and very dim, low-contrast subjects have little signal — those
get low confidence and are never rejected on an absolute threshold.

**Face/eye detection in concert conditions.** Coloured light, haze, small or
profile faces, microphones. Found while testing: (a) YuNet at one scale misses
frame-filling faces → two scales; (b) the landmark model hallucinates a face
onto an occluding object and reports random blink values → its eyes must land
on YuNet's independently detected eye points or the answer is discarded;
(c) eye state is only evaluated for faces ≥ 70 px. Closed eyes are often
intentional on stage, so they only break ties, and only reject when another
frame of the same burst has them open.

**Motion blur is the weakest signal.** Directional gradients also come from
mic stands, strings and light beams, so the cue is used only relative to the
burst's median and only on frames that are already softer than the best one.
It cannot tell intentional panning from shake.

**Calibration is from synthetic data.** Thresholds were set on synthetically
degraded photos (top pick truly sharp in 31/31 bursts, no sharp frame
rejected), not on real concert RAWs. Treat the reject thresholds as a starting
point and run `culler eval` on one real event before trusting auto-rejects.

**Lightroom does not read pick/reject flags from XMP** – they live only in the
catalog. Picks/rejects are exported as colour labels (Green/Red by default)
plus `xmp:Rating`. Label names must match the Lightroom label set in use.
Lightroom also ignores sidecars for JPEG-only files; since originals are never
modified, JPEG-only decisions are not delivered (they are still copied as
keepers).

**Existing sidecars** may hold develop settings: they are merged (only
`xmp:Rating`/`xmp:Label` change), backed up to the cache first, written
atomically; an unparsable sidecar is left alone and reported.

**Not verified here:** portrait-orientation RAWs (no sample; orientation comes
from LibRaw's flip), Lightroom actually importing the sidecars (no Lightroom in
the build environment — the format follows the XMP spec), macOS/Windows, GPU.
MediaPipe needs `libEGL` on a headless Linux box; without MediaPipe the app
runs with faces but no eye state and tells you.
