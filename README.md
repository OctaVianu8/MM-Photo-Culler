# MM Photo Culler

Local culling for concert and night photography: point it at a folder of
RAW (CR3 / ARW / NEF / … anything LibRaw reads) and JPEG files, it groups
bursts into moments, scores focus on the subject, pre-selects the best frame(s)
of each moment, and you confirm or override with Lightroom-style keys. Results
go to Lightroom as XMP sidecars. Originals are only ever read.

## Install

```bash
python -m venv .venv
```
```bash
. .venv/bin/activate
```
```bash
pip install -e ".[faces,dev]"
```
```bash
culler models
```

`faces` pulls in MediaPipe (closed eyes, head pose, person boxes). Without it
everything else still works. `culler models` downloads ~11 MB of detector
weights once (checksum-verified). No GPU needed.

## Use

```bash
culler serve /path/to/event
```

Opens `http://127.0.0.1:8765`. Thumbnails appear while analysis is still
running; re-opening a folder later is instant.

| key | action |
|---|---|
| `←` `→` / `↑` `↓` | frame / moment |
| `P` `X` `U` | pick, reject, unflag |
| `Shift+P` | pick this frame, reject the rest of the burst |
| `1`–`5`, `0` | stars |
| `A` | accept the automatic choice for this moment and go to the next |
| `C` | compare the burst side by side |
| `Z` / `Shift+Z` | 100 % on the detected subject / same from a real RAW decode |
| `Enter`, `G` | loupe, grid |
| `B` | blind mode (hides suggestions) |
| `Ctrl+Z` | undo |
| `?` | all shortcuts |

Dashed outline = suggested by the machine, solid = your decision. Yours always wins.

Then **Export…** in the UI, or:

```bash
culler export /path/to/event --dry-run
```
```bash
culler export /path/to/event --keepers /path/to/keepers
```

In Lightroom: select the photos → *Metadata → Read Metadata from Files*. Picks
arrive with the label **Green**, rejects **Red** (Lightroom cannot read
pick/reject flags from files), stars as stars.

Other commands: `culler process FOLDER` (headless), `culler eval FOLDER`.

## Measuring agreement with your own taste

1. Open an event, press `B` (blind mode) so suggestions cannot bias you.
2. Cull a few hundred frames normally with `P` / `X`.
3. Run:

```bash
culler eval /path/to/event --sweep --labels labels.csv
```

It reports how often the automatic #1 frame is one of your picks (against the
random baseline), how many auto-rejects you actually kept, the bursts where
you disagreed, and whether different weights would have done better. Tunables
live in `culler/config.py`; override with `--config my.json`.

## More

* `docs/DESIGN.md` – architecture, scoring pipeline, data model, known risks.
* `scripts/make_demo.py` – builds a synthetic "event" from ordinary photos.
* Cache location: `~/.cache/mm-photo-culler` (Linux), `~/Library/Caches/…`
  (macOS), `%LOCALAPPDATA%` (Windows); override with `MMCULL_CACHE`. Safe to delete,
  except that it holds your not-yet-exported decisions.

```bash
pytest
```
