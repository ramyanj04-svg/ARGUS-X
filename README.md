# ARGUS-X — Track the Untrackable

Long-horizon multi-object identity recovery under camera motion, occlusion and
appearance drift. Built for the A05 Advanced Computer Vision brief: detection,
motion estimation, data association, identity memory, re-identification and
confidence-aware track management, with an explicit bias against false identity
matches.

---

## The problem, precisely

A detector plus a tracker ID is not an identity. As soon as an object is
occluded for longer than the tracker's buffer, the tracker throws the ID away —
the object comes back as a stranger. Naively re-attaching the "closest
appearance" is worse: two people in similar clothes produce a false match, and a
false identity is harder to notice than a lost one.

ARGUS-X separates the two concerns:

| Layer | Lifetime | Purpose |
| --- | --- | --- |
| **Track** | seconds | motion hypothesis, Kalman filter, association |
| **Identity** | whole session | durable appearance gallery, timeline, recovery |

Tracks die. Identities don't.

---

## Run it

```bash
# dashboard (recommended)
python app.py                 # http://127.0.0.1:8000

# CLI on your own footage
python run_argus.py --video test.mp4 --max-frames 240 --out runs/argus_x.mp4

# labelled benchmark with CLEAR-MOT + IDF1
python run_argus.py --benchmark
```

Dependencies are in `requirements.txt` (Ultralytics, OpenCV, NumPy, SciPy,
PyTorch). A CPU is enough: ~3–6 fps at 1280 px with `yolo11n` at 512 px.

---

## Pipeline

1. **Camera compensation** — ORB + RANSAC estimates the scene translation each
   frame; every Kalman prediction is shifted into the current frame before
   matching. Without this, a pan looks like every object teleporting.
2. **Motion prediction** — per-track Kalman filter on `(cx, cy, aspect, height)`
   with damped velocity while the object is missing.
3. **Appearance embedding** — 6×12 and 8×16 Lab colour grids, a 16×6 HSV
   histogram and a gradient-orientation histogram, **each block standardised**
   before concatenation. Standardisation is what makes the descriptor usable:
   raw colour vectors are all positive and score ~0.92 cosine between different
   people, which no threshold can separate.
4. **Association** — two-stage gated Hungarian matching. High-confidence
   detections use `0.45·IoU + 0.55·appearance`; low-confidence detections require
   **both** strong motion and strong appearance. After a gap, motion is only a
   prediction, so appearance must confirm it.
5. **Identity memory** — EMA embedding plus a 12-crop gallery, quality-gated;
   identity state machine `ACTIVE → LOST → ARCHIVED` with a 900-frame horizon.
6. **Verified recovery** — five gates (class, time, motion/scale, adaptive
   *multi-cue* appearance, ambiguity margin over **every** known identity), a
   joint Hungarian assignment, an active-claim guard, and a 3-frame
   confirmation before the recovery is committed. Failure rolls the identity
   back to `LOST`. An unidentified new track may also wait (`identity_grace`)
   for a clean view before it mints a permanent ID, so an actor stepping out
   from behind an occluder claims the ID that is waiting for it instead of
   creating a duplicate.

An identity is only created after `n_init = 3` hits, so detector noise cannot
invent identities. Tracks that vanish for more than 12 frames are dropped
deliberately: a stale ghost box is exactly what steals the wrong person, so
long-gap re-acquisition must go through the strict recovery path instead.

### Why the ambiguity gate looks at *live* identities

A traced false recovery showed the hole: identity `AX-001` (actor in the striped
jacket) was handed a detection of a *different* actor while that actor's
double was still being tracked. The score was 0.787 for the lost identity and
0.780 for the live one — a lead of 0.007. Because the gate only compared against
other **lost** identities, nothing consulted the look-alike that was on screen.
The gate now scores a candidate against the whole identity memory, active or
not, and reports the losing rival in the decision log.

A second cause was the global cosine itself. Look-alike outfits share colour, so
the colour blocks and the histogram agreed while only the texture block
disagreed, and they outvoted it. The same pair now measures 0.79 fused versus
0.90 for the true actor once each cue block is compared separately: a 0.115
gap instead of 0.046. Recovery uses the fused score; frame-to-frame association
keeps the cheaper global cosine.

---

## Measured behaviour

`python run_argus.py --benchmark` renders a 300-frame scene with a sinusoidal
camera pan, a 120 px pillar producing 77–116 frame full occlusions, two
look-alike jackets in the same colour, 5 % detector dropout, 2.5 px box jitter
and random false positives. Both trackers receive *identical* detections.

| Metric | IoU baseline | ARGUS-X |
| --- | --- | --- |
| IDF1 | 0.754 | **0.931** |
| MOTA | 0.875 | **0.924** |
| Precision | 0.928 | **1.000** |
| Recall | **0.952** | 0.925 |
| ID switches | 3 | **1** |
| Unique identities created | 73 | **5** |

The baseline is not a strawman: because it never re-associates after a gap it
has no false matches, so it edges ARGUS-X on raw recall. ARGUS-X trades 2.7
points of recall for 7 points of precision, 5 points of MOTA, two thirds fewer
ID switches and **73 → 5** identities for 4 actors. Its two verified
re-acquisitions are a 132-frame gap (4.4 s) and a 100-frame gap, each of which
lands back on the correct actor — traced frame by frame against the labels, not
just counted.

All four ground-truth actors map to exactly one identity each for the whole
sequence. The fifth identity is a duplicate created for one actor that stayed
unidentified past its grace window; the tracker never mixes two actors inside
one identity, which is the property that matters here.

These numbers come from the synthetic sequence in `core/benchmark.py`; they are
not a claim about any public benchmark. The scene, the noise and the ground
truth are all generated locally and are reproducible with a fixed seed.

---

## Dashboard

`python app.py` serves a single-page mission-control view:

- **Annotated stream** — MJPEG; identities in colour, lost identities as dashed
  ghosts with a missing-frame counter, re-acquired identities in gold, recovery
  banner on the frame it happens.
- **Identity timeline** — presence segments per identity, gaps visible, dots at
  verified re-acquisitions.
- **Decision log** — every create / loss / recovery / rejection with confidence
  and reason (including "ambiguous match" and "object already claimed").
- **Live camera** — start the webcam and the same engine runs on live frames:
  boxes, identity labels, lost ghosts and the HUD are drawn in real time.
- **Benchmark panel** — the table above, computed live.

API: `POST /api/run`, `POST /api/upload`, `GET /api/job/<id>`,
`GET /api/benchmark`, `GET /media/<job>/stream.mjpg`,
`POST /api/camera/start`, `POST /api/camera/stop`,
`GET /api/camera/status`, `GET /media/camera/stream.mjpg`.

---

## Layout

```
app.py                 launcher (server + browser)
server.py              stdlib HTTP server, job runner, MJPEG stream
run_argus.py           CLI
core/config.py         every tunable in one dataclass
core/detector.py       YOLO front-end -> Detection
core/motion.py         Kalman filter, IoU, camera motion estimation
core/appearance.py     standardised multi-cue embedding + quality score
core/association.py    gated Hungarian matching
core/memory.py         identities, galleries, timeline, events
core/reid.py           margin-aware appearance matching
core/recovery.py       five-gate recovery proposals
core/tracking.py       the per-frame engine
core/metrics.py        MOTA / IDF1 / ID switches / fragmentation
core/benchmark.py      labelled synthetic sequence + IoU baseline
core/pipeline.py       video -> annotated video + JSON report
core/live.py           webcam capture thread -> annotated MJPEG frames
```

## Known limitations

- Appearance is a handcrafted descriptor, not a learned ReID network. It is fast,
  explainable and CPU-friendly, but on dense crowds with near-identical clothing
  the margin gate will refuse matches rather than guess — by design. The
  benchmark's look-alike pair is the measured example: genuine return 0.90,
  impostor 0.79, threshold 0.80 plus drift, so the impostor is refused by a
  margin of ~0.01–0.05 and the cost is a duplicated identity rather than a
  swapped one.
- A new identity waits up to `identity_grace = 40` frames for a cleaner view
  while a plausible lost identity exists; those frames are shown as provisional
  boxes and count as false negatives in the metrics.
- Recovery is bounded by the identity archive horizon (900 frames); a longer
  absence starts a new identity.
- The report is per-process; there is no database, so jobs are not persisted
  across server restarts.
