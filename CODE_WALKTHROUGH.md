# Gauge — Technical Walkthrough for Developers

This document walks through the actual pipeline code end-to-end: what each
stage does, why it's built this way, and the real functions/snippets from
`backend/pipeline.py` (the only file that matters for the CV logic).
High-level submission context (AI tools used, reused vs. custom code, test
evidence, cost) is in `DELIVERY_NOTES.md`; this doc is the "how does it
actually work" companion to that.

## 1. The pipeline in one picture

```
photo (JPEG/HEIC upload)
   │
   ▼
check_exposure() + blur_score()        ─── cheap whole-photo gates, before
   │ ok                                    any detection work is attempted
   ▼
detect_and_rectify()                    ─── find the 20mm ArUco marker,
   │ rectified image + marker_rect          homography-warp to a fixed
   ▼                                        15 px/mm scale
locate_outer_ring_circle()              ─── Hough circle, coarse ROI only —
   │ (cx, cy, r) hint                       never trusted as the measurement
   ▼
build_band_hypotheses()                 ─── the hint could be the ring's
   │ [(label, inner_r, outer_r), ...]        OUTER edge or its INNER hole
   ▼                                         edge — try both readings
segment_ring_body_mobilesam()           ─── prompt MobileSAM per hypothesis:
   │ per hypothesis: body mask               "here's the band material,
   ▼                                         here's the hole, here's outside"
analyze_mask_topology()                 ─── read the hole from the mask's
   │ per hypothesis: ok/reject + ellipse     OWN contour hierarchy, gate on
   ▼                                         solidity + axis ratio
pick winner (higher solidity)           ─── classical ellipse fit → mm via
   │                                         the ArUco-calibrated scale
   ▼
diameter_mm, ring_size, overlay_img
```

Two things to notice about this shape, because they're the two decisions
the rest of the codebase review usually asks about:

- **ArUco → rectify → Hough is pure geometry.** It never changes across
  revisions and never decides what the measurement is — it only says
  "the ring is roughly here, roughly this big."
- **Everything after that is "which pixels are ring material, and what's
  the hole" — answered by a promptable segmentation model (MobileSAM),
  not by classical thresholding.** The diameter itself is still a
  classical `cv2.fitEllipse` on that mask's own contour. The model never
  estimates a size; it only answers a segmentation question.

## 2. Calibration and scale

```python
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())

MARKER_ID = 0
MARKER_MM = 20.0
PX_PER_MM_OUT = 15.0
```

`find_marker_corners()` is the single source of truth for locating the
marker by exact ID — used both by `detect_and_rectify()` and any
diagnostic code, specifically so two code paths can't silently disagree
on which marker was found (this actually happened once — see the comment
in the code).

`detect_and_rectify()` then:
1. Rejects a marker that's only partially in frame, too small in pixels
   (`MARKER_TOO_SMALL`, a corner-localization-noise floor) or too small
   relative to a *separate*, higher floor for pixel density
   (`RESOLUTION_TOO_LOW` — this one exists because a downscaled copy of an
   already-passing real photo reproduced a failure the original never had).
2. Rejects too steep a shooting angle (`side_ratio` of the marker's own
   warped side lengths vs. `PERSPECTIVE_TOO_HIGH`).
3. Computes a homography from the marker's 4 corners to a fixed
   20mm × 15px/mm square, then **dynamically sizes the output canvas** by
   projecting the whole source photo through that homography first — a
   fixed canvas size silently clipped the ring on photos where it sat
   further from the marker than expected.

Everything downstream works in this rectified, fixed-scale image, so "pixels
÷ 15" is always mm from this point on.

## 3. Coarse localization — `locate_outer_ring_circle()`

A `cv2.HoughCircles` pass over the rectified grayscale image (with the
marker's own region blanked out so its square edges can't be picked up as
a circle). This is explicitly **not** the measurement:

```python
"""
Locate the ring region using a strong approximately circular edge.

This is localization only: the returned circle is NEVER used as the
measurement or as evidence for which boundary is correct - it only
tells the rest of measure_ring roughly where to look...
```

The real subtlety documented here: on a clean, high-contrast photo, Hough
can lock onto the ring's **inner hole edge** just as easily as its outer
edge — there's no way to tell which one it found from the circle alone.
That ambiguity is resolved one step later, not here.

## 4. The two-hypothesis strategy — `build_band_hypotheses()`

Rather than guessing whether the Hough hint is the outer or inner edge,
the pipeline tries **both readings**:

```python
# H1: outer_ring_hint IS the outer edge; search inward for the hole.
inner_cand = radial_inner_candidate(rect_gray, probe, (cx, cy), r, ...)
if inner_cand is not None:
    hypotheses.append(("r=outer", inner_cand["diam_px"] / 2.0, r))

# H2: outer_ring_hint IS the hole edge; search outward for the band.
outer_cand = radial_outer_candidate(rect_gray, probe, (cx, cy), r)
if outer_cand is not None:
    hypotheses.append(("r=inner", r, outer_cand["radius"]))
```

Both radial searches use the same trick (`_radial_edge_histogram`): build a
histogram of Canny edge pixels by radius, but **only count edge pixels
whose local gradient points radially** (toward/away from the candidate
center). This one check is why the pipeline survives textured backgrounds
— wood grain produces plenty of Canny edges at every radius, but its
gradient direction is set by the grain pattern, not by any candidate
ring center, so it gets filtered out before the histogram is built.

Each hypothesis that comes back with a plausible `(inner_r, outer_r)` pair
goes on, independently, to segmentation. Neither, one, or both can survive
this step.

## 5. Segmentation — `segment_ring_body_mobilesam()`

This is the part that changed most recently. For each hypothesis, the
pipeline builds point prompts (`build_ring_prompts`) — positive points
around the band's own midpoint radius, negative at the hole center,
negative well outside the ring — and runs MobileSAM on a crop around the
hypothesis's window:

```python
points, labels = build_ring_prompts(lcx, lcy, inner_r, outer_r, win_r)
results = sam_model.predict(crop, points=[points], labels=[labels],
                             device="cpu", verbose=False)
```

The model is told "this is what the ring band looks like, this is clearly
not the ring" and decides the actual boundary itself — it's a prompt, not
a classification rule. One binarization detail worth knowing about if you
touch this function: `cv2.resize`'s interpolation leaves non-binary values
along the mask boundary, which `cv2.findContours` otherwise treats as
foreground, silently growing the mask. The re-threshold after resize fixes
that:

```python
m_resized = cv2.resize((m * 255).astype(np.uint8), (crop.shape[1], crop.shape[0]))
_, m_resized = cv2.threshold(m_resized, 255 // 2, 255, cv2.THRESH_BINARY)
```

**Why MobileSAM and not something simpler.** Three approaches were
compared head-to-head on the real test photos, judged primarily by
*inspecting the resulting mask*, not by which one happened to land closer
to a known diameter:

| Approach | Result |
|---|---|
| Properly-seeded GrabCut (same geometric priors as the two hypotheses) | Topologically clean on 2/3 real photos, but on the third — a ring with a strong internal reflection gradient — it produced a mask that was *still topologically clean by every structural check* yet visibly wrong: the color-GMM classified the bright half of the true opening as ring material. Clean topology is necessary but not sufficient. |
| Prompted FastSAM | Returned no mask for every hypothesis on every photo — investigated directly and found its `.prompt()` only *filters* a handful of pre-generated "segment everything" masks, it doesn't generate a new mask conditioned on the prompt. Not a real promptable segmenter for this task. |
| MobileSAM | Visually complete, correctly-shaped hole boundaries on all 3 real photos, including the reflective-ring case that defeated GrabCut. |

Full writeup with the visual comparison is in `DELIVERY_NOTES.md`.

## 6. Deciding what the hole is — `analyze_mask_topology()`

This is the core idea of the current architecture: the hole isn't found by
color-matching anything, it's read off the **mask's own contour
hierarchy**.

```python
cnts, hier = cv2.findContours(fg_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
...
body_idx = max(top_level, key=lambda i: cv2.contourArea(cnts[i]))   # largest top-level contour = ring body
...
hole_idx = max(children, key=lambda i: cv2.contourArea(cnts[i]))   # largest contour whose PARENT is the body = hole
```

The hole must then pass a validity gate before its ellipse is trusted:

```python
MIN_HOLE_SOLIDITY = 0.85
MIN_HOLE_AXIS_RATIO = 0.7

ok = (n_bodies == 1 and n_holes == 1
      and solidity >= MIN_HOLE_SOLIDITY
      and axis_ratio >= MIN_HOLE_AXIS_RATIO)
```

`solidity` is hole area ÷ convex-hull area of that *same* hole contour —
this is the GrabCut-failure-mode defense: an incomplete or bitten crescent
scores low here even if it would look "clean" by contour count alone. The
convex hull is used **only for this check**. The actual ellipse fit always
runs on the raw hole contour:

```python
ellipse = cv2.fitEllipse(hole_contour)   # NOT cv2.fitEllipse(hull)
```

So a partial mask can fail the gate, but it can never be silently
hull-reconstructed into a convincing wrong measurement.

## 7. Picking the winner — `measure_ring()`

```python
valid = [(label, inner_r, outer_r, topo) for ... if topo["ok"]]
...
if not valid:
    res.reason = "HOLE_NOT_SUFFICIENTLY_VISIBLE"   # honest retake, not a guess
    return res

winner = max(valid, key=lambda t: t[3]["solidity"])
...
if len(valid) == 2:
    res.detection_spread_mm = abs(diam_mm - other_diam_mm)  # agreement signal, not gating
```

If neither hypothesis's mask passes the topology gate, the API returns a
structured retake reason — never a number the mask itself doesn't support.
If both hypotheses pass, the higher-solidity one wins and the other's
diameter is reported as `detection_spread_mm` — an agreement signal
between two independent readings of the same photo, explicitly **not** a
calibrated measurement-uncertainty estimate (it says nothing about printer
scale error, lens distortion, or ArUco corner error — see README.md).

## 8. Running it yourself

```bash
cd backend
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python3 -c "from ultralytics import SAM; SAM('weights/mobile_sam.pt')"   # fetch checkpoint once
uvicorn main:app --host 0.0.0.0 --port 8000
```

Set `RING_DEBUG=1` for a per-hypothesis console trace (radii, solidity,
axis ratio, accept/reject reason) or `RING_DEBUG_OVERLAY=1` to also get the
failure reason burned into the returned debug image. The regression suite
(`python3 backend/regression_suite.py`) runs every labeled test photo
(real + synthetic) through `measure_ring()` and checks both the expected
accept/reject outcome and, where a trustworthy ground truth exists, the
error tolerance — and separately statically scans this file for any
ground-truth value or test-photo name leaking into the detection logic
itself (see the module docstring's "no-overfitting policy").

## 9. API contract

`POST /api/measure` — multipart form, field `file` = the photo.

```json
{
  "status": "success",
  "diameter_mm": 16.8,
  "ring_size": "6.5",
  "ring_size_in_range": true,
  "detection_spread_mm": 0.01,
  "family_estimates": {"r=outer": 16.79, "r=inner": 16.80},
  "processing_time_ms": 6408.7,
  "overlay_image": "data:image/jpeg;base64,..."
}
```

or, on a rejected photo:

```json
{
  "status": "retake",
  "reason": "HOLE_NOT_SUFFICIENTLY_VISIBLE",
  "message": "...",
  "processing_time_ms": 340.2,
  "overlay_image": null
}
```

Reason codes worth knowing when reading logs: `MARKER_NOT_FOUND`,
`MARKER_PARTIAL`, `MARKER_TOO_SMALL`, `RESOLUTION_TOO_LOW`,
`PERSPECTIVE_TOO_HIGH`, `UNDEREXPOSED`, `OVEREXPOSED`, `BLUR`,
`RING_NOT_FOUND`, `NO_BAND_HYPOTHESIS`, `SEGMENTATION_MODEL_UNAVAILABLE`,
`HOLE_NOT_SUFFICIENTLY_VISIBLE`. Each maps to a specific, user-facing retake
message in `main.py`'s `RETAKE_MESSAGES`.

## 10. Where to look for more detail

- `backend/pipeline.py`'s module docstring — the full history of why
  revisions 1-4 were each dropped, in more depth than section 5 above.
- `README.md` — architecture summary, API examples, known limitations.
- `DELIVERY_NOTES.md` — AI tools used, reused vs. custom code, the full
  test evidence table, dev time, processing-time numbers, and concrete
  examples of how AI-generated output was independently verified (not just
  trusted).
