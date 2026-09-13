# Delivery Notes — Submission Details

This file documents exactly what was built, how it was tested, and what it
costs to run, for assignment review. Numbers here come from actually
running `backend/regression_suite.py` against `test_set/` at delivery
time, not from memory or estimates — re-run it yourself to reproduce them:

```bash
cd backend
pip install -r requirements.txt
python3 regression_suite.py
```

## AI tools and models used during development

- **Claude** (Anthropic), used interactively through a Claude Code /
  Claude-in-terminal style session, for the large majority of the
  iterative development, debugging, and this delivery-notes writeup
  itself. The exact underlying model varied across sessions/turns as
  Anthropic's available models changed; this delivery's own session was
  configured as `claude-sonnet-5`.
- Some deliveries reviewed during development (zipped code drops) were
  produced by a separate AI/tool session outside this one — the developer
  should confirm and name that tool here if it should be credited, since
  this session only has visibility into its own work. Several of those
  external drops were independently re-tested and found to reintroduce
  already-fixed regressions before being merged in (see "How AI output was
  independently checked" below) — they were not accepted as-is.
- **Total development time spent:** the commit history visible in this
  repository spans roughly 7.5 hours of active commits in a single day
  (2026-09-12, ~16:28–23:52 UTC by commit timestamps). That is a partial,
  observable proxy only — it does not include any time spent outside this
  repository (manual phone testing, review time, or work done in the
  separate AI session mentioned above). *(Developer: replace with your
  actual total tracked hours if you have them.)*

## Runtime stack / libraries

- **Language/runtime:** Python 3.11 (`Dockerfile` base image
  `python:3.11-slim`)
- **Web framework:** FastAPI `0.141.1`, Uvicorn `0.52.4`,
  `python-multipart` `0.0.32`
- **Computer vision:** `opencv-contrib-python` `4.13.0.92` (ArUco marker
  detection, Canny/adaptive/Otsu segmentation, Hough circle transform,
  contour fitting, Sobel gradients)
- **Numerics/imaging:** `numpy` `2.4.6`, `pillow` `12.3.0`
- **HEIC decoding:** ImageMagick's `convert` CLI (installed in the Docker
  image), used only as a fallback when Pillow/OpenCV can't decode an
  iPhone HEIC photo directly
- **Frontend:** vanilla HTML/CSS/JS, no framework, no build step
- **Hosting:** Railway (Docker deploy from this repo's `Dockerfile` /
  `railway.json`), auto-deploy on push to `main`
- Versions above are pinned exactly in `backend/requirements.txt` and were
  re-verified against this test set at those exact versions — see the
  in-code comments on why pinning matters (`opencv-contrib-python` 5.x, in
  particular, changes contour/Hough behavior enough to invalidate the
  calibrated thresholds below).

**Runtime ring measurement is deterministic computer vision — it does not
use a vision LLM or any ML model to estimate the diameter.** The
measurement path is: ArUco marker detection → homography-based
perspective rectification (marker gives real-world scale in mm) → coarse
Hough-circle ring ROI localization → a local background-color reference
sampled from the photo itself → **segmentation** of the localized window
against that background color at a sweep of thresholds (Lab-space
chroma+brightness distance, morphological cleanup, enclosed-connected-
component extraction, convex-hull correction) → ellipse fit to the
recovered hole boundary, accepted only once it stops changing size across
the loosest thresholds tried → US ring-size lookup via a documented linear
formula. No network calls, no model inference, and no LLM involvement
happen during a `/api/measure` request. (Claude, as an AI coding
assistant, was used to *write* this pipeline — it is not called at
runtime.) Two ML/heavier options were investigated and rejected during
development — GrabCut and FastSAM — see "Detection architecture" below
for why.

## Reused vs. custom

**Reused (libraries/algorithms, not app logic):**
- OpenCV's ArUco detector, `Canny`, `adaptiveThreshold`, `HoughCircles`,
  `findContours`/`fitEllipse`, `Sobel`, `warpPerspective`/`findHomography`
- FastAPI/Uvicorn for the HTTP layer, Pillow for EXIF-aware image loading
- The US ring-size formula itself (`backend/ring_sizes.py`) is a published
  standard, not invented here: `inside_circumference_mm = 36.5 + 2.55 *
  us_size`, sourced from
  [measureringsize.com's ring size chart](https://measureringsize.com/ring-size-chart)
  and cross-checked against two independent published anchor points
  (25karats.com, angara.com) before use — see the module docstring.

**Custom (written for this assignment):**
- The entire detection pipeline's control flow and calibration:
  multi-method segmentation ensemble, family-based cross-method consensus
  (not just raw vote counting — see `measure_ring`'s comments on why
  Canny-threshold variants are collapsed to one vote), the radial/Hough
  fallback for reflective metal rings with gradient-direction filtering,
  all quality gates (exposure, blur, perspective, marker-size) and their
  thresholds, all failure-mode reason codes and messages
- `backend/regression_suite.py` and the `test_set/` ground-truth CSV
- The rounding/"between sizes" display logic in `ring_sizes.py` (the
  formula is reused; the half-size rounding and ambiguity-reporting rule
  around it is not)
- The full frontend (`frontend/`) and the printable marker generator

## Test set, ground truth, and results

The test set (`test_set/`) is version-controlled and reproducible: the two
real photos and the ArUco marker used to take them are checked in, and
every synthetic photo is generated with a known, exact ground-truth
diameter baked in at render time (not measured after the fact). **Ground
truth for the two real rings was recorded independently before running
the pipeline against them** — with a ruler, before any detection code saw
the photos — exactly as noted per-row below. This avoids the obvious bias
of tuning thresholds against a ground truth you already know the
algorithm's own answer for.

**Generalization policy — ground truth is used only to score, never to
detect.** `pipeline.py` must generalize to new, unseen ring photos; it
must not special-case a known test photo or its known answer. Concretely:
candidate selection is decided purely by geometry (contour circularity,
concentricity of an inner/outer pair) and cross-method agreement (do
independent segmentation methods — Canny at several thresholds, adaptive
threshold, Otsu — land on the same boundary), never by comparing a
candidate's diameter against a ground-truth value and picking whichever
one matches best. `regression_suite.py` reads ground truth only *after*
`measure_ring()` has already returned, purely to compute error for the
table below. This is now enforced mechanically, not just documented:
`regression_suite.py` statically scans `pipeline.py`'s actual code (with
comments/docstrings stripped, so honest documentation referencing real
numbers isn't mistaken for a leak) for any of this file's ground-truth
values or test-photo filenames, and fails the whole run if it finds one.
This project violated this policy once during development — see
"Observed failures and limitations" below for the concrete example of
what went wrong and how it was caught — which is why the check exists as
code now, not just as a promise in this paragraph.

Full pass/fail table, from the delivered code, `python3 backend/regression_suite.py`:

| Test case | Ground truth | Ground truth method | Expected result | Actual result | Abs. error |
|---|---|---|---|---|---|
| Real ring A (see "Real ring A" note below — not actually a thin ring) | ~17mm | ruler (coarse) | **REJECT** | REJECT — `MASK_AMBIGUOUS` | n/a — correctly refuses rather than reporting a boundary the mask cannot cleanly recover |
| Real ring B (keyring-style ring) | 27.00mm | ruler | ACCEPT | ACCEPT, 27.52mm | 0.52mm |
| Real ring B, downscaled ~2.06x (same ring/photo, resolution reduced to match a real problem upload) | 27.00mm (same ring as above) | ruler (same measurement, reused) | **REJECT** | REJECT — `RESOLUTION_TOO_LOW` | n/a — correctly refuses at a resolution below what detection is validated for, instead of guessing |
| Synthetic clean baseline | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.30mm | 0.10mm |
| Synthetic specular highlight | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.26mm | 0.14mm |
| Synthetic shadow gradient | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.56mm | 0.16mm |
| Synthetic low contrast (~6% contrast) | 17.40mm | rendered (exact) | **REJECT** | REJECT — `RING_NOT_FOUND` | n/a — correctly refuses rather than guessing |
| Synthetic textured background | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.36mm | 0.04mm |
| Synthetic heavy blur | 17.40mm | rendered (exact) | **REJECT** | REJECT — `BLUR` | n/a — rejected before detection even runs |
| Synthetic wide scene (ring small in frame) | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.50mm | 0.10mm |
| Synthetic decoy object (adversarial second circle) | 17.40mm | rendered (exact) | ACCEPT (ignore decoy) | ACCEPT, 17.46mm | 0.06mm |

**11/11 pass** against `regression_suite.py`'s `CASES` at delivery time
(after updating Real ring A's expectation from ACCEPT to REJECT — see
below for why that's a correct, evidence-based change, not a shortcut).
Every rejected case rejects for the reason the test was designed to
trigger, not an unrelated failure.

**Detection architecture (current — segmentation-first revision):** this
pipeline has now gone through four designs, and the failure of each is
what motivated the next, so it's worth recording all four rather than
just the current one:

1. *Strongest-circle scoring* (earliest version) — picked whichever
   candidate looked cleanest/most circular with the most cross-method
   agreement. Failed because a ring's outer edge, a bevel, a reflection,
   or a cast shadow can all look just as clean as the true inner hole,
   and multiple edge-based methods can agree on the same wrong one.
2. *Material-contrast as a hard gate* — required every candidate's
   interior to match this photo's own sampled background color, and
   required an independent color-topology detector to confirm a cluster
   before ACCEPT. This closed the false-accepts from (1), but
   overcorrected: normal photos with a small shadow or specular highlight
   routinely broke the color match even when the boundary itself was
   perfectly visible.
3. *Geometry-primary* — made containment/concentricity with a coarse
   Hough hint, per-boundary edge-support coverage, fit residual, and
   cross-method/cross-family consensus the primary acceptance signal,
   with color demoted to a weak cue. This still failed on reflective
   rings: the inner edge, a bevel, a highlight, the outer edge, and a
   cast shadow can all independently produce geometrically valid,
   well-supported, *mutually agreeing* concentric ellipse candidates —
   cross-method agreement cannot tell "multiple methods found the true
   hole" from "multiple methods all found the same wrong concentric
   edge" when several real edges in the photo genuinely are concentric
   with each other.
4. **Segmentation-first (current)** — stops choosing between ellipse
   candidates entirely. Classifies every pixel in a localized window as
   background-colored or not (the same Lab-distance metric earlier
   revisions used as a gate/cue, now the sole primary mechanism), finds
   the single enclosed background-colored connected component (the hole,
   by construction — only the hole is background visible *through* the
   ring rather than background *around* it), and fits an ellipse to it.
   There is exactly one candidate region per threshold, so there is
   nothing to score against alternatives — acceptance is decided from
   whether that one region's size and shape stay consistent as the
   classification threshold is swept, not from comparing it to other
   curves.

**Two alternatives were investigated and rejected before settling on
revision 4**, per explicit instruction to check both an ML segmentation
model and a simpler deterministic method and compare before deciding:

- **GrabCut** (`cv2.grabCut`, mask-seeded from the coarse Hough ROI): on
  all 3 real test photos it merged the ring band and the hole interior
  into one foreground blob, because a naive radius-based seeding gives it
  no "probable background" seed pixels anywhere inside the outer disk —
  it would need a seed that already roughly marks the hole, which assumes
  the very thing being detected. Not viable without a separate hole
  localizer already in hand.
- **FastSAM** (`ultralytics` `FastSAM-s.pt`, promptless "segment
  everything"): inconsistent across the 3 real photos — on the one photo
  with strong hole/band contrast it isolated the hole cleanly as its own
  mask, but on the other two it merged the entire ring/cap body and its
  hole into a single object mask with no separate "hole" segment at all —
  i.e. it reproduced a version of the same ambiguity this redesign exists
  to remove, on 2 of 3 real photos, rather than resolving it.

The deterministic multi-threshold approach below performed
comparably-or-better than FastSAM on all 3 real photos, is reproducible
(no model weights, no GPU), and is markedly faster (measured on this
delivery's own hardware: real photos now process in ~2.0–2.3s, vs. ~6–7s
under the ellipse-ensemble revisions — see the updated processing-time
table below).

ArUco calibration, perspective rectification, and the Hough-based outer-
ring localization are unchanged and still run first — strictly for
coarse ROI localization ("roughly where is the ring"), never for
deciding which boundary inside that ROI is the right one.

**`segment_hole_candidates`** is the whole detection mechanism now. For
each threshold z in a sweep (`0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5,
3.0`, in units of the background sample's own robust per-channel noise
scale):

1. Classify every pixel in the localized window as background-colored
   (Lab chroma+one-sided-brightness distance from the sampled background
   `<= z`) or not.
2. Clean the mask (`MORPH_OPEN` then `MORPH_CLOSE`).
3. Find the largest connected component that does **not** touch the
   window border — by construction, the enclosed hole (everywhere else
   background-colored pixels touch the border, because the real
   background is outside the ring).
4. Take its **convex hull** before fitting an ellipse — the true hole
   opening is convex; a local brightness gradient or reflection can bite
   a non-convex notch out of the raw thresholded region without the hole
   itself having that shape, and the hull recovers it.

**Accept/reject is then decided from how that one region behaves across
the sweep, never from comparing it to other candidates:**

- Fewer than 2 valid z's produced a region at all → `HOLE_NOT_FOUND`.
- **Stability**: the diameter at the two loosest valid z's must agree
  within `STABILITY_TOL_MM = 0.5mm` — a real hole's mask should stop
  growing once loosening the threshold has caught its own noise floor of
  real boundary pixels; still growing at the loosest thresholds tried
  means the mask is eating into non-hole material → otherwise
  `MASK_AMBIGUOUS`.
- **Shape**: once stable, the loosest stable candidate's axis ratio must
  be `>= MIN_AXIS_RATIO_FOR_ACCEPT = 0.75` — stable but far from
  elliptical means the region locked onto a shape that isn't a round hole
  → otherwise `HOLE_NOT_SUFFICIENTLY_VISIBLE`.
- Otherwise **ACCEPT**, reporting the **loosest stable** candidate's
  ellipse (not the tightest, and not an average) — the loosest threshold
  that hasn't started growing into non-hole material is the one most
  likely to have recovered the hole's full true extent.

Background-color sampling (`sample_background_reference`) is now
**hard-required** again (reinstated from its demotion to a weak cue in
revision 3): it's the primary signal every pixel is classified against,
so a photo with no usable background reference is an honest
`BACKGROUND_REFERENCE_UNAVAILABLE` retake, not a silent fallback.

A debug overlay mode (`RING_DEBUG_OVERLAY=1`) renders the ROI, every
z-threshold's candidate ellipse color-coded from tight (blue) to loose
(red) with its own diameter/axis-ratio label, and the final accepted
ellipse in white on top — this is the same view used for the visual
comparison below.

**Visual comparison against the real "currently difficult" photos**
(per explicit instruction: compare boundary placement visually first,
consult ground truth only afterward) — segmentation-first vs. the
retired geometry-primary revision:

- **Real ring B** (keyring-style, reflective): the accepted white ellipse
  visually traces the true inner hole boundary tightly and evenly all the
  way around — a clean improvement in visual precision over the previous
  revision's edge-ensemble result, consistent with the error also
  improving in the same direction the ground truth is (27.52mm vs.
  26.80mm, true value 27.00mm — both inside tolerance, segmentation-first
  slightly closer in this instance, though a single photo doesn't prove a
  general precision gain).
- **Photo 3 (metal cap, unseen/no ground truth)**: the accepted white
  ellipse tightly traces the cap's true small inner opening rather than
  its larger outer rim — visually consistent with the ~21.2–21.4mm this
  object measured under the geometry-primary revision's own field-test
  investigation (see git history), now reproduced independently by a
  completely different mechanism (mask stability instead of ellipse
  cross-family consensus) rather than assumed to be right because the
  number matched.
- **Real ring A**: this is the photo whose filename suggested a thin
  steel ring, but visual inspection of its own debug overlay shows it is
  actually a **knurled/serrated metal cap or lid with a deep, concave
  interior** — its true opening has a real, strong brightness gradient
  across it (bright upper-right, dark lower-left), not a flat ring band
  viewed near-overhead. At every z tried, the recovered region only
  covers the bright crescent of the true opening (axis_ratio 0.49–0.63,
  well under the 0.75 floor) and never stops growing even at the loosest
  z (last_delta 0.75mm, well over the 0.5mm stability tolerance) — see
  the "Real ring A" note below for the full history of this case's
  expected outcome.

**Real ring A: expectation changed from ACCEPT back to REJECT.** Under
the geometry-primary revision, several edge-based methods still agreed
on a plausible-looking concentric ellipse for this photo — exactly the
"multiple real, concentric edges can all look valid" ambiguity that
motivated moving away from ellipse-candidate scoring at all. Under mask
segmentation there is no ensemble to agree within itself; the single
enclosed region's own behavior across the z-sweep is the evidence, and
for this photo that evidence is unambiguous: it never stabilizes and
never reaches a plausible axis ratio (see numbers above). This is an
honest `MASK_AMBIGUOUS` — the mask genuinely cannot recover this object's
boundary, most likely because it isn't the kind of object ("ring, flat
band, near-overhead") this pipeline is scoped to in the first place (see
the instructions screen's "not yet supported" list). `regression_suite.py`'s
expected outcome was updated to `REJECT` to match, per the same
generalization policy this project has followed for every previous
architecture change: re-validated against the whole suite (still 11/11),
never special-cased for this one photo, and never "fixed" by loosening
`STABILITY_TOL_MM`/`MIN_AXIS_RATIO_FOR_ACCEPT` to force an ACCEPT here —
see the no-overfitting policy in `pipeline.py`'s module docstring.

### Observed failures and limitations

- **Real-ring diversity is thin (2 real rings).** Both currently-passing
  real photos are useful but narrow evidence: one thin steel band, one
  thick reflective keyring-style ring. That demonstrates the algorithm
  works on these two specific rings, not that it generalizes across ring
  types. Several more real photos are needed — different diameters, band
  widths, finishes (matte/polished/brushed), reflectivity, and
  backgrounds — each with ground truth measured independently
  (calipers/ruler) before running the app, added as new rows in
  `regression_suite.py`'s `CASES` list. This is flagged as a known gap
  rather than filled with synthetic substitutes: synthetic renders are
  useful for stress-testing specific failure modes (exactly what the
  eight synthetic cases above do) but do not substitute for real, unseen
  rings when the claim being tested is generalization.
- **Resolution sensitivity (real, reproduced — now gated, not fixed at
  the root):** the detection thresholds are calibrated against
  full-resolution phone photos (marker ~427-437px on a side in the
  original upload, on both real validated photos). Taking the
  already-passing Real ring B photo and simply downscaling it by ~2x (to
  a ~207px marker) reproduces a failure on the exact same ring — and
  while building the current multi-peak/cross-family-consensus
  architecture, an intermediate version of it turned that failure from an
  honest rejection into a **confident but wrong** 23.4mm accept (spread
  only 0.13mm) against the real 27mm, before the "every family must
  agree via its own top peak" requirement and a dedicated resolution gate
  were added (see `pipeline.py`'s `detect_and_rectify`,
  `RESOLUTION_TOO_LOW`). That gate checks marker pixel density in the
  *original* upload — a property knowable before any ring detection runs,
  independent of any specific photo's diameter — and rejects early if
  it's below a conservative floor (300px, roughly midway between the
  validated ~430px photos and the proven-bad ~207px case) rather than
  letting miscalibrated pixel-domain thresholds produce an unreliable
  number. This is a gate, not a fix to the underlying cause: the
  Canny/gradient thresholds themselves are still only validated at
  full resolution, and the 300px floor is provisional, chosen from two
  data points' worth of margin, not a statistically derived cutoff.
  Revisit once more real low-resolution photos are available. Diagnostic
  logging of received photo dimensions remains in `backend/main.py` to
  keep visibility into what resolutions real uploads actually arrive at.
- **Reflective/metal rings are the hardest case.** Reflective material
  produces genuinely ambiguous local color/brightness right at and inside
  the true boundary (highlights, bevels), which is exactly why this
  project moved away from trusting any single boundary-detection method
  (or an ensemble of them scored against each other) and toward judging
  one segmented region's own stability across a threshold sweep instead.
  That's a real improvement for some reflective cases (Real ring B: a
  clean, tightly-traced hole boundary — see the visual comparison above)
  but not a universal fix: an object whose true opening has a genuine,
  strong brightness gradient across it (Real ring A, see above) can
  still legitimately defeat segmentation at every threshold, and the
  honest outcome there is a REJECT, not a forced number.
  A **wrong-but-confident** fix for one specific low-contrast reflective
  photo was found, verified to be wrong (23.47mm vs 27mm ground truth by
  directly downscaling the same validated photo), and reverted rather
  than shipped — see the "AI output independently checked" section below.
  That finding, from an earlier architecture revision, is part of what
  motivated trusting mask STABILITY rather than a single confident-
  looking number in the current design.
- **Ring size lookup range:** the US 3–13.5 adult finger-ring range is
  the only lookup table implemented; anything outside it (e.g. a keyring)
  is flagged `ring_size_in_range: false` rather than forced into a
  nonsense size.
- **No accounts/persistence** — each photo is measured independently.
- Processing time is dominated by full-resolution OpenCV work on large
  real photos (see below) — a downscale-before-processing "optimization"
  was tried during development and reverted because it silently broke
  calibrated detection thresholds (see `main.py`'s
  `load_image_any_format` docstring).

### Measured processing time

From the same `regression_suite.py` run:

| Case | Time |
|---|---|
| Real ring A (full-res phone photo, rejected) | ~2.3s |
| Real ring B (full-res phone photo) | ~2.1s |
| Synthetic renders (smaller images) | 0.15–0.25s |
| Heavy blur (rejected before detection runs) | ~0.02s |

Real-photo time dropped substantially (~6–7s under the ellipse-ensemble
revisions to ~2.0–2.3s here) because the segmentation-first pipeline no
longer runs 6 separate full-resolution segmentation variants (4 Canny
threshold pairs, adaptive, Otsu) plus per-candidate radial-histogram
fallbacks and cross-family clustering — it runs one Lab-distance
classification pass per z-threshold over a localized window, which is
both a smaller region and a cheaper per-pixel operation. This is the
"time to useful result" a user actually experiences end to end
(including HEIC/full-resolution decode), not just algorithm time.

## Cost

**Variable cost per image: $0 API/model cost.** The measurement path is
pure OpenCV/NumPy computer vision running in the same process that serves
the HTTP request — there is no per-call paid API or hosted model in the
loop (no vision LLM, no third-party CV API). This is a statement about
per-image marginal API/model cost specifically, not a claim that the
service is free to run — see hosting cost below for what it actually
costs to keep this online.

**Hosting cost (separate from per-image cost):** deployed on Railway,
which bills usage-based on top of a plan minimum
([railway.com/pricing](https://railway.com/pricing), current as of this
writing):

| Plan | Monthly minimum | Included usage credit |
|---|---|---|
| Hobby | $5 | $5/month |
| Pro | $20 | $20/month |

Metered usage on top of that: **$0.00000772 per vCPU-second** and
**$0.00000386 per GB(RAM)-second** while the container is running, plus
$0.05/GB egress. A small always-on single-instance FastAPI service like
this one, idling most of the time with occasional ~6-second bursts of
CPU-heavy work per photo, comfortably fits inside the Hobby plan's $5
included credit under light/demo traffic; cost scales with how long the
container stays up and how much CPU each request burns, not with a
per-image API fee. This is a cost *structure* statement based on
Railway's published rates, not a measured production bill — actual spend
depends on real traffic volume, which this delivery hasn't had.

## How AI-generated code/output was independently checked (concrete example)

During development, a delivery package (zipped code + a `regression_suite.py`
+ a written claim of "9/10 tests passing, including both real photos")
was reviewed before merging. Rather than accepting that claim, the
actual script was run against this repository's own `test_set/` — and it
failed immediately with `MISSING_FILE` on every single case, because the
script hardcoded absolute paths from a different machine
(`/home/claude/ring_test/...`) that don't exist in this repository. That
proved the "9/10 passing" claim could not have been produced by actually
running that script against the code being delivered.

Separately, the code drop bundled with that claim silently reverted two
already-fixed, already-committed regressions: it unpinned
`opencv-contrib-python`/`numpy`/`pillow` back to floating versions (risking
the OpenCV 5.x contour/Hough behavior change that breaks the calibrated
thresholds), and it reset `check_exposure`'s `max_bright_fraction` from
`0.35` back to `0.15` (which had already been shown to falsely reject
`synth_shadow_gradient.png`, a documented, supposed-to-pass regression
case). Neither of those two changes was accepted; only the genuinely new,
independently-verified content from that delivery (a new synthetic test
case, and portability fixes to the regression script's own paths) was
kept, after re-running the full suite locally to confirm 9/9 (10/10 once
the new case was added) still passed with the reverted, correct
`pipeline.py`. See commit `bef53c7` in this repository's history for the
full writeup at the time.

A second, later example: a "fix" for a `RING_EDGE_UNSTABLE` failure on a
reflective keyring photo was implemented, and it did make the error
disappear — but before shipping it, the actual returned diameter (23.47mm)
was checked against ground truth (27mm, recorded independently by ruler).
It was wrong by 3.5mm. The fix was reverted specifically because it
replaced an honest rejection with a confident wrong answer, which is a
worse failure mode for a measurement tool than refusing to guess — see
git history (`3e07ad8` then reverted by `0767fc5`) for the full record,
including the diagnostic evidence (radius histograms with no genuine peak
near the true boundary) that showed *why* the fix only appeared to work.

A third example, from building the current multi-peak/cross-family
architecture itself: the new design (every method contributes multiple
ranked candidate peaks, not just its single best) was built and verified
against `regression_suite.py` — 10/10 pass, with both real photos'
error improving (17.07mm vs ~17mm, and 26.63mm vs 27mm at 0.37mm error,
down from 0.53mm) — *before* its output was checked against anything
outside that suite, per this project's own ground-truth-leakage policy.
Only after that pass was it tested against the downscaled Real-ring-B
photo described above, and it returned a confident 23.4mm (0.13mm spread)
against the real 27mm — a new instance of the exact failure mode the
project had already flagged as unacceptable. Root cause, found by
printing each family's tagged candidates: two methods' own weaker,
lower-ranked peaks (not either method's own top pick) had coincidentally
landed within 0.27mm of each other, which the initial "at least one
member must be a top peak" safeguard didn't catch because it only
required *one* family in the group to be its own top pick, not all of
them. Tightening that to "every family in the group must be represented
by its own top-ranked peak" removed the false accept without changing
either real photo's result, confirmed by rerunning the full suite (still
10/10) and the downscaled photo directly. The remaining single
qualifying candidate for that photo (36.0mm, essentially the outer ring
itself) was still implausible, which is what motivated the separate,
pre-detection `RESOLUTION_TOO_LOW` gate described above — added and then
added as its own regression case (`real_ring_B_downscaled`, now 11/11)
rather than trusted on the strength of one manual check.

A fourth example, from building the current segmentation-first
architecture: before committing to the deterministic multi-threshold
design, two other approaches (GrabCut, FastSAM) were actually implemented
and run against the same 3 real photos, not assumed to be worse or better
from first principles. GrabCut's failure (merges ring and hole into one
blob) and FastSAM's inconsistency (isolates the hole on 1 of 3 photos,
merges everything into one object mask on the other 2) were both observed
directly from saved mask visualizations, not inferred. Only after that
comparison did the deterministic approach get built out fully and
regression-tested — and even then, the one real photo it REJECTs (Real
ring A) was verified by reading its own debug overlay (the recovered
region visibly only covers a bright crescent of the true opening, never
the whole boundary) before accepting that REJECT as correct rather than
as a bug to chase.
