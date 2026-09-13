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
perspective rectification (marker gives real-world scale in mm) →
multiple independent classical segmentation methods (Canny at 4
threshold pairs, adaptive threshold, Otsu threshold) → contour/ellipse
fitting or a radius-histogram fallback for reflective rings → cross-method
consensus check → US ring-size lookup via a documented linear formula. No
network calls, no model inference, and no LLM involvement happen during a
`/api/measure` request. (Claude, as an AI coding assistant, was used to
*write* this pipeline — it is not called at runtime.)

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
| Real ring A (steel ring, mild angle) | ~17mm | ruler (coarse) | ACCEPT | ACCEPT, 17.07mm | not precisely scoreable — ruler too coarse for sub-mm ground truth |
| Real ring B (keyring-style ring) | 27.00mm | ruler | ACCEPT | ACCEPT, 26.63mm | 0.37mm |
| Real ring B, downscaled ~2.06x (same ring/photo, resolution reduced to match a real problem upload) | 27.00mm (same ring as above) | ruler (same measurement, reused) | **REJECT** | REJECT — `RESOLUTION_TOO_LOW` | n/a — correctly refuses at a resolution below what detection is validated for, instead of guessing |
| Synthetic clean baseline | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.36mm | 0.04mm |
| Synthetic specular highlight | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.40mm | 0.00mm |
| Synthetic shadow gradient | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.36mm | 0.04mm |
| Synthetic low contrast (~6% contrast) | 17.40mm | rendered (exact) | **REJECT** | REJECT — `RING_NOT_FOUND` | n/a — correctly refuses rather than guessing |
| Synthetic textured background | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.36mm | 0.04mm |
| Synthetic heavy blur | 17.40mm | rendered (exact) | **REJECT** | REJECT — `BLUR` | n/a — rejected before detection even runs |
| Synthetic wide scene (ring small in frame) | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.57mm | 0.17mm |
| Synthetic decoy object (adversarial second circle) | 17.40mm | rendered (exact) | ACCEPT (ignore decoy) | ACCEPT, 17.53mm | 0.13mm |

**11/11 pass** against `test_set/expected_vs_actual.csv` at delivery time.
Every rejected case rejects for the reason the test was designed to
trigger, not an unrelated failure.

**Detection architecture (current):** for each of the 6 segmentation
variants (4 Canny threshold pairs, adaptive threshold, Otsu), the primary
closed-contour detector runs first; if it can't get a clean inner+outer
contour pair, a radial-histogram fallback (gradient-direction-filtered
edge pixels binned by radius around an independently Hough-validated
outer circle) now returns *every* genuine local peak in its histogram,
not just the single tallest one, each tagged with its own rank. All
candidates from all variants — contour-based and radial — are pooled,
collapsed to one family each (canny/adaptive/otsu, since the 4 Canny
variants are correlated), then grouped by proximity (1mm chaining) across
families. A group only counts as genuine cross-method agreement, and can
only be accepted, if it has candidates from at least 2 distinct families
*and* every family in the group is represented by its own top-ranked
peak — not a weak secondary peak that happens to land nearby. Groups that
clear that bar must also agree spatially (family centroids within 2mm of
each other in real space), which catches two methods finding two
different round things rather than the same hole. This replaced an
earlier design that collapsed each family down to one value *before*
checking agreement, which could throw away a method's genuinely-correct
candidate if its own single best guess happened to be wrong.

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
- **Reflective/metal rings are the hardest case.** The primary
  closed-contour detector often can't get a clean boundary on a
  reflective ring at all (0 of 6 segmentation variants produced a closed
  contour on Real ring B in testing); the radial-histogram fallback that
  handles this is real but coarser (cross-method spread 0.53–0.8mm on
  real photos vs. ~0.04mm on clean synthetic renders).
  A **wrong-but-confident** fix for one specific low-contrast reflective
  photo was found, verified to be wrong (23.47mm vs 27mm ground truth by
  directly downscaling the same validated photo), and reverted rather
  than shipped — see the "AI output independently checked" section below.
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
| Real ring A (full-res phone photo) | ~7.0s |
| Real ring B (full-res phone photo) | ~6.2s |
| Synthetic renders (smaller images) | 0.3–0.6s |
| Heavy blur (rejected before detection runs) | ~0.02s |

The real-photo time is dominated by HEIC/full-resolution decode plus
running 6 segmentation variants × Hough/contour detection at full
resolution — this is the "time to useful result" a user actually
experiences end to end, not just algorithm time.

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
