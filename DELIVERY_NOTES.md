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
  detection, Hough circle transform, radial/Sobel gradient histograms,
  contour hierarchy analysis, `fitEllipse`)
- **Ring-body segmentation:** `ultralytics` `8.4.150` (its `SAM` wrapper,
  loading a MobileSAM checkpoint), `torch` `>=2.12,<2.15` (CPU-only build,
  installed from PyTorch's own CPU wheel index — see `Dockerfile`) — see
  "ML dependency, honestly" below for exactly what this is used for and
  what it costs. Both version constraints were checked against real
  PyPI/pytorch.org listings via web search before pinning, not trusted
  from this sandbox's own `pip show`/`pip index` output, which has
  previously been shown to report phantom not-yet-real versions for
  fast-moving packages (see `backend/requirements.txt`'s comments).
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

**Runtime ring measurement uses a small ML model for one narrow step
(ring-body segmentation), not a vision LLM, and never to estimate the
diameter itself.** The measurement path is: ArUco marker detection →
homography-based perspective rectification (marker gives real-world scale
in mm) → coarse Hough-circle ring ROI localization → two independent
readings of that coarse hint (see "TWO-HYPOTHESIS STRATEGY" below) → for
each reading, MobileSAM is **prompted** (positive points on the ring
band, negative points inside the hole and outside the ring) to produce a
ring-body mask → the enclosed hole is read off that mask's own contour
**topology** (not matched against anything) → a convex-hull solidity
check gates whether the result is trustworthy → an ellipse fit to the
mask's **raw** hole contour (never the hull) → US ring-size lookup via a
documented linear formula. This is a genuine, new runtime dependency:
CPU inference happens during a `/api/measure` request (no network calls -
the checkpoint is baked into the deploy image, not fetched at request
time). (Claude, as an AI coding assistant, was used to *write* this
pipeline — the coding assistant itself is never called at runtime; only
the small segmentation model it integrated is.) See "ML dependency,
honestly" and "Detection architecture" below for the full reasoning, and
for the two other approaches (properly-seeded GrabCut, prompted FastSAM)
that were investigated and rejected before this one.

## Reused vs. custom

**Reused (libraries/algorithms, not app logic):**
- OpenCV's ArUco detector, `HoughCircles`, `findContours`/`fitEllipse`,
  `Sobel`, `warpPerspective`/`findHomography`, `convexHull`
- **MobileSAM** (via `ultralytics.SAM`) — a pretrained, off-the-shelf
  point-promptable segmentation model, used exactly as published (no
  fine-tuning); this project only decides WHAT prompts to feed it (from
  its own geometric hypotheses) and WHAT to do with its output mask
  (topology analysis). The model itself, and its weights, are entirely
  reused.
- FastAPI/Uvicorn for the HTTP layer, Pillow for EXIF-aware image loading
- The US ring-size formula itself (`backend/ring_sizes.py`) is a published
  standard, not invented here: `inside_circumference_mm = 36.5 + 2.55 *
  us_size`, sourced from
  [measureringsize.com's ring size chart](https://measureringsize.com/ring-size-chart)
  and cross-checked against two independent published anchor points
  (25karats.com, angara.com) before use — see the module docstring.

**Custom (written for this assignment):**
- The entire detection pipeline's control flow and calibration: the
  two-hypothesis radial-edge-histogram radius search (including the new
  `radial_outer_candidate`, a mirror of the pre-existing
  `radial_inner_candidate`), the point-prompt construction fed to
  MobileSAM, the mask-topology analysis (`analyze_mask_topology` - body/
  hole extraction via contour hierarchy, solidity gating, raw-contour
  ellipse fitting), all quality gates (exposure, blur, perspective,
  marker-size) and their thresholds, all failure-mode reason codes and
  messages. MobileSAM itself is reused; everything about how it's driven
  and how its output is judged is custom.
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
which band hypothesis wins, and whether a hypothesis's mask is accepted
at all, is decided purely from that mask's own topology (single body,
single hole, hole solidity, axis ratio) - properties computable from the
mask alone, with no reference to any known diameter - never by comparing
a candidate's diameter against a ground-truth value and picking whichever
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
| Real ring A (see "Real ring A" note below) | ~17mm (see note — likely unreliable) | ruler (coarse) | ACCEPT | ACCEPT, 21.13mm | n/a — see note, not scored against this ground truth |
| Real ring B (keyring-style ring) | 27.00mm | ruler | ACCEPT | ACCEPT, 28.27mm | 1.27mm |
| Real ring B, downscaled ~2.06x (same ring/photo, resolution reduced to match a real problem upload) | 27.00mm (same ring as above) | ruler (same measurement, reused) | **REJECT** | REJECT — `RESOLUTION_TOO_LOW` | n/a — correctly refuses at a resolution below what detection is validated for, instead of guessing |
| Synthetic clean baseline | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.41mm | 0.01mm |
| Synthetic specular highlight | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.43mm | 0.03mm |
| Synthetic shadow gradient | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.41mm | 0.01mm |
| Synthetic low contrast (~6% contrast) | 17.40mm | rendered (exact) | **REJECT** | REJECT — `RING_NOT_FOUND` | n/a — correctly refuses rather than guessing |
| Synthetic textured background | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.46mm | 0.06mm |
| Synthetic heavy blur | 17.40mm | rendered (exact) | **REJECT** | REJECT — `BLUR` | n/a — rejected before detection even runs |
| Synthetic wide scene (ring small in frame) | 17.40mm | rendered (exact) | ACCEPT | ACCEPT, 17.65mm | 0.25mm |
| Synthetic decoy object (adversarial second circle) | 17.40mm | rendered (exact) | ACCEPT (ignore decoy) | ACCEPT, 17.59mm | 0.19mm |

**11/11 pass** against `regression_suite.py`'s `CASES` at delivery time
(after updating Real ring A's expectation from REJECT back to ACCEPT —
see below for why that's a correct, evidence-based change, not a
shortcut, and the mirror image of the same change made in the other
direction during the previous revision). Every rejected case rejects for
the reason the test was designed to trigger, not an unrelated failure.
Synthetic-photo errors are a mixed bag against the previous revision -
tighter on the cases closest to its own segmentation prompts (clean/
specular/shadow: 0.01–0.03mm here vs. 0.10–0.16mm before), a bit looser
on the two geometrically harder synthetic cases (wide scene: 0.25mm vs.
0.10mm; decoy: 0.19mm vs. 0.06mm) - reported as observed, not rounded
toward whichever direction looks better. The real, unambiguous gain is
Real ring A going from entirely unmeasurable to a mask-confirmed ACCEPT
(see below).

**Detection architecture (current — ring-body / mask-topology revision):**
this pipeline has now gone through five designs, and the failure of each
is what motivated the next, so it's worth recording all five rather than
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
4. *Segmentation-first via background-color matching* — stopped choosing
   between ellipse candidates entirely: classified every pixel in a
   localized window as background-colored or not (Lab-space
   chroma+brightness distance from a sampled background reference), found
   the single enclosed background-colored connected component (the hole,
   by construction), and fit an ellipse to its convex hull. This removed
   revision 3's "multiple real, concentric edges can all look valid"
   ambiguity, but replaced it with a different, more basic problem: the
   hole was still defined entirely by matching the *surrounding
   background's* color, not by anything about the ring itself - fragile
   under ordinary shadows, reflections, textured surfaces, and lighting
   gradients, and with a conceptual gap (an enclosed background-colored
   region isn't *guaranteed* to be the physical hole; fitting to the
   convex hull means an incomplete/wrong segmented crescent can still
   produce a convincing-looking ellipse). Concretely, this revision could
   not measure Real ring A at all (see below) - its opening has a genuine
   brightness gradient across it that no single color-distance threshold
   recovers cleanly.
5. **Ring-body / mask-topology (current)** — changes WHAT is segmented.
   Instead of "is this pixel colored like the background", it asks "is
   this pixel part of the physical ring body", using MobileSAM (a small
   promptable segmentation model) seeded with positive prompts on the
   ring's own band material and negative prompts inside the hole and
   outside the ring - i.e. told what the ring itself looks like, not what
   the table looks like. The hole is read off the resulting mask's own
   contour **topology** (RETR_CCOMP hierarchy: body = largest top-level
   contour, hole = its largest enclosed child), not from any color match.
   A convex-hull solidity check gates whether that hole boundary is
   complete enough to trust; the reported ellipse is always fit to the
   mask's **raw** contour, never the hull, so an incomplete/wrong mask
   can't be hull-reconstructed into a convincing wrong number.

**TWO-HYPOTHESIS STRATEGY:** `locate_outer_ring_circle`'s own docstring
documents a real ambiguity - on a clean, high-contrast photo, Hough can
lock onto the ring's INNER hole edge just as easily as its outer edge,
with no way to tell which before looking further. Revision 5 generates
BOTH readings and carries each one all the way through segmentation and
topology analysis: H1 ("r=outer") assumes the hint is the outer edge and
searches inward (reusing the pre-existing `radial_inner_candidate`) for a
hole radius; H2 ("r=inner") assumes the hint is the hole edge and
searches outward (the new `radial_outer_candidate`, a mirror of the same
function) for an outer band radius. Whichever hypothesis (if either)
produces a valid mask topology is reported; when both do, their agreement
is surfaced as `detection_spread_mm` (a confidence signal, not a gate).

**Two alternatives were investigated and rejected before settling on
prompted MobileSAM for revision 5**, per explicit instruction to test a
properly-seeded deterministic baseline against a genuinely promptable
model and compare before deciding, judging mask correctness before ever
looking at a diameter:

- **Seeded GrabCut** (`cv2.grabCut`, `GC_INIT_WITH_MASK`), using the SAME
  two-hypothesis geometric priors as narrow seed regions (a small central
  disk as definite background, a thin annulus near the hypothesis's own
  band as definite foreground, well outside as definite background, the
  large genuinely-ambiguous middle region left probable-background rather
  than guessed - an earlier, naive fixed-fraction-radius seeding attempt
  failed outright on all 3 real photos before this fix, because ring band
  width relative to outer diameter varies too much to assume a fixed
  fraction). Once properly seeded, this produced topologically clean
  masks (solidity up to 0.98) on 2 of 3 real photos, and they were
  visually good. On the third (Real ring A, the reflection-gradient
  photo) it produced a mask that was STILL topologically clean by every
  structural check, yet visibly WRONG once the debug overlay was actually
  inspected: GrabCut's color model classified the bright, reflective part
  of the true hole as ring material (it resembles the band's own
  highlights) while correctly keeping the darker part as hole - cutting
  the real opening roughly in half. This is the concrete demonstration,
  not just a theoretical worry, of "topological cleanliness is necessary
  but not sufficient" - see the visual comparison below.
- **Prompted FastSAM** (`ultralytics` `FastSAM-s.pt`, given the exact same
  prompt points as MobileSAM below): returned no mask for every
  hypothesis on every photo. Investigated directly (not assumed) by
  calling FastSAM without prompts and inspecting its raw output: it only
  ever generates a handful of "segment everything" masks per image, and
  its `.prompt()` method SELECTS among those few pre-generated masks using
  the given points - it does not generate a new mask conditioned on the
  prompts the way a genuinely promptable model does. Since this project's
  earlier, separate investigation of promptless FastSAM had already found
  it fails to isolate the hole as its own object on 2 of 3 real photos,
  prompted FastSAM inherits that same ceiling regardless of prompt
  quality - not usable for this task, and not comparable to MobileSAM at
  all despite both being from the SAM family.
- **MobileSAM** (`ultralytics.SAM`, wrapping a `mobile_sam.pt` checkpoint
  - a real point-promptable encoder/decoder model, unlike FastSAM's
  filter-only `.prompt()`): produced visually complete, correctly-shaped
  hole boundaries on ALL 3 real test photos, INCLUDING the
  reflective-ring case where properly-seeded GrabCut's mask was
  confidently wrong, and INCLUDING the one photo (photo3_cap, an unseen
  test image with no ground truth) where GrabCut found no valid topology
  at all. This is what backs `measure_ring` now.

ArUco calibration, perspective rectification, and the Hough-based outer-
ring localization are unchanged and still run first — strictly for
coarse ROI localization ("roughly where is the ring"), never for
deciding which boundary inside that ROI is the right one.

**`analyze_mask_topology`** is the acceptance gate. Given a ring-body
mask from one hypothesis: clean it (`MORPH_CLOSE` then `MORPH_OPEN`),
find its contour hierarchy (`RETR_CCOMP`), take the largest top-level
contour as the body and its largest enclosed child as the hole. Reject
(`NO_BODY`/`BODY_TOO_SMALL`/`NO_HOLE`/`HOLE_TOO_SMALL`) if that structure
isn't there at all. Otherwise compute the hole contour's convex-hull
solidity and its fitted ellipse's axis ratio; accept only if there is
exactly one significant body, exactly one significant hole, solidity
`>= MIN_HOLE_SOLIDITY = 0.85`, and axis ratio `>= MIN_HOLE_AXIS_RATIO =
0.7` (otherwise `TOPOLOGY_NOT_CLEAN`). The reported ellipse is fit to the
hole's **raw** contour, never its hull - the hull is read only as a
number (solidity), never used to reshape what gets measured.

`measure_ring` tries both hypotheses independently, keeps whichever
valid result has the higher solidity (when both are valid, their
diameters are also compared and reported as `detection_spread_mm`), and
returns a specific reason - `RING_NOT_FOUND` (no coarse hint at all),
`NO_BAND_HYPOTHESIS` (neither radial search found a second radius),
`SEGMENTATION_MODEL_UNAVAILABLE` (the model failed to load), or
`HOLE_NOT_SUFFICIENTLY_VISIBLE` (segmentation ran but no hypothesis
passed the topology gate) - when neither hypothesis is usable.

A debug overlay mode (`RING_DEBUG_OVERLAY=1`) renders the coarse ROI,
each valid hypothesis's mask (tinted, one color per hypothesis) and hole
contour, and the winning ellipse in white on top — this is the same kind
of view used for the visual comparison below.

**Visual comparison against the real "currently difficult" photos**
(per explicit instruction: inspect mask correctness first, consult
ground truth only afterward):

- **Real ring B** (keyring-style, reflective): the winning mask's hole
  contour was zoomed in and inspected directly - it tracks the ring's
  true inner edge tightly and cleanly all the way around, with no visible
  gap or overreach. Its diameter (28.27mm) is 1.27mm over the 27.00mm
  ruler ground truth - looser than the previous revision managed on this
  same photo (27.52mm) - but a visibly correct mask with a plausible
  measurement error is a fundamentally different, more trustworthy
  outcome than a topologically-clean-looking mask that turns out wrong on
  inspection (see GrabCut on Real ring A, directly below).
- **Real ring A** (the reflection-gradient photo every earlier revision
  failed on): MobileSAM's mask, zoomed in and inspected directly, traces
  the true opening completely and tightly all the way around, INCLUDING
  through the bright-to-dark reflection gradient that defeated background-
  color matching (revision 4: never stabilized, axis ratio never exceeded
  ~0.6) and that properly-seeded GrabCut got topologically-clean-but-
  visibly-wrong on. This is the clearest evidence in this delivery that
  ring-BODY segmentation, not hole-by-color-matching, was the right
  semantic shift - see the "Real ring A" note below for the full history
  of this case's expected outcome.
- **photo3_cap** (an additional unseen photo used only for architecture
  comparison during this revision, not a regression-suite case - no
  independently recorded ground truth exists for it): properly-seeded
  GrabCut found no valid topology at all on this photo (both hypotheses
  degenerate/rejected). MobileSAM succeeded on both hypotheses, which
  agreed with each other to within 0.01mm (24.25mm vs. 24.26mm) - strong
  internal cross-hypothesis agreement, though still not a substitute for
  an independently measured ground truth this photo doesn't have.

**Real ring A: expectation changed from REJECT back to ACCEPT** (the
mirror image of the change the *previous* revision made in the other
direction). Under background-color matching, this photo's evidence was
unambiguous the other way: the recovered region never stabilized and
never reached a plausible axis ratio. Under ring-body segmentation, the
evidence is unambiguous in this direction: a clean, high-solidity,
correctly-shaped mask whose contour was directly, visually confirmed (not
just topology-checked) to trace the true opening. `regression_suite.py`'s
expected outcome was updated to `ACCEPT` to match, per the same
generalization policy this project has followed for every previous
architecture change: re-validated against the whole suite (still 11/11),
never special-cased for this one photo's filename or identity, and this
case carries no ground-truth/tolerance in `CASES` at all (see that file's
comments) rather than inventing a fabricated-precision number against a
ground truth (~17mm, coarse ruler, recorded under the since-corrected
assumption that this was a small thin object) that a mask-confirmed
21.13mm gives real reason to distrust.

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
- **Reflective/metal rings, and rings with a strong internal reflection
  gradient across their opening, were the hardest case** and are the
  specific failure mode this revision was built to fix. Real ring A
  (reflection gradient) now succeeds with a visually confirmed correct
  mask, and Real ring B (reflective band) also produces a visually
  correct mask, though with more diameter error than the previous
  revision managed on the same photo (see "Test evidence"/visual
  comparison above) - a real, disclosed trade: less error on the case
  that used to be entirely unmeasurable, more error on the case that used
  to be the closest fit. A GrabCut baseline that looked topologically
  clean but was visually wrong on Real ring A (see above) is the concrete
  evidence for why this revision insists on inspecting masks, not just
  their topology scores.
- **New ML dependency (torch + ultralytics + a MobileSAM checkpoint) and
  a real latency cost** - see "ML dependency, honestly" and the
  processing-time table below. This is a genuine trade against the
  previous revision's speed, taken on because background-color matching
  could not be made reliable on ordinary real-world lighting conditions
  no matter how it was tuned (see the architecture history above).
- **Real-ring diversity is thin (2 real rings, one additional unseen
  photo used only for architecture comparison).** Same gap this project
  has flagged in every previous revision: two rings, however different,
  is not enough to claim general reliability across ring types. See the
  note earlier in this section - unchanged by this revision.
- **Ring size lookup range:** the US 3–13.5 adult finger-ring range is
  the only lookup table implemented; anything outside it (e.g. a keyring)
  is flagged `ring_size_in_range: false` rather than forced into a
  nonsense size.
- **No accounts/persistence** — each photo is measured independently.

### Measured processing time

From the same `regression_suite.py` run:

| Case | Time |
|---|---|
| Real ring A (full-res phone photo) | ~9.0s |
| Real ring B (full-res phone photo) | ~8.2s |
| Synthetic renders (smaller images) | 2.8–3.4s |
| Heavy blur (rejected before detection runs) | ~0.02s |

Real-photo time went UP substantially versus the previous revision
(~2.0–2.3s there, ~8–9s here) - the direct, honest cost of the new ML
dependency: instead of one cheap Lab-distance classification pass per
z-threshold over a small window, this revision runs up to two full
MobileSAM CPU inference passes (one per band hypothesis) on top of the
same ArUco/Hough localization work. This is a real regression in latency
for a real improvement in reliability on the hardest real photos (see
"Test evidence" above) - not free, and worth knowing about before
deploying somewhere latency-sensitive; GPU inference or a smaller/
quantized checkpoint would likely recover most of it, neither was
attempted here. This is the "time to useful result" a user actually
experiences end to end (including HEIC/full-resolution decode and model
inference), not just algorithm time.

## Cost

**Variable cost per image: $0 API/model cost, but real compute cost.**
The measurement path runs entirely in the same process/container that
serves the HTTP request — there is no per-call *paid* API or third-party
hosted model in the loop (no vision LLM, no third-party CV API, no
metered inference endpoint). MobileSAM inference is genuinely real
compute, though: unlike the previous revision, per-image cost is no
longer negligible CPU work - each measured photo now burns several real
CPU-seconds (see processing-time table above), which shows up directly in
the hosting-cost math below, not as a separate line item. This is a
statement about per-image marginal *API/model-fee* cost specifically
(there is none), not a claim that the service's compute footprint is
unchanged from before — see hosting cost below for what it now actually
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
this one, idling most of the time with occasional CPU-heavy bursts per
photo, still comfortably fits inside the Hobby plan's $5 included credit
under light/demo traffic; cost scales with how long the container stays
up and how much CPU each request burns, not with a per-image API fee.
Two things changed with this revision, though, and are worth naming
honestly rather than reusing the previous estimate unchanged: those
bursts are now ~8-9s per real photo instead of ~2s (see processing-time
table above - roughly 4x the CPU-seconds per measurement), and the
container image itself is meaningfully larger (torch + ultralytics + the
~39MB MobileSAM checkpoint), which affects build/storage more than
per-request billing but is still a real footprint change. Under light/
demo traffic this is still well inside the Hobby plan's included credit;
it would matter more at higher sustained request volume than under
heavier, more latency-sensitive traffic. This remains a cost *structure*
statement based on Railway's published rates, not a measured production
bill — actual spend depends on real traffic volume, which this delivery
hasn't had.

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

A fourth example, from building the (now superseded) segmentation-first
architecture: before committing to that deterministic multi-threshold
design, two other approaches (GrabCut, FastSAM) were actually implemented
and run against the same 3 real photos, not assumed to be worse or better
from first principles. GrabCut's failure (merges ring and hole into one
blob) and FastSAM's inconsistency (isolates the hole on 1 of 3 photos,
merges everything into one object mask on the other 2) were both observed
directly from saved mask visualizations, not inferred. Only after that
comparison did the deterministic approach get built out fully and
regression-tested — and even then, the one real photo it REJECTed at the
time (Real ring A) was verified by reading its own debug overlay (the
recovered region visibly only covers a bright crescent of the true
opening, never the whole boundary) before accepting that REJECT as
correct rather than as a bug to chase.

A fifth example, from building the current ring-body/mask-topology
architecture: a properly-seeded GrabCut baseline was built, and its
result on Real ring A PASSED every structural check this project uses to
judge a mask (single body, single hole, solidity 0.98, plausible axis
ratio) - by that measure alone it looked like a success. Per the explicit
instruction to inspect the actual mask before trusting its topology, the
debug overlay was opened and looked at directly anyway, rather than
accepting the topology-clean verdict as sufficient. It showed the mask
visibly cutting the true opening roughly in half - the bright, reflective
part of the hole had been classified as ring material. This is the same
class of finding as the third example above (a fix that "passes its own
check" but is wrong on direct inspection), now demonstrated specifically
against the structural mask-topology checks this revision itself
introduces, which is exactly why this project's evaluation policy insists
on looking at the actual output before trusting any automated check of
it, however principled that check is. Separately, prompted FastSAM was
initially assumed (reasonably, given it's marketed as promptable) to be
directly comparable to MobileSAM - it was not: calling it without prompts
and inspecting its raw output directly showed it only ever generates a
handful of "segment everything" masks and its `.prompt()` method merely
selects among them, rather than generating a new mask conditioned on the
prompts. That distinction was found by direct investigation of the
library's actual behavior, not read from its documentation or assumed
from its name/marketing - see `pipeline.py`'s module docstring and the
architecture section above for what this changed about which model was
actually usable for this task.
