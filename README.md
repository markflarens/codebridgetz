# Gauge — Ring Diameter Measurement

**Submission details** (AI tools/models used, exact runtime stack, reused
vs. custom code, reproducible test set with independently recorded ground
truth, expected vs. actual per image, dev time, processing time, cost
breakdown, and a concrete example of how AI-generated output was
independently checked) are in **[DELIVERY_NOTES.md](DELIVERY_NOTES.md)**.

Mobile-friendly web app that measures a ring's internal diameter from a
photo, using a printed 20mm calibration marker for scale. No manual
diameter entry, no manual scale calibration.

## Architecture

- **backend/** — FastAPI service wrapping a deterministic OpenCV pipeline
  (ArUco marker detection → homography rectification → coarse ring ROI
  localization → background-color segmentation of the ring's enclosed
  hole → US ring size lookup). No ML model, no LLM in the measurement
  path — see `pipeline.py`'s module docstring and inline comments for
  why, and for the specific real-photo failures that shaped each
  guardrail. ArUco/homography/Hough are used only to locate roughly where
  the ring is. Which pixels inside that area are the true inner hole is
  then decided by SEGMENTATION, not by scoring candidate boundary curves
  against each other: every pixel is classified as background-colored or
  not (against the photo's own locally-sampled background), the single
  enclosed background-colored region is the hole by construction, and its
  ellipse is accepted once its size and shape stop changing across a
  sweep of classification thresholds. Two alternatives (GrabCut, and the
  lightweight ML segmenter FastSAM) were tried and found less reliable
  than this deterministic approach — see DELIVERY_NOTES.md for the
  investigation. Set `RING_DEBUG_OVERLAY=1` to get back an overlay image
  showing every threshold's candidate ellipse (color-coded tight→loose)
  and the one accepted.
- **frontend/** — vanilla HTML/CSS/JS single-page app (no build step).
  Talks to the backend over `fetch`.

## Running locally

Requirements: Python 3.10+, and **ImageMagick** installed on the host
(`convert` command) for HEIC photos from iPhones — Pillow/OpenCV can't
decode HEIC without it.

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` on your phone (same network as the host) or
in a desktop browser. The frontend is served by the same FastAPI process
at `/`; the API lives under `/api/*`.

To use from an actual phone camera, the host needs to be reachable from
the phone — either run on a machine on the same Wi-Fi and use its LAN IP,
or deploy behind HTTPS (`capture="environment"` camera input generally
requires a secure context on iOS Safari).

## API

`POST /api/measure` — multipart form, field `file` = the photo.

Success:
```json
{
  "status": "success",
  "diameter_mm": 16.8,
  "ring_size": "6.5",
  "ring_size_in_range": true,
  "sizing_standard": "US/Canada ring size (linear inside-circumference scale)",
  "sizing_source": "https://measureringsize.com/ring-size-chart",
  "rounding_rule": "...",
  "detection_spread_mm": 0.8,
  "family_estimates": {"canny": 16.53, "adaptive": 16.8, "otsu": 17.33},
  "processing_time_ms": 6408.7,
  "overlay_image": "data:image/jpeg;base64,..."
}
```

Retake:
```json
{
  "status": "retake",
  "reason": "PERSPECTIVE_TOO_HIGH",
  "message": "Take the photo more directly from above.",
  "processing_time_ms": 340.2,
  "overlay_image": null
}
```

`GET /api/marker` — downloads the printable 20mm calibration marker (PDF).

## What "detection_spread_mm" is and isn't

Under the current segmentation-first pipeline, this is the remaining gap
between the diameter recovered at the two LOOSEST background-classification
thresholds tried in the sweep — i.e. how much the recovered hole boundary
was still changing right at the point it was judged "stable enough" to
accept. It is **not** a full measurement-uncertainty estimate — it
doesn't account for printer scale error, lens distortion, or ArUco corner
localization error. The frontend says this explicitly in the result
screen's disclosure section; don't repurpose the number as a lab-grade
error bar without further calibration against more caliper-measured
rings.

## Test evidence (from development)

Two real rings, photographed on a wood table with the printed marker,
measured against this pipeline:

| Photo | Caliper/ruler ground truth | Pipeline result | Abs. error |
|---|---|---|---|
| IMG_9784 (keyring-style ring) | 27mm (ruler) | 27.52mm | 0.52mm |
| IMG_9783 | ~17mm (ruler, coarse) | REJECT — `MASK_AMBIGUOUS` | n/a |

IMG_9783 is now an honest, investigated REJECT, not an ACCEPT: visual
inspection of its own debug overlay shows the object is actually a
knurled/serrated metal cap with a deep, concave interior carrying a real
brightness gradient across the true opening — not a flat ring band. The
segmentation never stabilizes and never reaches a plausible shape for
this photo at any threshold tried, which is the correct outcome for an
object whose true boundary the mask genuinely cannot recover cleanly, not
a regression — see DELIVERY_NOTES.md for the full architecture history,
the GrabCut/FastSAM investigation, and a debug-overlay comparison for all
three difficult real photos evaluated during this redesign.

Six synthetic stress photos (clean, specular highlight, shadow gradient,
low contrast, textured background, heavy blur) and one adversarial decoy
(a second circular object in frame) are in the `test_set/` folder used
during development — see delivery notes for the full pass/fail table.

## Known limitations (first version)

- One ring, one marker, same plane, near-overhead only — by design (see
  the "not yet supported" list on the instructions screen).
- Processing time on real test photos is now ~2-2.3s (down from ~6-8s
  under the earlier ellipse-ensemble pipeline — the segmentation-first
  design runs one classification pass per threshold over a localized
  window instead of six full-resolution segmentation variants plus
  fallbacks), mostly HEIC decoding and full-resolution OpenCV work. A
  downscale-before-processing optimization was tried and reverted — it
  silently broke calibrated detection thresholds on both real photos (see
  `main.py`'s `load_image_any_format` docstring). Revisit with a properly
  re-validated set of thresholds, not by re-adding the same shortcut.
- Reflective/metal rings remain the hardest case. Segmentation handles
  ordinary highlights/bevels better than the retired ellipse-candidate
  approach (see DELIVERY_NOTES.md's visual comparison), but an object
  whose true opening carries a genuine, strong brightness gradient across
  it can still legitimately defeat every threshold in the sweep — the
  pipeline reports `MASK_AMBIGUOUS`/`HOLE_NOT_SUFFICIENTLY_VISIBLE` rather
  than a number it can't stand behind. `detection_spread_mm` on an
  ACCEPTed real photo is typically well under 0.1mm now (the sweep had
  already stabilized), which is a stability signal, not a lab-grade
  accuracy guarantee — see "What detection_spread_mm is and isn't" above.
- Detection thresholds are only validated at full phone-camera resolution
  (marker ~430px on a side in the original upload). A photo uploaded at
  roughly half that pixel density is rejected with `RESOLUTION_TOO_LOW`
  rather than measured — see `pipeline.py`'s `detect_and_rectify` and
  DELIVERY_NOTES.md for how this was found (a downscaled copy of an
  already-passing real photo produced a confident wrong answer before
  this gate existed).
- Ring size lookup only covers the standard US 3-13.5 adult finger-ring
  range; a measured diameter outside that (e.g. a keyring) is flagged via
  `ring_size_in_range: false` rather than forcing a nonsense size.
- No accounts, no persistence — every request is measured independently;
  nothing about a previous photo influences the next one.
