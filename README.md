# Gauge — Ring Diameter Measurement

Mobile-friendly web app that measures a ring's internal diameter from a
photo, using a printed 20mm calibration marker for scale. No manual
diameter entry, no manual scale calibration.

## Architecture

- **backend/** — FastAPI service wrapping a deterministic OpenCV pipeline
  (ArUco marker detection → homography rectification → multi-method inner
  boundary detection → US ring size lookup). No ML model, no LLM in the
  measurement path — see `pipeline.py`'s module docstring and inline
  comments for why, and for the specific real-photo failures that shaped
  each guardrail.
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

It's the disagreement between independent detection methods (Canny at
several thresholds collapsed to one vote, adaptive threshold, Otsu) on
the *same* photo. It is **not** a full measurement-uncertainty estimate —
it doesn't account for printer scale error, lens distortion, or ArUco
corner localization error. The frontend says this explicitly in the
result screen's disclosure section; don't repurpose the number as a lab-
grade error bar without further calibration against more caliper-measured
rings.

## Test evidence (from development)

Two real rings, photographed on a wood table with the printed marker,
measured against this pipeline:

| Photo | Caliper/ruler ground truth | Pipeline result | Abs. error |
|---|---|---|---|
| IMG_9783 (steel ring, mild angle) | ~17mm (ruler, coarse) | 16.80mm | — (ruler too coarse to score precisely) |
| IMG_9784 (keyring-style ring) | 27mm (ruler) | 26.47mm | 0.53mm |

Six synthetic stress photos (clean, specular highlight, shadow gradient,
low contrast, textured background, heavy blur) and one adversarial decoy
(a second circular object in frame) are in the `test_set/` folder used
during development — see delivery notes for the full pass/fail table.

## Known limitations (first version)

- One ring, one marker, same plane, near-overhead only — by design (see
  the "not yet supported" list on the instructions screen).
- Processing time is real but not fast: ~6-8s per photo on the two real
  test photos, mostly full-resolution OpenCV work (Hough/Canny/Sobel) plus
  HEIC decoding. A downscale-before-processing optimization was tried and
  reverted — it silently broke calibrated detection thresholds on both
  real photos (see `main.py`'s `load_image_any_format` docstring). Revisit
  with a properly re-validated set of thresholds, not by re-adding the
  same shortcut.
- The Hough/radial fallback (used when the primary closed-contour detector
  can't get a clean boundary, e.g. on reflective metal) is coarser than
  the primary path: cross-method spread on real photos was 0.67-0.80mm,
  versus ~0.05mm on clean synthetic renders. Report this honestly; don't
  let the UI imply lab-grade precision it hasn't earned.
- Ring size lookup only covers the standard US 3-13.5 adult finger-ring
  range; a measured diameter outside that (e.g. a keyring) is flagged via
  `ring_size_in_range: false` rather than forcing a nonsense size.
- No accounts, no persistence — every request is measured independently;
  nothing about a previous photo influences the next one.
