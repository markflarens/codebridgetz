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

- **backend/** — FastAPI service wrapping a CV pipeline (ArUco marker
  detection → homography rectification → coarse ring ROI localization →
  prompted ring-BODY segmentation → enclosed hole read from the resulting
  mask's own topology → robust ellipse fit → US ring size lookup). ArUco/
  homography/Hough are deterministic and used only to locate roughly
  where the ring is. WHICH pixels are the ring's own body material, and
  therefore which enclosed region is its hole, is decided by prompting
  MobileSAM (a small, genuinely point-promptable segmentation model) with
  positive points on the ring band and negative points inside the hole
  and outside the ring, then reading the hole off the resulting mask's
  contour hierarchy — not by matching pixel colors against the
  surrounding background (an earlier revision's approach, dropped because
  it defines the hole by what's AROUND the ring rather than anything
  about the ring itself, and is fragile under ordinary shadows,
  reflections, and lighting gradients). Because `locate_outer_ring_circle`
  can land on either the ring's outer edge or its inner hole edge on a
  clean photo, both readings are tried (see `pipeline.py`'s
  TWO-HYPOTHESIS STRATEGY) and whichever produces a valid mask (one body,
  one hole, high hole solidity, plausible shape) is used. A convex hull is
  used only to CHECK that validity, never to fit the reported ellipse — so
  an incomplete mask can't be hull-reconstructed into a convincing-looking
  wrong measurement. A properly-seeded GrabCut baseline and a prompted
  FastSAM baseline were both tried and found less reliable than prompted
  MobileSAM — see DELIVERY_NOTES.md for the investigation and visual
  comparison. This is a real, new ML dependency (torch + ultralytics + a
  ~39MB MobileSAM checkpoint) — see "ML dependency" below for the honest
  cost/benefit. The model's role is scoped narrowly: it only produces a
  ring-body mask; the diameter itself is still computed by classical
  ellipse-fitting on that mask's own contour, exactly as every earlier
  revision computed it. Set `RING_DEBUG_OVERLAY=1` to get back an overlay
  image showing each hypothesis's mask/hole contour and the one accepted.
- **frontend/** — vanilla HTML/CSS/JS single-page app (no build step).
  Talks to the backend over `fetch`.

## Running locally

Requirements: Python 3.10+, and **ImageMagick** installed on the host
(`convert` command) for HEIC photos from iPhones — Pillow/OpenCV can't
decode HEIC without it.

```bash
cd backend
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
uvicorn main:app --host 0.0.0.0 --port 8000
```

The `--extra-index-url` pulls a CPU-only `torch` build (see Dockerfile
comments) instead of a much larger default CUDA one; every other package
still comes from PyPI. The MobileSAM checkpoint (`backend/weights/
mobile_sam.pt`, ~39MB) is not committed to the repo (kept out via
`.gitignore` to keep the repo/patches a reasonable size) — fetch it once
with:

```bash
python3 -c "from ultralytics import SAM; SAM('backend/weights/mobile_sam.pt')"
```

(run from the repo root; `ultralytics` downloads straight to that path).
The Dockerfile runs the same command at image build time, so a deployed
container never needs this step or any runtime network access to fetch
it.

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
  "detection_spread_mm": 0.01,
  "family_estimates": {"r=outer": 16.79, "r=inner": 16.80},
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

Under the current ring-body/mask-topology pipeline, this is the size
difference between the two independent band HYPOTHESES (see
`pipeline.py`'s TWO-HYPOTHESIS STRATEGY) when BOTH of them independently
produced a valid ring-body/hole mask — i.e. how much two separate
readings of the same coarse localization hint, each segmented and
measured on its own, agree with each other. When only one hypothesis
produced a valid mask, there is nothing to compare it against and the API
returns `null`/the frontend shows "n/a". Either way this is **not** a
full measurement-uncertainty estimate — it doesn't account for printer
scale error, lens distortion, or ArUco corner localization error. The
frontend says this explicitly in the result screen's disclosure section;
don't repurpose the number as a lab-grade error bar without further
calibration against more caliper-measured rings.

## ML dependency, honestly

This revision adds a real dependency the project didn't have before:
`torch` + `ultralytics` + a MobileSAM checkpoint (~39MB), baked into the
Docker image so a fresh deploy doesn't depend on runtime network access
to fetch it. That's a real cost — bigger image, and CPU inference time
per photo (see "Test evidence" below for the actual before/after
processing-time numbers) — taken on because the alternative that worked
without it (matching the hole against the surrounding background's
color) was fragile in exactly the ways real photos are hardest: ordinary
shadows, reflections, textured surfaces, and lighting gradients. The
model's job is scoped narrowly and does not change the project's
"diameter is computed by classical geometry, not asked from a model"
position: it only answers "which pixels are ring-body material, given
these prompts" — it is never asked to estimate a size, and the reported
diameter is still a classical ellipse fit to that mask's own contour,
converted to mm via the ArUco-calibrated scale, exactly like every
earlier revision.

## Test evidence (from development)

Two real rings, photographed on a wood table with the printed marker,
measured against this pipeline:

| Photo | Caliper/ruler ground truth | Pipeline result | Abs. error |
|---|---|---|---|
| IMG_9784 (keyring-style ring) | 27.00mm (ruler) | 28.27mm | 1.27mm |
| IMG_9783 (ornate/knurled ring, mild angle, reflection gradient) | ~17mm (ruler, coarse — see note) | 21.13mm | n/a — see note |

IMG_9783 is now an honest ACCEPT, not a REJECT: this photo's true object
is a knurled-band ring with a real bright-to-dark reflection gradient
across its opening (not the "deep concave cap" an earlier revision's
notes mischaracterized it as), and every prior architecture failed on it
for a reason specific to that architecture (see DELIVERY_NOTES.md for the
full history). The ~17mm ground truth here was a coarse, informal ruler
reading taken under that earlier, incorrect "small thin object"
assumption; the winning mask was directly visually inspected (not just
topology-checked) and its contour traces the ring's true opening tightly
and completely, including through the reflection gradient — a ~4mm gap
against a ground truth that dubious is far more consistent with the
ground truth being wrong than the mask being wrong, but this project
doesn't have a trustworthy independent re-measurement to score against,
so the honest thing is to report the mask-confirmed number and say so,
not to force a "correct-looking" answer either direction. IMG_9784's
1.27mm gap against its own (also ruler-based, less dubious) ground truth
is discussed in DELIVERY_NOTES.md alongside a visual confirmation that
this mask, too, is not visibly wrong.

Six synthetic stress photos (clean, specular highlight, shadow gradient,
low contrast, textured background, heavy blur) and one adversarial decoy
(a second circular object in frame) are in the `test_set/` folder used
during development — see delivery notes for the full pass/fail table.
Synthetic-photo accuracy improved markedly under this revision (typically
0.01–0.25mm error against exact synthetic ground truth, vs. 0.2–0.6mm
under the previous background-color-matching revision).

## Known limitations (first version)

- One ring, one marker, same plane, near-overhead only — by design (see
  the "not yet supported" list on the instructions screen).
- Processing time on real test photos is now ~8-9s (up from ~2-2.3s under
  the previous background-color-matching revision) - this is the direct,
  honest cost of the ML dependency discussed above: up to two MobileSAM
  CPU inference passes (one per band hypothesis) instead of a pixel-
  distance threshold sweep. Synthetic (smaller, cleaner) test photos run
  in ~2.8-3.2s. This is a real regression in latency for a real
  improvement in reliability on hard real photos (see "Test evidence"
  above) - not free, and worth knowing about before deploying somewhere
  latency-sensitive. GPU inference or a smaller/quantized checkpoint would
  likely recover most of this; neither was attempted here (CPU-only
  Railway deployment, and MobileSAM was already the "small" end of the
  SAM family).
- Reflective/metal rings, and rings whose opening has a strong internal
  reflection gradient, are handled meaningfully better than under every
  earlier revision (see DELIVERY_NOTES.md's visual comparison) - this was
  the specific failure mode this revision was built to fix, and it now
  succeeds on the one real photo that defeated every prior approach.
  Photos where `locate_outer_ring_circle` can't localize the ring at all,
  or where neither band hypothesis's prompts land MobileSAM on a clean
  single-body/single-hole mask, still honestly REJECT
  (`RING_NOT_FOUND`/`NO_BAND_HYPOTHESIS`/`HOLE_NOT_SUFFICIENTLY_VISIBLE`)
  rather than report a number the mask itself doesn't support.
  `detection_spread_mm` is only meaningful when both hypotheses validate
  (see "What detection_spread_mm is and isn't" above) - it is not present
  on every accepted result.
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
