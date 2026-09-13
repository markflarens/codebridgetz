"""
Ring measurement API.

POST /api/measure   - upload a photo, get back a measurement or a retake reason
GET  /api/marker     - download the printable 20mm calibration marker (PDF)
GET  /                - serves the frontend

Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""
import base64
import io
import subprocess
import tempfile
import time
import os

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import pipeline
from ring_sizes import lookup_ring_size

app = FastAPI(title="Ring Measurement API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MARKER_PDF_PATH = os.path.join(BACKEND_DIR, "ring_marker_20mm.pdf")

# Human-readable retake messages for every structured reason code the
# pipeline can return. Keeping this mapping in one place means the API
# contract (reason code) and the user-facing wording can change
# independently - the frontend only needs to render `message`.
RETAKE_MESSAGES = {
    "MARKER_NOT_FOUND": "Marker not detected. Make sure the whole marker is flat, unobstructed, and in the photo.",
    "MARKER_PARTIAL": "Marker is not fully visible. Keep the entire marker inside the frame.",
    "MARKER_TOO_SMALL": "Marker appears too small in the photo. Move the camera closer to the ring and marker.",
    "RESOLUTION_TOO_LOW": "Photo resolution is too low relative to the marker's size in frame for a reliable measurement. Retake at the camera's normal/full resolution (avoid pre-shrunk or heavily compressed images), or move closer so the marker fills more of the frame.",
    "PERSPECTIVE_TOO_HIGH": "Too much perspective. Take the photo more directly from above.",
    "BLUR": "Photo is too blurry. Hold the phone steady and retake.",
    "UNDEREXPOSED": "Photo is too dark. Move to a brighter area and retake.",
    "OVEREXPOSED": "Photo is too bright or reflective. Use softer, more even light and avoid direct reflections on the ring or marker, then retake.",
    "BACKGROUND_REFERENCE_UNAVAILABLE": "Could not reliably sample the table/background around the ring in this photo. Make sure the background is clearly visible around the ring (not filled entirely by the ring, marker, or other objects), then retake.",
    "RING_NOT_FOUND": "Ring edge is unclear. Avoid reflections and place the ring on a contrasting surface.",
    "RING_EDGE_UNSTABLE": "Ring edge is unclear. Avoid reflections and place the ring on a contrasting surface.",
    "INCONSISTENT_DETECTION": "Detected more than one possible circular object. Make sure only one ring is in the photo, away from other round objects.",
}
DEFAULT_RETAKE_MESSAGE = "Could not get a reliable measurement from this photo. Please retake following the instructions."


def load_image_any_format(raw_bytes: bytes) -> np.ndarray:
    """
    Load an uploaded photo into an OpenCV BGR array, handling two real-world
    issues that a plain cv2.imdecode misses:

    1. EXIF orientation: phone cameras often save a photo "sideways" with an
       EXIF tag saying how to rotate it for display. cv2 ignores this
       entirely, so a portrait photo can silently arrive as landscape (or
       upside down), which would corrupt the marker/ring geometry. PIL's
       ImageOps.exif_transpose() applies the correct rotation before we
       ever touch pixels.
    2. HEIC/HEIF: iOS saves photos in HEIC by default when a user picks
       "choose from library" instead of the live camera capture. Neither
       PIL nor cv2 can decode HEIC without an extra plugin, so we fall back
       to ImageMagick's `convert` (must be installed on the host - see
       README) if the direct decode fails.

    NOTE: a downscale-before-processing "optimization" was tried here and
    reverted. It seemed reasonable (the scale reference is the physical
    marker, not pixel count) but measured results on the two real test
    photos proved otherwise: IMG_9783 silently changed from a verified
    16.80mm to a wrong 27.93mm, and IMG_9784 flipped from a correct
    measurement to a false RETAKE. The Hough/Canny/gradient thresholds
    throughout pipeline.py were empirically calibrated against full-
    resolution real photos; changing input resolution changes edge
    sharpness and noise characteristics enough to invalidate that
    calibration. Full resolution is kept until this can be re-validated
    properly, not assumed safe from first principles.
    """
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    # Fallback: HEIC or any other format PIL can't decode directly.
    with tempfile.NamedTemporaryFile(suffix=".heic", delete=False) as f_in:
        f_in.write(raw_bytes)
        in_path = f_in.name
    out_path = in_path + ".jpg"
    try:
        subprocess.run(["convert", in_path, out_path], check=True, capture_output=True, timeout=20)
        img = Image.open(out_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.remove(p)


def encode_overlay_base64(bgr_image: np.ndarray, max_dim=1200) -> str:
    """Downscale (the rectified debug image can be huge on modern phone
    photos) and JPEG-encode the overlay for embedding directly in the
    JSON response as a data URL."""
    h, w = bgr_image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        bgr_image = cv2.resize(bgr_image, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


@app.post("/api/measure")
async def measure(file: UploadFile = File(...)):
    t0 = time.time()
    raw = await file.read()

    try:
        photo = load_image_any_format(raw)
    except Exception as e:
        return JSONResponse({
            "status": "retake",
            "reason": "UNREADABLE_PHOTO",
            "message": "Could not read this photo. Please try a different image (JPEG or PNG).",
        })

    # DIAGNOSTIC LOGGING (temporary): print what actually arrives at the
    # server. Added because a downscaled copy of an already-validated,
    # passing photo (real_ring_B) was proven locally to reproduce the exact
    # same RING_EDGE_UNSTABLE failure users are seeing on new phone photos -
    # but there's no visibility into what resolution phones are actually
    # uploading in production. This line has zero effect on measurement
    # behavior; it only writes to Railway's logs so we can check actual
    # upload sizes against real failure reports instead of guessing.
    ph, pw = photo.shape[:2]
    print(f"[measure] received photo: {pw}x{ph}px, {len(raw)/1024:.0f}KB, filename={file.filename!r}")

    result = pipeline.measure_ring(photo)
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    overlay_b64 = encode_overlay_base64(result.debug_img) if result.debug_img is not None else None

    if not result.ok:
        return JSONResponse({
            "status": "retake",
            "reason": result.reason,
            "message": RETAKE_MESSAGES.get(result.reason, DEFAULT_RETAKE_MESSAGE),
            "processing_time_ms": elapsed_ms,
            "overlay_image": overlay_b64,
        })

    size_info = lookup_ring_size(result.diameter_mm)

    return JSONResponse({
        "status": "success",
        "diameter_mm": round(result.diameter_mm, 2),
        "ring_size": size_info["display"],
        "ring_size_in_range": size_info["in_range"],
        "sizing_standard": size_info["standard"],
        "sizing_source": size_info["source"],
        "rounding_rule": size_info["rounding_rule"],
        "detection_spread_mm": round(result.detection_spread_mm, 2),
        "family_estimates": {k: round(v, 2) for k, v in result.family_estimates.items()},
        "processing_time_ms": elapsed_ms,
        "overlay_image": overlay_b64,
    })


@app.get("/api/marker")
async def get_marker():
    # Served INLINE, not as an attachment: the frontend opens this in a new
    # tab (target="_blank", no `download` attribute) so the user can view,
    # print, or save it with the browser's own controls rather than having
    # it silently auto-download. `filename=` alone on Starlette's
    # FileResponse defaults content_disposition_type to "attachment",
    # which forces a download regardless of how the link was opened - so
    # content_disposition_type is set explicitly here instead. `filename`
    # is still passed so "Save As" (if the user chooses it from the PDF
    # viewer) offers a sensible name instead of a random one.
    return FileResponse(
        MARKER_PDF_PATH,
        media_type="application/pdf",
        filename="ring_marker_20mm.pdf",
        content_disposition_type="inline",
    )


# Serve the frontend static files (index.html, style.css, app.js) at "/"
FRONTEND_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
