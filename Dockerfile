# Gauge — Ring Diameter Measurement
#
# ImageMagick's `convert` is required for HEIC photos from iPhones
# (backend/main.py's load_image_any_format falls back to it — neither
# Pillow nor OpenCV can decode HEIC directly). Railway's default
# nixpacks build for a Python app does NOT install ImageMagick, so it
# is installed explicitly here via apt. See backend/README notes on why
# this matters and what breaks without it.
FROM python:3.11-slim

# libgl1/libglib2.0-0: required by opencv-contrib-python's GUI-adjacent
# code paths even in a headless server context (cv2 import fails
# without them on a slim base image). imagemagick: HEIC fallback above.
RUN apt-get update && apt-get install -y --no-install-recommends \
    imagemagick \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
# torch/ultralytics (ring-body segmentation, see backend/pipeline.py's
# module docstring "ML DEPENDENCY, HONESTLY") are installed from
# PyTorch's own CPU-only wheel index as an extra index - the default
# index's torch build pulls in CUDA libraries that are both unusable
# (Railway's default compute here is CPU-only) and add well over 1GB to
# the image for nothing. --extra-index-url (not --index-url) keeps PyPI
# itself as the primary source for every other package in this file.
RUN pip install --no-cache-dir -r backend/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Bake the MobileSAM checkpoint into the image AT BUILD TIME, rather than
# committing a ~39MB binary to the repo or relying on ultralytics' own
# auto-download at first REQUEST: a first-request-triggered download would
# make the very first real measurement after every deploy unpredictably
# slow, and would fail outright on a deploy with no runtime egress to
# GitHub release assets. `ultralytics.SAM(path)` downloads straight to
# `path` when nothing is there yet (verified directly, not assumed from
# docs) - `backend/weights/mobile_sam.pt` matches the default path
# pipeline.py's _get_sam_model() reads (overridable via MOBILE_SAM_WEIGHTS),
# so this places the checkpoint exactly where the app expects it, as a
# build-time step (Railway's build environment has network access; the
# deployed container's runtime egress doesn't need to).
RUN python3 -c "from ultralytics import SAM; SAM('backend/weights/mobile_sam.pt')"

WORKDIR /app/backend

# Railway injects $PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
