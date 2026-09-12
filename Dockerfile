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
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend

# Railway injects $PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
