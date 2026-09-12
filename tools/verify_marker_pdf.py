"""
Verifies the PHYSICAL scale of a rendered ring_marker_20mm.pdf by rasterizing it
at a known DPI and measuring pixel distances - not by re-reading the
generator source. The whole point is to catch conversion bugs between
"what the code intended" and "what actually printed", the way the
wkhtmltopdf issue was found.

Run: python3 verify_marker_pdf.py
Exits non-zero and prints FAIL if either check is off by more than 0.3mm.
"""
import subprocess
import numpy as np
from PIL import Image
import sys

DPI = 300
MM_PER_PX = 25.4 / DPI
TOLERANCE_MM = 0.3

def rasterize():
    subprocess.run(["pdftoppm", "-r", str(DPI), "ring_marker_20mm.pdf", "verify_render", "-png"],
                    check=True, capture_output=True)
    return Image.open("verify_render-1.png").convert("RGB")

def measure_square(arr):
    """Find the black 20x20mm reference square (hollow rect) drawn in the
    SCALE CHECK block - distinct from the ArUco marker's filled bits by
    being a simple outline; we detect it by looking at the known region
    via the ruler's black pixels bounding box heuristically. To keep this
    robust and simple, we scan for the largest hollow-square-like contour
    using OpenCV instead of hand-rolled heuristics."""
    import cv2
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # looking for a roughly-square, roughly axis-aligned outline sized
        # near 20mm (in px at this DPI), NOT the small ArUco module rects
        expected_px = 20/MM_PER_PX
        if abs(w-expected_px) < 15 and abs(h-expected_px) < 15 and 0.9 < w/max(h,1) < 1.1:
            candidates.append((w,h,x,y,area))
    return candidates

def main():
    img = rasterize()
    arr = np.array(img)

    sq_candidates = measure_square(arr)
    if not sq_candidates:
        print("FAIL: could not locate the 20mm scale-check square in the rendered PDF")
        sys.exit(1)

    # pick the most square-like / largest candidate
    w,h,x,y,area = max(sq_candidates, key=lambda c: c[4])
    sq_w_mm = w*MM_PER_PX
    sq_h_mm = h*MM_PER_PX
    print(f"Scale-check square measured: {sq_w_mm:.3f} x {sq_h_mm:.3f} mm (expected 20.0 x 20.0)")

    ok = abs(sq_w_mm-20.0) < TOLERANCE_MM and abs(sq_h_mm-20.0) < TOLERANCE_MM
    if ok:
        print(f"PASS: within {TOLERANCE_MM}mm tolerance")
    else:
        err_pct = (sq_w_mm/20.0 - 1)*100
        print(f"FAIL: scale error {err_pct:+.1f}% - DO NOT PRINT/USE this PDF for measurement")
        sys.exit(1)

    # also sanity-check the ArUco marker itself decodes correctly at this scale
    import cv2
    DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(DICT, params)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and 0 in ids.flatten():
        idx = list(ids.flatten()).index(0)
        c = corners[idx][0]
        marker_w_mm = np.linalg.norm(c[0]-c[1]) * MM_PER_PX
        print(f"ArUco marker detected: ID=0, measured side ~{marker_w_mm:.3f} mm (expected 20.0)")
        if abs(marker_w_mm-20.0) > TOLERANCE_MM:
            print(f"FAIL: ArUco marker itself is off-scale by {(marker_w_mm/20-1)*100:+.1f}%")
            sys.exit(1)
    else:
        print("FAIL: ArUco marker ID=0 not detected in rendered PDF")
        sys.exit(1)

    print("\nPDF geometry passed scale verification. Verify the physical print")
    print("before use: even correct PDF geometry can be printed wrong if the")
    print("print dialog uses \"Fit to page\" instead of \"Actual size / 100%\".")

if __name__ == "__main__":
    main()
