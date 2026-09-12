"""
Real-photo diagnostic CLI for the ring measurement pipeline.

Deliberately NOT a polished UI at this stage: what's needed right now is
full visibility into WHY a measurement succeeded or failed on a real
photo (marker corners, side_ratio, blur score, per-family diameter
estimates, spread) - not a pretty mobile card. See build_marker.py for
the printable calibration marker.

Usage:
    python3 analyze_photo.py path/to/photo.jpg [--gt 17.42] [--label "Ring A overhead"]

Appends one row to test_set_log.csv with the schema:
    label, photo, ground_truth_mm, app_diameter_mm, abs_error_mm,
    detection_spread_mm, decision, reason, side_ratio, blur_score,
    n_families, family_estimates, processing_time_ms
"""
import sys
import csv
import time
import argparse
import os
import cv2
import numpy as np
from pipeline import measure_ring, detect_and_rectify, blur_score, DETECTOR, find_marker_corners

LOG_PATH = "test_set_log.csv"
LOG_FIELDS = ["label","photo","ground_truth_mm","app_diameter_mm","abs_error_mm",
              "detection_spread_mm","decision","reason","side_ratio","blur_score",
              "n_families","family_estimates","processing_time_ms"]

def compute_side_ratio(photo):
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    # IMPORTANT: use the SAME marker-selection logic as the measurement
    # pipeline (find_marker_corners, filtered by exact MARKER_ID). An
    # earlier version of this function used corners[0][0] directly, which
    # could pick a DIFFERENT marker than the one measure_ring() actually
    # used if more than one ArUco-like tag were in frame - silently
    # desynchronizing the logged side_ratio from the real measurement.
    marker_corners_px = find_marker_corners(gray)
    if marker_corners_px is None:
        return None
    d = marker_corners_px
    top = np.linalg.norm(d[0]-d[1]); bottom = np.linalg.norm(d[3]-d[2])
    left = np.linalg.norm(d[0]-d[3]); right = np.linalg.norm(d[1]-d[2])
    return float(max(top,bottom,left,right) / max(1e-6, min(top,bottom,left,right)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo_path")
    ap.add_argument("--gt", type=float, default=None, help="Ground truth diameter in mm (caliper-measured, recorded BEFORE running this)")
    ap.add_argument("--label", type=str, default=None, help="Short label for the test set log, e.g. 'Ring A overhead'")
    args = ap.parse_args()

    photo = cv2.imread(args.photo_path)
    if photo is None:
        print(f"ERROR: could not read image at {args.photo_path}")
        sys.exit(1)

    label = args.label or os.path.basename(args.photo_path)

    t0 = time.time()
    res = measure_ring(photo)
    elapsed_ms = (time.time() - t0) * 1000

    side_ratio = compute_side_ratio(photo)
    sharpness = blur_score(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY))

    print(f"\n{'='*60}")
    print(f"Photo:        {args.photo_path}")
    print(f"Label:        {label}")
    print(f"Processing:   {elapsed_ms:.0f} ms")
    print(f"Blur score:   {sharpness:.1f}  (reject threshold: 40)")
    print(f"Side ratio:   {side_ratio if side_ratio else 'N/A (marker not found)'}")
    print(f"{'-'*60}")

    if res.ok:
        print(f"DECISION:     ACCEPT")
        print(f"Diameter:            {res.diameter_mm:.2f} mm")
        print(f"Cross-method spread: {res.detection_spread_mm:.2f} mm  (NOT full measurement uncertainty - see README)")
        print(f"Families:     {res.family_estimates}")
        if args.gt is not None:
            err = abs(res.diameter_mm - args.gt)
            print(f"Ground truth: {args.gt:.2f} mm")
            print(f"Abs. error:   {err:.2f} mm")
    else:
        print(f"DECISION:     RETAKE")
        print(f"Reason:       {res.reason}")
        if res.family_estimates:
            print(f"(partial) families found: {res.family_estimates}")

    # save debug overlay next to the input photo for visual inspection
    if res.debug_img is not None:
        dbg_path = os.path.splitext(args.photo_path)[0] + "_debug.png"
        cv2.imwrite(dbg_path, res.debug_img)
        print(f"Debug overlay saved: {dbg_path}")
    print(f"{'='*60}\n")

    # append to CSV test-set log
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        row = {
            "label": label,
            "photo": args.photo_path,
            "ground_truth_mm": args.gt if args.gt is not None else "",
            "app_diameter_mm": round(res.diameter_mm,3) if res.ok else "",
            "abs_error_mm": round(abs(res.diameter_mm-args.gt),3) if (res.ok and args.gt is not None) else "",
            "detection_spread_mm": round(res.detection_spread_mm,3) if res.ok else "",
            "decision": "ACCEPT" if res.ok else "RETAKE",
            "reason": res.reason or "",
            "side_ratio": round(side_ratio,3) if side_ratio else "",
            "blur_score": round(sharpness,1),
            "n_families": len(res.family_estimates),
            "family_estimates": str({k: round(v,3) for k,v in res.family_estimates.items()}),
            "processing_time_ms": round(elapsed_ms,1),
        }
        w.writerow(row)
    print(f"Logged to {LOG_PATH}")

if __name__ == "__main__":
    main()
