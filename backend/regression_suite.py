"""
Single consolidated regression suite for the ring measurement pipeline.

Run this before every deploy - if it isn't all green, do not deploy.
This exists specifically because a stale-deployment mismatch (local code
fixed and verified, but an old build still running in production) was
mistaken for a code regression. This script is the one source of truth
for "is the CURRENT CODE actually healthy" - it takes zero dependency on
what's currently deployed anywhere.

For each case: expected status, actual status, diameter (if any), ground
truth (if known), absolute error (if scoreable), and reject reason (if
rejected) are all printed in one table. A case fails the run if its
actual status doesn't match its expected status - a photo that used to
work and now doesn't is a hard failure, not a "huh, interesting".

Paths are relative to this file's own directory (repo_root/test_set/...),
so this runs the same way regardless of who checks the repo out or where -
no hardcoded personal paths.

Usage: python3 regression_suite.py
Exit code 0 = all pass, 1 = at least one failure.
"""
import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import measure_ring

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SET_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "test_set")

# (label, filename, expected_status, ground_truth_mm_or_None, max_allowed_error_mm_or_None)
# Ground truth / expected outcomes match test_set/expected_vs_actual.csv.
CASES = [
    ("real_ring_A",            "real_ring_A_IMG_9783.jpg",              "ACCEPT", None,  None),
    ("real_ring_B",            "real_ring_B_IMG_9784.jpg",              "ACCEPT", 27.0,  1.0),
    ("synth_clean_baseline",   "synth_clean_baseline.png",              "ACCEPT", 17.40, 0.3),
    ("synth_specular",         "synth_specular_highlight.png",          "ACCEPT", 17.40, 0.3),
    ("synth_shadow_gradient",  "synth_shadow_gradient.png",             "ACCEPT", 17.40, 0.3),
    ("synth_low_contrast",     "synth_low_contrast_expect_retake.png",  "REJECT", None,  None),
    ("synth_textured_bg",      "synth_textured_background.png",         "ACCEPT", 17.40, 0.3),
    ("synth_heavy_blur",       "synth_heavy_blur_expect_retake.png",    "REJECT", None,  None),
    ("synth_decoy_adversarial","synth_decoy_object_adversarial.png",    "ACCEPT", 17.40, 0.3),
]
# NOTE: real_ring_A has no precise ground truth (ruler-measured, ~17mm coarse -
# see test_set/expected_vs_actual.csv) so it's checked for ACCEPT status only,
# not against a numeric tolerance.


def run():
    rows = []
    all_pass = True

    for label, filename, expected_status, gt, max_err in CASES:
        path = os.path.join(TEST_SET_DIR, filename)
        photo = cv2.imread(path)
        if photo is None:
            rows.append((label, expected_status, "MISSING_FILE", "-", "-", "-", "-", "-", "FAIL"))
            all_pass = False
            continue

        t0 = time.time()
        res = measure_ring(photo)
        elapsed_ms = (time.time() - t0) * 1000

        actual_status = "ACCEPT" if res.ok else "REJECT"
        diam_str = f"{res.diameter_mm:.2f}" if res.ok else "-"
        gt_str = f"{gt:.2f}" if gt is not None else "-"

        err_str = "-"
        status_ok = (actual_status == expected_status)

        if res.ok and gt is not None:
            err = abs(res.diameter_mm - gt)
            err_str = f"{err:.2f}"
            if max_err is not None and err > max_err:
                status_ok = False

        verdict = "pass" if status_ok else "FAIL"
        if not status_ok:
            all_pass = False

        reason_str = res.reason if not res.ok else "-"
        rows.append((label, expected_status, actual_status, diam_str, gt_str, err_str, reason_str, f"{elapsed_ms:.0f}ms", verdict))

    headers = ["case", "expected", "actual", "diam_mm", "gt_mm", "abs_err", "reject_reason", "time", "verdict"]
    widths = [26, 9, 9, 8, 7, 8, 18, 8, 7]
    print(" ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))
    for row in rows:
        print(" ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    print()
    if all_pass:
        print("ALL PASS - safe to deploy.")
    else:
        print("FAILURES PRESENT - do NOT deploy. A fix that breaks a previously-passing")
        print("case is not an accepted fix, per the project's own regression policy.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run())
