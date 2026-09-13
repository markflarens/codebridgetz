"""
Single consolidated regression suite for the ring measurement pipeline.

Run this before every deploy - if it isn't all green, do not deploy.

PORTABILITY NOTE (fixed after a real failure): an earlier version of this
script hardcoded this sandbox's own absolute paths
(/home/claude/ring_test/...). That meant a "9/10 pass" run reported from
inside this sandbox could not be reproduced by anyone else running the
exact same script against the exact same shipped repo - it would fail
with MISSING_FILE on every case outside this one machine. Paths below are
now resolved relative to this script's own location, pointing at the
test_set/ folder shipped alongside backend/ in the same delivery - so
`cd backend && python3 regression_suite.py` works from a fresh clone of
the repo, not just from the original sandbox that produced it.

For each case: expected status, actual status, diameter (if any), ground
truth (if known), absolute error (if scoreable), and reject reason (if
rejected) are all printed in one table. A case fails the run if its
actual status doesn't match its expected status - a photo that used to
work and now doesn't is a hard failure.

Usage: python3 regression_suite.py   (run from inside backend/)
Exit code 0 = all pass, 1 = at least one failure.
"""
import ast
import io
import os
import sys
import time
import tokenize
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline
from pipeline import measure_ring

_HERE = os.path.dirname(os.path.abspath(__file__))
TEST_SET_DIR = os.path.normpath(os.path.join(_HERE, "..", "test_set"))

# (label, filename in test_set/, expected_status, ground_truth_mm_or_None, max_allowed_error_mm_or_None)
#
# REAL-RING DIVERSITY: only two real rings are in this set today (A: thin
# steel band, mild angle; B: thick reflective keyring-style ring). That is
# not enough to claim the algorithm generalizes across ring types - it
# only demonstrates it works on these two specific rings. To actually test
# generalization, add several more REAL photos (not synthetic renders) of
# visually different rings: different diameters, band widths, finishes
# (matte vs. polished vs. brushed), reflectivity, and backgrounds. For
# each new photo: measure the ring's true diameter with calipers/a ruler
# BEFORE running it through the app (ground truth must be recorded
# independently of this pipeline's own output - see the policy note
# below), then add one row here with that filename and ground truth.
CASES = [
    # Expected status changed back from REJECT to ACCEPT, for the second
    # time, as a documented CONSEQUENCE of the geometry-primary redesign
    # (see pipeline.py's module docstring, "DETECTION ARCHITECTURE
    # (GEOMETRY-PRIMARY REVISION)") - not a fix tuned to make this one
    # photo pass. History, for anyone reading this later: this ring's band
    # is heavily hammered/textured AND has a strong specular highlight
    # bisecting the visible hole interior itself. An earlier revision
    # required the independent color-topology detector ("holecolor") to
    # confirm every accepted cluster (has_topological_confirmation); that
    # highlight splits the true hole's background-colored region into a
    # non-convex crescent that failed holecolor's own circularity filter,
    # so this photo REJECTed even though every edge-based family
    # (canny/adaptive/otsu) still found and tightly agreed on the correct
    # boundary. That was reported at the time as an intentional, accepted
    # trade-off of a stricter but more conservative system - but it turned
    # out, in combination with the same mandatory-topological-confirmation
    # rule rejecting other normal photos for the same underlying reason
    # (a local shadow/highlight breaking the *color* match, not the
    # *boundary*), to make the whole pipeline too conservative in exactly
    # the way this redesign was asked to fix: color/background similarity
    # is now a weak secondary cue, never a requirement, and boundary
    # coverage is evaluated with partial-arc tolerance (edge_support_-
    # fraction) rather than requiring one contour to already be fully
    # closed. Under that rule this photo now ACCEPTs from geometry alone,
    # at ~17.07mm - consistent with the edge families' own long-standing
    # measurement and this ring's previously-recorded ~17mm ground truth,
    # not a new or different number produced by loosening anything to
    # match it.
    ("real_ring_A_9783",        "real_ring_A_IMG_9783.jpg",             "ACCEPT", 17.0,  1.0),
    ("real_ring_B_9784",        "real_ring_B_IMG_9784.jpg",             "ACCEPT", 27.0,  1.0),
    # NOT a new/unseen ring - this is the SAME photo as real_ring_B_9784
    # above, downscaled ~2.06x (3024x4032 -> 1466x1956, matching a real
    # user-uploaded photo's resolution) before being saved. Added as an
    # honestly-labeled regression case after this exact file was proven to
    # reproduce a real failure: the pre-multi-peak pipeline rejected it
    # (RING_EDGE_UNSTABLE, an honest miss); an intermediate version of the
    # multi-peak/cross-family-consensus rework instead ACCEPTED it at a
    # confident-looking but wrong 23.4mm (detection_spread 0.13mm) against
    # this ring's real 27mm - a regression from "honestly unsure" to
    # "confidently wrong" that a diameter-blind test (this one expects
    # REJECT, not a specific wrong number) is specifically designed to
    # catch. Do not "fix" this case by tuning candidate selection toward
    # 27mm - the point is that this resolution is outside what this
    # pipeline's pixel-domain thresholds are validated for (see
    # RESOLUTION_TOO_LOW in pipeline.py's detect_and_rectify), so an honest
    # retake is the correct, generalizable answer - not a correct-looking
    # guess.
    ("real_ring_B_downscaled",  "real_ring_B_downscaled_2x_resolution_check.png", "REJECT", None, None),
    ("synth_clean_baseline",    "synth_clean_baseline.png",             "ACCEPT", 17.40, 0.3),
    ("synth_specular",          "synth_specular_highlight.png",         "ACCEPT", 17.40, 0.3),
    ("synth_shadow_gradient",   "synth_shadow_gradient.png",            "ACCEPT", 17.40, 0.3),
    ("synth_low_contrast",      "synth_low_contrast_expect_retake.png", "REJECT", None,  None),
    ("synth_textured_bg",       "synth_textured_background.png",        "ACCEPT", 17.40, 0.3),
    ("synth_heavy_blur",        "synth_heavy_blur_expect_retake.png",   "REJECT", None,  None),
    ("synth_wide_scene",        "synth_wide_scene.png",                 "ACCEPT", 17.40, 1.0),
    ("synth_decoy_adversarial", "synth_decoy_object_adversarial.png",   "ACCEPT", 17.40, 0.3),
]


def check_no_ground_truth_leakage():
    """
    Enforce, mechanically rather than by promise, that pipeline.py never
    reads or branches on this file's ground-truth values or filenames.
    Candidate selection / threshold decisions must come from geometry and
    cross-method agreement on whatever photo is actually being measured -
    never from already knowing the right answer for a specific test photo.
    This project shipped exactly that mistake once (a "fix" tuned by
    checking its output against one photo's known diameter, reverted after
    it turned out to silently give a wrong answer - see git history around
    commits 3e07ad8 / 0767fc5) - this check exists so a future change
    can't reintroduce it unnoticed.
    """
    pipeline_src_path = os.path.join(os.path.dirname(os.path.abspath(pipeline.__file__)), "pipeline.py")
    with open(pipeline_src_path, "r") as f:
        src = f.read()

    # Scan CODE only, not prose: this file's own docstrings legitimately
    # narrate past findings using real numbers/filenames (e.g. "the true
    # peak landed within 0.2mm of the ~27mm ground truth" as an
    # explanation of why a threshold is calibrated the way it is) - that's
    # documentation, not a control-flow dependency on ground truth, and
    # flagging it would just teach people to stop writing honest comments.
    # Strip every docstring and comment before checking; what remains is
    # actual executable code, which is where a real leak would live.
    code_src = src
    try:
        tree = ast.parse(src)
        docstrings = [ast.get_docstring(tree, clean=False)]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstrings.append(ast.get_docstring(node, clean=False))
        for doc in docstrings:
            if doc:
                code_src = code_src.replace(doc, "", 1)
        code_src = tokenize.untokenize(
            tok for tok in tokenize.generate_tokens(io.StringIO(code_src).readline)
            if tok.type != tokenize.COMMENT
        )
    except Exception as e:
        print(f"WARNING: could not strip comments/docstrings for leakage check ({e}); scanning raw source instead.")
        code_src = src

    leaks = []
    for label, filename, _, gt, _ in CASES:
        # Ground-truth mm values, formatted the same ways they'd plausibly
        # be typed as a literal (2dp, or bare like "27").
        if gt is not None:
            for form in (f"{gt:.2f}", f"{gt:g}"):
                if form in code_src:
                    leaks.append(f"{label}: ground-truth value {form!r} appears literally in pipeline.py")
        # Filenames/labels naming a specific test photo.
        stem = os.path.splitext(filename)[0]
        for needle in (filename, stem, label):
            if needle in code_src:
                leaks.append(f"{label}: identifier {needle!r} appears in pipeline.py (looks like a per-photo special case)")

    return leaks


def run():
    rows = []
    all_pass = True

    leaks = check_no_ground_truth_leakage()
    if leaks:
        print("FATAL: possible ground-truth / per-photo leakage into pipeline.py:")
        for l in leaks:
            print(f"  - {l}")
        print("Candidate selection must not know a test photo's identity or answer.")
        print("Fix pipeline.py (or this check, if it's a genuine false positive) before trusting any pass below.\n")
        all_pass = False

    if not os.path.isdir(TEST_SET_DIR):
        print(f"FATAL: test_set directory not found at {TEST_SET_DIR}")
        print("This script expects to run from inside backend/, with a sibling test_set/ folder.")
        return 1

    for label, filename, expected_status, gt, max_err in CASES:
        path = os.path.join(TEST_SET_DIR, filename)
        photo = cv2.imread(path)
        if photo is None:
            rows.append((label, expected_status, "MISSING_FILE", "-", "-", "-", path, "-", "FAIL"))
            all_pass = False
            continue

        t0 = time.time()
        res = measure_ring(photo)
        elapsed_ms = (time.time() - t0) * 1000

        # Invariant: a supported photo returns a diameter, an unsupported/
        # ambiguous one returns a specific reason - never both, never
        # neither (an accepted result with no number, or a rejection with
        # no reason, is a contract violation regardless of whether the
        # status itself happens to match what was expected).
        if res.ok and res.diameter_mm is None:
            rows.append((label, expected_status, "ACCEPT", "-", "-", "-", "MALFORMED: ok=True but diameter_mm is None", f"{elapsed_ms:.0f}ms", "FAIL"))
            all_pass = False
            continue
        if not res.ok and not res.reason:
            rows.append((label, expected_status, "REJECT", "-", "-", "-", "MALFORMED: not ok but no reason code", f"{elapsed_ms:.0f}ms", "FAIL"))
            all_pass = False
            continue

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
    widths = [26, 9, 9, 8, 7, 8, 20, 8, 7]
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
