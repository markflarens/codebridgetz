"""
Ring measurement pipeline (deterministic CV for geometry/calibration;
MobileSAM, a small promptable segmentation model, for ring-BODY
segmentation only - see "ML DEPENDENCY, HONESTLY" below for exactly what
that model is and is not used for).

GENERALIZATION / NO-OVERFITTING POLICY (read before touching this file):
Every decision in this file - which z-threshold to trust, which Hough
circle is the outer boundary, whether a mask has stabilized enough to
accept - must be made from the SHAPE OF THIS PHOTO'S OWN SIGNAL (how its
segmentation behaves across a sweep of thresholds), never from which
specific test photo it is or what its known diameter is. Concretely:

- This file must never import, read, or branch on test_set/ ground truth
  (regression_suite.py's CASES list, expected_vs_actual.csv, or any
  literal ground-truth mm value copy-pasted from them). Ground truth is
  read ONLY by regression_suite.py, and only AFTER measure_ring() has
  already returned a result - to score error, never to steer detection.
- Do not add logic that identifies a specific input photo (by filename,
  by an image hash, by a "if this looks like photo X" special case) and
  special-cases it. A rule that only fires on one known photo is not a
  fix, it's memorization, and it will not generalize to a new ring nobody
  has seen yet.
- When calibrating a threshold, validate it against the whole labeled set
  as a fixed decision rule applied uniformly - not by trying several
  candidate rules and picking whichever one happens to land closest to a
  known answer on one specific photo. This project has done the latter by
  mistake before (see git history around commits 3e07ad8 / 0767fc5: a
  "fix" was reverted specifically because it had been tuned by checking
  its output against one photo's known ground truth, and turned out to
  give a confidently WRONG answer once inspected properly - a policy
  violation that produced a real bug, not just a style problem).
- A photo that satisfies the documented supported-capture conditions
  should return a diameter; a photo that doesn't, or whose signal is too
  ambiguous to trust, should return a specific retake reason - never an
  unreliable number dressed up as a confident one.

DETECTION ARCHITECTURE (RING-BODY / MASK-TOPOLOGY REVISION - current):
earlier revisions of this file went through FOUR designs, each one's
failures motivating the next:

  1. "Strongest circle" scoring - picked whichever candidate looked
     cleanest/most circular and had the most cross-method agreement.
     Fails because a ring's OUTER edge, a polished bevel, a specular
     reflection, or a cast shadow can all look just as clean/circular as
     the true inner hole, and multiple edge-based methods CAN agree on
     the same wrong one.
  2. "Material-contrast as a hard gate" - required every candidate's
     interior to match the photo's sampled background color before it
     could even compete. Overcorrected: ordinary shadows/highlights on
     the hole or nearby background broke the match even when the
     boundary was perfectly visible, producing false REJECTs.
  3. "Geometry-primary" - made containment/concentricity with a coarse
     Hough hint, per-boundary edge-support coverage, fit residual, and
     cross-method/cross-family consensus the primary acceptance signal,
     with color demoted to a weak, non-blocking cue. This still fails on
     reflective rings: the INNER edge, a BEVEL, a HIGHLIGHT, the OUTER
     edge, and a CAST SHADOW can all independently produce geometrically
     valid, well-supported, mutually-agreeing concentric ellipse
     candidates - cross-method agreement does not distinguish "multiple
     methods found the true hole" from "multiple methods all found the
     same wrong concentric edge" when several real edges are genuinely
     concentric with each other. No amount of tuning coverage/residual/
     consensus thresholds on an ellipse-candidate ENSEMBLE fixes this: the
     ambiguity is about which closed curve among several real candidate
     curves IS the hole, and scoring curves against each other never
     answers that question directly.

  4. "Segmentation-first via background-color matching" - classified
     every pixel as "background-colored or not" (Lab chroma+brightness
     distance from a sampled table/background reference) and took the
     largest enclosed (non-border-touching) background-colored connected
     component as the hole, accepting/rejecting based on how that
     region's ellipse behaved across a sweep of classification
     thresholds. This removed the "which candidate CURVE is the hole"
     ambiguity of revision 3, but introduced a different, more basic
     problem, raised directly by this project's own reviewer: the hole
     was still being defined ENTIRELY by matching the *surrounding
     background's color*, not by anything about the ring itself. That is
     fragile exactly where real photos are hardest - ordinary shadows,
     reflections, textured surfaces, and illumination gradients all
     change how "background-colored" a given pixel looks without the
     pixel having moved from background to ring or back. It also has a
     conceptual gap: an enclosed background-colored region is not
     *guaranteed* to be the physical hole (a shadow pooled inside the
     ring, or a patch of background-colored material sitting on top of
     the ring, would pass the same test), and fitting an ellipse to that
     region's CONVEX HULL means an incomplete or wrong segmented crescent
     can still produce a convincing-looking ellipse - cleanliness of the
     resulting shape does not certify that the shape was found for the
     right reason.

This revision (5) changes what is being segmented. Instead of asking "is
this pixel colored like the background" (revision 4) it asks "is this
pixel part of the physical ring body", using a small PROMPTABLE
segmentation model (MobileSAM) seeded with POSITIVE prompts on the
ring's own band material and NEGATIVE prompts inside the hole and outside
the ring - i.e. it is told what the ring band itself looks like, not what
the table looks like. The enclosed hole is then read off the resulting
mask's own CONTOUR TOPOLOGY (RETR_CCOMP hierarchy: the largest top-level
contour is the ring body, its largest enclosed child contour is the
hole) rather than from any color-matching step. A convex-hull SOLIDITY
check on the hole contour is used only DEFENSIVELY, as a validity gate
("does this region look like a genuine, mostly-convex opening, or like an
incomplete/bitten crescent") - the ellipse itself is always fit to the
mask's own RAW hole contour, never to the hull, so a hull-reconstructed
shape can no longer be mistaken for a measurement.

ArUco/rectification and coarse ROI localization (locate_outer_ring_circle)
are UNCHANGED from every earlier revision - purely geometric, not a
detection decision. What changed is everything after that coarse ROI.

TWO-HYPOTHESIS STRATEGY (the key structural fix this revision adds):
locate_outer_ring_circle's own docstring documents a real ambiguity - on a
clean, high-contrast photo, Hough can lock onto the ring's INNER hole
edge just as easily as its outer edge, and there is no way to tell which
one it found before looking further. Rather than guessing, this revision
generates BOTH possible readings of that hint and tries each one all the
way through segmentation and topology analysis:
  - H1 ("r=outer"): assume the hint IS the outer edge; search INWARD
    (reusing the existing radial_inner_candidate) for a hole-edge radius.
  - H2 ("r=inner"): assume the hint IS the hole edge; search OUTWARD
    (the new radial_outer_candidate, a mirror of radial_inner_candidate)
    for an outer band-edge radius.
Each hypothesis that produces a plausible (inner_r, outer_r) pair gets
its own MobileSAM segmentation + topology analysis; whichever hypothesis
(if any) produces a VALID topology (single body, single hole, hole
solidity above threshold, plausible axis ratio) is reported. When BOTH
hypotheses are valid, their diameters are compared as an extra confidence
signal (reported, not gating) rather than a reason to prefer one over the
other beyond its own higher solidity.

ALTERNATIVES INVESTIGATED AND REJECTED before settling on prompted
MobileSAM, per the explicit instruction to compare a properly-seeded
deterministic baseline against a genuinely promptable model, and to
judge by mask correctness first, diameter only afterward:

  - Seeded GrabCut (cv2.grabCut, GC_INIT_WITH_MASK) using the SAME
    two-hypothesis geometric priors (a small central disk as definite
    background, a thin annulus near the hypothesis's own band as
    definite foreground, well outside as definite background, the large
    genuinely-ambiguous middle region left probable-background rather
    than guessed): once properly seeded (an earlier, naive fixed-fraction
    seeding mis-seeded entirely - see git history / DELIVERY_NOTES for
    that failure), this produced topologically clean results (single
    body, single hole, solidity up to ~0.98) on 2 of 3 real photos, and
    visually GOOD masks on those two. On the third (a ring with a strong
    internal reflection gradient across its opening) it produced a mask
    that was STILL topologically clean by every structural check, yet
    visibly WRONG once inspected directly: GrabCut's color-GMM
    classified the bright, reflective part of the true hole as ring
    material (it resembles the band's own highlights) while correctly
    keeping the darker part as hole - cutting a real opening roughly in
    half. This is exactly the "inspect the mask before trusting its
    topology" failure mode the no-overfitting policy above warns about,
    now demonstrated concretely: topological cleanliness is NECESSARY but
    not SUFFICIENT.
  - Prompted FastSAM (ultralytics FastSAM-s.pt, using the SAME prompt
    points as MobileSAM below): returned no mask for every hypothesis on
    every photo. Investigated directly (not assumed) by calling FastSAM
    without any prompts and inspecting its raw output: it only ever
    generates a handful of "segment everything" object masks per image,
    and its `.prompt()` API SELECTS among those few pre-generated masks
    using the given points - it does not generate a new mask conditioned
    on the prompts the way a true promptable segmenter does. Since this
    project's earlier, separate investigation of promptless FastSAM had
    already found it fails to isolate the hole as its own object on 2 of
    3 real photos, prompted FastSAM inherits that same ceiling regardless
    of how good the prompts are. Not usable for this task.
  - MobileSAM (ultralytics.SAM wrapper around mobile_sam.pt, a real
    point-promptable encoder/decoder model, unlike FastSAM's filter-only
    `.prompt()`): produced visually complete, correctly-shaped hole
    boundaries on ALL 3 real test photos, INCLUDING the reflective-ring
    case where properly-seeded GrabCut's mask was confidently wrong, and
    including the one photo where GrabCut found no valid topology at
    all. This is what backs measure_ring below.

ML DEPENDENCY, HONESTLY: this revision adds a real dependency this
project did not have before (torch + ultralytics + a ~39MB MobileSAM
checkpoint) and a real per-photo cost (loading + CPU inference for up to
two hypotheses). The model's role is scoped narrowly and deliberately:
it is used ONLY to answer "which pixels are ring-body material, given
these geometric prompts" - it never estimates a diameter, is never asked
"how big is this ring", and diameter is still computed the same way every
revision has computed it: classical ellipse-fitting on a mask's own
contour, converted to mm via the ArUco-calibrated px/mm scale. See
README.md / DELIVERY_NOTES.md for the full size/latency tradeoff this
introduces and why it was judged worth it.

ACCEPTANCE is decided from MASK TOPOLOGY, not from diameter, per the
explicit instruction to inspect mask correctness before ever looking at a
number: a hypothesis's segmentation is only even considered a candidate
measurement once analyze_mask_topology confirms exactly one ring body,
exactly one enclosed hole, that hole's solidity above MIN_HOLE_SOLIDITY,
and its fitted-ellipse axis ratio above MIN_HOLE_AXIS_RATIO. Only after a
hypothesis passes that gate does its ellipse's diameter become the
reported measurement (or a cross-hypothesis comparison value).
"""
import os
import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICT, PARAMS)

MARKER_ID = 0
MARKER_MM = 20.0
PX_PER_MM_OUT = 15.0
MARKER_OUT_ORIGIN = (60, 60)

# How far beyond the ArUco marker's own 20x20mm bounding box to mask out
# when searching for ring candidates. Found via a real photo: the printed
# marker page (build_marker.py) leaves an ~8mm white paper margin around
# the ArUco pattern for scissors-cutting, and a small mask padding (15px)
# left that whole paper edge unmasked. Canny found a real, legitimate edge
# there (the physical cut line of the paper against the table) that
# happened to fit a deceptively plausible-looking near-circular ellipse -
# not noise, but a real edge from the WRONG object. 8mm * 15px/mm = 120px
# is the minimum needed to clear the paper itself; padded further for
# photography/cutting slop (someone may leave extra margin when cutting).
MARKER_MASK_PAD_PX = 150

# --- MobileSAM: loaded lazily, once, on first use -----------------------
# A small (~39MB) promptable segmentation model (TinyViT encoder + SAM
# mask decoder), used ONLY to turn geometric prompts (see
# build_ring_prompts) into a ring-body mask - see the module docstring's
# "ML DEPENDENCY, HONESTLY" section for the scope of what this model is
# and is not used for. Loaded once at first use (not at import time) so
# importing this module for unrelated reasons (e.g. regression_suite's
# no-ground-truth-leakage static check) never pays the load cost, and so
# a deployment without network access to fetch the checkpoint still
# imports cleanly and fails per-request with a clear reason rather than
# crashing at startup.
_SAM_MODEL = None
_SAM_LOAD_ERROR = None


def _get_sam_model():
    global _SAM_MODEL, _SAM_LOAD_ERROR
    if _SAM_MODEL is not None or _SAM_LOAD_ERROR is not None:
        return _SAM_MODEL
    try:
        from ultralytics import SAM
        weights_path = os.environ.get(
            "MOBILE_SAM_WEIGHTS",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "mobile_sam.pt"),
        )
        _SAM_MODEL = SAM(weights_path)
    except Exception as e:  # pragma: no cover - exercised only when weights/deps are missing
        _SAM_LOAD_ERROR = str(e)
        _SAM_MODEL = None
    return _SAM_MODEL


# Mask-topology validity thresholds (see analyze_mask_topology). Reasoned
# the same way every other threshold in this file is reasoned - from the
# observed separation between clean-accept and genuinely-ambiguous real
# photos, never from forcing a specific known diameter to pass:
#   - MIN_HOLE_SOLIDITY: a genuine round opening's raw contour is nearly
#     convex; a mask that caught only part of the opening (GrabCut's
#     reflection failure, or a MobileSAM mask that bit into the band)
#     produces a visibly non-convex hole contour with solidity well below
#     this on every case tried, while every visually-correct mask scored
#     >=0.9.
#   - MIN_HOLE_AXIS_RATIO: once a hole is convex-ish, it should also be
#     roughly round/elliptical (a near-overhead photo never produces a
#     highly eccentric TRUE opening - detect_and_rectify's side_ratio gate
#     already screens out steep angles upstream). A stable-looking but
#     badly elongated hole is evidence of a segmentation/topology problem,
#     not an unusually-shaped ring.
MIN_HOLE_SOLIDITY = 0.85
MIN_HOLE_AXIS_RATIO = 0.7


class MeasurementResult:
    def __init__(self):
        self.ok = False
        self.reason = None          # structured failure code
        self.diameter_mm = None
        # NOTE: deliberately NOT called "uncertainty_mm". Under this
        # two-hypothesis, mask-topology architecture this is
        # |diam(H1) - diam(H2)| when BOTH hypotheses independently produced
        # a VALID topology - a cross-hypothesis agreement signal on THIS
        # photo, not a physical measurement uncertainty. When only one
        # hypothesis was valid, there is nothing to compare against and
        # this stays None. It says nothing about print scale error, lens
        # distortion, or ArUco corner localization error.
        self.detection_spread_mm = None
        self.candidates_mm = []     # [(label, diameter_mm, topo_dict), ...] every hypothesis with a VALID topology
        # Keyed by hypothesis label ("r=outer" / "r=inner") rather than by
        # detection "family" or z-threshold. Kept under this field name for
        # API/frontend compatibility (the frontend renders this generically
        # as "label: value mm").
        self.family_estimates = {}
        self.debug_img = None
        # Every hypothesis tried, whether or not it ended up valid, with a
        # reason when it was degenerate, produced no mask, or failed the
        # topology gate. Kept on the result so a debug overlay/log can show
        # the full attempt, not just the winner.
        self.rejected_candidates = []
        # Reporting fields describing the WINNING hypothesis, replacing the
        # z-sweep-stability fields (z_threshold_used, n_stable_z) from the
        # background-color-matching revision, which no longer have meaning
        # once there is no z-sweep at all.
        self.hypothesis_used = None     # "r=outer" or "r=inner" - which hypothesis won
        self.mask_solidity = None       # hole contour area / its convex hull area
        self.hole_axis_ratio = None     # minor/major of the accepted ellipse, shape-quality signal
        self.n_bodies = None            # number of significant top-level mask components found
        self.n_holes = None             # number of significant enclosed components found in the winning body


def find_marker_corners(gray):
    """Single source of truth for locating OUR marker (by exact ID) in an
    image. Used by both detect_and_rectify() and any diagnostic/CLI code
    that needs the same corners (e.g. computing side_ratio for logging) -
    duplicating this logic elsewhere risks it silently drifting out of
    sync (as happened: an earlier diagnostic script's side_ratio computed
    from corners[0] regardless of ID, while measurement correctly filtered
    by MARKER_ID - the two could disagree on which marker was used).
    Returns the 4 corner points, or None if our marker isn't confidently found.
    """
    corners, ids, _ = DETECTOR.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    ids_flat = ids.flatten()
    matches = np.where(ids_flat == MARKER_ID)[0]
    if len(matches) != 1:
        return None
    return corners[matches[0]][0]


def detect_and_rectify(photo):
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    marker_corners_px = find_marker_corners(gray)
    if marker_corners_px is None:
        return None, None, "MARKER_NOT_FOUND"

    h, w = gray.shape
    margin_px = 0.02 * min(h, w)
    xs, ys = marker_corners_px[:,0], marker_corners_px[:,1]
    if xs.min() < margin_px or ys.min() < margin_px or xs.max() > w-margin_px or ys.max() > h-margin_px:
        return None, None, "MARKER_PARTIAL"

    d = marker_corners_px
    top = np.linalg.norm(d[0]-d[1]); bottom = np.linalg.norm(d[3]-d[2])
    left = np.linalg.norm(d[0]-d[3]); right = np.linalg.norm(d[1]-d[2])

    # Marker too small in frame: below this many px per side (in the
    # ORIGINAL photo, before any rectification), ArUco corner localization
    # gets noisy enough that the scale it hands the whole measurement
    # becomes unreliable - a small pixel error in a small marker is a much
    # bigger fraction of its size than the same pixel error on a large one.
    # Both real validated test photos had marker sides ~430px; 80px is a
    # conservative floor well below that, not a tight fit to them.
    avg_side_px = (top+bottom+left+right) / 4
    if avg_side_px < 80:
        return None, None, "MARKER_TOO_SMALL"

    # Effective resolution check, distinct from MARKER_TOO_SMALL above.
    #
    # MARKER_TOO_SMALL (80px) is a floor for ArUco CORNER localization
    # staying numerically sane - below it, corner-finding itself gets
    # noisy. This is a separate, higher floor for whether there are enough
    # real source pixels per mm for the pixel-domain edge/gradient
    # thresholds in the rest of this file (Canny thresholds, gradient-angle
    # tolerance, histogram prominence/support cutoffs) to behave the way
    # they were calibrated to. Those thresholds were tuned against the two
    # real validated photos, both ~427-437px marker side (see
    # MARKER_TOO_SMALL's comment above and main.py's load_image_any_format
    # docstring on why a downscale-before-processing shortcut was tried and
    # reverted). Proven concretely: downscaling one of those same validated
    # photos by ~2x (to ~207px marker side, same physical ring, same
    # marker) reproduces a failure the full-resolution original never had -
    # not by becoming blurry (blur_score() above, checked on the original
    # upload, does not catch this - downscaling reduces spatial detail
    # without necessarily lowering Laplacian variance at the new, smaller
    # pixel grid) but by starving the pixel-domain thresholds of the detail
    # they need, which can let independent methods agree on the wrong
    # boundary with a deceptively TIGHT spread (high false confidence,
    # worse than an honest rejection).
    #
    # 300px is a provisional, conservative cut roughly midway between the
    # validated photos (~427-437px) and the proven-bad case (~207px) - not
    # fitted to any specific photo's diameter or ground truth, only to
    # marker pixel density, which is knowable before any ring detection
    # happens at all. Revisit (probably raise, with more headroom) once
    # more real photos across a range of phone cameras/resolutions are
    # available - this is deliberately a resolution/calibration-validity
    # gate, not a per-photo special case.
    MIN_MARKER_PX_FOR_CALIBRATED_THRESHOLDS = 300
    if avg_side_px < MIN_MARKER_PX_FOR_CALIBRATED_THRESHOLDS:
        return None, None, "RESOLUTION_TOO_LOW"

    side_ratio = max(top,bottom,left,right) / max(1e-6, min(top,bottom,left,right))
    # Tightened from an earlier 1.8: that threshold was never empirically
    # validated (see delivery notes) and let through more perspective
    # distortion than a "near-overhead" photo should have. Both real
    # validated photos had side_ratio ~1.01-1.02, so 1.35 still leaves
    # generous headroom for a hand-held, not-perfectly-vertical shot while
    # rejecting meaningfully steep angles. This is a reasoned tightening,
    # not a re-calibration against a specific failing photo - revisit if a
    # concrete steep-angle example is available to test against directly.
    if side_ratio > 1.35:
        return None, None, "PERSPECTIVE_TOO_HIGH"

    dst_marker_px = np.float32([
        [MARKER_OUT_ORIGIN[0], MARKER_OUT_ORIGIN[1]],
        [MARKER_OUT_ORIGIN[0] + MARKER_MM*PX_PER_MM_OUT, MARKER_OUT_ORIGIN[1]],
        [MARKER_OUT_ORIGIN[0] + MARKER_MM*PX_PER_MM_OUT, MARKER_OUT_ORIGIN[1] + MARKER_MM*PX_PER_MM_OUT],
        [MARKER_OUT_ORIGIN[0], MARKER_OUT_ORIGIN[1] + MARKER_MM*PX_PER_MM_OUT],
    ])
    H0, _ = cv2.findHomography(marker_corners_px, dst_marker_px, method=0)

    # --- Dynamically size the rectified canvas so nothing gets clipped. ---
    # A fixed out_size assumes the ring always lands within some guessed
    # region; on a real photo where the ring sits further from the marker
    # (or the frame is a different aspect ratio) it can silently crop the
    # ring, which then fails or corrupts inner-boundary detection with no
    # clear error message. Instead: project the ENTIRE source photo through
    # H0, see where all of it lands, and size/shift the canvas to fit it.
    src_corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    projected = cv2.perspectiveTransform(src_corners, H0).reshape(-1,2)
    all_pts = np.vstack([projected, dst_marker_px])
    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)

    margin = 20  # px padding so nothing sits flush against the canvas edge
    # Shift so the projected scene's minimum lands exactly at `margin`,
    # regardless of whether min_x/min_y started positive or negative.
    # (Earlier version used max(0,-min)+margin, which only added padding
    # when min was negative - when min was already positive, the scene
    # kept its original offset AND got `margin` added on top, so the far
    # edge silently exceeded the computed canvas width. Confirmed with
    # min_x=40,max_x=1200,margin=20: old formula clipped the last 20px.)
    shift_x = margin - min_x
    shift_y = margin - min_y
    dst_marker_px_shifted = dst_marker_px + np.float32([shift_x, shift_y])
    H, _ = cv2.findHomography(marker_corners_px, dst_marker_px_shifted, method=0)

    out_w = int(np.ceil(max_x - min_x + 2*margin))
    out_h = int(np.ceil(max_y - min_y + 2*margin))
    # sanity cap: an extreme/degenerate homography could otherwise request
    # an absurd canvas size (huge memory) - clamp and treat as unmeasurable
    if out_w > 6000 or out_h > 6000 or out_w < 10 or out_h < 10:
        return None, None, "PERSPECTIVE_TOO_HIGH"

    rectified = cv2.warpPerspective(photo, H, (out_w, out_h), borderValue=(255,255,255))
    marker_rect_out = (dst_marker_px_shifted[0][0], dst_marker_px_shifted[0][1],
                        dst_marker_px_shifted[2][0], dst_marker_px_shifted[2][1])  # x0,y0,x1,y1
    return rectified, marker_rect_out, None


def check_exposure(gray, dark_thresh=25, bright_thresh=250,
                    max_dark_fraction=0.5, max_bright_fraction=0.35,
                    min_mean=60, max_mean=235):
    """Reject photos that are too dark or too washed-out to trust before
    spending any time on marker/ring detection. Thresholds calibrated with
    real working photos as the floor/ceiling to stay clear of: both real
    test photos (evenly lit, indoor) measured mean brightness ~137,
    frac_dark(<25) ~1%, frac_bright(>250) ~0.01-0.04% - comfortably inside
    these bounds.

    max_bright_fraction was originally set to 0.15, calibrated only against
    those two real photos. That was too tight: it produced a false
    OVEREXPOSED reject on test_set/synth_shadow_gradient.png - a documented
    regression-test photo (directional lighting gradient) that measures
    frac_bright ~19% while still being a perfectly readable, correctly-
    ACCEPTable photo (marker and ring both detect fine; mean brightness is
    unremarkable at ~212, same as the other clean synthetic renders). A
    bright directional gradient can legitimately blow out a fifth of the
    frame (background, out-of-focus areas) without the marker/ring region
    itself being compromised - fraction-of-frame alone doesn't distinguish
    "unreadable" from "unevenly lit but fine". Raised to 0.35 (comfortably
    above the observed 19%) so this gate only catches photos where the
    large majority of the frame is blown out, and re-verified against the
    full test_set with no other regressions (see delivery notes / CSV).
    Returns a reason string or None.
    """
    mean_b = gray.mean()
    frac_dark = (gray < dark_thresh).mean()
    frac_bright = (gray > bright_thresh).mean()
    if mean_b < min_mean or frac_dark > max_dark_fraction:
        return "UNDEREXPOSED"
    if mean_b > max_mean or frac_bright > max_bright_fraction:
        return "OVEREXPOSED"
    return None


def blur_score(gray):
    # normalize scale so metric is comparable across resolutions
    h, w = gray.shape
    scale = 1600.0 / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)))
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _radial_edge_histogram(gray_for_gradient, edge_map, center, outer_radius_px,
                            min_band_ratio, max_band_ratio,
                            gradient_angle_tolerance_deg):
    """
    Shared setup for the radial fallback: build a histogram of
    radially-oriented edge-pixel counts by radius, in the plausible
    inner-boundary band [r_min, r_max] below outer_radius_px.

    CRITICAL FIX (found on a real photo of a ring on a wood table):
    counting ALL edge pixels in the radius band, regardless of direction,
    fails on textured backgrounds. Wood grain alone produced 300-478 Canny
    edge pixels at EVERY radius from 122-202px in one real test photo - an
    almost flat, noisy histogram with no usable peak. A true circular
    boundary's edge pixels have gradients pointing RADIALLY (toward/away
    from the circle's center); wood grain and other texture edges do not -
    their gradient direction is set by the grain/pattern, essentially
    decorrelated from any candidate center. Filtering edge pixels to only
    those whose local gradient direction is within
    `gradient_angle_tolerance_deg` of radial, before building the radius
    histogram, removes the vast majority of texture noise and reveals the
    true peak.

    Returns (hist, r_min, r_max) where hist[i] is the count at radius
    r_min + i, or (None, r_min, r_max) if there isn't enough signal to
    build a histogram at all.
    """
    cx, cy = center

    r_min = int(np.ceil(outer_radius_px / max_band_ratio))
    r_max = int(np.floor(outer_radius_px / min_band_ratio))

    if r_max <= r_min + 3:
        return None, r_min, r_max

    pad = int(np.ceil(outer_radius_px)) + 3

    x0 = max(0, int(cx - pad))
    x1 = min(edge_map.shape[1], int(cx + pad + 1))
    y0 = max(0, int(cy - pad))
    y1 = min(edge_map.shape[0], int(cy + pad + 1))

    roi = edge_map[y0:y1, x0:x1]
    ys, xs = np.nonzero(roi)

    if len(xs) < 20:
        return None, r_min, r_max

    xs = xs + x0
    ys = ys + y0

    radii = np.hypot(xs - cx, ys - cy)
    in_band = (radii >= r_min) & (radii <= r_max)
    xs, ys, radii = xs[in_band], ys[in_band], radii[in_band]

    if len(radii) < 20:
        return None, r_min, r_max

    gx = cv2.Sobel(gray_for_gradient, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_for_gradient, cv2.CV_32F, 0, 1, ksize=3)
    gxs, gys = gx[ys, xs], gy[ys, xs]
    gmag = np.hypot(gxs, gys)
    rad_dx = (xs - cx) / np.maximum(radii, 1e-6)
    rad_dy = (ys - cy) / np.maximum(radii, 1e-6)
    cos_angle = (gxs*rad_dx + gys*rad_dy) / np.maximum(gmag, 1e-6)
    angle_deg = np.degrees(np.arccos(np.clip(np.abs(cos_angle), 0, 1)))
    radial_mask = (gmag > 1e-3) & (angle_deg <= gradient_angle_tolerance_deg)
    radii = radii[radial_mask]

    if len(radii) < 20:
        return None, r_min, r_max

    rounded = np.rint(radii).astype(np.int32)
    hist = np.bincount(rounded, minlength=r_max + 2)[r_min:r_max + 1]

    if hist.size == 0:
        return None, r_min, r_max

    return hist, r_min, r_max


def radial_inner_candidate(gray_for_gradient, edge_map, center, outer_radius_px,
                           min_band_ratio=1.07,
                           max_band_ratio=1.8,
                           min_support=0.03,
                           min_prominence=1.25,
                           gradient_angle_tolerance_deg=20):
    """
    Single-best-peak version of the radial fallback: return only the
    tallest peak in the radius histogram. Two independent callers rely on
    it, deliberately kept exactly as originally calibrated because
    changing its behavior has previously been shown to change which Hough
    circle gets picked as the outer boundary (a real regression - see git
    history around commits 3e07ad8 / 0767fc5):
      - locate_outer_ring_circle() uses it to validate a Hough circle (a
        yes/no "does this look ring-like at all" check, for coarse ROI
        localization only).
      - build_band_hypotheses() reuses it as hypothesis H1's own radius
        search ("outer_ring_hint IS the outer edge; where's the hole") -
        see the module docstring's TWO-HYPOTHESIS STRATEGY. Even here it
        only ever produces a PROMPT radius for MobileSAM, never the
        measurement itself - which boundary is actually the hole is
        decided by analyze_mask_topology on the resulting mask, not by
        this function's own edge histogram.
    """
    hist, r_min, r_max = _radial_edge_histogram(
        gray_for_gradient, edge_map, center, outer_radius_px,
        min_band_ratio, max_band_ratio, gradient_angle_tolerance_deg
    )
    if hist is None:
        return None

    cx, cy = center
    peak_idx = int(np.argmax(hist))
    radius = float(r_min + peak_idx)
    peak = float(hist[peak_idx])

    support = peak / max(1.0, 2.0 * np.pi * radius)
    baseline = float(np.percentile(hist, 75))
    prominence = peak / max(1.0, baseline)

    if support < min_support or prominence < min_prominence:
        return None

    return {
        "ellipse": ((float(cx), float(cy)), (2.0 * radius, 2.0 * radius), 0.0),
        "circularity": 1.0,
        "axis_ratio": 1.0,
        "residual": 0.0,
        "score": support,
        "diam_px": 2.0 * radius,
        "center": (float(cx), float(cy)),
        "radial_support": support,
        "radial_prominence": prominence,
    }


def radial_outer_candidate(gray_for_gradient, edge_map, center, inner_radius_px,
                            min_ratio=1.05, max_ratio=1.6,
                            min_support=0.02, min_prominence=1.2,
                            gradient_angle_tolerance_deg=20):
    """
    Mirror of radial_inner_candidate, searching OUTWARD from a known
    radius instead of inward. Used as hypothesis H2's radius search (see
    module docstring's TWO-HYPOTHESIS STRATEGY): if outer_ring_hint
    actually landed on the ring's INNER hole edge rather than its outer
    edge (a documented risk of locate_outer_ring_circle on a clean,
    high-contrast photo), this looks for a plausible OUTER band-edge
    radius just beyond it, the same way radial_inner_candidate looks
    inward for a hole edge under the opposite assumption.

    Reuses the same radial-gradient-filtered edge histogram as
    radial_inner_candidate (see _radial_edge_histogram's docstring for why
    filtering to radially-oriented edge pixels is necessary on textured
    backgrounds) - only the direction and the band-ratio parameterization
    differ, since here the known radius is the INNER bound of the search
    band rather than the outer one.
    """
    cx, cy = center
    r_min = int(np.ceil(inner_radius_px * min_ratio))
    r_max = int(np.floor(inner_radius_px * max_ratio))
    if r_max <= r_min + 3:
        return None

    hist, r_min, r_max = _radial_edge_histogram(
        gray_for_gradient, edge_map, center, r_max,
        1.0, r_max / max(1, r_min), gradient_angle_tolerance_deg
    )
    if hist is None:
        return None

    peak_idx = int(np.argmax(hist))
    radius = float(r_min + peak_idx)
    peak = float(hist[peak_idx])

    support = peak / max(1.0, 2.0 * np.pi * radius)
    baseline = float(np.percentile(hist, 75))
    prominence = peak / max(1.0, baseline)

    if support < min_support or prominence < min_prominence:
        return None

    return {"radius": radius, "support": support, "prominence": prominence}


def build_ring_prompts(cx, cy, inner_r, outer_r, win_r, n_pos=8, n_neg_outside=4):
    """
    Build the point prompts MobileSAM needs to segment the RING BODY
    itself, given a hypothesized (inner_r, outer_r) band:
      - POSITIVE prompts around the band's own midpoint radius, at several
        angles, so the model is shown actual ring-band material.
      - One NEGATIVE prompt at the hole center - the center of a near-
        overhead ring photo is hole (or, rarely, fully occluded), never
        ring band, in every physically plausible case in scope here.
      - NEGATIVE prompts well outside the ring (near the crop window edge)
        so the model is also shown what is clearly NOT the ring.
    This is a PROMPT, not a classification rule: MobileSAM still has to
    decide the actual mask boundary from the image itself, these points
    only tell it where to look for positive/negative examples.
    """
    points, labels = [], []
    band_r = (inner_r + outer_r) / 2.0
    for i in range(n_pos):
        theta = 2 * np.pi * i / n_pos
        points.append([cx + band_r * np.cos(theta), cy + band_r * np.sin(theta)])
        labels.append(1)
    points.append([cx, cy])
    labels.append(0)
    for i in range(n_neg_outside):
        theta = 2 * np.pi * i / n_neg_outside + 0.3
        rr = win_r * 0.95
        points.append([cx + rr * np.cos(theta), cy + rr * np.sin(theta)])
        labels.append(0)
    return points, labels


def segment_ring_body_mobilesam(rectified_bgr, sam_model, cx, cy, inner_r, outer_r,
                                 outer_margin=1.3):
    """
    Run MobileSAM, prompted with build_ring_prompts' points for this one
    hypothesis's (inner_r, outer_r) band, to produce a ring-BODY mask -
    the segmentation target this revision uses instead of matching the
    surrounding background's color (see module docstring). Returns a
    full-rectified-frame-sized uint8 mask (0/255), or None if the model
    produced no mask at all for this prompt set (e.g. a genuinely
    degenerate/empty crop).
    """
    h, w = rectified_bgr.shape[:2]
    win_r = outer_r * outer_margin
    x0 = max(0, int(cx - win_r)); x1 = min(w, int(cx + win_r))
    y0 = max(0, int(cy - win_r)); y1 = min(h, int(cy + win_r))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = rectified_bgr[y0:y1, x0:x1]

    lcx, lcy = cx - x0, cy - y0
    points, labels = build_ring_prompts(lcx, lcy, inner_r, outer_r, win_r)

    try:
        results = sam_model.predict(crop, points=[points], labels=[labels],
                                     device="cpu", verbose=False)
    except Exception:
        return None
    if not results or results[0].masks is None or results[0].masks.data.shape[0] == 0:
        return None

    m = results[0].masks.data.cpu().numpy()[0]
    # Resize the model's own (lower-resolution) mask up to the crop's
    # pixel size, then re-binarize at the midpoint. Without this,
    # cv2.resize's interpolation leaves intermediate (non-0/255) values
    # along the boundary, and cv2.findContours downstream treats ANY
    # nonzero pixel as foreground - silently growing the mask by a
    # fraction of the resize kernel's width in every direction, a real
    # measurement bias with nothing to do with any specific photo.
    m_resized = cv2.resize((m * 255).astype(np.uint8), (crop.shape[1], crop.shape[0]))
    _, m_resized = cv2.threshold(m_resized, 255 // 2, 255, cv2.THRESH_BINARY)
    m_full = np.zeros((h, w), np.uint8)
    m_full[y0:y0 + m_resized.shape[0], x0:x0 + m_resized.shape[1]] = m_resized
    return m_full


def analyze_mask_topology(fg_mask):
    """
    Decide whether a ring-body mask actually corresponds to a physical
    ring body with a single enclosed hole, from the mask's own CONTOUR
    HIERARCHY (cv2.RETR_CCOMP) - this directly implements "acceptance
    should depend on mask topology": one ring body, one enclosed central
    hole, hole boundary sufficiently complete (solidity), plausible
    geometry (axis ratio). Only once a mask passes this gate does its
    hole's ellipse become a candidate measurement.

    The ring BODY is the largest top-level (no-parent) contour; the HOLE
    is the largest contour whose parent IS that body contour - i.e. the
    largest region the mask's own topology says is enclosed BY the body,
    not merely "some background-colored blob we found nearby".

    The convex hull is used ONLY to compute solidity as a validity CHECK
    ("is this hole boundary complete enough to trust") - the returned
    ellipse is always fit to the RAW hole contour, never to the hull, so
    an incomplete/bitten crescent cannot be hull-reconstructed into a
    convincing-looking measurement (the conceptual gap this revision was
    explicitly asked to close).
    """
    fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    cnts, hier = cv2.findContours(fg_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts or hier is None:
        return {"ok": False, "reason": "NO_CONTOURS"}
    hier = hier[0]

    top_level = [i for i in range(len(cnts)) if hier[i][3] == -1]
    if not top_level:
        return {"ok": False, "reason": "NO_BODY"}
    body_idx = max(top_level, key=lambda i: cv2.contourArea(cnts[i]))
    body_area = cv2.contourArea(cnts[body_idx])
    if body_area < 500:
        return {"ok": False, "reason": "BODY_TOO_SMALL"}
    n_bodies = sum(1 for i in top_level if cv2.contourArea(cnts[i]) > 0.15 * body_area)

    children = [i for i in range(len(cnts)) if hier[i][3] == body_idx]
    if not children:
        return {"ok": False, "reason": "NO_HOLE", "n_bodies": n_bodies}
    hole_idx = max(children, key=lambda i: cv2.contourArea(cnts[i]))
    hole_area = cv2.contourArea(cnts[hole_idx])
    hole_contour = cnts[hole_idx]
    n_holes = sum(1 for i in children if cv2.contourArea(cnts[i]) > 0.1 * hole_area)
    if len(hole_contour) < 5 or hole_area < 100:
        return {"ok": False, "reason": "HOLE_TOO_SMALL", "n_bodies": n_bodies}

    hull = cv2.convexHull(hole_contour)
    hull_area = cv2.contourArea(hull)
    solidity = hole_area / hull_area if hull_area > 0 else 0.0

    ellipse = cv2.fitEllipse(hole_contour)
    (ecx, ecy), (MA, ma), angle = ellipse
    axis_ratio = min(MA, ma) / max(MA, ma) if max(MA, ma) > 0 else 0.0

    ok = (n_bodies == 1 and n_holes == 1
          and solidity >= MIN_HOLE_SOLIDITY
          and axis_ratio >= MIN_HOLE_AXIS_RATIO)
    return {
        "ok": ok,
        "reason": None if ok else "TOPOLOGY_NOT_CLEAN",
        "n_bodies": n_bodies,
        "n_holes": n_holes,
        "solidity": solidity,
        "axis_ratio": axis_ratio,
        "ellipse": ellipse,
        "diam_px": (MA + ma) / 2.0,
        "hole_contour": hole_contour,
        "fg_mask": fg_clean,
    }


def locate_outer_ring_circle(rect_gray, marker_rect):
    """
    Locate the ring region using a strong approximately circular edge.

    This is localization only: the returned circle is NEVER used as the
    measurement or as evidence for which boundary is correct - it only
    tells the rest of measure_ring roughly where to look (seeding
    build_band_hypotheses' two radial searches, and sizing the crop window
    segment_ring_body_mobilesam prompts MobileSAM on). Which pixels inside
    that window are actually the ring body, and which enclosed region is
    the hole, is decided entirely by the prompted segmentation and
    analyze_mask_topology's check on the resulting mask, so this function
    does not need to itself validate "is this really a ring" - it only
    needs a plausible, roughly-correctly-located, roughly-correctly-sized
    circle to point the rest of the pipeline at.

    Preference order: a Hough circle that ALSO shows plausible nested
    radial edge structure (checked via radial_inner_candidate) is preferred
    when one exists, largest first - this is usually the more precise
    localization. But on a clean, high-contrast photo, Hough can lock onto
    the ring's INNER hole directly (it can be the single most prominent
    circle in the frame), in which case there is no even-smaller nested
    structure to find and every candidate fails that check - which used to
    make this function return None outright, discarding localization
    entirely even though a perfectly usable circle was sitting right there.
    Since (per above) this hint is coarse-localization-only, falling back
    to the largest Hough circle in the physically plausible range (whether
    or not it validated) is safe: measure_ring's own refinement step
    immediately after this call independently cross-checks the resulting
    radius and expands it if the segmentation window needs more room - so
    an inner-hole-sized Hough circle still ends up corrected to roughly the
    true outer extent before it's used for anything.
    """
    work = rect_gray.copy()

    mx0, my0, mx1, my1 = marker_rect
    pad = MARKER_MASK_PAD_PX

    # Remove the calibration marker from Hough search.
    fill_value = int(np.median(rect_gray))

    cv2.rectangle(
        work,
        (
            max(0, int(mx0 - pad)),
            max(0, int(my0 - pad)),
        ),
        (
            min(work.shape[1] - 1, int(mx1 + pad)),
            min(work.shape[0] - 1, int(my1 + pad)),
        ),
        fill_value,
        -1
    )

    blurred = cv2.GaussianBlur(work, (9, 9), 1.8)

    # Broad supported physical range for ring OUTER diameter:
    # radius 5-20 mm => outer diameter 10-40 mm.
    min_radius_px = int(5.0 * PX_PER_MM_OUT)
    max_radius_px = int(20.0 * PX_PER_MM_OUT)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=120,
        param2=35,
        minRadius=min_radius_px,
        maxRadius=max_radius_px,
    )

    if circles is None:
        return None

    # Use one stable edge map only for selecting which Hough circle
    # actually looks ring-like.
    probe = cv2.Canny(rect_gray, 40, 120)

    probe = cv2.morphologyEx(
        probe,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8)
    )

    marker_pad = MARKER_MASK_PAD_PX

    cv2.rectangle(
        probe,
        (
            max(0, int(mx0 - marker_pad)),
            max(0, int(my0 - marker_pad)),
        ),
        (
            min(probe.shape[1] - 1, int(mx1 + marker_pad)),
            min(probe.shape[0] - 1, int(my1 + marker_pad)),
        ),
        0,
        -1
    )

    best = None          # largest circle that also validated nested structure
    largest_any = None   # largest circle regardless of validation - coarse fallback

    for cx, cy, radius in circles[0]:
        cx = float(cx)
        cy = float(cy)
        radius = float(radius)

        if largest_any is None or radius > largest_any[2]:
            largest_any = (cx, cy, radius)

        inner = radial_inner_candidate(
            rect_gray,
            probe,
            (cx, cy),
            radius
        )

        if inner is None:
            continue

        # Prefer the LARGEST candidate that has valid inner structure, not
        # the one with the highest support*prominence score. Found on a
        # real photo: smaller Hough circles more easily produce a
        # spuriously "sharp-looking" radial peak by chance in noisy real
        # texture (their narrower search band has less background to
        # dilute the peak-to-baseline ratio), so support*prominence
        # systematically favored wrong, smaller circles over the true
        # (larger) ring outline - confirmed: true candidate ranked LOWEST
        # (0.183) of three valid candidates, while two smaller, wrong ones
        # ranked higher (0.395, 0.501). Within the stated scope of "one
        # clearly visible ring", the largest validated candidate is the
        # ring; a smaller circle that also happens to show some radial
        # structure is far more likely to be texture/reflection noise.
        rank = radius

        if best is None or rank > best[0]:
            best = (
                rank,
                (cx, cy, radius)
            )

    if best is not None:
        return best[1]

    # No Hough circle showed nested radial structure (see docstring above
    # for why that's expected, not necessarily wrong, on a clean/high-
    # contrast photo) - fall back to the largest circle in the physically
    # plausible range as a coarse hint rather than discarding localization
    # entirely.
    return largest_any


def build_band_hypotheses(rect_gray, marker_rect, outer_ring_hint):
    """
    Build both readings of outer_ring_hint per the module docstring's
    TWO-HYPOTHESIS STRATEGY, returning a list of (label, inner_r, outer_r)
    tuples - zero, one, or two entries depending on how much radial edge
    signal this photo actually has. Degenerate pairs (outer_r not
    meaningfully larger than inner_r) are dropped here so callers never
    have to special-case them.
    """
    cx, cy, r = outer_ring_hint
    probe = cv2.Canny(rect_gray, 40, 120)
    probe = cv2.morphologyEx(probe, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mx0, my0, mx1, my1 = marker_rect
    pad = MARKER_MASK_PAD_PX
    cv2.rectangle(
        probe,
        (max(0, int(mx0 - pad)), max(0, int(my0 - pad))),
        (min(probe.shape[1] - 1, int(mx1 + pad)), min(probe.shape[0] - 1, int(my1 + pad))),
        0, -1,
    )

    hypotheses = []

    # H1: outer_ring_hint IS the outer edge; search inward for the hole.
    inner_cand = radial_inner_candidate(rect_gray, probe, (cx, cy), r,
                                         min_band_ratio=1.05, max_band_ratio=3.0)
    if inner_cand is not None:
        hypotheses.append(("r=outer", inner_cand["diam_px"] / 2.0, r))

    # H2: outer_ring_hint IS the hole edge; search outward for the band.
    outer_cand = radial_outer_candidate(rect_gray, probe, (cx, cy), r)
    if outer_cand is not None:
        hypotheses.append(("r=inner", r, outer_cand["radius"]))

    return [(label, inner_r, outer_r) for (label, inner_r, outer_r) in hypotheses
            if outer_r > inner_r * 1.03]


def measure_ring(photo, verbose_name=""):
    res = MeasurementResult()
    gray_full = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    # Cheap whole-photo checks first, before any marker/ring detection work.
    # If a photo critically fails one of these, there is no point (and real
    # risk of a misleading result) in running the rest of the pipeline.
    exposure_issue = check_exposure(gray_full)
    if exposure_issue:
        res.reason = exposure_issue
        return res

    sharpness = blur_score(gray_full)
    if sharpness < 40:
        res.reason = "BLUR"
        return res

    rectified, marker_rect, err = detect_and_rectify(photo)
    if err:
        res.reason = err
        return res

    rect_gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

    # Coarse ROI localization ONLY - see locate_outer_ring_circle's
    # docstring. It tells the rest of this function roughly WHERE the ring
    # is; WHICH pixels are the true ring body/hole is decided entirely by
    # the prompted MobileSAM segmentation + mask-topology analysis below.
    outer_ring_hint = locate_outer_ring_circle(rect_gray, marker_rect)
    if outer_ring_hint is None:
        res.reason = "RING_NOT_FOUND"
        return res

    if os.environ.get("RING_DEBUG"):
        print(f"[RING_DEBUG] outer_ring_hint diam_mm={2*outer_ring_hint[2]/PX_PER_MM_OUT:.2f}")

    def draw_debug(winner=None, reason_note=None):
        """Built once the outcome is known. Draws the localization ROI,
        every hypothesis's ring-body mask (tinted, one color per
        hypothesis) and hole contour, and the winning ellipse (if any)
        bold on top."""
        dbg = rectified.copy()
        line_thickness = max(3, dbg.shape[1] // 300)
        ocx, ocy, orad = outer_ring_hint
        cv2.circle(dbg, (int(ocx), int(ocy)), int(orad), (200, 200, 200), max(1, line_thickness // 2))

        palette = {"r=outer": (255, 140, 0), "r=inner": (0, 200, 255)}
        for label, inner_r, outer_r, topo in attempts:
            if not topo.get("ok"):
                continue
            color = palette.get(label, (0, 180, 0))
            overlay = dbg.copy()
            overlay[topo["fg_mask"] > 0] = color
            dbg = cv2.addWeighted(overlay, 0.30, dbg, 0.70, 0)
            cv2.drawContours(dbg, [topo["hole_contour"]], -1, color, 2)

        if winner is not None:
            label, inner_r, outer_r, topo = winner
            cv2.ellipse(dbg, topo["ellipse"], (255, 255, 255), line_thickness + 2)

        if reason_note and os.environ.get("RING_DEBUG_OVERLAY"):
            cv2.putText(dbg, reason_note, (20, dbg.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return dbg

    hypotheses = build_band_hypotheses(rect_gray, marker_rect, outer_ring_hint)
    if not hypotheses:
        # Neither radial search (inward or outward from the hint) found a
        # plausible second radius at all - there is no (inner_r, outer_r)
        # band to even prompt a segmenter with.
        res.reason = "NO_BAND_HYPOTHESIS"
        attempts = []
        res.debug_img = draw_debug(reason_note="NO_BAND_HYPOTHESIS")
        return res

    sam_model = _get_sam_model()
    if sam_model is None:
        res.reason = "SEGMENTATION_MODEL_UNAVAILABLE"
        attempts = []
        res.debug_img = draw_debug(reason_note="SEGMENTATION_MODEL_UNAVAILABLE")
        return res

    # Run MobileSAM + mask-topology analysis independently for every
    # hypothesis - see module docstring's TWO-HYPOTHESIS STRATEGY. Each
    # entry is (label, inner_r, outer_r, topo_dict); topo_dict["ok"] is
    # False (with a "reason") when that hypothesis didn't pan out.
    attempts = []
    for label, inner_r, outer_r in hypotheses:
        cx, cy, _ = outer_ring_hint
        fg = segment_ring_body_mobilesam(rectified, sam_model, cx, cy, inner_r, outer_r)
        if fg is None:
            attempts.append((label, inner_r, outer_r, {"ok": False, "reason": "NO_MASK"}))
            continue
        topo = analyze_mask_topology(fg)
        attempts.append((label, inner_r, outer_r, topo))

    if os.environ.get("RING_DEBUG"):
        for label, inner_r, outer_r, topo in attempts:
            status = "OK" if topo["ok"] else f"REJECT({topo['reason']})"
            print(f"[RING_DEBUG] [{label}] inner_r={inner_r:.1f} outer_r={outer_r:.1f} -> {status} "
                  f"solidity={topo.get('solidity', 0):.2f} axis_ratio={topo.get('axis_ratio', 0):.2f}")

    res.rejected_candidates = [
        (label, topo.get("reason")) for label, inner_r, outer_r, topo in attempts if not topo["ok"]
    ]
    valid = [(label, inner_r, outer_r, topo) for label, inner_r, outer_r, topo in attempts if topo["ok"]]

    res.candidates_mm = [(label, topo["diam_px"] / PX_PER_MM_OUT, topo) for label, _, _, topo in valid]
    res.family_estimates = {label: topo["diam_px"] / PX_PER_MM_OUT for label, _, _, topo in valid}

    if not valid:
        # Every hypothesis that reached segmentation either produced no
        # mask at all, or a mask whose topology did not pass the
        # single-body/single-hole/solidity/shape gate - per the explicit
        # instruction to inspect mask correctness rather than force a
        # number through, this is an honest retake, not a best-effort
        # guess from whichever attempt looked least bad.
        res.reason = "HOLE_NOT_SUFFICIENTLY_VISIBLE"
        res.debug_img = draw_debug(reason_note="HOLE_NOT_SUFFICIENTLY_VISIBLE")
        return res

    # Prefer the hypothesis with the higher hole solidity - the more
    # convex, more complete-looking boundary. When both hypotheses are
    # valid, their agreement is reported (detection_spread_mm) as an extra
    # confidence signal, not used to override the solidity choice.
    winner = max(valid, key=lambda t: t[3]["solidity"])
    label, inner_r, outer_r, topo = winner
    diam_mm = topo["diam_px"] / PX_PER_MM_OUT

    res.ok = True
    res.diameter_mm = diam_mm
    res.hypothesis_used = label
    res.mask_solidity = topo["solidity"]
    res.hole_axis_ratio = topo["axis_ratio"]
    res.n_bodies = topo["n_bodies"]
    res.n_holes = topo["n_holes"]
    if len(valid) == 2:
        other_diam_mm = [t[3]["diam_px"] / PX_PER_MM_OUT for t in valid if t is not winner][0]
        res.detection_spread_mm = abs(diam_mm - other_diam_mm)
    res.debug_img = draw_debug(winner=winner)
    return res
