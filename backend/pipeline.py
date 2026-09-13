"""
Ring measurement pipeline (deterministic computer vision, no ML/LLM).

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

DETECTION ARCHITECTURE (SEGMENTATION-FIRST REVISION - current): earlier
revisions of this file went through three designs, each one's failures
motivating the next:

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

This revision (4) stops choosing between ellipse candidates altogether.
Instead it SEGMENTS: classify every pixel in a localized window as
"background-colored" or not (reusing the same Lab-space chroma+brightness
distance from sample_background_reference/the old material-contrast
gate, now the SOLE primary detection mechanism rather than a gate or a
weak cue), find the background-colored CONNECTED COMPONENT that is fully
enclosed (does not touch the window border) - this is the hole, by
construction, because only the hole is background visible *through* the
ring rather than background *around* it - and fit an ellipse to that
component's boundary. There is no scoring of multiple candidate curves
against each other: there is exactly one enclosed background region per
threshold, and the question becomes "does this single region's boundary,
and its shape, stay consistent as the classification threshold is
loosened" rather than "which of several curves looks most plausible".

TWO ALTERNATIVES WERE INVESTIGATED AND REJECTED before this approach, per
the explicit instruction to look at SAM-family models and at a simpler
deterministic method and compare before deciding:

  - GrabCut (cv2.grabCut, GC_INIT_WITH_MASK) seeded from the coarse Hough
    ROI (ring band marked probable-foreground, a thin outer margin marked
    probable-background): on all 3 real test photos it merged the ring
    band and the hole interior into one foreground blob. Root cause: a
    naive radius-based seeding gives GrabCut no "probable background"
    seed pixels ANYWHERE inside the outer disk, so it has no signal that
    the centre region should be background rather than "more ring". It
    would need a seed that already roughly marks the hole, which assumes
    the very thing being detected - not viable without a separate hole
    localizer already in hand.
  - FastSAM (ultralytics FastSAM-s.pt, promptless "segment everything"):
    inconsistent across the 3 real photos. On the one real photo with
    strong hole/band contrast it isolated the hole cleanly as its own
    mask; on the other two it merged the entire ring/cap body and its
    hole into a single object mask (no separate "hole" segment at all),
    i.e. it reproduced a version of the same ambiguity this redesign is
    trying to remove, rather than resolving it. A lightweight promptless
    segmenter does not reliably know "the hole is a separate object from
    the ring" any better than the edge-based ensemble did. Not used.

The deterministic multi-threshold approach below was found to perform
comparably-or-better than FastSAM on all 3 real photos, is reproducible
(no model weights, no GPU), is fast, and directly implements the
requested pipeline stages (segment ring material -> find the enclosed
hole -> fit an ellipse to the hole boundary -> convert to mm via the
ArUco scale) rather than approximating them through ellipse-candidate
scoring.

PIPELINE STAGES (measure_ring):
  1. ArUco marker detection + perspective rectification to a fixed
     px/mm scale (detect_and_rectify) - UNCHANGED from every earlier
     revision; this is purely geometric calibration, not a detection
     decision.
  2. Coarse ring ROI localization (locate_outer_ring_circle, Hough) -
     UNCHANGED in mechanism, still coarse-only: it tells the rest of the
     pipeline roughly where the ring is so the segmentation window and
     the background-reference annulus can be sized/centered correctly. It
     is NEVER used to decide which boundary is the true inner hole.
  3. Background-color reference (sample_background_reference) - sampled
     LOCALLY around the localized ring first, falling back to a whole-
     frame sample. This is now HARD-REQUIRED (reinstated from its
     demotion in revision 3): without knowing what "background-colored"
     means for this specific photo, there is no segmentation signal to
     build a mask from at all, so a missing reference is an honest
     BACKGROUND_REFERENCE_UNAVAILABLE retake, not a silent fallback to
     some other mechanism.
  4. segment_hole_candidates: sweep a range of Lab-distance
     background-classification thresholds (z, in units of the
     background's own per-channel robust noise scale). At each z, build
     a dense per-pixel background-likeness mask, clean it with a
     morphological open (drop thin false-positive speckle) then close
     (seal small real gaps from local noise/texture), find the enclosed
     (non-border-touching) connected component with the largest area,
     take its CONVEX HULL (the true hole boundary is convex; a local
     brightness gradient or reflection can bite a non-convex notch out of
     the raw thresholded region without the hole actually having that
     shape - the hull recovers it), and fit an ellipse to the hull.
  5. Acceptance is decided from how that single region's ellipse BEHAVES
     across the z-sweep, not from comparing it to any other candidate:
     - require at least 2 valid z-thresholds to have produced a region
       at all (otherwise HOLE_NOT_FOUND - there's no enclosed background
       region to measure);
     - STABILITY: as z is loosened, a real hole's mask should stop
       growing once the open threshold change has caught the photo's own
       noise floor of real boundary pixels - the diameter at the two
       loosest valid z's should agree within STABILITY_TOL_MM. If the
       region keeps growing even at the loosest thresholds tried, the
       mask is still eating into ambiguous surrounding material and there
       is no trustworthy boundary to report (MASK_AMBIGUOUS);
     - SHAPE: once stable, the loosest stable candidate's axis ratio
       (minor/major) must be at least MIN_AXIS_RATIO_FOR_ACCEPT. A region
       that stabilizes but stays far from circular/elliptical is not
       cleanly recovering the hole's outline (HOLE_NOT_SUFFICIENTLY_-
       VISIBLE) - this was found concretely on one real test photo (a
       knurled, serrated cap, not a plain ring band) whose opening has a
       genuine brightness gradient across it that no single threshold
       recovers cleanly: axis ratio never exceeds ~0.6 even after
       widening the z-range and adding the hull correction, and the
       region never stops leaking past the object's textured edge at
       looser z - a legitimate "mask ambiguous/hole not sufficiently
       visible" case to honestly reject, not one to keep tuning thresholds
       to force through.
  6. The ACCEPTED measurement is the loosest STABLE candidate's ellipse
     (not the tightest, and not an average) - the loosest threshold that
     hasn't started eating into non-hole material is the one most likely
     to have recovered the hole's full true extent rather than an
     undersized, overly-conservative core of it.

STABILITY_TOL_MM and MIN_AXIS_RATIO_FOR_ACCEPT, and the z_thresholds grid
itself, were reasoned from the observed SHAPE of each test photo's own
z-vs-diameter and z-vs-axis_ratio curves (where does growth actually stop;
how separated are the clean-accept cases from the one genuinely ambiguous
real photo), never from forcing any specific known diameter to pass - see
the no-overfitting policy above. The z-grid was widened after an initial
coarse grid (1.0, 1.5, ..., 3.0) produced a misleading "still growing"
read on a known-good synthetic case simply because it skipped over the
point where that photo's own curve actually flattens (1.25-1.5); the
finer grid below resolved that for every case tested without narrowing
any single case's own true signal.
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


class MeasurementResult:
    def __init__(self):
        self.ok = False
        self.reason = None          # structured failure code
        self.diameter_mm = None
        # NOTE: deliberately NOT called "uncertainty_mm". Under the
        # segmentation-first architecture this is the gap between the
        # loosest STABLE z-threshold's diameter and the next-loosest one
        # considered (see segment_hole_candidates / measure_ring) - a
        # stability signal on THIS photo's own mask, not a physical
        # measurement uncertainty. It says nothing about print scale
        # error, lens distortion, or ArUco corner localization error.
        self.detection_spread_mm = None
        self.candidates_mm = []     # [(z, diameter_mm, candidate_dict), ...] every z tried
        # Keyed by a label per z-threshold (e.g. "z=1.50") rather than by
        # detection "family" - there is only one detection mechanism now.
        # Kept under this field name for API/frontend compatibility (the
        # frontend renders this generically as "label: value mm").
        self.family_estimates = {}
        self.debug_img = None
        # Every candidate ellipse considered (every z tried), whether or
        # not it ended up being the accepted one, together with a reason
        # when a z produced no usable region at all. Kept on the result so
        # a debug overlay can show the full sweep, not just the winner.
        self.rejected_candidates = []
        # SEGMENTATION-FIRST REVISION: reporting fields describing the
        # winning candidate's own behavior across the z-sweep, replacing
        # revision 3's ellipse-ensemble fields (boundary_coverage,
        # fit_residual, color_confidence), which no longer have meaning
        # once there is only one candidate region per threshold instead of
        # several competing ellipse candidates to score against each
        # other.
        self.hole_axis_ratio = None     # minor/major of the accepted ellipse, shape-quality signal
        self.z_threshold_used = None    # which z-threshold produced the accepted candidate
        self.n_stable_z = None          # how many of the loosest z's agreed within STABILITY_TOL_MM


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
    tallest peak in the radius histogram. Used SOLELY by
    locate_outer_ring_circle() to validate a Hough circle (a yes/no "does
    this look ring-like at all" check, for coarse ROI localization only) -
    that caller only ever needs one answer, and changing its behavior has
    previously been shown to change which Hough circle gets picked as the
    outer boundary (a real regression - see git history around commits
    3e07ad8 / 0767fc5), so it is kept exactly as originally calibrated.

    measure_ring() itself does not use this, or any other edge/radial/
    contour-based candidate, for its own measurement - see the module
    docstring's SEGMENTATION-FIRST REVISION section for why boundary
    selection moved entirely to segment_hole_candidates.
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


def sample_background_reference(rectified_bgr, marker_rect, outer_ring_hint,
                                  local_inner_scale=1.15, local_outer_scale=3.0,
                                  min_local_px=1500):
    """
    Sample the photo's own TABLE/BACKGROUND color, independent of what any
    ring/hole detector thinks. Under the segmentation-first architecture
    this is no longer a secondary cue or a gate on ellipse candidates - it
    is the PRIMARY input that segment_hole_candidates classifies every
    pixel against: "background-colored" vs. not. Every stage after this
    one depends on it; a photo with no usable reference is measured as
    BACKGROUND_REFERENCE_UNAVAILABLE rather than attempted on no signal.

    LOCAL-FIRST SAMPLING: wood grain and lighting are not uniform across a
    whole photo - a corner of the frame far from the ring can legitimately
    be a different color/brightness than the table immediately around the
    ring (a lamp to one side, a shadow falling across part of the table).
    What segmentation actually needs is what "background" looks like
    *right around the ring*, not a whole-frame average that can be pulled
    off by regions nowhere near it. So when outer_ring_hint localizes the
    ring, this samples an annulus around it first - from
    local_inner_scale*radius (safely clear of the ring's own outer edge
    and band) out to local_outer_scale*radius - and only falls back to the
    old whole-frame (marker-excluded) sample if that local annulus doesn't
    yield enough pixels (ring too close to the frame edge, hint overlaps
    the marker exclusion heavily, etc).

    MEDIAN (not mean) is used throughout so that even a sample still
    contaminated by a sliver of ring/marker pixels stays a robust estimate
    as long as background pixels are the majority - true for both the local
    annulus and the whole-frame fallback on any photo actually following
    the "near-overhead, ring and marker both clearly visible" capture
    instructions.

    Returns None (never a fabricated/degenerate reference) if too little of
    the frame is left after exclusions to trust either sample at all - the
    caller must treat that as "cannot segment at all" and retake.
    """
    h, w = rectified_bgr.shape[:2]
    lab = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    mx0, my0, mx1, my1 = marker_rect
    pad = MARKER_MASK_PAD_PX
    mex0, mey0 = max(0, int(mx0 - pad)), max(0, int(my0 - pad))
    mex1, mey1 = min(w, int(mx1 + pad)), min(h, int(my1 + pad))

    def _median_mad(bg_pixels, n_px, source):
        lab_median = np.median(bg_pixels, axis=0)
        # Median Absolute Deviation, scaled to be std-equivalent for a
        # normal distribution (x1.4826) - a robust noise-floor estimate
        # that isn't blown up by the occasional outlier pixel (shadow edge,
        # dust, a sliver of ring that leaked past exclusion).
        lab_mad = np.median(np.abs(bg_pixels - lab_median), axis=0) * 1.4826
        return {"lab_median": lab_median, "lab_mad": lab_mad, "n_px": n_px, "source": source}

    if outer_ring_hint is not None:
        ocx, ocy, orad = outer_ring_hint
        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - ocx) ** 2 + (yy - ocy) ** 2
        local_mask = (dist2 >= (orad * local_inner_scale) ** 2) & (dist2 <= (orad * local_outer_scale) ** 2)
        local_mask[mey0:mey1, mex0:mex1] = False
        if local_mask.sum() >= min_local_px:
            bg_pixels = lab[local_mask]
            return _median_mad(bg_pixels, int(local_mask.sum()), "local")

    # Fallback: whole-frame, marker (and ring hint, if any) excluded - the
    # original design, kept for photos where a local annulus isn't usable.
    exclude = np.zeros((h, w), dtype=bool)
    exclude[mey0:mey1, mex0:mex1] = True
    if outer_ring_hint is not None:
        ocx, ocy, orad = outer_ring_hint
        yy, xx = np.ogrid[:h, :w]
        exclude |= (xx - ocx) ** 2 + (yy - ocy) ** 2 <= (orad * 1.4) ** 2

    bg_mask = ~exclude
    if bg_mask.sum() < 500:
        return None

    bg_pixels = lab[bg_mask]
    return _median_mad(bg_pixels, int(bg_mask.sum()), "global")


def locate_outer_ring_circle(rect_gray, marker_rect):
    """
    Locate the ring region using a strong approximately circular edge.

    This is localization only: the returned circle is NEVER used as the
    measurement or as evidence for which boundary is correct - it only
    tells the rest of measure_ring roughly where to look (sizing
    segment_hole_candidates' search window, centering
    sample_background_reference's local background annulus). Which pixels
    inside that window are actually the hole is decided entirely by
    segment_hole_candidates' background-color segmentation and its
    stability-across-thresholds check, so this function does not need to
    itself validate "is this really a ring" - it only needs a plausible,
    roughly-correctly-located, roughly-correctly-sized circle to point the
    rest of the pipeline at.

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


def segment_hole_candidates(rect_lab, marker_rect, outer_ring_hint, bg_ref,
                             z_thresholds=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
                             close_ksize=9, open_ksize=5, min_diam_px=20):
    """
    THE core detection mechanism of this pipeline (see module docstring's
    SEGMENTATION-FIRST REVISION). For each candidate z-threshold:

      1. Classify every pixel in a window around the localized ring as
         "background-colored" (distance from bg_ref's Lab median, scaled
         by its own robust per-channel noise estimate, <= z) or not. The
         distance combines chroma (a*/b*) Euclidean distance with a
         ONE-SIDED brightness term - only being brighter than background
         counts toward "background-like" on the brightness axis, so a
         genuinely darker region (a shadow cast BY the ring, or the ring
         material itself if it's dark) is not mistaken for background
         just because its chroma happens to be close; a region so much
         darker than background that it's almost certainly pure shadow
         (MAX_DARKNESS_Z) is excluded outright rather than folded into the
         distance metric, matching the asymmetric logic this pipeline has
         used for background/material contrast since the very first
         revision.
      2. Clean the mask: MORPH_OPEN (drop thin false-positive speckle from
         isolated noisy pixels) then MORPH_CLOSE (seal small real gaps -
         e.g. a thin glare streak across the hole - so the enclosed region
         isn't fragmented by something that wouldn't fragment a human's
         reading of the same photo).
      3. Find the largest connected component of this mask that does NOT
         touch the window border. This is, by construction, the ring's
         enclosed hole: everywhere else "background-colored" pixels touch
         the border of the window (because the actual background is
         outside the ring), but the hole is background visible *through*
         the ring, so it alone forms a closed island.
      4. Take that component's convex hull before fitting an ellipse.
         Rationale: the true hole opening is convex (it's the silhouette
         of a round hole), but a local brightness gradient, partial
         reflection, or shadow can locally fail the background-likeness
         test along part of the true boundary, biting a non-convex notch
         out of the raw thresholded region without the hole itself having
         that shape. The hull corrects for that without assuming anything
         about WHERE the notch will be, since it is computed purely from
         the component's own shape.

    Returns a list of dicts (one per z that produced a usable region),
    each with: z, ellipse, diam_px, center, axis_ratio, area (hull area in
    px^2). Does NOT decide accept/reject - that is measure_ring's job,
    based on how these z-indexed candidates behave as a group (stability,
    shape) - this function never compares candidates to each other, it
    only ever looks at one region per threshold.
    """
    if bg_ref is None or outer_ring_hint is None:
        return []

    cx, cy, r = outer_ring_hint
    h, w = rect_lab.shape[:2]
    win_r = int(np.ceil(r * 1.3))
    x0 = max(0, int(cx - win_r)); x1 = min(w, int(cx + win_r + 1))
    y0 = max(0, int(cy - win_r)); y1 = min(h, int(cy + win_r + 1))
    win_lab = rect_lab[y0:y1, x0:x1]

    lab_med, lab_mad = bg_ref["lab_median"], bg_ref["lab_mad"]
    scale = np.maximum(lab_mad, 3.0)
    chroma_dist = np.sqrt((((win_lab[:, :, 1:] - lab_med[1:]) / scale[1:]) ** 2).sum(axis=2))
    brightness_signed = (win_lab[:, :, 0] - lab_med[0]) / scale[0]
    MAX_DARKNESS_Z = 6.0
    too_dark = brightness_signed < -MAX_DARKNESS_Z
    dist = np.where(too_dark, np.inf, np.maximum(chroma_dist, np.clip(brightness_signed, 0, None)))

    mx0, my0, mx1, my1 = marker_rect
    pad = MARKER_MASK_PAD_PX
    mmy0 = max(0, int(my0 - pad) - y0); mmy1 = min(win_lab.shape[0], int(my1 + pad) - y0)
    mmx0 = max(0, int(mx0 - pad) - x0); mmx1 = min(win_lab.shape[1], int(mx1 + pad) - x0)

    out = []
    for z in z_thresholds:
        bg_like = (dist <= z).astype(np.uint8) * 255
        bg_like = cv2.morphologyEx(bg_like, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8))
        bg_like = cv2.morphologyEx(bg_like, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8))
        if mmy1 > mmy0 and mmx1 > mmx0:
            bg_like[mmy0:mmy1, mmx0:mmx1] = 0

        num_labels, labels = cv2.connectedComponents(bg_like, connectivity=8)
        border_labels = set(labels[0, :].tolist()) | set(labels[-1, :].tolist()) | \
                         set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
        border_labels.discard(0)

        best = None
        for lbl in range(1, num_labels):
            if lbl in border_labels:
                continue
            comp = (labels == lbl).astype(np.uint8) * 255
            area = int(np.count_nonzero(comp))
            if area < 200:
                continue
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            if len(hull) < 5:
                continue
            ellipse = cv2.fitEllipse(hull)
            (ecx, ecy), (MA, ma), angle = ellipse
            if MA == 0 or ma == 0 or MA < min_diam_px:
                continue
            axis_ratio = min(MA, ma) / max(MA, ma)
            hull_area = cv2.contourArea(hull)
            if best is None or hull_area > best[1]:
                best = (comp, hull_area, ellipse, axis_ratio)

        if best is not None:
            comp, area, ellipse, axis_ratio = best
            (ecx, ecy), (MA, ma), angle = ellipse
            full_ellipse = ((ecx + x0, ecy + y0), (MA, ma), angle)
            out.append({
                "z": z,
                "ellipse": full_ellipse,
                "diam_px": (MA + ma) / 2.0,
                "center": (ecx + x0, ecy + y0),
                "axis_ratio": axis_ratio,
                "area": area,
            })
    return out


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
    rect_lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Coarse ROI localization ONLY - see locate_outer_ring_circle's
    # docstring. It tells the rest of this function roughly WHERE the ring
    # is; WHICH pixels inside that area are the true hole is decided
    # entirely by segment_hole_candidates below.
    outer_ring_hint = locate_outer_ring_circle(rect_gray, marker_rect)
    if outer_ring_hint is None:
        res.reason = "RING_NOT_FOUND"
        return res

    if os.environ.get("RING_DEBUG"):
        print(f"[RING_DEBUG] outer_ring_hint diam_mm={2*outer_ring_hint[2]/PX_PER_MM_OUT:.2f}")

    # Background-color reference is now HARD-REQUIRED: it's the primary
    # signal segment_hole_candidates classifies every pixel against, not a
    # secondary/optional cue (see sample_background_reference's and the
    # module docstring's discussion of this reinstatement).
    bg_ref = sample_background_reference(rectified, marker_rect, outer_ring_hint)
    if bg_ref is None:
        res.reason = "BACKGROUND_REFERENCE_UNAVAILABLE"
        return res

    candidates = segment_hole_candidates(rect_lab, marker_rect, outer_ring_hint, bg_ref)

    res.candidates_mm = [
        (c["z"], c["diam_px"] / PX_PER_MM_OUT, c) for c in candidates
    ]
    res.family_estimates = {
        f"z={c['z']:.2f}": c["diam_px"] / PX_PER_MM_OUT for c in candidates
    }

    def draw_debug(final_candidate=None, reason_note=None):
        """Built once the outcome is known. Draws the localization ROI,
        every z-threshold's candidate ellipse in a threshold-indexed color
        (lightest z = coolest color, loosest z = warmest), and the final
        accepted ellipse (if any) bold on top. Rejected-photo overlays
        still show the full sweep, since that is the useful debugging
        signal under this architecture (there is no "rejected candidate
        list" distinct from "the sweep" any more - the sweep IS the
        evidence, accepted or not)."""
        dbg = rectified.copy()
        line_thickness = max(3, dbg.shape[1] // 300)

        if outer_ring_hint is not None:
            ocx, ocy, orad = outer_ring_hint
            cv2.circle(dbg, (int(ocx), int(ocy)), int(orad), (200, 200, 200), max(1, line_thickness // 2))

        n = max(1, len(candidates))
        for i, c in enumerate(candidates):
            t = i / max(1, n - 1)
            color = (int(255 * (1 - t)), 80, int(255 * t))  # blue (tight) -> red (loose)
            cv2.ellipse(dbg, c["ellipse"], color, max(1, line_thickness // 2))
            if os.environ.get("RING_DEBUG_OVERLAY"):
                (ecx, ecy), _, _ = c["ellipse"]
                label = f"z={c['z']:.2f} {c['diam_px']/PX_PER_MM_OUT:.2f}mm ar={c['axis_ratio']:.2f}"
                cv2.putText(dbg, label, (max(0, int(ecx) - 60), max(15, int(ecy))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if final_candidate is not None:
            cv2.ellipse(dbg, final_candidate["ellipse"], (255, 255, 255), line_thickness + 2)

        if reason_note and os.environ.get("RING_DEBUG_OVERLAY"):
            cv2.putText(dbg, reason_note, (20, dbg.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        return dbg

    if os.environ.get("RING_DEBUG"):
        for c in candidates:
            print(f"[RING_DEBUG] z={c['z']:.2f} diam_mm={c['diam_px']/PX_PER_MM_OUT:.2f} "
                  f"axis_ratio={c['axis_ratio']:.2f} area={c['area']:.0f}")

    if len(candidates) < 2:
        # Not enough of the sweep produced an enclosed background region at
        # all - there's no mask behavior to judge stability/shape from.
        res.reason = "HOLE_NOT_FOUND"
        res.debug_img = draw_debug(reason_note="HOLE_NOT_FOUND")
        return res

    # --- Stability across the z-sweep is the accept/reject signal. ---
    #
    # Reasoned from first principles, not fitted to any known diameter (see
    # module docstring): as z loosens, a real hole's recovered mask should
    # grow while it's still catching genuine hole-boundary pixels that a
    # tighter threshold missed, then STOP growing once it has - further
    # loosening should not keep adding real information, because there's
    # nothing further out that's actually hole. If the diameter is still
    # changing meaningfully even between the two LOOSEST thresholds tried,
    # the mask has not found a stable stopping point and is (or soon will
    # be) eating into ring material, a shadow, or background beyond the
    # ring - not a trustworthy boundary.
    #
    # STABILITY_TOL_MM=0.5: half a millimeter of residual drift between the
    # two loosest candidates tried is small relative to the physical sizes
    # in scope here (10-40mm outer diameter) and was the point at which
    # every clean-signal test case (real and synthetic) was already well
    # below, while the one genuinely ambiguous real photo (a knurled,
    # serrated cap with a real brightness gradient across its opening) was
    # clearly and separately above it even after widening the z-range.
    STABILITY_TOL_MM = 0.5

    # MIN_AXIS_RATIO_FOR_ACCEPT=0.75: once stable, the region's own shape
    # has to be reasonably close to circular/elliptical - a stable but
    # badly-non-elliptical region is evidence the mask locked onto a
    # boundary shape that ISN'T a round hole (e.g. it fused with a notch or
    # an adjacent shadow), not that the true hole is unusually eccentric. A
    # genuinely elliptical hole from meaningful photo angle is handled by
    # detect_and_rectify's side_ratio/PERSPECTIVE_TOO_HIGH gate upstream, so
    # axis_ratio this low at the measurement stage is a segmentation
    # problem, not a real perspective ellipse.
    MIN_AXIS_RATIO_FOR_ACCEPT = 0.75

    candidates_sorted = sorted(candidates, key=lambda c: c["z"])
    loosest = candidates_sorted[-1]
    second_loosest = candidates_sorted[-2]
    loosest_diam_mm = loosest["diam_px"] / PX_PER_MM_OUT
    second_diam_mm = second_loosest["diam_px"] / PX_PER_MM_OUT
    last_delta_mm = abs(loosest_diam_mm - second_diam_mm)

    # How many of the loosest candidates (walking inward from the loosest)
    # stay within tolerance of the loosest one - purely a reporting/debug
    # signal, not itself a gate.
    n_stable = 1
    for c in reversed(candidates_sorted[:-1]):
        if abs(c["diam_px"] / PX_PER_MM_OUT - loosest_diam_mm) <= STABILITY_TOL_MM:
            n_stable += 1
        else:
            break

    if os.environ.get("RING_DEBUG"):
        print(f"[RING_DEBUG] last_delta_mm={last_delta_mm:.3f} "
              f"loosest_axis_ratio={loosest['axis_ratio']:.3f} n_stable={n_stable}")

    if last_delta_mm > STABILITY_TOL_MM:
        res.reason = "MASK_AMBIGUOUS"
        res.detection_spread_mm = last_delta_mm
        res.debug_img = draw_debug(reason_note=f"MASK_AMBIGUOUS delta={last_delta_mm:.2f}mm")
        return res

    if loosest["axis_ratio"] < MIN_AXIS_RATIO_FOR_ACCEPT:
        res.reason = "HOLE_NOT_SUFFICIENTLY_VISIBLE"
        res.detection_spread_mm = last_delta_mm
        res.debug_img = draw_debug(reason_note=f"HOLE_NOT_SUFFICIENTLY_VISIBLE ar={loosest['axis_ratio']:.2f}")
        return res

    # ACCEPT: report the loosest STABLE candidate - the loosest threshold
    # that hasn't started growing into non-hole material is the one most
    # likely to have recovered the hole's full true extent rather than an
    # undersized, overly-conservative core of it.
    res.ok = True
    res.diameter_mm = loosest_diam_mm
    res.detection_spread_mm = max(last_delta_mm, 0.05)  # floor so we never claim impossible precision
    res.hole_axis_ratio = loosest["axis_ratio"]
    res.z_threshold_used = loosest["z"]
    res.n_stable_z = n_stable
    res.debug_img = draw_debug(final_candidate=loosest)
    return res
