"""
Ring measurement pipeline (deterministic computer vision, no ML/LLM).

GENERALIZATION / NO-OVERFITTING POLICY (read before touching this file):
Every decision in this file - which segmentation variant to trust, which
Hough circle is the outer boundary, whether cross-method agreement is
tight enough to accept - must be made from GEOMETRY and CROSS-METHOD
AGREEMENT on the photo actually being measured, never from which specific
test photo it is or what its known diameter is. Concretely:

- This file must never import, read, or branch on test_set/ ground truth
  (regression_suite.py's CASES list, expected_vs_actual.csv, or any
  literal ground-truth mm value copy-pasted from them). Ground truth is
  read ONLY by regression_suite.py, and only AFTER measure_ring() has
  already returned a result - to score error, never to steer detection.
- Do not add logic that identifies a specific input photo (by filename,
  by an image hash, by a "if this looks like the keyring photo" special
  case) and special-cases it. A rule that only fires on one known photo
  is not a fix, it's memorization, and it will not generalize to a new
  ring nobody has seen yet.
- When calibrating a threshold, validate it against the whole labeled set
  as a fixed decision rule applied uniformly - not by trying several
  candidate rules and picking whichever one happens to land closest to a
  known answer on one specific photo. This project has done the latter
  by mistake before (see git history around commits 3e07ad8 / 0767fc5:
  a "fix" was reverted specifically because it had been tuned by checking
  its output against one photo's known ground truth, and turned out to
  give a confidently WRONG answer once inspected properly - a policy
  violation that produced a real bug, not just a style problem).
- A photo that satisfies the documented supported-capture conditions
  should return a diameter; a photo that doesn't, or whose signal is too
  ambiguous to trust, should return a specific retake reason - never an
  unreliable number dressed up as a confident one.

DETECTION ARCHITECTURE (GEOMETRY-PRIMARY REVISION - current): earlier
revisions of this file went through two designs worth understanding
because the failures of each motivated the next:

  1. "Strongest circle" scoring - picked whichever candidate looked
     cleanest/most circular and had the most cross-method agreement. Fails
     because a ring's OUTER edge, a polished bevel, a specular reflection,
     or a cast shadow can all look just as clean/circular as the true
     inner hole, and multiple edge-based methods CAN agree on the same
     wrong one (confirmed: the dilate-bias bug made all 4 canny variants
     wrong in the same direction at once).
  2. "Material-contrast as a hard gate" - required every candidate's
     interior to match this photo's own sampled background color
     (validate_material_contrast) before it could even compete, and
     required the independent color-topology detector (enclosed_hole_-
     candidates, family "holecolor") to confirm a cluster before ACCEPT
     (has_topological_confirmation). This closed the false-accept failures
     from (1), but turned out to overcorrect: normal consumer photos with
     small shadows, specular highlights, or local illumination changes on
     the hole interior or the background right around the ring routinely
     broke the background/hole color match even though the boundary
     itself was perfectly visible and correctly located by every
     edge-based method - a real, reported regression (a known-good real
     ring photo, and a previously-false-accepting photo that correctly
     started REJECTing instead of being fixed, both started being refused
     for a reason that had nothing to do with whether the boundary could
     actually be found).

This revision (3) makes GEOMETRY the primary acceptance signal and
demotes color/background similarity to a weak, non-blocking supporting
cue, per explicit instruction: a photo should be accepted when its inner
boundary can be geometrically recovered with sufficient stability, even
if a shadow or highlight locally discolors part of the hole or the
background next to the ring - and rejected only when the boundary itself
cannot be recovered that way, not because of a color mismatch alone.

ArUco calibration, perspective rectification, and Hough circle detection
(locate_outer_ring_circle) are kept exactly as before, strictly for
coarse ROI localization - "roughly where is the ring" - never for
deciding which boundary inside that ROI is correct.

The boundary decision now runs on GEOMETRIC EVIDENCE, computed per
candidate (see geometric_plausibility, edge_support_fraction):
  - containment inside the outer ring boundary, and concentricity with it
    (geometric_plausibility, checked against outer_ring_hint - no longer
    against another independently-detected "outer partner" candidate,
    which was a much stricter and more fragile requirement);
  - visible boundary coverage % (edge_support_fraction): the fraction of
    the candidate ellipse's own perimeter that has real edge support
    SOMEWHERE across the union of every segmentation variant, sampled in
    angular bins rather than requiring one single contour to be fully
    closed. This is the PARTIAL-BOUNDARY TOLERANCE the old closed-contour-
    only circularity >= 0.65 gate did not allow: a boundary that is, say,
    80% visible with a 20% arc obscured by a shadow or a specular
    highlight now scores 0.80 coverage and can still be accepted, rather
    than being thrown out because the one contour that happened to trace
    it was broken into two pieces by that arc;
  - ellipse/circle fit residual (how well the candidate's points actually
    lie on its fitted ellipse - unchanged metric, now reported alongside
    coverage rather than being the only closure signal);
  - cross-variant / cross-family stability: a candidate only drives an
    ACCEPT when candidates from at least two independent families
    (canny/adaptive/otsu/holecolor/radial-fallback) land within
    CLUSTER_TOL_MM of each other AND agree spatially (not just on
    diameter) - unchanged from earlier revisions, this is still the
    "more than one method found the same thing" check, it is just no
    longer required to specifically include the "holecolor" family.

Color/background similarity (validate_material_contrast,
enclosed_hole_candidates' background-color segmentation,
sample_background_reference) is still computed and still contributes
"holecolor" as one more independent family/vote in the geometry-driven
consensus above - it is simply no longer a pass/fail gate that can reject
a geometrically well-supported boundary on its own. It is used only: (a)
as one more candidate source among several, exactly like any edge-based
family, and (b) as a tie-breaker/confidence annotation when more than one
cluster would otherwise qualify. A photo with no usable background
reference (sample_background_reference returns None) no longer hard-
rejects - it simply proceeds without that one supporting signal, relying
on geometry alone, which is consistent with "color is a secondary
confidence cue, not a requirement."
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
        # NOTE: deliberately NOT called "uncertainty_mm". This is the spread
        # between independent segmentation-method estimates on THIS photo -
        # it says nothing about print scale error, lens distortion, or
        # ArUco corner localization error. Calling it "measurement
        # uncertainty" would overstate what we actually know until it's
        # been calibrated empirically against real caliper-measured rings.
        self.detection_spread_mm = None
        self.candidates_mm = []
        self.family_estimates = {}
        self.debug_img = None
        # Populated by the material-contrast validation gate (see
        # validate_material_contrast): every candidate boundary that looked
        # geometrically plausible but was rejected because its interior
        # didn't match the photo's own background, or its immediate
        # exterior didn't sufficiently differ from it - i.e. candidates
        # that were probably the outer edge, a bevel, a reflection, or a
        # cast shadow rather than the true inner hole. Each entry is
        # (family_name, diameter_mm, reason). Kept on the result (not just
        # logged) so a debug overlay can show exactly what was rejected and
        # why, not just what was accepted.
        self.rejected_candidates = []
        # GEOMETRY-PRIMARY REVISION: reporting fields for the winning
        # candidate's own geometric evidence, per the explicit ask to be
        # able to report "selected inner boundary, fitted ellipse, visible
        # boundary coverage %, fit residual, accepted/rejected, measured
        # diameter" for any photo evaluated against this pipeline.
        self.boundary_coverage = None   # fraction [0,1], see edge_support_fraction
        self.fit_residual = None        # None when the winning candidate came from a
                                         # source with no residual concept (radial peak/holecolor)
        self.color_confidence = None    # weak secondary cue only, never gates ACCEPT/REJECT

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

def segmentation_candidates(rect_gray, marker_mask):
    """Multiple independent segmentation methods -> list of binary edge maps."""
    variants = []

    for lo, hi in [(20,60),(30,90),(40,120),(60,150)]:
        e = cv2.Canny(rect_gray, lo, hi)
        variants.append(("canny_%d_%d"%(lo,hi), e))

    adap = cv2.adaptiveThreshold(rect_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 35, 5)
    variants.append(("adaptive", cv2.Canny(adap, 50, 150)))

    _, otsu = cv2.threshold(rect_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    variants.append(("otsu", cv2.Canny(otsu, 50, 150)))

    out = []
    for name, e in variants:
        e = cv2.bitwise_and(e, e, mask=marker_mask)
        # IMPORTANT: use morphological CLOSE, not dilate, to bridge small gaps
        # in the edge line. dilate() thickens the line into a band, and since
        # we deliberately pick the SMALLEST circular candidate (the hole),
        # that band's own inner edge gets mistaken for the ring's true inner
        # boundary -> systematic inward bias (~0.2mm, confirmed empirically).
        # close() re-erodes after dilating, closing small gaps while leaving
        # the line's true centerline position essentially unchanged.
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        out.append((name, e))
    return out

def all_ring_candidates(edge_map, min_diam_px=30, min_circularity=0.35, min_axis_ratio=0.6):
    """Return every plausible elliptical contour, not just the single 'best' one.
    We deliberately do NOT pick by score alone: a ring's OUTER edge is often just
    as circular/clean as its INNER edge (the hole), so a pure quality score
    tends to latch onto the outer boundary. Domain knowledge - the hole is
    always the smaller of the (roughly concentric) candidates - is applied by
    the caller via geometric_plausibility(), not here.

    GEOMETRY-PRIMARY REVISION: min_circularity was 0.65, which in practice
    required a contour to be close to a fully CLOSED loop - exactly the
    requirement that made this pipeline reject normal photos where a small
    shadow or specular highlight splits the true inner-boundary contour into
    an open arc or two disconnected pieces. Lowered to 0.35 (still enough to
    exclude genuinely non-circular junk - a stray line segment or a sharp
    corner scores far below that) so a partial boundary can still produce an
    ellipse-fit candidate here; whether that partial boundary is actually
    trustworthy is now decided downstream by edge_support_fraction (how much
    of the fitted ellipse's full perimeter has real edge support ANYWHERE
    across all segmentation variants, not just in this one contour) and by
    fit residual - genuine geometric evidence, not by requiring the one
    contour that happened to trace it to already be a closed loop.
    min_axis_ratio similarly loosened (0.75 -> 0.6): a partial arc's own
    least-squares ellipse fit is noisier than a fully closed contour's, so a
    slightly looser axis-ratio floor avoids discarding an otherwise-good
    partial candidate purely for a fitting artifact - axis_ratio still feeds
    into the candidate's score either way.
    """
    contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for c in contours:
        if len(c) < 20:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0: continue
        area = cv2.contourArea(c)
        circularity = 4*np.pi*area/(perim*perim)
        if circularity < min_circularity:
            continue
        if len(c) < 5:
            continue  # fitEllipse requires >= 5 points
        ellipse = cv2.fitEllipse(c)
        (cx,cy),(MA,ma),angle = ellipse
        if MA == 0 or ma == 0 or MA < min_diam_px:
            continue
        axis_ratio = min(MA,ma)/max(MA,ma)
        if axis_ratio < min_axis_ratio:
            continue
        pts = c.reshape(-1,2).astype(np.float32)
        dists = []
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        for (x,y) in pts[::max(1,len(pts)//60)]:
            xt = (x-cx)*cos_a + (y-cy)*sin_a
            yt = -(x-cx)*sin_a + (y-cy)*cos_a
            r = np.hypot(xt/(MA/2), yt/(ma/2))
            dists.append(abs(r-1.0))
        residual = np.mean(dists) if dists else 1.0
        score = circularity * axis_ratio * (1.0/(1.0+residual*5))
        candidates.append(dict(ellipse=ellipse, circularity=circularity, axis_ratio=axis_ratio,
                                residual=residual, score=score, diam_px=(MA+ma)/2, center=(cx,cy)))
    return candidates


def edge_distance_transform(edge_map):
    """Precompute, once per edge map, the distance (in px) from every pixel
    to the nearest edge pixel. Reused by edge_support_fraction for every
    candidate ellipse checked against this edge map, so the expensive part
    (the transform itself) is paid once, not once per candidate."""
    inv = cv2.bitwise_not(edge_map)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def edge_support_fraction(edge_dist, ellipse, shape, n_bins=72, tol_px=4.0):
    """
    THE partial-boundary-tolerant coverage metric: what fraction of a
    candidate ellipse's own perimeter has real edge support SOMEWHERE,
    sampled at n_bins evenly-spaced angles rather than requiring one single
    contour to already be a fully closed loop.

    This is deliberately computed against edge_dist - typically a distance
    transform of the UNION of every segmentation variant's edge map, not
    just the one variant that happened to produce this particular candidate
    - so a boundary that's cleanly visible in, say, adaptive-threshold but
    broken into two arcs by a highlight in canny still gets credited with
    high coverage: the true physical boundary is there, even if any single
    method's own contour-closure test would have missed it.

    A small shadow or specular highlight affecting a MINORITY arc of an
    otherwise-visible boundary costs that arc's share of bins, not the
    whole candidate - exactly the "accept if a large majority of the inner
    boundary is visible" tolerance asked for, instead of an all-or-nothing
    closed-contour requirement.

    Returns a float in [0, 1]; 0.0 if no perimeter samples land inside the
    image (degenerate ellipse).
    """
    (cx, cy), (MA, ma), angle = ellipse
    h, w = shape
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    thetas = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    ex = (MA/2.0) * np.cos(thetas)
    ey = (ma/2.0) * np.sin(thetas)
    xs = cx + ex*cos_a - ey*sin_a
    ys = cy + ex*sin_a + ey*cos_a
    xi = np.round(xs).astype(np.int32)
    yi = np.round(ys).astype(np.int32)
    in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if not np.any(in_bounds):
        return 0.0
    xi, yi = xi[in_bounds], yi[in_bounds]
    dists = edge_dist[yi, xi]
    supported = np.count_nonzero(dists <= tol_px)
    return float(supported) / float(len(xi))


def geometric_plausibility(diam_px, center, outer_ring_hint,
                            min_band_ratio=1.03, max_band_ratio=3.2,
                            max_center_offset_frac=0.4):
    """
    Structural "could this be the ring's inner hole, given roughly where the
    ring is" check - CONTAINMENT inside the outer ring boundary, and
    CONCENTRICITY with it, per the geometry-primary redesign. Checked
    against outer_ring_hint (the coarse Hough/contour-refined localization
    that is already computed once per photo), not against a second,
    independently-detected "outer partner" candidate the way an earlier
    revision's inner_boundary_from_candidates required - that was a much
    stricter, more fragile requirement (it needed TWO separate contours in
    the SAME edge map to both survive and pair up), and it's redundant with
    outer_ring_hint, which already answers "roughly where and how big is the
    ring" for the whole photo.

    max_band_ratio widened from the old inner/outer pairing's 1.8 to 3.2:
    that 1.8 cap was calibrated for "this candidate's own paired outer
    contour", which by construction couldn't be far off; here it's compared
    against the coarse Hough hint, which can legitimately be a good deal
    larger than the true inner hole for a thick-banded ring. 3.2 stays a
    physically-reasoned, generous but finite bound (a ring whose hole is
    less than ~a third of its outer diameter is already an unusually thick
    band) rather than removing the containment check altogether.

    max_center_offset_frac=0.4 (not a tight value): outer_ring_hint is
    documented elsewhere in this file as COARSE localization only, and on
    an irregular/hammered band its Hough-fitted center can genuinely be
    some way off from the ring's true center - confirmed directly on a real
    photo during this redesign's own testing: an independent edge family
    AND the independent holecolor (color-topology) family agreed tightly
    with each other (within ~0.1mm) on a candidate whose center sat ~0.32x
    outer_ring_hint's own radius away from outer_ring_hint's center - well
    past an earlier, tighter 0.18 value this function was first written
    with, which discarded that cross-family-corroborated candidate for no
    reason other than the coarse hint itself being imprecisely centered.
    0.4 still rejects a candidate that's nowhere near the localized ring at
    all (the marker, background clutter, a decoy object elsewhere in
    frame), while no longer second-guessing a candidate that two
    INDEPENDENT families already agree on just because the one-shot Hough
    hint's own center estimate was off - the cross-family spatial consensus
    check later in measure_ring (CENTER_AGREEMENT_THRESHOLD_MM) is the
    better-informed judge of "do independent methods agree on WHERE",
    since it compares candidates to each other, not to a single coarse
    hint. This was found and reasoned from that structural testing
    observation, not from comparing output to this photo's own (unknown)
    true diameter - no ground truth for that photo exists in this
    pipeline's test set.

    Returns (passed: bool, info: dict) with the raw offset/ratio for
    reporting, even on a pass.
    """
    if outer_ring_hint is None:
        return True, {"center_offset_frac": None, "outer_inner_ratio": None}
    ocx, ocy, orad = outer_ring_hint
    cx, cy = center
    offset = float(np.hypot(cx - ocx, cy - ocy))
    offset_frac = offset / max(1.0, orad)
    ratio = orad / max(1.0, diam_px / 2.0)
    info = {"center_offset_frac": offset_frac, "outer_inner_ratio": ratio}
    if offset_frac > max_center_offset_frac:
        info["reason"] = "NOT_CONCENTRIC"
        return False, info
    if not (min_band_ratio <= ratio <= max_band_ratio):
        info["reason"] = "NOT_CONTAINED"
        return False, info
    return True, info


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
    tallest peak in the radius histogram. Used by locate_outer_ring_circle()
    to validate a Hough circle (a yes/no "does this look ring-like at all"
    check) - that caller only ever needs one answer, and changing its
    behavior has previously been shown to change which Hough circle gets
    picked as the outer boundary (a real regression - see git history
    around commits 3e07ad8 / 0767fc5), so it is kept exactly as originally
    calibrated and NOT reused internally by the multi-candidate path below.

    measure_ring() itself no longer uses this for its own final
    measurement - see radial_multi_peak_candidates() and the "Cross-family
    candidate consensus" section of measure_ring for why a single
    global-argmax peak per method is not enough to trust on its own.
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


def radial_multi_peak_candidates(gray_for_gradient, edge_map, center, outer_radius_px,
                                  min_band_ratio=1.07,
                                  max_band_ratio=1.8,
                                  min_support=0.03,
                                  min_prominence=1.25,
                                  gradient_angle_tolerance_deg=20,
                                  max_peaks=3):
    """
    Multi-candidate version of the radial fallback, used for the actual
    measurement (unlike radial_inner_candidate's single best-peak, which is
    reserved for Hough-circle validation - see its docstring).

    WHY MULTIPLE PEAKS, NOT ONE: taking only the single tallest bin
    (argmax) throws away information whenever a photo has more than one
    locally-strong radius - which real reflective/textured photos do
    often, from causes that have nothing to do with the true inner
    boundary (the outer edge's own blur/specular halo bleeding a little
    way into the search band; an unrelated internal reflection). Picking
    "the tallest one" per method and then only comparing those single
    picks across methods (the old design) meant that if two different
    methods' tallest peaks happened to be two different ARTIFACTS, cross-
    method agreement could never rescue the truth even when the true peak
    was sitting right there as each method's SECOND-best peak. Returning
    every genuine local peak (not just the best) lets the caller look for
    agreement across the full candidate set instead of just each method's
    single guess - the true boundary only has to be a locally-strong edge
    NOMINATED by multiple independent methods, not necessarily each
    method's own top pick.

    A "peak" here is a strict local maximum of the (already
    gradient-direction-filtered) radius histogram - hist[i] > hist[i-1]
    and hist[i] >= hist[i+1] - that clears the same per-candidate support/
    prominence bar as the single-peak version. Prominence is computed
    against the histogram's own 75th percentile as a local-noise-floor
    baseline (same idea as before, not a value fitted to any one photo).
    Peaks are returned ranked by prominence, capped at max_peaks so one
    noisy method can't flood the cross-method vote with many weak entries.
    """
    hist, r_min, r_max = _radial_edge_histogram(
        gray_for_gradient, edge_map, center, outer_radius_px,
        min_band_ratio, max_band_ratio, gradient_angle_tolerance_deg
    )
    if hist is None:
        return []

    cx, cy = center
    baseline = float(np.percentile(hist, 75))

    peaks = []
    n = len(hist)
    for i in range(n):
        v = float(hist[i])
        if v <= 0:
            continue
        left_ok = (i == 0) or (v > hist[i - 1])
        right_ok = (i == n - 1) or (v >= hist[i + 1])
        if not (left_ok and right_ok):
            continue

        radius = float(r_min + i)
        support = v / max(1.0, 2.0 * np.pi * radius)
        prominence = v / max(1.0, baseline)
        if support < min_support or prominence < min_prominence:
            continue

        peaks.append({
            "ellipse": ((float(cx), float(cy)), (2.0 * radius, 2.0 * radius), 0.0),
            "circularity": 1.0,
            "axis_ratio": 1.0,
            "residual": 0.0,
            "score": support,
            "diam_px": 2.0 * radius,
            "center": (float(cx), float(cy)),
            "radial_support": support,
            "radial_prominence": prominence,
        })

    peaks.sort(key=lambda c: c["radial_prominence"], reverse=True)
    peaks = peaks[:max_peaks]
    for rank, p in enumerate(peaks):
        p["peak_rank"] = rank  # 0 = this variant's single strongest peak
    return peaks


def sample_background_reference(rectified_bgr, marker_rect, outer_ring_hint,
                                  local_inner_scale=1.15, local_outer_scale=3.0,
                                  min_local_px=1500):
    """
    Sample the photo's own TABLE/BACKGROUND color, independent of what any
    ring/hole detector thinks. This is the missing ingredient that let
    earlier versions of this pipeline accept the outer edge, a bevel, a
    specular reflection, or a cast shadow as if it were the inner hole:
    every one of those is *geometrically* a plausible circle, and multiple
    segmentation methods can agree on the same wrong one (that's exactly
    the false-confidence failure mode this redesign targets) - but none of
    them is background-COLORED the way the true hole is, because the true
    hole shows the same table/background surface visible everywhere else
    around the ring.

    LOCAL-FIRST SAMPLING: wood grain and lighting are not uniform across a
    whole photo - a corner of the frame far from the ring can legitimately
    be a different color/brightness than the table immediately around the
    ring (a lamp to one side, a shadow falling across part of the table).
    What the material-contrast gate and the enclosed-hole detector actually
    need is what "background" looks like *right around the ring*, not a
    whole-frame average that can be pulled off by regions nowhere near it.
    So when outer_ring_hint localizes the ring, this samples an annulus
    around it first - from local_inner_scale*radius (safely clear of the
    ring's own outer edge and band, matching the margin validate_material_
    contrast already uses for its own exterior band) out to
    local_outer_scale*radius - and only falls back to the old whole-frame
    (marker-excluded) sample if that local annulus doesn't yield enough
    pixels (ring too close to the frame edge, hint overlaps the marker
    exclusion heavily, etc).

    MEDIAN (not mean) is used throughout so that even a sample still
    contaminated by a sliver of ring/marker pixels stays a robust estimate
    as long as background pixels are the majority - true for both the local
    annulus and the whole-frame fallback on any photo actually following
    the "near-overhead, ring and marker both clearly visible" capture
    instructions.

    Returns None (never a fabricated/degenerate reference) if too little of
    the frame is left after exclusions to trust either sample at all - the
    caller must treat that as "cannot validate the physical boundary" and
    retake, not fall back to geometry-only acceptance.
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


def _annulus_mask(shape, ellipse, inner_scale, outer_scale):
    (cx, cy), (MA, ma), angle = ellipse
    h, w = shape
    outer_m = np.zeros((h, w), dtype=np.uint8)
    axes_o = (max(1, int(MA / 2 * outer_scale)), max(1, int(ma / 2 * outer_scale)))
    cv2.ellipse(outer_m, (int(round(cx)), int(round(cy))), axes_o, angle, 0, 360, 255, -1)
    inner_m = np.zeros((h, w), dtype=np.uint8)
    axes_i = (max(1, int(MA / 2 * inner_scale)), max(1, int(ma / 2 * inner_scale)))
    cv2.ellipse(inner_m, (int(round(cx)), int(round(cy))), axes_i, angle, 0, 360, 255, -1)
    return cv2.bitwise_and(outer_m, cv2.bitwise_not(inner_m))


def validate_material_contrast(rect_lab, ellipse, bg_ref,
                                interior_scale=0.75,
                                ext_inner_scale=1.02, ext_outer_scale=1.10,
                                max_interior_chroma_z=3.0,
                                max_interior_brightness_z=3.0,
                                max_interior_darkness_z=6.0,
                                min_contrast_z=1.0):
    """
    THE core defense against "geometrically clean circle, wrong physical
    edge": accept a candidate boundary only if the region strictly INSIDE
    it looks like this photo's own background (consistent with "this is an
    open hole showing the table/background through it"), AND the region
    immediately OUTSIDE it looks meaningfully DIFFERENT from that
    background (consistent with "this is where the ring's material
    starts"). Both conditions are necessary and neither is optional:

    - Outer-edge-selected-as-inner fails on contrast: just outside the true
      outer edge is the table again, so d_out stays small - the exterior
      never becomes not-background, and the required contrast margin is
      never met.
    - A bevel or specular reflection on the ring's own surface fails on
      the interior check: its interior is still metal, not the background
      color, or a bright highlight ON metal - either way its CHROMA (Lab
      a*/b*) does not match the background's, or (for a highlight) it is
      much BRIGHTER than the background, so it's caught below.
    - A cast shadow next to the ring fails on contrast the other way: both
      inside AND outside a shadow blob are still the table (just dimmer),
      so the required "outside must differ from background, more than
      inside does" margin isn't met either.

    INTERIOR CHECK IS SPLIT INTO CHROMA AND (ONE-SIDED) BRIGHTNESS, not one
    combined isotropic Lab distance, for a real physical reason found while
    testing this against real photos (not synthetic renders): a ring
    physically sits ABOVE the table and casts its own shadow through/into
    its own hole, so the background surface seen THROUGH a genuine hole is
    often legitimately DARKER than the open background sampled right next
    to the ring, without being a different surface at all - same wood, same
    hue, just less lit. A single combined Lab distance can't tell that
    apart from "this is metal/a bevel, not background" without either
    rejecting genuine self-shadowed holes or (if loosened enough to let
    them through) also loosening the color-identity check that's actually
    doing the discriminating work. Splitting the checks keeps both: CHROMA
    (a*, b*) must match the background closely regardless of how dark it
    reads (self-shadowing changes lightness, not the underlying material's
    hue), while brightness is penalized asymmetrically: any amount BRIGHTER
    than background is suspicious (consistent with a highlight/reflection
    on metal, or with the physical reality that a background surface
    behind/under the ring cannot be lit MORE brightly than the open
    background right beside it), while DARKER is tolerated up to
    max_interior_darkness_z before being rejected too. That bound matters:
    an unconditional "any amount darker is fine" rule (tried first, and
    found to be wrong) turns out to erase the interior check's usefulness
    whenever the ring material happens to be close to achromatic (many
    metals are, and so are many light-toned backgrounds) - a dark metal
    band's only real distinguishing property from a light background can
    be brightness alone, so a rule that ignores darkness entirely lets the
    ring's own band read as "background" too. Real self-shadowing (the ring
    casting a shadow into its own hole) is a real but BOUNDED effect;
    max_interior_darkness_z=6.0 is a deliberately generous but finite
    allowance for it (roughly 1.5x the largest self-shadow darkening
    actually observed on a real validated photo), chosen for that physical
    reasoning and then checked against the whole regression suite as a
    fixed rule, not fitted to match any single photo's own reading.

    Distances are z-score-like (Lab distance divided by the background's
    own per-channel MAD, floored at 3.0 to avoid blowing up on a near-zero
    reading from a suspiciously uniform background) - dimensionless and
    self-calibrated per photo, not fixed absolute color/brightness values,
    so the same two thresholds are meant to generalize across lighting and
    table color rather than being tuned to any one photo's palette.
    max_interior_z=3.0 and min_contrast_z=2.5 are round, conservative
    starting values chosen from that reasoning alone, then checked against
    the whole regression suite as a fixed rule (never adjusted by looking
    at any single photo's known diameter) - see regression_suite.py.

    ext_inner_scale/ext_outer_scale are kept DELIBERATELY TIGHT (1.02-1.10,
    i.e. sampling only the 2%-10% band just past the candidate radius)
    rather than a wider band further out. Found empirically (structurally,
    not by tuning to a known diameter) while first building this check: a
    wider annulus (originally 1.15-1.45) reliably fails on thin-banded
    rings, because an annulus's pixel COUNT grows with radius, so a wide
    annulus is dominated by whichever part of it has the largest radius -
    if the true ring band ends before the annulus's outer edge (any band
    whose own outer/inner radius ratio is below the annulus's outer scale;
    real bands validated elsewhere in this file range from ~1.07x to 1.8x),
    the annulus's median silently drifts back to background-like, and a
    perfectly correct candidate gets wrongly rejected as
    INSUFFICIENT_BAND_CONTRAST. A tight band close to the boundary stays
    inside the ring material for the whole plausible band-thickness range
    this file already validates elsewhere, at the cost of being slightly
    more sensitive to boundary antialiasing - an acceptable trade since
    interior_scale=0.75 already keeps the interior sample well clear of
    that same antialiasing on the other side.

    Returns (passed: bool, info: dict) - info always carries enough (d_in,
    d_out, and on failure a reason code) to drive a debug overlay.

    GEOMETRY-PRIMARY REVISION: this function's result is no longer a hard
    gate in measure_ring() - it is one soft, secondary confidence signal
    (color_confidence) among several, used only to break ties between
    otherwise-equally-good geometric clusters. A `False` return here no
    longer removes a candidate from consideration on its own.

    When bg_ref is None (sample_background_reference couldn't get a
    trustworthy background sample - e.g. the ring/marker occupying nearly
    the whole frame, or the local annulus and the whole-frame fallback both
    coming up short), measure_ring() does not even call this function - it
    simply proceeds on geometry alone with color_confidence left neutral,
    per the explicit instruction that color/background similarity is a
    weak supporting cue, not a requirement. (An earlier revision returned
    False unconditionally here and had measure_ring() hard-reject with a
    dedicated retake reason, BACKGROUND_REFERENCE_UNAVAILABLE, before any
    candidate was generated - that reason code no longer exists.)
    """
    if bg_ref is None:
        return False, {"reason": "NO_BACKGROUND_REFERENCE", "d_in": None, "d_out": None}

    h, w = rect_lab.shape[:2]
    interior_mask = np.zeros((h, w), dtype=np.uint8)
    (cx, cy), (MA, ma), angle = ellipse
    axes_in = (max(1, int(MA / 2 * interior_scale)), max(1, int(ma / 2 * interior_scale)))
    cv2.ellipse(interior_mask, (int(round(cx)), int(round(cy))), axes_in, angle, 0, 360, 255, -1)
    ext_mask = _annulus_mask((h, w), ellipse, ext_inner_scale, ext_outer_scale)

    ys_in, xs_in = np.nonzero(interior_mask)
    ys_out, xs_out = np.nonzero(ext_mask)
    if len(xs_in) < 15 or len(xs_out) < 15:
        return False, {"reason": "INSUFFICIENT_SAMPLING", "d_in": None, "d_out": None}

    lab_med, lab_mad = bg_ref["lab_median"], bg_ref["lab_mad"]
    scale = np.maximum(lab_mad, 3.0)

    interior_med = np.median(rect_lab[ys_in, xs_in], axis=0)
    ext_med = np.median(rect_lab[ys_out, xs_out], axis=0)

    # Full (chroma + lightness) isotropic distance - used for the exterior
    # sample and the contrast margin, exactly as before: an outer edge or a
    # cast shadow both need to look like background in every channel to be
    # rejected there, and lightness is exactly what a cast shadow (dimmer,
    # same hue, both sides) needs to be judged on.
    d_in = float(np.sqrt(np.sum(((interior_med - lab_med) / scale) ** 2)))
    d_out = float(np.sqrt(np.sum(((ext_med - lab_med) / scale) ** 2)))

    # Interior-only checks - see docstring for why these are split from the
    # isotropic distance above.
    d_in_chroma = float(np.sqrt(np.sum(((interior_med[1:] - lab_med[1:]) / scale[1:]) ** 2)))
    d_in_brightness_signed = float((interior_med[0] - lab_med[0]) / scale[0])

    info = {
        "d_in": d_in, "d_out": d_out, "contrast": d_out - d_in,
        "d_in_chroma": d_in_chroma, "d_in_brightness_signed": d_in_brightness_signed,
    }

    if (d_in_chroma > max_interior_chroma_z
            or d_in_brightness_signed > max_interior_brightness_z
            or d_in_brightness_signed < -max_interior_darkness_z):
        info["reason"] = "INTERIOR_NOT_BACKGROUND"
        return False, info
    if (d_out - d_in) < min_contrast_z:
        info["reason"] = "INSUFFICIENT_BAND_CONTRAST"
        return False, info
    return True, info


def enclosed_hole_candidates(rect_lab, marker_rect, outer_ring_hint, bg_ref, min_diam_px=30):
    """
    THE actual "find the hole, not the strongest circle" detector, per the
    redesign: rather than scoring candidate circles by how clean/circular
    they look (which the outer edge, a bevel, or a reflection can all win),
    this directly segments background-colored pixels within the localized
    ring ROI, then uses connected-component topology to find regions of
    that background color that are FULLY ENCLOSED - i.e. not reachable
    from the ROI window's border without crossing non-background pixels.
    A component that touches the window border is just "the background
    around the ring" and is discarded; one that doesn't is, by
    construction, an island of background-colored pixels surrounded by
    ring material - which is exactly what the ring's hole physically is,
    and what an outer edge/bevel/reflection/shadow structurally is NOT
    (none of those enclose a background-colored island; they're edges
    within or beside the ring material itself).

    Runs two independent thresholds ("loose"/"strict" on background-match
    distance) as two sub-variants of one family ("holecolor_loose" /
    "holecolor_strict" -> family "holecolor" once split on "_", same
    convention as the canny_20_60 style names elsewhere in this file) so a
    single mis-set threshold can't be this method's only chance to
    contribute a candidate.

    Requires both outer_ring_hint (for the ROI window) and bg_ref (for
    what "background" means in this photo); returns [] without either -
    this method is a member of the family ensemble, not the only one, so
    an absent hint just means one fewer independent vote, not a crash.

    "Background-colored" is judged the same physically-motivated way
    validate_material_contrast judges its interior sample (see that
    function's docstring): by CHROMA (Lab a*/b*) matching closely, plus
    brightness only disqualifying a pixel when it's BRIGHTER than the
    background, never when it's darker. A ring sitting on a table casts its
    own shadow into its own hole, so the true hole is frequently a
    self-shadowed (darker, same-hue) patch of the background rather than an
    identically-lit one - judging by full brightness+chroma distance here
    would silently fail to find the hole at all on exactly those (common,
    real) photos, not because the topology is wrong but because the pixels
    never get classified as "background-like" in the first place.
    """
    if bg_ref is None or outer_ring_hint is None:
        return []

    cx, cy, r = outer_ring_hint
    h, w = rect_lab.shape[:2]
    win_r = int(np.ceil(r * 1.3))
    x0 = max(0, int(cx - win_r)); x1 = min(w, int(cx + win_r + 1))
    y0 = max(0, int(cy - win_r)); y1 = min(h, int(cy + win_r + 1))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return []
    win_lab = rect_lab[y0:y1, x0:x1]

    lab_med, lab_mad = bg_ref["lab_median"], bg_ref["lab_mad"]
    scale = np.maximum(lab_mad, 3.0)
    chroma_dist = np.sqrt((((win_lab[:, :, 1:] - lab_med[1:]) / scale[1:]) ** 2).sum(axis=2))
    brightness_signed = (win_lab[:, :, 0] - lab_med[0]) / scale[0]
    # "Background-like" = chroma within z_thresh AND not brighter than
    # background by more than z_thresh, with darker tolerated up to a fixed
    # bound (MAX_DARKNESS_Z, matching validate_material_contrast's
    # max_interior_darkness_z - see that function's docstring for why
    # "any amount darker" turned out to be wrong: it erased the ability to
    # tell an achromatic ring band from an achromatic background). Beyond
    # that bound, a pixel is genuinely too dark to be self-shadowed
    # background and is excluded regardless of chroma - this is what keeps
    # the ring band itself (a much larger drop than shadow alone produces)
    # from being misclassified as background-colored.
    MAX_DARKNESS_Z = 6.0
    too_dark = brightness_signed < -MAX_DARKNESS_Z
    dist = np.where(too_dark, np.inf, np.maximum(chroma_dist, np.clip(brightness_signed, 0, None)))

    mx0, my0, mx1, my1 = marker_rect
    pad = MARKER_MASK_PAD_PX

    candidates = []
    for suffix, z_thresh in [("loose", 2.5), ("strict", 1.5)]:
        bg_like = (dist <= z_thresh).astype(np.uint8) * 255

        # Bridge single/few-pixel false-positive "background-like" leaks
        # through the ring band before computing topology. A per-pixel
        # color threshold applied to a real (noisy, sometimes gradient-lit)
        # photo can transiently classify a thin strip of the ring band as
        # background wherever its shaded color briefly crosses close to the
        # background reference (seen concretely on a directional-lighting
        # test photo: a 1-2px leak connected the hole to the outer
        # background through an otherwise-solid band, so nothing was
        # topologically "enclosed" at all). MORPH_OPEN (erode then dilate)
        # removes thin spurious strands like that while leaving genuinely
        # large background regions (the true hole, the table around the
        # ring) intact, the same rationale as the morphological CLOSE
        # already used elsewhere in this file to bridge small edge gaps.
        bg_like = cv2.morphologyEx(bg_like, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Zero out any marker-mask overlap that happens to fall inside this
        # window (the ring can, per capture instructions, be placed close
        # to the marker) so the marker's own white paper background can
        # never seed or merge into a hole component.
        mmy0 = max(0, int(my0 - pad) - y0); mmy1 = min(bg_like.shape[0], int(my1 + pad) - y0)
        mmx0 = max(0, int(mx0 - pad) - x0); mmx1 = min(bg_like.shape[1], int(mx1 + pad) - x0)
        if mmy1 > mmy0 and mmx1 > mmx0:
            bg_like[mmy0:mmy1, mmx0:mmx1] = 0

        num_labels, labels = cv2.connectedComponents(bg_like, connectivity=8)
        border_labels = set(labels[0, :].tolist()) | set(labels[-1, :].tolist()) | \
                         set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
        border_labels.discard(0)

        for lbl in range(1, num_labels):
            if lbl in border_labels:
                continue  # reachable from the window edge - ordinary background, not an enclosed hole
            comp = (labels == lbl).astype(np.uint8) * 255
            area = int(np.count_nonzero(comp))
            if area < 200:
                continue
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            if len(c) < 20:
                continue
            ellipse = cv2.fitEllipse(c)
            (ecx, ecy), (MA, ma), angle = ellipse
            if MA == 0 or ma == 0 or MA < min_diam_px:
                continue
            axis_ratio = min(MA, ma) / max(MA, ma)
            if axis_ratio < 0.75:
                continue
            full_ellipse = ((ecx + x0, ecy + y0), (MA, ma), angle)
            candidates.append({
                "name": f"holecolor_{suffix}",
                "ellipse": full_ellipse,
                "circularity": None,
                "axis_ratio": axis_ratio,
                "residual": None,
                "score": axis_ratio,
                "diam_px": (MA + ma) / 2,
                "center": (ecx + x0, ecy + y0),
                "peak_rank": 0,
            })
    return candidates


def locate_outer_ring_circle(rect_gray, marker_rect):
    """
    Locate the ring region using a strong approximately circular edge.

    This is localization only: the returned circle is NEVER used as the
    measurement or as evidence for which boundary is correct - it only
    tells the rest of measure_ring roughly where to look (sizing the
    enclosed_hole_candidates search window, centering
    sample_background_reference's local background annulus). That
    downstream decision-making runs entirely through
    validate_material_contrast and cross-family/cross-topology agreement,
    so this function no longer needs to itself validate "is this really a
    ring" - it only needs a plausible, roughly-correctly-located, roughly-
    correctly-sized circle to point the rest of the pipeline at.

    Preference order: a Hough circle that ALSO shows plausible nested
    radial edge structure (checked via radial_inner_candidate) is preferred
    when one exists, largest first - this is usually the more precise
    localization. But on a clean, high-contrast photo, Hough can lock onto
    the ring's INNER hole directly (it can be the single most prominent
    circle in the frame), in which case there is no even-smaller nested
    structure to find and every candidate fails that check - which used to
    make this function return None outright, discarding localization
    entirely even though a perfectly usable circle was sitting right there.
    Since (per above) this hint is coarse-localization-only now, falling
    back to the largest Hough circle in the physically plausible range
    (whether or not it validated) is safe: measure_ring's own refinement
    step immediately after this call independently cross-checks the
    resulting radius against the edge-based contour candidates it already
    computes, and expands it if those show a larger extent nearby (see that
    refinement's docstring in measure_ring) - so an inner-hole-sized Hough
    circle still ends up corrected to roughly the true outer extent before
    it's used for anything.
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
    # entirely. measure_ring's own contour-based radius refinement corrects
    # this if it understates the true outer extent.
    return largest_any

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
    mx0, my0, mx1, my1 = marker_rect
    mask = np.ones_like(rect_gray)*255
    pad = MARKER_MASK_PAD_PX
    cv2.rectangle(mask, (int(mx0-pad), int(my0-pad)), (int(mx1+pad), int(my1+pad)), 0, -1)

    variants = segmentation_candidates(rect_gray, mask)

    # Coarse ROI localization ONLY - kept for exactly that purpose per this
    # redesign's scope. locate_outer_ring_circle (Hough) and the ArUco/
    # perspective rectification above tell us roughly WHERE the ring is;
    # neither is trusted to say WHICH boundary inside that ROI is the true
    # inner hole - that judgment now runs on geometric evidence (below),
    # not on "which circle looks strongest" and not on color/background
    # similarity either.
    outer_ring_hint = locate_outer_ring_circle(
        rect_gray,
        marker_rect
    )
    if os.environ.get("RING_DEBUG") and outer_ring_hint is not None:
        print(f"[RING_DEBUG] outer_ring_hint diam_mm={2*outer_ring_hint[2]/PX_PER_MM_OUT:.2f}")

    # Run every segmentation variant's raw contour search once, up front -
    # needed both for the ROI-radius refinement immediately below and for
    # the per-variant inner-boundary/radial-fallback decisions further down
    # (which reuse these same all_cands rather than recomputing them).
    per_variant_all_cands = []
    for name, edge_map in variants:
        per_variant_all_cands.append((name, edge_map, all_ring_candidates(edge_map)))

    # Coarse-ROI-radius refinement: locate_outer_ring_circle's Hough search
    # validates only "is there SOME smaller nested circular structure inside
    # this one", not specifically "is this circle the ring's OUTER edge" -
    # on a clean/high-contrast photo the inner hole itself can be the more
    # prominent Hough circle, in which case its radius understates the
    # ring's true outer extent. That matters now even though Hough is only
    # used for coarse localization: sample_background_reference's local
    # annulus and enclosed_hole_candidates' search window both start just
    # outside outer_ring_hint's radius, so an understated radius places them
    # partly ON the ring material instead of clear of it, contaminating the
    # very background reference the material-contrast gate depends on.
    # Fixed by cross-checking against the independent, already-computed
    # edge-based contour candidates: any of them centered near the Hough
    # hint is evidence of the ring's real extent regardless of which one
    # measure_ring later decides is the true inner boundary, so the hint's
    # radius is expanded (never shrunk, center never moved) to at least the
    # largest such candidate.
    if outer_ring_hint is not None:
        ocx, ocy, cur_r = outer_ring_hint
        best_r = cur_r
        for _, _, all_cands in per_variant_all_cands:
            for c in all_cands:
                (ex, ey), (MA, ma), _ = c['ellipse']
                if np.hypot(ex - ocx, ey - ocy) <= 25:
                    best_r = max(best_r, (MA + ma) / 4.0)
        if best_r != cur_r:
            outer_ring_hint = (ocx, ocy, best_r)
            if os.environ.get("RING_DEBUG"):
                print(f"[RING_DEBUG] outer_ring_hint radius refined: "
                      f"{2*cur_r/PX_PER_MM_OUT:.2f}mm -> {2*best_r/PX_PER_MM_OUT:.2f}mm")

    # This photo's own background color/luminance, sampled independently of
    # any ring/hole candidate - still computed (used below as the weak
    # SECONDARY confidence cue, and to drive enclosed_hole_candidates/
    # "holecolor" as one more independent candidate family), but per the
    # geometry-primary redesign this is no longer required for measurement
    # to proceed at all. A photo where the local annulus and whole-frame
    # fallback both come up short (ring/marker filling nearly the whole
    # frame) simply proceeds on geometry alone, with "holecolor" unable to
    # contribute a vote and color_confidence left neutral - it is NOT an
    # honest retake by itself anymore, because a missing color reference
    # says nothing about whether the boundary itself is geometrically
    # recoverable.
    rect_lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_ref = sample_background_reference(rectified, marker_rect, outer_ring_hint)

    # Union edge map across every segmentation variant, and its distance
    # transform, computed ONCE and reused by edge_support_fraction for
    # every candidate from every source (contour-based, radial-peak, and
    # holecolor alike) - see edge_support_fraction's docstring for why the
    # UNION (not any single variant's own edge map) is the right thing to
    # score "visible boundary coverage" against: a boundary that's cleanly
    # closed in adaptive-threshold but broken into two arcs by a highlight
    # in canny is still a boundary that's genuinely there, which the union
    # correctly credits.
    union_edge = np.zeros_like(rect_gray)
    for _, edge_map in variants:
        union_edge = cv2.bitwise_or(union_edge, edge_map)
    union_edge_dist = edge_distance_transform(union_edge)
    union_shape = rect_gray.shape

    # Geometric acceptance thresholds - reasoned from first principles (what
    # "a large majority of the boundary is visible" and "a well-fitting
    # ellipse" should mean), NOT fitted by checking against any specific
    # photo's known diameter (see the module's no-overfitting policy).
    # MIN_ACCEPT_COVERAGE=0.55: a boundary with the clear majority (>55%) of
    # its own perimeter showing real edge support somewhere across every
    # segmentation variant is "geometrically recoverable" per the explicit
    # instruction to tolerate a small shadow/highlight arc rather than
    # requiring full 100% closure. MAX_FIT_RESIDUAL=0.4: residual is the
    # mean absolute deviation (in normalized ellipse-radius units) of a
    # contour's own points from their fitted ellipse - 0.4 is a generous
    # but finite cap that still excludes a badly-fitting, likely-spurious
    # shape while tolerating ordinary photographic/segmentation noise.
    MIN_ACCEPT_COVERAGE = 0.55
    MAX_FIT_RESIDUAL = 0.4

    diam_candidates = []   # (name, diameter_mm, cand) - passed all geometric checks
    rejected = []           # (name, diameter_mm, cand, reason) - for the debug overlay

    def consider(name, cand):
        """
        Score ONE candidate ellipse (from any source - closed/partial
        contour, radial histogram peak, or holecolor connected-component)
        against the geometric evidence this redesign actually decides on:
        containment + concentricity with the coarse ROI (geometric_-
        plausibility), visible boundary coverage % (edge_support_fraction,
        the partial-boundary tolerance), and fit residual where that
        concept applies. Candidates that fail any of these are recorded in
        `rejected` with a specific reason for the debug overlay, never
        silently dropped. Color/background similarity is NOT checked here -
        see the soft color_confidence annotation added after this filtering
        step, which never gates acceptance.
        """
        ellipse = cand['ellipse']
        diam_px = cand['diam_px']
        diameter_mm = diam_px / PX_PER_MM_OUT
        coverage = edge_support_fraction(union_edge_dist, ellipse, union_shape)
        plausible, geo_info = geometric_plausibility(diam_px, cand['center'], outer_ring_hint)
        cand = dict(cand)
        cand['coverage'] = coverage
        cand['geo_info'] = geo_info

        if not plausible:
            rejected.append((name, diameter_mm, cand, geo_info.get("reason", "NOT_PLAUSIBLE")))
            return
        if coverage < MIN_ACCEPT_COVERAGE:
            rejected.append((name, diameter_mm, cand, "LOW_BOUNDARY_COVERAGE"))
            return
        residual = cand.get('residual')
        if residual is not None and residual > MAX_FIT_RESIDUAL:
            rejected.append((name, diameter_mm, cand, "HIGH_FIT_RESIDUAL"))
            return

        diam_candidates.append((name, diameter_mm, cand))
        if os.environ.get("RING_DEBUG"):
            print(f"[RING_DEBUG] candidate fam={name:<16} diam={diameter_mm:7.2f}mm "
                  f"coverage={coverage:.2f} residual={residual}")

    # Source 1: contour/ellipse-fit candidates per segmentation variant -
    # EVERY plausible one, not a single pre-selected "best", per point 1 of
    # the redesign ("detect multiple closed contour / ellipse candidates").
    # Which one (if any) is trustworthy is now decided entirely by consider()
    # above, not by requiring this contour to already have a separately-
    # detected concentric outer partner in the same edge map.
    for name, edge_map, all_cands in per_variant_all_cands:
        for cand in all_cands:
            consider(name, cand)

    # Source 2: radial-histogram peaks - kept as a supplementary hypothesis
    # source for every variant (not only when the contour path found
    # nothing), since a real reflective/textured ring can have its true
    # boundary show up more clearly as a radial peak than as any one clean
    # contour. geometric_plausibility + coverage filtering above decide
    # whether any of these are actually worth keeping.
    if outer_ring_hint is not None:
        outer_cx, outer_cy, outer_radius = outer_ring_hint
        for name, edge_map, _all_cands in per_variant_all_cands:
            radial_cands = radial_multi_peak_candidates(
                rect_gray, edge_map, (outer_cx, outer_cy), outer_radius
            )
            for rc in radial_cands:
                consider(name, rc)

    # Source 3: the direct hole-region ("holecolor") detector - still one
    # more independent candidate family in the same pool, per point 4 of
    # the redesign ("color/background similarity only a weak supporting
    # feature, not a hard gate") - it contributes votes exactly like any
    # edge-based family, but is no longer REQUIRED for any cluster to
    # qualify (see has_topological_confirmation's removal below).
    for hc in enclosed_hole_candidates(rect_lab, marker_rect, outer_ring_hint, bg_ref):
        consider(hc['name'], hc)

    res.rejected_candidates = rejected

    # Color/background similarity as a WEAK, NON-BLOCKING secondary cue per
    # point 4 of the redesign: every candidate that already passed the
    # geometric filtering above gets an additional, purely informational
    # color_confidence annotation (1.0 = looks like an actual hole against
    # this photo's own sampled background, 0.4 = doesn't, 0.5 = no
    # trustworthy background sample to judge by at all). This NEVER removes
    # a candidate - it is only used below to break ties between more than
    # one otherwise-equally-good cluster, and to report an honest
    # "confidence" alongside the final answer.
    for i, (name, diameter_mm, cand) in enumerate(diam_candidates):
        if bg_ref is not None:
            passed_color, color_info = validate_material_contrast(rect_lab, cand['ellipse'], bg_ref)
            cand['color_confidence'] = 1.0 if passed_color else 0.4
            cand['color_info'] = color_info
        else:
            cand['color_confidence'] = 0.5
            cand['color_info'] = None
        diam_candidates[i] = (name, diameter_mm, cand)

    def draw_debug(final_ellipse=None):
        """
        Built and returned only once the outcome is known, so the overlay
        always matches what was actually decided (not a snapshot from
        mid-computation). Always draws the localization ROI and every
        candidate that survived geometric filtering; additionally draws
        REJECTED candidates (labeled with their rejection reason) when
        RING_DEBUG_OVERLAY is set, since that view is verbose/technical and
        not meant for the default end-user image. The final selected
        boundary (if any) is drawn last, bold, on top.
        """
        dbg = rectified.copy()
        line_thickness = max(3, dbg.shape[1] // 300)
        family_colors = {
            "canny": (0, 255, 0),
            "adaptive": (255, 0, 0),
            "otsu": (0, 255, 255),
            "holecolor": (255, 0, 255),
        }
        fallback_colors = [(0, 128, 255), (255, 255, 0), (128, 0, 255)]

        if outer_ring_hint is not None:
            ocx, ocy, orad = outer_ring_hint
            cv2.circle(dbg, (int(ocx), int(ocy)), int(orad), (200, 200, 200), max(1, line_thickness // 2))

        for name, diameter_mm, cand in diam_candidates:
            family = name.split("_")[0]
            color = family_colors.get(family, fallback_colors[hash(family) % len(fallback_colors)])
            cv2.ellipse(dbg, cand['ellipse'], color, line_thickness)

        if os.environ.get("RING_DEBUG_OVERLAY"):
            for name, diameter_mm, cand, reason in rejected:
                (rcx, rcy), (rMA, rma), rangle = cand['ellipse']
                overlay = dbg.copy()
                cv2.ellipse(overlay, cand['ellipse'], (0, 0, 180), max(1, line_thickness // 2))
                dbg = cv2.addWeighted(overlay, 0.6, dbg, 0.4, 0)
                cov = cand.get('coverage')
                cov_str = f"cov={cov:.2f}" if cov is not None else ""
                label = f"{name}: {diameter_mm:.1f}mm REJECTED {reason} {cov_str}"
                text_y = max(15, int(rcy - rMA / 2) - 6)
                cv2.putText(dbg, label, (max(0, int(rcx - rMA / 2)), text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1, cv2.LINE_AA)

        if final_ellipse is not None:
            cv2.ellipse(dbg, final_ellipse, (255, 255, 255), line_thickness + 2)

        return dbg

    res.candidates_mm = diam_candidates

    # --- Cross-family candidate CONSENSUS, not per-family single-best-vote. ---
    #
    # canny_20_60 .. canny_60_150 are the same underlying detector with only
    # a threshold changed: they are correlated and tend to succeed or fail
    # together (proved with the dilate-bias bug - all 4 canny variants were
    # wrong in the same direction at once). Counting them as "4 independent
    # confirmations" overstates confidence, so a FAMILY (canny/adaptive/
    # otsu) is still the unit of "one independent method", exactly as
    # before.
    #
    # What's different from the old design: instead of first collapsing
    # each family down to ONE value (that family's single best/only guess)
    # and only then checking whether those single per-family values agree,
    # every surviving candidate from every family is pooled together and
    # grouped by proximity FIRST. This matters because a family's own best
    # guess can be wrong while a candidate value close to the truth still
    # exists somewhere in its (now multi-candidate) output - collapsing too
    # early can throw that away before cross-family agreement ever gets a
    # chance to find it. The accepted answer is the tightest group of
    # candidate diameters that has members from at least MIN_FAMILIES
    # distinct families - i.e. "multiple independent methods each nominate
    # a candidate at essentially the same physical size", which is what
    # "these methods agree" should actually mean, rather than "each
    # method's single top pick happens to match another method's single
    # top pick".
    MIN_FAMILIES = 2         # need at least 2 genuinely different methods to agree
    CLUSTER_TOL_MM = 1.0     # same tolerance the old design used for "do family votes agree"
    # NOT loosened for holecolor. An earlier version of this file raised
    # this from 2.0mm to 3.0mm specifically because real_ring_B showed a
    # ~2.5mm holecolor-vs-canny center gap - which is exactly the kind of
    # per-photo tolerance tuning this project's own policy forbids (see the
    # module docstring). The actual cause was traced to
    # sample_background_reference using a whole-frame background estimate:
    # under uneven lighting, a global reference can mismatch the wood color
    # *right around the ring*, causing enclosed_hole_candidates' background-
    # color segmentation to include/exclude pixels asymmetrically and pull
    # its fitted center off from the true one. That's fixed at the source by
    # sampling background locally (see sample_background_reference's LOCAL-
    # FIRST SAMPLING), which removes the need to loosen this threshold at
    # all. Kept at 2.0mm - the value that was already validated (not
    # discovered by tuning) against synth_decoy_object_adversarial's
    # genuinely-different-object case.
    CENTER_AGREEMENT_THRESHOLD_MM = 2.0

    tagged = []  # (family, diameter_mm, cx, cy, peak_rank, ellipse, cand)
    for name, diameter_mm, cand in diam_candidates:
        family = name.split("_")[0]   # "canny_20_60" -> "canny"
        cx, cy = cand['center']
        # Contour-based candidates are a single, direct measurement (there
        # is no "2nd-best contour"), so treat them as rank 0 - the same
        # standing as a variant's own strongest radial peak.
        peak_rank = cand.get('peak_rank', 0)
        tagged.append((family, diameter_mm, cx, cy, peak_rank, cand['ellipse'], cand))

    if os.environ.get("RING_DEBUG"):
        for t in sorted(tagged, key=lambda x: x[1]):
            print(f"[RING_DEBUG] fam={t[0]:<8} diam={t[1]:7.2f}mm center=({t[2]:.0f},{t[3]:.0f}) peak_rank={t[4]}")

    all_families = {t[0] for t in tagged}
    res.family_estimates = {
        fam: float(np.median([t[1] for t in tagged if t[0] == fam]))
        for fam in all_families
    }

    if len(all_families) < MIN_FAMILIES:
        res.reason = "RING_NOT_FOUND"
        res.debug_img = draw_debug()
        return res

    # Greedy 1D chain-clustering on diameter_mm: sorted candidates within
    # CLUSTER_TOL_MM of their cluster's running extent join the same
    # cluster. Simple and deterministic - no fitting to any known answer,
    # just "how close together are these numbers".
    tagged_sorted = sorted(tagged, key=lambda t: t[1])
    clusters = []
    for item in tagged_sorted:
        if clusters and item[1] - clusters[-1][-1][1] <= CLUSTER_TOL_MM:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    def families_in(cluster):
        return {c[0] for c in cluster}

    def has_a_top_peak(cluster):
        # Require EVERY distinct family contributing to this cluster to be
        # represented there by its OWN strongest signal (rank 0) - not just
        # one family's top pick plus another family's weak/low-ranked
        # secondary peak that happens to land nearby. Checked and tightened
        # after finding a concrete counter-example: an earlier version of
        # this check required only ANY member to be rank 0, which let a
        # cluster through where one family's top pick (rank 0) coincided
        # with a SECOND family's 2nd-choice peak (rank 1) - a materially
        # weaker form of agreement than "two methods each independently
        # nominate this as their own best answer". On a test photo this
        # produced a confidently wrong, deceptively tight-spread result
        # while a different cluster - where every contributing family's own
        # top pick agreed - was the one actually consistent with the known
        # answer. Requiring full top-pick agreement across every family in
        # the cluster (not just one) is still a purely structural,
        # photo-agnostic property of the candidate set, not a value fitted
        # to that photo's own numbers.
        fams = families_in(cluster)
        return all(any(c[0] == fam and c[4] == 0 for c in cluster) for fam in fams)

    # GEOMETRY-PRIMARY REVISION: a mandatory "holecolor + >=1 edge family"
    # requirement (has_topological_confirmation) used to sit here. Removed
    # per explicit instruction - color/background similarity is now a weak
    # supporting cue (see the color_confidence annotation above and the
    # tie-breaking in cluster_key below), never a requirement for a cluster
    # to qualify. Cross-FAMILY agreement (>=2 distinct families, whichever
    # they are - edge-based or holecolor) combined with each family's own
    # top pick (has_a_top_peak) and, for every surviving candidate, the
    # geometric evidence already enforced in consider() above (containment,
    # concentricity, boundary coverage, fit residual) is what now has to
    # carry the "this is trustworthy" judgment on its own.
    qualifying = [
        c for c in clusters
        if len(families_in(c)) >= MIN_FAMILIES and has_a_top_peak(c)
    ]

    if os.environ.get("RING_DEBUG"):
        print(f"[RING_DEBUG] {len(qualifying)} qualifying cluster(s):")
        for c in qualifying:
            diams = [x[1] for x in c]
            print(f"[RING_DEBUG]   range=[{min(diams):.2f},{max(diams):.2f}] families={families_in(c)}")

    if not qualifying:
        # Enough independent methods produced SOMETHING, but no group of
        # candidates from different methods landed close together - the
        # methods never actually agreed on a single boundary.
        res.reason = "RING_EDGE_UNSTABLE"
        fam_values = np.array(list(res.family_estimates.values()))
        res.detection_spread_mm = float(fam_values.max() - fam_values.min())
        res.debug_img = draw_debug()
        return res

    # Prefer the cluster with the most distinct families behind it (still
    # the primary, purely structural signal: more independent methods
    # agreeing is stronger evidence regardless of any one photo's answer).
    # Ties are broken first by GEOMETRIC evidence (higher average boundary
    # coverage + lower fit residual - still geometry, still primary), and
    # only after that by color_confidence, per the explicit instruction
    # that color/background similarity is a weak secondary cue, not a
    # primary ranking signal - it only ever matters when geometry alone
    # left more than one cluster tied. Final tie-break is spread
    # (tightest group wins). None of these are fitted to any known answer.
    def cluster_key(c):
        diams = [x[1] for x in c]
        spread = max(diams) - min(diams)
        coverages = [m[6].get('coverage', 0.0) for m in c]
        residuals = [m[6].get('residual') for m in c if m[6].get('residual') is not None]
        avg_coverage = float(np.mean(coverages)) if coverages else 0.0
        avg_residual = float(np.mean(residuals)) if residuals else 0.0
        colors = [m[6].get('color_confidence', 0.5) for m in c]
        avg_color = float(np.mean(colors)) if colors else 0.5
        geometric_quality = avg_coverage - avg_residual  # higher coverage, lower residual = better
        return (-len(families_in(c)), -geometric_quality, -avg_color, spread)

    best_cluster = min(qualifying, key=cluster_key)

    # Per-family representative value WITHIN the winning cluster (median,
    # in case a family contributed more than one peak to it) - this keeps
    # the same "one vote per family" semantics the rest of the app/UI
    # already documents "cross-method spread" as meaning.
    cluster_family_diams = {}
    cluster_family_centers = {}
    for fam, dmm, cx, cy, _peak_rank, _ellipse, _cand in best_cluster:
        cluster_family_diams.setdefault(fam, []).append(dmm)
        cluster_family_centers.setdefault(fam, []).append((cx, cy))

    # Representative ellipse for the debug overlay's final highlight, and
    # for the "selected inner boundary / fitted ellipse / coverage % / fit
    # residual" reporting fields below - doesn't affect the reported
    # diameter (still the per-family median below). Prefer a member that
    # was its own method's top pick.
    _top_rank_members = [m for m in best_cluster if m[4] == 0]
    _rep_member = _top_rank_members[0] if _top_rank_members else best_cluster[0]
    winning_ellipse = _rep_member[5]
    _rep_cand = _rep_member[6]

    family_estimates_in_cluster = {
        fam: float(np.median(vals)) for fam, vals in cluster_family_diams.items()
    }
    family_centers_in_cluster = {
        fam: (float(np.median([c[0] for c in vals])), float(np.median([c[1] for c in vals])))
        for fam, vals in cluster_family_centers.items()
    }

    # --- Spatial consensus: families must agree on WHERE, not just diameter. ---
    # Comparing diameters alone can't tell "two methods found the same hole"
    # from "two methods found two different circular things in the scene"
    # (a reflection here, a background pattern there) that happen to be
    # similar sizes. Require the family centroids (within the winning
    # cluster) to be close together, in the same physical units (mm) as
    # everything else we report.
    centers_mm = {fam: (x/PX_PER_MM_OUT, y/PX_PER_MM_OUT) for fam,(x,y) in family_centers_in_cluster.items()}
    fams = list(centers_mm.keys())
    max_center_dist_mm = 0.0
    for i in range(len(fams)):
        for j in range(i+1, len(fams)):
            (x1,y1), (x2,y2) = centers_mm[fams[i]], centers_mm[fams[j]]
            dist = np.hypot(x1-x2, y1-y2)
            max_center_dist_mm = max(max_center_dist_mm, dist)

    if max_center_dist_mm > CENTER_AGREEMENT_THRESHOLD_MM:
        # families found geometrically DIFFERENT objects, not the same hole
        # measured slightly differently - this is a distinct failure mode
        # from "same object, disagreeing size" and deserves its own code.
        res.reason = "INCONSISTENT_DETECTION"
        res.debug_img = draw_debug()
        return res

    fam_values = np.array(list(family_estimates_in_cluster.values()))
    spread = float(fam_values.max() - fam_values.min())
    median = float(np.median(fam_values))

    res.ok = True
    res.diameter_mm = median
    res.detection_spread_mm = max(spread, 0.05)  # floor so we never claim impossible precision
    res.boundary_coverage = _rep_cand.get('coverage')
    res.fit_residual = _rep_cand.get('residual')
    res.color_confidence = _rep_cand.get('color_confidence')
    res.debug_img = draw_debug(final_ellipse=winning_ellipse)
    return res
