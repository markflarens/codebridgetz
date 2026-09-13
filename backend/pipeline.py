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

DETECTION ARCHITECTURE (redesigned around inner-hole segmentation, not
strongest-circle detection): earlier versions of this file scored
candidate circles by how clean/closed/circular they looked and by how
many segmentation methods agreed on one - which a ring's OUTER edge, a
polished bevel, a specular reflection, or a cast shadow can all satisfy
just as well as the true inner hole, and multiple methods CAN agree on
the same wrong one (a real, confirmed failure mode: two methods agreeing
is not evidence they found the right edge, only that they found the same
edge). ArUco calibration, perspective rectification, and Hough circle
detection (locate_outer_ring_circle) are kept, but strictly for coarse
ROI localization - "roughly where is the ring" - never for deciding which
boundary inside that ROI is correct.

The actual boundary decision now runs through validate_material_contrast:
every candidate, from every family (contour-based, radial-histogram, and
the enclosed-hole detector below), must have an INTERIOR that matches
this photo's own sampled background color (sample_background_reference)
and an immediate EXTERIOR that is meaningfully different from it, before
it is even allowed to compete for cross-method agreement. This directly
targets each of the reported failure modes: the outer edge fails because
just past it is background again (no contrast); a bevel/reflection fails
because its interior is still ring material, not background; a cast
shadow fails because both sides of it are still background. Cross-family
agreement (still required, still >=2 distinct families) is therefore
agreement among candidates that have ALREADY been confirmed to look like
an actual hole - low spread among rejected candidates can no longer
produce a false accept, because rejected candidates never reach that
stage.

enclosed_hole_candidates is a new, independent family ("holecolor") that
finds the hole directly rather than inferring it from edges: it segments
background-colored pixels in the ROI and uses connected-component
topology to find islands of that color fully enclosed by non-background
material - which is what a hole structurally IS, and what an edge
artifact structurally is NOT.
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

def all_ring_candidates(edge_map, min_diam_px=30):
    """Return every plausible circular contour, not just the single 'best' one.
    We deliberately do NOT pick by score alone: a ring's OUTER edge is often just
    as circular/clean as its INNER edge (the hole), so a pure quality score
    tends to latch onto the outer boundary. Domain knowledge - the hole is
    always the smaller of the (roughly concentric) candidates - has to be
    applied explicitly by the caller."""
    contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300 or len(c) < 20:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0: continue
        circularity = 4*np.pi*area/(perim*perim)
        if circularity < 0.65:
            continue
        ellipse = cv2.fitEllipse(c)
        (cx,cy),(MA,ma),angle = ellipse
        if MA == 0 or ma == 0 or MA < min_diam_px:
            continue
        axis_ratio = min(MA,ma)/max(MA,ma)
        if axis_ratio < 0.75:
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

def inner_boundary_from_candidates(candidates, quality_floor=0.5,
                                    min_band_ratio=1.03, max_band_ratio=1.8,
                                    max_center_offset_px=8):
    """Among plausible circular contours, pick the one that looks like the
    INSIDE of a ring band - not just "the smallest circle anywhere in the
    scene". A lone small circle (a decoy: coin, button, printed dot,
    reflection) can otherwise win purely for being small and round, and -
    critically - do so CONSISTENTLY across every segmentation method, since
    the bias is in the shared heuristic, not in per-method noise. Cross-
    family spatial consensus alone does NOT catch this, because agreeing
    families can agree on the same wrong object.

    Fix: require a genuine ring-band structure. For a candidate to be
    accepted as the inner boundary, there must ALSO be a larger, roughly
    concentric candidate in the same edge map (its plausible outer edge).
    A lone circle with no larger concentric partner is rejected outright,
    regardless of how clean/circular it looks.
    """
    good = [c for c in candidates if c['score'] > quality_floor]
    if not good:
        return None

    good_sorted = sorted(good, key=lambda c: c['diam_px'])
    for inner in good_sorted:
        icx, icy = inner['center']
        for outer in good:
            if outer is inner:
                continue
            ocx, ocy = outer['center']
            center_offset = np.hypot(icx-ocx, icy-ocy)
            ratio = outer['diam_px'] / inner['diam_px']
            if center_offset <= max_center_offset_px and min_band_ratio <= ratio <= max_band_ratio:
                return inner  # confirmed: this circle has a plausible concentric outer partner
    return None  # no candidate had a matching ring-band partner - don't guess


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


def sample_background_reference(rectified_bgr, marker_rect, outer_ring_hint):
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

    Excludes the marker (plus its existing mask padding) and, when
    available, the localized ring region (inflated 1.4x) from the sample.
    Falls back to "marker-only excluded" when outer_ring_hint is None
    (Hough couldn't localize anything) - a ring normally occupies a
    minority of the frame, and MEDIAN (not mean) is used specifically so
    that even a sample still contaminated by ring pixels stays a robust
    background estimate as long as background pixels are the majority,
    which holds for any photo actually following the "near-overhead, ring
    and marker both clearly visible" capture instructions.

    Returns None (never a fabricated/degenerate reference) if too little
    of the frame is left after exclusions to trust the sample at all.
    """
    h, w = rectified_bgr.shape[:2]
    exclude = np.zeros((h, w), dtype=bool)

    mx0, my0, mx1, my1 = marker_rect
    pad = MARKER_MASK_PAD_PX
    ex0, ey0 = max(0, int(mx0 - pad)), max(0, int(my0 - pad))
    ex1, ey1 = min(w, int(mx1 + pad)), min(h, int(my1 + pad))
    exclude[ey0:ey1, ex0:ex1] = True

    if outer_ring_hint is not None:
        ocx, ocy, orad = outer_ring_hint
        yy, xx = np.ogrid[:h, :w]
        exclude |= (xx - ocx) ** 2 + (yy - ocy) ** 2 <= (orad * 1.4) ** 2

    bg_mask = ~exclude
    if bg_mask.sum() < 500:
        return None

    lab = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_pixels = lab[bg_mask]
    lab_median = np.median(bg_pixels, axis=0)
    # Median Absolute Deviation, scaled to be std-equivalent for a normal
    # distribution (x1.4826) - a robust noise-floor estimate that isn't
    # blown up by the occasional outlier pixel (shadow edge, dust, a sliver
    # of ring that leaked past exclusion).
    lab_mad = np.median(np.abs(bg_pixels - lab_median), axis=0) * 1.4826
    return {"lab_median": lab_median, "lab_mad": lab_mad, "n_px": int(bg_mask.sum())}


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
                                max_interior_z=3.0, min_contrast_z=1.0):
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
      the interior check: its interior is still metal (or a bright
      highlight ON metal), not the background color/luminance, so d_in is
      large.
    - A cast shadow next to the ring fails on contrast the other way: both
      inside AND outside a shadow blob are still the table (just dimmer),
      so the required "outside must differ from background, more than
      inside does" margin isn't met either.

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

    When bg_ref is None (sample_background_reference couldn't get a
    trustworthy background sample - documented rare edge case, e.g. the
    ring/marker occupying nearly the whole frame), this check is SKIPPED
    (returns True) rather than blocking every candidate - a known,
    intentional fallback to the older geometry-only behavior for that
    narrow case, not a silent weakening of the general rule.
    """
    if bg_ref is None:
        return True, {"skipped": "no_background_reference"}

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

    d_in = float(np.sqrt(np.sum(((interior_med - lab_med) / scale) ** 2)))
    d_out = float(np.sqrt(np.sum(((ext_med - lab_med) / scale) ** 2)))
    info = {"d_in": d_in, "d_out": d_out, "contrast": d_out - d_in}

    if d_in > max_interior_z:
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
    dist = np.sqrt((((win_lab - lab_med) / scale) ** 2).sum(axis=2))

    mx0, my0, mx1, my1 = marker_rect
    pad = MARKER_MASK_PAD_PX

    candidates = []
    for suffix, z_thresh in [("loose", 2.5), ("strict", 1.5)]:
        bg_like = (dist <= z_thresh).astype(np.uint8) * 255

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
    Locate the ring region using a strong approximately circular outer edge.

    This is localization only: the outer circle is NOT used as the
    measurement. Diameter still comes from the detected inner boundary.

    A Hough candidate is accepted only if it also contains plausible
    inner circular edge evidence, which prevents a lone arbitrary circle
    from automatically becoming a ring.
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

    best = None

    for cx, cy, radius in circles[0]:
        cx = float(cx)
        cy = float(cy)
        radius = float(radius)

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

    if best is None:
        return None

    return best[1]

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
    # inner hole - that judgment now belongs entirely to the material-
    # contrast gate below, not to "which circle looks strongest".
    outer_ring_hint = locate_outer_ring_circle(
        rect_gray,
        marker_rect
    )
    if os.environ.get("RING_DEBUG") and outer_ring_hint is not None:
        print(f"[RING_DEBUG] outer_ring_hint diam_mm={2*outer_ring_hint[2]/PX_PER_MM_OUT:.2f}")

    # This photo's own background color/luminance, sampled independently of
    # any ring/hole candidate - see sample_background_reference's docstring
    # for why this is the actual fix for "methods agreeing on the wrong
    # physical edge", not just another geometric heuristic.
    rect_lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_ref = sample_background_reference(rectified, marker_rect, outer_ring_hint)

    diam_candidates = []
    for i,(name, edge_map) in enumerate(variants):
        all_cands = all_ring_candidates(edge_map)

        # Primary detector: clean closed inner + outer contours.
        cand = inner_boundary_from_candidates(all_cands)

        # Fallback for real reflective rings: allow a partially broken
        # inner boundary, but ONLY around a ring location that
        # independently showed outer + inner circular structure.
        #
        # MULTIPLE candidates per variant, not one: see
        # radial_multi_peak_candidates' docstring for why taking only the
        # single tallest histogram peak per method throws away exactly the
        # information that lets independent methods rescue each other from
        # unrelated artifacts. Every returned peak becomes its own entry in
        # diam_candidates, same as if it were its own detection - the
        # cross-family clustering below decides which one (if any) has
        # genuine multi-method support.
        if cand is None and outer_ring_hint is not None:
            outer_cx, outer_cy, outer_radius = outer_ring_hint
            radial_cands = radial_multi_peak_candidates(
                rect_gray,
                edge_map,
                (outer_cx, outer_cy),
                outer_radius
            )
            for rc in radial_cands:
                (cx,cy),(MA,ma),angle = rc['ellipse']
                diameter_mm = (MA+ma)/2/PX_PER_MM_OUT
                diam_candidates.append((name, diameter_mm, rc))
            continue

        if cand is None:
            continue
        (cx,cy),(MA,ma),angle = cand['ellipse']
        diameter_mm = (MA+ma)/2/PX_PER_MM_OUT
        diam_candidates.append((name, diameter_mm, cand))

    # A DIRECT hole-region detector, independent of every edge/contour
    # method above: segments this photo's own background color within the
    # localized ROI and looks for background-colored islands fully
    # enclosed by non-background material - see enclosed_hole_candidates'
    # docstring. Added as its own family ("holecolor") into the same pool,
    # not a replacement for the others - cross-family agreement still
    # requires >=2 independent methods, and this is one vote among them,
    # not a trusted oracle by itself.
    for hc in enclosed_hole_candidates(rect_lab, marker_rect, outer_ring_hint, bg_ref):
        (cx,cy),(MA,ma),angle = hc['ellipse']
        diameter_mm = (MA+ma)/2/PX_PER_MM_OUT
        diam_candidates.append((hc['name'], diameter_mm, hc))

    # --- THE material-contrast gate: reject candidates before they ever ---
    # --- reach cross-method voting, not after.                          ---
    #
    # This is the actual fix for "low cross-method spread producing false
    # confidence because multiple methods can agree on the same wrong
    # edge": that failure mode is only possible if the outer edge, a
    # bevel, a specular reflection, or a cast shadow is allowed to reach
    # the consensus stage as a candidate at all. validate_material_contrast
    # checks each candidate against THIS PHOTO's own sampled background -
    # never against a known diameter or any other photo's numbers - and
    # candidates that fail are recorded (not silently dropped) in
    # res.rejected_candidates for the debug overlay, then excluded from
    # every step below. Agreement between two methods that both failed
    # this gate is no longer possible, because neither reaches the pool
    # agreement is computed over.
    validated = []
    rejected = []
    for name, diameter_mm, cand in diam_candidates:
        passed, info = validate_material_contrast(rect_lab, cand['ellipse'], bg_ref)
        if passed:
            validated.append((name, diameter_mm, cand))
        else:
            rejected.append((name, diameter_mm, cand, info.get("reason", "MATERIAL_CONTRAST_FAILED")))
            if os.environ.get("RING_DEBUG"):
                print(f"[RING_DEBUG] REJECTED fam={name:<16} diam={diameter_mm:7.2f}mm "
                      f"reason={info.get('reason')} d_in={info.get('d_in')} d_out={info.get('d_out')}")
    diam_candidates = validated
    res.rejected_candidates = rejected

    def draw_debug(final_ellipse=None):
        """
        Built and returned only once the outcome is known, so the overlay
        always matches what was actually decided (not a snapshot from
        mid-computation). Always draws the localization ROI and every
        candidate that survived the material-contrast gate; additionally
        draws REJECTED candidates (dashed, labeled with their rejection
        reason) when RING_DEBUG_OVERLAY is set, since that view is
        verbose/technical and not meant for the default end-user image.
        The final selected boundary (if any) is drawn last, bold, on top.
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
                label = f"{name}: {diameter_mm:.1f}mm REJECTED {reason}"
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
    # Loosened from 2.0mm after adding the "holecolor" family: it locates
    # the hole via connected-component mask boundaries (blocky, pixel-
    # quantized) rather than a sub-pixel contour fit, so its own center
    # estimate is inherently coarser than the edge-based families' - a
    # real photo (real_ring_B) where every family clearly found the SAME
    # physical hole still showed a ~2.5mm holecolor-vs-canny center gap
    # from that precision difference alone, not from finding different
    # objects. 3.0mm keeps real cross-object rejection (see
    # synth_decoy_object_adversarial, whose decoy sits far outside this
    # margin) while not punishing holecolor for being coarser, not wrong.
    CENTER_AGREEMENT_THRESHOLD_MM = 3.0

    tagged = []  # (family, diameter_mm, cx, cy, peak_rank, ellipse)
    for name, diameter_mm, cand in diam_candidates:
        family = name.split("_")[0]   # "canny_20_60" -> "canny"
        cx, cy = cand['center']
        # Contour-based candidates are a single, direct measurement (there
        # is no "2nd-best contour"), so treat them as rank 0 - the same
        # standing as a variant's own strongest radial peak.
        peak_rank = cand.get('peak_rank', 0)
        tagged.append((family, diameter_mm, cx, cy, peak_rank, cand['ellipse']))

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

    # Prefer the cluster with the most distinct families behind it; break
    # ties by whichever is tightest (smallest spread) - both are purely
    # geometric/statistical properties of the candidate set itself, not
    # anything fitted to a known answer.
    def cluster_key(c):
        diams = [x[1] for x in c]
        spread = max(diams) - min(diams)
        return (-len(families_in(c)), spread)

    best_cluster = min(qualifying, key=cluster_key)

    # Per-family representative value WITHIN the winning cluster (median,
    # in case a family contributed more than one peak to it) - this keeps
    # the same "one vote per family" semantics the rest of the app/UI
    # already documents "cross-method spread" as meaning.
    cluster_family_diams = {}
    cluster_family_centers = {}
    for fam, dmm, cx, cy, _peak_rank, _ellipse in best_cluster:
        cluster_family_diams.setdefault(fam, []).append(dmm)
        cluster_family_centers.setdefault(fam, []).append((cx, cy))

    # Representative ellipse for the debug overlay's final highlight only -
    # doesn't affect the reported diameter (still the per-family median
    # below). Prefer a member that was its own method's top pick.
    _top_rank_members = [m for m in best_cluster if m[4] == 0]
    _rep_member = _top_rank_members[0] if _top_rank_members else best_cluster[0]
    winning_ellipse = _rep_member[5]

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
    res.debug_img = draw_debug(final_ellipse=winning_ellipse)
    return res
