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
                    max_dark_fraction=0.5, max_bright_fraction=0.15,
                    min_mean=60, max_mean=235):
    """Reject photos that are too dark or too washed-out to trust before
    spending any time on marker/ring detection. Thresholds calibrated with
    real working photos as the floor/ceiling to stay clear of: both real
    test photos (evenly lit, indoor) measured mean brightness ~137,
    frac_dark(<25) ~1%, frac_bright(>250) ~0.01-0.04% - comfortably inside
    these bounds. Returns a reason string or None.
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


def radial_inner_candidate(gray_for_gradient, edge_map, center, outer_radius_px,
                           min_band_ratio=1.07,
                           max_band_ratio=1.8,
                           min_support=0.03,
                           min_prominence=1.25,
                           gradient_angle_tolerance_deg=20):
    """
    Fallback for real metal rings where reflections/shadows break the
    boundary into arcs and findContours() cannot produce a clean closed
    contour.

    CRITICAL FIX (found on a real photo of a ring on a wood table):
    counting ALL edge pixels in the radius band, regardless of direction,
    fails on textured backgrounds. Wood grain alone produced 300-478 Canny
    edge pixels at EVERY radius from 122-202px in one real test photo - an
    almost flat, noisy histogram with no usable peak (prominence ~1.16,
    needed >1.7 under the old thresholds). A true circular boundary's edge
    pixels have gradients pointing RADIALLY (toward/away from the circle's
    center); wood grain and other texture edges do not - their gradient
    direction is set by the grain/pattern, essentially decorrelated from
    any candidate center. Filtering edge pixels to only those whose local
    gradient direction is within `gradient_angle_tolerance_deg` of radial,
    before building the radius histogram, removes the vast majority of
    texture noise and reveals the true peak (confirmed on the same photo:
    filtered peak landed within ~0.2mm of the ~27mm ground truth, versus a
    wrong-by-8mm peak on the unfiltered histogram). Thresholds below are
    recalibrated for the filtered (much lower absolute count, much
    cleaner) histogram - they are not the same numbers that applied to an
    unfiltered count.

    The candidate must:
    - be plausibly inside the outer ring edge;
    - have enough angular edge support (of radially-oriented edges only);
    - form a clear radial peak rather than ordinary background texture.
    """
    cx, cy = center

    r_min = int(np.ceil(outer_radius_px / max_band_ratio))
    r_max = int(np.floor(outer_radius_px / min_band_ratio))

    if r_max <= r_min + 3:
        return None

    pad = int(np.ceil(outer_radius_px)) + 3

    x0 = max(0, int(cx - pad))
    x1 = min(edge_map.shape[1], int(cx + pad + 1))
    y0 = max(0, int(cy - pad))
    y1 = min(edge_map.shape[0], int(cy + pad + 1))

    roi = edge_map[y0:y1, x0:x1]
    ys, xs = np.nonzero(roi)

    if len(xs) < 20:
        return None

    xs = xs + x0
    ys = ys + y0

    radii = np.hypot(xs - cx, ys - cy)
    in_band = (radii >= r_min) & (radii <= r_max)
    xs, ys, radii = xs[in_band], ys[in_band], radii[in_band]

    if len(radii) < 20:
        return None

    # --- Gradient-direction filter (the actual fix) ---
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
        return None

    rounded = np.rint(radii).astype(np.int32)

    hist = np.bincount(
        rounded,
        minlength=r_max + 2
    )[r_min:r_max + 1]

    if hist.size == 0:
        return None

    peak_idx = int(np.argmax(hist))
    radius = float(r_min + peak_idx)
    peak = float(hist[peak_idx])

    # Approximate fraction of the circumference supported by edge pixels.
    support = peak / max(1.0, 2.0 * np.pi * radius)

    # Reject broad/noisy radial distributions from wood grain/background.
    baseline = float(np.percentile(hist, 75))
    prominence = peak / max(1.0, baseline)

    if support < min_support:
        return None

    if prominence < min_prominence:
        return None

    return {
        "ellipse": (
            (float(cx), float(cy)),
            (2.0 * radius, 2.0 * radius),
            0.0
        ),
        "circularity": 1.0,
        "axis_ratio": 1.0,
        "residual": 0.0,
        "score": support,
        "diam_px": 2.0 * radius,
        "center": (float(cx), float(cy)),
        "radial_support": support,
        "radial_prominence": prominence,
    }


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

    # Real-photo fallback localization.
    #
    # The existing contour-based detector remains the PRIMARY path because
    # it is precise when clean closed contours exist.
    #
    # Hough/radial detection is only used when that primary detector cannot
    # obtain a valid inner+outer contour pair.
    outer_ring_hint = locate_outer_ring_circle(
        rect_gray,
        marker_rect
    )

    diam_candidates = []
    dbg = rectified.copy()
    colors = [(0,255,0),(255,0,0),(0,255,255),(255,0,255),(0,128,255),(255,255,0)]
    for i,(name, edge_map) in enumerate(variants):
        all_cands = all_ring_candidates(edge_map)

        # Primary detector: clean closed inner + outer contours.
        cand = inner_boundary_from_candidates(all_cands)

        # Fallback for real reflective rings: allow a partially broken
        # inner boundary, but ONLY around a ring location that
        # independently showed outer + inner circular structure.
        if cand is None and outer_ring_hint is not None:
            outer_cx, outer_cy, outer_radius = outer_ring_hint
            cand = radial_inner_candidate(
                rect_gray,
                edge_map,
                (outer_cx, outer_cy),
                outer_radius
            )

        if cand is None:
            continue
        (cx,cy),(MA,ma),angle = cand['ellipse']
        diameter_mm = (MA+ma)/2/PX_PER_MM_OUT
        diam_candidates.append((name, diameter_mm, cand))
        # Line thickness scaled to image size: a fixed 1px was fine for our
        # own full-resolution diagnostic viewing, but real phone photos
        # rectify to large canvases (3000px+) that then get downscaled for
        # display in the app - at 1px the annotation became nearly
        # invisible once shrunk to a phone-sized result card. Scale with
        # width so it stays visible after downscaling, whatever the source
        # photo's resolution.
        line_thickness = max(3, dbg.shape[1] // 300)
        cv2.ellipse(dbg, cand['ellipse'], colors[i%len(colors)], line_thickness)

    res.candidates_mm = diam_candidates
    res.debug_img = dbg

    # --- Aggregate by INDEPENDENT METHOD FAMILY, not raw candidate count. ---
    # canny_20_60 .. canny_60_150 are the same underlying detector with only
    # a threshold changed: they are correlated and tend to succeed or fail
    # together (we proved this with the dilate-bias bug - all 4 canny variants
    # were wrong in the same direction at once). Counting them as "4 independent
    # confirmations" overstates confidence. We collapse each family to a single
    # vote (its median) before measuring cross-family agreement.
    families = {}
    for name, diameter_mm, cand in diam_candidates:
        family = name.split("_")[0]   # "canny_20_60" -> "canny"
        cx, cy = cand['center']
        families.setdefault(family, []).append((diameter_mm, cx, cy))

    family_estimates = {}
    family_centers = {}
    for fam, vals in families.items():
        diams = [v[0] for v in vals]
        cxs = [v[1] for v in vals]
        cys = [v[2] for v in vals]
        family_estimates[fam] = float(np.median(diams))
        family_centers[fam] = (float(np.median(cxs)), float(np.median(cys)))

    res.family_estimates = family_estimates

    MIN_FAMILIES = 2   # need at least 2 genuinely different methods to agree
    if len(family_estimates) < MIN_FAMILIES:
        res.reason = "RING_NOT_FOUND"
        return res

    # --- Spatial consensus: families must agree on WHERE, not just diameter. ---
    # Comparing diameters alone can't tell "two methods found the same hole"
    # from "two methods found two different circular things in the scene"
    # (a reflection here, a background pattern there) that happen to be
    # similar sizes. Require the family centroids to be close together, in
    # the same physical units (mm) as everything else we report.
    centers_mm = {fam: (x/PX_PER_MM_OUT, y/PX_PER_MM_OUT) for fam,(x,y) in family_centers.items()}
    fams = list(centers_mm.keys())
    max_center_dist_mm = 0.0
    for i in range(len(fams)):
        for j in range(i+1, len(fams)):
            (x1,y1), (x2,y2) = centers_mm[fams[i]], centers_mm[fams[j]]
            dist = np.hypot(x1-x2, y1-y2)
            max_center_dist_mm = max(max_center_dist_mm, dist)

    CENTER_AGREEMENT_THRESHOLD_MM = 2.0
    if max_center_dist_mm > CENTER_AGREEMENT_THRESHOLD_MM:
        # families found geometrically DIFFERENT objects, not the same hole
        # measured slightly differently - this is a distinct failure mode
        # from "same object, disagreeing size" and deserves its own code.
        res.reason = "INCONSISTENT_DETECTION"
        return res

    fam_values = np.array(list(family_estimates.values()))
    spread = fam_values.max() - fam_values.min()
    median = float(np.median(fam_values))

    if spread > 1.0:   # mm -- families disagree too much, unreliable
        res.reason = "RING_EDGE_UNSTABLE"
        # Deliberately NOT setting res.diameter_mm here even though we have
        # a median value: res.ok is False, and a number sitting on a
        # "not ok" result is a landmine for any future code path that reads
        # diameter_mm without checking ok first. detection_spread_mm is
        # still useful to keep for diagnostics (it's WHY this was rejected).
        res.detection_spread_mm = spread
        return res

    res.ok = True
    res.diameter_mm = median
    res.detection_spread_mm = max(spread, 0.05)  # floor so we never claim impossible precision
    return res
