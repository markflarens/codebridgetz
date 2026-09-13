"""
Generates a printable A4 calibration marker page directly in physical units
via ReportLab (mm), avoiding any browser/CSS unit-conversion step.

Why not HTML/wkhtmltopdf: an earlier version rendered via wkhtmltopdf and,
when the *rendered PDF* was actually measured (not just the HTML source),
the 20mm marker printed at 15.37mm - a ~23% scale error from wkhtmltopdf's
CSS-mm -> PDF-point handling. For a tool whose entire measurement depends
on "this printed square is exactly 20mm", ReportLab draws directly in mm
with no browser/DPI layer in between. verify_marker_pdf.py independently
re-measures the OUTPUT afterwards - never trust the generator alone.

Two bugs fixed after visual review of the first ReportLab attempt:
  1. Coordinate direction: ReportLab's origin is bottom-left with y
     increasing upward, which is the opposite of how the layout was
     authored (top-down). All placements now go through y_from_top(),
     a single helper, instead of ad hoc page_h-minus-X arithmetic
     scattered through the code (which had put instructions at the
     bottom of the page and the marker near the top).
  2. Cyrillic text: the default Helvetica font has no Cyrillic glyphs -
     instruction text rendered as solid black boxes. Fixed by embedding
     DejaVuSans (supports Cyrillic) via pdfmetrics. All instruction text
     below is now English-only (production copy requirement) - DejaVuSans
     is kept anyway since it's already verified working and dropping it
     for Helvetica would be an unrelated, unverified change.

Run: python3 build_marker.py
Output: ring_marker_20mm.pdf
Verification: run verify_marker_pdf.py afterwards - do not trust this
              script's intent, trust the measured output.
"""
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER_ID = 0
BORDER_BITS = 1
MARKER_MM = 20.0

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def extract_marker_bits():
    grid = DICT.markerSize + 2*BORDER_BITS
    res = 600
    marker_img = cv2.aruco.generateImageMarker(DICT, MARKER_ID, res, borderBits=BORDER_BITS)
    cell = res / grid
    bits = np.zeros((grid, grid), dtype=int)
    for r in range(grid):
        for c in range(grid):
            cy, cx = int((r+0.5)*cell), int((c+0.5)*cell)
            bits[r, c] = 0 if marker_img[cy, cx] < 128 else 1
    return bits

def wrap_text(text, font_name, font_size, max_width_pt):
    """Word-wrap by MEASURED string width, not by eyeballing line lengths.
    Manually authoring line breaks to "look about right" is exactly the
    kind of thing that quietly breaks the moment font/size/margin changes -
    measuring is cheap and removes the guesswork."""
    words = text.split(" ")
    lines, current = [], ""
    for w in words:
        trial = (current + " " + w).strip()
        if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width_pt:
            current = trial
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines

def main():
    pdfmetrics.registerFont(TTFont("DejaVuSans", FONT_REGULAR))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", FONT_BOLD))

    bits = extract_marker_bits()
    grid = bits.shape[0]
    module_mm = MARKER_MM / grid

    page_w_pt, page_h_pt = A4
    page_h_mm = page_h_pt / mm
    LEFT_MARGIN_MM = 15
    RIGHT_MARGIN_MM = 15
    max_text_width_pt = page_w_pt - (LEFT_MARGIN_MM + RIGHT_MARGIN_MM) * mm

    def y_from_top(top_mm):
        """Single source of truth for vertical placement: give a distance
        from the TOP of the page in mm, get back a ReportLab y coordinate
        (points, origin bottom-left). Every placement below goes through
        this - no other place in the file computes y by hand."""
        return (page_h_mm - top_mm) * mm

    c = canvas.Canvas("ring_marker_20mm.pdf", pagesize=A4)

    # --- Instructions block, top of page ---
    c.setFont("DejaVuSans-Bold", 14)
    c.drawString(15*mm, y_from_top(15), "Ring Measurement — Calibration Marker (20 mm)")

    body_font, body_size = "DejaVuSans", 9.5
    c.setFont(body_font, body_size)
    raw_lines = [
        "Step 1. Print this page at 100% / Actual size. Turn OFF \"Fit to page\" / \"Scale to fit\".",
        "Step 2. Check the SCALE CHECK block below with an ordinary ruler: the square must measure exactly 20x20 mm, and the ruler must measure exactly 50.0 mm. If not, reprint with the correct scale. Do NOT photograph the ring until this is confirmed.",
        "Step 3. Cut out the ArUco marker along the gray dashed line (this leaves a white border around it).",
        "Step 4. Place the marker NEXT TO the ring (as close as possible), on the same flat surface, in the same plane. The closer the marker is to the ring in frame, the less the camera lens' own distortion affects the result.",
        "Step 5. Photograph from directly above (near-overhead): marker and ring both clearly visible, without heavy blur or camera tilt.",
    ]
    top = 24
    line_height = 5.2
    for raw in raw_lines:
        for line in wrap_text(raw, body_font, body_size, max_text_width_pt):
            c.drawString(15*mm, y_from_top(top), line)
            top += line_height
        top += 1.5  # small gap between steps

    c.setFillColorRGB(0.8, 0, 0)
    c.setFont("DejaVuSans-Bold", 9.5)
    for line in wrap_text("A 2% print scale error means roughly a 2% error in the measured ring size.",
                          "DejaVuSans-Bold", 9.5, max_text_width_pt):
        c.drawString(15*mm, y_from_top(top+2), line)
        top += line_height
    c.setFillColorRGB(0, 0, 0)

    # --- ArUco marker: pure black/white, nothing else drawn on/over it ---
    marker_top = top + 12       # comfortably below the instructions block
    marker_x0 = 15*mm

    for r in range(grid):
        for col in range(grid):
            if bits[r, col] == 0:
                x = marker_x0 + col*module_mm*mm
                y = y_from_top(marker_top + (r+1)*module_mm)  # +1 row -> bottom edge of this module
                c.rect(x, y, module_mm*mm, module_mm*mm, fill=1, stroke=0)

    # dashed cut guide around the marker (clear of the marker's own bits)
    cut_margin = 8
    c.setStrokeColorRGB(0.6, 0.6, 0.6)
    c.setDash(2, 2)
    c.setLineWidth(0.4)
    c.rect(marker_x0 - cut_margin*mm, y_from_top(marker_top + MARKER_MM + cut_margin),
           (MARKER_MM + 2*cut_margin)*mm, (MARKER_MM + 2*cut_margin)*mm, fill=0, stroke=1)
    c.setDash()
    c.setStrokeColorRGB(0, 0, 0)

    # --- SCALE CHECK block: fully separate area of the page, own square + ruler ---
    sc_top = marker_top + MARKER_MM + cut_margin*2 + 15
    c.setFont("DejaVuSans-Bold", 10)
    c.drawString(15*mm, y_from_top(sc_top), "SCALE CHECK (verify with a ruler before photographing the ring):")

    sq_top = sc_top + 8
    sq_x0 = 15*mm
    c.setLineWidth(0.4)
    c.rect(sq_x0, y_from_top(sq_top + 20), 20*mm, 20*mm, fill=0, stroke=1)
    c.setFont("DejaVuSans", 7)
    c.drawString(sq_x0, y_from_top(sq_top + 24), "20.0 x 20.0 mm")

    ruler_x0 = 60*mm
    ruler_top = sq_top + 10
    ruler_y = y_from_top(ruler_top)
    c.line(ruler_x0, ruler_y, ruler_x0 + 50*mm, ruler_y)
    for m in range(0, 51, 10):
        c.line(ruler_x0 + m*mm, ruler_y - 1.5*mm, ruler_x0 + m*mm, ruler_y + 1.5*mm)
        c.drawCentredString(ruler_x0 + m*mm, y_from_top(ruler_top + 6), str(m))
    c.drawString(ruler_x0, y_from_top(ruler_top - 4), "50.0 mm")

    c.showPage()
    c.save()
    print("Wrote ring_marker_20mm.pdf (ReportLab, direct mm units, Cyrillic-capable font)")

if __name__ == "__main__":
    main()
