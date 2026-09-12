"""
US ring size lookup from an internal diameter measurement.

STANDARD USED: US/Canada ring size scale, linear in inside circumference.
    inside_circumference_mm = 36.5 + 2.55 * us_size
    diameter_mm = inside_circumference_mm / pi

SOURCE: https://measureringsize.com/ring-size-chart - "The scale is linear:
each full size adds about 2.55 mm of circumference. ... US size =
(circumference - 36.5) / 2.55". Cross-checked against two independent
published anchor points before use:
    - https://www.25karats.com (25karats.com ring size chart): "A ring
      measuring 16.5 mm inside is a US 6."      -> formula gives 16.49mm. OK
    - https://www.angara.com (Angara ring size guide): "A US size 7 ring
      measures 17.3 mm in inside diameter"        -> formula gives 17.30mm. OK

RANGE: US sizes 3 to 13.5 (standard adult range; half sizes supported).

ROUNDING RULE: convert measured diameter to an exact (fractional) US size,
then round to the nearest half-size (0.5 increments, the finest granularity
commonly sold). If the exact size falls within the middle 30% of the gap
between two half-sizes (i.e. more than 0.175 away from the nearer
half-size, out of the 0.25 max possible), report it as "between X and Y"
rather than forcing a single answer - a genuine measurement near a size
boundary should not be silently rounded into false precision.
"""
import math

CIRCUMFERENCE_INTERCEPT_MM = 36.5
CIRCUMFERENCE_PER_SIZE_MM = 2.55
MIN_SIZE = 3.0
MAX_SIZE = 13.5
BETWEEN_THRESHOLD = 0.175  # fraction of a half-size step; see docstring


def diameter_to_exact_size(diameter_mm):
    """Inverse of the standard formula: diameter -> fractional US size."""
    circumference_mm = diameter_mm * math.pi
    return (circumference_mm - CIRCUMFERENCE_INTERCEPT_MM) / CIRCUMFERENCE_PER_SIZE_MM


def lookup_ring_size(diameter_mm):
    """
    Returns a dict describing the ring size result:
      {
        "exact_size": 7.18,
        "display": "7" | "7.5" | "between 7 and 7.5",
        "in_range": True,
        "standard": "US/Canada (linear circumference scale)",
        "source": "https://measureringsize.com/ring-size-chart",
        "rounding_rule": "<human-readable string>"
      }
    """
    exact = diameter_to_exact_size(diameter_mm)

    rounding_rule = (
        "Rounded to nearest half US size; reported as \"between X and Y\" "
        "when the measurement falls in the middle 30% of the gap between "
        "two half-sizes."
    )
    standard = "US/Canada ring size (linear inside-circumference scale)"
    source = "https://measureringsize.com/ring-size-chart"

    in_range = MIN_SIZE <= exact <= MAX_SIZE

    nearest_half = round(exact * 2) / 2
    distance = abs(exact - nearest_half)  # 0..0.25

    if distance > BETWEEN_THRESHOLD:
        lower = math.floor(exact * 2) / 2
        upper = lower + 0.5
        display = f"between {fmt_size(lower)} and {fmt_size(upper)}"
    else:
        display = fmt_size(nearest_half)

    return {
        "exact_size": round(exact, 2),
        "display": display,
        "in_range": in_range,
        "standard": standard,
        "source": source,
        "rounding_rule": rounding_rule,
    }


def fmt_size(size):
    """7.0 -> '7', 7.5 -> '7.5'"""
    if size == int(size):
        return str(int(size))
    return str(size)
