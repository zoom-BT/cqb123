"""
loop_builder.py — ordered contour reconstruction
DeepCAD curves within a loop are NOT stored in connected order.
This module chains them into a continuous, ordered contour so that
generators can emit valid closed wires.

Core idea:
  - Each curve has a start and end point (circles are self-closed).
  - We greedily chain curves end-to-start, flipping direction when needed,
    using a spatial tolerance to match endpoints.
"""

from __future__ import annotations
from typing import Optional
import math

from parser import Loop, Curve, CurveLine, CurveCircle, CurveArc, Vec3


TOL = 1e-4  # endpoint matching tolerance in metres (~0.1 mm)


def _pt(v: Vec3) -> tuple[float, float]:
    """2D point in mm."""
    return (v.x * 1000, v.y * 1000)


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _endpoints(curve: Curve) -> Optional[tuple[tuple, tuple]]:
    """
    Return (start, end) points in mm for a curve.
    Circles return None (they are self-closed, handled separately).
    """
    if isinstance(curve, CurveLine):
        return _pt(curve.start), _pt(curve.end)
    if isinstance(curve, CurveArc):
        return _pt(curve.start), _pt(curve.end)
    return None  # circle


class OrderedCurve:
    """A curve with a guaranteed orientation (start → end)."""
    __slots__ = ("kind", "start", "end", "center", "radius", "is_circle")

    def __init__(self, kind, start=None, end=None, center=None,
                 radius=0.0, is_circle=False):
        self.kind      = kind          # "line" | "arc" | "circle"
        self.start     = start         # (x, y) mm
        self.end       = end           # (x, y) mm
        self.center    = center        # (x, y) mm
        self.radius    = radius        # mm
        self.is_circle = is_circle


def _to_ordered(curve: Curve) -> OrderedCurve:
    if isinstance(curve, CurveCircle):
        return OrderedCurve(
            kind="circle",
            center=_pt(curve.center),
            radius=curve.radius_mm,
            is_circle=True,
        )
    if isinstance(curve, CurveLine):
        return OrderedCurve(
            kind="line",
            start=_pt(curve.start),
            end=_pt(curve.end),
        )
    if isinstance(curve, CurveArc):
        return OrderedCurve(
            kind="arc",
            start=_pt(curve.start),
            end=_pt(curve.end),
            center=_pt(curve.center),
            radius=curve.radius_mm,
        )
    raise ValueError(f"Unknown curve type: {type(curve)}")


def order_loop(loop: Loop) -> list[OrderedCurve]:
    """
    Chain a loop's curves into a continuous ordered contour.

    Returns a list of OrderedCurve objects oriented start → end so that
    each curve's end matches the next curve's start.

    Circles are returned as standalone self-closed contours.
    """
    raw = [_to_ordered(c) for c in loop.curves]

    # Separate circles (self-closed) from chainable curves
    circles    = [c for c in raw if c.is_circle]
    chainable  = [c for c in raw if not c.is_circle]

    if not chainable:
        return circles

    # Greedy chaining
    ordered: list[OrderedCurve] = []
    remaining = chainable[:]

    # Start with the first curve
    current = remaining.pop(0)
    ordered.append(current)
    chain_end = current.end

    while remaining:
        best_idx  = None
        best_flip = False
        best_dist = float("inf")

        for i, cand in enumerate(remaining):
            d_start = _dist(chain_end, cand.start)
            d_end   = _dist(chain_end, cand.end)

            if d_start < best_dist:
                best_dist = d_start
                best_idx  = i
                best_flip = False
            if d_end < best_dist:
                best_dist = d_end
                best_idx  = i
                best_flip = True

        if best_idx is None:
            break

        nxt = remaining.pop(best_idx)
        if best_flip:
            # Flip orientation
            nxt.start, nxt.end = nxt.end, nxt.start

        ordered.append(nxt)
        chain_end = nxt.end

    return circles + ordered


def is_closed(ordered: list[OrderedCurve], tol_mm: float = 0.5) -> bool:
    """Check if the chained contour forms a closed loop."""
    chainable = [c for c in ordered if not c.is_circle]
    if not chainable:
        return True  # circles are always closed
    first = chainable[0].start
    last  = chainable[-1].end
    return _dist(first, last) <= tol_mm


def ordered_points(ordered: list[OrderedCurve]) -> list[tuple[float, float]]:
    """
    Return the ordered list of vertex points (start of each segment) in mm,
    for building a polyline. Only valid for line-only contours.
    """
    chainable = [c for c in ordered if not c.is_circle]
    if not chainable:
        return []
    pts = [c.start for c in chainable]
    return pts


def is_polygon(ordered: list[OrderedCurve]) -> bool:
    """True if the contour is made only of straight lines."""
    chainable = [c for c in ordered if not c.is_circle]
    return len(chainable) > 0 and all(c.kind == "line" for c in chainable)


def detect_rectangle(ordered: list[OrderedCurve]
                     ) -> Optional[tuple[float, float, float, float]]:
    """
    Detect an axis-aligned rectangle from an ordered line contour.
    Returns (cx, cy, width, height) in mm, or None.
    """
    if not is_polygon(ordered):
        return None
    pts = ordered_points(ordered)
    if len(pts) != 4:
        return None

    xs = sorted(set(round(p[0], 4) for p in pts))
    ys = sorted(set(round(p[1], 4) for p in pts))

    if len(xs) == 2 and len(ys) == 2:
        w  = xs[1] - xs[0]
        h  = ys[1] - ys[0]
        cx = (xs[0] + xs[1]) / 2
        cy = (ys[0] + ys[1]) / 2
        return (cx, cy, w, h)
    return None


# ── Nesting detection (holes) ─────────────────────────────────────────────────

def contour_centroid(ordered: list[OrderedCurve]) -> tuple[float, float]:
    """Approximate centroid of a contour (mm)."""
    pts = []
    for c in ordered:
        if c.is_circle:
            return c.center
        pts.append(c.start)
    if not pts:
        return (0.0, 0.0)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return (cx, cy)


def contour_area(ordered: list[OrderedCurve]) -> float:
    """Approximate absolute area of a contour (mm^2)."""
    import math
    if len(ordered) == 1 and ordered[0].is_circle:
        return math.pi * ordered[0].radius ** 2
    pts = [c.start for c in ordered if not c.is_circle]
    if len(pts) < 3:
        return 0.0
    # Shoelace formula
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def point_in_contour(pt: tuple, ordered: list[OrderedCurve]) -> bool:
    """Ray-casting point-in-polygon test. For circles, distance check."""
    import math
    if len(ordered) == 1 and ordered[0].is_circle:
        c = ordered[0]
        d = math.hypot(pt[0] - c.center[0], pt[1] - c.center[1])
        return d < c.radius
    pts = [c.start for c in ordered if not c.is_circle]
    if len(pts) < 3:
        return False
    x, y = pt
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def classify_loops(loops_ordered: list[list[OrderedCurve]]) -> list[dict]:
    """
    Given a list of ordered contours from ONE profile, classify each as
    outer (solid) or inner (hole) based on spatial nesting.

    Returns a list of dicts: {"contour": ordered, "is_hole": bool}
    A contour is a hole if its centroid lies inside a larger contour.
    """
    # Sort by area descending — largest first
    indexed = [(i, c, contour_area(c)) for i, c in enumerate(loops_ordered)]
    indexed.sort(key=lambda t: t[2], reverse=True)

    result = [{"contour": c, "is_hole": False} for _, c, _ in indexed]

    for a in range(len(indexed)):
        for b in range(len(indexed)):
            if a == b:
                continue
            # is contour b inside contour a (a larger than b)?
            if indexed[a][2] <= indexed[b][2]:
                continue
            centroid_b = contour_centroid(indexed[b][1])
            if point_in_contour(centroid_b, indexed[a][1]):
                # b is nested inside a → b is a hole
                result[b]["is_hole"] = True

    return result
