"""
build123d_gen.py — Build123d code generator
Converts a parsed CADModel into an executable Build123d Python script.

Coverage (V1):
  - Line3D   → polyline / Rectangle detection
  - Circle3D → Circle
  - Planes   → XY, XZ, XZ_neg, YZ, YZ_neg, XY_neg + custom fallback
  - Operations → new_body, cut, join
  - Extent   → one_side, two_sides
  - Orphan sketches → commented out
  - Multi-profile extrudes → multiple shapes in same BuildSketch context
"""

from __future__ import annotations
from typing import Optional

from parser import (
    CADModel, Sketch, Extrude,
    Profile, Loop, Curve,
    CurveLine, CurveCircle, CurveArc,
    Workplane, Vec3,
)


# ── Plane helpers ─────────────────────────────────────────────────────────────

# Build123d standard Plane names
_B3D_PLANE_MAP = {
    "XY"     : "Plane.XY",
    "XY_neg" : "Plane.XY.offset(-0)",   # handled via origin
    "XZ"     : "Plane.XZ",
    "XZ_neg" : "Plane.XZ",
    "YZ"     : "Plane.YZ",
    "YZ_neg" : "Plane.YZ",
}


def _b3d_plane(wp: Workplane) -> str:
    """
    Return the Build123d Plane expression for a workplane.
    Standard cardinal planes → Plane.XY / Plane.XZ / Plane.YZ
    Custom planes → Plane(origin=..., z_dir=...)
    """
    name = wp.plane_name

    if name and not wp.has_nonzero_origin:
        return _B3D_PLANE_MAP[name]

    # Custom plane or non-zero origin
    o = wp.origin
    z = wp.z_axis
    return (
        f"Plane("
        f"origin=({o.x*1000:.4f}, {o.y*1000:.4f}, {o.z*1000:.4f}), "
        f"z_dir=({z.x:.4f}, {z.y:.4f}, {z.z:.4f}))"
    )


# ── Curve helpers ─────────────────────────────────────────────────────────────

def _is_rect(curves: list[Curve]) -> Optional[tuple[float, float, float, float]]:
    """
    Detect axis-aligned rectangle from 4 Line3D segments.
    Returns (cx, cy, width, height) in mm, or None.
    """
    if len(curves) != 4:
        return None
    if not all(isinstance(c, CurveLine) for c in curves):
        return None

    xs, ys = set(), set()
    for c in curves:
        xs.update([round(c.start.x * 1000, 6), round(c.end.x * 1000, 6)])
        ys.update([round(c.start.y * 1000, 6), round(c.end.y * 1000, 6)])

    if len(xs) == 2 and len(ys) == 2:
        x_vals = sorted(xs)
        y_vals = sorted(ys)
        w  = x_vals[1] - x_vals[0]
        h  = y_vals[1] - y_vals[0]
        cx = (x_vals[0] + x_vals[1]) / 2
        cy = (y_vals[0] + y_vals[1]) / 2
        return (cx, cy, w, h)
    return None


def _gen_loop_b3d(loop: Loop, indent: str = "        ") -> list[str]:
    """
    Generate Build123d sketch lines for a single loop.
    Returns a list of code lines ready to be indented inside a BuildSketch block.
    """
    lines = []
    curves = loop.curves

    if not curves:
        return lines

    # ── Circle ───────────────────────────────────────────────────────────────
    if len(curves) == 1 and isinstance(curves[0], CurveCircle):
        c = curves[0]
        cx = c.center.x * 1000
        cy = c.center.y * 1000
        if abs(cx) > 1e-6 or abs(cy) > 1e-6:
            lines.append(f"{indent}with Locations(({cx:.4f}, {cy:.4f})):")
            lines.append(f"{indent}    Circle({c.radius_mm:.4f})")
        else:
            lines.append(f"{indent}Circle({c.radius_mm:.4f})")
        return lines

    # ── Axis-aligned rectangle ────────────────────────────────────────────────
    rect = _is_rect(curves)
    if rect:
        cx, cy, w, h = rect
        if abs(cx) > 1e-6 or abs(cy) > 1e-6:
            lines.append(f"{indent}with Locations(({cx:.4f}, {cy:.4f})):")
            lines.append(f"{indent}    Rectangle({w:.4f}, {h:.4f})")
        else:
            lines.append(f"{indent}Rectangle({w:.4f}, {h:.4f})")
        return lines

    # ── Generic polyline (Line3D segments) ────────────────────────────────────
    if all(isinstance(c, CurveLine) for c in curves):
        pts = [(c.start.x * 1000, c.start.y * 1000) for c in curves]
        pts_str = ", ".join(f"({x:.4f}, {y:.4f})" for x, y in pts)
        lines.append(f"{indent}Polyline([{pts_str}], close=True)")
        return lines

    # ── Arc3D ─────────────────────────────────────────────────────────────────
    if len(curves) == 1 and isinstance(curves[0], CurveArc):
        arc = curves[0]
        sx, sy = arc.start.x  * 1000, arc.start.y  * 1000
        ex, ey = arc.end.x    * 1000, arc.end.y    * 1000
        cx, cy = arc.center.x * 1000, arc.center.y * 1000
        lines.append(
            f"{indent}ThreePointArc("
            f"({sx:.4f}, {sy:.4f}), "
            f"({cx:.4f}, {cy:.4f}), "
            f"({ex:.4f}, {ey:.4f}))"
        )
        return lines

    # ── Mixed curves fallback ─────────────────────────────────────────────────
    for curve in curves:
        if isinstance(curve, CurveLine):
            sx, sy = curve.start.x * 1000, curve.start.y * 1000
            ex, ey = curve.end.x   * 1000, curve.end.y   * 1000
            lines.append(f"{indent}Line(({sx:.4f}, {sy:.4f}), ({ex:.4f}, {ey:.4f}))")
        elif isinstance(curve, CurveArc):
            sx, sy = curve.start.x  * 1000, curve.start.y  * 1000
            ex, ey = curve.end.x    * 1000, curve.end.y    * 1000
            cx, cy = curve.center.x * 1000, curve.center.y * 1000
            lines.append(
                f"{indent}ThreePointArc("
                f"({sx:.4f}, {sy:.4f}), "
                f"({cx:.4f}, {cy:.4f}), "
                f"({ex:.4f}, {ey:.4f}))"
            )

    return lines


# ── Operation helpers ─────────────────────────────────────────────────────────

_B3D_MODE_MAP = {
    "new_body" : "Mode.ADD",
    "join"     : "Mode.ADD",
    "cut"      : "Mode.SUBTRACT",
    "intersect": "Mode.INTERSECT",
}


# ── Main generator ────────────────────────────────────────────────────────────

def generate(model: CADModel) -> str:
    """
    Generate a Build123d Python script from a CADModel.

    Args:
        model: parsed CADModel from parser.parse()

    Returns:
        A string containing the complete executable Build123d script.
    """
    code: list[str] = []

    # Header
    code.append('"""')
    code.append(f'Build123d script generated from {model.source_file}')
    code.append('Generated by CQB123 — github.com/zoom-BT/cqb123')
    code.append('"""')
    code.append("from build123d import *")
    code.append("")

    # Map sketch entity_id → variable name
    sketch_vars: dict[str, str] = {}
    sketch_counter = 0
    for step in model.steps:
        if step.step_type == "sketch":
            sketch_counter += 1
            sketch_vars[step.entity.entity_id] = f"sketch_{sketch_counter}"

    has_solid = False

    for step in model.steps:

        # ── Sketch ───────────────────────────────────────────────────────────
        if step.step_type == "sketch":
            sketch: Sketch = step.entity
            vname     = sketch_vars[sketch.entity_id]
            plane_str = _b3d_plane(sketch.workplane)

            if sketch.is_orphan:
                code.append(f"# Orphan sketch '{sketch.name}' — not extruded")
                code.append(f"# with BuildSketch({plane_str}) as {vname}:")
                for profile in sketch.profiles.values():
                    for loop in profile.loops:
                        for line in _gen_loop_b3d(loop):
                            code.append(f"#  {line.strip()}")
                code.append("")
                continue

            code.append(f"with BuildSketch({plane_str}) as {vname}:")

            for profile in sketch.profiles.values():
                for loop in profile.loops:
                    loop_lines = _gen_loop_b3d(loop)
                    code.extend(loop_lines)

            code.append("")

        # ── Extrude ──────────────────────────────────────────────────────────
        elif step.step_type == "extrude":
            extrude: Extrude = step.entity
            op   = extrude.operation
            mode = _B3D_MODE_MAP.get(op, "Mode.ADD")
            dist = extrude.distance_mm

            # Collect unique referenced sketch variables
            ref_sketch_ids = list(dict.fromkeys(
                r.sketch_id for r in extrude.profile_refs
            ))

            if not has_solid and op == "new_body":
                # First solid — open the BuildPart context
                code.append("with BuildPart() as part:")
                has_solid = True

            for sketch_id in ref_sketch_ids:
                src_var = sketch_vars.get(sketch_id)
                if src_var is None:
                    continue

                # Add sketch into the part context
                code.append(f"    add({src_var}.sketch)")

            # Extrude inside the part context
            if extrude.extent_type == "two_sides":
                code.append(
                    f"    extrude(amount={dist:.4f}, "
                    f"both=True, mode={mode})"
                )
            else:
                code.append(
                    f"    extrude(amount={dist:.4f}, mode={mode})"
                )

            code.append("")

    # Footer
    code.append("# Display")
    if has_solid:
        code.append("show_object(part.part)")
    else:
        code.append("# No solid generated")

    return "\n".join(code)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from parser import parse

    if len(sys.argv) < 2:
        print("Usage: python build123d_gen.py <path_to_json>")
        sys.exit(1)

    model = parse(sys.argv[1])
    print(generate(model))
