"""
cadquery_gen.py — CadQuery code generator
Converts a parsed CADModel into an executable CadQuery Python script.

Coverage (V1):
  - Line3D   → polyline / rect detection
  - Circle3D → circle
  - Planes   → XY, XZ, XZ_neg, YZ, YZ_neg, XY_neg + custom fallback
  - Operations → new_body, cut, join
  - Extent   → one_side, two_sides
  - Orphan sketches → commented out
  - Multi-profile extrudes → face selector loop
"""

from __future__ import annotations
from typing import Optional
import textwrap

from parser import (
    CADModel, CADStep, Sketch, Extrude,
    Profile, Loop, Curve,
    CurveLine, CurveCircle, CurveArc,
    Workplane, Vec3,
)


# ── Plane helpers ─────────────────────────────────────────────────────────────

# CadQuery standard plane names
_CQ_PLANE_MAP = {
    "XY"     : '"XY"',
    "XY_neg" : '"XY"',   # will flip via origin offset
    "XZ"     : '"XZ"',
    "XZ_neg" : '"XZ"',
    "YZ"     : '"YZ"',
    "YZ_neg" : '"YZ"',
}

def _cq_plane(wp: Workplane) -> str:
    """
    Return the CadQuery Workplane plane argument string.
    For standard planes returns a string literal ("XY", "XZ", "YZ").
    For custom planes builds a cq.Plane(...) expression.
    """
    name = wp.plane_name

    if name and not wp.has_nonzero_origin:
        return _CQ_PLANE_MAP[name]

    # Custom plane or non-zero origin → build explicit cq.Plane
    o  = wp.origin
    x  = wp.x_axis
    z  = wp.z_axis
    return (
        f"cq.Plane("
        f"origin=({o.x*1000:.4f}, {o.y*1000:.4f}, {o.z*1000:.4f}), "
        f"xDir=({x.x:.4f}, {x.y:.4f}, {x.z:.4f}), "
        f"normal=({z.x:.4f}, {z.y:.4f}, {z.z:.4f}))"
    )


# ── Curve helpers ─────────────────────────────────────────────────────────────

def _is_rect(curves: list[Curve]) -> Optional[tuple[float, float, float, float]]:
    """
    Detect if 4 Line3D curves form an axis-aligned rectangle.
    Returns (x_min, y_min, width, height) in mm, or None.
    """
    if len(curves) != 4:
        return None
    if not all(isinstance(c, CurveLine) for c in curves):
        return None

    xs = set()
    ys = set()
    for c in curves:
        xs.update([round(c.start.x * 1000, 6), round(c.end.x * 1000, 6)])
        ys.update([round(c.start.y * 1000, 6), round(c.end.y * 1000, 6)])

    if len(xs) == 2 and len(ys) == 2:
        x_vals = sorted(xs)
        y_vals = sorted(ys)
        w = x_vals[1] - x_vals[0]
        h = y_vals[1] - y_vals[0]
        cx = (x_vals[0] + x_vals[1]) / 2
        cy = (y_vals[0] + y_vals[1]) / 2
        return (cx, cy, w, h)
    return None


def _gen_loop_cq(loop: Loop, indent: str = "    ") -> list[str]:
    """
    Generate CadQuery sketch lines for a single loop.
    Returns a list of code lines (without trailing newline).
    """
    lines = []
    curves = loop.curves

    if not curves:
        return lines

    # ── Circle ──────────────────────────────────────────────────────────────
    if len(curves) == 1 and isinstance(curves[0], CurveCircle):
        c = curves[0]
        cx = c.center.x * 1000
        cy = c.center.y * 1000
        if abs(cx) > 1e-6 or abs(cy) > 1e-6:
            lines.append(f"{indent}.transformed(offset=cq.Vector({cx:.4f}, {cy:.4f}, 0))")
        lines.append(f"{indent}.circle({c.radius_mm:.4f})")
        return lines

    # ── Axis-aligned rectangle ───────────────────────────────────────────────
    rect = _is_rect(curves)
    if rect:
        cx, cy, w, h = rect
        if abs(cx) > 1e-6 or abs(cy) > 1e-6:
            lines.append(f"{indent}.transformed(offset=cq.Vector({cx:.4f}, {cy:.4f}, 0))")
        lines.append(f"{indent}.rect({w:.4f}, {h:.4f})")
        return lines

    # ── Generic polyline ─────────────────────────────────────────────────────
    # Collect ordered points from Line3D segments
    if all(isinstance(c, CurveLine) for c in curves):
        pts = [(c.start.x * 1000, c.start.y * 1000) for c in curves]
        pts_str = ", ".join(f"({x:.4f}, {y:.4f})" for x, y in pts)
        lines.append(f"{indent}.polyline([{pts_str}]).close()")
        return lines

    # ── Arc3D ────────────────────────────────────────────────────────────────
    if len(curves) == 1 and isinstance(curves[0], CurveArc):
        arc = curves[0]
        sx, sy = arc.start.x * 1000, arc.start.y * 1000
        ex, ey = arc.end.x * 1000,   arc.end.y * 1000
        mx = (arc.center.x * 1000 + arc.radius_mm *
              ((sx - arc.center.x * 1000) / max(arc.radius_mm, 1e-9)))
        my = (arc.center.y * 1000 + arc.radius_mm *
              ((sy - arc.center.y * 1000) / max(arc.radius_mm, 1e-9)))
        lines.append(
            f"{indent}.threePointArc(({sx:.4f}, {sy:.4f}), "
            f"({mx:.4f}, {my:.4f}), ({ex:.4f}, {ey:.4f}))"
        )
        return lines

    # ── Mixed curves fallback ────────────────────────────────────────────────
    for curve in curves:
        if isinstance(curve, CurveLine):
            sx, sy = curve.start.x * 1000, curve.start.y * 1000
            ex, ey = curve.end.x   * 1000, curve.end.y   * 1000
            lines.append(f"{indent}.moveTo({sx:.4f}, {sy:.4f})")
            lines.append(f"{indent}.lineTo({ex:.4f}, {ey:.4f})")
        elif isinstance(curve, CurveArc):
            sx, sy = curve.start.x * 1000, curve.start.y * 1000
            ex, ey = curve.end.x   * 1000, curve.end.y   * 1000
            cx, cy = curve.center.x * 1000, curve.center.y * 1000
            lines.append(
                f"{indent}.threePointArc(({sx:.4f}, {sy:.4f}), "
                f"({cx:.4f}, {cy:.4f}), ({ex:.4f}, {ey:.4f}))"
            )
    if lines:
        lines.append(f"{indent}.close()")

    return lines


# ── Operation helpers ─────────────────────────────────────────────────────────

_CQ_OP_MAP = {
    "new_body" : None,          # default — no combine kwarg needed
    "join"     : "combine=True",
    "cut"      : "combine='cut'",
    "intersect": "combine='intersect'",
}


# ── Main generator ────────────────────────────────────────────────────────────

def generate(model: CADModel) -> str:
    """
    Generate a CadQuery Python script from a CADModel.

    Args:
        model: parsed CADModel from parser.parse()

    Returns:
        A string containing the complete executable CadQuery script.
    """
    code: list[str] = []

    # Header
    code.append('"""')
    code.append(f'CadQuery script generated from {model.source_file}')
    code.append('Generated by CQB123 — github.com/zoom-BT/cqb123')
    code.append('"""')
    code.append("import cadquery as cq")
    code.append("")

    # Build a sketch registry: sketch_id → variable name
    sketch_vars: dict[str, str] = {}
    sketch_counter = 0
    result_var = "result"

    # First pass — assign variable names to sketches
    for step in model.steps:
        if step.step_type == "sketch":
            sketch_counter += 1
            vname = f"sketch_{sketch_counter}"
            sketch_vars[step.entity.entity_id] = vname

    # Track whether we have a solid yet
    has_solid = False

    # Second pass — generate code
    for step in model.steps:

        # ── Sketch ──────────────────────────────────────────────────────────
        if step.step_type == "sketch":
            sketch: Sketch = step.entity
            vname = sketch_vars[sketch.entity_id]
            plane_str = _cq_plane(sketch.workplane)

            if sketch.is_orphan:
                code.append(f"# Orphan sketch '{sketch.name}' — not extruded")
                code.append(f"# {vname} = (")
                code.append(f'#     cq.Workplane({plane_str})')
                for profile in sketch.profiles.values():
                    for loop in profile.loops:
                        for line in _gen_loop_cq(loop):
                            code.append(f"#  {line.strip()}")
                code.append("# )")
                code.append("")
                continue

            code.append(f"{vname} = (")
            code.append(f"    cq.Workplane({plane_str})")

            for profile in sketch.profiles.values():
                for loop in profile.loops:
                    loop_lines = _gen_loop_cq(loop)
                    code.extend(loop_lines)

            code.append(")")
            code.append("")

        # ── Extrude ──────────────────────────────────────────────────────────
        elif step.step_type == "extrude":
            extrude: Extrude = step.entity
            op = extrude.operation
            combine_kwarg = _CQ_OP_MAP.get(op)

            # Collect unique referenced sketch ids in this extrude
            ref_sketch_ids = list(dict.fromkeys(
                r.sketch_id for r in extrude.profile_refs
            ))

            # Build extrude call
            dist     = extrude.distance_mm
            dist_two = extrude.distance_two_mm

            if extrude.extent_type == "two_sides":
                extrude_call = (
                    f".extrude({dist:.4f}, both=True)"
                )
            else:
                extrude_call = f".extrude({dist:.4f})"

            if combine_kwarg:
                extrude_call = extrude_call.replace(")", f", {combine_kwarg})")

            for i, sketch_id in enumerate(ref_sketch_ids):
                src_var = sketch_vars.get(sketch_id)
                if src_var is None:
                    continue

                if not has_solid and op == "new_body":
                    # First solid — assign to result
                    code.append(f"{result_var} = (")
                    code.append(f"    {src_var}")
                    code.append(f"    {extrude_call}")
                    code.append(")")
                    has_solid = True
                else:
                    # Subsequent operations — union/cut/join onto result
                    code.append(f"{result_var} = (")
                    code.append(f"    {result_var}")
                    code.append(f"    .union({src_var}{extrude_call})"
                                 if op == "join" else
                                 f"    .cut({src_var}{extrude_call})"
                                 if op == "cut" else
                                 f"    .union({src_var}{extrude_call})")
                    code.append(")")

                code.append("")

    # Footer — show result
    code.append("# Display")
    if has_solid:
        code.append(f"show_object({result_var})")
    else:
        code.append("# No solid generated")

    return "\n".join(code)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from parser import parse

    if len(sys.argv) < 2:
        print("Usage: python cadquery_gen.py <path_to_json>")
        sys.exit(1)

    model = parse(sys.argv[1])
    print(generate(model))
