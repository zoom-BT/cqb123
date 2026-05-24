"""
parser.py — DeepCAD JSON parser
Reads a DeepCAD JSON file and returns a normalized CADModel structure
ready to be consumed by cadquery_gen.py and build123d_gen.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import math


# ── Primitive curves ────────────────────────────────────────────────────────

@dataclass
class Vec3:
    x: float
    y: float
    z: float

    @classmethod
    def from_dict(cls, d: dict) -> "Vec3":
        return cls(
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            z=float(d.get("z", 0.0)),
        )

    def to_mm(self) -> "Vec3":
        return Vec3(self.x * 1000, self.y * 1000, self.z * 1000)

    def is_zero(self) -> bool:
        return abs(self.x) < 1e-9 and abs(self.y) < 1e-9 and abs(self.z) < 1e-9

    def rounded(self, decimals: int = 4) -> tuple:
        return (
            round(self.x, decimals),
            round(self.y, decimals),
            round(self.z, decimals),
        )


@dataclass
class CurveLine:
    type: str = "Line3D"
    start: Vec3 = None
    end: Vec3 = None


@dataclass
class CurveCircle:
    type: str = "Circle3D"
    center: Vec3 = None
    radius_m: float = 0.0
    normal: Vec3 = None

    @property
    def radius_mm(self) -> float:
        return self.radius_m * 1000


@dataclass
class CurveArc:
    type: str = "Arc3D"
    start: Vec3 = None
    end: Vec3 = None
    center: Vec3 = None
    radius_m: float = 0.0

    @property
    def radius_mm(self) -> float:
        return self.radius_m * 1000


Curve = CurveLine | CurveCircle | CurveArc


# ── Profile and loop ─────────────────────────────────────────────────────────

@dataclass
class Loop:
    is_outer: bool
    curves: list[Curve] = field(default_factory=list)


@dataclass
class Profile:
    profile_id: str
    loops: list[Loop] = field(default_factory=list)

    @property
    def outer_loop(self) -> Optional[Loop]:
        for loop in self.loops:
            if loop.is_outer:
                return loop
        return self.loops[0] if self.loops else None


# ── Workplane ────────────────────────────────────────────────────────────────

STANDARD_PLANES = {
    (0.0, 0.0, 1.0): "XY",
    (0.0, 0.0, -1.0): "XY_neg",
    (0.0, 1.0, 0.0): "XZ",
    (0.0, -1.0, 0.0): "XZ_neg",
    (1.0, 0.0, 0.0): "YZ",
    (-1.0, 0.0, 0.0): "YZ_neg",
}


@dataclass
class Workplane:
    origin: Vec3
    x_axis: Vec3
    y_axis: Vec3
    z_axis: Vec3

    @property
    def plane_name(self) -> Optional[str]:
        """Return standard plane name if applicable, else None."""
        key = self.z_axis.rounded(4)
        return STANDARD_PLANES.get(key)

    @property
    def is_standard(self) -> bool:
        return self.plane_name is not None

    @property
    def has_nonzero_origin(self) -> bool:
        return not self.origin.is_zero()


# ── Sketch and Extrude ───────────────────────────────────────────────────────

@dataclass
class Sketch:
    entity_id: str
    name: str
    workplane: Workplane
    profiles: dict[str, Profile] = field(default_factory=dict)
    is_orphan: bool = False


OPERATION_MAP = {
    "NewBodyFeatureOperation": "new_body",
    "JoinFeatureOperation": "join",
    "CutFeatureOperation": "cut",
    "IntersectFeatureOperation": "intersect",
}

EXTENT_MAP = {
    "OneSideFeatureExtentType": "one_side",
    "TwoSidesFeatureExtentType": "two_sides",
}


@dataclass
class ProfileRef:
    sketch_id: str
    profile_id: str


@dataclass
class Extrude:
    entity_id: str
    name: str
    operation: str          # new_body | join | cut | intersect
    extent_type: str        # one_side | two_sides
    distance_mm: float
    distance_two_mm: float  # only for two_sides
    taper_angle: float
    profile_refs: list[ProfileRef] = field(default_factory=list)


# ── CAD step (union type) ────────────────────────────────────────────────────

@dataclass
class CADStep:
    index: int
    step_type: str          # "sketch" | "extrude"
    entity: Sketch | Extrude = None


# ── Top-level model ──────────────────────────────────────────────────────────

@dataclass
class CADModel:
    source_file: str
    steps: list[CADStep] = field(default_factory=list)

    @property
    def sketches(self) -> list[Sketch]:
        return [s.entity for s in self.steps if s.step_type == "sketch"]

    @property
    def extrudes(self) -> list[Extrude]:
        return [s.entity for s in self.steps if s.step_type == "extrude"]


# ── Parsing functions ────────────────────────────────────────────────────────

def _parse_curve(d: dict) -> Optional[Curve]:
    ctype = d.get("type")

    if ctype == "Line3D":
        return CurveLine(
            start=Vec3.from_dict(d["start_point"]),
            end=Vec3.from_dict(d["end_point"]),
        )

    if ctype == "Circle3D":
        return CurveCircle(
            center=Vec3.from_dict(d.get("center_point", {})),
            radius_m=float(d.get("radius", 0.0)),
            normal=Vec3.from_dict(d.get("normal", {"z": 1.0})),
        )

    if ctype == "Arc3D":
        return CurveArc(
            start=Vec3.from_dict(d["start_point"]),
            end=Vec3.from_dict(d["end_point"]),
            center=Vec3.from_dict(d.get("center_point", {})),
            radius_m=float(d.get("radius", 0.0)),
        )

    return None  # unknown curve type — skip


def _parse_loop(d: dict) -> Loop:
    curves = []
    for curve_dict in d.get("profile_curves", []):
        curve = _parse_curve(curve_dict)
        if curve:
            curves.append(curve)
    return Loop(is_outer=d.get("is_outer", True), curves=curves)


def _parse_profile(profile_id: str, d: dict) -> Profile:
    loops = [_parse_loop(loop) for loop in d.get("loops", [])]
    return Profile(profile_id=profile_id, loops=loops)


def _parse_workplane(d: dict) -> Workplane:
    return Workplane(
        origin=Vec3.from_dict(d.get("origin", {})),
        x_axis=Vec3.from_dict(d.get("x_axis", {"x": 1.0})),
        y_axis=Vec3.from_dict(d.get("y_axis", {"y": 1.0})),
        z_axis=Vec3.from_dict(d.get("z_axis", {"z": 1.0})),
    )


def _parse_sketch(entity_id: str, d: dict) -> Sketch:
    workplane = _parse_workplane(d.get("transform", {}))
    profiles = {
        pid: _parse_profile(pid, pdata)
        for pid, pdata in d.get("profiles", {}).items()
    }
    return Sketch(
        entity_id=entity_id,
        name=d.get("name", "Sketch"),
        workplane=workplane,
        profiles=profiles,
    )


def _parse_extrude(entity_id: str, d: dict) -> Extrude:
    op_raw = d.get("operation", "NewBodyFeatureOperation")
    ext_raw = d.get("extent_type", "OneSideFeatureExtentType")

    dist_one = (d.get("extent_one", {})
                 .get("distance", {})
                 .get("value", 0.0)) * 1000

    dist_two = (d.get("extent_two", {})
                 .get("distance", {})
                 .get("value", 0.0)) * 1000

    taper = (d.get("extent_one", {})
              .get("taper_angle", {})
              .get("value", 0.0))

    refs = [
        ProfileRef(sketch_id=p["sketch"], profile_id=p["profile"])
        for p in d.get("profiles", [])
    ]

    return Extrude(
        entity_id=entity_id,
        name=d.get("name", "Extrude"),
        operation=OPERATION_MAP.get(op_raw, "new_body"),
        extent_type=EXTENT_MAP.get(ext_raw, "one_side"),
        distance_mm=dist_one,
        distance_two_mm=dist_two,
        taper_angle=taper,
        profile_refs=refs,
    )


def _mark_orphan_sketches(model: CADModel) -> None:
    """Mark sketches not referenced by any extrude as orphans."""
    referenced = set()
    for extrude in model.extrudes:
        for ref in extrude.profile_refs:
            referenced.add(ref.sketch_id)
    for sketch in model.sketches:
        if sketch.entity_id not in referenced:
            sketch.is_orphan = True


def parse(path: str | Path) -> CADModel:
    """
    Parse a DeepCAD JSON file into a CADModel.

    Args:
        path: path to the .json file

    Returns:
        CADModel with ordered steps (sketches + extrudes)

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if the JSON structure is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "sequence" not in data or "entities" not in data:
        raise ValueError(f"Invalid DeepCAD JSON structure: {path.name}")

    entities = data["entities"]
    model = CADModel(source_file=str(path))

    for step_dict in data["sequence"]:
        index = step_dict.get("index", 0)
        stype = step_dict.get("type")
        eid   = step_dict.get("entity")

        entity_data = entities.get(eid)
        if entity_data is None:
            continue  # missing entity — skip step

        if stype == "Sketch":
            entity = _parse_sketch(eid, entity_data)
            model.steps.append(CADStep(index=index,
                                       step_type="sketch",
                                       entity=entity))

        elif stype == "ExtrudeFeature":
            entity = _parse_extrude(eid, entity_data)
            model.steps.append(CADStep(index=index,
                                       step_type="extrude",
                                       entity=entity))

    _mark_orphan_sketches(model)
    return model


# ── Quick debug helper ───────────────────────────────────────────────────────

def summarize(model: CADModel) -> str:
    """Return a human-readable summary of a CADModel."""
    lines = [f"File : {Path(model.source_file).name}",
             f"Steps: {len(model.steps)}"]

    for step in model.steps:
        e = step.entity
        if step.step_type == "sketch":
            plane = e.workplane.plane_name or "custom"
            orphan = " [orphan]" if e.is_orphan else ""
            n_profiles = len(e.profiles)
            curve_types = set()
            for p in e.profiles.values():
                for loop in p.loops:
                    for c in loop.curves:
                        curve_types.add(c.type)
            lines.append(
                f"  [{step.index}] Sketch '{e.name}' "
                f"plane={plane} profiles={n_profiles} "
                f"curves={sorted(curve_types)}{orphan}"
            )
        elif step.step_type == "extrude":
            n_refs = len(e.profile_refs)
            lines.append(
                f"  [{step.index}] Extrude '{e.name}' "
                f"op={e.operation} dist={e.distance_mm:.2f}mm "
                f"extent={e.extent_type} refs={n_refs}"
            )

    return "\n".join(lines)


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python parser.py <path_to_json>")
        sys.exit(1)
    model = parse(sys.argv[1])
    print(summarize(model))
