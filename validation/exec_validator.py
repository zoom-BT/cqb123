"""
exec_validator.py — CQB123 execution validator
Executes generated CadQuery and Build123d scripts, checks compilation,
and measures geometric equivalence via Chamfer distance.

Metrics produced per pair:
  - cq_ok        : CadQuery script compiles and runs
  - b3d_ok       : Build123d script compiles and runs
  - both_ok      : both scripts succeed
  - chamfer_mm   : Chamfer distance between the two solids (mm)
  - parametric   : both scripts contain named dimension variables

Usage (single pair):
  python exec_validator.py \
    --cq  path/to/script_cq.py \
    --b3d path/to/script_b3d.py

Usage (batch from manifest):
  python exec_validator.py \
    --manifest /kaggle/working/cqb123_dataset/manifest.jsonl \
    --dataset  /kaggle/working/cqb123_dataset \
    --output   /kaggle/working/cqb123_validation.jsonl \
    --limit    1000 \
    --workers  4
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    file_id:      str
    cq_ok:        bool
    b3d_ok:       bool
    both_ok:      bool
    cq_error:     str
    b3d_error:    str
    chamfer_mm:   float   # -1.0 if not computable
    n_points:     int     # points sampled per solid
    status:       str     # "ok" | "partial" | "error"


# ── Solid execution ───────────────────────────────────────────────────────────

def _run_cq_script(script_path: str) -> tuple[bool, str, object]:
    """
    Execute a CadQuery script and return the resulting solid.
    Returns (success, error_msg, solid_or_None)
    """
    try:
        import cadquery as cq

        code = Path(script_path).read_text(encoding="utf-8")

        # Replace show_object with a capture
        captured = {}
        namespace = {
            "cq": cq,
            "show_object": lambda obj, **kw: captured.update({"solid": obj}),
        }

        exec(compile(code, script_path, "exec"), namespace)

        solid = captured.get("solid")
        if solid is None:
            # Try to find any CQ object in namespace
            for v in namespace.values():
                if isinstance(v, cq.Workplane):
                    solid = v
                    break

        if solid is None:
            return False, "No solid produced", None

        return True, "", solid

    except Exception as e:
        return False, str(e), None


def _run_b3d_script(script_path: str) -> tuple[bool, str, object]:
    """
    Execute a Build123d script and return the resulting solid.
    Returns (success, error_msg, solid_or_None)
    """
    try:
        import build123d as b3d

        code = Path(script_path).read_text(encoding="utf-8")

        captured = {}
        namespace = {
            **{name: getattr(b3d, name) for name in dir(b3d)},
            "show_object": lambda obj, **kw: captured.update({"solid": obj}),
        }

        exec(compile(code, script_path, "exec"), namespace)

        solid = captured.get("solid")
        if solid is None:
            # Try to find a BuildPart result
            for v in namespace.values():
                if hasattr(v, "part"):
                    solid = v.part
                    break

        if solid is None:
            return False, "No solid produced", None

        return True, "", solid

    except Exception as e:
        return False, str(e), None


# ── Point sampling ────────────────────────────────────────────────────────────

def _sample_points_cq(solid, n_points: int = 512) -> Optional[np.ndarray]:
    """Sample n_points from the surface of a CadQuery solid."""
    try:
        import cadquery as cq
        # Get the underlying OCCT shape
        shape = solid.val() if hasattr(solid, "val") else solid
        # Use BRep sampling via OCC
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Plane

        mesh = BRepMesh_IncrementalMesh(shape.wrapped, 0.1)
        mesh.Perform()

        pts = []
        explorer = TopExp_Explorer(shape.wrapped, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            location = BRep_Tool.Location_s(face)
            triangulation = BRep_Tool.Triangulation_s(face, location)
            if triangulation is not None:
                for i in range(1, triangulation.NbNodes() + 1):
                    node = triangulation.Node(i)
                    pts.append([node.X(), node.Y(), node.Z()])
            explorer.Next()

        if not pts:
            return None

        pts = np.array(pts, dtype=np.float32)

        # Subsample if too many points
        if len(pts) > n_points:
            idx = np.random.choice(len(pts), n_points, replace=False)
            pts = pts[idx]

        return pts

    except Exception:
        return None


def _sample_points_b3d(solid, n_points: int = 512) -> Optional[np.ndarray]:
    """Sample n_points from the surface of a Build123d solid."""
    try:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.BRep import BRep_Tool

        # Get wrapped OCC shape
        shape = solid
        if hasattr(solid, "wrapped"):
            shape_wrapped = solid.wrapped
        elif hasattr(solid, "part"):
            shape_wrapped = solid.part.wrapped
        else:
            return None

        mesh = BRepMesh_IncrementalMesh(shape_wrapped, 0.1)
        mesh.Perform()

        pts = []
        explorer = TopExp_Explorer(shape_wrapped, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            location = BRep_Tool.Location_s(face)
            triangulation = BRep_Tool.Triangulation_s(face, location)
            if triangulation is not None:
                for i in range(1, triangulation.NbNodes() + 1):
                    node = triangulation.Node(i)
                    pts.append([node.X(), node.Y(), node.Z()])
            explorer.Next()

        if not pts:
            return None

        pts = np.array(pts, dtype=np.float32)

        if len(pts) > n_points:
            idx = np.random.choice(len(pts), n_points, replace=False)
            pts = pts[idx]

        return pts

    except Exception:
        return None


# ── Chamfer distance ──────────────────────────────────────────────────────────

def chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """
    Compute the symmetric Chamfer distance between two point clouds.
    Both arrays are (N, 3) float32.
    Returns distance in mm.
    """
    # A → B : for each point in A, find nearest in B
    diff_ab = pts_a[:, None, :] - pts_b[None, :, :]   # (N, M, 3)
    dist_ab = np.sqrt((diff_ab ** 2).sum(axis=2))       # (N, M)
    min_ab  = dist_ab.min(axis=1).mean()                # scalar

    # B → A
    dist_ba = dist_ab.T
    min_ba  = dist_ba.min(axis=1).mean()

    # Convert m → mm (DeepCAD units are metres)
    return float((min_ab + min_ba) / 2 * 1000)


# ── Single pair validation ────────────────────────────────────────────────────

def validate_pair(
    cq_path: str,
    b3d_path: str,
    file_id: str = "",
    n_points: int = 512,
) -> ValidationResult:
    """
    Validate a single (CadQuery, Build123d) pair.
    Executes both scripts and computes Chamfer distance if both succeed.
    """
    cq_ok,  cq_err,  cq_solid  = _run_cq_script(cq_path)
    b3d_ok, b3d_err, b3d_solid = _run_b3d_script(b3d_path)

    chamfer = -1.0
    n_pts   = 0

    if cq_ok and b3d_ok:
        pts_cq  = _sample_points_cq(cq_solid,  n_points)
        pts_b3d = _sample_points_b3d(b3d_solid, n_points)

        if pts_cq is not None and pts_b3d is not None and \
           len(pts_cq) > 0 and len(pts_b3d) > 0:
            chamfer = chamfer_distance(pts_cq, pts_b3d)
            n_pts   = min(len(pts_cq), len(pts_b3d))

    both_ok = cq_ok and b3d_ok
    status  = "ok" if both_ok else ("partial" if (cq_ok or b3d_ok) else "error")

    return ValidationResult(
        file_id    = file_id,
        cq_ok      = cq_ok,
        b3d_ok     = b3d_ok,
        both_ok    = both_ok,
        cq_error   = cq_err,
        b3d_error  = b3d_err,
        chamfer_mm = chamfer,
        n_points   = n_pts,
        status     = status,
    )


# ── Batch worker ──────────────────────────────────────────────────────────────

def _worker(args: tuple) -> ValidationResult:
    cq_path, b3d_path, file_id, n_points = args
    try:
        return validate_pair(cq_path, b3d_path, file_id, n_points)
    except Exception as e:
        return ValidationResult(
            file_id    = file_id,
            cq_ok      = False,
            b3d_ok     = False,
            both_ok    = False,
            cq_error   = str(e),
            b3d_error  = str(e),
            chamfer_mm = -1.0,
            n_points   = 0,
            status     = "error",
        )


# ── Batch validation ──────────────────────────────────────────────────────────

def validate_batch(
    manifest_path: str,
    dataset_dir: str,
    output_path: str,
    limit: Optional[int] = None,
    workers: int = 4,
    n_points: int = 512,
    split_filter: Optional[str] = None,
) -> None:
    """
    Run validation over all pairs listed in a manifest.jsonl file.
    Writes results to output_path as JSONL.
    """
    dataset = Path(dataset_dir)
    output  = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    # Filter
    entries = [e for e in entries if e["status"] == "ok"]
    if split_filter:
        entries = [e for e in entries if e["split"] == split_filter]
    if limit:
        entries = entries[:limit]

    n = len(entries)
    print(f"Validating {n:,} pairs  (workers={workers}, n_points={n_points})")

    tasks = [
        (
            str(dataset / e["cq_path"]),
            str(dataset / e["b3d_path"]),
            e["file_id"],
            n_points,
        )
        for e in entries
    ]

    counters = {"ok": 0, "partial": 0, "error": 0}
    chamfer_vals = []

    with open(output, "w", encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, t): t for t in tasks}
            done = 0
            for future in as_completed(futures):
                result: ValidationResult = future.result()
                done += 1
                counters[result.status] += 1
                if result.chamfer_mm >= 0:
                    chamfer_vals.append(result.chamfer_mm)

                out_f.write(json.dumps(asdict(result)) + "\n")

                if done % 100 == 0 or done == n:
                    mean_ch = (
                        f"{np.mean(chamfer_vals):.4f} mm"
                        if chamfer_vals else "n/a"
                    )
                    print(
                        f"  [{done:6,}/{n:6,}]  "
                        f"ok={counters['ok']:,}  "
                        f"partial={counters['partial']:,}  "
                        f"errors={counters['error']:,}  "
                        f"chamfer_mean={mean_ch}"
                    )

    # Final report
    print(f"\n{'='*55}")
    print(f"Validation complete — {n:,} pairs")
    print(f"  Both OK  : {counters['ok']:,}  "
          f"({counters['ok']/n*100:.1f}%)")
    print(f"  Partial  : {counters['partial']:,}  "
          f"({counters['partial']/n*100:.1f}%)")
    print(f"  Errors   : {counters['error']:,}  "
          f"({counters['error']/n*100:.1f}%)")

    if chamfer_vals:
        arr = np.array(chamfer_vals)
        print(f"\n  Chamfer distance (mm) — {len(arr):,} pairs")
        print(f"    mean   : {arr.mean():.4f}")
        print(f"    median : {np.median(arr):.4f}")
        print(f"    p95    : {np.percentile(arr, 95):.4f}")
        print(f"    max    : {arr.max():.4f}")
        print(f"    == 0   : {(arr == 0).sum():,} pairs  "
              f"({(arr == 0).mean()*100:.1f}%)")
        print(f"    < 1mm  : {(arr < 1).sum():,} pairs  "
              f"({(arr < 1).mean()*100:.1f}%)")

    print(f"\n  Results : {output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate CQB123 generated script pairs"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Single pair
    single = subparsers.add_parser("single", help="Validate one pair")
    single.add_argument("--cq",  required=True, help="CadQuery script path")
    single.add_argument("--b3d", required=True, help="Build123d script path")
    single.add_argument("--id",  default="",    help="File ID label")

    # Batch
    batch = subparsers.add_parser("batch", help="Validate from manifest")
    batch.add_argument("--manifest", required=True)
    batch.add_argument("--dataset",  required=True)
    batch.add_argument("--output",   required=True)
    batch.add_argument("--limit",    type=int, default=None)
    batch.add_argument("--workers",  type=int, default=4)
    batch.add_argument("--points",   type=int, default=512)
    batch.add_argument("--split",    default=None,
                       choices=["train", "validation", "test"])

    args = parser.parse_args()

    if args.command == "single":
        result = validate_pair(args.cq, args.b3d, args.id)
        print(json.dumps(asdict(result), indent=2))

    elif args.command == "batch":
        validate_batch(
            manifest_path = args.manifest,
            dataset_dir   = args.dataset,
            output_path   = args.output,
            limit         = args.limit,
            workers       = args.workers,
            n_points      = args.points,
            split_filter  = args.split,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
