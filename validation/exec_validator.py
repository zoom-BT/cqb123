"""
exec_validator.py — CQB123 execution validator (V2)
Executes generated CadQuery and Build123d scripts, checks compilation,
and optionally measures geometric equivalence via Chamfer distance.

Key changes vs V1:
  - Per-pair timeout (prevents BrokenProcessPool on heavy meshes)
  - Chamfer disabled by default (compute_chamfer flag)
  - Lighter, protected point sampler with coarse mesh
  - Robust worker isolation — one crash never kills the batch

Usage (compilation only — fast):
  validate_batch(manifest, dataset, output, limit=500, workers=4)

Usage (with Chamfer — slower):
  validate_batch(..., compute_chamfer=True, timeout=20)
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    file_id:    str
    cq_ok:      bool
    b3d_ok:     bool
    both_ok:    bool
    cq_error:   str
    b3d_error:  str
    chamfer_mm: float   # -1.0 if not computed
    n_points:   int
    status:     str     # ok | partial | error | timeout


# ── Timeout via signal (per worker process) ──────────────────────────────────

class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


# ── Script execution ──────────────────────────────────────────────────────────

def _run_cq(script_path: str):
    """Execute a CadQuery script. Returns (ok, error, solid)."""
    try:
        import cadquery as cq
        code = Path(script_path).read_text(encoding="utf-8")
        captured = {}
        ns = {"cq": cq,
              "show_object": lambda o, **k: captured.update({"s": o})}
        exec(compile(code, script_path, "exec"), ns)
        solid = captured.get("s")
        if solid is None:
            for v in ns.values():
                if isinstance(v, cq.Workplane):
                    solid = v
                    break
        if solid is None:
            return False, "No solid produced", None
        return True, "", solid
    except Exception as e:
        return False, str(e)[:150], None


def _run_b3d(script_path: str):
    """Execute a Build123d script. Returns (ok, error, solid)."""
    try:
        import build123d as b3d
        code = Path(script_path).read_text(encoding="utf-8")
        captured = {}
        ns = {**{n: getattr(b3d, n) for n in dir(b3d)},
              "show_object": lambda o, **k: captured.update({"s": o})}
        exec(compile(code, script_path, "exec"), ns)
        solid = captured.get("s")
        if solid is None:
            for v in ns.values():
                if hasattr(v, "part"):
                    solid = v.part
                    break
        if solid is None:
            return False, "No solid produced", None
        return True, "", solid
    except Exception as e:
        return False, str(e)[:150], None


# ── Point sampling (coarse, protected) ────────────────────────────────────────

def _sample_occ(wrapped, n_points: int = 256, deflection: float = 1.0):
    """
    Sample surface points from an OCCT shape using a COARSE mesh.
    deflection=1.0 keeps meshing fast even for complex parts.
    """
    try:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.BRep import BRep_Tool

        BRepMesh_IncrementalMesh(wrapped, deflection).Perform()

        pts = []
        exp = TopExp_Explorer(wrapped, TopAbs_FACE)
        while exp.More():
            face = exp.Current()
            loc = BRep_Tool.Location_s(face)
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is not None:
                for i in range(1, tri.NbNodes() + 1):
                    node = tri.Node(i)
                    pts.append([node.X(), node.Y(), node.Z()])
            exp.Next()

        if not pts:
            return None
        arr = np.array(pts, dtype=np.float32)
        if len(arr) > n_points:
            idx = np.random.choice(len(arr), n_points, replace=False)
            arr = arr[idx]
        return arr
    except Exception:
        return None


def _get_wrapped(solid):
    """Extract the OCCT wrapped shape from a CQ or B3D object."""
    try:
        if hasattr(solid, "val"):          # CadQuery Workplane
            return solid.val().wrapped
        if hasattr(solid, "wrapped"):      # B3D Solid/Part
            return solid.wrapped
        if hasattr(solid, "part"):         # B3D BuildPart
            return solid.part.wrapped
    except Exception:
        return None
    return None


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance in mm."""
    diff = a[:, None, :] - b[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) / 2 * 1000)


# ── Single pair validation ────────────────────────────────────────────────────

def validate_pair(
    cq_path: str,
    b3d_path: str,
    file_id: str = "",
    compute_chamfer: bool = False,
    n_points: int = 256,
) -> ValidationResult:

    cq_ok, cq_err, cq_solid = _run_cq(cq_path)
    b3d_ok, b3d_err, b3d_solid = _run_b3d(b3d_path)

    chamfer = -1.0
    n_pts = 0

    if compute_chamfer and cq_ok and b3d_ok:
        wa = _get_wrapped(cq_solid)
        wb = _get_wrapped(b3d_solid)
        if wa is not None and wb is not None:
            pa = _sample_occ(wa, n_points)
            pb = _sample_occ(wb, n_points)
            if pa is not None and pb is not None and len(pa) and len(pb):
                chamfer = chamfer_distance(pa, pb)
                n_pts = min(len(pa), len(pb))

    both = cq_ok and b3d_ok
    status = "ok" if both else ("partial" if (cq_ok or b3d_ok) else "error")

    return ValidationResult(
        file_id=file_id, cq_ok=cq_ok, b3d_ok=b3d_ok, both_ok=both,
        cq_error=cq_err, b3d_error=b3d_err,
        chamfer_mm=chamfer, n_points=n_pts, status=status,
    )


# ── Worker with timeout ───────────────────────────────────────────────────────

def _worker(args):
    cq_path, b3d_path, file_id, compute_chamfer, n_points, timeout = args

    # Install SIGALRM timeout (unix only)
    if timeout > 0 and hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)

    try:
        result = validate_pair(cq_path, b3d_path, file_id,
                               compute_chamfer, n_points)
        if timeout > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        return result
    except _Timeout:
        return ValidationResult(
            file_id=file_id, cq_ok=False, b3d_ok=False, both_ok=False,
            cq_error="timeout", b3d_error="timeout",
            chamfer_mm=-1.0, n_points=0, status="timeout",
        )
    except Exception as e:
        if timeout > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        return ValidationResult(
            file_id=file_id, cq_ok=False, b3d_ok=False, both_ok=False,
            cq_error=str(e)[:150], b3d_error=str(e)[:150],
            chamfer_mm=-1.0, n_points=0, status="error",
        )


# ── Batch validation ──────────────────────────────────────────────────────────

def validate_batch(
    manifest_path: str,
    dataset_dir: str,
    output_path: str,
    limit: Optional[int] = None,
    workers: int = 4,
    compute_chamfer: bool = False,
    n_points: int = 256,
    timeout: int = 15,
    split_filter: Optional[str] = None,
) -> dict:

    dataset = Path(dataset_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]
    entries = [e for e in entries if e["status"] == "ok"]
    if split_filter:
        entries = [e for e in entries if e["split"] == split_filter]
    if limit:
        entries = entries[:limit]

    n = len(entries)
    mode = "compile + chamfer" if compute_chamfer else "compile only"
    print(f"Validating {n:,} pairs  ({mode}, workers={workers}, "
          f"timeout={timeout}s)")

    tasks = [
        (str(dataset / e["cq_path"]), str(dataset / e["b3d_path"]),
         e["file_id"], compute_chamfer, n_points, timeout)
        for e in entries
    ]

    counters = {"ok": 0, "partial": 0, "error": 0, "timeout": 0}
    cq_total = 0
    b3d_total = 0
    chamfer_vals = []

    def _record(r, out_f):
        nonlocal cq_total, b3d_total
        counters[r.status] += 1
        if r.cq_ok:
            cq_total += 1
        if r.b3d_ok:
            b3d_total += 1
        if r.chamfer_mm >= 0:
            chamfer_vals.append(r.chamfer_mm)
        out_f.write(json.dumps(asdict(r)) + "\n")

    # Process in chunks so a crashed worker only loses its chunk,
    # not the whole batch. A fresh executor is created per chunk.
    CHUNK = 500
    done = 0

    with open(output, "w", encoding="utf-8") as out_f:
        for chunk_start in range(0, n, CHUNK):
            chunk = tasks[chunk_start:chunk_start + CHUNK]
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(_worker, t): t for t in chunk}
                    for fut in as_completed(futures):
                        try:
                            r = fut.result(timeout=timeout + 10)
                        except Exception:
                            t = futures[fut]
                            r = ValidationResult(
                                file_id=t[2], cq_ok=False, b3d_ok=False,
                                both_ok=False, cq_error="worker_error",
                                b3d_error="worker_error", chamfer_mm=-1.0,
                                n_points=0, status="error")
                        _record(r, out_f)
                        done += 1
            except Exception:
                # Whole chunk pool died → fall back to sequential for this chunk
                for t in chunk:
                    try:
                        r = _worker(t)
                    except Exception:
                        r = ValidationResult(
                            file_id=t[2], cq_ok=False, b3d_ok=False,
                            both_ok=False, cq_error="seq_error",
                            b3d_error="seq_error", chamfer_mm=-1.0,
                            n_points=0, status="error")
                    _record(r, out_f)
                    done += 1

            pct = done / n * 100
            print(f"  [{done:6,}/{n:6,}]  {pct:5.1f}%  "
                  f"both={counters['ok']:,}  "
                  f"cq={cq_total:,}  b3d={b3d_total:,}  "
                  f"err={counters['error']:,}")

    # Report
    print(f"\n{'='*55}")
    print(f"Validation complete — {n:,} pairs")
    print(f"  Both OK   : {counters['ok']:,}  ({counters['ok']/n*100:.1f}%)")
    print(f"  CadQuery  : {cq_total:,}  ({cq_total/n*100:.1f}%)")
    print(f"  Build123d : {b3d_total:,}  ({b3d_total/n*100:.1f}%)")
    print(f"  Partial   : {counters['partial']:,}  "
          f"({counters['partial']/n*100:.1f}%)")
    print(f"  Errors    : {counters['error']:,}  "
          f"({counters['error']/n*100:.1f}%)")
    print(f"  Timeouts  : {counters['timeout']:,}  "
          f"({counters['timeout']/n*100:.1f}%)")

    if chamfer_vals:
        arr = np.array(chamfer_vals)
        print(f"\n  Chamfer (mm) — {len(arr):,} pairs")
        print(f"    mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
              f"p95={np.percentile(arr,95):.4f}")
        print(f"    <1mm: {(arr<1).mean()*100:.1f}%   "
              f"<0.1mm: {(arr<0.1).mean()*100:.1f}%")

    print(f"\n  Results : {output}")

    return {
        "n": n,
        "both_ok": counters["ok"],
        "cq_ok": cq_total,
        "b3d_ok": b3d_total,
        "errors": counters["error"],
        "timeouts": counters["timeout"],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--chamfer", action="store_true")
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--timeout", type=int, default=15)
    p.add_argument("--split", default=None)
    a = p.parse_args()
    validate_batch(
        a.manifest, a.dataset, a.output,
        limit=a.limit, workers=a.workers,
        compute_chamfer=a.chamfer, n_points=a.points,
        timeout=a.timeout, split_filter=a.split,
    )
