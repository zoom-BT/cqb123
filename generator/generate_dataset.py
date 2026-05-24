"""
generate_dataset.py — CQB123 dataset generator
Loops over DeepCAD JSON files and produces aligned pairs:
  (cadquery_script, build123d_script)

Output structure:
  output_dir/
    train/
      0000/
        00000007_cq.py
        00000007_b3d.py
    validation/
      ...
    test/
      ...
    manifest.jsonl   ← one JSON line per pair with metadata

Usage:
  python generate_dataset.py \
    --cad_json  /kaggle/input/.../cad_json \
    --split     /kaggle/input/.../train_val_test_split.json \
    --output    /kaggle/working/cqb123_dataset \
    --workers   4 \
    --split_name train          # train | validation | test | all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Allow running from repo root or generator/ subfolder
sys.path.insert(0, str(Path(__file__).parent))

from parser import parse, summarize
from cadquery_gen import generate as gen_cq
from build123d_gen import generate as gen_b3d


# ── Data classes ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, asdict


@dataclass
class PairResult:
    file_id: str          # e.g. "0000/00000007"
    split: str            # train | validation | test
    cq_path: str          # relative path to CadQuery script
    b3d_path: str         # relative path to Build123d script
    seq_length: int
    has_circle: bool
    has_line: bool
    has_arc: bool
    has_custom_plane: bool
    has_orphan_sketch: bool
    has_multi_profile: bool
    status: str           # "ok" | "error"
    error_msg: str = ""


# ── Worker function (runs in subprocess) ─────────────────────────────────────

def process_file(args: tuple) -> PairResult:
    """
    Parse one JSON file and write both generated scripts.
    Returns a PairResult with metadata.
    """
    json_path, file_id, split_name, output_dir = args

    folder, stem = file_id.split("/")
    out_folder = Path(output_dir) / split_name / folder
    out_folder.mkdir(parents=True, exist_ok=True)

    cq_path  = out_folder / f"{stem}_cq.py"
    b3d_path = out_folder / f"{stem}_b3d.py"

    try:
        model = parse(json_path)

        # Generate both scripts
        cq_code  = gen_cq(model)
        b3d_code = gen_b3d(model)

        # Write files
        cq_path.write_text(cq_code,  encoding="utf-8")
        b3d_path.write_text(b3d_code, encoding="utf-8")

        # Collect metadata
        sketches  = model.sketches
        extrudes  = model.extrudes

        has_circle = any(
            any(c.__class__.__name__ == "CurveCircle"
                for p in sk.profiles.values()
                for loop in p.loops
                for c in loop.curves)
            for sk in sketches
        )
        has_line = any(
            any(c.__class__.__name__ == "CurveLine"
                for p in sk.profiles.values()
                for loop in p.loops
                for c in loop.curves)
            for sk in sketches
        )
        has_arc = any(
            any(c.__class__.__name__ == "CurveArc"
                for p in sk.profiles.values()
                for loop in p.loops
                for c in loop.curves)
            for sk in sketches
        )
        has_custom_plane   = any(not sk.workplane.is_standard for sk in sketches)
        has_orphan_sketch  = any(sk.is_orphan for sk in sketches)
        has_multi_profile  = any(len(ex.profile_refs) > 1 for ex in extrudes)

        return PairResult(
            file_id           = file_id,
            split             = split_name,
            cq_path           = str(cq_path.relative_to(output_dir)),
            b3d_path          = str(b3d_path.relative_to(output_dir)),
            seq_length        = len(model.steps),
            has_circle        = has_circle,
            has_line          = has_line,
            has_arc           = has_arc,
            has_custom_plane  = has_custom_plane,
            has_orphan_sketch = has_orphan_sketch,
            has_multi_profile = has_multi_profile,
            status            = "ok",
        )

    except Exception as e:
        return PairResult(
            file_id           = file_id,
            split             = split_name,
            cq_path           = "",
            b3d_path          = "",
            seq_length        = 0,
            has_circle        = False,
            has_line          = False,
            has_arc           = False,
            has_custom_plane  = False,
            has_orphan_sketch = False,
            has_multi_profile = False,
            status            = "error",
            error_msg         = str(e),
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def build_args(
    cad_json_root: Path,
    split_ids: list[str],
    split_name: str,
    output_dir: Path,
) -> list[tuple]:
    """Build the list of (json_path, file_id, split_name, output_dir) tuples."""
    tasks = []
    for file_id in split_ids:
        json_path = cad_json_root / f"{file_id}.json"
        if json_path.exists():
            tasks.append((str(json_path), file_id, split_name, str(output_dir)))
    return tasks


def run(
    cad_json_root: str,
    split_path: str,
    output_dir: str,
    split_name: str = "train",
    workers: int = 4,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> None:

    cad_root   = Path(cad_json_root)
    output     = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load split
    with open(split_path, encoding="utf-8") as f:
        split_data = json.load(f)

    if split_name == "all":
        ids_per_split = {k: v for k, v in split_data.items()}
    else:
        if split_name not in split_data:
            raise ValueError(f"Split '{split_name}' not found. "
                             f"Available: {list(split_data.keys())}")
        ids_per_split = {split_name: split_data[split_name]}

    manifest_path = output / "manifest.jsonl"
    manifest_f    = open(manifest_path, "w", encoding="utf-8")

    total_ok     = 0
    total_errors = 0

    for sname, ids in ids_per_split.items():
        if limit:
            ids = ids[:limit]

        tasks = build_args(cad_root, ids, sname, output)
        n     = len(tasks)
        print(f"\n── {sname} : {n:,} files ──")

        done = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_file, t): t for t in tasks}
            for future in as_completed(futures):
                result: PairResult = future.result()
                done += 1

                if result.status == "ok":
                    total_ok += 1
                else:
                    total_errors += 1
                    if verbose:
                        print(f"  ERROR {result.file_id}: {result.error_msg}")

                # Write manifest line
                manifest_f.write(json.dumps(asdict(result)) + "\n")

                # Progress
                if done % 1000 == 0 or done == n:
                    pct = done / n * 100
                    print(f"  [{done:6,}/{n:6,}]  {pct:.1f}%  "
                          f"ok={total_ok:,}  errors={total_errors:,}")

    manifest_f.close()

    # Summary
    print(f"\n{'='*50}")
    print(f"Dataset generation complete")
    print(f"  OK     : {total_ok:,}")
    print(f"  Errors : {total_errors:,}")
    print(f"  Total  : {total_ok + total_errors:,}")
    print(f"  Output : {output}")
    print(f"  Manifest : {manifest_path}")

    # Write a small stats file
    stats = {
        "total_ok"    : total_ok,
        "total_errors": total_errors,
        "splits"      : list(ids_per_split.keys()),
        "output_dir"  : str(output),
    }
    with open(output / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate CQB123 paired dataset from DeepCAD JSONs"
    )
    parser.add_argument(
        "--cad_json", required=True,
        help="Path to DeepCAD cad_json root folder"
    )
    parser.add_argument(
        "--split", required=True,
        help="Path to train_val_test_split.json"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for generated scripts"
    )
    parser.add_argument(
        "--split_name", default="train",
        choices=["train", "validation", "test", "all"],
        help="Which split to generate (default: train)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of files per split (for testing)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print errors as they occur"
    )
    args = parser.parse_args()

    run(
        cad_json_root = args.cad_json,
        split_path    = args.split,
        output_dir    = args.output,
        split_name    = args.split_name,
        workers       = args.workers,
        limit         = args.limit,
        verbose       = args.verbose,
    )


if __name__ == "__main__":
    main()
