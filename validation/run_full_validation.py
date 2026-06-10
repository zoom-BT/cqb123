"""
run_full_validation.py — Batch validation orchestrator with resume.

Validates the entire train split in chunks, computing Chamfer distance,
and writes everything to a single resumable output file. If the Kaggle
session restarts, just re-run: it skips already-validated pairs.

Designed to be called from a Kaggle notebook cell:

    from run_full_validation import run_full
    run_full(
        dataset_dir = "/kaggle/working/cqb123_final",
        output      = "/kaggle/working/chamfer_full.jsonl",
        batch_size  = 20000,
        workers     = 2,
    )

Persist the output file to a Kaggle Dataset between sessions so progress
survives restarts.
"""

from __future__ import annotations

import json
from pathlib import Path

from exec_validator import validate_batch


def run_full(
    dataset_dir: str,
    output: str,
    manifest_path: str = None,
    batch_size: int = 20000,
    workers: int = 2,
    n_points: int = 256,
    timeout: int = 20,
    split_filter: str = "train",
    compute_chamfer: bool = True,
) -> None:
    """
    Validate the full split in resumable batches.

    The output file accumulates results. Re-running skips done pairs.
    """
    dataset = Path(dataset_dir)
    if manifest_path is None:
        manifest_path = str(dataset / "manifest.jsonl")

    # Count total eligible entries
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]
    entries = [e for e in entries
               if e["status"] == "ok"
               and (not split_filter or e["split"] == split_filter)]
    total = len(entries)

    # Count how many already done
    done = 0
    out_path = Path(output)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = sum(1 for l in f if l.strip())

    print(f"Total to validate : {total:,}")
    print(f"Already done      : {done:,}")
    print(f"Remaining         : {total - done:,}\n")

    # Process remaining in batches, always with resume=True
    while done < total:
        print(f"\n{'#'*55}")
        print(f"# Batch starting at offset {done:,}")
        print(f"{'#'*55}")

        validate_batch(
            manifest_path   = manifest_path,
            dataset_dir     = dataset_dir,
            output_path     = output,
            offset          = 0,            # resume handles positioning
            limit           = done + batch_size,
            workers         = workers,
            compute_chamfer = compute_chamfer,
            n_points        = n_points,
            timeout         = timeout,
            split_filter    = split_filter,
            resume          = True,
        )

        # Recount
        with open(out_path, encoding="utf-8") as f:
            done = sum(1 for l in f if l.strip())
        print(f"\n>>> Progress: {done:,}/{total:,} "
              f"({done/total*100:.1f}%)")

    print(f"\n{'='*55}")
    print(f"FULL VALIDATION COMPLETE — {done:,} pairs")
    print(f"Output: {output}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--batch_size", type=int, default=20000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--split", default="train")
    a = p.parse_args()
    run_full(
        dataset_dir=a.dataset, output=a.output,
        manifest_path=a.manifest, batch_size=a.batch_size,
        workers=a.workers, split_filter=a.split,
    )
