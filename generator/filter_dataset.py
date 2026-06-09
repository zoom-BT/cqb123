"""
filter_dataset.py — CQB123 dataset filtering on geometric equivalence
Reads a Chamfer validation file and produces a clean manifest containing
only pairs whose two solids are geometrically equivalent.

A pair is "verified" if:
  - both_ok is True (both scripts compile)
  - 0 <= chamfer_mm < threshold (default 0.5 mm)

Output:
  - clean_manifest.jsonl : verified pairs only
  - filter_report.json   : statistics

Usage:
  python filter_dataset.py \
    --validation /kaggle/working/chamfer_full.jsonl \
    --manifest   /kaggle/working/cqb123_final/manifest.jsonl \
    --output     /kaggle/working/cqb123_clean \
    --threshold  0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter


def filter_dataset(
    validation_path: str,
    manifest_path: str,
    output_dir: str,
    threshold: float = 0.5,
) -> dict:
    """
    Filter pairs on geometric equivalence and write a clean manifest.

    Args:
        validation_path: JSONL produced by exec_validator with chamfer_mm
        manifest_path:   original dataset manifest (for metadata)
        output_dir:      where clean_manifest.jsonl is written
        threshold:       max Chamfer distance (mm) to count as equivalent

    Returns:
        dict of statistics
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load validation results
    with open(validation_path, encoding="utf-8") as f:
        val = [json.loads(l) for l in f if l.strip()]

    # Load original manifest (metadata by file_id)
    meta = {}
    with open(manifest_path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                e = json.loads(l)
                meta[e["file_id"]] = e

    verified = []
    counters = Counter()

    for r in val:
        fid = r["file_id"]
        ch  = r["chamfer_mm"]

        if not r["both_ok"]:
            counters["not_both_ok"] += 1
            continue
        if ch < 0:
            counters["no_chamfer"] += 1
            continue
        if ch >= threshold:
            counters["divergent"] += 1
            continue

        # Verified pair
        counters["verified"] += 1
        entry = dict(meta.get(fid, {}))
        entry["chamfer_mm"] = ch
        entry["verified"]   = True
        verified.append(entry)

    # Write clean manifest
    clean_path = out / "clean_manifest.jsonl"
    with open(clean_path, "w", encoding="utf-8") as f:
        for e in verified:
            f.write(json.dumps(e) + "\n")

    total = len(val)
    report = {
        "total_validated": total,
        "verified": counters["verified"],
        "divergent": counters["divergent"],
        "not_both_ok": counters["not_both_ok"],
        "no_chamfer": counters["no_chamfer"],
        "threshold_mm": threshold,
        "verified_pct": round(counters["verified"] / total * 100, 2) if total else 0,
        "clean_manifest": str(clean_path),
    }

    with open(out / "filter_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"{'='*55}")
    print(f"Dataset filtering — threshold {threshold} mm")
    print(f"  Total validated : {total:,}")
    print(f"  Verified        : {counters['verified']:,}  "
          f"({report['verified_pct']:.1f}%)")
    print(f"  Divergent       : {counters['divergent']:,}")
    print(f"  Not both-OK     : {counters['not_both_ok']:,}")
    print(f"  No chamfer       : {counters['no_chamfer']:,}")
    print(f"\n  Clean manifest  : {clean_path}")

    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--validation", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()
    filter_dataset(a.validation, a.manifest, a.output, a.threshold)
