#!/usr/bin/env python3
"""
Extract and prepare all ML-Master submissions for grading.

This script:
1. Finds all submission_*.csv files in the workspace
2. Copies them to a grading directory
3. Creates submission files for each one that MLE-bench can grade
4. Generates a report of all submissions with their metrics

Usage:
    python extract_all_submissions.py <workspace_dir> <output_dir>
"""

import sys
import shutil
from pathlib import Path
import json
import re


def extract_submissions(workspace_dir: Path, output_dir: Path):
    """Extract all submissions from ML-Master workspace."""

    submission_dir = workspace_dir / "submission"
    if not submission_dir.exists():
        print(f"ERROR: Submission directory not found: {submission_dir}")
        return

    # Find all submission files
    submission_files = sorted(submission_dir.glob("submission_*.csv"))

    if not submission_files:
        print("WARNING: No submission files found")
        # Check for best submission
        best_submission = workspace_dir / "best_submission" / "submission.csv"
        if best_submission.exists():
            submission_files = [best_submission]
        else:
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load journal to get metrics for each submission
    journal_file = workspace_dir.parent / "logs" / workspace_dir.name / "journal.jsonl"
    node_metrics = {}
    if journal_file.exists():
        with open(journal_file, 'r') as f:
            for line in f:
                node = json.loads(line)
                node_id = node.get("id")
                metric = node.get("metric", {}).get("value")
                if node_id and metric is not None:
                    node_metrics[node_id] = metric

    # Extract each submission
    results = []
    for i, sub_file in enumerate(submission_files):
        # Extract node ID from filename
        match = re.search(r'submission_([a-f0-9]+)\.csv', sub_file.name)
        node_id = match.group(1) if match else f"node_{i}"
        metric = node_metrics.get(node_id, "unknown")

        # Copy to output directory with index
        output_file = output_dir / f"submission_{i:03d}_{node_id}.csv"
        shutil.copy(sub_file, output_file)

        results.append({
            "index": i,
            "node_id": node_id,
            "metric": metric,
            "file": str(output_file.relative_to(output_dir.parent)),
        })

        print(f"Extracted submission {i}: {node_id} (metric={metric})")

    # Save results manifest
    manifest_file = output_dir / "submissions_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump({
            "total_submissions": len(results),
            "workspace": str(workspace_dir),
            "submissions": results,
        }, f, indent=2)

    print(f"\nExtracted {len(results)} submissions to {output_dir}")
    print(f"Manifest saved to {manifest_file}")

    # Find best submission by metric
    best_sub = None
    best_metric = None
    for r in results:
        if r["metric"] != "unknown":
            if best_metric is None or r["metric"] > best_metric:
                best_metric = r["metric"]
                best_sub = r

    if best_sub:
        print(f"\nBest submission: {best_sub['node_id']} (metric={best_sub['metric']})")
        # Copy best submission as final submission.csv
        best_file = output_dir / "submission.csv"
        shutil.copy(output_dir / Path(best_sub['file']).name, best_file)
        print(f"Best submission copied to {best_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_all_submissions.py <workspace_dir> <output_dir>")
        sys.exit(1)

    workspace_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    extract_submissions(workspace_dir, output_dir)
