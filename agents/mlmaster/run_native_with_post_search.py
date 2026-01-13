#!/usr/bin/env python3
"""
Run ML-Master natively (no Docker) with AIDE-like artifacts + MLE-bench grading.

This script runs ML-Master once and relies on ML-Master to export these artifacts
into its log directory (AIDE-style):
- tree_plot.html
- journal.json / journal.jsonl
- final_selection.json
- submission_*.csv variants (post_search/raw/max_min/mean_minus_k_std/...)
- solutions/ (node_*.py + per-node submissions + metrics.jsonl)

Then it:
- Copies the artifacts into an easy-to-browse run directory
- Copies all discovered submission variants into `submission/`
- Grades every discovered variant using MLE-bench and writes JSON reports

Usage:
  python run_native_with_post_search.py <competition_dir> <steps>
Example:
  python run_native_with_post_search.py /home/ka3094/dataset_submit/playground_series_s5e8 5
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MLEBENCH_ROOT = Path(__file__).resolve().parents[2]
if str(MLEBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(MLEBENCH_ROOT))

MLEBENCH_DATA_DIR = MLEBENCH_ROOT / "data" / "competitions"

# AIDE-like file names written by ML-Master into its log dir.
SUBMISSION_VARIANTS = {
    "post_search": "submission_post_search.csv",
    "raw": "submission_raw.csv",
    "max_min": "submission_max_min.csv",
    "mean_minus_k_std": "submission_mean_minus_k_std.csv",
    "elite_maximin": "submission_elite_maximin.csv",
}


def _read_text_if_exists(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _build_full_instructions(task_description: str | None, data_description: str | None) -> str:
    parts: list[str] = []
    parts.append(
        "\n".join(
            [
                "You are running ML-Master on MLE-bench (native, no Docker).",
                "",
                "WORKSPACE CONVENTIONS (ML-MASTER):",
                "- Input files are available in `./input/` (copied from the competition folder).",
                "- Write your final predictions to `./submission/submission.csv`.",
                "- The submission format MUST match `./input/sample_submission.csv` exactly (columns and row count).",
                "",
            ]
        )
    )

    if task_description:
        parts.append("DATASET_SUBMIT TASK DESCRIPTION\n------\n\n" + task_description.strip() + "\n")
    if data_description:
        parts.append("DATASET_SUBMIT DATA DESCRIPTION\n------\n\n" + data_description.strip() + "\n")

    return "\n\n".join([p for p in parts if p])


def _find_desc_file(competition_dir: Path) -> Path:
    for candidate in [competition_dir / "description.md", competition_dir / "task_description.txt"]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No description file in {competition_dir}")


def _safe_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except Exception:
        link.write_text(str(target) + "\n", encoding="utf-8")


def run_mlmaster(
    competition_dir: Path, steps: int, run_dir: Path, *, desc_file: Path
) -> tuple[Path, Path, list[str], int]:
    """Run ML-Master and return (workspace_dir, log_dir, cmd, returncode)."""

    competition_id = competition_dir.name
    exp_name = f"mlebench-test-{competition_id}"

    top_workspace_dir = run_dir / "workspaces"
    top_log_dir = run_dir / "logs"
    top_workspace_dir.mkdir(parents=True, exist_ok=True)
    top_log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "/home/ka3094/ML-Master_submit/main_mcts.py",
        f"data_dir={competition_dir}",
        f"desc_file={desc_file}",
        f"agent.steps={steps}",
        "agent.k_fold_validation=5",
        f"exp_name={exp_name}",
        "start_cpu_id=0",
        "cpu_number=4",
        "agent.search.parallel_search_num=1",
        "agent.steerable_reasoning=false",
        "agent.code.model=gpt-4o-2024-08-06",
        "agent.code.base_url=https://api.openai.com/v1",
        "agent.code.api_key=${oc.env:OPENAI_API_KEY}",
        "agent.feedback.model=gpt-4o-2024-08-06",
        "agent.feedback.base_url=https://api.openai.com/v1",
        "agent.feedback.api_key=${oc.env:OPENAI_API_KEY}",
        "agent.search.use_bug_consultant=true",
        "agent.save_all_submission=true",
        "agent.check_format=false",
        f"workspace_dir={top_workspace_dir}",
        f"log_dir={top_log_dir}",
        # Post-search: ML-Master will export multiple submission variants into its log_dir.
        "post_search.enabled=true",
        "post_search.selection=elite_maximin",
        "post_search.top_k=50",
        "post_search.elite_top_k=3",
        "post_search.elite_ratio=0.05",
        "post_search.elite_k_std=2.0",
    ]

    (run_dir / "run_cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    print("=" * 60)
    print("Running ML-Master (native, no Docker)")
    print("=" * 60)
    print(f"Competition: {competition_id}")
    print(f"Steps: {steps}")
    print(f"Run dir: {run_dir}")
    print("Command:", " ".join(cmd))
    print()

    proc = subprocess.run(cmd, cwd="/home/ka3094/ML-Master_submit", check=False)

    workspace_dir = top_workspace_dir / exp_name
    log_dir = top_log_dir / exp_name
    return workspace_dir, log_dir, cmd, proc.returncode


def export_mlmaster_artifacts(run_dir: Path, run_log_dir: Path) -> dict[str, str]:
    """Copy/symlink key ML-Master artifacts into run_dir/mlmaster (AIDE-like)."""

    ml_dir = run_dir / "mlmaster"
    ml_dir.mkdir(parents=True, exist_ok=True)

    _safe_symlink(ml_dir / "log_dir", run_log_dir)

    copied: dict[str, str] = {}
    for name in [
        "tree_plot.html",
        "journal.json",
        "journal.jsonl",
        "final_selection.json",
        "config_mcts.yaml",
        "best_solution.py",
        "ml-master.log",
        "ml-master.verbose.log",
        *SUBMISSION_VARIANTS.values(),
    ]:
        src = run_log_dir / name
        if src.is_file():
            dst = ml_dir / name
            shutil.copy2(src, dst)
            copied[name] = str(dst)

    # AIDE uses `config.yaml`; keep an alias for convenience.
    cfg_src = run_log_dir / "config_mcts.yaml"
    if cfg_src.is_file():
        shutil.copy2(cfg_src, ml_dir / "config.yaml")
        copied["config.yaml"] = str(ml_dir / "config.yaml")

    for dname in ["solutions", "bug_consultant"]:
        src_dir = run_log_dir / dname
        if src_dir.is_dir():
            dst_dir = ml_dir / dname
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            copied[f"{dname}_dir"] = str(dst_dir)

    return copied


def collect_submission_variants(run_log_dir: Path, workspace_dir: Path) -> dict[str, Path]:
    variants: dict[str, Path] = {}
    for key, fname in SUBMISSION_VARIANTS.items():
        path = run_log_dir / fname
        if path.is_file():
            variants[key] = path

    # Fallback: if ML-Master didn't export selection variants, try the workspace best_submission.
    if not variants:
        best = workspace_dir / "best_submission" / "submission.csv"
        if best.is_file():
            variants["best_submission"] = best

    return variants


def pick_selected_variant(variants: dict[str, Path]) -> tuple[str, Path] | tuple[None, None]:
    for key in ["post_search", "raw", "elite_maximin", "mean_minus_k_std", "max_min", "best_submission"]:
        if key in variants:
            return key, variants[key]
    return None, None


def grade_variants(variants: dict[str, Path], competition_id: str, run_dir: Path) -> dict[str, dict]:
    grading_dir = run_dir / "grading"
    grading_dir.mkdir(parents=True, exist_ok=True)

    # Prepare MLE-bench data (offline/local mode).
    competition = None
    try:
        from mlebench.data import download_and_prepare_dataset
        from mlebench.registry import registry

        reg = registry.set_data_dir(MLEBENCH_DATA_DIR)
        competition = reg.get_competition(competition_id)
        download_and_prepare_dataset(
            competition=competition,
            keep_raw=True,
            overwrite_checksums=False,
            overwrite_leaderboard=False,
            skip_verification=True,
        )
    except Exception as e:
        print(f"WARNING: Failed to prepare MLE-bench data for grading: {e}")

    reports: dict[str, dict] = {}
    if competition is None:
        return reports

    from mlebench.grade import grade_csv

    for key, path in sorted(variants.items()):
        try:
            report = grade_csv(path, competition)
            reports[key] = report.to_dict()
            (grading_dir / f"grade_{key}.json").write_text(
                json.dumps(report.to_dict(), indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            reports[key] = {"error": f"{type(e).__name__}: {e}", "submission_path": str(path)}
            (grading_dir / f"grade_{key}.json").write_text(
                json.dumps(reports[key], indent=2, default=str) + "\n", encoding="utf-8"
            )

    return reports


def write_summary(
    run_dir: Path,
    *,
    competition_id: str,
    steps: int,
    selected_key: str | None,
    grading_reports: dict[str, dict],
) -> None:
    summary_path = run_dir / "SUMMARY.md"
    ml_dir = run_dir / "mlmaster"
    submission_dir = run_dir / "submission"
    grading_dir = run_dir / "grading"

    lines: list[str] = []
    lines.append("# ML-Master Native Run (no Docker)\n")
    lines.append(f"- Competition: `{competition_id}`\n")
    lines.append(f"- Steps: `{steps}`\n")
    lines.append(f"- Run dir: `{run_dir}`\n")
    if selected_key:
        lines.append(f"- Selected submission: `{selected_key}`\n")
    lines.append("\n## Key Artifacts\n")

    def add(path: Path) -> None:
        status = "✅" if path.exists() else "❌"
        lines.append(f"- {status} `{path}`\n")

    add(ml_dir / "tree_plot.html")
    add(ml_dir / "journal.json")
    add(ml_dir / "journal.jsonl")
    add(ml_dir / "final_selection.json")
    add(ml_dir / "solutions")
    add(submission_dir / "submission.csv")
    add(grading_dir)
    add(run_dir / "grading_reports.json")
    add(run_dir / "grading_report.json")

    lines.append("\n## Notes\n")
    lines.append(
        "- If grading `score` is `null`, the competition may ship placeholder private answers in this offline setup; the reports are still written.\n"
    )

    summary_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python run_native_with_post_search.py <competition_dir> <steps>")
        return 2

    competition_dir = Path(sys.argv[1])
    steps = int(sys.argv[2])

    if not competition_dir.is_dir():
        print(f"ERROR: Competition directory not found: {competition_dir}")
        return 2

    competition_id = competition_dir.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"/home/ka3094/mle-bench/native_runs/mlmaster_{competition_id}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=False)

    # Create an AIDE-like prompt file for reproducibility and easy inspection.
    task_description = _read_text_if_exists(competition_dir / "task_description.txt")
    data_description = _read_text_if_exists(competition_dir / "data_description.txt")
    full_desc_path = run_dir / "full_instructions.txt"
    full_desc_path.write_text(
        _build_full_instructions(task_description, data_description), encoding="utf-8"
    )

    # If the combined file is empty for some reason, fall back to the competition's own desc file.
    desc_file = full_desc_path if full_desc_path.stat().st_size > 0 else _find_desc_file(competition_dir)

    workspace_dir, log_dir, cmd, returncode = run_mlmaster(
        competition_dir, steps, run_dir, desc_file=desc_file
    )

    if returncode != 0:
        print(f"WARNING: ML-Master exited with code {returncode}")

    print("\n" + "=" * 60)
    print("Exporting artifacts")
    print("=" * 60)
    artifacts = export_mlmaster_artifacts(run_dir, log_dir)

    variants = collect_submission_variants(log_dir, workspace_dir)
    selected_key, selected_path = pick_selected_variant(variants)

    submission_dir = run_dir / "submission"
    code_dir = run_dir / "code"
    submission_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    # Copy submission variants into run_dir/submission (AIDE-like convenience).
    for key, path in variants.items():
        shutil.copy2(path, submission_dir / f"{key}.csv")

    if selected_key is not None and selected_path is not None:
        shutil.copy2(selected_path, submission_dir / "submission.csv")

    best_solution = log_dir / "best_solution.py"
    if best_solution.is_file():
        shutil.copy2(best_solution, code_dir / "best_solution.py")

    print("\n" + "=" * 60)
    print("Grading submissions")
    print("=" * 60)
    grading_reports = grade_variants(variants, competition_id, run_dir)

    if selected_key is not None and selected_key in grading_reports:
        (run_dir / "grading_report.json").write_text(
            json.dumps(
                {"selected_method": selected_key, **grading_reports[selected_key]},
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    (run_dir / "grading_reports.json").write_text(
        json.dumps(grading_reports, indent=2, default=str) + "\n", encoding="utf-8"
    )

    write_summary(
        run_dir,
        competition_id=competition_id,
        steps=steps,
        selected_key=selected_key,
        grading_reports=grading_reports,
    )

    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "workspace_dir": str(workspace_dir),
                "log_dir": str(log_dir),
                "submission_dir": str(submission_dir),
                "code_dir": str(code_dir),
                "selected_method": selected_key,
                "selected_submission": str(selected_path) if selected_path else None,
                "mlmaster_artifacts": artifacts,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("✅ Done")
    print("=" * 60)
    print(f"Run dir: {run_dir}")
    print(f"Logs: {log_dir}")
    if (run_dir / "mlmaster" / "tree_plot.html").is_file():
        print(f"Tree: {run_dir / 'mlmaster' / 'tree_plot.html'}")
    if (run_dir / "mlmaster" / "journal.json").is_file():
        print(f"Journal: {run_dir / 'mlmaster' / 'journal.json'}")
    if (submission_dir / "submission.csv").is_file():
        print(f"Submission: {submission_dir / 'submission.csv'}")
    if selected_key is not None:
        print(f"Selected method: {selected_key}")
        print(f"Score: {grading_reports.get(selected_key, {}).get('score')}")
    if grading_reports:
        print("Per-method scores:")
        for key, report in sorted(grading_reports.items()):
            print(f"  - {key}: {report.get('score', report.get('error', 'N/A'))}")

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
