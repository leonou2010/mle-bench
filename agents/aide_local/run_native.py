from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlebench.data import is_dataset_prepared  # noqa: E402
from mlebench.grade import grade_csv  # noqa: E402
from mlebench.registry import Registry  # noqa: E402
from mlebench.utils import get_timestamp  # noqa: E402


def _repo_root() -> Path:
    return REPO_ROOT


def _aide_repo_root() -> Path:
    # Expected sibling checkout: <parent>/aideml_submit
    return _repo_root().parent / "aideml_submit"


def _build_task_description(
    competition_description: str, task_description: str | None, data_description: str | None
) -> str:
    parts: list[str] = []
    parts.append(
        "\n".join(
            [
                "You are running on MLE-bench (native, no Docker).",
                "",
                "WORKSPACE CONVENTIONS (AIDE):",
                "- Input files are available in `./input/` (copied from the competition public data).",
                "- Write your final predictions to `./working/submission.csv`.",
                "- The submission format MUST match `./input/sample_submission.csv` exactly (columns and row count).",
                "",
            ]
        )
    )

    if task_description:
        parts.append("DATASET_SUBMIT TASK DESCRIPTION\n------\n\n" + task_description.strip() + "\n")
    if data_description:
        parts.append("DATASET_SUBMIT DATA DESCRIPTION\n------\n\n" + data_description.strip() + "\n")

    if competition_description:
        parts.append("MLE-BENCH COMPETITION DESCRIPTION\n------\n\n" + competition_description.strip() + "\n")

    return "\n\n".join([p for p in parts if p])


def _latest_subdir(top_dir: Path) -> Path | None:
    if not top_dir.is_dir():
        return None
    candidates = [p for p in top_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _export_aide_artifacts(run_dir: Path, run_log_dir: Path) -> dict[str, str]:
    aide_dir = run_dir / "aide"
    aide_dir.mkdir(parents=True, exist_ok=True)

    # Convenience symlink to the full AIDE log directory.
    link = aide_dir / "log_dir"
    try:
        link.symlink_to(run_log_dir, target_is_directory=True)
    except Exception:
        (aide_dir / "log_dir_path.txt").write_text(str(run_log_dir) + "\n")

    copied: dict[str, str] = {}
    for name in [
        "tree_plot.html",
        "journal.json",
        "final_selection.json",
        "config.yaml",
        "best_solution.py",
        "submission_raw.csv",
        "submission_max_min.csv",
        "submission_mean_minus_k_std.csv",
        "submission_post_search.csv",
    ]:
        src = run_log_dir / name
        if src.is_file():
            dst = aide_dir / name
            shutil.copy2(src, dst)
            copied[name] = str(dst)

    solutions_src = run_log_dir / "solutions"
    if solutions_src.is_dir():
        solutions_dst = aide_dir / "solutions"
        shutil.copytree(solutions_src, solutions_dst, dirs_exist_ok=True)
        copied["solutions_dir"] = str(solutions_dst)

    return copied


def _pick_submission(run_log_dir: Path) -> Path | None:
    _, path = _pick_submission_with_method(run_log_dir)
    return path


def _collect_submissions(run_log_dir: Path) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    named = {
        "post_search": run_log_dir / "submission_post_search.csv",
        "raw": run_log_dir / "submission_raw.csv",
        "max_min": run_log_dir / "submission_max_min.csv",
        "mean_minus_k_std": run_log_dir / "submission_mean_minus_k_std.csv",
    }
    for method, path in named.items():
        if path.is_file():
            candidates[method] = path

    solutions_dir = run_log_dir / "solutions"
    if solutions_dir.is_dir():
        per_node = sorted(solutions_dir.glob("submission_node_*.csv"), key=lambda p: p.stat().st_mtime)
        if per_node:
            candidates["node_latest"] = per_node[-1]

    return candidates


def _pick_submission_with_method(run_log_dir: Path) -> tuple[str, Path] | tuple[None, None]:
    preferred = [
        ("post_search", run_log_dir / "submission_post_search.csv"),
        ("raw", run_log_dir / "submission_raw.csv"),
        ("max_min", run_log_dir / "submission_max_min.csv"),
        ("mean_minus_k_std", run_log_dir / "submission_mean_minus_k_std.csv"),
    ]
    for method, path in preferred:
        if path.is_file():
            return method, path

    # Fallback: newest per-node snapshot
    solutions_dir = run_log_dir / "solutions"
    if solutions_dir.is_dir():
        candidates = sorted(solutions_dir.glob("submission_node_*.csv"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return "node_latest", candidates[-1]
    return None, None


def _apply_variant_overrides(overrides: list[str], variant: str) -> None:
    # Start from a baseline where all 3 experiment features are OFF.
    overrides.extend(
        [
            "post_search.selection=best_valid",
            "plan_constraints.enabled=false",
            "agent.search.use_bug_consultant=false",
        ]
    )

    if variant == "post-search-on":
        overrides.extend(
            [
                "post_search.selection=elite_maximin",
                "post_search.top_k=50",
                "post_search.elite_top_k=3",
                "post_search.elite_ratio=0.05",
                "post_search.elite_k_std=2.0",
            ]
        )
        return

    if variant == "plan-constraints":
        overrides.extend(["plan_constraints.enabled=true", "plan_constraints.max_sentences=5"])
        return

    if variant == "bug-consultant":
        overrides.extend(
            [
                "agent.search.use_bug_consultant=true",
                "agent.search.bug_context_mode=consultant",
            ]
        )
        return

    raise ValueError(f"Unknown variant: {variant!r}")


def main() -> int:
    p = argparse.ArgumentParser(description="Run AIDE on an MLE-bench competition (no Docker).")
    p.add_argument("--competition-id", required=True)
    p.add_argument(
        "--variant",
        default="post-search-on",
        choices=["post-search-on", "plan-constraints", "bug-consultant"],
        help="Which AIDE experiment variant to run.",
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exec-timeout", type=int, default=3600)
    p.add_argument("--data-dir", default=None, help="MLEBench data dir (where <comp>/prepared/... lives).")
    p.add_argument("--out-dir", default=None, help="Output root (default: native_runs/aide_local).")
    args = p.parse_args()

    repo_root = _repo_root()

    registry = Registry()
    if args.data_dir:
        registry = registry.set_data_dir(Path(args.data_dir))

    competition = registry.get_competition(args.competition_id)
    if not is_dataset_prepared(competition):
        raise SystemExit(
            f"Dataset for `{competition.id}` is not prepared at {registry.get_data_dir()}.\n"
            f"Run: `./experiments/prepare_dataset_submit.sh` (or `mlebench prepare -c {competition.id} --data-dir {registry.get_data_dir()}`)"
        )

    root = Path(args.out_dir) if args.out_dir else repo_root / "native_runs" / "aide_local"
    root.mkdir(parents=True, exist_ok=True)

    run_dir = root / f"{get_timestamp()}_{competition.id}_{args.variant}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    task_description = None
    data_description = None
    if competition.raw_dir.is_dir():
        task_path = competition.raw_dir / "task_description.txt"
        if task_path.is_file():
            task_description = task_path.read_text()
        data_path = competition.raw_dir / "data_description.txt"
        if data_path.is_file():
            data_description = data_path.read_text()

    # Build an AIDE-friendly prompt file that matches the AIDE workspace contract.
    full_desc_path = run_dir / "full_instructions.txt"
    full_desc_path.write_text(
        _build_task_description(competition.description, task_description, data_description)
    )

    top_log_dir = run_dir / "logs"
    top_ws_dir = run_dir / "workspaces"
    top_log_dir.mkdir(parents=True, exist_ok=True)
    top_ws_dir.mkdir(parents=True, exist_ok=True)

    # Run AIDE (expects `aide` to be importable in this interpreter).
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["AIDE_SEED"] = str(args.seed)

    # Prefer the local fork if it exists.
    aide_root = _aide_repo_root()
    if aide_root.is_dir():
        env["PYTHONPATH"] = f"{aide_root}:{repo_root}:{env.get('PYTHONPATH', '')}"
    else:
        env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"

    exp_name = f"{competition.id}-{args.variant}-seed{args.seed}"

    overrides: list[str] = []
    _apply_variant_overrides(overrides, args.variant)

    cmd = [
        sys.executable,
        "-m",
        "aide.run",
        f"data_dir={competition.public_dir}",
        f"desc_file={full_desc_path}",
        f"exp_name={exp_name}",
        f"log_dir={top_log_dir}",
        f"workspace_dir={top_ws_dir}",
        f"agent.steps={args.steps}",
        "agent.k_fold_validation=5",
        "copy_data=false",
        "generate_report=false",
        f"exec.timeout={args.exec_timeout}",
        "agent.code.model=gpt-4o-2024-08-06",
        "agent.feedback.model=gpt-4o-2024-08-06",
        *overrides,
    ]
    (run_dir / "run_cmd.txt").write_text(" ".join(cmd) + "\n")

    proc = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)

    # Locate the produced log dir (AIDE creates an indexed subdir under log_dir).
    run_log_dir = _latest_subdir(top_log_dir)
    if run_log_dir is None:
        raise SystemExit(f"AIDE did not produce a run directory under {top_log_dir}")

    exported_aide_artifacts = _export_aide_artifacts(run_dir, run_log_dir)

    # Export in MLE-bench-style locations.
    submission_dir = run_dir / "submission"
    code_dir = run_dir / "code"
    submission_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    selected_method, submission_src = _pick_submission_with_method(run_log_dir)
    if selected_method is None or submission_src is None:
        raise SystemExit(f"No submission found under {run_log_dir}")
    (submission_dir / "submission.csv").write_bytes(submission_src.read_bytes())

    best_solution = run_log_dir / "best_solution.py"
    if best_solution.is_file():
        (code_dir / "best_solution.py").write_bytes(best_solution.read_bytes())

    # Grade all discovered submissions (post_search/raw/etc) + the selected one.
    grading_reports: dict[str, dict] = {}
    for method, path in sorted(_collect_submissions(run_log_dir).items()):
        try:
            report = grade_csv(path, competition)
            grading_reports[method] = report.to_dict()
        except Exception as e:
            grading_reports[method] = {"error": f"{type(e).__name__}: {e}"}

        # Mirror into run_dir for easy inspection.
        dst = submission_dir / f"{method}.csv"
        try:
            dst.write_bytes(path.read_bytes())
        except Exception:
            pass

    # Grade the selected submission (copied into submission/submission.csv).
    selected_report = grade_csv(submission_dir / "submission.csv", competition)
    (run_dir / "grading_report.json").write_text(
        json.dumps(
            {"selected_method": selected_method, **selected_report.to_dict()},
            indent=2,
            default=str,
        )
    )
    (run_dir / "grading_reports.json").write_text(json.dumps(grading_reports, indent=2, default=str))

    artifacts = {
        "run_dir": str(run_dir),
        "aide_log_dir": str(run_log_dir),
        "aide_artifacts": exported_aide_artifacts,
        "submission_dir": str(submission_dir),
        "code_dir": str(code_dir),
        "grading_report": str(run_dir / "grading_report.json"),
        "grading_reports": str(run_dir / "grading_reports.json"),
    }
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2, default=str))

    print(f"Run dir: {run_dir}")
    print(f"AIDE log dir: {run_log_dir}")
    if (run_dir / "aide" / "tree_plot.html").is_file():
        print(f"AIDE tree: {run_dir / 'aide' / 'tree_plot.html'}")
    if (run_dir / "aide" / "journal.json").is_file():
        print(f"AIDE journal: {run_dir / 'aide' / 'journal.json'}")
    print(f"Submission: {submission_dir / 'submission.csv'}")
    print(f"Selected method: {selected_method}")
    print(f"Score: {selected_report.score}")
    if grading_reports:
        print("Per-method scores:")
        for method, report in grading_reports.items():
            if "score" in report:
                print(f"  - {method}: {report['score']}")
            else:
                print(f"  - {method}: {report.get('error', 'unknown error')}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
