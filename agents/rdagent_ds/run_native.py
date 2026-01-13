from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import subprocess
import shutil

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlebench.data import is_dataset_prepared
from mlebench.registry import Registry
from mlebench.utils import get_timestamp


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rdagent_repo_root() -> Path:
    # Expected sibling checkout: <parent>/RD-Agent
    return _repo_root().parent / "RD-Agent"


def _symlink_tree(src_dir: Path, dst_dir: Path, *, exclude_names: set[str] | None = None) -> None:
    exclude_names = exclude_names or set()
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        if child.name in exclude_names:
            continue
        target = dst_dir / child.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(child)


def _write_combined_description(dst_path: Path, competition_description: str) -> None:
    # RD-Agent reads this file as the "competition description". For native MLE-bench runs we must
    # provide local (non-Docker) file conventions to avoid writing to paths like `/mnt/output`.
    prefix = "\n".join(
        [
            "You are running RD-Agent on MLE-bench (native, no Docker).",
            "",
            "WORKSPACE CONVENTIONS (RD-AGENT / MLE-BENCH NATIVE):",
            "- Input files are available in `./workspace_input/`.",
            "- Write your final predictions to `./submission.csv` (in the current working directory).",
            "- The submission format MUST match `./workspace_input/sample_submission.csv` exactly (columns + row count).",
            "- Also write `./scores.csv` (in the current working directory). Use a single metric column and include an `ensemble` row.",
            "- Do NOT write outputs to absolute paths like `/mnt/output` or `/kaggle/working`.",
            "",
            "COMPETITION INSTRUCTIONS",
            "------",
            "",
        ]
    )
    dst_path.write_text(prefix + (competition_description or ""))


def _apply_variant_env(env: dict[str, str], variant: str) -> None:
    if variant == "baseline":
        return
    if variant == "post-search-on":
        env["DS_SOTA_EXP_SELECTOR_NAME"] = (
            "rdagent.scenarios.data_science.proposal.exp_gen.select.submit.CVFoldsRobustSOTASelector"
        )
        env.setdefault("DS_POST_SEARCH_SELECTION", "elite_maximin")
        return
    if variant == "plan-constraints":
        env["DS_PLAN_CONSTRAINTS_ENABLED"] = "true"
        env.setdefault("DS_PLAN_CONSTRAINTS_MAX_SENTENCES", "5")
        return
    if variant == "bug-consultant":
        env["DS_USE_BUG_CONSULTANT"] = "true"
        env.setdefault("DS_BUG_CONTEXT_MODE", "consultant")
        return
    raise ValueError(f"Unknown variant: {variant!r}")


def _redact_env(env: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for k, v in env.items():
        lk = k.lower()
        if any(s in lk for s in ["api_key", "apikey", "token", "secret", "password"]):
            redacted[k] = "***REDACTED***" if v else ""
        else:
            redacted[k] = v
    return redacted


def _find_conda_exe(env: dict[str, str]) -> str | None:
    conda_exe = env.get("CONDA_EXE") or shutil.which("conda")
    if conda_exe and Path(conda_exe).is_file():
        return conda_exe
    for candidate in [
        str(Path.home() / "miniconda3" / "bin" / "conda"),
        str(Path.home() / "anaconda3" / "bin" / "conda"),
    ]:
        if Path(candidate).is_file():
            return candidate
    return None


def _find_conda_env_python(conda_exe: str, env_name: str) -> Path | None:
    proc = subprocess.run([conda_exe, "env", "list"], capture_output=True, text=True, check=False)
    env_path: Path | None = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        name = parts[0].rstrip("*")
        if name != env_name:
            continue
        if len(parts) >= 2:
            env_path = Path(parts[1])
            break
    if not env_path:
        return None
    py = env_path / "bin" / "python"
    return py if py.is_file() else None


def _conda_env_exists(env_name: str) -> bool:
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda_exe:
        for candidate in [
            str(Path.home() / "miniconda3" / "bin" / "conda"),
            str(Path.home() / "anaconda3" / "bin" / "conda"),
        ]:
            if Path(candidate).is_file():
                conda_exe = candidate
                break
    if not conda_exe:
        return False
    proc = subprocess.run([conda_exe, "env", "list"], capture_output=True, text=True, check=False)
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: <name> <path> or <name>* <path>
        first = line.split()[0].rstrip("*")
        if first == env_name:
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description="Run RD-Agent DS loop on an MLEBench competition (no Docker).")
    p.add_argument("--competition-id", required=True)
    p.add_argument(
        "--variant",
        default="post-search-on",
        choices=["baseline", "post-search-on", "plan-constraints", "bug-consultant"],
        help="Which RD-Agent experiment variant to run.",
    )
    p.add_argument("--data-dir", default=None, help="MLEBench data dir (where <comp>/prepared/... lives).")
    p.add_argument(
        "--conda-env",
        default="aide",
        help="Conda env name to use for both this runner and RD-Agent LocalEnv (default: aide).",
    )
    p.add_argument("--loop-n", type=int, default=1)
    p.add_argument("--step-n", type=int, default=None)
    p.add_argument("--fast", action="store_true", help="Smoke-test mode (reduces coder/runner loops).")
    p.add_argument(
        "--no-reexec",
        action="store_true",
        help="Do not auto re-exec under the target conda env python if this interpreter lacks deps.",
    )
    p.add_argument("--out-dir", default=None, help="Output root (default: native_runs/rdagent_ds).")
    args = p.parse_args()

    repo_root = _repo_root()
    rdagent_root = _rdagent_repo_root()
    if not rdagent_root.is_dir():
        raise SystemExit(f"RD-Agent repo not found at {rdagent_root}. Expected a sibling checkout.")

    # Ensure we run under the conda env that has RD-Agent deps installed (unless explicitly disabled).
    if not args.no_reexec and os.environ.get("MLEBENCH_RDA_REEXEC") != "1":
        probe_env = os.environ.copy()
        conda_exe = _find_conda_exe(probe_env)
        if conda_exe:
            target_py = _find_conda_env_python(conda_exe, args.conda_env)
            if target_py and Path(sys.executable).resolve() != target_py.resolve():
                new_env = os.environ.copy()
                new_env["MLEBENCH_RDA_REEXEC"] = "1"
                cmd = [str(target_py), str(Path(__file__).resolve()), *sys.argv[1:], "--no-reexec"]
                proc = subprocess.run(cmd, cwd=str(repo_root), env=new_env, check=False)
                return proc.returncode

    registry = Registry()
    if args.data_dir is None:
        # In this repo, prepared datasets are typically under `data/competitions/<id>/prepared/...`.
        default_data_dir = repo_root / "data" / "competitions"
        if default_data_dir.is_dir():
            args.data_dir = str(default_data_dir)
    if args.data_dir:
        registry = registry.set_data_dir(Path(args.data_dir))

    competition = registry.get_competition(args.competition_id)
    if not is_dataset_prepared(competition):
        raise SystemExit(
            f"Dataset for `{competition.id}` is not prepared at {registry.get_data_dir()}.\n"
            f"Run: `mlebench prepare -c {competition.id} --data-dir {registry.get_data_dir()}`"
        )

    root = Path(args.out_dir) if args.out_dir else repo_root / "native_runs" / "rdagent_ds"
    root.mkdir(parents=True, exist_ok=True)

    run_dir = root / f"{get_timestamp()}_{competition.id}_{args.variant}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Build RD-Agent expected dataset layout: DS_LOCAL_DATA_PATH/<competition_id>/...
    ds_local_data_path = run_dir / "rd_data"
    ds_comp_dir = ds_local_data_path / competition.id
    ds_comp_dir.mkdir(parents=True, exist_ok=True)

    _symlink_tree(competition.public_dir, ds_comp_dir, exclude_names={"description.md"})
    _write_combined_description(ds_comp_dir / "description.md", competition.description)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{repo_root}:{rdagent_root}:{env.get('PYTHONPATH', '')}"
    env["COMPETITION_ID"] = competition.id
    env["MLEBENCH_DATA_DIR"] = str(registry.get_data_dir())
    bin_dir = repo_root / "bin"
    if bin_dir.is_dir():
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    conda_exe = _find_conda_exe(env)
    if conda_exe:
        env["CONDA_EXE"] = conda_exe
        env["PATH"] = f"{Path(conda_exe).parent}:{env.get('PATH', '')}"

    # For MLEBench we point RD-Agent at an on-disk dataset layout and avoid Kaggle download flows.
    env.setdefault("DS_SCEN", "rdagent.scenarios.data_science.scen.DataScienceScen")

    # No docker: force RD-Agent to run code via LocalEnv (conda).
    env.setdefault("DS_Coder_CoSTEER_ENV_TYPE", "conda")
    env.setdefault("DS_Runner_CoSTEER_ENV_TYPE", "conda")
    # Keep the executing conda env consistent across the controller (this runner) and LocalEnv.
    env["DS_Coder_CoSTEER_CONDA_ENV_NAME"] = args.conda_env
    env["DS_Runner_CoSTEER_CONDA_ENV_NAME"] = args.conda_env
    # Avoid stale cached command outputs when iterating/debugging locally.
    env.setdefault("ENABLE_CACHE", "false")

    if args.fast:
        env.setdefault("DS_CODER_MAX_LOOP", "1")
        env.setdefault("DS_RUNNER_MAX_LOOP", "1")

    if env.get("DS_Coder_CoSTEER_ENV_TYPE") == "conda" or env.get("DS_Runner_CoSTEER_ENV_TYPE") == "conda":
        if not _conda_env_exists(args.conda_env):
            cur_env = os.environ.get("CONDA_DEFAULT_ENV")
            hint = (
                f"Try: `conda create -n {args.conda_env} --clone {cur_env}`"
                if cur_env
                else f"Create one, e.g. `conda create -n {args.conda_env} python=3.11` and install RD-Agent deps."
            )
            raise SystemExit(
                f"RD-Agent DS loop is configured for LocalEnv (conda) and expects a conda env named `{args.conda_env}`.\n"
                + hint
            )

    env["DS_LOCAL_DATA_PATH"] = str(ds_local_data_path)
    env["LOG_TRACE_PATH"] = str(run_dir / "log")
    env["WORKSPACE_PATH"] = str(run_dir / "rd_workspace")

    env["SUBMISSION_DIR"] = str(run_dir / "submission")
    env["CODE_DIR"] = str(run_dir / "code")

    _apply_variant_env(env, args.variant)

    log_dir = Path(env["LOG_TRACE_PATH"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_env.json").write_text(json.dumps(_redact_env(env), indent=2, sort_keys=True) + "\n")

    # Run RD-Agent DS loop in-process (so VS Code can debug into RD-Agent code).
    # Important: apply env vars before importing `rdagent`, since settings are read at import time.
    os.environ.update(env)
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(rdagent_root))

    ds_cmd = [
        sys.executable,
        "-m",
        "rdagent.app.data_science.loop",
        "--competition",
        competition.id,
        "--loop_n",
        str(args.loop_n),
    ]
    if args.step_n is not None:
        ds_cmd += ["--step_n", str(args.step_n)]

    (run_dir / "run_cmd.txt").write_text(" ".join(ds_cmd) + "\n")

    console_log = (log_dir / "console.log").open("w")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for s in self._streams:
                try:
                    s.write(data)
                except Exception:
                    pass

        def flush(self):
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass

    try:
        sys.stdout = _Tee(orig_stdout, console_log)
        sys.stderr = _Tee(orig_stderr, console_log)
        from rdagent.app.data_science.loop import main as rdagent_ds_main

        rdagent_ds_main(competition=competition.id, loop_n=args.loop_n, step_n=args.step_n)
    except subprocess.CalledProcessError as e:
        (run_dir / "run_error.txt").write_text(str(e) + "\n")
    except Exception as e:
        (run_dir / "run_error.txt").write_text(repr(e) + "\n")
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        try:
            console_log.close()
        except Exception:
            pass

    # Extract auto-selected and per-strategy selections into run_dir/submission/*.
    extract_cmd = [sys.executable, str(repo_root / "agents" / "rdagent_ds" / "extract_selected_submissions.py")]
    subprocess.run(extract_cmd, cwd=str(repo_root), env=env, check=False)

    print(f"Run dir: {run_dir}")
    print(f"Auto submission: {run_dir / 'submission' / 'submission.csv'}")
    print(f"Selections: {run_dir / 'submission' / 'selections'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
