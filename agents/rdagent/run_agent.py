#!/usr/bin/env python3
"""
RD-Agent launcher written in the same style as the agent-template.

The flow mirrors the simple template:
    1. Collect context (env vars and important paths).
    2. Make sure every output directory exists.
    3. Prepare the data that RD-Agent expects (mirrored copy, instructions).
    4. Run the agent loop.
    5. Save submission, logs, code snapshots.

Only the internals differ because RD-Agent ships as a library rather than a
single CLI. Replace the sections noted below if you need to customize the
behaviour further, but the overall shape should feel familiar.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# RD-Agent lives in /opt/rdagent inside the container. When debugging locally
# we fall back to the repository copy so imports keep working.
DEFAULT_RDAGENT_PATH = Path("/opt/rdagent")
if DEFAULT_RDAGENT_PATH.exists():
    sys.path.insert(0, str(DEFAULT_RDAGENT_PATH))
else:
    repo_rdagent = Path(__file__).resolve().parents[3] / "RD-Agent"
    if repo_rdagent.exists():
        sys.path.insert(0, str(repo_rdagent))

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.log.conf import LOG_SETTINGS
from rdagent.log.timer import RD_Agent_TIMER_wrapper
from rdagent.scenarios.data_science.loop import DataScienceRDLoop
from rdagent.scenarios.data_science.scen import DataScienceScen

logging.basicConfig(level=logging.INFO)

# Cache the description builder so we do not recompute prompts repeatedly.
MLE_DESCRIPTION_CACHE: Dict[str, str] = {}
_ORIGINAL_GET_DESCRIPTION = DataScienceScen._get_description
_INSTRUCTIONS_PATH: Optional[Path] = None


@dataclass
class RuntimePaths:
    """Important folders exposed by MLE-bench plus RD-Agent specifics."""

    agent_root: Path
    data_source: Path
    data_mirror_root: Path
    submission_path: Path
    code_dir: Path
    logs_dir: Path
    workspace_dir: Path
    trace_dir: Path


@dataclass
class RuntimeContext:
    """All configuration we need to run RD-Agent once."""

    competition_id: str
    step_limit: int
    time_limit_secs: int
    time_limit_hours: int
    hardware: str
    paths: RuntimePaths


# ---------------------------------------------------------------------------
# Step 1: Read the environment and build the RuntimeContext.
# ---------------------------------------------------------------------------


def gather_context() -> RuntimeContext:
    """Translate environment variables into a strongly typed context object."""

    competition_id = os.environ.get("COMPETITION_ID")
    if not competition_id:
        raise RuntimeError("COMPETITION_ID must be provided by MLE-bench.")

    step_limit = int(os.environ.get("STEP_LIMIT", "500") or 500)
    time_limit_secs = int(os.environ.get("TIME_LIMIT_SECS", "0") or 0)
    time_limit_hours = int(os.environ.get("TIME_LIMIT_HOURS", "0") or 0)
    if time_limit_hours <= 0 and time_limit_secs > 0:
        time_limit_hours = max(1, (time_limit_secs + 3599) // 3600)
    hardware = os.environ.get("HARDWARE", "CPU")

    agent_root = Path(os.environ.get("AGENT_DIR", "/home/agent")).resolve()

    # Read paths from environment or use Docker defaults
    submission_path = Path(os.environ.get("SUBMISSION_PATH", "/home/submission/submission.csv"))
    code_dir = Path(os.environ.get("CODE_DIR", "/home/code"))
    logs_dir = Path(os.environ.get("LOG_DIR", "/home/logs"))

    paths = RuntimePaths(
        agent_root=agent_root,
        data_source=Path(os.environ.get("DATA_DIR", "/home/data")).resolve(),
        data_mirror_root=(agent_root / "runtime" / "data"),
        submission_path=submission_path,
        code_dir=code_dir,
        logs_dir=logs_dir,
        workspace_dir=(agent_root / "runtime" / "workspace"),
        trace_dir=(agent_root / "runtime" / "logs"),
    )

    return RuntimeContext(
        competition_id=competition_id,
        step_limit=step_limit,
        time_limit_secs=time_limit_secs,
        time_limit_hours=time_limit_hours,
        hardware=hardware,
        paths=paths,
    )


# ---------------------------------------------------------------------------
# Step 2: Prepare filesystem layout.
# ---------------------------------------------------------------------------


def ensure_directories(context: RuntimeContext) -> None:
    """Replicate the template behaviour: create every directory we will write."""

    paths = context.paths
    for folder in (
        paths.submission_path.parent,
        paths.code_dir,
        paths.logs_dir,
        paths.workspace_dir,
        paths.trace_dir,
        paths.data_mirror_root,
    ):
        folder.mkdir(parents=True, exist_ok=True)


def mirror_competition_data(context: RuntimeContext) -> Path:
    """
    RD-Agent expects a writable data directory. We copy /home/data into
    AGENT_DIR/runtime/data/<competition>.
    """
    source = context.paths.data_source
    target = context.paths.data_mirror_root / context.competition_id

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    if source.exists():
        shutil.copytree(source, target, dirs_exist_ok=True)
        logger.info(f"Mirrored data from {source} to {target}")
    else:
        logger.warning(f"Competition data directory {source} is missing.")

    return target


# ---------------------------------------------------------------------------
# Step 3: Configure RD-Agent to use the mirrored paths and prompts.
# ---------------------------------------------------------------------------


def configure_rd_agent(context: RuntimeContext, data_root: Path) -> None:
    """Update global RD-Agent settings so it operates inside our sandbox."""

    paths = context.paths

    RD_AGENT_SETTINGS.workspace_path = paths.workspace_dir
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_SETTINGS.trace_path = str(paths.trace_dir / timestamp)

    DS_RD_SETTING.competition = context.competition_id
    # Point to the mirror ROOT so RD-Agent finds f"{local_data_path}/{competition}"
    DS_RD_SETTING.local_data_path = str(paths.data_mirror_root)
    DS_RD_SETTING.use_raw_description = True

    os.environ["DS_LOCAL_DATA_PATH"] = str(paths.data_mirror_root)
    os.environ["RD_AGENT_WORKSPACE_PATH"] = str(paths.workspace_dir)
    os.environ["RD_AGENT_LOG_PATH"] = str(paths.trace_dir)


def discover_instructions_path(context: Optional[RuntimeContext] = None) -> Optional[Path]:
    """Locate the MLE-bench instructions file without hardcoding repository paths."""
    candidates: list[Path] = []

    if env_instructions := os.environ.get("MLE_BENCH_INSTRUCTIONS"):
        candidates.append(Path(env_instructions).expanduser())

    if env_root := os.environ.get("MLE_BENCH_ROOT"):
        candidates.append(Path(env_root).expanduser() / "environment" / "instructions.txt")

    if context:
        agent_root = context.paths.agent_root.resolve()
        candidates.append(agent_root / "environment" / "instructions.txt")
        for parent in agent_root.parents:
            candidates.append(parent / "environment" / "instructions.txt")

    script_path = Path(__file__).resolve()
    candidates.append(script_path.parent / "environment" / "instructions.txt")
    for parent in script_path.parents:
        candidates.append(parent / "environment" / "instructions.txt")

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return candidate
    return None


def initialize_instructions_path(context: RuntimeContext) -> None:
    """Compute and cache the instructions path for patched description builder."""
    global _INSTRUCTIONS_PATH
    _INSTRUCTIONS_PATH = discover_instructions_path(context)


def _get_cached_instructions_path() -> Optional[Path]:
    """Return a cached instructions path or attempt to rediscover one."""
    global _INSTRUCTIONS_PATH
    if _INSTRUCTIONS_PATH and _INSTRUCTIONS_PATH.exists():
        return _INSTRUCTIONS_PATH
    rediscovered = discover_instructions_path()
    if rediscovered:
        _INSTRUCTIONS_PATH = rediscovered
    return rediscovered


def _stringify_description(data: Any) -> str:
    """Convert the competition description into a printable string."""
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, indent=2)
    except TypeError:
        return str(data)


def _render_additional_notes(hardware: str, time_limit: int, step_limit: int) -> str:
    """Construct the standard MLE-bench additional notes section."""
    return f"""ADDITIONAL NOTES
------

- **Compute**: You have access to {hardware} with the appropriate drivers installed.
- **Total Runtime**: You have a maximum of {time_limit} hours to attempt this task. You may iteratively develop your solution within this time frame, but bear in mind that program runtime also counts towards this limit.
- **Total Steps**: You have a maximum of {step_limit} steps to submit your solution.
- You will be informed of the remaining time and steps at each step.
- **Use as much of the available time as possible to refine and optimize your submission**: Your goal is to achieve the best possible score, not just to submit your first working solution. Consider strategies like validating your answer using the "Validate Submission" tool at the http://localhost:5000/validate endpoint, manually spot-checking predictions, building custom validation sets and grading functions, and comparing different algorithms. You may keep working until the time limit expires -- as long as your best submission is stored at the designated location at the end of your run, it will be picked up for grading.

"""


def patch_description_builder() -> None:
    """
    Replace RD-Agent's description lookup with a cache-aware version so the
    expensive prompt conversion runs once per competition.
    """

    def _patched_get_description(self: DataScienceScen) -> str:  # type: ignore[override]
        competition = getattr(self, "competition", None)
        if competition and competition in MLE_DESCRIPTION_CACHE:
            logger.info(f"Using cached description for {competition}")
            return MLE_DESCRIPTION_CACHE[competition]
        return _ORIGINAL_GET_DESCRIPTION(self)  # type: ignore[misc]

    if DataScienceScen._get_description is not _patched_get_description:  # type: ignore[comparison-overlap]
        DataScienceScen._get_description = _patched_get_description  # type: ignore[assignment]


def build_mle_description(context: RuntimeContext) -> str:
    """Construct the full MLE-bench description without relying on RD-Agent internals."""
    instructions_path = _get_cached_instructions_path()
    scen = DataScienceScen(competition=context.competition_id)
    if not instructions_path or not instructions_path.exists():
        logger.warning("Unable to locate MLE-bench instructions, using standard format")
        return _stringify_description(scen.raw_description)

    base_instructions = instructions_path.read_text()
    additional_notes = _render_additional_notes(
        hardware=context.hardware,
        time_limit=context.time_limit_hours,
        step_limit=context.step_limit,
    )

    comp_desc_path = Path(f"{DS_RD_SETTING.local_data_path}/{context.competition_id}/description.md")
    if comp_desc_path.exists():
        competition_description = comp_desc_path.read_text()
    else:
        logger.warning(f"Competition description not found at {comp_desc_path}, using fallback")
        competition_description = _stringify_description(scen.raw_description)

    full_description = f"""{base_instructions}

{additional_notes}

COMPETITION INSTRUCTIONS
------

{competition_description}
"""
    logger.info(f"Generated MLE-bench description ({len(full_description)} chars)")
    return full_description


# ---------------------------------------------------------------------------
# Step 4: Run RD-Agent's main loop.
# ---------------------------------------------------------------------------


async def run_rd_loop(context: RuntimeContext) -> DataScienceRDLoop:
    """Execute the RD-Agent data-science loop with time/step budgets."""
    loop = DataScienceRDLoop(DS_RD_SETTING)
    total_seconds = str(max(context.time_limit_hours, 0) * 3600)
    if context.time_limit_hours > 0:
        RD_Agent_TIMER_wrapper.timer.reset(all_duration=total_seconds)
    await loop.run(step_n=context.step_limit or None, all_duration=total_seconds)
    return loop


# ---------------------------------------------------------------------------
# Step 5: Export results (submission, code, logs, metadata).
# ---------------------------------------------------------------------------


def export_submission(loop: Optional[DataScienceRDLoop], context: RuntimeContext) -> bool:
    """Copy the best submission produced by RD-Agent, or fall back to dummy."""
    submission_path = context.paths.submission_path
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    candidate: Optional[Path] = None
    if loop and getattr(loop, "trace", None):
        sota = getattr(loop.trace, "sota_exp_to_submit", None)
        if sota and getattr(sota, "experiment_workspace", None):
            candidate = Path(sota.experiment_workspace.workspace_path) / "submission.csv"

    if candidate and candidate.exists():
        shutil.copy2(candidate, submission_path)
        logger.info(f"Submission copied from {candidate} to {submission_path}")
        return True

    submission_path.write_text("id,target\n0,0\n")
    logger.error("RD-Agent did not produce a submission; wrote placeholder.")
    return False


def export_workspace(loop: Optional[DataScienceRDLoop], context: RuntimeContext) -> Optional[Path]:
    """Grab RD-Agent's best experiment files for inspection."""
    if not loop or not getattr(loop, "trace", None):
        return None
    sota = getattr(loop.trace, "sota_exp_to_submit", None)
    if not sota or not getattr(sota, "experiment_workspace", None):
        return None

    workspace = Path(sota.experiment_workspace.workspace_path)
    target_dir = context.paths.code_dir / "best_experiment"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for relative_path, content in sota.experiment_workspace.file_dict.items():
        file_path = target_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    for name in ("scores.csv", "submission.csv"):
        src = workspace / name
        if src.exists():
            shutil.copy2(src, target_dir / name)

    return workspace


def export_logs(context: RuntimeContext) -> None:
    """Copy RD-Agent trace logs into /home/logs so MLE-bench captures them."""
    trace_path = Path(LOG_SETTINGS.trace_path)
    if not trace_path.exists():
        logger.warning(f"Trace path {trace_path} is missing; skipping log export.")
        return

    destination = context.paths.logs_dir / trace_path.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(trace_path, destination)


def write_run_metadata(
    context: RuntimeContext,
    loop: Optional[DataScienceRDLoop],
    submission_ready: bool,
    workspace_path: Optional[Path],
) -> None:
    """Store a short JSON summary next to the logs."""
    metadata = {
        "competition_id": context.competition_id,
        "submission_ready": submission_ready,
        "time_limit_hours": context.time_limit_hours,
        "step_limit": context.step_limit,
        "hardware": context.hardware,
    }

    if loop and getattr(loop, "trace", None):
        metadata["loop_iterations"] = getattr(loop, "loop_idx", None)
        sota = getattr(loop.trace, "sota_exp_to_submit", None)
        if sota and getattr(sota, "result", None):
            result = sota.result
            try:
                metadata["best_result"] = result.to_dict() if hasattr(result, "to_dict") else str(result)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Could not serialize RD-Agent result: {exc}")

    if workspace_path:
        scores = workspace_path / "scores.csv"
        if scores.exists():
            try:
                metadata["scores_preview"] = scores.read_text().splitlines()[:10]
            except Exception:  # pragma: no cover
                pass

    context.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = context.paths.logs_dir / "run_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


# ---------------------------------------------------------------------------
# Main entrypoint: glue the steps together exactly like the template.
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        context = gather_context()
    except Exception as exc:
        logger.error(f"Invalid runtime context: {exc}")
        return 1

    ensure_directories(context)
    data_root = mirror_competition_data(context)
    configure_rd_agent(context, data_root)
    initialize_instructions_path(context)
    patch_description_builder()
    description = build_mle_description(context)
    MLE_DESCRIPTION_CACHE[context.competition_id] = description

    loop: Optional[DataScienceRDLoop] = None
    exit_code = 0
    try:
        loop = asyncio.run(run_rd_loop(context))
        logger.info("RD-Agent loop completed.")
    except Exception as exc:  # pragma: no cover - runtime failure
        logger.error(f"RD-Agent loop failed: {exc}")
        traceback.print_exc()
        exit_code = 1

    workspace = export_workspace(loop, context)
    submission_ready = export_submission(loop, context)
    export_logs(context)
    write_run_metadata(context, loop, submission_ready, workspace)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
