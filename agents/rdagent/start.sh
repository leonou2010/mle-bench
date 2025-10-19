#!/bin/bash
set -x # Print commands and their arguments as they are executed

cd ${AGENT_DIR}

eval "$(conda shell.bash hook)" # make conda available to the shell
conda activate agent

# Determine hardware available
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
else
  HARDWARE="a CPU"
fi
export HARDWARE

# Check GPU availability
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')" || true

# Convert $TIME_LIMIT_SECS to hours for RD-Agent (ceil to ensure at least 1 when limit < 1h)
if [ "${TIME_LIMIT_SECS:-0}" -gt 0 ]; then
  export TIME_LIMIT_HOURS=$(((TIME_LIMIT_SECS + 3599) / 3600))
else
  export TIME_LIMIT_HOURS=24
fi

# Mirror competition data into a writable location for RD-Agent
DATA_MIRROR_ROOT=${AGENT_DIR}/runtime/data
DATA_MIRROR_PATH=${DATA_MIRROR_ROOT}/${COMPETITION_ID}
mkdir -p "${DATA_MIRROR_PATH}"
if [ -d /home/data ]; then
  find "${DATA_MIRROR_PATH}" -mindepth 1 -delete 2>/dev/null || true
  cp -a /home/data/. "${DATA_MIRROR_PATH}"/
fi

# Create .env file for RD-Agent in agent directory
cat > ${AGENT_DIR}/.env << EOF
# LLM Configuration
CHAT_MODEL=${CHAT_MODEL:-gpt-4o}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-text-embedding-3-small}
OPENAI_API_KEY=${OPENAI_API_KEY}

# Data Science Scenario Configuration
DS_LOCAL_DATA_PATH=${DATA_MIRROR_ROOT}
DS_CODER_ON_WHOLE_PIPELINE=True
DS_IF_USING_MLE_DATA=True
DS_SAMPLE_DATA_BY_LLM=False
DS_SCEN=rdagent.scenarios.data_science.scen.DataScienceScen
DS_USE_RAW_DESCRIPTION=True

# RD-Agent runtime configuration
RD_AGENT_WORKSPACE_PATH=${AGENT_DIR}/runtime/workspace
RD_AGENT_LOG_PATH=${AGENT_DIR}/runtime/logs
LOG_TRACE_PATH=${AGENT_DIR}/runtime/logs

# MLE-bench specific paths
COMPETITION_ID=${COMPETITION_ID}
MLE_BENCH_ROOT=/home
EOF

# Create RD-Agent runtime directories
mkdir -p ${AGENT_DIR}/runtime/workspace
mkdir -p ${AGENT_DIR}/runtime/logs

# Create a Python script to run RD-Agent with MLE-bench format
cat > ${AGENT_DIR}/run_rdagent_mle.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
RD-Agent runner for MLE-bench integration.

This script wires RD-Agent's data science loop into the MLE-bench runtime by:
1. Generating MLE-bench formatted instructions for the target competition
2. Running the full RD-Agent research & development loop
3. Exporting the best submission, code artifacts, and logs back to the host
"""

import asyncio
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, '/opt/rdagent')

from dotenv import load_dotenv

load_dotenv('/home/agent/.env', override=True)

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.log.conf import LOG_SETTINGS
from rdagent.log.timer import RD_Agent_TIMER_wrapper
from rdagent.scenarios.data_science.loop import DataScienceRDLoop
from rdagent.scenarios.data_science.scen import DataScienceScen


SUBMISSION_DEST = Path('/home/submission/submission.csv')
CODE_EXPORT_ROOT = Path('/home/code')
LOG_EXPORT_ROOT = Path('/home/logs')
WORKSPACE_ROOT = Path(os.environ.get('RD_AGENT_WORKSPACE_PATH', '/home/agent/runtime/workspace'))
LOG_ROOT = Path(os.environ.get('RD_AGENT_LOG_PATH', '/home/agent/runtime/logs'))
MLE_DESCRIPTION_CACHE: Dict[str, str] = {}
_ORIGINAL_GET_DESCRIPTION = DataScienceScen._get_description


def _patched_get_description(self: DataScienceScen) -> str:
    competition = getattr(self, 'competition', None)
    if competition and competition in MLE_DESCRIPTION_CACHE:
        logger.info(f"Using cached MLE-bench description for competition {competition}")
        return MLE_DESCRIPTION_CACHE[competition]
    return _ORIGINAL_GET_DESCRIPTION(self)


def _install_description_patch() -> None:
    if DataScienceScen._get_description is not _patched_get_description:  # type: ignore[comparison-overlap]
        DataScienceScen._get_description = _patched_get_description  # type: ignore[assignment]


def _ensure_runtime_paths() -> None:
    SUBMISSION_DEST.parent.mkdir(parents=True, exist_ok=True)
    CODE_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)


def _configure_settings() -> None:
    RD_AGENT_SETTINGS.workspace_path = WORKSPACE_ROOT
    timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    run_trace_path = LOG_ROOT / timestamp
    LOG_SETTINGS.trace_path = str(run_trace_path)
    logger.info(f"Workspace path set to {RD_AGENT_SETTINGS.workspace_path}")
    logger.info(f"Log trace path set to {LOG_SETTINGS.trace_path}")


def _copy_file(src: Path, dest: Path) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return True
    except Exception as exc:
        logger.warning(f"Failed to copy {src} to {dest}: {exc}")
        return False


def _write_text(dest: Path, content: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)


def _export_workspace(loop: Optional[DataScienceRDLoop]) -> Optional[Path]:
    if loop is None or getattr(loop, 'trace', None) is None:
        return None
    sota_exp = getattr(loop.trace, 'sota_exp_to_submit', None)
    if sota_exp is None:
        logger.warning('No SOTA experiment available to export workspace')
        return None
    workspace = getattr(sota_exp, 'experiment_workspace', None)
    if workspace is None:
        logger.warning('SOTA experiment lacks an experiment workspace')
        return None

    target_dir = CODE_EXPORT_ROOT / 'best_experiment'
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for relative_path, content in workspace.file_dict.items():
        dest = target_dir / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)

    for filename in ('scores.csv', 'submission.csv'):
        source = Path(workspace.workspace_path) / filename
        if source.exists():
            _copy_file(source, target_dir / filename)

    logger.info(f"Exported workspace artifacts to {target_dir}")
    return Path(workspace.workspace_path)


def _export_logs() -> None:
    trace_path = Path(LOG_SETTINGS.trace_path)
    if not trace_path.exists():
        logger.warning(f"Trace path {trace_path} does not exist; skipping log export")
        return
    dest = LOG_EXPORT_ROOT / trace_path.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(trace_path, dest)
    logger.info(f"Copied logs to {dest}")


def _extract_scores(workspace_path: Optional[Path]) -> Optional[dict]:
    if workspace_path is None:
        return None
    scores_path = workspace_path / 'scores.csv'
    if not scores_path.exists():
        return None
    try:
        with scores_path.open('r') as handle:
            lines = handle.read().strip().splitlines()
        return {'scores_csv_preview': lines[:10]}
    except Exception as exc:
        logger.warning(f"Failed to read scores.csv for metadata: {exc}")
        return None


def _export_submission(loop: Optional[DataScienceRDLoop]) -> bool:
    submission_src = None
    if loop and getattr(loop, 'trace', None):
        sota_exp = getattr(loop.trace, 'sota_exp_to_submit', None)
        if sota_exp is not None and getattr(sota_exp, 'experiment_workspace', None):
            candidate = Path(sota_exp.experiment_workspace.workspace_path) / 'submission.csv'
            if candidate.exists():
                submission_src = candidate

    if submission_src and submission_src.exists():
        copied = _copy_file(submission_src, SUBMISSION_DEST)
        if copied:
            logger.info(f"Submission copied from {submission_src} to {SUBMISSION_DEST}")
            return True

    logger.error('Submission file missing after RD-Agent run; writing placeholder submission')
    _write_text(SUBMISSION_DEST, 'id,target\n0,0\n')
    return False


def _write_metadata(loop: Optional[DataScienceRDLoop], submission_ready: bool, workspace_path: Optional[Path]) -> None:
    metadata = {
        'competition_id': DS_RD_SETTING.competition,
        'submission_ready': submission_ready,
        'time_limit_hours': int(os.environ.get('TIME_LIMIT_HOURS', 24)),
        'step_limit': int(os.environ.get('STEP_LIMIT', 500)),
        'hardware': os.environ.get('HARDWARE', 'a GPU'),
    }

    if loop and getattr(loop, 'trace', None):
        sota_exp = getattr(loop.trace, 'sota_exp_to_submit', None)
        if sota_exp is not None:
            result = getattr(sota_exp, 'result', None)
            if result is not None:
                try:
                    if hasattr(result, 'to_dict'):
                        metadata['best_result'] = result.to_dict()
                    else:
                        metadata['best_result'] = str(result)
                except Exception as exc:
                    logger.warning(f"Failed to serialize result for metadata: {exc}")
            metadata['loop_iterations'] = getattr(loop, 'loop_idx', None)

    scores_preview = _extract_scores(workspace_path)
    if scores_preview:
        metadata.update(scores_preview)

    metadata_path = LOG_EXPORT_ROOT / 'run_summary.json'
    try:
        metadata_path.write_text(json.dumps(metadata, indent=2))
    except Exception as exc:
        logger.warning(f"Failed to write run metadata: {exc}")


def _compute_mle_description(competition_id: str, hardware: str, time_limit: int, step_limit: int) -> str:
    scen = DataScienceScen(competition=competition_id)
    try:
        description = scen._get_description_mle_format(
            hardware=hardware,
            time_limit=time_limit,
            step_limit=step_limit,
        )
        logger.info(f"Generated MLE-bench description ({len(description)} chars)")
        return description
    except Exception as exc:
        logger.warning(f"Fell back to standard description: {exc}")
        return scen.raw_description


async def _run_loop(step_limit: int, time_limit_hours: int) -> DataScienceRDLoop:
    loop = DataScienceRDLoop(DS_RD_SETTING)
    all_duration = str(max(time_limit_hours, 0) * 3600)
    if time_limit_hours > 0:
        RD_Agent_TIMER_wrapper.timer.reset(all_duration=all_duration)
    await loop.run(step_n=step_limit or None, all_duration=all_duration)
    return loop


def main() -> int:
    _ensure_runtime_paths()

    competition_id = os.environ.get('COMPETITION_ID')
    if not competition_id:
        logger.error('COMPETITION_ID must be provided')
        return 1

    DS_RD_SETTING.competition = competition_id
    DS_RD_SETTING.use_raw_description = True
    data_root = Path(os.environ.get('DS_LOCAL_DATA_PATH', '/home/data')).resolve()
    try:
        DS_RD_SETTING.local_data_path = str(data_root)
    except Exception as exc:
        logger.warning(f"Failed to set DS_RD_SETTING.local_data_path directly: {exc}")
    os.environ['DS_LOCAL_DATA_PATH'] = str(data_root)
    logger.info(f"Using data root: {data_root}")
    logger.info(f"Runtime DS local path: {DS_RD_SETTING.local_data_path}")

    hardware = os.environ.get('HARDWARE', 'a GPU')
    time_limit_hours = int(os.environ.get('TIME_LIMIT_HOURS', 24))
    step_limit = int(os.environ.get('STEP_LIMIT', 500))

    logger.info(f"Configured run for competition={competition_id} time_limit_hours={time_limit_hours} step_limit={step_limit}")

    _configure_settings()

    mle_description = _compute_mle_description(competition_id, hardware, time_limit_hours, step_limit)
    MLE_DESCRIPTION_CACHE[competition_id] = mle_description
    _install_description_patch()

    loop: Optional[DataScienceRDLoop] = None
    exit_code = 0
    try:
        loop = asyncio.run(_run_loop(step_limit, time_limit_hours))
        logger.info('RD-Agent loop completed')
    except Exception as exc:
        logger.error(f"RD-Agent loop failed: {exc}")
        traceback.print_exc()
        exit_code = 1

    workspace_path = _export_workspace(loop)
    submission_ready = _export_submission(loop)
    _export_logs()
    _write_metadata(loop, submission_ready, workspace_path)

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
PYTHON_SCRIPT

chmod +x ${AGENT_DIR}/run_rdagent_mle.py

# Run RD-Agent with timeout
echo "Starting RD-Agent with MLE-bench integration..."
echo "Competition: ${COMPETITION_ID}"
echo "Time limit: ${TIME_LIMIT_HOURS} hours (${TIME_LIMIT_SECS} seconds)"
echo "Step limit: ${STEP_LIMIT}"
echo "Hardware: ${HARDWARE}"

cd ${AGENT_DIR}
timeout $TIME_LIMIT_SECS python run_rdagent_mle.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
  echo "RD-Agent timed out after ${TIME_LIMIT_HOURS} hours"
  exit 124
elif [ $EXIT_CODE -ne 0 ]; then
  echo "RD-Agent exited with error code $EXIT_CODE"
  exit $EXIT_CODE
else
  echo "RD-Agent completed successfully"
  exit 0
fi
