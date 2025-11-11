#!/bin/bash
#
# RD-Agent start script written exactly like the agent-template instructions.
# 1. Activate the conda env.
# 2. Print the run context so logs make sense.
# 3. Call run_agent.py (with timeout if provided).
# 4. Make sure a submission file exists.

set -euo pipefail
set -x

cd "${AGENT_DIR}"

# Activate the shared "agent" environment used by every MLE-bench image.
eval "$(conda shell.bash hook)"
conda activate agent

# Determine the hardware string in the same way the template does.
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ', ' -)
else
  HARDWARE="CPU"
fi
export HARDWARE

# Optional sanity check: show whether PyTorch sees a GPU.
python -c "import torch; print('PyTorch GPU:', torch.cuda.is_available())" || true

# Convert the raw second limit into a human-friendly display.
TIME_LIMIT_SECS=${TIME_LIMIT_SECS:-0}
if [[ "${TIME_LIMIT_SECS}" =~ ^[0-9]+$ && "${TIME_LIMIT_SECS}" -gt 0 ]]; then
  export TIME_LIMIT_HOURS=$(((TIME_LIMIT_SECS + 3599) / 3600))
  DISPLAY_TIME_LIMIT="${TIME_LIMIT_SECS} seconds"
else
  export TIME_LIMIT_HOURS=${TIME_LIMIT_HOURS:-24}
  DISPLAY_TIME_LIMIT="unbounded"
fi

STEP_LIMIT_VALUE=${STEP_LIMIT:-}
if [[ -z "${STEP_LIMIT_VALUE}" ]]; then
  STEP_LIMIT_VALUE="unbounded"
fi

echo "====================================="
echo "RD-Agent Starting"
echo "Competition: ${COMPETITION_ID}"
echo "Time limit: ${DISPLAY_TIME_LIMIT}"
echo "Step limit: ${STEP_LIMIT_VALUE}"
echo "Hardware: ${HARDWARE}"
echo "====================================="

# The template always calls run_agent.py and lets it do the heavy lifting.
if [[ "${TIME_LIMIT_SECS}" =~ ^[0-9]+$ && "${TIME_LIMIT_SECS}" -gt 0 ]]; then
  timeout "${TIME_LIMIT_SECS}" python "${AGENT_DIR}/run_agent.py" "$@"
  EXIT_CODE=$?
  if [[ "${EXIT_CODE}" -eq 124 ]]; then
    echo "RD-Agent timed out after ${TIME_LIMIT_SECS} seconds"
    exit 124
  fi
else
  python "${AGENT_DIR}/run_agent.py" "$@"
  EXIT_CODE=$?
fi

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  echo "RD-Agent exited with status ${EXIT_CODE}"
  exit "${EXIT_CODE}"
fi

SUBMISSION_PATH="${SUBMISSION_DIR}/submission.csv"
if [[ ! -f "${SUBMISSION_PATH}" ]]; then
  echo "Expected submission at ${SUBMISSION_PATH} was not created."
  exit 1
fi

echo "RD-Agent completed successfully"
echo "Submission located at ${SUBMISSION_PATH}"
