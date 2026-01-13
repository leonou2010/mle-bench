#!/bin/bash
set -euo pipefail
set -x # Print commands and their arguments as they are executed

cd ${AGENT_DIR}

eval "$(conda shell.bash hook)" # make conda available to the shell
conda activate agent

# determine hardware available
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
# check that we can use the GPU in PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')"
# check that we can use the GPU in TensorFlow
python -c "import tensorflow as tf; print('GPUs Available: ', tf.config.list_physical_devices('GPU'))"

# convert $TIME_LIMIT_SECS to more readable format for prompt
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)

# overwrite instructions.txt with instructions_obfuscated.txt if $OBFUSCATE is set
if [ "${OBFUSCATE:-}" = "true" ]; then
  if [ ! -w /home/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv /home/instructions_obfuscated.txt /home/instructions.txt
fi

# start a new file to store the full instructions, starting with general instructions
cp /home/instructions.txt ${AGENT_DIR}/full_instructions.txt

# Update instructions for agent-specific details: replace `/home/` paths to make paths relative
# (since the agent will have its own copies of these files in its workspace).
# e.g. /home/submission/submission.csv -> submission/submission.csv
sed -i 's|/home/||g' ${AGENT_DIR}/full_instructions.txt
# we'll take care of moving things to home/submission/ ourselves

# move on to agent-specific instructions, with a linebreak in between
# substitute env variables into additional_notes.txt and append result to full_instructions.txt
echo "" >> ${AGENT_DIR}/full_instructions.txt
envsubst < ${AGENT_DIR}/additional_notes.txt >> ${AGENT_DIR}/full_instructions.txt
# finally, append the comp instructions, with a linebreak in between
printf "\nCOMPETITION INSTRUCTIONS\n------\n\n" >> ${AGENT_DIR}/full_instructions.txt

# overwrite description.md with description_obfuscated.md if $OBFUSCATE is set
if [ "${OBFUSCATE:-}" = "true" ]; then
  if [ ! -w /home/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv /home/data/description_obfuscated.md /home/data/description.md
fi
cat /home/data/description.md >> ${AGENT_DIR}/full_instructions.txt

if [ -n "${SEED:-}" ] && [ -z "${AIDE_SEED:-}" ]; then
  export AIDE_SEED="${SEED}"
fi

AIDE_EXP_NAME="${COMPETITION_ID:-exp}"
AIDE_LOG_DIR="${LOGS_DIR}"
AIDE_WORKSPACE_DIR="${AGENT_DIR}/workspaces"

# Run with timeout and then export the best submission to MLE-bench's required location.
set +e
timeout $TIME_LIMIT_SECS aide \
  data_dir="/home/data/" \
  desc_file="${AGENT_DIR}/full_instructions.txt" \
  exp_name="${AIDE_EXP_NAME}" \
  log_dir="${AIDE_LOG_DIR}" \
  workspace_dir="${AIDE_WORKSPACE_DIR}" \
  $@ # forward the bash arguments to aide
exit_code=$?
set -e

if [ $exit_code -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi

# Find the run directory AIDE wrote under /home/logs (e.g., /home/logs/0-<exp_name>).
run_log_dir="$(
  ls -td "${AIDE_LOG_DIR}"/*/ 2>/dev/null | head -n 1 | sed 's:/*$::' || true
)"
if [ -z "${run_log_dir}" ] || [ ! -d "${run_log_dir}" ]; then
  echo "ERROR: Could not locate AIDE run directory under ${AIDE_LOG_DIR}"
  exit 1
fi

# Export best solution code (for debugging / inspection).
if [ -f "${run_log_dir}/best_solution.py" ]; then
  cp "${run_log_dir}/best_solution.py" "${CODE_DIR}/best_solution.py"
fi

# Pick a submission to hand to MLE-bench.
submission_src=""
if [ -f "${run_log_dir}/submission_post_search.csv" ]; then
  submission_src="${run_log_dir}/submission_post_search.csv"
elif [ -f "${run_log_dir}/final_selection.json" ]; then
  submission_src="$(
    python - <<'PY' "${run_log_dir}/final_selection.json" 2>/dev/null || true
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())

for key in ("post_search", "best_valid"):
    obj = data.get(key) or {}
    p = obj.get("submission_csv_path")
    if isinstance(p, str) and p:
        print(p)
        raise SystemExit(0)

raise SystemExit(1)
PY
  )"
fi

if [ -n "${submission_src}" ] && [ -f "${submission_src}" ]; then
  cp "${submission_src}" "${SUBMISSION_DIR}/submission.csv"
else
  # Fallback: use the newest per-node submission snapshot saved under solutions/.
  fallback_submission="$(
    ls -t "${run_log_dir}/solutions"/submission_node_*.csv 2>/dev/null | head -n 1 || true
  )"
  if [ -n "${fallback_submission}" ] && [ -f "${fallback_submission}" ]; then
    cp "${fallback_submission}" "${SUBMISSION_DIR}/submission.csv"
  else
    echo "ERROR: No submission found to copy to ${SUBMISSION_DIR}/submission.csv"
    exit 1
  fi
fi

exit $exit_code
