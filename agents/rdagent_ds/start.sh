#!/bin/bash
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
if [ "$OBFUSCATE" = "true" ]; then
  if [ ! -w /home/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv /home/instructions_obfuscated.txt /home/instructions.txt
fi

# overwrite description.md with description_obfuscated.md if $OBFUSCATE is set
if [ "$OBFUSCATE" = "true" ]; then
  if [ ! -w /home/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv /home/data/description_obfuscated.md /home/data/description.md
fi

# Setup data directory for RD-Agent DataScience scenario
# RD-Agent expects: DS_LOCAL_DATA_PATH/<competition_id>/...
RD_DATA_DIR=${AGENT_DIR}/rd_data/${COMPETITION_ID}
mkdir -p ${RD_DATA_DIR}

# Symlink all files from /home/data/* into RD-Agent's data structure
for file in /home/data/*; do
  if [ -f "$file" ] || [ -d "$file" ]; then
    # We'll write our own combined description.md below.
    if [ "$(basename "$file")" = "description.md" ]; then
      continue
    fi
    ln -s "$file" "${RD_DATA_DIR}/$(basename $file)"
  fi
done

# Ensure description.md is available in RD-Agent data dir.
# RD-Agent reads `<local_data_path>/<competition_id>/description.md` as the competition description.
# We prepend the benchmark instructions so the agent knows about the offline harness + submission path.
rm -f ${RD_DATA_DIR}/description.md
cat /home/instructions.txt > ${RD_DATA_DIR}/description.md
printf "\n\nCOMPETITION INSTRUCTIONS\n------\n\n" >> ${RD_DATA_DIR}/description.md
cat /home/data/description.md >> ${RD_DATA_DIR}/description.md

# Set RD-Agent environment variables
export DS_LOCAL_DATA_PATH=${AGENT_DIR}/rd_data
export DS_Coder_CoSTEER_ENV_TYPE=conda  # Use conda instead of docker to avoid Docker-in-Docker
export DS_Runner_CoSTEER_ENV_TYPE=conda # Runner also executes code; avoid Docker-in-Docker
export LOG_TRACE_PATH=${LOGS_DIR}/rdagent/${COMPETITION_ID}
export WORKSPACE_PATH=${AGENT_DIR}/rd_workspace

# Create log directory structure
mkdir -p ${LOG_TRACE_PATH}

# Run RD-Agent's DS loop with a hard timeout.
# NOTE: RD-Agent's DS entrypoint is `rdagent/app/data_science/loop.py` (fire-based CLI).
timeout $TIME_LIMIT_SECS python -m rdagent.app.data_science.loop --competition ${COMPETITION_ID}

if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi

# Extract selected submissions (including per-strategy selections) into /home/submission and best code into /home/code.
python ${AGENT_DIR}/wrapper/extract_selected_submissions.py

# Ensure submission.csv exists in the right place
if [ -f "${SUBMISSION_DIR}/submission.csv" ]; then
  echo "RD-Agent produced submission.csv"
else
  echo "WARNING: RD-Agent did not produce submission.csv"
fi
