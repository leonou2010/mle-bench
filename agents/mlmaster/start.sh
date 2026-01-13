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
envsubst < ${AGENT_DIR}/wrapper/additional_notes.txt >> ${AGENT_DIR}/full_instructions.txt
# finally, append the comp instructions, with a linebreak in between
printf "\nCOMPETITION INSTRUCTIONS\n------\n\n" >> ${AGENT_DIR}/full_instructions.txt

# overwrite description.md with description_obfuscated.md if $OBFUSCATE is set
if [ "$OBFUSCATE" = "true" ]; then
  if [ ! -w /home/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv /home/data/description_obfuscated.md /home/data/description.md
fi
cat /home/data/description.md >> ${AGENT_DIR}/full_instructions.txt

# Setup workspace and log directories for ML-Master
EXP_NAME="exp"
MLMASTER_WORKSPACE=${AGENT_DIR}/workspaces/${EXP_NAME}
MLMASTER_LOGS=${AGENT_DIR}/logs/${EXP_NAME}

# Create necessary directories
mkdir -p ${MLMASTER_WORKSPACE}
mkdir -p ${AGENT_DIR}/logs
mkdir -p ${MLMASTER_WORKSPACE}/submission

# Symbolic linking to MLE-bench expected paths
# ML-Master writes to workspace_dir/best_submission and workspace_dir/best_solution
ln -s ${CODE_DIR} ${MLMASTER_WORKSPACE}/best_solution
ln -s ${SUBMISSION_DIR} ${MLMASTER_WORKSPACE}/best_submission
ln -s ${LOGS_DIR} ${MLMASTER_LOGS}

# run ML-Master with timeout, and print if timeout occurs
# Note: Using Hydra override syntax for ML-Master config
timeout $TIME_LIMIT_SECS python ${AGENT_DIR}/ML-Master_submit/main_mcts.py \
  data_dir="/home/data" \
  desc_file="${AGENT_DIR}/full_instructions.txt" \
  exp_name="${EXP_NAME}" \
  workspace_dir="${MLMASTER_WORKSPACE}" \
  log_dir="${AGENT_DIR}/logs" \
  copy_data=False \
  $@ # forward the bash arguments to ML-Master

if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi

# Ensure submission.csv exists in the right place
if [ -f "${MLMASTER_WORKSPACE}/best_submission/submission.csv" ]; then
  echo "ML-Master produced submission.csv"
else
  echo "WARNING: ML-Master did not produce submission.csv"
fi
