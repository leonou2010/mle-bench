#!/bin/bash
# Run ML-Master natively (no Docker) with MLE-bench grading
#
# This is much simpler and faster for development!
# Only use Docker when you need reproducible/official results.
#
# Usage:
#   ./run_native.sh <competition_dir> <steps> [exp_name]
#
# Example:
#   ./run_native.sh /home/ka3094/dataset_submit/spaceship-titanic 5 test-run

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <competition_dir> <steps> [exp_name]"
    echo "Example: $0 /home/ka3094/dataset_submit/spaceship-titanic 5 test-run"
    exit 1
fi

COMPETITION_DIR="$1"
STEPS="$2"
EXP_NAME="${3:-native-test}"
COMPETITION_ID=$(basename "$COMPETITION_DIR")

echo "========================================="
echo "Running ML-Master Natively (No Docker)"
echo "========================================="
echo "Competition: $COMPETITION_ID"
echo "Data: $COMPETITION_DIR"
echo "Steps: $STEPS"
echo "Experiment: $EXP_NAME"
echo ""

# Check if competition directory exists
if [ ! -d "$COMPETITION_DIR" ]; then
    echo "ERROR: Competition directory not found: $COMPETITION_DIR"
    exit 1
fi

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/home/ka3094/mle-bench/native_runs/mlmaster_${COMPETITION_ID}_${TIMESTAMP}_${EXP_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Find description file
DESC_FILE="$COMPETITION_DIR/description.md"
if [ ! -f "$DESC_FILE" ]; then
    DESC_FILE="$COMPETITION_DIR/task_description.txt"
    if [ ! -f "$DESC_FILE" ]; then
        echo "ERROR: No description file found"
        exit 1
    fi
fi

echo "Running ML-Master..."
echo ""

# Run ML-Master directly (no Docker!)
cd /home/ka3094/ML-Master_submit
python main_mcts.py \
  data_dir="$COMPETITION_DIR" \
  desc_file="$DESC_FILE" \
  agent.steps="$STEPS" \
  agent.k_fold_validation=5 \
  exp_name="$EXP_NAME" \
  start_cpu_id=0 \
  cpu_number=4 \
  agent.search.parallel_search_num=1 \
  agent.steerable_reasoning=false \
  agent.code.model=gpt-4o-2024-08-06 \
  agent.code.base_url=https://api.openai.com/v1 \
  "agent.code.api_key=\${oc.env:OPENAI_API_KEY}" \
  agent.feedback.model=gpt-4o-2024-08-06 \
  agent.feedback.base_url=https://api.openai.com/v1 \
  "agent.feedback.api_key=\${oc.env:OPENAI_API_KEY}" \
  agent.search.use_bug_consultant=true \
  agent.save_all_submission=true \
  agent.check_format=false \
  workspace_dir="$OUTPUT_DIR/workspace" \
  log_dir="$OUTPUT_DIR/logs"

echo ""
echo "========================================="
echo "ML-Master Complete!"
echo "========================================="

# Copy outputs to organized structure (MLE-bench style)
echo "Organizing outputs..."
mkdir -p "$OUTPUT_DIR"/{submission,logs,code}

# Find the workspace directory
WORKSPACE="$OUTPUT_DIR/workspace/$EXP_NAME"
if [ ! -d "$WORKSPACE" ]; then
    echo "WARNING: Workspace not found at $WORKSPACE"
    WORKSPACE=$(find "$OUTPUT_DIR/workspace" -type d -name "$EXP_NAME" | head -1)
fi

if [ -d "$WORKSPACE" ]; then
    # Copy submissions
    if [ -d "$WORKSPACE/submission" ]; then
        cp -r "$WORKSPACE/submission"/* "$OUTPUT_DIR/submission/" 2>/dev/null || true
        SUBMISSION_COUNT=$(ls -1 "$OUTPUT_DIR/submission"/submission_*.csv 2>/dev/null | wc -l)
        echo "✓ Copied $SUBMISSION_COUNT submissions"
    fi

    # Copy best solution
    if [ -d "$WORKSPACE/best_solution" ]; then
        cp -r "$WORKSPACE/best_solution" "$OUTPUT_DIR/code/" 2>/dev/null || true
        echo "✓ Copied best solution"
    fi
fi

# Copy logs
if [ -d "$OUTPUT_DIR/logs/$EXP_NAME" ]; then
    cp -r "$OUTPUT_DIR/logs/$EXP_NAME"/* "$OUTPUT_DIR/logs/" 2>/dev/null || true
    echo "✓ Copied logs"
fi

echo ""
echo "Results:"
echo "  Directory: $OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/submission/submission.csv" ]; then
    echo "  ✓ submission.csv found"
else
    echo "  ✗ submission.csv NOT FOUND"
fi

SUBMISSION_COUNT=$(find "$OUTPUT_DIR/submission" -name "submission_*.csv" 2>/dev/null | wc -l)
echo "  ✓ Total submissions: $SUBMISSION_COUNT"

if [ -f "$OUTPUT_DIR/logs/journal.jsonl" ]; then
    NODE_COUNT=$(wc -l < "$OUTPUT_DIR/logs/journal.jsonl")
    echo "  ✓ Journal nodes: $NODE_COUNT"
fi

echo ""
echo "To grade with MLE-bench:"
echo "  cd /home/ka3094/mle-bench"
echo "  python -m mlebench.grade \\"
echo "    --submission $OUTPUT_DIR/submission/submission.csv \\"
echo "    --competition $COMPETITION_ID"
echo ""
echo "To view all submissions:"
echo "  ls -lh $OUTPUT_DIR/submission/"
echo ""
echo "To view logs:"
echo "  cat $OUTPUT_DIR/logs/journal.jsonl | jq"
