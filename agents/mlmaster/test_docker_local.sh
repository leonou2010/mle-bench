#!/bin/bash
# Test ML-Master Docker image locally (outside full MLE-bench)
#
# This script mimics what MLE-bench does but lets you test the Docker image directly
#
# Usage:
#   ./test_docker_local.sh <competition_dir> [steps]
#
# Example:
#   ./test_docker_local.sh /home/ka3094/dataset_submit/spaceship-titanic 5

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <competition_dir> [steps]"
    echo "Example: $0 /home/ka3094/dataset_submit/spaceship-titanic 5"
    exit 1
fi

COMPETITION_DIR="$1"
STEPS="${2:-5}"
COMPETITION_ID=$(basename "$COMPETITION_DIR")

echo "========================================="
echo "Testing ML-Master Docker Image"
echo "========================================="
echo "Competition: $COMPETITION_ID"
echo "Data: $COMPETITION_DIR"
echo "Steps: $STEPS"
echo ""

# Check if competition directory exists
if [ ! -d "$COMPETITION_DIR" ]; then
    echo "ERROR: Competition directory not found: $COMPETITION_DIR"
    exit 1
fi

# Check if Docker image exists
if ! docker image inspect mlmaster >/dev/null 2>&1; then
    echo "Docker image 'mlmaster' not found. Building..."
    cd /home/ka3094
    docker build --platform=linux/amd64 -t mlmaster -f mle-bench/agents/mlmaster/Dockerfile .
fi

# Create output directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/home/ka3094/mle-bench/test_runs/mlmaster_${COMPETITION_ID}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"/{submission,logs,code}

echo "Output directory: $OUTPUT_DIR"
echo ""

# Create description file if needed
DESC_FILE="$COMPETITION_DIR/description.md"
if [ ! -f "$DESC_FILE" ]; then
    echo "WARNING: description.md not found, checking for task_description.txt"
    DESC_FILE="$COMPETITION_DIR/task_description.txt"
    if [ ! -f "$DESC_FILE" ]; then
        echo "ERROR: No description file found"
        exit 1
    fi
fi

# Create minimal instructions.txt
INSTRUCTIONS_FILE="$OUTPUT_DIR/instructions.txt"
cat > "$INSTRUCTIONS_FILE" << 'EOF'
BENCHMARK RULES
------

You are an AI agent solving a Kaggle-style machine learning competition.

Your goal is to:
1. Analyze the provided data
2. Build a predictive model
3. Generate predictions on the test set
4. Save predictions to submission.csv

Resources:
- Data is available in /home/data/
- You must save your final submission to /home/submission/submission.csv
- Logs should be saved to /home/logs/
- Your best code should be in /home/code/

Remember:
- Use cross-validation for robust model evaluation
- The submission.csv format must match the sample submission
- You can validate your submission format at http://localhost:5000/validate
EOF

echo "Running ML-Master in Docker..."
echo ""

# Run Docker container (mimicking MLE-bench)
docker run --rm \
  --platform=linux/amd64 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  -e TIME_LIMIT_SECS=3600 \
  -e STEP_LIMIT="$STEPS" \
  -e COMPETITION_ID="$COMPETITION_ID" \
  --mount type=bind,source="$COMPETITION_DIR",target=/home/data,readonly \
  --mount type=bind,source="$INSTRUCTIONS_FILE",target=/home/instructions.txt,readonly \
  --mount type=bind,source="$OUTPUT_DIR/submission",target=/home/submission \
  --mount type=bind,source="$OUTPUT_DIR/logs",target=/home/logs \
  --mount type=bind,source="$OUTPUT_DIR/code",target=/home/code \
  mlmaster \
  bash /home/agent/start.sh \
    agent.steps="$STEPS" \
    agent.code.model=gpt-4o-2024-08-06 \
    agent.feedback.model=gpt-4o-2024-08-06 \
    agent.save_all_submission=true \
    agent.check_format=false

echo ""
echo "========================================="
echo "Docker run complete!"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check outputs
echo "Checking outputs..."
if [ -f "$OUTPUT_DIR/submission/submission.csv" ]; then
    echo "✓ submission.csv found"
else
    echo "✗ submission.csv NOT FOUND"
fi

SUBMISSION_COUNT=$(find "$OUTPUT_DIR/submission" -name "submission_*.csv" 2>/dev/null | wc -l)
echo "✓ Found $SUBMISSION_COUNT total submissions"

if [ -f "$OUTPUT_DIR/logs/journal.jsonl" ]; then
    NODE_COUNT=$(wc -l < "$OUTPUT_DIR/logs/journal.jsonl")
    echo "✓ journal.jsonl found ($NODE_COUNT nodes)"
else
    echo "✗ journal.jsonl NOT FOUND"
fi

if [ -d "$OUTPUT_DIR/code/best_solution" ]; then
    echo "✓ best_solution directory found"
else
    echo "✗ best_solution NOT FOUND"
fi

echo ""
echo "To view logs:"
echo "  cat $OUTPUT_DIR/logs/journal.jsonl | jq"
echo ""
echo "To view best code:"
echo "  cat $OUTPUT_DIR/code/best_solution/solution.py"
echo ""
echo "To view submissions:"
echo "  ls -lh $OUTPUT_DIR/submission/"
