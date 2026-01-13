#!/bin/bash
# Grade all ML-Master submissions with MLE-bench
#
# Usage:
#   ./grade_all_submissions.sh <run_dir> <competition_id>
#
# Example:
#   ./grade_all_submissions.sh runs/mlmaster_spaceship-titanic_20260111_123456 spaceship-titanic

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <run_dir> <competition_id>"
    echo "Example: $0 runs/mlmaster_spaceship-titanic_20260111_123456 spaceship-titanic"
    exit 1
fi

RUN_DIR="$1"
COMPETITION_ID="$2"

echo "========================================="
echo "Grading All ML-Master Submissions"
echo "========================================="
echo "Run directory: $RUN_DIR"
echo "Competition: $COMPETITION_ID"
echo ""

# Check if run directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Run directory not found: $RUN_DIR"
    exit 1
fi

# Find the workspace directory (should be extracted)
WORKSPACE_DIR="$RUN_DIR/code"
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "ERROR: Workspace directory not found: $WORKSPACE_DIR"
    echo "Make sure MLE-bench extracted the run outputs first"
    exit 1
fi

# Create grading directory
GRADING_DIR="$RUN_DIR/all_submissions"
mkdir -p "$GRADING_DIR"

echo "Extracting all submissions..."
# Look for submission files in the extracted workspace
# ML-Master saves them in workspaces/exp/submission/
SUBMISSION_SOURCE="$WORKSPACE_DIR/../submission"
if [ ! -d "$SUBMISSION_SOURCE" ]; then
    # Try alternate path
    SUBMISSION_SOURCE="$RUN_DIR/submission"
fi

if [ ! -d "$SUBMISSION_SOURCE" ]; then
    echo "ERROR: Submission directory not found"
    echo "Tried: $WORKSPACE_DIR/../submission and $RUN_DIR/submission"
    exit 1
fi

# Find all submission_*.csv files
SUBMISSION_FILES=$(find "$SUBMISSION_SOURCE" -name "submission_*.csv" | sort)
SUBMISSION_COUNT=$(echo "$SUBMISSION_FILES" | wc -l)

if [ -z "$SUBMISSION_FILES" ]; then
    echo "WARNING: No submission_*.csv files found in $SUBMISSION_SOURCE"
    echo "Checking for best submission only..."

    BEST_SUB="$SUBMISSION_SOURCE/submission.csv"
    if [ -f "$BEST_SUB" ]; then
        SUBMISSION_FILES="$BEST_SUB"
        SUBMISSION_COUNT=1
    else
        echo "ERROR: No submissions found at all"
        exit 1
    fi
fi

echo "Found $SUBMISSION_COUNT submission files"
echo ""

# Extract node metrics from journal if available
JOURNAL_FILE="$RUN_DIR/logs/journal.jsonl"
declare -A NODE_METRICS

if [ -f "$JOURNAL_FILE" ]; then
    echo "Loading metrics from journal..."
    while IFS= read -r line; do
        node_id=$(echo "$line" | jq -r '.id // empty')
        metric=$(echo "$line" | jq -r '.metric.value // empty')
        if [ -n "$node_id" ] && [ -n "$metric" ]; then
            NODE_METRICS["$node_id"]="$metric"
        fi
    done < "$JOURNAL_FILE"
    echo "Loaded ${#NODE_METRICS[@]} node metrics"
fi

# Copy and grade each submission
i=0
echo ""
echo "Processing submissions..."
echo ""

for sub_file in $SUBMISSION_FILES; do
    # Extract node ID from filename
    filename=$(basename "$sub_file")

    if [[ "$filename" =~ submission_([a-f0-9]+)\.csv ]]; then
        node_id="${BASH_REMATCH[1]}"
        metric="${NODE_METRICS[$node_id]:-unknown}"
    else
        node_id="final"
        metric="best"
    fi

    # Format index with leading zeros
    index=$(printf "%03d" $i)

    # Copy submission with informative name
    output_file="$GRADING_DIR/submission_${index}_${node_id}.csv"
    cp "$sub_file" "$output_file"

    echo "[$index] Node: $node_id | Metric: $metric"
    echo "       File: $output_file"

    # Grade this submission with MLE-bench
    echo "       Grading..."

    # Create temporary submission.jsonl for this file
    temp_submission="$GRADING_DIR/temp_submission_${index}.jsonl"
    echo "{\"competition_id\": \"$COMPETITION_ID\", \"submission_path\": \"$output_file\"}" > "$temp_submission"

    # Grade with MLE-bench
    grade_output="$GRADING_DIR/grade_${index}_${node_id}.json"
    if mlebench grade --submission "$temp_submission" --output-dir "$GRADING_DIR" > "$grade_output" 2>&1; then
        # Extract score from grading output
        score=$(cat "$grade_output" | jq -r '.score // "N/A"' 2>/dev/null || echo "N/A")
        echo "       Score: $score"
    else
        echo "       ERROR: Grading failed (see $grade_output)"
    fi

    rm -f "$temp_submission"
    echo ""

    i=$((i + 1))
done

# Create summary report
SUMMARY_FILE="$GRADING_DIR/grading_summary.txt"
echo "Creating summary report: $SUMMARY_FILE"

cat > "$SUMMARY_FILE" << EOF
ML-Master Submission Grading Summary
=====================================
Competition: $COMPETITION_ID
Run Directory: $RUN_DIR
Total Submissions: $SUBMISSION_COUNT
Graded: $(ls -1 "$GRADING_DIR"/grade_*.json 2>/dev/null | wc -l)

Submissions (sorted by index):
EOF

# Add each submission to summary
for grade_file in $(ls -1 "$GRADING_DIR"/grade_*.json 2>/dev/null | sort); do
    index=$(basename "$grade_file" | sed 's/grade_\([0-9]*\)_.*/\1/')
    node_id=$(basename "$grade_file" | sed 's/grade_[0-9]*_\(.*\)\.json/\1/')
    score=$(cat "$grade_file" | jq -r '.score // "N/A"' 2>/dev/null || echo "N/A")
    metric="${NODE_METRICS[$node_id]:-unknown}"

    echo "[$index] Node: $node_id | Val Metric: $metric | Test Score: $score" >> "$SUMMARY_FILE"
done

echo ""
echo "========================================="
echo "Grading Complete!"
echo "========================================="
echo "Results directory: $GRADING_DIR"
echo "Summary: $SUMMARY_FILE"
echo ""
cat "$SUMMARY_FILE"
