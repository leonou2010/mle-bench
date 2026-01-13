#!/usr/bin/env python3
"""
Add competitions from dataset_submit to MLE-bench.

This script creates MLE-bench-compatible competition directories
from existing Kaggle-style datasets without modifying the originals.

Usage:
    python add_dataset_to_mlebench.py <dataset_dir>
    python add_dataset_to_mlebench.py --all  # Process all datasets

Example:
    python add_dataset_to_mlebench.py /home/ka3094/dataset_submit/spaceship-titanic
    python add_dataset_to_mlebench.py --all
"""

import sys
import shutil
import hashlib
from pathlib import Path
import pandas as pd
import re


MLEBENCH_ROOT = Path("/home/ka3094/mle-bench")
COMPETITIONS_DIR = MLEBENCH_ROOT / "mlebench" / "competitions"
DATASET_SUBMIT_DIR = Path("/home/ka3094/dataset_submit")


def detect_metric_from_description(desc_file: Path) -> str:
    """Detect metric from task description."""

    if not desc_file.exists():
        return "accuracy"  # Default

    content = desc_file.read_text().lower()

    # Common patterns
    if "accuracy" in content or "classification" in content:
        return "accuracy"
    elif "rmse" in content or "root mean squared error" in content:
        return "rmse"
    elif "rmsle" in content or "root mean squared log" in content:
        return "rmsle"
    elif "mae" in content or "mean absolute error" in content:
        return "mae"
    elif "auc" in content or "area under" in content:
        return "auc"
    elif "f1" in content:
        return "f1"
    elif "log loss" in content or "logloss" in content:
        return "logloss"
    elif "mape" in content:
        return "mape"
    else:
        # Try to detect from sample submission
        return "accuracy"  # Default fallback


def detect_target_column(sample_submission: Path) -> str:
    """Detect target column from sample submission."""

    try:
        df = pd.read_csv(sample_submission, nrows=1)
        columns = list(df.columns)

        # Skip ID columns
        id_cols = ['id', 'Id', 'ID', 'PassengerId', 'index']
        for col in columns:
            if col not in id_cols and not col.lower().endswith('id'):
                return col

        # Fallback: use second column
        if len(columns) >= 2:
            return columns[1]

        return columns[0]
    except Exception as e:
        print(f"Warning: Could not detect target column: {e}")
        return "target"


def detect_id_column(sample_submission: Path) -> str:
    """Detect ID column from sample submission."""

    try:
        df = pd.read_csv(sample_submission, nrows=1)
        columns = list(df.columns)

        # Common ID column patterns
        id_patterns = ['id', 'Id', 'ID', 'PassengerId', 'index']
        for col in columns:
            if col in id_patterns or col.lower().endswith('id'):
                return col

        # Fallback: use first column
        return columns[0]
    except Exception as e:
        print(f"Warning: Could not detect ID column: {e}")
        return "id"


def create_config_yaml(comp_id: str, comp_name: str, metric: str) -> str:
    """Generate config.yaml content."""

    return f"""id: {comp_id}
name: {comp_name}
competition_type: simple
awards_medals: false
prizes: null
description: mlebench/competitions/{comp_id}/description.md

dataset:
  answers: {comp_id}/prepared/private/answers.csv
  sample_submission: {comp_id}/prepared/public/sample_submission.csv

grader:
  name: {metric}
  grade_fn: mlebench.competitions.{comp_id}.grade:grade

preparer: mlebench.competitions.{comp_id}.prepare:prepare
"""


def create_grade_py(metric: str, target_col: str, id_col: str) -> str:
    """Generate grade.py content based on metric."""

    if metric == "accuracy":
        return f"""import pandas as pd
from sklearn.metrics import accuracy_score

from mlebench.competitions.utils import prepare_for_accuracy_metric


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    accuracy_inputs = prepare_for_accuracy_metric(
        submission=submission, answers=answers, target_column="{target_col}", id_column="{id_col}"
    )
    return accuracy_score(**accuracy_inputs)
"""
    elif metric in ["rmse", "rmsle", "mae", "mape"]:
        sklearn_metric = {
            "rmse": "mean_squared_error",
            "rmsle": "mean_squared_log_error",
            "mae": "mean_absolute_error",
            "mape": "mean_absolute_percentage_error",
        }[metric]

        extra_args = ""
        if metric == "rmse":
            extra_args = ", squared=False"
        elif metric == "rmsle":
            extra_args = ", squared=False"

        return f"""import pandas as pd
from sklearn.metrics import {sklearn_metric}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    # Merge on ID column
    merged = submission.merge(answers, on="{id_col}", suffixes=("_pred", "_true"))

    y_true = merged["{target_col}_true"]
    y_pred = merged["{target_col}_pred"]

    return {sklearn_metric}(y_true, y_pred{extra_args})
"""
    else:
        # Generic template
        return f"""import pandas as pd


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    # Merge on ID column
    merged = submission.merge(answers, on="{id_col}", suffixes=("_pred", "_true"))

    y_true = merged["{target_col}_true"]
    y_pred = merged["{target_col}_pred"]

    # TODO: Implement {metric} metric
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)
"""


def create_prepare_py(dataset_dir: Path) -> str:
    """Generate prepare.py that copies already-prepared data."""

    return f"""from pathlib import Path
import shutil

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    \"\"\"
    Prepare competition data.

    Since data is already prepared in dataset_submit, we just copy it.
    For MLE-bench, we need to:
    - public/: Data visible to agents (train.csv, test.csv, sample_submission.csv)
    - private/: Ground truth for grading (answers.csv with test labels)
    \"\"\"

    # Copy public files (what agents see)
    if (raw / "train.csv").exists():
        shutil.copy(raw / "train.csv", public / "train.csv")

    if (raw / "test.csv").exists():
        shutil.copy(raw / "test.csv", public / "test.csv")

    if (raw / "sample_submission.csv").exists():
        shutil.copy(raw / "sample_submission.csv", public / "sample_submission.csv")

    # Create private answers file
    # In Kaggle datasets, we don't have true test labels, so we use sample submission as placeholder
    # For real grading, you'd need actual test labels
    if (raw / "sample_submission.csv").exists():
        # Use sample submission as answers (this is a placeholder)
        # In production, replace with actual test labels
        shutil.copy(raw / "sample_submission.csv", private / "answers.csv")

        print("WARNING: Using sample_submission.csv as answers (placeholder)")
        print("For real grading, replace with actual test labels")
"""


def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""

    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_checksums_yaml(dataset_dir: Path) -> str:
    """Generate checksums.yaml for dataset files."""

    checksums = {}

    for csv_file in dataset_dir.glob("*.csv"):
        checksum = calculate_checksum(csv_file)
        checksums[csv_file.name] = checksum

    lines = []
    for filename, checksum in sorted(checksums.items()):
        lines.append(f"{filename}: {checksum}")

    return "\n".join(lines) + "\n"


def create_leaderboard_csv() -> str:
    """Generate minimal leaderboard.csv."""

    return """TeamId,TeamName,SubmissionDate,Score
1,Baseline,2024-01-01,0.5
"""


def add_competition(dataset_dir: Path):
    """Add a competition to MLE-bench from dataset_submit."""

    comp_id = dataset_dir.name
    print(f"\n{'='*60}")
    print(f"Adding competition: {comp_id}")
    print(f"{'='*60}")

    # Check required files
    required_files = ["train.csv", "test.csv", "sample_submission.csv"]
    missing = [f for f in required_files if not (dataset_dir / f).exists()]

    if missing:
        print(f"❌ Missing required files: {missing}")
        print(f"   Skipping {comp_id}")
        return False

    # Create competition directory
    comp_dir = COMPETITIONS_DIR / comp_id
    if comp_dir.exists():
        print(f"⚠️  Competition directory already exists: {comp_dir}")
        response = input("   Overwrite? (y/N): ")
        if response.lower() != 'y':
            print(f"   Skipping {comp_id}")
            return False
        shutil.rmtree(comp_dir)

    comp_dir.mkdir(parents=True)
    print(f"✓ Created directory: {comp_dir}")

    # Detect competition properties
    desc_file = dataset_dir / "task_description.txt"
    if not desc_file.exists():
        desc_file = dataset_dir / "description.md"

    metric = detect_metric_from_description(desc_file)
    target_col = detect_target_column(dataset_dir / "sample_submission.csv")
    id_col = detect_id_column(dataset_dir / "sample_submission.csv")

    comp_name = comp_id.replace("-", " ").replace("_", " ").title()

    print(f"✓ Detected:")
    print(f"   - Metric: {metric}")
    print(f"   - Target column: {target_col}")
    print(f"   - ID column: {id_col}")

    # Create config.yaml
    config_content = create_config_yaml(comp_id, comp_name, metric)
    (comp_dir / "config.yaml").write_text(config_content)
    print(f"✓ Created config.yaml")

    # Create grade.py
    grade_content = create_grade_py(metric, target_col, id_col)
    (comp_dir / "grade.py").write_text(grade_content)
    print(f"✓ Created grade.py")

    # Create prepare.py
    prepare_content = create_prepare_py(dataset_dir)
    (comp_dir / "prepare.py").write_text(prepare_content)
    print(f"✓ Created prepare.py")

    # Copy description
    if desc_file.exists():
        shutil.copy(desc_file, comp_dir / "description.md")
        print(f"✓ Copied description.md")
    else:
        # Create minimal description
        (comp_dir / "description.md").write_text(f"# {comp_name}\n\nKaggle competition.\n")
        print(f"⚠️  Created minimal description.md")

    # Create checksums.yaml
    checksums_content = create_checksums_yaml(dataset_dir)
    (comp_dir / "checksums.yaml").write_text(checksums_content)
    print(f"✓ Created checksums.yaml")

    # Create leaderboard.csv
    leaderboard_content = create_leaderboard_csv()
    (comp_dir / "leaderboard.csv").write_text(leaderboard_content)
    print(f"✓ Created leaderboard.csv")

    # Copy raw data to MLE-bench data directory
    data_dir = MLEBENCH_ROOT / "data" / "competitions" / comp_id / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in dataset_dir.glob("*.csv"):
        shutil.copy(csv_file, data_dir / csv_file.name)

    if (dataset_dir / "data_description.txt").exists():
        shutil.copy(dataset_dir / "data_description.txt", data_dir / "data_description.txt")

    print(f"✓ Copied raw data to: {data_dir}")

    print(f"\n✅ Successfully added {comp_id} to MLE-bench!")
    print(f"   Location: {comp_dir}")
    print(f"\n   To prepare: mlebench prepare -c {comp_id}")
    print(f"   To test: python run_agent.py --agent-id mlmaster/dev --competition {comp_id}")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python add_dataset_to_mlebench.py <dataset_dir>")
        print("       python add_dataset_to_mlebench.py --all")
        print("\nExample:")
        print("  python add_dataset_to_mlebench.py /home/ka3094/dataset_submit/spaceship-titanic")
        print("  python add_dataset_to_mlebench.py --all")
        sys.exit(1)

    if sys.argv[1] == "--all":
        print(f"\n{'='*60}")
        print(f"Adding ALL competitions from {DATASET_SUBMIT_DIR}")
        print(f"{'='*60}\n")

        success_count = 0
        skip_count = 0

        for dataset_dir in sorted(DATASET_SUBMIT_DIR.iterdir()):
            if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                if add_competition(dataset_dir):
                    success_count += 1
                else:
                    skip_count += 1

        print(f"\n{'='*60}")
        print(f"Summary")
        print(f"{'='*60}")
        print(f"✅ Successfully added: {success_count} competitions")
        print(f"⚠️  Skipped: {skip_count} competitions")
        print(f"\nAll competitions are now available in MLE-bench!")
        print(f"Location: {COMPETITIONS_DIR}")
    else:
        dataset_dir = Path(sys.argv[1])

        if not dataset_dir.exists():
            print(f"ERROR: Dataset directory not found: {dataset_dir}")
            sys.exit(1)

        add_competition(dataset_dir)


if __name__ == "__main__":
    main()
