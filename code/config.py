"""Path and runtime configuration.

The evaluator runs ``code/main.py`` with ``dataset/`` sitting next to ``code/``.
We resolve everything from this file's location so the code works no matter the
current working directory, and we allow overrides via environment variables so
the same code runs unchanged during grading.

Environment overrides:
  DATASET_DIR  absolute path to the dataset folder (default: <repo>/dataset)
  OUTPUT_CSV   path to write predictions  (default: <repo>/output.csv)
"""

from __future__ import annotations

import os
from pathlib import Path

# This file lives in <repo>/code/config.py
CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent


def dataset_dir() -> Path:
    """Folder that holds the CSVs and the images/ tree."""
    env = os.environ.get("DATASET_DIR")
    return Path(env).resolve() if env else (REPO_ROOT / "dataset")


def output_csv_path() -> Path:
    """Where final predictions are written."""
    env = os.environ.get("OUTPUT_CSV")
    return Path(env).resolve() if env else (REPO_ROOT / "output.csv")


# Convenience accessors for the individual input files.
def claims_csv() -> Path:
    return dataset_dir() / "claims.csv"


def sample_claims_csv() -> Path:
    return dataset_dir() / "sample_claims.csv"


def user_history_csv() -> Path:
    return dataset_dir() / "user_history.csv"


def evidence_requirements_csv() -> Path:
    return dataset_dir() / "evidence_requirements.csv"
