"""Path, model, and runtime configuration.

The evaluator runs ``code/main.py`` with ``dataset/`` sitting next to ``code/``.
We resolve everything from this file's location so the code works no matter the
current working directory, and we allow overrides via environment variables so
the same code runs unchanged during grading.

Environment overrides:
  DATASET_DIR   absolute path to the dataset folder (default: <repo>/dataset)
  OUTPUT_CSV    path to write predictions     (default: <repo>/output.csv)
  EVR_BACKEND   "auto" | "vlm" | "heuristic"  (default: auto)
  EVR_ARCH      "pipeline" | "mega"           (default: pipeline)
  ANTHROPIC_API_KEY   required only for the VLM backend
"""

from __future__ import annotations

import os
from pathlib import Path

# This file lives in <repo>/code/config.py
CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent


def _load_dotenv() -> None:
    """Minimal, dependency-free .env loader.

    Looks for a .env file at the repo root (or code/) and loads simple
    ``KEY=VALUE`` lines into the environment. Real, already-exported environment
    variables always win, so an exported ANTHROPIC_API_KEY overrides the file.
    Secrets are never logged.
    """
    for candidate in (REPO_ROOT / ".env", CODE_DIR / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:   # exported env vars take precedence
                os.environ[key] = value


_load_dotenv()


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def dataset_dir() -> Path:
    """Folder that holds the CSVs and the images/ tree."""
    env = os.environ.get("DATASET_DIR")
    return Path(env).resolve() if env else (REPO_ROOT / "dataset")


def output_csv_path() -> Path:
    """Where final predictions are written."""
    env = os.environ.get("OUTPUT_CSV")
    return Path(env).resolve() if env else (REPO_ROOT / "output.csv")


def claims_csv() -> Path:
    return dataset_dir() / "claims.csv"


def sample_claims_csv() -> Path:
    return dataset_dir() / "sample_claims.csv"


def user_history_csv() -> Path:
    return dataset_dir() / "user_history.csv"


def evidence_requirements_csv() -> Path:
    return dataset_dir() / "evidence_requirements.csv"


# --------------------------------------------------------------------------- #
# Model tiering (see code/SOLUTION.md §"Technology choices")
#
# Tiered strategy: a cheap text model extracts the claim, a mid-tier vision
# model does the high-volume per-image perception, and the most capable model
# is reserved for the few genuinely ambiguous fusion decisions.
# --------------------------------------------------------------------------- #
MODEL_EXTRACT = os.environ.get("EVR_MODEL_EXTRACT", "claude-haiku-4-5")
MODEL_PERCEPTION = os.environ.get("EVR_MODEL_PERCEPTION", "claude-sonnet-4-6")
MODEL_FUSION = os.environ.get("EVR_MODEL_FUSION", "claude-opus-4-8")

# Pricing in USD per 1,000,000 tokens (cached 2026-06; Claude API reference).
PRICING = {
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

# Cache economics (prompt caching): writes ~1.25x base input, reads ~0.1x.
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10


def backend_choice() -> str:
    """auto -> vlm if a key is present, else heuristic."""
    return os.environ.get("EVR_BACKEND", "auto").strip().lower()


def architecture() -> str:
    """'pipeline' (multi-stage, default) or 'mega' (single mega-prompt baseline)."""
    return os.environ.get("EVR_ARCH", "pipeline").strip().lower()


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
