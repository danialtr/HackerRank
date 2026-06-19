"""Optional MCP server exposing the system's deterministic helpers as tools.

This is a DEMO and is OFF the hot path — neither code/main.py nor the evaluation
harness imports it (see README.md in this folder for the rationale). It shows how
the same deterministic helpers the pipeline already uses could be offered to an
interactive agent over the Model Context Protocol.

Requires the optional `mcp` package:  pip install mcp
Run:                                   python code/mcp_tools/server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the code/ modules importable when run directly.
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional dependency
    print("This demo needs the optional MCP SDK:  pip install mcp")
    raise SystemExit(1)

import config
from backends.heuristic import HeuristicBackend
from data_loader import load_user_history
from logging_setup import CostMeter
from models import ImageRef
from pipeline.evidence import _select_requirement, load_requirements

mcp = FastMCP("evidence-review-tools")
_backend = HeuristicBackend(CostMeter())
_requirements = load_requirements()
_histories = load_user_history()


@mcp.tool()
def assess_image_quality(image_path: str) -> dict:
    """Return deterministic image-quality signals (sharpness, brightness, flags)."""
    from data_loader import resolve_image
    ref: ImageRef = resolve_image(image_path, config.dataset_dir())
    if not ref.usable:
        return {"usable": False, "error": ref.load_error}
    blur, brightness = _backend._stats(ref)  # reuse the pipeline's CV stats
    flags = []
    if blur < _backend.blur_threshold:
        flags.append("blurry_image")
    if brightness < _backend.dark_threshold or brightness > _backend.glare_threshold:
        flags.append("low_light_or_glare")
    return {"usable": True, "sharpness": round(blur, 1),
            "brightness": round(brightness, 1), "flags": flags}


@mcp.tool()
def lookup_evidence_requirement(claim_object: str, object_part: str, issue_type: str) -> dict:
    """Return the minimum-image-evidence rule for a claim (object, part, issue)."""
    rid = _select_requirement(claim_object, object_part, issue_type)
    req = _requirements.get(rid, {})
    return {"requirement_id": rid,
            "minimum_image_evidence": req.get("minimum_image_evidence", ""),
            "applies_to": req.get("applies_to", "")}


@mcp.tool()
def get_user_history(user_id: str) -> dict:
    """Return the user-history risk context for a user_id (or empty if unknown)."""
    h = _histories.get(user_id)
    if not h:
        return {"user_id": user_id, "known": False}
    return {"user_id": user_id, "known": True, "past_claim_count": h.past_claim_count,
            "rejected_claim": h.rejected_claim, "last_90_days_claim_count": h.last_90_days_claim_count,
            "history_flags": h.history_flags, "history_summary": h.history_summary}


if __name__ == "__main__":
    mcp.run()
