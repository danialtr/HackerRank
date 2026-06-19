"""Output schema: the single source of truth for columns and allowed values.

Schema compliance is binary in grading — one wrong enum spelling or a column out
of order can tank a row. So every value the system emits is funnelled through the
normalisers here, which snap a stray value to the nearest allowed enum rather
than letting it through. This is the cheapest, highest-leverage correctness step
in the whole system.
"""

from __future__ import annotations

import difflib
from typing import Iterable

# --------------------------------------------------------------------------- #
# Output columns, in the exact order required by problem_statement.md
# --------------------------------------------------------------------------- #
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# --------------------------------------------------------------------------- #
# Allowed enum values
# --------------------------------------------------------------------------- #
CLAIM_STATUS = ["supported", "contradicted", "not_enough_information"]

ISSUE_TYPE = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
]

SEVERITY = ["none", "low", "medium", "high", "unknown"]

OBJECT_PARTS = {
    "car": [
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    ],
    "laptop": [
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base",
        "body", "unknown",
    ],
    "package": [
        "box", "package_corner", "package_side", "seal", "label", "contents", "item",
        "unknown",
    ],
}

# Every risk flag the schema permits (excluding the literal "none" sentinel).
RISK_FLAGS = [
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
]

# Quality / authenticity flags a single image can carry (used by perception).
IMAGE_FLAGS = [
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible",
    "possible_manipulation", "non_original_image", "text_instruction_present",
]


def all_object_parts() -> list[str]:
    seen: list[str] = []
    for parts in OBJECT_PARTS.values():
        for p in parts:
            if p not in seen:
                seen.append(p)
    return seen


# --------------------------------------------------------------------------- #
# Normalisers — snap any value to the nearest allowed enum
# --------------------------------------------------------------------------- #
def _snap(value: str, allowed: Iterable[str], default: str) -> str:
    allowed = list(allowed)
    v = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if v in allowed:
        return v
    match = difflib.get_close_matches(v, allowed, n=1, cutoff=0.6)
    return match[0] if match else default


def norm_bool(value) -> str:
    """Render any truthy/falsey input as the literal 'true'/'false' strings."""
    if isinstance(value, bool):
        return "true" if value else "false"
    v = str(value).strip().lower()
    return "true" if v in {"true", "1", "yes", "y", "t"} else "false"


def norm_claim_status(value: str) -> str:
    return _snap(value, CLAIM_STATUS, "not_enough_information")


def norm_issue_type(value: str) -> str:
    return _snap(value, ISSUE_TYPE, "unknown")


def norm_severity(value: str) -> str:
    return _snap(value, SEVERITY, "unknown")


def norm_object_part(value: str, claim_object: str) -> str:
    parts = OBJECT_PARTS.get((claim_object or "").strip().lower(), all_object_parts())
    return _snap(value, parts, "unknown")


def norm_risk_flags(flags: Iterable[str]) -> str:
    """De-dup, snap each flag to an allowed value, join with ';' (or 'none')."""
    out: list[str] = []
    for f in flags or []:
        snapped = _snap(f, RISK_FLAGS, "")
        if snapped and snapped not in out:
            out.append(snapped)
    return ";".join(out) if out else "none"


def norm_image_ids(ids: Iterable[str]) -> str:
    out: list[str] = []
    for i in ids or []:
        i = (i or "").strip()
        if i and i.lower() != "none" and i not in out:
            out.append(i)
    return ";".join(out) if out else "none"


def validate_row(row: dict) -> list[str]:
    """Return a list of schema problems for a finished output row (empty == OK)."""
    problems: list[str] = []
    for col in OUTPUT_COLUMNS:
        if col not in row:
            problems.append(f"missing column: {col}")

    obj = (row.get("claim_object") or "").strip().lower()
    checks = {
        "claim_status": CLAIM_STATUS,
        "issue_type": ISSUE_TYPE,
        "severity": SEVERITY,
        "object_part": OBJECT_PARTS.get(obj, all_object_parts()),
    }
    for col, allowed in checks.items():
        val = (row.get(col) or "").strip()
        if val and val not in allowed:
            problems.append(f"{col}={val!r} not in allowed values")

    for col in ("evidence_standard_met", "valid_image"):
        if (row.get(col) or "") not in {"true", "false"}:
            problems.append(f"{col}={row.get(col)!r} is not a boolean string")

    rf = (row.get("risk_flags") or "").strip()
    if rf and rf != "none":
        for f in rf.split(";"):
            if f and f not in RISK_FLAGS:
                problems.append(f"risk_flag {f!r} not allowed")
    return problems
