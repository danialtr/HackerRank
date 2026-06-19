"""Stage 6: evidence sufficiency — deterministic, no model call.

We look up the matching row in evidence_requirements.csv by (object, issue/part
family) and decide whether the submitted images are enough to evaluate the claim:
is there at least one valid image that actually shows the claimed part/condition?
This is pure code, so it is reliable, free, and reproducible.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import config
from logging_setup import log
from models import Claim, ClaimIntent, EvidenceDecision, PerceptionResult

# Which requirement_id covers a given (object, part, issue).
_CAR_GLASS_LIGHT = {"windshield", "side_mirror", "headlight", "taillight"}
_CAR_GLASS_ISSUES = {"crack", "glass_shatter", "broken_part", "missing_part"}
_LAPTOP_SCREEN_KB = {"screen", "keyboard", "trackpad"}
_PACKAGE_EXTERIOR_ISSUES = {"crushed_packaging", "torn_packaging"}
_PACKAGE_STAIN_ISSUES = {"water_damage", "stain"}
_PACKAGE_CONTENT_PARTS = {"contents", "item"}


def _select_requirement(claim_object: str, part: str, issue: str) -> str:
    o = claim_object.lower()
    if o == "car":
        if part in _CAR_GLASS_LIGHT or issue in _CAR_GLASS_ISSUES:
            return "REQ_CAR_GLASS_LIGHT_MIRROR"
        if issue in ("dent", "scratch"):
            return "REQ_CAR_BODY_PANEL"
        return "REQ_CAR_IDENTITY_OR_SIDE"
    if o == "laptop":
        if part in _LAPTOP_SCREEN_KB:
            return "REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD"
        return "REQ_LAPTOP_BODY_HINGE_PORT"
    if o == "package":
        if part in _PACKAGE_CONTENT_PARTS or issue == "missing_part":
            return "REQ_PACKAGE_CONTENTS"
        if issue in _PACKAGE_STAIN_ISSUES or part == "label":
            return "REQ_PACKAGE_LABEL_OR_STAIN"
        if issue in _PACKAGE_EXTERIOR_ISSUES or part in ("seal", "box", "package_corner", "package_side"):
            return "REQ_PACKAGE_EXTERIOR"
    return "REQ_GENERAL_OBJECT_PART"


def load_requirements(path: Optional[Path] = None) -> dict[str, dict]:
    path = path or config.evidence_requirements_csv()
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rid = (row.get("requirement_id") or "").strip()
            if rid:
                out[rid] = row
    return out


def decide_evidence(claim: Claim, intent: ClaimIntent,
                    perceptions: list[PerceptionResult],
                    requirements: dict[str, dict]) -> EvidenceDecision:
    rid = _select_requirement(claim.claim_object, intent.object_part, intent.issue_type)
    req = requirements.get(rid, {})

    relevant = [p for p in perceptions if p.valid_image and p.is_claimed_object and p.shows_claimed_part]
    n_valid = sum(1 for p in perceptions if p.valid_image)
    met = len(relevant) >= 1

    if met:
        reason = (f"{len(relevant)} valid image(s) show the claimed "
                  f"{intent.object_part} clearly enough to evaluate the claim "
                  f"({rid}).")
    elif n_valid == 0:
        reason = "No submitted image is usable for automated review, so the claim cannot be evaluated."
    else:
        reason = (f"The submitted image(s) do not clearly show the claimed "
                  f"{intent.object_part}/{intent.issue_type}, so the minimum evidence "
                  f"for {rid} is not satisfied.")

    log.debug("    [evidence] req=%s met=%s relevant=%d/%d", rid, met, len(relevant), len(perceptions))
    return EvidenceDecision(met=met, reason=reason, requirement_id=rid)
