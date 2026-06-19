"""Stage 8: decision fusion — the adjudicator.

Combines the claim intent, per-image perception, evidence sufficiency, and
history risk into the final verdict. The decision is deterministic by default;
genuinely ambiguous cases may be escalated to the Opus tie-breaker (when the VLM
backend is in use). The precedence rule is enforced structurally: history only
adds flags, it can never change ``supported`` ↔ ``contradicted`` ↔
``not_enough_information``.
"""

from __future__ import annotations

from logging_setup import log
from models import (Claim, ClaimIntent, EvidenceDecision, FusionResult,
                    HistoryRisk, PerceptionResult)

# Issues that are visually compatible (a claimed crack confirmed by a shatter, etc.)
_COMPATIBLE = {
    "crack": {"crack", "glass_shatter"},
    "glass_shatter": {"glass_shatter", "crack"},
    "broken_part": {"broken_part", "missing_part"},
    "missing_part": {"missing_part", "broken_part"},
    "dent": {"dent"},
    "scratch": {"scratch"},
    "torn_packaging": {"torn_packaging"},
    "crushed_packaging": {"crushed_packaging"},
    "water_damage": {"water_damage", "stain"},
    "stain": {"stain", "water_damage"},
}


def _compatible(visible: str, claimed: str) -> bool:
    if claimed in ("", "unknown"):
        return visible not in ("none", "unknown")
    return visible in _COMPATIBLE.get(claimed, {claimed})


def fuse(claim: Claim, intent: ClaimIntent, perceptions: list[PerceptionResult],
         evidence: EvidenceDecision, history: HistoryRisk, backend=None) -> FusionResult:
    valid = [p for p in perceptions if p.valid_image]
    overall_valid = len(valid) > 0 and not any(
        ("non_original_image" in p.flags or "possible_manipulation" in p.flags) for p in valid
    )

    confirming = [
        p for p in valid
        if p.is_claimed_object and p.shows_claimed_part
        and p.issue_type not in ("none", "unknown")
        and _compatible(p.issue_type, intent.issue_type)
    ]
    identity_mismatch = any(
        (not p.is_claimed_object or "wrong_object" in p.flags) for p in valid
    )
    contradicting = [
        p for p in valid
        if p.is_claimed_object and p.shows_claimed_part
        and (p.issue_type == "none" or "damage_not_visible" in p.flags
             or (intent.issue_type not in ("", "unknown")
                 and p.issue_type not in ("none", "unknown")
                 and not _compatible(p.issue_type, intent.issue_type)))
    ]

    # ----- core decision (images first) ---------------------------------- #
    if not evidence.met or not valid:
        status = "not_enough_information"
    elif confirming:
        status = "supported"
    elif identity_mismatch or contradicting:
        status = "contradicted"
    else:
        status = "not_enough_information"

    # ----- optional Opus tie-breaker for ambiguous cases ----------------- #
    escalated = False
    ambiguous = (
        (status == "not_enough_information" and evidence.met and valid)
        or (confirming and (contradicting or identity_mismatch))
    )
    opus = None
    if ambiguous and backend is not None:
        opus = backend.fuse_escalate(claim, intent, perceptions, {
            "evidence_met": evidence.met, "evidence_reason": evidence.reason,
            "history_note": history.note,
        })
    if opus:
        status = opus.get("claim_status", status)
        escalated = True

    res = FusionResult(claim_status=status, escalated=escalated, valid_image=overall_valid)

    # ----- choose the reported content fields ---------------------------- #
    relevant_shown = [p for p in valid if p.shows_claimed_part]
    if status == "supported":
        best = confirming[0]
        res.issue_type = best.issue_type
        res.object_part = best.object_part or intent.object_part
        res.severity = best.severity
        res.supporting_image_ids = [p.image_id for p in confirming]
    elif status == "contradicted":
        pool = contradicting or [p for p in valid if not p.is_claimed_object] or relevant_shown
        best = pool[0] if pool else (valid[0] if valid else None)
        res.issue_type = best.issue_type if best else "unknown"
        res.object_part = (best.object_part if best and best.object_part != "unknown"
                           else intent.object_part)
        res.severity = best.severity if best else "unknown"
        res.supporting_image_ids = [best.image_id] if best else []
    else:  # not_enough_information
        res.object_part = intent.object_part
        res.issue_type = relevant_shown[0].issue_type if relevant_shown else "unknown"
        res.severity = "unknown"
        res.supporting_image_ids = [p.image_id for p in relevant_shown]

    if opus:  # let the tie-breaker override the content too
        res.issue_type = opus.get("issue_type", res.issue_type)
        res.object_part = opus.get("object_part", res.object_part)
        res.severity = opus.get("severity", res.severity)
        if opus.get("supporting_image_ids"):
            res.supporting_image_ids = opus["supporting_image_ids"]

    # ----- risk flags (quality + mismatch + authenticity + history) ------ #
    flags: list[str] = []
    for p in perceptions:
        for f in p.flags:
            if f not in flags:
                flags.append(f)
    if status == "contradicted" and (contradicting or identity_mismatch):
        if "claim_mismatch" not in flags:
            flags.append("claim_mismatch")
    if identity_mismatch and "wrong_object" not in flags:
        flags.append("wrong_object")
    if intent.conversation_text_instruction and "text_instruction_present" not in flags:
        flags.append("text_instruction_present")
    # history flags last — context only
    for f in history.flags:
        if f not in flags:
            flags.append(f)
    if ("claim_mismatch" in flags or status == "contradicted"
            or "possible_manipulation" in flags or "non_original_image" in flags):
        if "manual_review_required" not in flags:
            flags.append("manual_review_required")
    res.risk_flags = flags

    # ----- justification -------------------------------------------------- #
    if opus and opus.get("justification"):
        res.justification = opus["justification"]
    else:
        res.justification = _justify(status, res, intent, perceptions, history)

    log.debug("    [fusion] status=%s issue=%s part=%s sev=%s imgs=%s escalated=%s",
              status, res.issue_type, res.object_part, res.severity,
              res.supporting_image_ids or "none", escalated)
    return res


def _justify(status: str, res: FusionResult, intent: ClaimIntent,
             perceptions: list[PerceptionResult], history: HistoryRisk) -> str:
    ids = ", ".join(res.supporting_image_ids) if res.supporting_image_ids else None
    note = next((p.note for p in perceptions if p.image_id in res.supporting_image_ids and p.note), "")
    hist = f" User history adds risk: {history.note}." if history.flags else ""
    if status == "supported":
        base = (f"The image evidence ({ids}) supports the claim: "
                f"a {res.issue_type} is visible on the {res.object_part}.")
    elif status == "contradicted":
        base = (f"The image evidence ({ids}) contradicts the claim: "
                f"the {res.object_part} is visible but the claimed "
                f"{intent.issue_type} is not what is shown.")
    else:
        base = ("The submitted images do not provide enough information to verify "
                f"the claimed {intent.issue_type} on the {intent.object_part}.")
    if note:
        base += f" Observed: {note}"
    return (base + hist).strip()
