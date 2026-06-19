"""Stage 7: user-history risk — deterministic rules.

History adds *risk context only*. It can raise ``user_history_risk`` and
``manual_review_required`` flags, but the fusion stage is built so these can
never flip a clear visual verdict — that is the spec's core precedence rule, and
keeping history in plain, auditable code is how we prove we honoured it.
"""

from __future__ import annotations

from logging_setup import log
from models import HistoryRisk, UserHistory

# Thresholds for deriving risk from the history numbers.
REJECTION_RATIO = 0.30      # >= 30% of prior claims rejected
HIGH_VELOCITY = 4           # >= 4 claims in the last 90 days


def assess_history(history: UserHistory | None) -> HistoryRisk:
    risk = HistoryRisk()
    if history is None:
        risk.note = "no user history on file"
        return risk

    flags: list[str] = []
    declared = (history.history_flags or "none").lower()
    if "user_history_risk" in declared:
        flags.append("user_history_risk")
    if "manual_review_required" in declared:
        flags.append("manual_review_required")

    ratio = (history.rejected_claim / history.past_claim_count) if history.past_claim_count else 0.0
    if ratio >= REJECTION_RATIO and history.past_claim_count >= 3:
        if "user_history_risk" not in flags:
            flags.append("user_history_risk")
    if history.last_90_days_claim_count >= HIGH_VELOCITY:
        if "user_history_risk" not in flags:
            flags.append("user_history_risk")

    # A risky history warrants a manual look, but only as a flag — never a verdict.
    if "user_history_risk" in flags and "manual_review_required" not in flags:
        if ratio >= REJECTION_RATIO or history.manual_review_claim >= 2:
            flags.append("manual_review_required")

    risk.flags = flags
    risk.note = (history.history_summary or "").strip() or (
        f"{history.past_claim_count} prior claims, {history.rejected_claim} rejected"
    )
    log.debug("    [history] flags=%s ratio=%.2f velocity=%d",
              flags or "none", ratio, history.last_90_days_claim_count)
    return risk
