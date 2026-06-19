"""Scoring metrics for the evaluation harness.

We score the 10 predicted columns against the gold labels in sample_claims.csv:
  * exact (normalised) accuracy for the scalar columns
  * a 3-class confusion matrix for claim_status (the headline metric)
  * set-overlap (Jaccard) for the two list columns: supporting_image_ids and
    risk_flags
"""

from __future__ import annotations

# The columns the system predicts (everything except the 4 input columns).
SCORED_COLUMNS = [
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]
EXACT_COLUMNS = [
    "evidence_standard_met", "issue_type", "object_part", "claim_status",
    "valid_image", "severity",
]
SET_COLUMNS = ["risk_flags", "supporting_image_ids"]
CLAIM_STATUS_LABELS = ["supported", "contradicted", "not_enough_information"]


def _norm(v) -> str:
    return (str(v) if v is not None else "").strip().lower()


def _as_set(v) -> set[str]:
    s = _norm(v)
    if not s or s == "none":
        return set()
    return {p.strip() for p in s.split(";") if p.strip() and p.strip() != "none"}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def score(preds: list[dict], golds: list[dict]) -> dict:
    assert len(preds) == len(golds), "preds/golds length mismatch"
    n = len(preds)
    result: dict = {"n": n, "exact": {}, "set_jaccard": {}}

    for col in EXACT_COLUMNS:
        correct = sum(1 for p, g in zip(preds, golds) if _norm(p.get(col)) == _norm(g.get(col)))
        result["exact"][col] = round(correct / n, 4) if n else 0.0

    for col in SET_COLUMNS:
        total = sum(_jaccard(_as_set(p.get(col)), _as_set(g.get(col))) for p, g in zip(preds, golds))
        result["set_jaccard"][col] = round(total / n, 4) if n else 0.0

    # 3-class confusion matrix for claim_status: matrix[gold][pred]
    matrix = {g: {p: 0 for p in CLAIM_STATUS_LABELS} for g in CLAIM_STATUS_LABELS}
    for p, g in zip(preds, golds):
        gv, pv = _norm(g.get("claim_status")), _norm(p.get("claim_status"))
        if gv in matrix and pv in matrix[gv]:
            matrix[gv][pv] += 1
    result["claim_status_confusion"] = matrix

    # macro F1 over claim_status
    f1s = []
    for label in CLAIM_STATUS_LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in CLAIM_STATUS_LABELS if g != label)
        fn = sum(matrix[label][p] for p in CLAIM_STATUS_LABELS if p != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1s.append(f1)
    result["claim_status_macro_f1"] = round(sum(f1s) / len(f1s), 4)

    # one headline number: mean of exact + set columns
    parts = list(result["exact"].values()) + list(result["set_jaccard"].values())
    result["overall_score"] = round(sum(parts) / len(parts), 4) if parts else 0.0
    return result


def format_confusion(matrix: dict) -> str:
    labels = CLAIM_STATUS_LABELS
    short = {"supported": "sup", "contradicted": "con", "not_enough_information": "nei"}
    header = "gold\\pred   " + "  ".join(f"{short[l]:>5}" for l in labels)
    lines = [header]
    for g in labels:
        row = "  ".join(f"{matrix[g][p]:>5}" for p in labels)
        lines.append(f"{short[g]:>9}   {row}")
    return "\n".join(lines)
