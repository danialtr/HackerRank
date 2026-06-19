"""Dynamic few-shot retrieval from sample_claims.csv.

Classic vector-DB RAG is overkill for ~20 labeled rows. The genuinely useful
retrieval here is structured: for the claim in hand, pull a couple of labeled
examples that match the same object type (and, when possible, the same issue
family) and inject them to calibrate the supported / contradicted / not-enough
boundary and the severity scale. This is VLM-only and entirely optional — the
deterministic pipeline does not depend on it.
"""

from __future__ import annotations

from typing import Optional

from data_loader import load_claims
from logging_setup import log

# Map an issue to a coarse family so "crack" can retrieve a "glass_shatter" example.
_FAMILY = {
    "dent": "surface", "scratch": "surface",
    "crack": "break", "glass_shatter": "break", "broken_part": "break", "missing_part": "break",
    "torn_packaging": "package", "crushed_packaging": "package",
    "water_damage": "stain", "stain": "stain",
}


class FewShotRetriever:
    def __init__(self, max_examples: int = 2):
        self.max_examples = max_examples
        self.rows: list = []
        try:
            self.rows = [c for c in load_claims(path=None, histories={}) if c.expected]
        except Exception as exc:  # noqa: BLE001
            log.debug("few-shot retrieval unavailable: %s", exc)

    @classmethod
    def from_sample(cls, max_examples: int = 2) -> "FewShotRetriever":
        import config
        r = cls(max_examples)
        r.rows = []
        try:
            r.rows = [c for c in load_claims(path=config.sample_claims_csv(), histories={}) if c.expected]
        except Exception as exc:  # noqa: BLE001
            log.debug("few-shot retrieval unavailable: %s", exc)
        return r

    def examples_text(self, claim_object: str, issue_hint: str = "") -> Optional[str]:
        if not self.rows:
            return None
        fam = _FAMILY.get(issue_hint, "")
        same_obj = [c for c in self.rows if c.claim_object == claim_object]
        scored = []
        for c in same_obj:
            exp = c.expected or {}
            score = 1
            if fam and _FAMILY.get((exp.get("issue_type") or "").strip(), "") == fam:
                score += 1
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        picks = [c for _s, c in scored[: self.max_examples]]
        if not picks:
            return None

        lines = ["Reference examples (labeled), for calibration only:"]
        for c in picks:
            e = c.expected
            lines.append(
                f"- {c.claim_object}: claim '{c.user_claim[:80]}...' -> "
                f"status={e.get('claim_status')}, issue={e.get('issue_type')}, "
                f"part={e.get('object_part')}, severity={e.get('severity')}"
            )
        return "\n".join(lines)
