"""Run the full per-claim pipeline and assemble normalised output rows.

Two architectures are supported (the evaluation ablation compares them):
  * "pipeline" — the multi-stage design (extract → perceive → evidence →
    history → fuse). This is the default.
  * "mega" — a single mega-prompt call per claim (VLM only); used as a baseline.

Every field that leaves this module is funnelled through schema normalisers, so
the output is schema-valid by construction.
"""

from __future__ import annotations

import time

import config
import schema
from logging_setup import log
from models import Claim
from pipeline.evidence import decide_evidence, load_requirements
from pipeline.fuse import fuse
from pipeline.history_risk import assess_history


def _row_from_fusion(claim: Claim, fusion, evidence) -> dict:
    met = fusion.claim_status != "not_enough_information"
    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths_raw,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": schema.norm_bool(met),
        "evidence_standard_met_reason": evidence.reason,
        "risk_flags": schema.norm_risk_flags(fusion.risk_flags),
        "issue_type": schema.norm_issue_type(fusion.issue_type),
        "object_part": schema.norm_object_part(fusion.object_part, claim.claim_object),
        "claim_status": schema.norm_claim_status(fusion.claim_status),
        "claim_status_justification": (fusion.justification or "").strip()[:600],
        "supporting_image_ids": schema.norm_image_ids(fusion.supporting_image_ids),
        "valid_image": schema.norm_bool(fusion.valid_image),
        "severity": schema.norm_severity(fusion.severity),
    }


def _row_from_mega(claim: Claim, mega: dict, history) -> dict:
    flags = list(mega.get("risk_flags") or [])
    for f in history.flags:                      # precedence: history adds context only
        if f not in flags:
            flags.append(f)
    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths_raw,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": schema.norm_bool(mega.get("evidence_standard_met")),
        "evidence_standard_met_reason": (mega.get("evidence_standard_met_reason") or "").strip()[:600],
        "risk_flags": schema.norm_risk_flags(flags),
        "issue_type": schema.norm_issue_type(mega.get("issue_type", "unknown")),
        "object_part": schema.norm_object_part(mega.get("object_part", "unknown"), claim.claim_object),
        "claim_status": schema.norm_claim_status(mega.get("claim_status", "not_enough_information")),
        "claim_status_justification": (mega.get("claim_status_justification") or "").strip()[:600],
        "supporting_image_ids": schema.norm_image_ids(mega.get("supporting_image_ids") or []),
        "valid_image": schema.norm_bool(mega.get("valid_image")),
        "severity": schema.norm_severity(mega.get("severity", "unknown")),
    }


def process_claim(claim: Claim, backend, requirements: dict, arch: str) -> dict:
    n_imgs = len(claim.images)
    if arch == "mega":
        mega = backend.analyze_mega(claim)
        if mega is not None:
            history = assess_history(claim.history)
            row = _row_from_mega(claim, mega, history)
            log.info("  %-9s %-7s | %d img | %-22s | %s (mega)", claim.user_id,
                     claim.claim_object, n_imgs, row["claim_status"], row["issue_type"])
            return row
        log.warning("  mega path unavailable for %s; using pipeline", claim.user_id)

    intent = backend.extract_claim(claim)
    perceptions = [backend.analyze_image(claim, im, intent) for im in claim.images]
    evidence = decide_evidence(claim, intent, perceptions, requirements)
    history = assess_history(claim.history)
    fusion = fuse(claim, intent, perceptions, evidence, history, backend=backend)
    row = _row_from_fusion(claim, fusion, evidence)

    problems = schema.validate_row(row)
    if problems:
        log.warning("  schema issues for %s: %s", claim.user_id, problems)
    log.info("  %-9s %-7s | %d img | %-22s | issue=%s part=%s sev=%s flags=%s%s",
             claim.user_id, claim.claim_object, n_imgs, row["claim_status"],
             row["issue_type"], row["object_part"], row["severity"],
             row["risk_flags"], " [escalated]" if fusion.escalated else "")
    return row


def run(claims: list[Claim], backend, arch: str | None = None) -> list[dict]:
    arch = arch or config.architecture()
    requirements = load_requirements()
    log.info("Loaded %d evidence requirements; architecture=%s", len(requirements), arch)
    rows: list[dict] = []
    t0 = time.time()
    for i, claim in enumerate(claims, 1):
        log.info("[%d/%d] processing claim by %s ...", i, len(claims), claim.user_id)
        rows.append(process_claim(claim, backend, requirements, arch))
    log.info("Processed %d claims in %.1fs", len(claims), time.time() - t0)
    return rows
