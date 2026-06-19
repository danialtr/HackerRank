#!/usr/bin/env python3
"""Run the pipeline on one or a few CHOSEN claims — for quick testing.

Unlike `main.py` (which processes the whole split), this lets you pick exactly
which claims to run, by user_id, by case folder, or by 1-based row number. It
defaults to the labeled `sample` split and prints the prediction side-by-side
with the gold answer so you can see what matches.

Examples:
    python code/run-sample.py --list                     # show selectable claims, then exit
    python code/run-sample.py --ids user_005             # one claim by user_id
    python code/run-sample.py --ids user_001,user_005    # several claims
    python code/run-sample.py --case case_008            # by image case folder
    python code/run-sample.py --index 1,4,7              # by 1-based row number
    python code/run-sample.py --split test --ids user_002   # a test-split claim (no gold)
    python code/run-sample.py --ids user_005 --backend vlm -v
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

# This file is in code/, so its directory is on sys.path[0] when run directly.
import config
import schema
from backends import build_backend
from data_loader import load_claims, load_user_history
from logging_setup import CostMeter, log, setup_logging
from pipeline.evidence import load_requirements
from pipeline.orchestrator import process_claim

SCORED = [
    "claim_status", "evidence_standard_met", "issue_type", "object_part",
    "severity", "valid_image", "supporting_image_ids", "risk_flags",
]
SET_COLS = {"supporting_image_ids", "risk_flags"}


def case_of(claim) -> str:
    """Parse the case folder (e.g. 'case_005') from the first image path."""
    m = re.search(r"case_\d+", claim.image_paths_raw or "")
    return m.group(0) if m else ""


def _norm(v) -> str:
    return (str(v) if v is not None else "").strip().lower()


def _set(v) -> set[str]:
    s = _norm(v)
    return set() if (not s or s == "none") else {p.strip() for p in s.split(";") if p.strip()}


def select(claims, args) -> list:
    ids = {x.strip() for x in (args.ids or "").split(",") if x.strip()}
    cases = {x.strip() for x in (args.case or "").split(",") if x.strip()}
    idxs = {int(x) for x in (args.index or "").replace(" ", "").split(",") if x.strip().isdigit()}
    if not (ids or cases or idxs) and not args.all:
        return []
    chosen = []
    for i, c in enumerate(claims, 1):
        if args.all or c.user_id in ids or case_of(c) in cases or i in idxs:
            chosen.append((i, c))
    return chosen


def print_listing(claims) -> None:
    print(f"{'#':>3}  {'user_id':<9}  {'case':<10}  {'object':<8}  claim (truncated)")
    print("-" * 92)
    for i, c in enumerate(claims, 1):
        chat = (c.user_claim or "").replace("\n", " ")[:55]
        print(f"{i:>3}  {c.user_id:<9}  {case_of(c):<10}  {c.claim_object:<8}  {chat}")


def show(idx: int, claim, row: dict) -> int:
    print("\n" + "=" * 92)
    gold = claim.expected
    print(f"[{claim.user_id} / {case_of(claim) or 'n/a'} / {claim.claim_object}]  (row {idx})")
    chat = (claim.user_claim or "").replace("\n", " ")
    print(f"claim   : {chat[:300]}{'...' if len(chat) > 300 else ''}")
    print(f"images  : {', '.join(im.image_id for im in claim.images) or 'none'} "
          f"({len(claim.usable_images)} usable)")
    print("-" * 92)
    print("PREDICTION")
    for col in SCORED:
        print(f"  {col:<24}: {row.get(col, '')}")
    print(f"  {'evidence_reason':<24}: {row.get('evidence_standard_met_reason', '')[:80]}")
    print(f"  {'justification':<24}: {row.get('claim_status_justification', '')[:100]}")

    if not gold:
        print("(test split — no gold answer to compare against)")
        return -1

    print("-" * 92)
    print(f"{'COMPARE vs GOLD':<26} {'predicted':<22} {'gold':<22} ok")
    correct = 0
    for col in SCORED:
        p, g = row.get(col, ""), gold.get(col, "")
        ok = (_set(p) == _set(g)) if col in SET_COLS else (_norm(p) == _norm(g))
        correct += int(ok)
        print(f"  {col:<24}: {str(p)[:20]:<22} {str(g)[:20]:<22} {'OK' if ok else 'x'}")
    print(f"  -> {correct}/{len(SCORED)} columns match")
    return correct


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the pipeline on chosen claims.")
    ap.add_argument("--split", choices=("sample", "test"), default="sample")
    ap.add_argument("--ids", help="comma-separated user_ids, e.g. user_001,user_005")
    ap.add_argument("--case", help="comma-separated case folders, e.g. case_005,case_008")
    ap.add_argument("--index", help="comma-separated 1-based row numbers, e.g. 1,4,7")
    ap.add_argument("--all", action="store_true", help="run every claim in the split")
    ap.add_argument("--list", action="store_true", help="list selectable claims and exit")
    ap.add_argument("--backend", choices=("auto", "vlm", "heuristic"), default=None)
    ap.add_argument("-v", "--verbose", action="store_true", help="show full per-stage pipeline trace")
    args = ap.parse_args()

    setup_logging(args.verbose)
    if not args.verbose:
        logging.getLogger("evidence_review").setLevel(logging.WARNING)  # keep output focused

    path = config.sample_claims_csv() if args.split == "sample" else config.claims_csv()
    histories = load_user_history()
    claims = load_claims(path=path, histories=histories)

    if args.list:
        print_listing(claims)
        return

    chosen = select(claims, args)
    if not chosen:
        print("No claims selected. Choose with --ids / --case / --index (or --all), "
              "or run --list to see what is available.\n")
        print_listing(claims)
        sys.exit(1)

    meter = CostMeter()
    backend = build_backend(meter, force=args.backend)
    requirements = load_requirements()
    arch = config.architecture()

    totals = []
    for idx, claim in chosen:
        row = process_claim(claim, backend, requirements, arch)
        c = show(idx, claim, row)
        if c >= 0:
            totals.append(c)
    backend.close()

    print("\n" + "=" * 92)
    if totals:
        print(f"Summary: {sum(totals)}/{len(totals) * len(SCORED)} scored cells matched "
              f"across {len(totals)} labeled claim(s).")
    s = meter.summary()
    print(f"Backend: {backend.name} | model calls: {s['model_calls']} | "
          f"images: {s['images_processed']} | cost: ${s['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
