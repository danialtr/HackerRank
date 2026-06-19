"""Entry point — Multi-Modal Evidence Review.

Stage 1 milestone: load the inputs (the claim conversation + its images, joined
with user history) and show that everything resolves cleanly. Later stages
(perception, evidence check, decision) plug in after this foundation.

Usage:
    python code/main.py                 # load test set (dataset/claims.csv)
    python code/main.py --split sample  # load labeled sample set
    python code/main.py --index 3       # inspect a specific claim
    python code/main.py --check         # validate that every image resolves
"""

from __future__ import annotations

import argparse
import textwrap

import PIL

import config
from data_loader import load_claims, load_user_history
from image_utils import AVIF_SUPPORT
from models import Claim


def _short(text: str, width: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def show_claim(claim: Claim, index: int) -> None:
    """Pretty-print one loaded claim and its resolved images."""
    print(f"\n=== Claim #{index}  ({claim.user_id} · {claim.claim_object}) ===")
    print("Conversation:")
    for turn in claim.conversation_turns():
        print("   " + "\n   ".join(textwrap.wrap(turn, width=92)))

    print(f"\nImages ({len(claim.usable_images)}/{len(claim.images)} usable):")
    for im in claim.images:
        if im.usable:
            print(f"   ✓ {im.image_id:8s} {im.width}x{im.height} {im.fmt}  ({im.rel_path})")
        else:
            print(f"   ✗ {im.image_id:8s} [{im.load_error}]  ({im.rel_path})")

    if claim.history:
        h = claim.history
        print(
            f"\nHistory: {h.past_claim_count} claims "
            f"(accepted={h.accept_claim}, review={h.manual_review_claim}, "
            f"rejected={h.rejected_claim}, last90d={h.last_90_days_claim_count}); "
            f"flags={h.history_flags}"
        )
        if h.history_summary:
            print(f"   {_short(h.history_summary, 110)}")
    else:
        print("\nHistory: (no matching user_history row)")

    if claim.expected:  # only present for sample_claims.csv
        print("\nExpected (gold):")
        for key in ("claim_status", "issue_type", "object_part", "severity",
                    "evidence_standard_met", "valid_image", "risk_flags",
                    "supporting_image_ids"):
            if key in claim.expected:
                print(f"   {key:22s}= {claim.expected[key]}")


def check_dataset(claims: list[Claim]) -> int:
    """Report any missing/broken images across the whole set. Returns problem count."""
    total_imgs = sum(len(c.images) for c in claims)
    broken = [(c, im) for c in claims for im in c.images if not im.usable]
    print(f"\nIntegrity check: {len(claims)} claims, {total_imgs} images, "
          f"{total_imgs - len(broken)} usable, {len(broken)} problems.")
    for c, im in broken:
        print(f"   ✗ {c.user_id}: {im.rel_path} -> {im.load_error}")
    return len(broken)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review — loader")
    parser.add_argument("--split", choices=("test", "sample"), default="test",
                        help="which claims file to load (default: test)")
    parser.add_argument("--index", type=int, default=0,
                        help="index of the claim to display in detail")
    parser.add_argument("--limit", type=int, default=1,
                        help="how many claims to display from --index (default: 1)")
    parser.add_argument("--check", action="store_true",
                        help="validate that every referenced image resolves and opens")
    args = parser.parse_args()

    path = config.sample_claims_csv() if args.split == "sample" else config.claims_csv()
    avif = "yes" if AVIF_SUPPORT else "NO  -> run: pip install -U Pillow"
    print(f"Pillow      : {PIL.__version__}  (AVIF decode: {avif})")
    print(f"Dataset dir : {config.dataset_dir()}")
    print(f"Claims file : {path}")

    histories = load_user_history()
    claims = load_claims(path=path, histories=histories)
    print(f"Loaded      : {len(claims)} claims, {len(histories)} user-history rows")

    if args.check:
        problems = check_dataset(claims)
        raise SystemExit(1 if problems else 0)

    if not claims:
        print("No claims loaded.")
        return

    start = max(0, args.index)
    for offset in range(args.limit):
        i = start + offset
        if i >= len(claims):
            break
        show_claim(claims[i], i)


if __name__ == "__main__":
    main()
