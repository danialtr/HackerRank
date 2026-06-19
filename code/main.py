"""Entry point — Multi-Modal Evidence Review.

Loads the inputs (claims + images + user history) and logs progress. Later
pipeline stages (perception, evidence, decision) hook into this same flow.

Usage:
    python code/main.py                 # load test split (dataset/claims.csv)
    python code/main.py --split sample  # load labeled sample split
    python code/main.py -v              # per-claim debug detail
"""

from __future__ import annotations

import argparse
import logging
import sys

import PIL

import config
from data_loader import load_claims, load_user_history
from image_utils import AVIF_SUPPORT

log = logging.getLogger("evidence_review")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Keep third-party debug chatter (e.g. Pillow's plugin imports) out of our log.
    logging.getLogger("PIL").setLevel(logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review")
    parser.add_argument("--split", choices=("test", "sample"), default="test",
                        help="which claims file to load (default: test)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="log every claim as it is processed")
    args = parser.parse_args()
    setup_logging(args.verbose)

    log.info("Pillow %s (AVIF decode: %s)", PIL.__version__,
             "yes" if AVIF_SUPPORT else "NO - run: pip install -U Pillow")

    path = config.sample_claims_csv() if args.split == "sample" else config.claims_csv()
    log.info("Loading '%s' split from %s", args.split, config.dataset_dir())

    histories = load_user_history()
    log.info("Loaded %d user-history rows", len(histories))

    claims = load_claims(path=path, histories=histories)
    log.info("Loaded %d claims from %s", len(claims), path.name)

    total = usable = problems = 0
    for i, c in enumerate(claims, 1):
        total += len(c.images)
        usable += len(c.usable_images)
        log.debug("[%d/%d] %s · %s · %d image(s)", i, len(claims),
                  c.user_id, c.claim_object, len(c.images))
        for im in c.images:
            if not im.usable:
                problems += 1
                log.warning("%s: image '%s' unusable (%s)",
                            c.user_id, im.image_id, im.load_error)

    log.info("Image integrity: %d/%d usable, %d problem(s)", usable, total, problems)
    log.info("Done loading '%s' split.", args.split)


if __name__ == "__main__":
    main()
