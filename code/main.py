"""Entry point — Multi-Modal Evidence Review.

Reads the dataset (claims + images + user history + evidence requirements), runs
each claim through the Claude VLM perception → deterministic decision pipeline,
and writes output.csv with the exact 14-column schema. The system is VLM-only and
requires ANTHROPIC_API_KEY (set it directly or via a .env file).

Usage:
    python code/main.py                       # test split -> output.csv
    python code/main.py --split sample        # labeled sample split
    python code/main.py --arch mega           # single mega-prompt ablation
    python code/main.py -v                     # per-stage debug detail
    python code/main.py --limit 3             # process only the first 3 claims
"""

from __future__ import annotations

import argparse
import csv
import json

import PIL

import config
import schema
from backends import build_backend
from data_loader import load_claims, load_user_history
from image_utils import AVIF_SUPPORT
from logging_setup import CostMeter, log, setup_logging
from pipeline.orchestrator import run as run_pipeline


def write_output(rows: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=schema.OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in schema.OUTPUT_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review")
    parser.add_argument("--split", choices=("test", "sample"), default="test")
    parser.add_argument("--arch", choices=("pipeline", "mega"), default=None)
    parser.add_argument("--output", default=None, help="output CSV path (default: <repo>/output.csv)")
    parser.add_argument("--limit", type=int, default=0, help="process only the first N claims")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logfile = setup_logging(args.verbose)
    log.info("=== Multi-Modal Evidence Review ===")
    log.info("Pillow %s (AVIF decode: %s); log file: %s", PIL.__version__,
             "yes" if AVIF_SUPPORT else "NO", logfile)

    path = config.sample_claims_csv() if args.split == "sample" else config.claims_csv()
    histories = load_user_history()
    claims = load_claims(path=path, histories=histories)
    if args.limit:
        claims = claims[: args.limit]
    log.info("Loaded %d user-history rows and %d claims from %s",
             len(histories), len(claims), path.name)

    total = sum(len(c.images) for c in claims)
    usable = sum(len(c.usable_images) for c in claims)
    log.info("Image integrity: %d/%d usable across %d claims", usable, total, len(claims))

    meter = CostMeter()
    backend = build_backend(meter)
    arch = args.arch or config.architecture()

    rows = run_pipeline(claims, backend, arch=arch)
    backend.close()

    if args.output:
        from pathlib import Path
        out_path = Path(args.output).resolve()
    else:
        out_path = config.output_csv_path()
    write_output(rows, out_path)
    log.info("Wrote %d rows to %s", len(rows), out_path)

    # ----- operational summary ------------------------------------------- #
    summary = meter.summary()
    summary["backend"] = backend.name
    summary["architecture"] = arch
    log.info("Operational summary: %s", json.dumps(summary, indent=2))

    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["claim_status"]] = statuses.get(r["claim_status"], 0) + 1
    log.info("claim_status distribution: %s", statuses)


if __name__ == "__main__":
    main()
