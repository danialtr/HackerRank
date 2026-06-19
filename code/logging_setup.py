"""Runtime logging — a clear, human-readable trace of what the agent is doing.

The user asked for "a good log when the agent is running that shows what it is
doing and what happens in the middle of running". This module gives every run:

  * a timestamped console + file log (logs/run_<timestamp>.log)
  * a per-claim / per-stage trace (which image, which model call, what verdict)
  * a live token + cost meter accumulated across the run
  * a tidy end-of-run summary

Nothing here is decorative: the same log is what you read to debug a wrong
verdict, and the token/cost meter feeds the operational analysis in the
evaluation report.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import config

log = logging.getLogger("evidence_review")


def setup_logging(verbose: bool = False, logfile: Path | None = None) -> Path:
    """Configure console + file logging. Returns the path of the log file."""
    log.handlers.clear()
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    log.addHandler(console)

    if logfile is None:
        logs_dir = config.CODE_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        logfile = logs_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # the file always keeps full detail
    fh.setFormatter(fmt)
    log.addHandler(fh)

    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logfile


@dataclass
class CostMeter:
    """Accumulates model usage so we can report calls, tokens, and $ at the end."""

    calls: int = 0
    images: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    per_model: dict = field(default_factory=dict)

    def record(self, model: str, usage) -> None:
        """Add one API call's usage (an anthropic Usage object or a dict)."""
        def g(name: str) -> int:
            if usage is None:
                return 0
            if isinstance(usage, dict):
                return int(usage.get(name) or 0)
            return int(getattr(usage, name, 0) or 0)

        in_tok = g("input_tokens")
        out_tok = g("output_tokens")
        cr = g("cache_read_input_tokens")
        cw = g("cache_creation_input_tokens")

        price = config.PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (
            in_tok * price["input"]
            + cr * price["input"] * config.CACHE_READ_MULT
            + cw * price["input"] * config.CACHE_WRITE_MULT
            + out_tok * price["output"]
        ) / 1_000_000

        self.calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.cache_read_tokens += cr
        self.cache_write_tokens += cw
        self.cost_usd += cost

        m = self.per_model.setdefault(
            model, {"calls": 0, "input": 0, "output": 0, "cost": 0.0}
        )
        m["calls"] += 1
        m["input"] += in_tok + cr + cw
        m["output"] += out_tok
        m["cost"] += cost

    def summary(self) -> dict:
        return {
            "model_calls": self.calls,
            "images_processed": self.images,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "per_model": self.per_model,
        }
