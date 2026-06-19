"""Perception backends: the swappable "eyes" of the system.

Two implementations share one interface so the rest of the pipeline never
changes:

  * ``vlm``       — the real Claude vision engine (used when a key is present)
  * ``heuristic`` — a deterministic CV + text fallback (always available)

``build_backend`` picks one based on configuration and whether an API key is
available, so the same command runs end-to-end with or without credentials.
"""

from __future__ import annotations

import config
from logging_setup import CostMeter, log

from .base import PerceptionBackend


def build_backend(meter: CostMeter, force: str | None = None) -> PerceptionBackend:
    choice = (force or config.backend_choice()).lower()
    if choice == "auto":
        choice = "vlm" if config.has_api_key() else "heuristic"

    if choice == "vlm":
        if not config.has_api_key():
            log.warning("VLM backend requested but no ANTHROPIC_API_KEY — falling back to heuristic.")
            choice = "heuristic"
        else:
            from .vlm import VLMBackend
            log.info("Backend: VLM (tiered: %s / %s / %s)",
                     config.MODEL_EXTRACT, config.MODEL_PERCEPTION, config.MODEL_FUSION)
            return VLMBackend(meter)

    from .heuristic import HeuristicBackend
    log.info("Backend: heuristic (deterministic CV + claim-text parsing, no API calls)")
    return HeuristicBackend(meter)


__all__ = ["PerceptionBackend", "build_backend"]
