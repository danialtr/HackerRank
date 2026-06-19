"""Perception backend: the "eyes" of the system.

The system is VLM-only: perception is always done by the Claude vision backend.
``build_backend`` constructs it and requires an API key — there is no fallback.
"""

from __future__ import annotations

import config
from logging_setup import CostMeter, log

from .base import PerceptionBackend


def build_backend(meter: CostMeter) -> PerceptionBackend:
    if not config.has_api_key():
        log.error(
            "This system is VLM-only and requires an API key. Set ANTHROPIC_API_KEY "
            "(or create a .env file from .env.example) and re-run. See code/README.md."
        )
        raise SystemExit(2)

    from .vlm import VLMBackend
    log.info("Backend: VLM (tiered: %s extract / %s perception)",
             config.MODEL_EXTRACT, config.MODEL_PERCEPTION)
    return VLMBackend(meter)


__all__ = ["PerceptionBackend", "build_backend"]
