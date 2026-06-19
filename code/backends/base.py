"""The PerceptionBackend interface shared by the VLM and heuristic backends.

A backend is responsible only for *perception* — turning a conversation and an
image into structured signals. It never makes the final ruling; that is the job
of the deterministic fusion stage.
"""

from __future__ import annotations

import abc
from typing import Optional

from logging_setup import CostMeter
from models import Claim, ClaimIntent, ImageRef, PerceptionResult


class PerceptionBackend(abc.ABC):
    name: str = "base"

    def __init__(self, meter: CostMeter):
        self.meter = meter

    @abc.abstractmethod
    def extract_claim(self, claim: Claim) -> ClaimIntent:
        """Stage 2: parse the conversation into a structured claim intent."""

    @abc.abstractmethod
    def analyze_image(self, claim: Claim, image: ImageRef, intent: ClaimIntent) -> PerceptionResult:
        """Stage 3-5: describe one image (object, part, issue, quality, authenticity)."""

    def analyze_mega(self, claim: Claim) -> Optional[dict]:
        """Optional single-call 'mega-prompt' path (the ablation baseline)."""
        return None

    def close(self) -> None:
        pass
