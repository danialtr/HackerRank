"""Typed data structures for the claim-verification pipeline.

These are plain dataclasses, deliberately light. The loader (``data_loader.py``)
fills them from the dataset CSVs and images; later pipeline stages read from
them. Keeping the inputs in typed objects means the rest of the system never
has to re-parse raw CSV strings or re-resolve image paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ImageRef:
    """One submitted image, resolved against the dataset folder.

    ``image_id`` is the filename without extension (e.g. ``img_1``) which is the
    identifier the output schema expects in ``supporting_image_ids``.
    """

    image_id: str
    rel_path: str            # exactly as written in the CSV
    abs_path: Path           # resolved on disk
    exists: bool             # file present?
    width: Optional[int] = None
    height: Optional[int] = None
    fmt: Optional[str] = None       # e.g. "JPEG"
    load_error: Optional[str] = None  # set if the file could not be opened

    @property
    def usable(self) -> bool:
        """True when the file exists and opened cleanly with sane dimensions."""
        return self.exists and self.load_error is None and bool(self.width) and bool(self.height)


@dataclass
class UserHistory:
    """A row from user_history.csv. Used later only for *risk context*."""

    user_id: str
    past_claim_count: int = 0
    accept_claim: int = 0
    manual_review_claim: int = 0
    rejected_claim: int = 0
    last_90_days_claim_count: int = 0
    history_flags: str = "none"
    history_summary: str = ""


@dataclass
class Claim:
    """One row from claims.csv (or sample_claims.csv), with inputs joined.

    ``expected`` holds the gold output columns when loaded from the labeled
    sample file; it is ``None`` for the test set.
    """

    user_id: str
    image_paths_raw: str
    user_claim: str
    claim_object: str                 # car | laptop | package
    images: list[ImageRef] = field(default_factory=list)
    history: Optional[UserHistory] = None
    expected: Optional[dict] = None   # gold columns from sample_claims.csv

    def conversation_turns(self) -> list[str]:
        """Split the chat transcript on ' | ' into individual turns."""
        return [t.strip() for t in self.user_claim.split("|") if t.strip()]

    @property
    def usable_images(self) -> list[ImageRef]:
        return [im for im in self.images if im.usable]
