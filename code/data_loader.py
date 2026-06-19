"""Load the dataset CSVs and images into typed objects (pipeline stage 1).

Responsibilities:
  * read claims / sample_claims / user_history into memory
  * parse the semicolon-separated ``image_paths`` and resolve each file
  * open each image to capture its size/format and flag broken/missing files
  * join each claim to its user history

Nothing here makes a model call or a decision; it only turns raw files into
clean, validated inputs for the rest of the system.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from PIL import Image

import config
from models import Claim, ImageRef, UserHistory

# Columns present in claims.csv (inputs only). Anything beyond these in
# sample_claims.csv is treated as the expected/gold output.
INPUT_COLUMNS = {"user_id", "image_paths", "user_claim", "claim_object"}


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default


def _image_id_from_path(rel_path: str) -> str:
    """Filename without extension, e.g. 'images/test/c1/img_1.jpg' -> 'img_1'."""
    return Path(rel_path).stem


def resolve_image(rel_path: str, dataset_root: Path) -> ImageRef:
    """Resolve one image path and probe the file without loading full pixels."""
    rel_path = rel_path.strip()
    abs_path = (dataset_root / rel_path).resolve()
    ref = ImageRef(
        image_id=_image_id_from_path(rel_path),
        rel_path=rel_path,
        abs_path=abs_path,
        exists=abs_path.is_file(),
    )
    if not ref.exists:
        ref.load_error = "file not found"
        return ref
    try:
        # Image.open is lazy; reading .size/.format triggers a header parse,
        # which is enough to confirm the file is a valid, openable image.
        with Image.open(abs_path) as im:
            ref.width, ref.height = im.size
            ref.fmt = im.format
    except Exception as exc:  # noqa: BLE001 - we want any decode failure flagged
        ref.load_error = f"{type(exc).__name__}: {exc}"
    return ref


def parse_image_paths(raw: str, dataset_root: Path) -> list[ImageRef]:
    """Split the semicolon-separated paths and resolve each one."""
    parts = [p for p in (raw or "").split(";") if p.strip()]
    return [resolve_image(p, dataset_root) for p in parts]


def load_user_history(path: Optional[Path] = None) -> dict[str, UserHistory]:
    """Index user_history.csv by user_id."""
    path = path or config.user_history_csv()
    histories: dict[str, UserHistory] = {}
    if not path.is_file():
        return histories
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            uid = (row.get("user_id") or "").strip()
            if not uid:
                continue
            histories[uid] = UserHistory(
                user_id=uid,
                past_claim_count=_to_int(row.get("past_claim_count")),
                accept_claim=_to_int(row.get("accept_claim")),
                manual_review_claim=_to_int(row.get("manual_review_claim")),
                rejected_claim=_to_int(row.get("rejected_claim")),
                last_90_days_claim_count=_to_int(row.get("last_90_days_claim_count")),
                history_flags=(row.get("history_flags") or "none").strip(),
                history_summary=(row.get("history_summary") or "").strip(),
            )
    return histories


def load_claims(
    path: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    histories: Optional[dict[str, UserHistory]] = None,
) -> list[Claim]:
    """Load claims, resolve their images, and join user history.

    Works for both claims.csv (inputs only) and sample_claims.csv (inputs +
    gold columns). Extra columns are captured into ``Claim.expected``.
    """
    path = path or config.claims_csv()
    dataset_root = dataset_root or config.dataset_dir()
    if histories is None:
        histories = load_user_history()

    claims: list[Claim] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        extra_cols = [c for c in (reader.fieldnames or []) if c not in INPUT_COLUMNS]
        for row in reader:
            uid = (row.get("user_id") or "").strip()
            raw_paths = (row.get("image_paths") or "").strip()
            claim = Claim(
                user_id=uid,
                image_paths_raw=raw_paths,
                user_claim=(row.get("user_claim") or "").strip(),
                claim_object=(row.get("claim_object") or "").strip().lower(),
                images=parse_image_paths(raw_paths, dataset_root),
                history=histories.get(uid),
                expected={c: row.get(c) for c in extra_cols} if extra_cols else None,
            )
            claims.append(claim)
    return claims
