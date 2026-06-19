"""Deterministic perception backend — no API calls.

This backend is the always-available fallback (it runs with no API key) and the
cheap baseline in the evaluation ablation. It combines two deterministic signals:

  * classical computer vision over the pixels (OpenCV-style blur via Laplacian
    variance and brightness via the luminance histogram) for image-quality flags
  * keyword parsing of the claim conversation for the alleged issue and part

It cannot truly *recognise* what is in an image, so for content (object, part,
issue) it trusts the conversation — which is exactly why comparing it against the
VLM backend quantifies how much the images add over the text alone.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

import schema
from logging_setup import CostMeter, log
from models import Claim, ClaimIntent, ImageRef, PerceptionResult

from .base import PerceptionBackend

# --------------------------------------------------------------------------- #
# Keyword vocabularies for parsing the conversation
# --------------------------------------------------------------------------- #
ISSUE_KEYWORDS = {
    "glass_shatter": ["shatter", "shattered", "smashed glass"],
    "crack": ["crack", "cracked", "cracking", "chipped"],
    "dent": ["dent", "dented", "ding", "hail"],
    "scratch": ["scratch", "scrape", "scuff", "scraped", "mark on", "scratched"],
    "broken_part": ["broke", "broken", "snapped", "came off", "not sitting", "broke off", "broken off"],
    "missing_part": ["missing", "not inside", "gone", "lost", "absent"],
    "torn_packaging": ["torn", "tear", "ripped", "opened", "phati", "phat", "torn open"],
    "crushed_packaging": ["crush", "crushed", "crumpled", "smashed box", "caved"],
    "water_damage": ["water damage", "water-damaged", "wet", "soaked", "moisture", "water"],
    "stain": ["stain", "sticky", "discolor", "discolour"],
}

PART_KEYWORDS = {
    "car": {
        "front_bumper": ["front bumper"],
        "rear_bumper": ["rear bumper", "back bumper"],
        "windshield": ["windshield", "windscreen", "front glass"],
        "side_mirror": ["side mirror", "mirror", "wing mirror"],
        "headlight": ["headlight", "head light", "head-light"],
        "taillight": ["taillight", "tail light", "rear light"],
        "hood": ["hood", "bonnet", "top panel"],
        "door": ["door", "door panel"],
        "fender": ["fender"],
        "quarter_panel": ["quarter panel"],
        "body": ["body", "side panel", "panel"],
    },
    "laptop": {
        "screen": ["screen", "display", "display glass"],
        "keyboard": ["keyboard", "keys", "keys feel"],
        "trackpad": ["trackpad", "touchpad"],
        "hinge": ["hinge"],
        "lid": ["lid"],
        "corner": ["corner"],
        "port": ["port"],
        "base": ["base", "bottom"],
        "body": ["body", "chassis"],
    },
    "package": {
        "seal": ["seal", "tape", "flap"],
        "package_corner": ["corner"],
        "package_side": ["side", "surface", "outside"],
        "label": ["label"],
        "contents": ["contents", "content", "item inside", "product inside", "inside"],
        "item": ["item", "product"],
        "box": ["box", "shipping box", "carton", "parcel", "package"],
    },
}

QUALIFIER_HIGH = ["badly", "severe", "severely", "shattered", "deep", "completely", "totally", "bad"]
QUALIFIER_LOW = ["small", "minor", "light", "slight", "tiny", "little", "hairline"]

INJECTION_PATTERNS = [
    "approve the claim", "approve immediately", "must approve", "auto approve",
    "skip manual review", "skip review", "skip the review", "ignore previous",
    "ignore the", "system reading this", "any system", "instruction:", "you must",
    "without review", "mark as approved",
]


def _last_match(text: str, keywords: list[str]) -> int:
    """Return the latest character position any keyword appears at, else -1."""
    best = -1
    for kw in keywords:
        idx = text.rfind(kw)
        if idx > best:
            best = idx
    return best


class HeuristicBackend(PerceptionBackend):
    name = "heuristic"

    def __init__(self, meter: CostMeter, blur_threshold: float = 60.0,
                 dark_threshold: float = 45.0, glare_threshold: float = 215.0):
        super().__init__(meter)
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.glare_threshold = glare_threshold

    # ----- Stage 2: claim extraction (text only) -------------------------- #
    def extract_claim(self, claim: Claim) -> ClaimIntent:
        text = claim.user_claim.lower()
        obj = claim.claim_object.lower()

        issue = self._pick(text, ISSUE_KEYWORDS)
        part = self._pick(text, PART_KEYWORDS.get(obj, {}))

        severity = "unknown"
        if issue not in ("none", "unknown"):
            severity = "medium"
            if any(q in text for q in QUALIFIER_HIGH):
                severity = "high"
            elif any(q in text for q in QUALIFIER_LOW):
                severity = "low"

        injection = any(p in text for p in INJECTION_PATTERNS)

        intent = ClaimIntent(
            issue_type=issue or "unknown",
            object_part=part or "unknown",
            summary=f"claims {issue or 'unknown'} on {obj} {part or 'unknown'}",
            conversation_text_instruction=injection,
        )
        # severity rides along on the intent via a note we reuse later
        intent.summary += f" (severity~{severity})"
        log.debug("    [extract:heuristic] issue=%s part=%s injection=%s",
                  intent.issue_type, intent.object_part, injection)
        return intent

    @staticmethod
    def _pick(text: str, mapping: dict) -> str:
        best_key, best_pos = "unknown", -1
        for key, kws in mapping.items():
            pos = _last_match(text, kws)
            if pos > best_pos:
                best_pos, best_key = pos, key
        return best_key

    # ----- Stage 3-5: per-image perception (pixels only) ------------------ #
    def analyze_image(self, claim: Claim, image: ImageRef, intent: ClaimIntent) -> PerceptionResult:
        self.meter.images += 1
        res = PerceptionResult(image_id=image.image_id, backend="heuristic")

        if not image.usable:
            res.valid_image = False
            res.is_claimed_object = False
            res.shows_claimed_part = False
            res.issue_type = "unknown"
            res.flags = ["cropped_or_obstructed"] if image.load_error else []
            res.note = f"image not usable ({image.load_error or 'unknown'})"
            log.debug("    [vision:heuristic] %s unusable", image.image_id)
            return res

        blur, brightness = self._stats(image)
        flags: list[str] = []
        valid = True
        if blur < self.blur_threshold:
            flags.append("blurry_image")
            valid = False
        if brightness < self.dark_threshold or brightness > self.glare_threshold:
            flags.append("low_light_or_glare")

        # The heuristic cannot recognise content, so it trusts the claim text:
        # it assumes a usable image shows the claimed object and part.
        res.valid_image = valid
        res.is_claimed_object = True
        res.object_part = intent.object_part
        res.shows_claimed_part = valid
        res.issue_type = intent.issue_type if valid else "unknown"
        res.severity = self._severity_from_intent(intent)
        res.flags = flags
        res.note = (f"{image.width}x{image.height}, sharpness~{blur:.0f}, "
                    f"brightness~{brightness:.0f}")
        log.debug("    [vision:heuristic] %s blur=%.0f bright=%.0f flags=%s",
                  image.image_id, blur, brightness, flags or "none")
        return res

    @staticmethod
    def _severity_from_intent(intent: ClaimIntent) -> str:
        m = re.search(r"severity~(\w+)", intent.summary)
        return m.group(1) if m else "unknown"

    def _stats(self, image: ImageRef) -> tuple[float, float]:
        """Return (laplacian variance ~ sharpness, mean brightness 0-255)."""
        try:
            with Image.open(image.abs_path) as im:
                im = im.convert("L")
                w, h = im.size
                if max(w, h) > 512:
                    s = 512 / max(w, h)
                    im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
                arr = np.asarray(im, dtype=np.float64)
            brightness = float(arr.mean())
            lap = (
                -4 * arr
                + np.roll(arr, 1, 0) + np.roll(arr, -1, 0)
                + np.roll(arr, 1, 1) + np.roll(arr, -1, 1)
            )[1:-1, 1:-1]
            return float(lap.var()), brightness
        except Exception as exc:  # noqa: BLE001
            log.debug("    [vision:heuristic] stats failed for %s: %s", image.image_id, exc)
            return 1e9, 128.0  # treat as sharp/normal if we cannot measure
