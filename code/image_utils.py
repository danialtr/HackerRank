"""Image decoding + normalization.

Several dataset files are AVIF/WEBP saved with a ``.jpg`` extension. Pillow
>= 11.3 decodes AVIF natively, so all we need is an up-to-date Pillow — no
extra plugins. If Pillow is too old, AVIF files fail to open; we detect that
case and return an actionable message instead of a cryptic error.

Before any vision call we still normalize every image to JPEG and downscale,
because the API does not accept AVIF and image tokens scale with resolution.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from PIL import Image, features

# Whether this Pillow build can decode AVIF (True for Pillow >= 11.3 wheels).
AVIF_SUPPORT = bool(features.check("avif"))

# Anthropic resizes images whose long edge exceeds ~1568px; sending larger just
# wastes tokens, so we downscale to this before encoding.
DEFAULT_MAX_EDGE = 1568


def _looks_like_avif_or_heif(path: Path) -> bool:
    """Sniff the ISO-BMFF brand so we can give a precise hint on decode failure."""
    try:
        head = path.read_bytes()[:16]
    except Exception:  # noqa: BLE001
        return False
    return head[4:8] == b"ftyp" and head[8:12] in (
        b"avif", b"avis", b"heic", b"heix", b"mif1", b"msf1",
    )


def probe_image(path: Path) -> tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    """Return (width, height, real_format, error) without decoding full pixels."""
    try:
        with Image.open(path) as im:
            return im.size[0], im.size[1], im.format, None
    except Exception as exc:  # noqa: BLE001
        if not AVIF_SUPPORT and _looks_like_avif_or_heif(path):
            return None, None, None, "AVIF/HEIF file but Pillow lacks AVIF support — run: pip install -U Pillow"
        return None, None, None, f"{type(exc).__name__}: {exc}"


def to_jpeg_bytes(
    path: Path,
    max_edge: int = DEFAULT_MAX_EDGE,
    quality: int = 85,
) -> tuple[bytes, str, int, int]:
    """Decode any supported image, downscale, and return (jpeg_bytes, media_type, w, h)."""
    with Image.open(path) as im:
        im = im.convert("RGB")  # drop alpha/palette; JPEG needs RGB
        w, h = im.size
        longest = max(w, h)
        if longest > max_edge:
            scale = max_edge / longest
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        out_w, out_h = im.size
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue(), "image/jpeg", out_w, out_h
