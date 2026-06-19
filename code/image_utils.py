"""Image decoding + normalization.

Why this module exists: in this dataset the file *extensions lie*. Many files
named ``.jpg`` are actually AVIF or WEBP. Two problems follow:

  1. A stock Pillow build cannot decode AVIF, so those images look "broken"
     even though they are fine (this is environment-dependent and a portability
     trap). We register optional AVIF/HEIF openers so decoding works anywhere.
  2. The Claude vision API accepts JPEG/PNG/GIF/WEBP — not AVIF. And image
     tokens scale with resolution. So before any model call we decode whatever
     the real format is, downscale, and re-encode to compact JPEG.

If the optional decoders are missing, we fail soft: real JPEG/PNG/WEBP still
work; only AVIF/HEIF would be unreadable, and the loader flags those clearly.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from PIL import Image

# --- Register optional formats so extension-mislabeled files decode anywhere ---
# pillow-avif-plugin ships self-contained wheels (incl. Windows) and registers
# AVIF with Pillow on import. pillow-heif is a fallback that also covers AVIF.
AVIF_SUPPORT = False
try:  # primary: AVIF plugin
    import pillow_avif  # noqa: F401  (import has the side effect of registering)
    AVIF_SUPPORT = True
except Exception:  # noqa: BLE001
    try:  # fallback: HEIF/AVIF opener
        from pillow_heif import register_heif_opener

        register_heif_opener()
        AVIF_SUPPORT = True
    except Exception:  # noqa: BLE001
        AVIF_SUPPORT = False

# Anthropic resizes images whose long edge exceeds ~1568px; sending anything
# larger just wastes tokens. We downscale to this by default.
DEFAULT_MAX_EDGE = 1568


def probe_image(path: Path) -> tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    """Return (width, height, real_format, error) without decoding full pixels."""
    try:
        with Image.open(path) as im:
            return im.size[0], im.size[1], im.format, None
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"{type(exc).__name__}: {exc}"


def to_jpeg_bytes(
    path: Path,
    max_edge: int = DEFAULT_MAX_EDGE,
    quality: int = 85,
) -> tuple[bytes, str, int, int]:
    """Decode any supported image, downscale, and return JPEG bytes.

    Returns (jpeg_bytes, media_type, out_width, out_height). The output is
    always ``image/jpeg`` and RGB, safe to hand to the vision API.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")  # drop alpha / palette; JPEG needs RGB
        w, h = im.size
        longest = max(w, h)
        if longest > max_edge:
            scale = max_edge / longest
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        out_w, out_h = im.size
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue(), "image/jpeg", out_w, out_h
