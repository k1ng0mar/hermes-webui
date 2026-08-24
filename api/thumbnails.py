"""Thumbnail serving for Nyx mobile artifact grid."""
import io
import mimetypes
import os
from pathlib import Path

THUMB_MAX = 320  # px longest edge


def mime_for(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def make_thumbnail(path: Path, max_size: int = THUMB_MAX) -> tuple[bytes, str] | None:
    """Return (bytes, mime) for a downscaled thumbnail, or None if unsupported."""
    ext = path.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        return None
    try:
        from PIL import Image, ImageOps
    except Exception:
        # No PIL — serve original bytes (still works, just not downscaled)
        return path.read_bytes(), mime_for(path.name)
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)
        im.thumbnail((max_size, max_size))
        if ext == ".png":
            fmt = "PNG"
        elif ext == ".webp":
            fmt = "WEBP"
        elif ext == ".gif":
            fmt = "GIF"
        else:
            fmt = "JPEG"
        buf = io.BytesIO()
        im.convert("RGBA" if fmt == "PNG" else "RGB").save(buf, fmt, quality=82)
        return buf.getvalue(), ("image/png" if fmt == "PNG" else f"image/{fmt.lower()}")
    except Exception:
        return None
