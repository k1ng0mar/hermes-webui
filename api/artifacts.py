"""Artifact listing for Nyx mobile.

Claude-Artifact-style: type-aware, recency-sorted, preview-able.
Filters out workspace noise (tarballs, backups, binaries).
"""
import os
import mimetypes
from pathlib import Path

# Extensions we treat as preview-able "real" artifacts
PREVIEW_EXT = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".svg": "image", ".bmp": "image",
    ".html": "html", ".htm": "html",
    ".md": "markdown", ".markdown": "markdown",
    ".ts": "code", ".tsx": "code", ".js": "code", ".jsx": "code",
    ".py": "code", ".go": "code", ".rs": "code", ".c": "code",
    ".cpp": "code", ".sh": "code", ".css": "code", ".json": "code",
    ".yaml": "code", ".yml": "code", ".sql": "code",
}

# Size cap: don't try to preview giant files
MAX_PREVIEW_SIZE = 5 * 1024 * 1024  # 5 MB

# Extension -> kind label shown on tiles
def kind_of(name: str) -> str:
    ext = Path(name).suffix.lower()
    return PREVIEW_EXT.get(ext, "file")


def is_noise(name: str) -> bool:
    """Tarballs, backups, archives, hidden, lockfiles — not artifacts."""
    low = name.lower()
    if low.startswith("."):
        return True
    noise = (".tar", ".tar.gz", ".tgz", ".zip", ".gz", ".bz2", ".xz",
             ".bak", ".backup", ".lock", ".tmp", ".part", ".log")
    for n in noise:
        if low.endswith(n):
            return True
    if "backup" in low:
        return True
    return False


def artifact_from_entry(entry: dict) -> dict | None:
    """Convert a workspace list entry to an artifact dict (or None if noise)."""
    name = entry.get("name", "")
    if entry.get("type") == "dir" or entry.get("is_dir"):
        return None
    if is_noise(name):
        return None
    size = entry.get("size") or 0
    mtime_ns = entry.get("mtime_ns")
    kind = kind_of(name)
    try:
        mtime_ms = (float(mtime_ns) / 1_000_000) if mtime_ns else None
    except (TypeError, ValueError):
        mtime_ms = None
    return {
        "id": entry.get("path") or name,
        "title": name,
        "kind": kind,
        "size": size,
        "mtime": mtime_ms,
        "previewable": kind != "file" and size <= MAX_PREVIEW_SIZE,
    }
