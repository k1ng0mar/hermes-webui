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


# Directories that are never artifact output, only machinery. Walking into
# node_modules or .git turns a 40-file workspace into a 400,000-file one.
SKIP_DIRS = {
    "node_modules", ".git", ".hg", ".svn", "__pycache__", ".venv", "venv",
    "env", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".next", ".expo", ".gradle", ".idea", ".vscode", "target", "vendor",
    "Pods", ".cache", ".tox", ".terraform", "site-packages",
}

# How deep to look. Agents write reports/ and docs/ one or two levels down;
# past that you are in source trees, not artifacts.
MAX_DEPTH = 3
# Hard ceiling on files examined, so a wrong workspace root cannot hang the
# request while it walks a home directory.
SCAN_LIMIT = 4000


def walk_artifacts(root: Path, max_depth: int = MAX_DEPTH, limit: int = SCAN_LIMIT) -> list[dict]:
    """Artifacts under ``root``, recursively.

    The listing used to be one non-recursive `list_dir(root, ".")`, so anything
    the agent wrote into a subdirectory — which is most of what it writes — was
    invisible. Symlinks are skipped outright rather than resolved: following
    them is how a walk escapes the workspace it is supposed to be bounded by.
    """
    out: list[dict] = []
    seen = 0
    root = Path(root)

    def walk(d: Path, rel: str, depth: int) -> None:
        nonlocal seen
        if depth > max_depth or seen >= limit:
            return
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            return
        for e in entries:
            if seen >= limit:
                return
            seen += 1
            name = e.name
            if name.startswith("."):
                continue
            try:
                if e.is_symlink():
                    continue
                if e.is_dir():
                    if name in SKIP_DIRS:
                        continue
                    walk(Path(e.path), f"{rel}{name}/", depth + 1)
                    continue
                if not e.is_file():
                    continue
                if is_noise(name):
                    continue
                st = e.stat()
            except OSError:
                continue
            path = f"{rel}{name}"
            kind = kind_of(name)
            out.append({
                "id": path,
                "path": path,          # what the revision lookup and file viewer need
                "title": name,
                "kind": kind,
                "size": st.st_size,
                "mtime": st.st_mtime * 1000.0,
                "previewable": kind != "file" and st.st_size <= MAX_PREVIEW_SIZE,
            })

    walk(root, "", 0)
    return out
