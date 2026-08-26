"""Atomic, corruption-safe JSON/blob persistence for the Nyx backend.

Every Nyx store was a truncate-then-write (``Path.write_text``) paired with a
loader that swallowed ``json.JSONDecodeError`` and returned an empty default.
That combination turns a torn write — a crash, an OOM kill, a full disk
mid-write — into *silent data loss*: the next read sees empty, and the next
write commits that emptiness over the wreckage. The command queue, the push
registrations and the pending manifest could all evaporate without a log line.

This module is the single writer/reader the ``nyx_*`` modules share:

  * writes go through tempfile + fsync + ``os.replace`` (text delegates to the
    already-hardened ``api.paths._atomic_write_text``, which also preserves
    ownership/mode and handles hard links, symlinks and xattrs);
  * a file that exists but does not parse is *quarantined*, not discarded, and
    the failure is logged — the caller still gets its default so the endpoint
    stays up, but the bytes are preserved for recovery;
  * a file that parses to the wrong shape (a list where a dict is expected)
    is treated as corrupt rather than crashing the handler with ``TypeError``.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from api.paths import _atomic_write_text, _fsync_directory

logger = logging.getLogger(__name__)


def atomic_write_text(path: Path, text: str) -> None:
    """Crash-safe replacement of *path*'s contents. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, text)


def atomic_write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    """Serialize *data* and commit it atomically."""
    atomic_write_text(path, json.dumps(data, indent=indent) + "\n")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Crash-safe replacement of *path* with *data*.

    The bytes twin of ``_atomic_write_text``. Kept separate (rather than
    encoding through the text helper) because revision content and reverted
    working files are arbitrary binary and must not round-trip through a codec.
    Mode is carried over from the existing file so reverting never widens or
    tightens permissions on the user's own source file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = os.stat(path).st_mode & 0o777
    except FileNotFoundError:
        mode = None

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        if mode is not None:
            os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _quarantine(path: Path, reason: str) -> None:
    """Move an unparseable store aside so the next write cannot bury it."""
    try:
        dest = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
        os.replace(path, dest)
        logger.error("Nyx store %s unreadable (%s); quarantined to %s", path, reason, dest)
    except OSError:
        logger.error("Nyx store %s unreadable (%s); quarantine failed", path, reason, exc_info=True)


def load_json(path: Path, default: Any, *, expect: type | tuple[type, ...] = dict) -> Any:
    """Read JSON from *path*, returning a fresh copy of *default* when absent.

    A missing file is normal and silent. A file that exists but is corrupt (bad
    JSON, or the wrong top-level type) is quarantined and logged before the
    default is returned, so a torn write is recoverable instead of overwritten.
    """
    path = Path(path)
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return json.loads(json.dumps(default))
    except OSError:
        logger.warning("Nyx store %s could not be read", path, exc_info=True)
        return json.loads(json.dumps(default))

    try:
        data = json.loads(raw) if raw.strip() else None
    except json.JSONDecodeError as exc:
        _quarantine(path, f"invalid JSON: {exc}")
        return json.loads(json.dumps(default))

    if data is None and not raw.strip():
        return json.loads(json.dumps(default))
    if not isinstance(data, expect):
        _quarantine(path, f"expected {expect}, got {type(data).__name__}")
        return json.loads(json.dumps(default))
    return data
