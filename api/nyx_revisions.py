"""Artifact revisions for Nyx mobile (design 4b).

Storage layout (per workspace):
    <workspace>/.artifacts/.revisions/<encoded_path>/<rev_id>/
        content     # the file bytes at this revision
        meta.json   # {rev, mtime, parent, message, author, size}

`<encoded_path>` is base64url of the artifact path, padding stripped.
`<rev_id>` is `r0`, `r1`, ... (monotonic; new = max + 1).

Revisions are user-curated: the mobile UI calls `POST snapshot` to
capture the current file state before a meaningful change. We do NOT
auto-snapshot on every write because the agent edits files in place
and there's no write-API hook in the backend today.

Endpoints (all require a valid session_id and a file that exists in
the session workspace):

    GET  /api/nyx/revisions?session_id=X&path=Y[&limit=N]
        List revisions for a file. Newest first.
        Response: {revisions: [{rev, mtime, size, message, author,
                                parent, stats: {added, removed}}]}

    POST /api/nyx/revisions/snapshot
        Body: {session_id, path, message?, author?}
        Capture the current file content as a new revision.
        Idempotent on identical content: returns the existing rev
        rather than creating a duplicate.
        Response: {rev, mtime, size, created: bool}

    GET  /api/nyx/revisions/diff?session_id=X&path=Y&from=A&to=B
        Unified diff between two revisions. `from` and `to` are
        rev_ids, or "head" for the current file, or "working" to
        alias the current file's uncommitted state.
        Response: {diff: "<unified diff text>", stats: {added,
        removed}}

    POST /api/nyx/revisions/revert
        Body: {session_id, path, rev}
        Restore a revision to the current file path. Captures the
        pre-revert state as a new revision so the revert is
        reversible.
        Response: {rev, mtime, size, restored_from: <rev>}

Errors:
    400 bad path / invalid input
    403 not in workspace / path traversal
    404 session not found / file not found / revision not found
    500 unexpected
"""
from __future__ import annotations

import base64
import difflib
import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs

from api.helpers import bad, j, safe_resolve
from api.nyx_store import atomic_write_bytes, atomic_write_text

# Configurable caps. Keep the working set small — old revisions fall
# off the end on a FIFO basis.
DEFAULT_REVISION_LIMIT = 50        # returned by GET /revisions
MAX_REVISIONS_KEPT = 200           # hard cap on disk per file
MAX_DIFF_BYTES = 5 * 1024 * 1024   # 5 MB diff cap (matches MAX_PREVIEW_SIZE)
MAX_REVISION_BYTES = 5 * 1024 * 1024  # refuse to snapshot huge files

_REV_PATTERN = re.compile(r"^r(\d+)$")

# Snapshot/revert/prune do read-modify-write on a shared revisions directory.
# Without this, two concurrent snapshots of the same file both resolve the same
# `_next_rev_id`, both mkdir(exist_ok=True), and the second silently overwrites
# the first one's content — a lost revision with no error anywhere.
_revision_lock = threading.Lock()


def _validate_rev_id(rev: str) -> str:
    """Reject anything that is not a literal rN revision id.

    `rev` arrives straight from the request body/query and is joined onto the
    revisions directory. Unvalidated, `../../..` walks out of the revisions root
    — and on the revert path the content found out there gets written into the
    user's workspace file. _REV_PATTERN existed but was only ever applied to
    directory names already read off disk, never to caller input.
    """
    rev = str(rev or "").strip()
    if not _REV_PATTERN.match(rev):
        raise ValueError(f"invalid revision id: {rev!r}")
    return rev


# ── Path encoding ───────────────────────────────────────────────────────────


def encode_path(rel: str) -> str:
    """base64url of the artifact path. Empty path -> '_root'."""
    if not rel:
        return "_root"
    return base64.urlsafe_b64encode(rel.encode("utf-8")).decode("ascii").rstrip("=")


def decode_path(encoded: str) -> str:
    """Reverse of encode_path."""
    if encoded == "_root":
        return ""
    pad = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded + pad).decode("utf-8")


# ── Workspace + session resolution (mirrors _handle_artifacts) ─────────────


def _resolve_workspace(sid: str) -> tuple[Path, dict | None]:
    """Return (workspace_path, session_or_None) for a session id.

    Mirrors _handle_artifacts: tries the in-memory session first, then
    falls back to the CLI session registry. Raises KeyError when the
    session is unknown.
    """
    from api.models import get_session

    s = None
    workspace = ""
    try:
        s = get_session(sid)
        workspace = s.workspace
    except KeyError:
        s = None
        try:
            cli_meta = None
            from api.models import get_cli_sessions
            for cs in get_cli_sessions():
                if cs["session_id"] == sid:
                    cli_meta = cs
                    break
            if not cli_meta:
                raise KeyError(sid)
            workspace = cli_meta.get("workspace", "")
        except Exception as exc:
            raise KeyError(sid) from exc

    # Trust the workspace unless we have no in-memory session.
    from api.workspace import (
        resolve_trusted_workspace,
        resolve_implicit_workspace_with_recovery,
        get_last_workspace,
    )
    if s is None:
        ws = Path(resolve_trusted_workspace(workspace))
    else:
        ws = Path(resolve_implicit_workspace_with_recovery(workspace, get_last_workspace)[0])
    return ws, s


# ── On-disk layout ──────────────────────────────────────────────────────────


def revisions_root(workspace: Path) -> Path:
    """The .artifacts/.revisions/ root for a workspace. Created lazily."""
    p = workspace / ".artifacts" / ".revisions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_rev_dir(workspace: Path, rel: str) -> Path:
    """Per-file revisions directory. Created lazily."""
    p = revisions_root(workspace) / encode_path(rel)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _next_rev_id(file_dir: Path) -> str:
    """Pick the next monotonic rev id, scanning existing dirs."""
    if not file_dir.exists():
        return "r0"
    max_n = -1
    for child in file_dir.iterdir():
        if not child.is_dir():
            continue
        m = _REV_PATTERN.match(child.name)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return f"r{max_n + 1}"


def _parse_rev_id(s: str) -> int | None:
    m = _REV_PATTERN.match(s)
    return int(m.group(1)) if m else None


# ── Snapshot ────────────────────────────────────────────────────────────────


def _read_meta(rev_dir: Path) -> dict:
    meta_path = rev_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_meta(rev_dir: Path, meta: dict) -> None:
    atomic_write_text(
        rev_dir / "meta.json", json.dumps(meta, indent=2, sort_keys=True)
    )


def _existing_rev_for_content(file_dir: Path, content: bytes) -> Path | None:
    """Idempotency check: if the latest revision has identical content,
    return it. Otherwise None.
    """
    if not file_dir.exists():
        return None
    # Walk revisions in rev_id order; the highest-numbered one with
    # the same content is the dedupe target.
    revs = []
    for child in file_dir.iterdir():
        if not child.is_dir():
            continue
        n = _parse_rev_id(child.name)
        if n is not None:
            revs.append((n, child))
    if not revs:
        return None
    revs.sort()
    latest = revs[-1][1]
    latest_content = (latest / "content").read_bytes() if (latest / "content").exists() else b""
    if latest_content == content:
        return latest
    return None


def snapshot_revision(workspace: Path, rel: str, message: str = "",
                      author: str = "") -> dict:
    """Capture the current file content as a new revision.

    Returns {rev, mtime, size, created: bool}. `created` is False if
    we deduped against an existing revision with identical content.
    Raises FileNotFoundError if `rel` doesn't exist.
    """
    target = safe_resolve(workspace, rel)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(rel)
    content = target.read_bytes()
    if len(content) > MAX_REVISION_BYTES:
        raise ValueError(
            f"file too large to snapshot ({len(content)} > {MAX_REVISION_BYTES} bytes)"
        )

    file_dir = file_rev_dir(workspace, rel)

    # Everything from here down is read-modify-write on file_dir: the dedup
    # scan, the _next_rev_id allocation and the FIFO prune all have to see a
    # consistent directory or concurrent snapshots clobber each other.
    with _revision_lock:
        # Idempotency: if the most recent rev has identical content, reuse it.
        existing = _existing_rev_for_content(file_dir, content)
        if existing is not None:
            meta = _read_meta(existing)
            return {
                "rev": existing.name,
                "mtime": meta.get("mtime", int(existing.stat().st_mtime * 1000)),
                "size": meta.get("size", len(content)),
                "created": False,
            }

        rev_id = _next_rev_id(file_dir)
        rev_dir = file_dir / rev_id
        rev_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(rev_dir / "content", content)

        # Find parent rev_id (the previous newest).
        parent = ""
        n = _parse_rev_id(rev_id)
        if n is not None and n > 0:
            parent = f"r{n - 1}"
            if not (file_dir / parent).exists():
                parent = ""

        mtime_ms = int(time.time() * 1000)
        meta = {
            "rev": rev_id,
            "mtime": mtime_ms,
            "size": len(content),
            "parent": parent,
            "message": message[:500] if message else "",
            "author": author[:100] if author else "nyx",
        }
        _write_meta(rev_dir, meta)

        # FIFO prune: keep at most MAX_REVISIONS_KEPT.
        _prune_old_revisions(file_dir)

    return {
        "rev": rev_id,
        "mtime": mtime_ms,
        "size": len(content),
        "created": True,
    }


def _prune_old_revisions(file_dir: Path) -> None:
    """Drop oldest revisions beyond MAX_REVISIONS_KEPT."""
    revs = []
    for child in file_dir.iterdir():
        if not child.is_dir():
            continue
        n = _parse_rev_id(child.name)
        if n is not None:
            revs.append((n, child))
    if len(revs) <= MAX_REVISIONS_KEPT:
        return
    revs.sort()
    for _, old in revs[: len(revs) - MAX_REVISIONS_KEPT]:
        try:
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
        except OSError:
            pass


# ── List ────────────────────────────────────────────────────────────────────


def list_revisions(workspace: Path, rel: str, limit: int = DEFAULT_REVISION_LIMIT) -> list[dict]:
    """Return up to `limit` revisions for a file, newest first."""
    target = safe_resolve(workspace, rel)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(rel)

    file_dir = file_rev_dir(workspace, rel)
    if not file_dir.exists():
        return []

    revs = []
    for child in file_dir.iterdir():
        if not child.is_dir():
            continue
        n = _parse_rev_id(child.name)
        if n is None:
            continue
        meta = _read_meta(child)
        revs.append({
            "rev": child.name,
            "mtime": meta.get("mtime", int(child.stat().st_mtime * 1000)),
            "size": meta.get("size", 0),
            "message": meta.get("message", ""),
            "author": meta.get("author", ""),
            "parent": meta.get("parent", ""),
        })
    revs.sort(key=lambda r: _parse_rev_id(r["rev"]) or 0, reverse=True)
    return revs[: max(1, min(limit, MAX_REVISIONS_KEPT))]


# ── Diff ────────────────────────────────────────────────────────────────────


def _read_rev_content(rev_dir: Path) -> bytes:
    p = rev_dir / "content"
    if not p.exists():
        return b""
    return p.read_bytes()


def _decode_for_diff(data: bytes) -> list[str] | None:
    """Try utf-8 decode. Return None for binary data."""
    try:
        return data.decode("utf-8").splitlines(keepends=False)
    except UnicodeDecodeError:
        return None


def diff_revisions(workspace: Path, rel: str, from_rev: str, to_rev: str) -> dict:
    """Compute a unified diff between two revisions (or the working file).

    from_rev / to_rev: "r0".."rN", "head" (current file), or
    "working" (alias for head).
    """
    target = safe_resolve(workspace, rel)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(rel)
    file_dir = file_rev_dir(workspace, rel)

    def fetch(rev: str) -> bytes:
        if rev in ("head", "working", "current"):
            return target.read_bytes()
        rev_dir = file_dir / _validate_rev_id(rev)
        if not rev_dir.exists():
            raise FileNotFoundError(f"revision {rev!r} not found")
        return _read_rev_content(rev_dir)

    from_bytes = fetch(from_rev)
    to_bytes = fetch(to_rev)

    if len(from_bytes) + len(to_bytes) > 2 * MAX_DIFF_BYTES:
        raise ValueError("file(s) too large to diff")

    from_lines = _decode_for_diff(from_bytes)
    to_lines = _decode_for_diff(to_bytes)

    # Binary files: no text diff possible.
    if from_lines is None or to_lines is None:
        return {
            "diff": "",
            "stats": {"added": 0, "removed": 0},
            "binary": True,
            "from": from_rev,
            "to": to_rev,
        }

    ud = list(difflib.unified_diff(
        from_lines,
        to_lines,
        fromfile=from_rev,
        tofile=to_rev,
        lineterm="",
    ))

    # Count +/- lines (skip the ---/+++ headers).
    added = sum(1 for line in ud if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in ud if line.startswith("-") and not line.startswith("---"))

    return {
        "diff": "\n".join(ud),
        "stats": {"added": added, "removed": removed},
        "binary": False,
        "from": from_rev,
        "to": to_rev,
    }


# ── Revert ──────────────────────────────────────────────────────────────────


def revert_to_revision(workspace: Path, rel: str, rev: str) -> dict:
    """Restore `rev` content to the working file.

    Captures the current working state as a new revision first, so
    the revert is itself reversible. Returns {rev, mtime, size,
    restored_from}.
    """
    rev = _validate_rev_id(rev)
    target = safe_resolve(workspace, rel)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(rel)
    file_dir = file_rev_dir(workspace, rel)
    rev_dir = file_dir / rev
    if not rev_dir.exists():
        raise FileNotFoundError(f"revision {rev!r} not found")

    # Capture the pre-revert state. If it's identical to the rev
    # we're about to write, skip the snapshot (it would be a no-op
    # anyway).
    pre_content = target.read_bytes()
    rev_content = _read_rev_content(rev_dir)
    if pre_content != rev_content:
        # NB: snapshot_revision takes _revision_lock itself, so it must be
        # called before we hold it — the lock is not reentrant.
        snapshot_revision(workspace, rel, message=f"pre-revert to {rev}", author="nyx")

    # Now overwrite the working file. This is the user's real source file, so
    # it goes through temp+fsync+rename: a crash mid-revert used to truncate
    # the very file the revision history exists to protect.
    atomic_write_bytes(target, rev_content)

    return {
        "rev": rev,
        "mtime": int(time.time() * 1000),
        "size": len(rev_content),
        "restored_from": rev,
    }


# ── HTTP handlers ───────────────────────────────────────────────────────────


def _require_session(qs) -> str:
    sid = qs.get("session_id", [""])[0]
    if not sid:
        raise ValueError("session_id is required")
    return sid


def _require_path(qs) -> str:
    p = qs.get("path", [""])[0]
    if not p:
        raise ValueError("path is required")
    return p


def handle_revisions_list(handler, parsed):
    qs = parse_qs(parsed.query)
    try:
        sid = _require_session(qs)
        rel = _require_path(qs)
        limit = int(qs.get("limit", [str(DEFAULT_REVISION_LIMIT)])[0])
    except ValueError as e:
        return bad(handler, str(e))
    try:
        workspace, _ = _resolve_workspace(sid)
        rels = list_revisions(workspace, rel, limit=limit)
    except KeyError:
        return bad(handler, "session not found", 404)
    except FileNotFoundError as e:
        return bad(handler, f"file not found: {e}", 404)
    except ValueError as e:
        return bad(handler, str(e), 400)
    except Exception as e:
        return bad(handler, f"unexpected: {e}", 500)
    return j(handler, {"revisions": rels, "path": rel})


def handle_revisions_snapshot(handler, body):
    try:
        sid = (body or {}).get("session_id", "")
        rel = (body or {}).get("path", "")
        message = (body or {}).get("message", "")
        author = (body or {}).get("author", "")
        if not sid or not rel:
            return bad(handler, "session_id and path are required")
        workspace, _ = _resolve_workspace(sid)
        result = snapshot_revision(workspace, rel, message=message, author=author)
    except KeyError:
        return bad(handler, "session not found", 404)
    except FileNotFoundError as e:
        return bad(handler, f"file not found: {e}", 404)
    except ValueError as e:
        return bad(handler, str(e), 400)
    except Exception as e:
        return bad(handler, f"unexpected: {e}", 500)
    return j(handler, result, status=201 if result.get("created") else 200)


def handle_revisions_diff(handler, parsed):
    qs = parse_qs(parsed.query)
    try:
        sid = _require_session(qs)
        rel = _require_path(qs)
        from_rev = qs.get("from", ["head"])[0]
        to_rev = qs.get("to", ["head"])[0]
    except ValueError as e:
        return bad(handler, str(e))
    try:
        workspace, _ = _resolve_workspace(sid)
        result = diff_revisions(workspace, rel, from_rev, to_rev)
    except KeyError:
        return bad(handler, "session not found", 404)
    except FileNotFoundError as e:
        return bad(handler, f"not found: {e}", 404)
    except ValueError as e:
        return bad(handler, str(e), 400)
    except Exception as e:
        return bad(handler, f"unexpected: {e}", 500)
    return j(handler, result)


def handle_revisions_revert(handler, body):
    try:
        sid = (body or {}).get("session_id", "")
        rel = (body or {}).get("path", "")
        rev = (body or {}).get("rev", "")
        if not sid or not rel or not rev:
            return bad(handler, "session_id, path, and rev are required")
        workspace, _ = _resolve_workspace(sid)
        result = revert_to_revision(workspace, rel, rev)
    except KeyError:
        return bad(handler, "session not found", 404)
    except FileNotFoundError as e:
        return bad(handler, f"not found: {e}", 404)
    except ValueError as e:
        return bad(handler, str(e), 400)
    except Exception as e:
        return bad(handler, f"unexpected: {e}", 500)
    return j(handler, result)
