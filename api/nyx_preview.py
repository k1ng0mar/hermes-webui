"""Path-style preview so relative CSS/images resolve.

GET /api/nyx/preview/<session_id>/<relpath>
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import unquote

from api.helpers import bad, safe_resolve


def handle_preview(handler, parsed):
    rest = unquote(parsed.path[len("/api/nyx/preview/") :]).lstrip("/")
    sid, _, rel = rest.partition("/")
    if not sid or not rel:
        return bad(handler, "need /api/nyx/preview/<session>/<path>")
    try:
        from api.models import get_session_for_file_ops
        session = get_session_for_file_ops(sid)
    except Exception:
        session = None
    if session is None:
        return bad(handler, "session not found", 404)
    try:
        target = safe_resolve(Path(session.workspace), rel)
    except ValueError:
        return bad(handler, "bad path", 400)
    if not target.exists() or not target.is_file():
        return bad(handler, "not found", 404)
    mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    data = target.read_bytes()
    if mime == "text/html":
        base = f"/api/nyx/preview/{sid}/{rel.rsplit('/', 1)[0] + '/' if '/' in rel else ''}"
        inject = f'<base href="{base}">'.encode()
        if b"<head>" in data.lower():
            # case-insensitive insert after first <head>
            lower = data.lower()
            i = lower.find(b"<head>")
            if i >= 0:
                data = data[: i + 6] + inject + data[i + 6 :]
        else:
            data = inject + data
        mime = "text/html; charset=utf-8"
    handler.send_response(200)
    handler.send_header("Content-Type", mime)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)
    return True
