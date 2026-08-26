"""Path-style preview so relative CSS/images resolve.

GET /api/nyx/preview/<session_id>/<relpath>

Security note
-------------
This endpoint serves *workspace* content — files an agent wrote — from the
WebUI's own origin, which is where the authenticated session cookie lives. It
therefore does three things the original did not:

  * caps the response size, so a large artifact cannot be read entirely into
    memory (the WebUI shares this host with the agent);
  * escapes the injected ``<base href>``, which interpolates a URL-derived
    path into an HTML attribute;
  * sends a sandboxing CSP on HTML. ``sandbox allow-scripts allow-forms``
    without ``allow-same-origin`` puts the document in an opaque origin:
    interactive artifacts still run their own JS, but that JS can no longer
    read ``document.cookie`` or call the authenticated API as the user.
    This handler writes its headers directly rather than going through
    ``j()``, so nothing else applies them on its behalf.
"""
from __future__ import annotations

import html
import mimetypes
from pathlib import Path
from urllib.parse import unquote

from api.helpers import bad, safe_resolve

# Matches MAX_PREVIEW_SIZE / MAX_DIFF_BYTES in api.nyx_revisions.
MAX_PREVIEW_BYTES = 5 * 1024 * 1024


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

    # Check the size before reading: read_bytes() on a multi-GB artifact would
    # pull the whole file into the server's address space.
    try:
        size = target.stat().st_size
    except OSError:
        return bad(handler, "not found", 404)
    if size > MAX_PREVIEW_BYTES:
        return bad(
            handler,
            f"file too large to preview ({size} > {MAX_PREVIEW_BYTES} bytes)",
            413,
        )

    mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    data = target.read_bytes()
    is_html = mime == "text/html"
    if is_html:
        base = f"/api/nyx/preview/{sid}/{rel.rsplit('/', 1)[0] + '/' if '/' in rel else ''}"
        # `rel` comes from the request path; unescaped, a filename containing a
        # double quote breaks out of the href attribute.
        inject = f'<base href="{html.escape(base, quote=True)}">'.encode()
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
    handler.send_header("X-Content-Type-Options", "nosniff")
    if is_html:
        # No allow-same-origin: scripts run, but in an opaque origin with no
        # access to this origin's cookies or authenticated endpoints.
        handler.send_header(
            "Content-Security-Policy", "sandbox allow-scripts allow-forms"
        )
    handler.end_headers()
    handler.wfile.write(data)
    return True
