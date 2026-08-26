"""Route-level integration tests for the 4b revision endpoints.

These exercise the handle_* functions in nyx_revisions.py with the
real session/workspace resolution pipeline — but stub out the HTTP
handler (no actual socket traffic). Catches issues like:

  - URL parsing & query string handling
  - JSON body parsing for POSTs
  - Error response shapes (status codes, error messages)
  - Auth/session resolution plumbing

We don't spin up the full server because the existing test suite
covers server boot elsewhere.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

# NB: this used to `sys.path.insert(0, "/home/ubuntu/hermes-webui")` — the
# DEPLOYED tree, not this repo. Run alone, that made these tests import
# /home/ubuntu/hermes-webui/api/nyx_revisions.py and pass against whatever was
# last deployed, while reporting green for the working copy. It only appeared
# to test repo code in a full run, because an alphabetically earlier test
# module had already imported `api.*` from the repo into sys.modules.
# The repo root is on sys.path via pytest's rootdir; no insert is needed.


# ── Tiny stub handler so we can call the real handle_* functions ────────────


class _Stub:
    """Mimics the methods the real handle_* functions touch."""

    def __init__(self, *, body: bytes = b"") -> None:
        self.headers = {"Content-Length": str(len(body))}
        self._body = body
        self.response_code: int | None = None
        self.response_payload: dict | None = None
        self.rfile = BytesIO(body)
        self.wfile = BytesIO()
        self._sent_headers: dict[str, str] = {}

    def send_response(self, code: int) -> None:
        self.response_code = code

    def send_header(self, key: str, value: str) -> None:
        pass

    def end_headers(self) -> None:
        pass


def _build_handler(parsed, body: bytes = b"", b: dict | None = None):
    """Build a stub handler + a parsed URL for the real handlers.

    The handle_* functions for snapshot/revert accept a dict body
    (not a parsed URL); list/diff accept a parsed URL. We support
    both styles here. The dict body is passed positionally, not
    via the stub, so this helper just returns the stub.
    """
    handler = _Stub(body=body)
    return handler, parsed


# ── Server boot: do we even start? ──────────────────────────────────────────


def test_routes_module_imports_clean() -> None:
    """Routes module must import with our new dispatches in place."""
    from api import routes  # noqa: F401
    from api import nyx_revisions  # noqa: F401
    assert hasattr(nyx_revisions, "handle_revisions_list")
    assert hasattr(nyx_revisions, "handle_revisions_diff")
    assert hasattr(nyx_revisions, "handle_revisions_snapshot")
    assert hasattr(nyx_revisions, "handle_revisions_revert")


# ── End-to-end via the real handle_* functions ──────────────────────────────


def test_handle_list_unknown_session_returns_404(tmp_path: Path) -> None:
    from api.nyx_revisions import handle_revisions_list
    parsed = urlparse("/api/nyx/revisions?session_id=does-not-exist&path=foo.py")
    h, _ = _build_handler(parsed)
    result = handle_revisions_list(h, parsed)
    assert h.response_code == 404


def test_handle_list_requires_session_id() -> None:
    from api.nyx_revisions import handle_revisions_list
    parsed = urlparse("/api/nyx/revisions?path=foo.py")
    h, _ = _build_handler(parsed)
    handle_revisions_list(h, parsed)
    assert h.response_code == 400


def test_handle_list_requires_path() -> None:
    from api.nyx_revisions import handle_revisions_list
    parsed = urlparse("/api/nyx/revisions?session_id=x")
    h, _ = _build_handler(parsed)
    handle_revisions_list(h, parsed)
    assert h.response_code == 400


def test_handle_snapshot_requires_path_and_session() -> None:
    from api.nyx_revisions import handle_revisions_snapshot
    h, _ = _build_handler(None, b"", None)
    handle_revisions_snapshot(h, {"session_id": "x"})
    assert h.response_code == 400


def test_handle_revert_requires_rev() -> None:
    from api.nyx_revisions import handle_revisions_revert
    h, _ = _build_handler(None, b"", None)
    handle_revisions_revert(h, {"session_id": "x", "path": "foo.py"})
    assert h.response_code == 400


# ── Happy path through the HTTP handlers (with stubbed session) ───────────


def test_end_to_end_list_handler(tmp_path: Path, monkeypatch) -> None:
    """Full round-trip: snapshot via the core fn, list via the HTTP handler.
    Session resolution is stubbed to point at tmp_path.
    """
    from api import nyx_revisions

    f = tmp_path / "hello.py"
    f.write_text("print('v1')\n")
    monkeypatch.setattr(
        nyx_revisions, "_resolve_workspace", lambda sid: (tmp_path, None)
    )

    # Snapshot via the POST handler
    h_snap, _ = _build_handler(None, b"", None)
    nyx_revisions.handle_revisions_snapshot(
        h_snap, {"session_id": "test", "path": "hello.py", "message": "initial"}
    )
    assert h_snap.response_code == 201
    snap_body = json.loads(h_snap.wfile.getvalue().decode("utf-8"))
    assert snap_body["rev"] == "r0"
    assert snap_body["created"] is True

    # List via the GET handler
    parsed = urlparse("/api/nyx/revisions?session_id=test&path=hello.py")
    h_list, _ = _build_handler(parsed)
    nyx_revisions.handle_revisions_list(h_list, parsed)
    assert h_list.response_code == 200
    list_body = json.loads(h_list.wfile.getvalue().decode("utf-8"))
    assert len(list_body["revisions"]) == 1
    assert list_body["revisions"][0]["message"] == "initial"

    # Edit the file, then diff via the GET handler
    f.write_text("print('v2')\n")
    diff_parsed = urlparse(
        "/api/nyx/revisions/diff?session_id=test&path=hello.py&from=r0&to=head"
    )
    h_diff, _ = _build_handler(diff_parsed)
    nyx_revisions.handle_revisions_diff(h_diff, diff_parsed)
    assert h_diff.response_code == 200
    diff_body = json.loads(h_diff.wfile.getvalue().decode("utf-8"))
    assert diff_body["stats"]["removed"] == 1
    assert diff_body["stats"]["added"] == 1
    assert "v1" in diff_body["diff"]
    assert "v2" in diff_body["diff"]

    # Revert via the POST handler
    h_revert, _ = _build_handler(None, b"", None)
    nyx_revisions.handle_revisions_revert(
        h_revert,
        {"session_id": "test", "path": "hello.py", "rev": "r0"},
    )
    assert h_revert.response_code == 200
    assert f.read_text() == "print('v1')\n"
    # The pre-revert state was captured as a new rev
    revs = nyx_revisions.list_revisions(tmp_path, "hello.py")
    assert revs[0]["message"].startswith("pre-revert to r0")
