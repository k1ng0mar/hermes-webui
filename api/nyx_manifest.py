"""Nyx Mobile client manifest — agent proposes, owner applies.

GET  /api/nyx/manifest
POST /api/nyx/manifest          { tabs?, more?, theme?, stt?, note? }  → pending
POST /api/nyx/manifest/apply
POST /api/nyx/manifest/reject
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from api.helpers import bad, j

_STORE = Path.home() / ".hermes" / "webui" / "nyx-manifest.json"
_lock = threading.Lock()

ALLOWED_TABS = ("home", "sessions", "artifacts", "memory", "more")
ALLOWED_MORE = (
    "Analytics",
    "Cron",
    "Skills",
    "Tools",
    "Voice",
    "Browser",
    "Router",
    "Projects",
    "Settings",
)
ALLOWED_STT = ("server", "groq", "openai")
ALLOWED_THEMES = ("paper", "night", "plum", "moss", "sand", "ink", "terminal", "ocean")

DEFAULT = {
    "tabs": ["home", "sessions", "artifacts", "memory", "more"],
    "more": list(ALLOWED_MORE),
    "theme": "paper",
    "stt": "server",
}


def _load() -> dict:
    if not _STORE.is_file():
        return {"applied": dict(DEFAULT), "pending": None}
    try:
        data = json.loads(_STORE.read_text())
    except Exception:
        return {"applied": dict(DEFAULT), "pending": None}
    if not isinstance(data.get("applied"), dict):
        data["applied"] = dict(DEFAULT)
    return data


def _save(data: dict) -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    _STORE.write_text(json.dumps(data, indent=2))


def sanitize(raw: dict) -> dict:
    out = dict(DEFAULT)
    tabs = raw.get("tabs")
    if isinstance(tabs, list):
        clean = [t for t in tabs if t in ALLOWED_TABS]
        if "home" not in clean:
            clean.insert(0, "home")
        if "more" not in clean:
            clean.append("more")
        # de-dupe preserve order
        seen = set()
        ordered = []
        for t in clean:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        out["tabs"] = ordered
    more = raw.get("more")
    if isinstance(more, list):
        clean_m = [m for m in more if m in ALLOWED_MORE]
        if "Settings" not in clean_m:
            clean_m.append("Settings")
        seen = set()
        ordered_m = []
        for m in clean_m:
            if m not in seen:
                seen.add(m)
                ordered_m.append(m)
        out["more"] = ordered_m
    theme = raw.get("theme")
    if isinstance(theme, str) and theme in ALLOWED_THEMES:
        out["theme"] = theme
    stt = raw.get("stt")
    if isinstance(stt, str) and stt in ALLOWED_STT:
        out["stt"] = stt
    note = raw.get("note")
    if isinstance(note, str) and note.strip():
        out["note"] = note.strip()[:200]
    return out


def handle_get(handler):
    with _lock:
        data = _load()
    return j(handler, {"applied": data.get("applied") or DEFAULT, "pending": data.get("pending")})


def handle_propose(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    pending = sanitize(body)
    pending["proposed_at"] = int(time.time())
    with _lock:
        data = _load()
        data["pending"] = pending
        _save(data)
    try:
        from api.nyx_push import notify
        notify("info", "Nyx wants to change the app", pending.get("note") or "Review the client manifest.", None)
    except Exception:
        pass
    return j(handler, {"ok": True, "pending": pending})


def handle_apply(handler, body):
    with _lock:
        data = _load()
        pending = data.get("pending")
        if not isinstance(pending, dict):
            return bad(handler, "nothing pending", 409)
        applied = sanitize(pending)
        data["applied"] = applied
        data["pending"] = None
        _save(data)
    return j(handler, {"ok": True, "applied": applied})


def handle_reject(handler, body):
    with _lock:
        data = _load()
        data["pending"] = None
        _save(data)
    return j(handler, {"ok": True, "pending": None})
