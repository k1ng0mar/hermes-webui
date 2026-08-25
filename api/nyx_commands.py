"""Nyx device-command queue — the backend half of the hands-free channel.

The phone runs a foreground service that polls for pending commands and acks
them. Commands are a strict allow-list; the agent never gets an arbitrary
shell. Store is a small JSON file under ~/.hermes/webui so it survives a
backend restart.

Endpoints (registered in routes.py):
  POST /api/nyx/commands            { action, params, confirm? } -> enqueue
  GET  /api/nyx/commands/pending?device=<id>  -> oldest unacked (or [])
  POST /api/nyx/commands/ack        { id, ok, error? }          -> mark done
  GET  /api/nyx/commands/status                                  -> queue counts
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from api.helpers import bad, j

_STORE = Path.home() / ".hermes" / "webui" / "nyx-commands.json"
_lock = threading.Lock()

# The allow-list: only these actions are executable on the phone. `confirm`
# means the agent must require explicit user confirmation before enqueueing.
ALLOWED_ACTIONS: dict[str, dict] = {
    "call": {"confirm": True, "params": ["number"]},
    "open_app": {"confirm": False, "params": ["package"]},
    "bluetooth": {"confirm": False, "params": ["enabled"]},
    "flashlight": {"confirm": False, "params": ["enabled"]},
    "volume": {"confirm": False, "params": ["stream", "level"]},
    "brightness": {"confirm": False, "params": ["level"]},
    "dnd": {"confirm": False, "params": ["enabled"]},
}


def _load() -> dict:
    if not _STORE.is_file():
        return {"commands": []}
    try:
        return json.loads(_STORE.read_text())
    except Exception:
        return {"commands": []}


def _save(data: dict) -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    _STORE.write_text(json.dumps(data, indent=2))


def enqueue(action: str, params: dict, confirm_required: bool = False) -> dict:
    """Validate against the allow-list and append a command."""
    spec = ALLOWED_ACTIONS.get(action)
    if not spec:
        raise ValueError(f"unknown action: {action}")
    # re-confirm on the server side even if the caller lied about confirm_required
    for key in spec["params"]:
        if key not in params:
            raise ValueError(f"{action} requires param: {key}")

    rec = {
        "id": uuid.uuid4().hex[:16],
        "action": action,
        "params": params,
        "confirm_required": bool(spec["confirm"]) or bool(confirm_required),
        "status": "pending",
        "created_at": time.time(),
        "acked_at": None,
        "result": None,
    }
    with _lock:
        data = _load()
        data.setdefault("commands", []).append(rec)
        # cap history to the last 200 to keep the file small
        data["commands"] = data["commands"][-200:]
        _save(data)
    return rec


def _pop_pending(device: str) -> dict | None:
    """Return (and lock) the oldest unacked command for this device."""
    with _lock:
        data = _load()
        for rec in data["commands"]:
            if rec.get("status") == "pending":
                rec["status"] = "dispatched"
                rec["dispatched_at"] = time.time()
                rec["device"] = device
                _save(data)
                return rec
    return None


def _ack(command_id: str, ok: bool, error: str | None) -> bool:
    with _lock:
        data = _load()
        for rec in data["commands"]:
            if rec.get("id") == command_id:
                rec["status"] = "done" if ok else "failed"
                rec["result"] = "ok" if ok else (error or "failed")
                rec["acked_at"] = time.time()
                _save(data)
                return True
    return False


def handle_enqueue(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    action = str(body.get("action") or "").strip()
    params = body.get("params") or {}
    if not isinstance(params, dict):
        return bad(handler, "params must be an object")
    try:
        rec = enqueue(action, params)
    except ValueError as e:
        return bad(handler, str(e), 400)
    return j(handler, {"ok": True, "command": rec})


def handle_pending(handler, parsed):
    from urllib.parse import parse_qs

    qs = parse_qs(parsed.query or "")
    device = (qs.get("device", [""])[0] or "unknown").strip()
    rec = _pop_pending(device)
    if rec is None:
        # long-poll: block up to ~20s for a command, then return empty.
        deadline = time.time() + 20.0
        while time.time() < deadline:
            time.sleep(1.0)
            rec = _pop_pending(device)
            if rec is not None:
                break
    return j(handler, {"command": rec})


def handle_ack(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    command_id = str(body.get("id") or "").strip()
    if not command_id:
        return bad(handler, "id required")
    ok = bool(body.get("ok"))
    error = str(body.get("error") or "").strip() or None
    found = _ack(command_id, ok, error)
    if not found:
        return bad(handler, "command not found", 404)
    return j(handler, {"ok": True})


def handle_status(handler):
    with _lock:
        data = _load()
    cmds = data.get("commands", [])
    pending = [c for c in cmds if c.get("status") == "pending"]
    dispatched = [c for c in cmds if c.get("status") == "dispatched"]
    return j(
        handler,
        {
            "total": len(cmds),
            "pending": len(pending),
            "dispatched": len(dispatched),
            "allowed_actions": sorted(ALLOWED_ACTIONS.keys()),
        },
    )
