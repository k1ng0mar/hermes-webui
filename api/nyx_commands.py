"""Nyx device-command queue — the backend half of the hands-free channel.

The phone runs a foreground service that polls for pending commands and acks
them. Commands are a strict allow-list; the agent never gets an arbitrary
shell. Store is a small JSON file under ~/.hermes/webui so it survives a
backend restart.

Endpoints (registered in routes.py):
  POST /api/nyx/commands            { action, params, device?, confirmed? } -> enqueue
  GET  /api/nyx/commands/pending?device=<id>  -> oldest unacked for that device (or [])
  POST /api/nyx/commands/ack        { id, ok, error? }          -> mark done
  GET  /api/nyx/commands/status                                  -> queue counts
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from pathlib import Path

from api.helpers import bad, j
from api.nyx_store import atomic_write_json, load_json

_STORE = Path.home() / ".hermes" / "webui" / "nyx-commands.json"
# One lock guards the store; the Condition on top of it lets long-pollers sleep
# until an enqueue wakes them instead of re-reading the file once a second.
_lock = threading.Lock()
_arrived = threading.Condition(_lock)

MAX_HISTORY = 200
LONG_POLL_S = 20.0

# The allow-list: only these actions are executable on the phone. `confirm`
# means the request MUST carry an explicit confirmation from the owner — the
# server enforces it below, it is not a hint for the caller to honour.
ALLOWED_ACTIONS: dict[str, dict] = {
    "call": {"confirm": True, "params": ["number"]},
    "open_app": {"confirm": False, "params": ["package"]},
    "bluetooth": {"confirm": False, "params": ["enabled"]},
    "flashlight": {"confirm": False, "params": ["enabled"]},
    # `stream` was listed as REQUIRED here, but NyxCommandService hardcodes
    # STREAM_MUSIC and never reads it — so every volume command in the
    # documented shape ({level: 0.5}) was rejected with "requires param:
    # stream". It stays accepted-but-optional for forward compatibility.
    "volume": {"confirm": False, "params": ["level"], "optional": ["stream"]},
    "brightness": {"confirm": False, "params": ["level"]},
    "dnd": {"confirm": False, "params": ["enabled"]},
}

ALLOWED_STREAMS = ("music", "ring", "alarm", "notification", "call", "system")
_NUMBER_RE = re.compile(r"^\+?[0-9][0-9 ()\-.]{2,31}$")
_PACKAGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+$")
_DEVICE_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "1", "yes", "on"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "0", "no", "off"):
        return False
    raise ValueError("expected a boolean")


def _coerce_level(value) -> float:
    """A 0.0–1.0 fraction, matching what the phone consumes.

    NyxCommandService reads `level` as a double and scales it (× stream max for
    volume, × 255 for brightness). It coerces into range on its side, but an
    out-of-range value here means the caller misunderstood the unit — dialling
    brightness to "50" would silently mean full brightness — so reject rather
    than clamp.
    """
    if isinstance(value, bool):
        raise ValueError("level must be a number between 0.0 and 1.0")
    try:
        level = float(value)
    except (TypeError, ValueError):
        raise ValueError("level must be a number between 0.0 and 1.0") from None
    if not 0.0 <= level <= 1.0:
        raise ValueError("level must be between 0.0 and 1.0")
    return level


def _validate_params(action: str, params: dict) -> dict:
    """Return the normalized params for *action*, or raise ValueError.

    Values reach a foreground service that actually places calls and opens apps,
    so every one is type- and range-checked here rather than trusted.
    """
    spec = ALLOWED_ACTIONS[action]
    for key in spec["params"]:
        if key not in params:
            raise ValueError(f"{action} requires param: {key}")

    clean: dict = {}
    if action == "call":
        number = str(params["number"]).strip()
        if not _NUMBER_RE.match(number):
            raise ValueError("number must be a plain phone number")
        clean["number"] = number
    elif action == "open_app":
        package = str(params["package"]).strip()
        if not _PACKAGE_RE.match(package):
            raise ValueError("package must be an Android package name")
        clean["package"] = package
    elif action in ("bluetooth", "flashlight", "dnd"):
        clean["enabled"] = _coerce_bool(params["enabled"])
    elif action == "volume":
        clean["level"] = _coerce_level(params["level"])
        if params.get("stream") is not None:
            stream = str(params["stream"]).strip().lower()
            if stream not in ALLOWED_STREAMS:
                raise ValueError(f"stream must be one of: {', '.join(ALLOWED_STREAMS)}")
            clean["stream"] = stream
    elif action == "brightness":
        clean["level"] = _coerce_level(params["level"])
    return clean


def _load() -> dict:
    """Read the queue. A corrupt store is quarantined by load_json rather than
    silently replaced with an empty queue that the next _save would commit."""
    data = load_json(_STORE, {"commands": []})
    if not isinstance(data.get("commands"), list):
        data["commands"] = []
    return data


def _save(data: dict) -> None:
    atomic_write_json(_STORE, data)


def enqueue(action: str, params: dict, confirmed: bool = False,
            device: str = "") -> dict:
    """Validate against the allow-list and append a command.

    `device` addresses the command at one phone; empty means "whichever device
    asks first". `confirmed` must be True for actions the allow-list marks as
    requiring confirmation — the server refuses them otherwise, so a caller
    cannot place a phone call by simply omitting the flag.
    """
    spec = ALLOWED_ACTIONS.get(action)
    if not spec:
        raise ValueError(f"unknown action: {action}")
    if spec["confirm"] and not confirmed:
        raise ValueError(f"{action} requires explicit owner confirmation")
    if device and not _DEVICE_RE.match(device):
        raise ValueError("invalid device id")

    clean_params = _validate_params(action, params)

    rec = {
        "id": uuid.uuid4().hex[:16],
        "action": action,
        "params": clean_params,
        "confirm_required": bool(spec["confirm"]),
        "confirmed": bool(confirmed),
        "target_device": device or "",
        "status": "pending",
        "created_at": time.time(),
        "acked_at": None,
        "result": None,
    }
    with _arrived:
        data = _load()
        data["commands"].append(rec)
        # cap history to the last MAX_HISTORY to keep the file small
        data["commands"] = data["commands"][-MAX_HISTORY:]
        _save(data)
        # Wake every long-poller; each re-checks whether the new command is
        # addressed to it.
        _arrived.notify_all()
    return rec


def _claim_locked(device: str) -> dict | None:
    """Claim the oldest pending command this device may run. Caller holds _lock.

    A command with an empty `target_device` is unaddressed and any device may
    take it; one addressed to another phone is skipped. Previously the device
    argument was ignored entirely and then stamped onto whatever came back, so
    with two phones connected either could steal the other's command.
    """
    data = _load()
    for rec in data["commands"]:
        if rec.get("status") != "pending":
            continue
        target = str(rec.get("target_device") or "")
        if target and target != device:
            continue
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
    device = str(body.get("device") or "").strip()
    confirmed = bool(body.get("confirmed"))
    try:
        rec = enqueue(action, params, confirmed=confirmed, device=device)
    except ValueError as e:
        return bad(handler, str(e), 400)
    return j(handler, {"ok": True, "command": rec})


def handle_pending(handler, parsed):
    from urllib.parse import parse_qs

    qs = parse_qs(parsed.query or "")
    device = (qs.get("device", [""])[0] or "unknown").strip()
    if not _DEVICE_RE.match(device):
        return bad(handler, "invalid device id", 400)

    # Long-poll: sleep on the condition until an enqueue wakes us or the
    # deadline passes. The previous loop re-read and re-parsed the whole JSON
    # store once a second for 20s per connected phone; this wakes only on a
    # real enqueue and re-reads once per wake.
    deadline = time.monotonic() + LONG_POLL_S
    with _arrived:
        rec = _claim_locked(device)
        while rec is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _arrived.wait(remaining)
            rec = _claim_locked(device)
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
            "confirm_required": sorted(
                a for a, s in ALLOWED_ACTIONS.items() if s["confirm"]
            ),
        },
    )
