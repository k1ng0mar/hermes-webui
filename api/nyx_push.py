"""Nyx Mobile push — Expo tokens + optional webhook (ntfy etc).

POST /api/nyx/push/register  { token, platform }
POST /api/nyx/push/config    { webhook }
GET  /api/nyx/push/status
POST /api/nyx/push/test
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from api.helpers import bad, j
from api.nyx_store import atomic_write_json, load_json

logger = logging.getLogger(__name__)

_STORE = Path.home() / ".hermes" / "webui" / "nyx-push.json"
_lock = threading.Lock()


def _load() -> dict:
    """Read the push store. A corrupt file is quarantined, never silently reset:
    losing the device registrations means the phone stops getting notifications
    with nothing in the log to explain it."""
    data = load_json(_STORE, {"devices": [], "webhook": ""})
    if not isinstance(data.get("devices"), list):
        data["devices"] = []
    return data


def _save(data: dict) -> None:
    atomic_write_json(_STORE, data)


def notify(kind: str, title: str, body: str, session_id: str | None = None) -> None:
    threading.Thread(
        target=_dispatch,
        args=(kind, title, body, session_id),
        daemon=True,
        name="nyx-push",
    ).start()


def _dispatch(kind: str, title: str, body: str, session_id: str | None) -> None:
    with _lock:
        data = _load()
        devices = list(data.get("devices") or [])
        webhook = str(data.get("webhook") or "").strip()
    payload = {"kind": kind, "title": title, "body": body, "sessionId": session_id or ""}
    expo = [d.get("token") for d in devices if str(d.get("token") or "").startswith("ExponentPushToken")]
    if expo:
        try:
            req = urllib.request.Request(
                "https://exp.host/--/api/v2/push/send",
                data=json.dumps(
                    [{"to": t, "title": title, "body": body, "data": payload, "sound": "default"} for t in expo]
                ).encode(),
                method="POST",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            urllib.request.urlopen(req, timeout=8).read()
        except Exception:
            logger.debug("expo push failed", exc_info=True)
    if webhook:
        try:
            req = urllib.request.Request(
                webhook,
                data=body.encode(),
                method="POST",
                headers={
                    "Title": title,
                    "Content-Type": "text/plain",
                    "X-Nyx-Kind": kind,
                    "X-Nyx-Session": session_id or "",
                },
            )
            urllib.request.urlopen(req, timeout=8).read()
        except Exception:
            logger.debug("webhook push failed", exc_info=True)


def handle_status(handler):
    with _lock:
        data = _load()
    return j(
        handler,
        {
            "devices": len(data.get("devices") or []),
            "webhook": bool(str(data.get("webhook") or "").strip()),
        },
    )


def handle_register(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    token = str(body.get("token") or "").strip()
    platform = str(body.get("platform") or "android").strip()
    if not token:
        return bad(handler, "token is required")
    with _lock:
        data = _load()
        devices = [d for d in (data.get("devices") or []) if d.get("token") != token]
        devices.append({"token": token, "platform": platform, "updated": int(time.time())})
        data["devices"] = devices[-20:]
        _save(data)
    return j(handler, {"ok": True, "devices": len(data["devices"])})


def handle_config(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    webhook = str(body.get("webhook") or "").strip()
    with _lock:
        data = _load()
        data["webhook"] = webhook
        _save(data)
    return j(handler, {"ok": True, "webhook": bool(webhook)})


def handle_test(handler, body):
    notify("info", "Nyx", "Push test — if you see this, the pipe works.")
    return j(handler, {"ok": True})
