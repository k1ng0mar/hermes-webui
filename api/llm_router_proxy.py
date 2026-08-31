"""Proxy llm-router admin (redacted) through WebUI so the phone never hits :8015.

GET  /api/llm-router/pools   -> { default, pools, listen }
POST /api/llm-router/pools   { pool, entries }  -> rewrite that pool's fallback list

Reads router_key from the on-disk router.yaml. Never returns keys.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from api.helpers import bad, j

_DEFAULT_YAML = Path(os.environ.get("NYX_ROUTER_YAML", "/home/ubuntu/llm-router/router.yaml"))


def _load_router_auth():
    try:
        import yaml  # type: ignore
    except Exception:
        return None, None
    if not _DEFAULT_YAML.is_file():
        return None, None
    data = yaml.safe_load(_DEFAULT_YAML.read_text()) or {}
    key = str(data.get("router_key") or "").strip()
    listen = str(data.get("listen") or "127.0.0.1:8015").strip()
    if "://" not in listen:
        listen = f"http://{listen}"
    return key, listen.rstrip("/")


def _router_request(method: str, path: str, body: dict | None = None):
    key, base = _load_router_auth()
    if not key or not base:
        return None, "llm-router config not found on this host"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode() or "{}"), None
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode() or "{}")
        except Exception:
            payload = {"error": str(e)}
        return payload, f"router {e.code}"
    except Exception as e:
        return None, str(e)


def handle_llm_router_get(handler):
    payload, err = _router_request("GET", "/api/config")
    if err and payload is None:
        return bad(handler, err, 502)
    data = payload or {}
    pools = data.get("pools") or {}
    providers = data.get("providers") or {}
    slim = []
    cooling = 0
    key_total = 0
    for name, p in (providers.items() if isinstance(providers, dict) else []):
        if not isinstance(p, dict):
            continue
        keys = p.get("keys") or []
        key_total += len(keys) if isinstance(keys, list) else 0
        slim.append(
            {
                "id": name,
                "base_url": p.get("base_url") or "",
                "enabled": p.get("enabled", True),
                "keyCount": len(keys) if isinstance(keys, list) else 0,
                "keyHints": keys if isinstance(keys, list) else [],
                "api_mode": p.get("api_mode") or "openai",
                "preset": bool(p.get("preset")),
            }
        )
    live, _ = _router_request("GET", "/api/status")
    if isinstance(live, dict):
        cooling = int(live.get("keysCooling") or 0)
        if live.get("keyTotal"):
            key_total = int(live.get("keyTotal") or key_total)
    quota = _read_quota()
    if not quota:
        # Fall back to Nyx resetwatch cards so the phone isn't blank when
        # quota.json isn't wired into router.yaml yet.
        try:
            from api.nyx_ops import read_quota
            nyx = read_quota() or {}
            cards = nyx.get("cards") or []
            if isinstance(cards, list) and cards:
                quota = [
                    {
                        "id": str(c.get("id") or ""),
                        "remaining": c.get("remaining"),
                        "note": c.get("note") or c.get("plan") or "",
                        "reset_at": c.get("reset_at"),
                    }
                    for c in cards
                    if isinstance(c, dict) and c.get("id")
                ]
            elif isinstance(nyx.get("providers"), dict):
                quota = []
                for name, rec in nyx["providers"].items():
                    if not isinstance(rec, dict):
                        continue
                    remaining = rec.get("percent_left", rec.get("remaining"))
                    try:
                        remaining = float(remaining) if remaining is not None else None
                    except Exception:
                        remaining = None
                    quota.append({"id": str(name), "remaining": remaining, "note": rec.get("note") or rec.get("plan") or "", "reset_at": rec.get("reset_at")})
        except Exception:
            pass
    return j(
        handler,
        {
            "default": data.get("default"),
            "pools": pools,
            "listen": str((_load_router_auth()[1] or "")),
            "providers": slim,
            "fallback": data.get("fallback") or {},
            "classifier": data.get("classifier") or {},
            "keysCooling": cooling,
            "keysDead": int((live or {}).get("keysDead") or 0) if isinstance(live, dict) else 0,
            "providersCooling": int((live or {}).get("providersCooling") or 0) if isinstance(live, dict) else 0,
            "keyTotal": key_total,
            "quota": quota,
        },
    )


def handle_llm_router_set_provider(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    payload, err = _router_request("POST", "/api/config/providers", body)
    if err and payload is None:
        return bad(handler, err, 502)
    if payload and payload.get("error"):
        return bad(handler, str(payload.get("error")), 400)
    return j(handler, payload or {"ok": True})


def handle_llm_router_models(handler, parsed):
    from urllib.parse import parse_qs

    qs = parse_qs(parsed.query or "")
    q = (qs.get("q", [""])[0] or "").strip().lower()
    payload, err = _router_request("GET", "/v1/models")
    if err and payload is None:
        return bad(handler, err, 502)
    data = payload or {}
    items = data.get("data") if isinstance(data, dict) else data
    out = []
    for m in items or []:
        if isinstance(m, dict):
            mid = str(m.get("id") or "")
        else:
            mid = str(m)
        if not mid:
            continue
        if q and q not in mid.lower():
            continue
        out.append({"id": mid, "label": mid})
        if len(out) >= 80:
            break
    return j(handler, {"models": out})


def handle_llm_router_logs(handler, parsed):
    """Proxy GET /api/requests from llm-router — per-request status, provider, cascades, latency.

    Returns { entries: [...], total } where each entry carries final_status,
    final_provider, final_model, total_ms, cost and an attempts[] cascade list.
    """
    from urllib.parse import parse_qs

    qs = parse_qs(parsed.query or "")
    limit = qs.get("limit", ["40"])[0] or "40"
    try:
        int(limit)
    except (TypeError, ValueError):
        limit = "40"
    path = f"/api/requests?limit={limit}"
    for k in ("provider", "status", "pool"):
        v = (qs.get(k, [""])[0] or "").strip()
        if v:
            path += f"&{k}={v}"
    payload, err = _router_request("GET", path)
    if err and payload is None:
        return bad(handler, err, 502)
    entries = payload.get("data") if isinstance(payload, dict) else None
    if entries is None:
        entries = payload if isinstance(payload, list) else []
    return j(
        handler,
        {
            "total": payload.get("total") if isinstance(payload, dict) else None,
            "entries": entries,
        },
    )


def handle_llm_router_set_default(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    pool = str(body.get("pool") or "").strip()
    if not pool:
        return bad(handler, "pool is required")
    payload, err = _router_request("POST", "/api/config/default", {"pool": pool})
    if err and payload is None:
        return bad(handler, err, 502)
    if payload and payload.get("error"):
        return bad(handler, str(payload.get("error")), 400)
    return j(handler, payload or {"ok": True, "default": pool})


def handle_llm_router_set_pool(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    name = str(body.get("pool") or "").strip()
    entries = body.get("entries")
    if not name:
        return bad(handler, "pool is required")
    if not isinstance(entries, list) or not all(isinstance(x, str) for x in entries):
        return bad(handler, "entries must be a list of strings")
    payload, err = _router_request("POST", "/api/config/pools", {"pool": name, "entries": entries})
    if err and payload is None:
        return bad(handler, err, 502)
    if payload and payload.get("error"):
        return bad(handler, str(payload.get("error")), 400)
    return j(handler, payload or {"ok": True, "pool": name, "entries": entries})


def _card_reset_at(rec: dict):
    """Soonest reset for a resetwatch card.

    Cards carry no top-level ``reset_at``; the resets live one level down in
    ``windows`` (a session window and a weekly window, typically). The soonest
    of them is the one worth showing, since that is the next time anything
    actually frees up.
    """
    if not isinstance(rec, dict):
        return None
    stamps = [rec.get("reset_at")] if rec.get("reset_at") else []
    for w in rec.get("windows") or []:
        if isinstance(w, dict) and w.get("reset_at"):
            stamps.append(w["reset_at"])
    stamps = [str(x) for x in stamps if x]
    return min(stamps) if stamps else None


def _read_quota() -> list:
    """Remaining-quota snapshots from quota.json. Empty list if not configured."""
    try:
        import yaml  # type: ignore
    except Exception:
        return []
    if not _DEFAULT_YAML.is_file():
        return []
    try:
        data = yaml.safe_load(_DEFAULT_YAML.read_text()) or {}
    except Exception:
        return []
    path = str(data.get("quota_file") or "").strip()
    if not path:
        return []
    p = Path(path)
    if not p.is_absolute():
        p = _DEFAULT_YAML.parent / p
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return []
    if isinstance(raw, dict) and isinstance(raw.get("_cards"), list) and raw["_cards"]:
        out = []
        for rec in raw["_cards"]:
            if not isinstance(rec, dict) or not rec.get("id"):
                continue
            remaining = rec.get("remaining")
            try:
                remaining = float(remaining) if remaining is not None else None
            except Exception:
                remaining = None
            out.append({
                "id": rec["id"],
                "remaining": remaining,
                "note": rec.get("note") or rec.get("plan") or "",
                "reset_at": _card_reset_at(rec),
            })
        return out
    items = raw.get("providers") if isinstance(raw, dict) else raw
    if isinstance(raw, dict) and not items:
        items = raw
    out = []
    if isinstance(items, dict):
        for name, rec in items.items():
            if str(name).startswith("_") or name in ("providers",):
                continue
            if not isinstance(rec, dict):
                continue
            remaining = rec.get("percent_left", rec.get("remaining_pct", rec.get("remaining", rec.get("pct"))))
            try:
                remaining = float(remaining)
            except Exception:
                remaining = None
            out.append({"id": name, "remaining": remaining, "note": rec.get("note") or rec.get("plan") or "", "reset_at": rec.get("reset_at")})
    elif isinstance(items, list):
        for rec in items:
            if not isinstance(rec, dict):
                continue
            name = rec.get("id") or rec.get("provider") or rec.get("name")
            if not name:
                continue
            remaining = rec.get("percent_left", rec.get("remaining_pct", rec.get("remaining", rec.get("pct"))))
            try:
                remaining = float(remaining)
            except Exception:
                remaining = None
            out.append({"id": str(name), "remaining": remaining, "note": rec.get("note") or rec.get("plan") or "", "reset_at": rec.get("reset_at")})
    return out


def handle_llm_router_set_fallback(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    payload, err = _router_request("POST", "/api/config/fallback", body)
    if err and payload is None:
        return bad(handler, err, 502)
    if payload and payload.get("error"):
        return bad(handler, str(payload.get("error")), 400)
    return j(handler, payload or {"ok": True})


def handle_llm_router_add_key(handler, body):
    """Append or drop a key on an existing llm-router provider without wiping the stack."""
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    name = str(body.get("name") or body.get("provider") or "").strip()
    if not name:
        return bad(handler, "name is required")
    action = str(body.get("action") or "add").strip().lower()
    try:
        import yaml  # type: ignore
    except Exception:
        return bad(handler, "yaml missing", 500)
    if not _DEFAULT_YAML.is_file():
        return bad(handler, "router.yaml missing", 500)
    data = yaml.safe_load(_DEFAULT_YAML.read_text()) or {}
    providers = data.get("providers") or {}
    rec = None
    if isinstance(providers, dict):
        rec = providers.get(name)
        if rec is None:
            custom = providers.get("custom")
            if isinstance(custom, dict):
                rec = custom.get(name)
    if not isinstance(rec, dict):
        return bad(handler, f"provider {name} not in router.yaml", 404)
    keys = [str(k) for k in (rec.get("keys") or []) if str(k).strip()]
    labels = [str(x) for x in (rec.get("key_labels") or [])]
    if action == "remove":
        try:
            idx = int(body.get("index"))
        except Exception:
            return bad(handler, "index required")
        if idx < 0 or idx >= len(keys):
            return bad(handler, "index out of range", 400)
        keys.pop(idx)
        if idx < len(labels):
            labels.pop(idx)
    else:
        new_key = str(body.get("api_key") or body.get("key") or "").strip()
        if not new_key:
            return bad(handler, "api_key required")
        if new_key in keys:
            return j(handler, {"ok": True, "name": name, "key_count": len(keys), "deduped": True})
        keys.append(new_key)
        label = str(body.get("label") or "").strip()
        if label:
            while len(labels) < len(keys) - 1:
                labels.append("")
            labels.append(label)
    payload, err = _router_request(
        "POST",
        "/api/config/keys",
        {"name": name, "keys": keys, "key_labels": labels},
    )
    if err and payload is None:
        return bad(handler, err, 502)
    if payload and payload.get("error"):
        return bad(handler, str(payload.get("error")), 400)
    return j(handler, {"ok": True, "name": name, "key_count": len(keys)})
