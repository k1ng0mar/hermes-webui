"""Nyx agent model routing + provider keys.

GET  /api/nyx/routing
POST /api/nyx/routing/primary   {model, provider?}
POST /api/nyx/routing/fallback  {entries: [{provider, model}]}
POST /api/nyx/routing/aux       {task, provider, model}
POST /api/nyx/routing/key       {provider, api_key}
GET  /api/nyx/models?q=
"""
from __future__ import annotations

from api.helpers import bad, j


def handle_get(handler):
    from api.config import get_auxiliary_models, reload_config, cfg
    from api.providers import get_providers

    reload_config()
    aux = get_auxiliary_models()
    main = aux.get("main") or {}
    raw_fb = cfg.get("fallback_providers") if isinstance(cfg, dict) else []
    fallback = []
    if isinstance(raw_fb, list):
        for e in raw_fb:
            if not isinstance(e, dict):
                continue
            provider = str(e.get("provider") or "").strip()
            model = str(e.get("model") or "").strip()
            if provider and model:
                fallback.append({"provider": provider, "model": model})
    providers = []
    try:
        raw = get_providers()
        lst = raw.get("providers") if isinstance(raw, dict) else raw
        if isinstance(lst, list):
            for p in lst:
                if not isinstance(p, dict):
                    continue
                providers.append(
                    {
                        "id": str(p.get("id") or ""),
                        "name": str(p.get("display_name") or p.get("name") or p.get("id") or ""),
                        "hasKey": bool(p.get("has_key") or p.get("configured")),
                        "keyHint": "••••" if p.get("has_key") else None,
                        "configurable": bool(p.get("configurable", True)),
                    }
                )
    except Exception:
        providers = []
    tasks = []
    for t in aux.get("tasks") or []:
        tasks.append(
            {
                "role": t.get("task") or t.get("key"),
                "label": t.get("label") or t.get("task"),
                "provider": t.get("provider") or "auto",
                "model": t.get("model") or "",
            }
        )
    return j(
        handler,
        {
            "primary": {
                "provider": main.get("provider") or "",
                "model": main.get("model") or "",
            },
            "fallback": fallback,
            "aux": tasks,
            "providers": [p for p in providers if p.get("id")],
            "pools": _list_pools(),
            "disabled_toolsets": list((cfg.get("agent") or {}).get("disabled_toolsets") or []) if isinstance(cfg, dict) else [],
        },
    )


def handle_primary(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    model = str(body.get("model") or "").strip()
    provider = str(body.get("provider") or "").strip() or None
    if not model:
        return bad(handler, "model is required")
    from api.config import set_hermes_default_model

    try:
        return j(handler, set_hermes_default_model(model, provider=provider))
    except Exception as e:
        return bad(handler, str(e), 400)


def handle_fallback(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    entries = body.get("entries")
    if not isinstance(entries, list):
        return bad(handler, "entries must be a list")
    cleaned = []
    for e in entries:
        if isinstance(e, str) and ":" in e:
            provider, model = e.split(":", 1)
        elif isinstance(e, dict):
            provider = str(e.get("provider") or "").strip()
            model = str(e.get("model") or "").strip()
        else:
            continue
        provider, model = provider.strip(), model.strip()
        if provider and model:
            cleaned.append({"provider": provider, "model": model})
    from api.config import _cfg_lock, _get_config_path, _load_yaml_config_file, _save_yaml_config_file, reload_config

    path = _get_config_path()
    with _cfg_lock:
        data = _load_yaml_config_file(path)
        data["fallback_providers"] = cleaned
        _save_yaml_config_file(path, data)
    reload_config()
    return j(handler, {"ok": True, "fallback": cleaned})


def handle_aux(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    task = str(body.get("task") or "").strip()
    provider = str(body.get("provider") or "auto").strip() or "auto"
    model = str(body.get("model") or "").strip()
    if not task:
        return bad(handler, "task is required")
    from api.config import set_auxiliary_model

    try:
        return j(handler, set_auxiliary_model(task, provider, model))
    except Exception as e:
        return bad(handler, str(e), 400)


def handle_key(handler, body):
    """Append a credential-pool key (does not overwrite existing keys)."""
    return handle_pool_add(handler, body)


def handle_pool_add(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    provider = str(body.get("provider") or "").strip().lower()
    api_key = str(body.get("api_key") or "").strip()
    label = str(body.get("label") or "").strip()
    if not provider or not api_key:
        return bad(handler, "provider and api_key are required")
    try:
        import uuid

        from agent.credential_pool import (
            AUTH_TYPE_API_KEY,
            CUSTOM_POOL_PREFIX,
            SOURCE_MANUAL,
            PooledCredential,
            load_pool,
        )

        pool = load_pool(provider)
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label or f"key #{len(pool.entries()) + 1}",
            auth_type=AUTH_TYPE_API_KEY,
            priority=0,
            source=SOURCE_MANUAL,
            access_token=api_key,
        )
        pool.add_entry(entry)
        if not provider.startswith(CUSTOM_POOL_PREFIX):
            try:
                from hermes_cli.auth import _load_auth_store, unsuppress_credential_source

                suppressed = _load_auth_store().get("suppressed_sources", {})
                for src in list(suppressed.get(provider, []) or []):
                    unsuppress_credential_source(provider, src)
            except Exception:
                pass
        try:
            from api.config import invalidate_credential_pool_cache, invalidate_models_cache, invalidate_providers_cache

            invalidate_credential_pool_cache(provider)
            invalidate_models_cache()
            invalidate_providers_cache()
        except Exception:
            pass
        return j(handler, {"ok": True, "provider": provider, "count": len(pool.entries())})
    except Exception as e:
        return bad(handler, str(e), 400)


def handle_pool_remove(handler, body):
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    provider = str(body.get("provider") or "").strip().lower()
    try:
        index = int(body.get("index"))
    except Exception:
        return bad(handler, "index is required (1-based)")
    if not provider or index < 1:
        return bad(handler, "provider and 1-based index required")
    try:
        from agent.credential_pool import load_pool
        from agent.credential_sources import find_removal_step
        from hermes_cli.auth import suppress_credential_source

        pool = load_pool(provider)
        removed = pool.remove_index(index)
        if removed is None:
            return bad(handler, f"index {index} out of range", 404)
        src = getattr(removed, "source", "") or ""
        step = find_removal_step(provider, src)
        if step is not None:
            try:
                result = step.remove_fn(provider, removed)
                if getattr(result, "suppress", False) and src:
                    suppress_credential_source(provider, src)
            except Exception:
                if src:
                    try:
                        suppress_credential_source(provider, src)
                    except Exception:
                        pass
        try:
            from api.config import invalidate_credential_pool_cache, invalidate_models_cache, invalidate_providers_cache

            invalidate_credential_pool_cache(provider)
            invalidate_models_cache()
            invalidate_providers_cache()
        except Exception:
            pass
        return j(handler, {"ok": True, "provider": provider, "count": len(pool.entries())})
    except Exception as e:
        return bad(handler, str(e), 400)


def handle_mcp_tool(handler, body):
    """Enable/disable a tool via agent.disabled_toolsets (official Hermes knob)."""
    if not isinstance(body, dict):
        return bad(handler, "JSON object required")
    name = str(body.get("name") or "").strip()
    if not name:
        return bad(handler, "name is required")
    enabled = bool(body.get("enabled"))
    from api.config import _cfg_lock, _get_config_path, _load_yaml_config_file, _save_yaml_config_file, reload_config

    path = _get_config_path()
    with _cfg_lock:
        data = _load_yaml_config_file(path)
        agent = data.get("agent")
        if not isinstance(agent, dict):
            agent = {}
            data["agent"] = agent
        disabled = [str(x) for x in (agent.get("disabled_toolsets") or []) if str(x).strip()]
        if enabled:
            disabled = [x for x in disabled if x != name]
        elif name not in disabled:
            disabled.append(name)
        agent["disabled_toolsets"] = disabled
        _save_yaml_config_file(path, data)
    reload_config()
    return j(handler, {"ok": True, "name": name, "enabled": enabled, "disabled_toolsets": disabled})


def _list_pools() -> list:
    out = []
    try:
        from agent.credential_pool import load_pool
        from hermes_cli.auth import read_credential_pool
        from hermes_cli.config import redact_key
    except Exception:
        return out
    try:
        raw = read_credential_pool() or {}
    except Exception:
        return out
    for pid in sorted(raw.keys()):
        try:
            pool = load_pool(pid)
            entries = pool.entries() or []
        except Exception:
            continue
        rows = []
        cooling = 0
        for i, e in enumerate(entries, start=1):
            status = str(getattr(e, "last_status", "") or "ok")
            if status and status not in ("ok", "none", "None"):
                cooling += 1
            token = getattr(e, "access_token", "") or ""
            try:
                preview = redact_key(token) if token else ""
            except Exception:
                preview = "••••"
            rows.append(
                {
                    "index": i,
                    "id": getattr(e, "id", None),
                    "label": getattr(e, "label", None) or f"#{i}",
                    "auth_type": getattr(e, "auth_type", None),
                    "source": getattr(e, "source", None),
                    "last_status": status,
                    "request_count": getattr(e, "request_count", 0),
                    "token_preview": preview,
                }
            )
        if rows:
            out.append({"provider": pid, "entries": rows, "cooling": cooling, "count": len(rows)})
    return out


def handle_models(handler, parsed):
    from urllib.parse import parse_qs

    from api.config import get_available_models

    qs = parse_qs(parsed.query or "")
    q = (qs.get("q", [""])[0] or "").strip().lower()
    try:
        raw = get_available_models()
    except Exception as e:
        return bad(handler, str(e), 502)
    groups = raw.get("providers") or raw.get("groups") or raw.get("catalog") or []
    if isinstance(raw, list):
        groups = raw
    out = []
    if isinstance(groups, list):
        for g in groups:
            if not isinstance(g, dict):
                continue
            pid = str(g.get("provider_id") or g.get("id") or g.get("name") or "")
            pname = str(g.get("name") or g.get("provider") or pid)
            for m in g.get("models") or []:
                if isinstance(m, str):
                    mid, label = m, m
                elif isinstance(m, dict):
                    mid = str(m.get("id") or m.get("model") or "")
                    label = str(m.get("label") or mid)
                else:
                    continue
                if not mid:
                    continue
                hay = f"{pid} {pname} {mid} {label}".lower()
                if q and q not in hay:
                    continue
                out.append({"id": mid, "label": label, "provider": pid or pname})
                if len(out) >= 80:
                    break
            if len(out) >= 80:
                break
    return j(handler, {"models": out, "q": q})
