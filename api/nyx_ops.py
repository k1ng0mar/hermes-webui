"""Nyx ops: quota.json, skill hub, MCP catalog, curator, goals, heartbeats.

These wrap live Hermes CLI / profile stores. No invented numbers.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from api.helpers import bad, j

QUOTA_PATH = Path(os.environ.get("NYX_QUOTA_JSON", "/home/ubuntu/llm-router/quota.json"))
ROUTER_YAML = Path(os.environ.get("NYX_ROUTER_YAML", "/home/ubuntu/llm-router/router.yaml"))
HERMES = os.environ.get("NYX_HERMES_BIN", "/home/ubuntu/.local/bin/hermes")
STATE_DB = Path.home() / ".hermes" / "state.db"
ENV_FILE = Path.home() / ".hermes" / ".env"


def _load_env() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENV_FILE.is_file():
        return out
    for line in ENV_FILE.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _run(args: list[str], timeout: int = 45) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "HERMES_ACCEPT_HOOKS": "1"},
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except Exception as e:
        return 1, "", str(e)


def _http_json(url: str, key: str, timeout: int = 10):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


PROBE = Path(os.environ.get("NYX_RESETWATCH_PROBE", "/home/ubuntu/vault/projects/hermes-resetwatch/probe.py"))
QUOTA_STALE_S = 5 * 60

# Probe provider → llm-router pool provider names (normed later).
_PROBE_ALIASES = {
    "anthropic": ["anthropic", "claude"],
    "openai-codex": ["openai-codex", "openai", "codex"],
    "ollama": ["ollama", "ollama-cloud"],
    "openrouter": ["openrouter"],
    "nous": ["nous"],
    "grok": ["grok", "xai", "xai-oauth"],
    "kimi": ["kimi"],
    "glm": ["glm", "zai"],
    "deepseek": ["deepseek"],
    "minimax": ["minimax"],
}


def _tightest_window(windows: list) -> dict | None:
    best = None
    best_pct = 1e9
    for w in windows or []:
        if not isinstance(w, dict):
            continue
        pct = w.get("remaining_percent")
        if pct is None:
            continue
        try:
            val = float(pct)
        except Exception:
            continue
        if val < best_pct:
            best_pct = val
            best = w
    return best


def _snapshots_to_quota(snaps: list, now: str) -> tuple[dict, list]:
    out: dict = {}
    cards = []
    for snap in snaps:
        if not isinstance(snap, dict):
            continue
        provider = str(snap.get("provider") or "").strip()
        if not provider or provider == "resetwatch":
            continue
        windows = [w for w in (snap.get("windows") or []) if isinstance(w, dict)]
        tight = _tightest_window(windows)
        pct = None if not tight else tight.get("remaining_percent")
        reset = None if not tight else tight.get("reset_at")
        bits = []
        if snap.get("plan"):
            bits.append(str(snap["plan"]))
        for w in windows:
            if w.get("remaining_percent") is None:
                continue
            label = w.get("label") or "limit"
            bits.append(f"{label} {float(w['remaining_percent']):.0f}%")
        note = " · ".join(bits)
        rec = {
            "percent_left": pct,
            "reset_at": reset,
            "note": note,
            "plan": snap.get("plan"),
            "windows": [
                {
                    "label": w.get("label"),
                    "remaining_percent": w.get("remaining_percent"),
                    "used_percent": w.get("used_percent"),
                    "reset_at": w.get("reset_at"),
                    "detail": w.get("detail"),
                }
                for w in windows
            ],
            "source": "resetwatch",
            "updated_at": now,
        }
        key = provider.lower()
        out[key] = rec
        for alias in _PROBE_ALIASES.get(key, []):
            if alias not in out:
                out[alias] = rec
        cards.append({"id": key, "remaining": pct, "note": note, "plan": snap.get("plan"), "windows": rec["windows"]})
    return out, cards


def run_resetwatch(fresh: bool = False) -> tuple[list, str]:
    if not PROBE.is_file():
        return [], f"probe missing: {PROBE}"
    args = ["python3", str(PROBE)]
    if fresh:
        args.append("--fresh")
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=55)
    except Exception as e:
        return [], str(e)
    text = (p.stdout or "").strip()
    # probe prints only JSON; ignore any leading noise
    start = text.find("[")
    if start < 0:
        return [], (p.stderr or "empty probe")[:400]
    try:
        data = json.loads(text[start:])
    except Exception as e:
        return [], f"probe json: {e}"
    if not isinstance(data, list):
        return [], "probe did not return an array"
    return data, ""


def refresh_quota(fresh: bool = True) -> dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    snaps, err = run_resetwatch(fresh=fresh)
    notes = [err] if err else []
    snap, cards = _snapshots_to_quota(snaps, now)

    # Keep only manual overlays that resetwatch did not cover.
    if QUOTA_PATH.is_file():
        try:
            existing = json.loads(QUOTA_PATH.read_text()) or {}
        except Exception:
            existing = {}
        if isinstance(existing, dict):
            for k, v in existing.items():
                if k.startswith("_") or k in snap:
                    continue
                if isinstance(v, dict) and v.get("source") == "manual" and v.get("percent_left") is not None:
                    snap[k] = v

    QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(snap)
    payload["_updated_at"] = now
    payload["_source"] = "hermes-resetwatch"
    if notes:
        payload["_errors"] = notes
    payload["_cards"] = cards
    QUOTA_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    try:
        text = ROUTER_YAML.read_text()
        if 'quota_file: ""' in text or "quota_file: ''" in text:
            ROUTER_YAML.write_text(
                text.replace('quota_file: ""', f'quota_file: "{QUOTA_PATH}"').replace(
                    "quota_file: ''", f'quota_file: "{QUOTA_PATH}"'
                )
            )
    except Exception:
        pass

    return {
        "path": str(QUOTA_PATH),
        "providers": {k: v for k, v in payload.items() if not k.startswith("_")},
        "cards": cards,
        "errors": notes,
        "updated_at": now,
        "source": "resetwatch",
    }


def read_quota() -> dict:
    if not QUOTA_PATH.is_file():
        return {"path": str(QUOTA_PATH), "providers": {}, "empty": True, "source": "resetwatch"}
    try:
        raw = json.loads(QUOTA_PATH.read_text())
    except Exception as e:
        return {"path": str(QUOTA_PATH), "providers": {}, "error": str(e)}
    providers = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                pct = v.get("percent_left", v.get("remaining_pct", v.get("remaining")))
                providers[k] = {
                    "percent_left": pct,
                    "reset_at": v.get("reset_at"),
                    "note": v.get("note") or "",
                    "plan": v.get("plan"),
                    "windows": v.get("windows") or [],
                    "source": v.get("source") or "",
                    "updated_at": v.get("updated_at"),
                }
            else:
                try:
                    providers[k] = {"percent_left": float(v), "note": ""}
                except Exception:
                    continue
    return {
        "path": str(QUOTA_PATH),
        "providers": providers,
        "cards": raw.get("_cards") if isinstance(raw, dict) else [],
        "updated_at": raw.get("_updated_at") if isinstance(raw, dict) else None,
        "source": raw.get("_source") if isinstance(raw, dict) else None,
        "errors": raw.get("_errors") if isinstance(raw, dict) else [],
    }


def _quota_age_s() -> float:
    if not QUOTA_PATH.is_file():
        return 1e9
    try:
        raw = json.loads(QUOTA_PATH.read_text())
        ts = str((raw or {}).get("_updated_at") or "")
        if not ts:
            return time.time() - QUOTA_PATH.stat().st_mtime
        # 2026-08-22T17:42:18Z
        t = time.strptime(ts.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        return time.time() - time.mktime(t) + time.timezone
    except Exception:
        return 1e9


def handle_quota_get(handler):
    if _quota_age_s() > QUOTA_STALE_S:
        refresh_quota(fresh=False)
    return j(handler, read_quota())


def handle_quota_refresh(handler, body=None):
    body = body or {}
    if body.get("providers"):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        current = {}
        if QUOTA_PATH.is_file():
            try:
                current = json.loads(QUOTA_PATH.read_text()) or {}
            except Exception:
                current = {}
        if not isinstance(current, dict):
            current = {}
        for name, rec in (body.get("providers") or {}).items():
            if not name:
                continue
            if isinstance(rec, (int, float)):
                rec = {"percent_left": float(rec)}
            if not isinstance(rec, dict):
                continue
            current[str(name)] = {
                "percent_left": rec.get("percent_left"),
                "reset_at": rec.get("reset_at"),
                "note": rec.get("note") or "manual",
                "source": "manual",
                "updated_at": now,
            }
        current["_updated_at"] = now
        QUOTA_PATH.write_text(json.dumps(current, indent=2) + "\n")
        return j(handler, read_quota())
    return j(handler, refresh_quota(fresh=bool(body.get("fresh", True))))


def handle_skills_hub(handler, parsed):
    from urllib.parse import parse_qs

    qs = parse_qs(parsed.query or "")
    q = (qs.get("q") or [""])[0].strip() or "agent"
    limit = (qs.get("limit") or ["20"])[0]
    code, out, err = _run([HERMES, "skills", "search", q, "--limit", str(limit), "--json"], timeout=60)
    items = []
    if out.strip().startswith("["):
        try:
            items = json.loads(out)
        except Exception:
            items = []
    return j(handler, {"query": q, "results": items, "ok": code == 0, "error": err.strip()[:400] if code else ""})


def handle_skills_install(handler, body):
    ident = str(body.get("identifier") or body.get("name") or "").strip()
    if not ident:
        return bad(handler, "identifier required")
    code, out, err = _run([HERMES, "skills", "install", ident, "--yes"], timeout=120)
    return j(handler, {"ok": code == 0, "identifier": ident, "output": (out or err)[-2000:]})


def handle_skills_toggle(handler, body):
    name = str(body.get("name") or "").strip()
    enabled = bool(body.get("enabled"))
    if not name:
        return bad(handler, "name required")
    # webui already has POST /api/skills/toggle — reuse if present
    try:
        from api.routes import _handle_skill_toggle  # type: ignore
    except Exception:
        _handle_skill_toggle = None
    if _handle_skill_toggle:
        return _handle_skill_toggle(handler, body)
    action = "enable" if enabled else "disable"
    code, out, err = _run([HERMES, "skills", "config", action, name], timeout=30)
    return j(handler, {"ok": code == 0, "name": name, "enabled": enabled, "output": (out or err)[-800:]})


def handle_mcp_catalog(handler):
    try:
        import sys

        sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent"))
        from hermes_cli.mcp_catalog import list_catalog  # type: ignore

        entries = []
        for e in list_catalog():
            entries.append(
                {
                    "name": getattr(e, "name", None) or getattr(e, "id", ""),
                    "description": getattr(e, "description", "") or "",
                    "status": getattr(e, "status", "") or "available",
                    "transport": getattr(e, "transport", "") or "",
                }
            )
        return j(handler, {"catalog": entries})
    except Exception:
        code, out, err = _run([HERMES, "mcp", "catalog"], timeout=30)
        rows = []
        for line in (out or "").splitlines():
            line = line.strip()
            if not line or line.startswith("Name") or line.startswith("-") or line.startswith("MCP") or line.startswith("Install"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append({"name": parts[0], "status": parts[1], "description": " ".join(parts[2:])})
        return j(handler, {"catalog": rows, "raw": out[-1500:], "ok": code == 0})


def handle_mcp_install(handler, body):
    name = str(body.get("name") or "").strip()
    if not name:
        return bad(handler, "name required")
    code, out, err = _run([HERMES, "mcp", "install", name], timeout=90)
    return j(handler, {"ok": code == 0, "name": name, "output": (out or err)[-2000:]})


def handle_curator_get(handler):
    code, out, err = _run([HERMES, "curator", "status"], timeout=30)
    usage = {}
    try:
        from api.skill_usage import read_skill_usage

        usage = read_skill_usage() or {}
    except Exception:
        usage = {}
    return j(
        handler,
        {
            "ok": code == 0,
            "status_text": out.strip(),
            "error": err.strip()[:400] if code else "",
            "usage_count": len(usage.get("usage") or usage) if isinstance(usage, dict) else 0,
        },
    )


def handle_curator_post(handler, body):
    action = str(body.get("action") or "run").strip().lower()
    name = str(body.get("name") or "").strip()
    allowed = {"run", "pause", "resume", "pin", "unpin", "archive", "restore", "adopt"}
    if action not in allowed:
        return bad(handler, f"action must be one of {sorted(allowed)}")
    args = [HERMES, "curator", action]
    if action in {"pin", "unpin", "archive", "restore", "adopt"}:
        if not name:
            return bad(handler, "name required")
        args.append(name)
    code, out, err = _run(args, timeout=180)
    return j(handler, {"ok": code == 0, "action": action, "output": (out or err)[-2000:]})


def _state_rows(prefix: str) -> list[dict]:
    if not STATE_DB.is_file():
        return []
    con = sqlite3.connect(str(STATE_DB))
    try:
        rows = con.execute("SELECT key, value FROM state_meta WHERE key LIKE ?", (prefix + "%",)).fetchall()
    finally:
        con.close()
    out = []
    for key, value in rows:
        sid = key.split(":", 1)[-1]
        try:
            data = json.loads(value)
        except Exception:
            data = {"raw": value}
        data["session_id"] = sid
        out.append(data)
    return out


def handle_goals_get(handler):
    items = _state_rows("goal:")
    items.sort(key=lambda g: float(g.get("last_turn_at") or g.get("created_at") or 0), reverse=True)
    return j(handler, {"goals": items[:80], "total": len(items)})


def handle_goal_action(handler, body):
    """Pause/resume/clear a goal row in state_meta — mirrors GoalManager.pause/resume/clear."""
    sid = str((body or {}).get("session_id") or "").strip()
    action = str((body or {}).get("action") or "").strip().lower()
    if not sid:
        return bad(handler, "session_id required")
    key = f"goal:{sid}"
    con = sqlite3.connect(str(STATE_DB))
    try:
        row = con.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
        if not row:
            return bad(handler, "no goal on that session", 404)
        data = json.loads(row[0])
        if action == "pause":
            data["status"] = "paused"
            data["paused_reason"] = "user-paused (mobile)"
            data["waiting_on_session"] = None
            data["waiting_until"] = 0.0
        elif action == "resume":
            data["status"] = "active"
            data["paused_reason"] = None
            data["waiting_until"] = 0.0
            data["turns_used"] = 0
        elif action == "clear":
            data["status"] = "cleared"
        else:
            return bad(handler, "action must be pause|resume|clear")
        con.execute("UPDATE state_meta SET value = ? WHERE key = ?", (json.dumps(data), key))
        con.commit()
    finally:
        con.close()
    return j(handler, {"ok": True, "session_id": sid, "status": data["status"]})


def handle_goal_create(handler, body):
    """Attach a goal to a session — the /goal contract: goal + session_id."""
    sid = str((body or {}).get("session_id") or "").strip()
    goal = str((body or {}).get("goal") or "").strip()
    max_turns = int((body or {}).get("max_turns") or 1000)
    if not sid or not goal:
        return bad(handler, "session_id and goal required")
    rec = {
        "goal": goal,
        "status": "active",
        "turns_used": 0,
        "max_turns": max_turns,
        "created_at": time.time(),
        "last_turn_at": time.time(),
        "created_by": "mobile",
    }
    con = sqlite3.connect(str(STATE_DB))
    try:
        con.execute(
            "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
            (f"goal:{sid}", json.dumps(rec)),
        )
        con.commit()
    finally:
        con.close()
    return j(handler, {"ok": True, "goal": {**rec, "session_id": sid}})


def handle_heartbeats_get(handler):
    items = _state_rows("heartbeat:")
    return j(handler, {"heartbeats": items, "total": len(items)})


def handle_heartbeat_set(handler, body):
    """Write a heartbeat into state_meta. Session-scoped; gateway must be running to fire."""
    sid = str(body.get("session_id") or "").strip()
    prompt = str(body.get("prompt") or "").strip()
    interval = str(body.get("interval") or "10m").strip()
    action = str(body.get("action") or "set").strip().lower()
    if not sid:
        return bad(handler, "session_id required")
    if action == "clear":
        con = sqlite3.connect(str(STATE_DB))
        try:
            con.execute("DELETE FROM state_meta WHERE key = ?", (f"heartbeat:{sid}",))
            con.commit()
        finally:
            con.close()
        return j(handler, {"ok": True, "cleared": sid})
    if action in {"pause", "resume"}:
        con = sqlite3.connect(str(STATE_DB))
        try:
            row = con.execute("SELECT value FROM state_meta WHERE key = ?", (f"heartbeat:{sid}",)).fetchone()
            if not row:
                return bad(handler, "no heartbeat on that session", 404)
            data = json.loads(row[0])
            data["status"] = "paused" if action == "pause" else "active"
            con.execute("UPDATE state_meta SET value = ? WHERE key = ?", (json.dumps(data), f"heartbeat:{sid}"))
            con.commit()
        finally:
            con.close()
        return j(handler, {"ok": True, "session_id": sid, "status": data["status"]})
    if not prompt:
        return bad(handler, "prompt required")
    # parse interval
    seconds = 600
    try:
        import sys

        sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent"))
        from hermes_cli.heartbeat import parse_interval  # type: ignore

        parsed = parse_interval(interval)
        if parsed and parsed > 0:
            seconds = parsed
    except Exception:
        pass
    rec = {
        "prompt": prompt,
        "interval_seconds": seconds,
        "status": "active",
        "created_at": time.time(),
        "last_fired_at": 0.0,
        "fire_count": 0,
    }
    con = sqlite3.connect(str(STATE_DB))
    try:
        con.execute(
            "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
            (f"heartbeat:{sid}", json.dumps(rec)),
        )
        con.commit()
    finally:
        con.close()
    return j(handler, {"ok": True, "heartbeat": {**rec, "session_id": sid}})
