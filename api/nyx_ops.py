"""Nyx ops: quota.json, skill hub, MCP catalog, curator, goals, heartbeats.

These wrap live Hermes CLI / profile stores. No invented numbers.
"""
from __future__ import annotations

import calendar
import datetime
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs

from api.helpers import bad, j
from api.nyx_store import atomic_write_json, atomic_write_text, load_json

logger = logging.getLogger(__name__)

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
    for k, v in load_json(QUOTA_PATH, {}).items():
        if k.startswith("_") or k in snap:
            continue
        if isinstance(v, dict) and v.get("source") == "manual" and v.get("percent_left") is not None:
            snap[k] = v

    payload = dict(snap)
    payload["_updated_at"] = now
    payload["_source"] = "hermes-resetwatch"
    if notes:
        payload["_errors"] = notes
    payload["_cards"] = cards
    atomic_write_json(QUOTA_PATH, payload)

    # router.yaml holds the only copy of every provider API key. This rewrite
    # only ever fills in an empty `quota_file:`, but a truncate-then-write here
    # meant a crash mid-write destroyed the keys with no backup. Go through the
    # atomic writer (temp + fsync + rename, ownership/mode preserved) and only
    # after re-reading and confirming the substitution actually changed
    # something, so a no-op never rewrites the file at all.
    try:
        text = ROUTER_YAML.read_text()
        if 'quota_file: ""' in text or "quota_file: ''" in text:
            updated = text.replace(
                'quota_file: ""', f'quota_file: "{QUOTA_PATH}"'
            ).replace("quota_file: ''", f'quota_file: "{QUOTA_PATH}"')
            if updated != text:
                atomic_write_text(ROUTER_YAML, updated)
    except Exception:
        logger.warning("Could not point router.yaml at the quota file", exc_info=True)

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
    raw = load_json(QUOTA_PATH, {})
    providers = {}
    if raw:
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
        "cards": raw.get("_cards") or [],
        "updated_at": raw.get("_updated_at"),
        "source": raw.get("_source"),
        "errors": raw.get("_errors") or [],
    }


def _quota_age_s() -> float:
    if not QUOTA_PATH.is_file():
        return 1e9
    raw = load_json(QUOTA_PATH, {})
    ts = str(raw.get("_updated_at") or "")
    if not ts:
        try:
            return time.time() - QUOTA_PATH.stat().st_mtime
        except OSError:
            return 1e9
    try:
        # 2026-08-22T17:42:18Z — the stamp is UTC, so it must be converted with
        # calendar.timegm. The previous `mktime(t) + time.timezone` read it as
        # local time and corrected by the *standard* offset, so during DST the
        # age was off by an hour — enough to call a fresh file stale (or a stale
        # one fresh) right around the 5-minute boundary.
        t = time.strptime(ts.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        return time.time() - calendar.timegm(t)
    except (ValueError, OverflowError):
        return 1e9


# Serializes every quota refresh AND every manual-overlay write. Two guarantees:
#   * one resetwatch probe at a time — a plain GET used to fan out an unbounded
#     number of 55s `python3 probe.py` subprocesses (one per concurrent stale
#     request), all racing to write quota.json;
#   * read-modify-write of the manual overlays can't interleave and lose an entry.
_quota_lock = threading.Lock()


def _refresh_quota_deduped(fresh: bool) -> None:
    """Refresh at most once per staleness window, no matter how many callers.

    Latecomers that arrive while a probe is in flight block on the lock, then
    re-check the age and return immediately — they get the fresh result the
    winner just wrote instead of launching a probe of their own.
    """
    with _quota_lock:
        if _quota_age_s() <= QUOTA_STALE_S:
            return
        refresh_quota(fresh=fresh)


def handle_quota_get(handler):
    if _quota_age_s() > QUOTA_STALE_S:
        _refresh_quota_deduped(fresh=False)
    return j(handler, read_quota())


def handle_quota_refresh(handler, body=None):
    body = body or {}
    if body.get("providers"):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with _quota_lock:
            _apply_manual_overlays(body.get("providers") or {}, now)
        return j(handler, read_quota())
    with _quota_lock:
        return j(handler, refresh_quota(fresh=bool(body.get("fresh", True))))


def _apply_manual_overlays(providers: dict, now: str) -> None:
    """Merge caller-supplied manual quota figures. Caller holds _quota_lock."""
    current = load_json(QUOTA_PATH, {})
    for name, rec in (providers or {}).items():
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
    atomic_write_json(QUOTA_PATH, current)


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


# Install identifiers are passed as argv to the Hermes CLI. There is no shell,
# so no shell injection — but a value starting with "-" is parsed by the CLI as
# a FLAG rather than a package name, which turns "install this skill" into
# "install with these options" on an endpoint that already passes --yes.
# Restrict to the shapes a real skill/MCP identifier takes (name,
# scope/name, owner/repo, or a URL) and reject anything leading with a dash.
_IDENT_RE = re.compile(r"^(?:https?://[^\s]{1,400}|[A-Za-z0-9][A-Za-z0-9._@/+-]{0,200})$")


def _validate_identifier(value: str, field: str = "identifier") -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{field} required")
    if value.startswith("-"):
        raise ValueError(f"{field} may not start with '-'")
    if not _IDENT_RE.match(value):
        raise ValueError(f"invalid {field}")
    return value


def handle_skills_install(handler, body):
    try:
        ident = _validate_identifier(
            (body or {}).get("identifier") or (body or {}).get("name") or ""
        )
    except ValueError as e:
        return bad(handler, str(e))
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


def _ensure_agent_on_path() -> None:
    """Put the hermes-agent tree on sys.path exactly once.

    The unguarded `sys.path.insert(0, ...)` this replaces ran on EVERY request
    to the calling endpoints, so sys.path grew without bound for the life of the
    process and every import in the server paid a longer linear scan.
    """
    import sys

    agent_dir = str(Path.home() / ".hermes" / "hermes-agent")
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)


def handle_mcp_catalog(handler):
    try:
        _ensure_agent_on_path()
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
    try:
        name = _validate_identifier((body or {}).get("name") or "", "name")
    except ValueError as e:
        return bad(handler, str(e))
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
        try:
            name = _validate_identifier(name, "name")
        except ValueError as e:
            return bad(handler, str(e))
        args.append(name)
    code, out, err = _run(args, timeout=180)
    return j(handler, {"ok": code == 0, "action": action, "output": (out or err)[-2000:]})


# state.db is written concurrently by the agent (GoalManager, the heartbeat
# scheduler). sqlite's default 5s busy timeout turns any contended read into a
# 500, so give writers room to finish; isolation_level=None puts us in explicit
# transaction control, which the read-modify-write helpers below need.
_STATE_DB_TIMEOUT_S = 15.0


def _state_db() -> sqlite3.Connection:
    return sqlite3.connect(
        str(STATE_DB), timeout=_STATE_DB_TIMEOUT_S, isolation_level=None
    )


def _state_rows(prefix: str) -> list[dict]:
    if not STATE_DB.is_file():
        return []
    con = _state_db()
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
    if action not in ("pause", "resume", "clear"):
        return bad(handler, "action must be pause|resume|clear")
    key = f"goal:{sid}"
    con = _state_db()
    try:
        # BEGIN IMMEDIATE takes the write lock up front, so the agent's own
        # GoalManager cannot land an update between this SELECT and the UPDATE
        # below and have it silently discarded.
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
        if not row:
            con.execute("ROLLBACK")
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
        else:
            data["status"] = "cleared"
        con.execute("UPDATE state_meta SET value = ? WHERE key = ?", (json.dumps(data), key))
        con.execute("COMMIT")
    except BaseException:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        con.close()
    return j(handler, {"ok": True, "session_id": sid, "status": data["status"]})


def handle_goal_create(handler, body):
    """Attach a goal to a session — the /goal contract: goal + session_id."""
    sid = str((body or {}).get("session_id") or "").strip()
    goal = str((body or {}).get("goal") or "").strip()
    if not sid or not goal:
        return bad(handler, "session_id and goal required")
    # A bare int() on request input raised ValueError straight past the handler
    # into the server's catch-all, surfacing as an opaque 500 instead of a 400.
    # Note the default is applied ONLY when the field is absent: writing
    # `int(x or 1000)` lets every falsy value — 0, "", {}, [] — short-circuit
    # past the range check and silently become 1000.
    raw_max = (body or {}).get("max_turns")
    if raw_max is None or raw_max == "":
        max_turns = 1000
    elif isinstance(raw_max, bool) or not isinstance(raw_max, (int, float, str)):
        return bad(handler, "max_turns must be an integer")
    else:
        try:
            max_turns = int(raw_max)
        except (TypeError, ValueError):
            return bad(handler, "max_turns must be an integer")
    if not 1 <= max_turns <= 100000:
        return bad(handler, "max_turns must be between 1 and 100000")
    rec = {
        "goal": goal,
        "status": "active",
        "turns_used": 0,
        "max_turns": max_turns,
        "created_at": time.time(),
        "last_turn_at": time.time(),
        "created_by": "mobile",
    }
    con = _state_db()
    try:
        con.execute(
            "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
            (f"goal:{sid}", json.dumps(rec)),
        )
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
        con = _state_db()
        try:
            con.execute("DELETE FROM state_meta WHERE key = ?", (f"heartbeat:{sid}",))
        finally:
            con.close()
        return j(handler, {"ok": True, "cleared": sid})
    if action in {"pause", "resume"}:
        con = _state_db()
        try:
            # Same lost-update hazard as goals: the heartbeat scheduler updates
            # last_fired_at / fire_count on this row while the phone toggles it.
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT value FROM state_meta WHERE key = ?", (f"heartbeat:{sid}",)).fetchone()
            if not row:
                con.execute("ROLLBACK")
                return bad(handler, "no heartbeat on that session", 404)
            data = json.loads(row[0])
            data["status"] = "paused" if action == "pause" else "active"
            con.execute("UPDATE state_meta SET value = ? WHERE key = ?", (json.dumps(data), f"heartbeat:{sid}"))
            con.execute("COMMIT")
        except BaseException:
            try:
                con.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            con.close()
        return j(handler, {"ok": True, "session_id": sid, "status": data["status"]})
    if not prompt:
        return bad(handler, "prompt required")
    # parse interval
    seconds = 600
    try:
        _ensure_agent_on_path()
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
    con = _state_db()
    try:
        con.execute(
            "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
            (f"heartbeat:{sid}", json.dumps(rec)),
        )
    finally:
        con.close()
    return j(handler, {"ok": True, "heartbeat": {**rec, "session_id": sid}})


# ── MCP call counts (design 6e "41 calls today") ────────────────────────────

# MCP tool calls are persisted on `messages.tool_name` namespaced as
# `mcp__<server>__<tool>`, with the server slug lowercased and its dashes
# turned into underscores (config `unity-mcp` → `unity_mcp`). That is the only
# record of MCP activity anywhere: `get_mcp_status()` reports just
# name/connected/disabled/tools/transport, and nothing in the agent or the
# webui counts invocations. So this reads the message log rather than a
# counter that does not exist.
_MCP_TOOL_RE = re.compile(r"^mcp__([a-z0-9_]+?)__(.+)$", re.IGNORECASE)


def handle_mcp_calls(handler, parsed):
    """GET /api/nyx/mcp/calls?days=1 — per-server MCP invocation counts.

    Returns counts only. A server with no activity in the window is simply
    absent from the map, so a caller can render nothing rather than a zero.
    """
    qs = parse_qs(parsed.query or "")
    try:
        days = int((qs.get("days", ["1"])[0] or "1").strip())
    except ValueError:
        return bad(handler, "days must be an integer")
    if days < 1 or days > 3650:
        return bad(handler, "days must be between 1 and 3650")

    if days == 1:
        # "today" means since local midnight, not the last 24 hours — the
        # label the UI shows says "today".
        since = datetime.datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
    else:
        since = time.time() - days * 86400

    calls: dict[str, int] = {}
    tools: dict[str, dict[str, int]] = {}
    con = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT tool_name FROM messages "
            "WHERE tool_name LIKE 'mcp\\_\\_%' ESCAPE '\\' AND timestamp >= ?",
            (since,),
        )
        for (tool_name,) in rows:
            m = _MCP_TOOL_RE.match(tool_name or "")
            if not m:
                continue
            server, tool = m.group(1).lower(), m.group(2)
            calls[server] = calls.get(server, 0) + 1
            tools.setdefault(server, {})
            tools[server][tool] = tools[server].get(tool, 0) + 1
    except sqlite3.Error as e:
        return bad(handler, f"could not read the message log: {e}", 500)
    finally:
        con.close()

    return j(handler, {"days": days, "since": since, "calls": calls, "tools": tools})
