"""Nyx plugin management — enable, disable, install, search.

    POST /api/nyx/plugins/enable   {name}                 -> enable a plugin
    POST /api/nyx/plugins/disable  {name}                 -> disable a plugin
    POST /api/nyx/plugins/install  {identifier, ref?}     -> install from Git
    GET  /api/nyx/plugins/search?q=...                    -> community index

`GET /api/plugins` stays the read path and stays read-only; it reports the
live plugin manager's view. These are the write operations, and they go
through the Hermes CLI (`hermes plugins ...`) rather than editing config.yaml
directly. That matters: `plugins.enabled` and `plugins.disabled` interact with
plugins that are on by default, so hand-editing one list gets the semantics
wrong — 52 of 59 plugins on this host report enabled while only 5 appear in
`plugins.enabled`.

**Nothing here reloads the running agent.** PluginManager sets
`LoadedPlugin.enabled` at load time and exposes no enable/disable, so a change
takes effect when the gateway restarts. Every mutating response carries
`restart_required: true` and the caller is expected to say so rather than
implying the change is already live.

Two CLI details that will hang a subprocess if missed, both verified against
`--help`:

  * `plugins install` prompts unless `--enable` or `--no-enable` is passed.
    We always pass `--no-enable`: installing and enabling are separate
    decisions, and a fresh install should not start executing in the agent
    because someone tapped once.
  * `plugins enable` prompts unless `--allow-tool-override` or
    `--no-allow-tool-override` is passed. We always pass
    `--no-allow-tool-override`. Granting a plugin permission to replace
    built-in tools like shell_exec or write_file is not something a phone tap
    should be able to do silently.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs

from api.helpers import bad, j

logger = logging.getLogger(__name__)

# Installing pulls and runs third-party code inside the agent, so it gets a
# longer budget than a local enable/disable, but still a bounded one.
_ENABLE_TIMEOUT_S = 90
_INSTALL_TIMEOUT_S = 300
_SEARCH_TIMEOUT_S = 60

# A plugin name as the CLI accepts it. Deliberately strict: this value becomes
# a subprocess argument.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")
# A 40-character git SHA, for --ref.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# An https git URL. Only https is accepted: git@ needs a key we do not manage,
# and schemes like file:// or ext:: would turn "install a plugin" into "read an
# arbitrary path" or "run an arbitrary helper".
_URL_RE = re.compile(r"^https://[A-Za-z0-9.-]+(?::\d+)?/[A-Za-z0-9._/-]{1,180}$")


def _cli_prefix() -> list[str]:
    """Build the `python hermes_cli/main.py [--profile X]` prefix.

    Mirrors _handle_gateway_lifecycle rather than inventing a second way to
    find the CLI.
    """
    from api import config as api_config
    from api.profiles import get_active_profile_name

    agent_dir = getattr(api_config, "_AGENT_DIR", None)
    if not agent_dir:
        raise FileNotFoundError("Hermes agent checkout not found")
    agent_dir = Path(agent_dir).expanduser().resolve()
    main_py = agent_dir / "hermes_cli" / "main.py"
    if not main_py.exists():
        raise FileNotFoundError("Hermes agent CLI entrypoint not found")

    cmd = [str(getattr(api_config, "PYTHON_EXE", sys.executable)), str(main_py)]
    try:
        profile = str(get_active_profile_name() or "").strip()
    except Exception:
        profile = ""
    if profile and profile != "default":
        cmd.extend(["--profile", profile])
    return cmd


def _run_cli(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    from api import config as api_config

    agent_dir = Path(api_config._AGENT_DIR).expanduser().resolve()
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    # The CLI may try to open a browser on some paths; keep it inert.
    env.setdefault("BROWSER", "echo")
    return subprocess.run(
        _cli_prefix() + args,
        cwd=str(agent_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _tail(text: str, lines: int = 4) -> str:
    out = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(out[-lines:])


def _validated_name(body, field: str = "name") -> tuple[str | None, str | None]:
    """Return (value, error). Rejects anything that could act as a flag."""
    if not isinstance(body, dict):
        return None, "body must be an object"
    raw = body.get(field)
    if not isinstance(raw, str):
        return None, f"{field} is required"
    value = raw.strip()
    if not value:
        return None, f"{field} is required"
    # A leading '-' would be parsed as an option by the CLI.
    if value.startswith("-"):
        return None, f"{field} must not start with '-'"
    if not _NAME_RE.match(value):
        return None, f"{field} contains characters that are not allowed"
    return value, None


def handle_plugin_enable(handler, body):
    name, err = _validated_name(body)
    if err:
        return bad(handler, err)
    try:
        # --no-allow-tool-override: never let a remote tap grant a plugin the
        # right to replace built-in tools. It also stops the CLI prompting.
        proc = _run_cli(["plugins", "enable", name, "--no-allow-tool-override"], _ENABLE_TIMEOUT_S)
    except FileNotFoundError as exc:
        return bad(handler, str(exc), 404)
    except subprocess.TimeoutExpired:
        return bad(handler, "enabling timed out", 504)
    if proc.returncode != 0:
        return bad(handler, _tail(proc.stderr or proc.stdout) or "enable failed", 500)
    return j(handler, {
        "ok": True,
        "name": name,
        "enabled": True,
        "restart_required": True,
        "output": _tail(proc.stdout),
    })


def handle_plugin_disable(handler, body):
    name, err = _validated_name(body)
    if err:
        return bad(handler, err)
    try:
        proc = _run_cli(["plugins", "disable", name], _ENABLE_TIMEOUT_S)
    except FileNotFoundError as exc:
        return bad(handler, str(exc), 404)
    except subprocess.TimeoutExpired:
        return bad(handler, "disabling timed out", 504)
    if proc.returncode != 0:
        return bad(handler, _tail(proc.stderr or proc.stdout) or "disable failed", 500)
    return j(handler, {
        "ok": True,
        "name": name,
        "enabled": False,
        "restart_required": True,
        "output": _tail(proc.stdout),
    })


def _validated_identifier(body) -> tuple[str | None, str | None]:
    """An install target: an https git URL, owner/repo, or an index name.

    Kept separate from _validated_name because a URL contains characters a
    plugin name must not, and loosening the name pattern to fit URLs would
    weaken every other endpoint that uses it.
    """
    if not isinstance(body, dict):
        return None, "body must be an object"
    raw = body.get("identifier")
    if not isinstance(raw, str) or not raw.strip():
        return None, "identifier is required"
    value = raw.strip()
    if value.startswith("-"):
        return None, "identifier must not start with '-'"
    if value.startswith("https://"):
        if not _URL_RE.match(value):
            return None, "that does not look like an https git URL"
        return value, None
    if "://" in value or value.startswith("git@"):
        return None, "only https git URLs are accepted"
    if not _NAME_RE.match(value):
        return None, "identifier contains characters that are not allowed"
    return value, None


def handle_plugin_install(handler, body):
    """Install from a Git URL, owner/repo shorthand, or community index name.

    Installed DISABLED (`--no-enable`). Pulling third-party code and running it
    inside the agent are two different decisions, and this endpoint only makes
    the first one.
    """
    identifier, err = _validated_identifier(body)
    if err:
        return bad(handler, err)
    args = ["plugins", "install", identifier, "--no-enable"]

    ref = (body or {}).get("ref")
    if ref is not None:
        if not isinstance(ref, str) or not _SHA_RE.match(ref.strip()):
            return bad(handler, "ref must be a full 40-character commit sha")
        args.extend(["--ref", ref.strip()])

    try:
        proc = _run_cli(args, _INSTALL_TIMEOUT_S)
    except FileNotFoundError as exc:
        return bad(handler, str(exc), 404)
    except subprocess.TimeoutExpired:
        return bad(handler, "install timed out", 504)
    if proc.returncode != 0:
        return bad(handler, _tail(proc.stderr or proc.stdout) or "install failed", 500)
    return j(handler, {
        "ok": True,
        "identifier": identifier,
        # Installed but not enabled — the caller must enable it deliberately.
        "enabled": False,
        "restart_required": True,
        "output": _tail(proc.stdout, 8),
    })


def handle_plugin_search(handler, parsed):
    """Search the community plugin index. `q` may be empty to browse."""
    qs = parse_qs(parsed.query or "")
    term = (qs.get("q", [""])[0] or "").strip()
    if term.startswith("-"):
        return bad(handler, "q must not start with '-'")
    if len(term) > 100:
        return bad(handler, "q is too long")
    args = ["plugins", "search", "--json"]
    if term:
        args.append(term)
    try:
        proc = _run_cli(args, _SEARCH_TIMEOUT_S)
    except FileNotFoundError as exc:
        return bad(handler, str(exc), 404)
    except subprocess.TimeoutExpired:
        return bad(handler, "search timed out", 504)
    if proc.returncode != 0:
        return bad(handler, _tail(proc.stderr or proc.stdout) or "search failed", 502)
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return bad(handler, "the plugin index returned unreadable output", 502)
    results = data if isinstance(data, list) else data.get("results") or data.get("plugins") or []
    if not isinstance(results, list):
        results = []
    return j(handler, {"results": results[:60], "term": term})
