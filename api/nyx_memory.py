"""Nyx memory backends — native (MEMORY.md) + GalaxyMem if installed.

GET  /api/nyx/memory/backends
GET  /api/nyx/memory?backend=native|galaxymem
POST /api/nyx/memory/forget  {id}   -> archive one GalaxyMem memory
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from api.helpers import bad, j

_PLUGIN = Path.home() / ".hermes" / "plugins" / "galaxymem"
_DB = Path.home() / ".galaxymem" / "db"

# WebUI's hermes-agent venv does not ship pandas, which galaxymem.store needs.
# System python (or the plugin's own env) can open the lance store. We try
# in-process first, then fall back to a one-shot subprocess.
_READER = r"""
import json, sys
from pathlib import Path
plug = Path.home() / ".hermes" / "plugins" / "galaxymem"
if plug.exists():
    sys.path.insert(0, str(plug))
from galaxymem.store import Store
db = Path.home() / ".galaxymem" / "db"
s = Store(db)
s.open()
try:
    mems = s.list_memories(limit=200) or []
    ents = {e.id: getattr(e, "label", e.id) for e in (s.list_entities() or [])}
finally:
    s.close()
out = []
for m in mems:
    net = getattr(getattr(m, "network", None), "value", None) or str(getattr(m, "network", "") or "world")
    status = getattr(getattr(m, "status", None), "value", None) or str(getattr(m, "status", "") or "")
    text = (getattr(m, "text", None) or "").strip()
    label = text.split("\n", 1)[0][:80] or str(m.id)
    created = getattr(m, "created_at", None)
    created_s = created.isoformat() if hasattr(created, "isoformat") else created
    eids = getattr(m, "entity_ids", None) or []
    cat = ents.get(eids[0], net) if eids else net
    rc = getattr(m, "recall_count", 0) or 0
    lra = getattr(m, "last_recalled_at", None)
    lra_s = lra.isoformat() if hasattr(lra, "isoformat") else (lra or None)
    src = getattr(m, "source_session_id", None)
    out.append({
        "id": str(m.id),
        "label": label,
        "text": text,
        "category": cat,
        "network": net,
        "status": status,
        "score": 0.8 if status == "active" else 0.35,
        "created_at": created_s,
        "recall_count": int(rc),
        "last_recalled_at": lra_s,
        "used": bool(rc and rc > 0),
        "source_session_id": str(src) if src else None,
    })
print(json.dumps(out))
"""


_FORGET = r"""
import json, sys
from pathlib import Path
plug = Path.home() / ".hermes" / "plugins" / "galaxymem"
if plug.exists():
    sys.path.insert(0, str(plug))
from galaxymem.store import Store
from galaxymem.models import MemoryStatus
mem_id = sys.argv[1]
db = Path.home() / ".galaxymem" / "db"
s = Store(db)
s.open()
try:
    before = s.get_memory(mem_id)
    if before is None:
        print(json.dumps({"ok": False, "error": "not found"}))
        raise SystemExit(0)
    s.update_memory_status(mem_id, MemoryStatus.archived)
    after = s.get_memory(mem_id)
    status = getattr(getattr(after, "status", None), "value", None) or ""
finally:
    s.close()
print(json.dumps({"ok": True, "id": mem_id, "status": status}))
"""


def _galaxymem_present() -> bool:
    return (_DB / "memories.lance").exists()


def handle_backends(handler):
    backends = [{"id": "native", "label": "Native"}]
    if _galaxymem_present():
        backends.append({"id": "galaxymem", "label": "GalaxyMem"})
    return j(handler, {"backends": backends, "default": "native"})


def handle_read(handler, parsed):
    from urllib.parse import parse_qs

    qs = parse_qs(parsed.query or "")
    backend = (qs.get("backend", ["native"])[0] or "native").strip().lower()
    if backend in ("galaxymem", "galaxy"):
        return j(handler, _galaxymem_payload())
    return None


def _mems_to_payload(mems: list) -> dict:
    clusters: dict[str, dict] = {}
    for n in mems:
        key = n.get("category") or "uncategorized"
        cl = clusters.setdefault(key, {"id": key, "label": key, "nodes": []})
        cl["nodes"].append(n)
    return {"nodes": mems, "clusters": list(clusters.values()), "backend": "galaxymem"}


# Importing the plugin into the WebUI process pulls third-party code (lance,
# pyarrow, pandas) into the server: a segfault or a leak in that stack takes the
# backend down with it, and the sys.path insert is process-global and permanent.
# The one-shot subprocess is isolated and only ~200ms slower on a cold read, so
# it is the default; set NYX_GALAXYMEM_INPROCESS=1 to opt back in.
_INPROCESS_OK = os.environ.get("NYX_GALAXYMEM_INPROCESS", "").strip().lower() in ("1", "true", "yes")


def _try_inprocess() -> list | None:
    if not _INPROCESS_OK:
        return None
    try:
        if str(_PLUGIN) not in sys.path and _PLUGIN.exists():
            sys.path.insert(0, str(_PLUGIN))
        from galaxymem.store import Store

        store = Store(_DB)
        store.open()
        try:
            mems = store.list_memories(limit=200) or []
            ents = {e.id: getattr(e, "label", e.id) for e in (store.list_entities() or [])}
        finally:
            store.close()
    except Exception:
        return None
    out = []
    for m in mems:
        net = getattr(getattr(m, "network", None), "value", None) or str(getattr(m, "network", "") or "world")
        status = getattr(getattr(m, "status", None), "value", None) or str(getattr(m, "status", "") or "")
        text = (getattr(m, "text", None) or "").strip()
        label = text.split("\n", 1)[0][:80] or str(m.id)
        created = getattr(m, "created_at", None)
        created_s = created.isoformat() if hasattr(created, "isoformat") else created
        eids = getattr(m, "entity_ids", None) or []
        cat = ents.get(eids[0], net) if eids else net
        out.append(
            {
                "id": str(m.id),
                "label": label,
                "text": text,
                "category": cat,
                "network": net,
                "status": status,
                "score": 0.8 if status == "active" else 0.35,
                "created_at": created_s,
            }
        )
    return out


def _try_subprocess() -> list:
    pythons = ["/usr/bin/python3", sys.executable]
    last_err = "no python"
    for py in pythons:
        if not Path(py).exists():
            continue
        try:
            proc = subprocess.run(
                [py, "-c", _READER],
                capture_output=True,
                text=True,
                timeout=25,
                check=False,
            )
        except Exception as e:
            last_err = str(e)
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError as e:
                last_err = f"reader json: {e}"
                continue
            if not isinstance(data, list):
                last_err = "reader did not return an array"
                continue
            return data
        last_err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[:400]
    raise RuntimeError(last_err)


def _galaxymem_payload() -> dict:
    if not _galaxymem_present():
        return {"nodes": [], "clusters": [], "backend": "galaxymem", "error": "GalaxyMem db missing"}
    try:
        mems = _try_inprocess()
        if mems is None:
            mems = _try_subprocess()
        return _mems_to_payload(mems)
    except Exception as e:
        return {"nodes": [], "clusters": [], "backend": "galaxymem", "error": str(e)}


def _run_forget(mem_id: str) -> dict:
    """Archive one memory through the plugin's own store.

    Runs out-of-process for the same reason reads do: the WebUI venv has no
    pandas, which galaxymem.store needs to open the lance store.
    """
    last_err = "no python"
    for py in ("/usr/bin/python3", sys.executable):
        if not Path(py).exists():
            continue
        try:
            proc = subprocess.run(
                [py, "-c", _FORGET, mem_id],
                capture_output=True,
                text=True,
                timeout=25,
                check=False,
            )
        except Exception as e:
            last_err = str(e)
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError as e:
                last_err = f"writer json: {e}"
                continue
        last_err = (proc.stderr or "").strip().splitlines()[-1:] or [f"exit {proc.returncode}"]
        last_err = last_err[0]
    return {"ok": False, "error": last_err}


def handle_forget(handler, body):
    """POST /api/nyx/memory/forget {id}

    Archives the memory — it is never hard-deleted. GalaxyMem's model marks
    `archived` as "explicit user intent only (D13: never hard-deleted)", so the
    row stays queryable and the client renders it struck through rather than
    making it vanish.
    """
    if not isinstance(body, dict):
        return bad(handler, "body must be an object")
    mem_id = str(body.get("id") or "").strip()
    if not mem_id:
        return bad(handler, "id is required")
    if len(mem_id) > 200:
        return bad(handler, "id is too long")
    if not _galaxymem_present():
        return bad(handler, "GalaxyMem is not installed", 404)
    result = _run_forget(mem_id)
    if not result.get("ok"):
        err = str(result.get("error") or "forget failed")
        return bad(handler, err, 404 if "not found" in err.lower() else 500)
    return j(handler, result)
