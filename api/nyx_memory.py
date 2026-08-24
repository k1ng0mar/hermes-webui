"""Nyx memory backends — native (MEMORY.md) + GalaxyMem if installed.

GET /api/nyx/memory/backends
GET /api/nyx/memory?backend=native|galaxymem
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from api.helpers import j

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


def _try_inprocess() -> list | None:
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
            return json.loads(proc.stdout)
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
