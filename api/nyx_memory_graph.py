"""Nyx memory graph — a real graph of GalaxyMem, laid out server-side.

GET /api/nyx/memory/graph?k=8&min_sim=0.62&limit=2500&iters=130

Why this exists
---------------
The plain `/api/nyx/memory` payload is a *list* (nodes + category buckets). It
carries no relationships, so the phone had to invent them: one synthetic hub
per category with every member spoked to it. That is a rendering of the
category list, not a graph — every real node ends up with degree 1 and the
topology can never show structure that is actually in the store.

This endpoint returns the real thing:

  * **semantic** edges — *mutual* k-nearest-neighbour over the 384-d embedding
    every memory already carries, above a cosine threshold. Mutual matters:
    plain k-NN hands every node exactly k neighbours, so the graph has no hubs
    and no leaves and embeds as a featureless disc.
  * **stored** edges — the `edges` table GalaxyMem already maintains
    (`temporal`, `shared_entity`, `derived_from`, `supersedes`, `contests`),
    carried through with kind + weight so the client can filter by kind.

Communities come from Louvain over the combined graph (not from "first entity
id", which collapses to one dominant bucket; and not label propagation, which
collapsed ~2/3 of this graph into a single community). Coordinates are
computed here — the phone cannot afford an O(n^2) force layout over thousands
of nodes, and numpy can.

Heavy lifting runs in a subprocess under system python: the WebUI venv ships
numpy + lancedb but not pandas, which `galaxymem.store` needs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from api.helpers import j

_DB = Path.home() / ".galaxymem" / "db"
_PLUGIN = Path.home() / ".hermes" / "plugins" / "galaxymem"

# (params, db_mtime) -> (built_at, payload)
_CACHE: dict[tuple, tuple[float, dict]] = {}
_CACHE_TTL = 300.0

# A full build is ~40s (mutual k-NN over 2358x384, Louvain, then 130 force
# passes). Far too long to hold a phone's request open, so builds run on a
# background thread: the first call kicks one off and answers `building: true`,
# and the client polls until the payload lands. Any previously built payload
# keeps being served in the meantime.
_LOCK = threading.Lock()
_INFLIGHT: set[tuple] = set()
_LAST: dict | None = None

# Ceiling on concurrent builds. Each one is a CPU-bound subprocess (numpy k-NN
# + Louvain + force layout) with a 180s timeout, on a host that is also running
# the agent — two at a time is already generous.
_MAX_INFLIGHT = 2

# Defaults tuned against the live store (2358 memories, 803 stored edges).
# k=8/min_sim=0.62 under mutual k-NN gives ~5.2k semantic edges, degree spread
# 1..14 (median 4), ~106 orphans and ~160 communities — a connected core with a
# real halo. Loosening min_sim past ~0.68 collapses 2/3 of the graph into one
# blob; tightening past ~0.75 shatters it into dust.
_DEFAULTS = {"k": 8, "min_sim": 0.62, "limit": 2500, "iters": 130}
_LIMITS = {"k": (1, 24), "min_sim": (0.0, 0.99), "limit": (1, 20000), "iters": (0, 600)}


_BUILDER = r'''
import json, sys, math
from pathlib import Path

K        = int(sys.argv[1])
MIN_SIM  = float(sys.argv[2])
LIMIT    = int(sys.argv[3])
ITERS    = int(sys.argv[4])

import numpy as np
import lancedb

DB = Path.home() / ".galaxymem" / "db"
db = lancedb.connect(str(DB))

def table_names(conn):
    # lancedb renamed this: <=0.34 has table_names(), 0.36 has list_tables()
    # returning a ListTablesResponse whose .tables holds the names.
    for attr in ("list_tables", "table_names"):
        fn = getattr(conn, attr, None)
        if fn is None:
            continue
        try:
            r = fn()
        except Exception:
            continue
        if hasattr(r, "tables"):
            return set(r.tables or [])
        if isinstance(r, (list, tuple, set)):
            return set(r)
    return set()

names = table_names(db)

# ── load ──────────────────────────────────────────────────────────────────
mem = db.open_table("memories").to_pandas()
# newest first, then cap: a truncated graph should keep recent memory
if "created_at" in mem.columns:
    mem = mem.sort_values("created_at", ascending=False, kind="mergesort")
total_memories = len(mem)
mem = mem.head(LIMIT).reset_index(drop=True)
n = len(mem)
if n == 0:
    print(json.dumps({"nodes": [], "edges": [], "communities": [],
                      "stats": {"memories_total": 0}}))
    raise SystemExit(0)

ent_label = {}
if "entities" in names:
    e = db.open_table("entities").to_pandas()
    for _, r in e.iterrows():
        ent_label[str(r["id"])] = str(r.get("label") or r["id"])

def parse_json_list(v):
    if v is None: return []
    if isinstance(v, (list, tuple)): return [str(x) for x in v]
    s = str(v).strip()
    if not s or s == "[]": return []
    try:
        out = json.loads(s)
        return [str(x) for x in out] if isinstance(out, list) else [str(out)]
    except Exception:
        return []

ids   = [str(x) for x in mem["id"].tolist()]
index = {mid: i for i, mid in enumerate(ids)}

# ── vectors ───────────────────────────────────────────────────────────────
V = np.vstack([np.asarray(v, dtype=np.float32) for v in mem["vector"].tolist()])
norms = np.linalg.norm(V, axis=1, keepdims=True)
norms[norms == 0] = 1.0
V = V / norms

# ── semantic edges: MUTUAL k-NN (blocked cosine, no full n^2 matrix held) ──
# Mutual, not plain, k-NN. Plain k-NN hands every node exactly k neighbours, so
# the graph is a uniform mesh with no hubs and no leaves — degree heterogeneity
# is impossible by construction, and a uniform mesh always embeds as a filled
# disc. Requiring the relationship to be reciprocated ("j is one of my nearest
# AND i is one of j's") restores real structure: a memory many others consider
# their nearest becomes a hub, a memory nobody reciprocates falls out to the
# halo, and the degree distribution stops being flat.
topk = [None] * n
BLK = 512
for s0 in range(0, n, BLK):
    s1 = min(n, s0 + BLK)
    sims = V[s0:s1] @ V.T                       # (blk, n)
    np.fill_diagonal(sims[:, s0:s1], -2.0)      # never self
    kk = min(K, n - 1)
    idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
    for bi in range(s1 - s0):
        i = s0 + bi
        keep = {}
        for jj in idx[bi]:
            w = float(sims[bi, jj])
            if w >= MIN_SIM and int(jj) != i:
                keep[int(jj)] = w
        topk[i] = keep
    del sims

sem = {}
for i in range(n):
    for jx, w in topk[i].items():
        if i not in topk[jx]:
            continue                            # not reciprocated — drop it
        a, b = (i, jx) if i < jx else (jx, i)
        prev = sem.get((a, b))
        wj = topk[jx][i]
        wmax = w if w > wj else wj
        if prev is None or wmax > prev:
            sem[(a, b)] = wmax

edges = [{"a": a, "b": b, "kind": "semantic", "w": round(w, 4)}
         for (a, b), w in sem.items()]

# ── stored edges ──────────────────────────────────────────────────────────
stored_kinds = {}
dropped_stored = 0
if "edges" in names:
    et = db.open_table("edges").to_pandas()
    seen = set()
    for _, r in et.iterrows():
        a = index.get(str(r["from_id"]))
        b = index.get(str(r["to_id"]))
        if a is None or b is None:
            dropped_stored += 1
            continue
        if a == b:
            continue
        kind = str(r.get("kind") or "related")
        lo, hi = (a, b) if a < b else (b, a)
        key = (lo, hi, kind)
        if key in seen:
            continue
        seen.add(key)
        try:
            w = float(r.get("weight") or 1.0)
        except Exception:
            w = 1.0
        edges.append({"a": lo, "b": hi, "kind": kind, "w": round(w, 4)})
        stored_kinds[kind] = stored_kinds.get(kind, 0) + 1

# ── adjacency (undirected, union of all kinds) ─────────────────────────────
adj = [[] for _ in range(n)]
for e in edges:
    adj[e["a"]].append(e["b"])
    adj[e["b"]].append(e["a"])
degree = np.array([len(a) for a in adj], dtype=np.int32)

# ── communities ───────────────────────────────────────────────────────────
# Louvain (modularity) where networkx is available: plain label propagation
# collapses ~2/3 of this graph into one community, which is useless for hue.
# Edge weight folds in kind, so a temporal chain does not bind as tightly as
# a semantic neighbour.
KIND_W = {"semantic": 1.0, "shared_entity": 1.1, "derived_from": 0.9,
          "supersedes": 0.9, "contests": 0.5, "temporal": 0.25}

labels = None
try:
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for e in edges:
        w = KIND_W.get(e["kind"], 0.6) * max(0.05, float(e["w"]))
        if G.has_edge(e["a"], e["b"]):
            G[e["a"]][e["b"]]["weight"] += w
        else:
            G.add_edge(e["a"], e["b"], weight=w)
    comms = nx.community.louvain_communities(G, weight="weight", seed=9731)
    labels = np.zeros(n, dtype=np.int64)
    for ci, members in enumerate(comms):
        for i in members:
            labels[i] = ci
except Exception:
    labels = None

if labels is None:
    # fallback: seeded label propagation, smallest-label tie-break
    labels = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(9731)
    order = np.arange(n)
    for _ in range(18):
        rng.shuffle(order)
        moved = 0
        for i in order:
            nb = adj[i]
            if not nb:
                continue
            counts = {}
            for jx in nb:
                lj = labels[jx]
                counts[lj] = counts.get(lj, 0) + 1
            best = max(sorted(counts.items()), key=lambda kv: (kv[1], -kv[0]))[0]
            if best != labels[i]:
                labels[i] = best
                moved += 1
        if moved == 0:
            break

# relabel by size, largest = 0 (stable palette assignment)
uniq, counts = np.unique(labels, return_counts=True)
singles = uniq[counts == 1]
ranked = [u for u, _ in sorted(zip(uniq, counts), key=lambda t: (-t[1], t[0]))]
remap = {u: i for i, u in enumerate(ranked)}
community = np.array([remap[l] for l in labels], dtype=np.int32)

# name each community by its most common category signal
cat_of = []
for i in range(n):
    eids = [x for x in parse_json_list(mem["entity_ids"].iloc[i]) if x]
    net = str(mem["network"].iloc[i] or "world")
    cat_of.append(ent_label.get(eids[0], net) if eids else net)

# Naming by most-common category is useless here: nearly every memory carries
# the "self" entity, so every community came out called "Umar". Score terms by
# how concentrated they are in a community versus the whole store instead.
_STOP = set("""a an and are as at be been but by for from had has have he her him his
i if in into is it its me my no not of on or our out she so than that the their them
then there these they this to too us was we were what when which who will with you your
just really very much more most some any all can cant dont im ive youre thats got get
about after again also always because before being between both did does doing done
down during each few further here how once only other over own same should such
then through under until up while why would about
user users assistant assistants uses use used using ask asks asked say said says
tell told want wants wanted need needs needed like likes liked know knows knew
think thinks thought make makes made work works working thing things time way
one two three new old good bad help helps prefers prefer wants doesnt didnt
message messages reply replies asking saying telling wanting needing""".split())

def _terms(txt):
    out = []
    word = []
    for ch in txt.lower():
        if ch.isalnum():
            word.append(ch)
        else:
            if len(word) > 2:
                out.append("".join(word))
            word = []
    if len(word) > 2:
        out.append("".join(word))
    return [w for w in out if w not in _STOP and not w.isdigit()]

n_comms = len(ranked)
node_terms = [set(_terms(str(mem["text"].iloc[i] or "")[:400])) for i in range(n)]
df = {}
for c in range(n_comms):
    seen_terms = set()
    for i in np.nonzero(community == c)[0]:
        seen_terms |= node_terms[i]
    for t in seen_terms:
        df[t] = df.get(t, 0) + 1

comm_names = {}
for c in range(n_comms):
    members = np.nonzero(community == c)[0]
    size = max(1, len(members))
    tally = {}
    for i in members:
        for t in node_terms[i]:
            tally[t] = tally.get(t, 0) + 1
    best, best_score = None, 0.0
    for t, cnt in tally.items():
        if cnt < 2 and size > 2:
            continue
        score = (cnt / size) * math.log(1.0 + n_comms / max(1, df.get(t, 1)))
        if score > best_score or (score == best_score and best and t < best):
            best, best_score = t, score
    if best is None:
        tally2 = {}
        for i in members:
            tally2[cat_of[i]] = tally2.get(cat_of[i], 0) + 1
        best = max(sorted(tally2.items()), key=lambda kv: (kv[1], kv[0]))[0] if tally2 else "misc"
    comm_names[c] = best

# ── layout ────────────────────────────────────────────────────────────────
# PCA of the embeddings is a semantically meaningful starting position, so the
# force pass only has to resolve overlap instead of discovering global shape.
core = np.nonzero(degree > 0)[0]
orphans = np.nonzero(degree == 0)[0]

pos = np.zeros((n, 2), dtype=np.float32)
if len(core) >= 3:
    Vc = V[core]
    Vc = Vc - Vc.mean(axis=0, keepdims=True)
    # top-2 right singular vectors
    _, _, Wt = np.linalg.svd(Vc, full_matrices=False)
    P = Vc @ Wt[:2].T
    scale = np.abs(P).max() or 1.0
    pos[core] = (P / scale) * 380.0
elif len(core):
    for t, i in enumerate(core):
        a = t * 2.39996
        pos[i] = (math.cos(a) * 40, math.sin(a) * 40)

# force refinement over the connected core only.
#
# This is d3-force's model rather than a naive spring/charge sum, because the
# naive version crystallises: uniform 1/d^2 repulsion across ~2300 nodes pushes
# everything into an evenly spaced disc with no dense core and no voids. Two
# things fix that:
#   * repulsion is *local* (cut off past CUTOFF) so distant clusters stop
#     shoving each other apart and are free to contract,
#   * link strength is normalised by degree (1/min(deg_a, deg_b)) so a hub is
#     not dragged apart by each of its many links — hubs sit in tight knots and
#     leaf nodes dangle outward, which is what reads as filament structure.
if len(core) > 1 and ITERS > 0:
    sub = {int(g): t for t, g in enumerate(core)}
    m = len(core)
    P = pos[core].copy()
    ea, eb, ew = [], [], []
    for e in edges:
        a = sub.get(e["a"]); b = sub.get(e["b"])
        if a is None or b is None:
            continue
        ea.append(a); eb.append(b)
        ew.append(KIND_W.get(e["kind"], 0.6) * max(0.05, float(e["w"])))
    ea = np.asarray(ea, dtype=np.int32)
    eb = np.asarray(eb, dtype=np.int32)
    ew = np.asarray(ew, dtype=np.float32)

    # Community-aware rest length is what turns one even disc into lobes: a
    # link inside a community pulls its members into a tight knot, while a link
    # that bridges two communities is allowed to stay long, so the knots push
    # apart and leave voids between them.
    scomm = community[core]
    same = (scomm[ea] == scomm[eb]) if len(ea) else np.zeros(0, dtype=bool)

    sdeg = degree[core].astype(np.float32)
    np.maximum(sdeg, 1.0, out=sdeg)
    if len(ea):
        # d3: strength = 1/min(deg); bias splits the correction between ends by
        # relative degree, so the lighter node moves further.
        pair_min = np.minimum(sdeg[ea], sdeg[eb])
        strength = ew / pair_min
        tot = sdeg[ea] + sdeg[eb]
        bias_a = (sdeg[eb] / tot).astype(np.float32)
        bias_b = (sdeg[ea] / tot).astype(np.float32)

    CHARGE = 340.0        # per-node repulsion
    CUTOFF = 190.0        # beyond this, nodes ignore each other
    CUT2 = CUTOFF * CUTOFF
    LEN_INTRA = 18.0      # inside a community: contract into a knot
    LEN_INTER = 110.0     # across communities: a long bridge, not a clamp
    GRAVITY = 0.020
    if len(ea):
        link_len = np.where(same, LEN_INTRA, LEN_INTER).astype(np.float32)
        # a bridge should also pull more weakly than an internal link
        strength = strength * np.where(same, 1.0, 0.35).astype(np.float32)
    BLKN = 768          # bigger blocks = fewer python-level loop trips
    for it in range(ITERS):
        alpha = 1.0 - it / max(1, ITERS)
        alpha = 0.10 + 0.90 * alpha * alpha   # ease out, so late passes settle

        # ── local repulsion ────────────────────────────────────────────────
        F = np.zeros((m, 2), dtype=np.float32)
        for b0 in range(0, m, BLKN):
            b1 = min(m, b0 + BLKN)
            d = P[b0:b1, None, :] - P[None, :, :]
            d2 = (d * d).sum(axis=2)
            np.maximum(d2, 4.0, out=d2)
            inv = CHARGE / d2
            inv[d2 > CUT2] = 0.0
            np.fill_diagonal(inv[:, b0:b1], 0.0)
            F[b0:b1] += (d * inv[:, :, None]).sum(axis=1)
        P += F * alpha

        # ── links (degree-normalised, applied as position corrections) ─────
        if len(ea):
            d = P[eb] - P[ea]
            dist = np.sqrt((d * d).sum(axis=1))
            np.maximum(dist, 1e-3, out=dist)
            corr = ((dist - link_len) / dist * alpha * strength)[:, None] * d
            np.add.at(P, ea, corr * bias_a[:, None])
            np.add.at(P, eb, -corr * bias_b[:, None])

        # ── gravity: holds detached fragments in frame ─────────────────────
        P -= P * (GRAVITY * alpha)

    pos[core] = P

# orphans ring the core — the halo of unconnected dots in the reference shots
if len(orphans):
    if len(core):
        R = float(np.abs(pos[core]).max()) * 1.18 + 60.0
    else:
        R = 300.0
    for t, i in enumerate(orphans):
        a = (t / len(orphans)) * math.tau
        rr = R * (1.0 + 0.06 * ((t * 7 % 5) / 5.0))
        pos[i] = (math.cos(a) * rr, math.sin(a) * rr)

# normalise to 0..1
mn = pos.min(axis=0)
mx = pos.max(axis=0)
span = np.maximum(mx - mn, 1e-6)
span = np.array([span.max(), span.max()], dtype=np.float32)  # keep aspect square
ctr = (mn + mx) / 2.0
norm = (pos - ctr) / span + 0.5

# ── payload ───────────────────────────────────────────────────────────────
def s(v, cap=None):
    if v is None: return None
    out = str(v)
    return out[:cap] if cap else out

nodes = []
for i in range(n):
    text = (str(mem["text"].iloc[i] or "")).strip()
    label = text.split("\n", 1)[0][:80] or ids[i]
    rc = mem["recall_count"].iloc[i]
    try: rc = int(rc)
    except Exception: rc = 0
    nodes.append({
        "id": ids[i],
        "label": label,
        "preview": text[:120],
        "category": cat_of[i],
        "network": s(mem["network"].iloc[i]) or "world",
        "status": s(mem["status"].iloc[i]) or "",
        "created_at": s(mem["created_at"].iloc[i]),
        "recall_count": rc,
        "degree": int(degree[i]),
        "community": int(community[i]),
        "x": round(float(norm[i, 0]), 5),
        "y": round(float(norm[i, 1]), 5),
    })

communities = []
for c in range(len(ranked)):
    size = int((community == c).sum())
    communities.append({"id": c, "label": comm_names[c], "size": size})

kind_counts = {}
for e in edges:
    kind_counts[e["kind"]] = kind_counts.get(e["kind"], 0) + 1

print(json.dumps({
    "nodes": nodes,
    "edges": edges,
    "communities": communities,
    "stats": {
        "memories_total": int(total_memories),
        "nodes": n,
        "edges": len(edges),
        "edge_kinds": kind_counts,
        "stored_edges_dropped": int(dropped_stored),
        "orphans": int(len(orphans)),
        "linked": int(len(core)),
        "communities": len(ranked),
        "singleton_communities": int(len(singles)),
        "params": {"k": K, "min_sim": MIN_SIM, "limit": LIMIT, "iters": ITERS},
    },
}))
'''


def _present() -> bool:
    return _DB.exists() and (_DB / "memories.lance").exists()


def _db_stamp() -> float:
    """Newest mtime across the lance tables — cheap change detector."""
    newest = 0.0
    try:
        for p in _DB.glob("*.lance"):
            newest = max(newest, p.stat().st_mtime)
    except Exception:
        pass
    return newest


def _clamp(name: str, raw, fallback):
    lo, hi = _LIMITS[name]
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return fallback
    if v != v:  # NaN
        return fallback
    v = max(lo, min(hi, v))
    return v if name == "min_sim" else int(v)


def _params(parsed) -> dict:
    from urllib.parse import parse_qs

    q = parse_qs(parsed.query or "")
    out = {}
    for name, default in _DEFAULTS.items():
        vals = q.get(name)
        out[name] = _clamp(name, vals[0], default) if vals else default
    return out


def _build(p: dict) -> dict:
    env = dict(os.environ)
    if _PLUGIN.exists():
        env["PYTHONPATH"] = str(_PLUGIN) + os.pathsep + env.get("PYTHONPATH", "")
    last = "no interpreter"
    for py in ("/usr/bin/python3", sys.executable):
        if not Path(py).exists():
            continue
        try:
            proc = subprocess.run(
                [py, "-c", _BUILDER, str(p["k"]), str(p["min_sim"]),
                 str(p["limit"]), str(p["iters"])],
                capture_output=True, text=True, timeout=180, check=False, env=env,
            )
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                return json.loads(proc.stdout)
            except Exception as exc:
                last = f"bad JSON from builder: {exc}"
                continue
        last = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[:600]
    raise RuntimeError(last)


def _build_async(key: tuple, p: dict) -> None:
    global _LAST
    try:
        payload = _build(p)
        payload["backend"] = "galaxymem"
    except Exception as exc:
        payload = {
            "nodes": [], "edges": [], "communities": [],
            "backend": "galaxymem", "error": str(exc),
        }
    with _LOCK:
        _CACHE.clear()      # only the newest parameterisation is worth holding
        _CACHE[key] = (time.time(), payload)
        _LAST = payload
        _INFLIGHT.discard(key)


def handle_graph(handler, parsed):
    """GET /api/nyx/memory/graph — real memory graph with server-side layout.

    Answers immediately. When nothing is cached for these parameters the reply
    carries `building: true` and the client should poll.
    """
    if not _present():
        return j(handler, {
            "nodes": [], "edges": [], "communities": [],
            "backend": "galaxymem", "error": "GalaxyMem db missing",
        })

    p = _params(parsed)
    key = (p["k"], p["min_sim"], p["limit"], p["iters"], _db_stamp())
    now = time.time()

    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < _CACHE_TTL:
            payload = dict(hit[1])
            payload["cached"] = True
            payload["building"] = False
            return j(handler, payload)

        # _INFLIGHT dedupes per parameter set, but every DISTINCT set used to
        # get its own thread and its own ~40s (up to 180s) subprocess, with
        # nothing bounding how many ran at once — a client sweeping k/min_sim,
        # or just a poll loop that varies params, could pin the box. Cap the
        # concurrent builds; refused callers still get the stale graph and the
        # next poll starts a build once a slot frees.
        building = key in _INFLIGHT
        if not building and len(_INFLIGHT) < _MAX_INFLIGHT:
            _INFLIGHT.add(key)
            threading.Thread(
                target=_build_async, args=(key, p), daemon=True,
                name="nyx-memory-graph",
            ).start()
        stale = _LAST

    # Serve the previous graph while the new one builds, so the screen has
    # something to draw instead of flashing empty.
    if stale is not None:
        payload = dict(stale)
        payload["cached"] = True
        payload["building"] = True
        payload["stale"] = True
        return j(handler, payload)

    return j(handler, {
        "nodes": [], "edges": [], "communities": [],
        "backend": "galaxymem", "building": True, "cached": False,
        "stats": {"params": p},
    })
