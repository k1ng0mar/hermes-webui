"""Nyx analytics: aggregate session_model_usage from state.db into phone-sized cards.

GET /api/nyx/analytics?days=7|30|90
"""
from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from urllib.parse import parse_qs

from api.helpers import bad, j

STATE_DB = None  # resolved lazily like nyx_ops


def _db():
    global STATE_DB
    if STATE_DB is None:
        from pathlib import Path

        STATE_DB = Path.home() / ".hermes" / "state.db"
    return str(STATE_DB)


def handle_analytics(handler, parsed):
    qs = parse_qs(parsed.query or "")
    try:
        days = int((qs.get("days", ["30"])[0] or "30"))
    except ValueError:
        days = 30
    days = max(1, min(days, 365))
    since = time.time() - days * 86400

    con = sqlite3.connect(_db())
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT model, billing_provider,
                   SUM(api_call_count) AS calls,
                   SUM(input_tokens)   AS input_toks,
                   SUM(output_tokens)  AS output_toks,
                   SUM(cache_read_tokens + cache_write_tokens + reasoning_tokens) AS extra_toks,
                   MAX(last_seen)      AS last_seen,
                   strftime('%Y-%m-%d', last_seen, 'unixepoch') AS day
            FROM session_model_usage
            WHERE last_seen >= ?
            GROUP BY day, model, billing_provider
            """,
            (since,),
        ).fetchall()
        cost_rows = con.execute(
            """
            SELECT model, billing_provider, SUM(COALESCE(estimated_cost_usd,0)) AS cost
            FROM session_model_usage
            WHERE last_seen >= ? AND cost_status IN ('estimated','actual','provider_models_api')
              AND estimated_cost_usd < 1000
              -- per-model sanity: implied $/M-input must be under $50 or the row is a pricing bug
              AND estimated_cost_usd < 50 * (input_tokens + cache_read_tokens) / 1e6 + 1.0
            GROUP BY model, billing_provider
            """,
            (since,),
        ).fetchall()
    finally:
        con.close()

    by_model: dict[str, dict] = {}
    by_day: dict[str, float] = defaultdict(float)
    toks_by_day: dict[str, int] = defaultdict(int)
    total_cost = 0.0
    total_in = 0
    total_out = 0
    total_extra = 0
    calls = 0

    for r in rows:
        m = f"{r['model']}"
        prov = r["billing_provider"] or ""
        slot = by_model.setdefault(
            m,
            {"model": m, "provider": prov, "calls": 0, "tokens": 0, "cost": 0.0},
        )
        t = int(r["input_toks"] or 0) + int(r["output_toks"] or 0) + int(r["extra_toks"] or 0)
        slot["calls"] += int(r["calls"] or 0)
        slot["tokens"] += t
        total_in += int(r["input_toks"] or 0)
        total_out += int(r["output_toks"] or 0)
        total_extra += int(r["extra_toks"] or 0)
        calls += int(r["calls"] or 0)
        day = r["day"]
        if day:
            toks_by_day[day] += t

    # cost comes only from rows with a trusted cost_status, capped at sane values
    for r in cost_rows or []:
        key = f"{r['model']}"
        c = min(float(r["cost"] or 0.0), 1000.0)
        slot = by_model.setdefault(
            key,
            {"model": key, "provider": r["billing_provider"] or "", "calls": 0, "tokens": 0, "cost": 0.0},
        )
        slot["cost"] += c
        total_cost += c

    models = sorted(by_model.values(), key=lambda x: (-x["cost"], -x["tokens"]))
    for m in models[:20]:
        m["cost"] = round(m["cost"], 2)
        m["share"] = round(100 * m["cost"] / total_cost, 1) if total_cost > 0 else None

    # prior window for delta
    con = sqlite3.connect(_db())
    try:
        row = con.execute(
            """
            SELECT SUM(COALESCE(estimated_cost_usd,0)) FROM session_model_usage
            WHERE last_seen >= ? AND last_seen < ?
              AND cost_status IN ('estimated','actual','provider_models_api')
              AND estimated_cost_usd < 1000
            """,
            (since - days * 86400, since),
        ).fetchone()
        prev_window_cost = float(row[0] or 0.0) if row else 0.0
    finally:
        con.close()

    delta_pct = None
    if prev_window_cost > 0 and total_cost > 0:
        delta_pct = round(100 * (total_cost - prev_window_cost) / prev_window_cost, 1)

    series = [
        {"day": d, "cost": round(by_day.get(d, 0.0), 4), "tokens": toks_by_day.get(d, 0)}
        for d in sorted(toks_by_day)
    ]

    return j(
        handler,
        {
            "days": days,
            "spend": round(total_cost, 2),
            "delta_pct": delta_pct,
            "tokens": total_in + total_out + total_extra,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "calls": calls,
            "avg_per_day_tokens": int((total_in + total_out + total_extra) / days),
            "models": models,
            "series": series,
            "generated_at": time.time(),
        },
    )
