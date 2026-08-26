"""Regression tests for the Nyx backend hardening pass.

Each test here pins a defect that was live in the nyx_* modules and had no
coverage at all:

  nyx_store    - atomic writes; a corrupt store is quarantined, not silently
                 replaced with an empty default that the next write commits
  nyx_commands - device-addressed commands are not stolen by another phone;
                 confirm-gated actions are refused without explicit
                 confirmation; params are type/range checked; `volume` no
                 longer demands a `stream` the phone never reads
  nyx_revisions- caller-supplied rev ids cannot escape the revisions root
"""
import json
import threading

import pytest

from api import nyx_commands
from api.nyx_revisions import _validate_rev_id, diff_revisions, revert_to_revision, snapshot_revision
from api.nyx_store import atomic_write_bytes, atomic_write_json, atomic_write_text, load_json


# ── nyx_store ───────────────────────────────────────────────────────────────


def test_atomic_write_text_replaces_contents(tmp_path):
    target = tmp_path / "store.json"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second")
    assert target.read_text() == "second"


def test_atomic_write_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deeper" / "store.json"
    atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}


def test_atomic_write_bytes_roundtrips_binary(tmp_path):
    target = tmp_path / "blob"
    payload = bytes(range(256))
    atomic_write_bytes(target, payload)
    assert target.read_bytes() == payload


def test_atomic_write_bytes_preserves_mode(tmp_path):
    target = tmp_path / "script.sh"
    target.write_bytes(b"old")
    target.chmod(0o750)
    atomic_write_bytes(target, b"new")
    assert target.stat().st_mode & 0o777 == 0o750


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "store.json"
    atomic_write_json(target, {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["store.json"]


def test_load_json_missing_file_returns_default(tmp_path):
    assert load_json(tmp_path / "nope.json", {"commands": []}) == {"commands": []}


def test_load_json_default_is_copied_not_shared(tmp_path):
    default = {"commands": []}
    first = load_json(tmp_path / "a.json", default)
    first["commands"].append("x")
    second = load_json(tmp_path / "b.json", default)
    assert second == {"commands": []}
    assert default == {"commands": []}


def test_load_json_quarantines_corrupt_file(tmp_path):
    """A torn write must not be silently converted into an empty store."""
    target = tmp_path / "store.json"
    target.write_text('{"commands": [{"id": "abc"')  # truncated mid-write
    assert load_json(target, {"commands": []}) == {"commands": []}
    assert not target.exists()
    quarantined = list(tmp_path.glob("store.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text() == '{"commands": [{"id": "abc"'


def test_load_json_quarantines_wrong_toplevel_type(tmp_path):
    """A list where a dict is expected used to raise TypeError in the handler."""
    target = tmp_path / "store.json"
    target.write_text("[1, 2, 3]")
    assert load_json(target, {"commands": []}) == {"commands": []}
    assert list(tmp_path.glob("store.json.corrupt-*"))


def test_load_json_empty_file_is_not_quarantined(tmp_path):
    target = tmp_path / "store.json"
    target.write_text("")
    assert load_json(target, {"commands": []}) == {"commands": []}
    assert target.exists()


# ── nyx_commands ────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(nyx_commands, "_STORE", tmp_path / "nyx-commands.json")
    return tmp_path / "nyx-commands.json"


def test_confirm_gated_action_refused_without_confirmation(store):
    with pytest.raises(ValueError, match="explicit owner confirmation"):
        nyx_commands.enqueue("call", {"number": "+15550001111"})


def test_confirm_gated_action_accepted_with_confirmation(store):
    rec = nyx_commands.enqueue("call", {"number": "+15550001111"}, confirmed=True)
    assert rec["action"] == "call"
    assert rec["confirmed"] is True


def test_unconfirmed_flag_is_not_forgeable_by_omitting_params(store):
    """Neither a truthy param nor an extra key can stand in for `confirmed`."""
    with pytest.raises(ValueError):
        nyx_commands.enqueue("call", {"number": "+15550001111", "confirmed": True})


def test_unknown_action_rejected(store):
    with pytest.raises(ValueError, match="unknown action"):
        nyx_commands.enqueue("wipe_device", {})


@pytest.mark.parametrize("number", ["", "not-a-number", "+1; rm -rf /", "12"])
def test_invalid_phone_numbers_rejected(store, number):
    with pytest.raises(ValueError):
        nyx_commands.enqueue("call", {"number": number}, confirmed=True)


@pytest.mark.parametrize("package", ["", "spotify", "../../etc/passwd", "com spotify"])
def test_invalid_packages_rejected(store, package):
    with pytest.raises(ValueError):
        nyx_commands.enqueue("open_app", {"package": package})


@pytest.mark.parametrize("level", [-0.1, 1.1, 50, "loud", None, True])
def test_out_of_range_level_rejected_not_clamped(store, level):
    with pytest.raises(ValueError):
        nyx_commands.enqueue("brightness", {"level": level})


@pytest.mark.parametrize("level", [0.0, 0.5, 1.0])
def test_valid_levels_accepted(store, level):
    rec = nyx_commands.enqueue("brightness", {"level": level})
    assert rec["params"]["level"] == pytest.approx(level)


def test_volume_no_longer_requires_stream(store):
    """The phone hardcodes STREAM_MUSIC; requiring `stream` rejected every
    volume command sent in the documented shape."""
    rec = nyx_commands.enqueue("volume", {"level": 0.4})
    assert rec["params"] == {"level": pytest.approx(0.4)}


def test_volume_validates_stream_when_supplied(store):
    assert nyx_commands.enqueue("volume", {"level": 0.4, "stream": "alarm"})["params"]["stream"] == "alarm"
    with pytest.raises(ValueError, match="stream must be one of"):
        nyx_commands.enqueue("volume", {"level": 0.4, "stream": "bogus"})


def test_toggle_actions_coerce_boolean_strings(store):
    assert nyx_commands.enqueue("flashlight", {"enabled": "true"})["params"]["enabled"] is True
    assert nyx_commands.enqueue("dnd", {"enabled": "off"})["params"]["enabled"] is False


def test_addressed_command_is_not_stolen_by_another_device(store):
    """The pre-fix _pop_pending ignored `device` entirely and handed the
    oldest pending command to whoever asked first."""
    nyx_commands.enqueue("flashlight", {"enabled": True}, device="pixel-8")
    with nyx_commands._arrived:
        assert nyx_commands._claim_locked("honor-x6") is None
        claimed = nyx_commands._claim_locked("pixel-8")
    assert claimed is not None
    assert claimed["target_device"] == "pixel-8"


def test_unaddressed_command_is_claimable_by_any_device(store):
    nyx_commands.enqueue("flashlight", {"enabled": True})
    with nyx_commands._arrived:
        claimed = nyx_commands._claim_locked("any-phone")
    assert claimed is not None
    assert claimed["device"] == "any-phone"


def test_claim_skips_addressed_command_to_reach_a_claimable_one(store):
    nyx_commands.enqueue("flashlight", {"enabled": True}, device="pixel-8")
    nyx_commands.enqueue("bluetooth", {"enabled": False}, device="honor-x6")
    with nyx_commands._arrived:
        claimed = nyx_commands._claim_locked("honor-x6")
    assert claimed["action"] == "bluetooth"


def test_command_is_claimed_exactly_once(store):
    nyx_commands.enqueue("flashlight", {"enabled": True})
    with nyx_commands._arrived:
        assert nyx_commands._claim_locked("phone") is not None
        assert nyx_commands._claim_locked("phone") is None


def test_concurrent_claims_never_hand_out_the_same_command(store):
    for _ in range(20):
        nyx_commands.enqueue("flashlight", {"enabled": True})
    claimed, barrier = [], threading.Barrier(8)

    def worker(name):
        barrier.wait()
        while True:
            with nyx_commands._arrived:
                rec = nyx_commands._claim_locked(name)
            if rec is None:
                return
            claimed.append(rec["id"])

    threads = [threading.Thread(target=worker, args=(f"phone-{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(claimed) == 20
    assert len(set(claimed)) == 20


def test_invalid_device_id_rejected(store):
    with pytest.raises(ValueError, match="invalid device id"):
        nyx_commands.enqueue("flashlight", {"enabled": True}, device="../../etc")


def test_queue_survives_a_corrupt_store(store):
    nyx_commands.enqueue("flashlight", {"enabled": True})
    store.write_text('{"commands": [{"id')  # simulate a torn write
    rec = nyx_commands.enqueue("bluetooth", {"enabled": True})
    assert rec["action"] == "bluetooth"
    assert list(store.parent.glob("nyx-commands.json.corrupt-*"))


def test_store_is_valid_json_after_every_write(store):
    for i in range(5):
        nyx_commands.enqueue("flashlight", {"enabled": bool(i % 2)})
    assert len(json.loads(store.read_text())["commands"]) == 5


# ── nyx_revisions: rev-id validation ────────────────────────────────────────


@pytest.mark.parametrize("rev", ["r0", "r1", "r42", "r1000"])
def test_valid_rev_ids_accepted(rev):
    assert _validate_rev_id(rev) == rev


@pytest.mark.parametrize("rev", [
    "../../../etc",
    "..",
    "r1/../../..",
    "/etc/passwd",
    "r-1",
    "rABC",
    "",
    "head",
])
def test_traversing_or_malformed_rev_ids_rejected(rev):
    with pytest.raises(ValueError, match="invalid revision id"):
        _validate_rev_id(rev)


def test_revert_rejects_traversing_rev(tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("original")
    with pytest.raises(ValueError, match="invalid revision id"):
        revert_to_revision(tmp_path, "note.txt", "../../../etc")
    assert target.read_text() == "original"


def test_diff_rejects_traversing_rev(tmp_path):
    (tmp_path / "note.txt").write_text("original")
    snapshot_revision(tmp_path, "note.txt")
    with pytest.raises(ValueError, match="invalid revision id"):
        diff_revisions(tmp_path, "note.txt", "r0", "../../..")


def test_concurrent_snapshots_do_not_clobber_each_other(tmp_path):
    """Two snapshots racing on _next_rev_id both used to resolve the same id."""
    target = tmp_path / "note.txt"
    results, barrier = [], threading.Barrier(6)

    def worker(i):
        target_text = f"content-{i}"
        barrier.wait()
        (tmp_path / f"note-{i}.txt").write_text(target_text)
        results.append(snapshot_revision(tmp_path, f"note-{i}.txt"))

    target.write_text("seed")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(results) == 6
    assert all(r["created"] for r in results)


def test_revert_writes_atomically_and_preserves_content(tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("v1")
    snapshot_revision(tmp_path, "note.txt")
    target.write_text("v2-longer-content")
    result = revert_to_revision(tmp_path, "note.txt", "r0")
    assert target.read_text() == "v1"
    assert result["restored_from"] == "r0"
    # the temp file used for the atomic rename must not be left behind
    assert not list(tmp_path.glob(".note.txt.*"))


# ── stub handler ────────────────────────────────────────────────────────────


class _Stub:
    """Records what a handler was told to send, so header/status assertions work."""

    def __init__(self, body: bytes = b"") -> None:
        self.headers = {"Content-Length": str(len(body))}
        self.status = None
        self.sent = {}
        self.rfile = __import__("io").BytesIO(body)
        self.wfile = __import__("io").BytesIO()

    def send_response(self, code):
        self.status = code

    def send_header(self, k, v):
        self.sent[k] = v

    def end_headers(self):
        pass

    def payload(self):
        return json.loads(self.wfile.getvalue().decode() or "{}")


# ── nyx_preview: size cap, escaping, sandboxing ─────────────────────────────


@pytest.fixture
def preview(tmp_path, monkeypatch):
    """handle_preview with a fake session whose workspace is tmp_path."""
    import api.models as models
    from urllib.parse import urlparse

    class _Session:
        workspace = str(tmp_path)

    monkeypatch.setattr(models, "get_session_for_file_ops", lambda sid: _Session(), raising=False)

    def call(rel):
        from api.nyx_preview import handle_preview
        h = _Stub()
        handle_preview(h, urlparse(f"/api/nyx/preview/sess1/{rel}"))
        return h

    return call


def test_preview_serves_a_small_file(preview, tmp_path):
    (tmp_path / "note.txt").write_text("hello")
    h = preview("note.txt")
    assert h.status == 200
    assert h.wfile.getvalue() == b"hello"


def test_preview_refuses_an_oversized_file(preview, tmp_path):
    """read_bytes() on a huge artifact used to pull it entirely into memory."""
    from api.nyx_preview import MAX_PREVIEW_BYTES
    big = tmp_path / "big.bin"
    big.write_bytes(b"\0" * (MAX_PREVIEW_BYTES + 1))
    h = preview("big.bin")
    assert h.status == 413
    assert h.wfile.getvalue() != b"\0" * (MAX_PREVIEW_BYTES + 1)


def test_preview_allows_a_file_exactly_at_the_cap(preview, tmp_path):
    from api.nyx_preview import MAX_PREVIEW_BYTES
    (tmp_path / "edge.bin").write_bytes(b"x" * MAX_PREVIEW_BYTES)
    assert preview("edge.bin").status == 200


def test_preview_sandboxes_html(preview, tmp_path):
    """No allow-same-origin: artifact JS must not reach this origin's cookies."""
    (tmp_path / "page.html").write_text("<html><head></head><body>hi</body></html>")
    h = preview("page.html")
    csp = h.sent.get("Content-Security-Policy", "")
    assert "sandbox" in csp
    assert "allow-same-origin" not in csp
    assert h.sent.get("X-Content-Type-Options") == "nosniff"


def test_preview_does_not_sandbox_non_html(preview, tmp_path):
    (tmp_path / "a.txt").write_text("plain")
    h = preview("a.txt")
    assert "Content-Security-Policy" not in h.sent


def test_preview_escapes_the_injected_base_href(preview, tmp_path):
    """`rel` is interpolated into an HTML attribute; a quote must not escape it."""
    d = tmp_path / 'ev"il'
    d.mkdir()
    (d / "page.html").write_text("<html><head></head><body>x</body></html>")
    h = preview('ev"il/page.html')
    out = h.wfile.getvalue().decode()
    assert "&quot;" in out
    assert '<base href="/api/nyx/preview/sess1/ev"il/' not in out


def test_preview_still_injects_a_base_tag(preview, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "page.html").write_text("<html><head></head><body>x</body></html>")
    out = preview("sub/page.html").wfile.getvalue().decode()
    assert '<base href="/api/nyx/preview/sess1/sub/">' in out


def test_preview_rejects_traversal(preview, tmp_path):
    (tmp_path.parent / "outside.txt").write_text("secret")
    assert preview("../outside.txt").status in (400, 404)


# ── nyx_ops: identifier validation ──────────────────────────────────────────


@pytest.mark.parametrize("ident", [
    "my-skill", "scope/name", "owner/repo", "pkg.name", "a_b+c",
    "https://example.com/skill.zip",
])
def test_valid_identifiers_accepted(ident):
    from api.nyx_ops import _validate_identifier
    assert _validate_identifier(ident) == ident


@pytest.mark.parametrize("ident", [
    "--force", "-x", "--config=/etc/passwd", "", "   ",
    "has space", "semi;colon", "pipe|thing", "$(whoami)", "back`tick`",
])
def test_dangerous_identifiers_rejected(ident):
    """A leading '-' is read by the CLI as a flag, not a package name."""
    from api.nyx_ops import _validate_identifier
    with pytest.raises(ValueError):
        _validate_identifier(ident)


def test_skills_install_rejects_flag_identifier_without_running_anything(monkeypatch):
    import api.nyx_ops as ops
    called = []
    monkeypatch.setattr(ops, "_run", lambda *a, **k: called.append(a) or (0, "", ""))
    h = _Stub()
    ops.handle_skills_install(h, {"identifier": "--yes"})
    assert h.status == 400
    assert called == []


def test_mcp_install_rejects_flag_name_without_running_anything(monkeypatch):
    import api.nyx_ops as ops
    called = []
    monkeypatch.setattr(ops, "_run", lambda *a, **k: called.append(a) or (0, "", ""))
    h = _Stub()
    ops.handle_mcp_install(h, {"name": "--version"})
    assert h.status == 400
    assert called == []


def test_skills_install_runs_for_a_valid_identifier(monkeypatch):
    import api.nyx_ops as ops
    called = []
    monkeypatch.setattr(ops, "_run", lambda *a, **k: (called.append(a[0]), (0, "ok", ""))[1])
    h = _Stub()
    ops.handle_skills_install(h, {"identifier": "some/skill"})
    assert called and "some/skill" in called[0]


# ── nyx_ops: state.db handling ──────────────────────────────────────────────


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    import sqlite3
    import api.nyx_ops as ops
    db = tmp_path / "state.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
    con.commit()
    con.close()
    monkeypatch.setattr(ops, "STATE_DB", db)
    return db


def test_state_db_sets_a_busy_timeout_and_manual_transactions(state_db):
    """Default 5s + implicit transactions turned a contended read into a 500."""
    import api.nyx_ops as ops
    con = ops._state_db()
    try:
        assert con.isolation_level is None
    finally:
        con.close()
    assert ops._STATE_DB_TIMEOUT_S >= 10


@pytest.mark.parametrize("bad_value", ["abc", "1e9999", {}, []])
def test_goal_create_rejects_non_integer_max_turns(state_db, bad_value):
    """int() on request input escaped the handler as an opaque 500."""
    import api.nyx_ops as ops
    h = _Stub()
    ops.handle_goal_create(h, {"session_id": "s1", "goal": "g", "max_turns": bad_value})
    assert h.status == 400


@pytest.mark.parametrize("bad_value", [0, -5, 10**9])
def test_goal_create_rejects_out_of_range_max_turns(state_db, bad_value):
    import api.nyx_ops as ops
    h = _Stub()
    ops.handle_goal_create(h, {"session_id": "s1", "goal": "g", "max_turns": bad_value})
    assert h.status == 400


def test_goal_create_accepts_a_sane_max_turns(state_db):
    import api.nyx_ops as ops
    h = _Stub()
    ops.handle_goal_create(h, {"session_id": "s1", "goal": "ship it", "max_turns": 50})
    assert h.status == 200
    assert h.payload()["goal"]["max_turns"] == 50


def test_goal_action_rejects_unknown_action_before_touching_the_db(state_db):
    import api.nyx_ops as ops
    h = _Stub()
    ops.handle_goal_action(h, {"session_id": "s1", "action": "destroy"})
    assert h.status == 400


def test_goal_pause_then_resume_roundtrips(state_db):
    import api.nyx_ops as ops
    ops.handle_goal_create(_Stub(), {"session_id": "s1", "goal": "g"})
    h = _Stub()
    ops.handle_goal_action(h, {"session_id": "s1", "action": "pause"})
    assert h.payload()["status"] == "paused"
    h = _Stub()
    ops.handle_goal_action(h, {"session_id": "s1", "action": "resume"})
    assert h.payload()["status"] == "active"


def test_goal_action_on_missing_session_is_404_and_leaves_no_open_transaction(state_db):
    import api.nyx_ops as ops
    h = _Stub()
    ops.handle_goal_action(h, {"session_id": "nope", "action": "pause"})
    assert h.status == 404
    # A leaked write lock would make this second call block until timeout.
    ops.handle_goal_create(_Stub(), {"session_id": "s2", "goal": "g"})


# ── nyx_memory_graph: bounded concurrent builds ─────────────────────────────


def test_graph_caps_concurrent_builds(monkeypatch):
    """Distinct params each spawned an unbounded ~40-180s subprocess build."""
    from urllib.parse import urlparse
    import api.nyx_memory_graph as g

    monkeypatch.setattr(g, "_present", lambda: True)
    monkeypatch.setattr(g, "_db_stamp", lambda: 1)
    monkeypatch.setattr(g, "_CACHE", {})
    monkeypatch.setattr(g, "_INFLIGHT", set())
    monkeypatch.setattr(g, "_LAST", None)

    spawned = []

    class _FakeThread:
        def __init__(self, **kw):
            spawned.append(kw.get("args"))

        def start(self):
            pass

    monkeypatch.setattr(g.threading, "Thread", lambda *a, **k: _FakeThread(**k))

    for k in range(1, 9):
        g.handle_graph(_Stub(), urlparse(f"/api/nyx/memory/graph?k={k}"))

    assert len(spawned) == g._MAX_INFLIGHT
    assert g._MAX_INFLIGHT <= 4


# ── nyx_analytics: per-day cost was always zero ─────────────────────────────


@pytest.fixture
def usage_db(tmp_path, monkeypatch):
    """A state.db with session_model_usage rows across two days."""
    import sqlite3
    import time as _time
    import api.nyx_analytics as an

    db = tmp_path / "state.db"
    con = sqlite3.connect(str(db))
    con.execute(
        """CREATE TABLE session_model_usage (
               model TEXT, billing_provider TEXT, api_call_count INT,
               input_tokens INT, output_tokens INT, cache_read_tokens INT,
               cache_write_tokens INT, reasoning_tokens INT,
               last_seen REAL, estimated_cost_usd REAL, cost_status TEXT)"""
    )
    now = _time.time()
    rows = [
        ("opus", "anthropic", 3, 1_000_000, 2000, 0, 0, 0, now - 3600, 2.50, "actual"),
        ("opus", "anthropic", 1, 1_000_000, 500, 0, 0, 0, now - 90000, 1.25, "actual"),
        ("sonnet", "anthropic", 5, 2_000_000, 900, 0, 0, 0, now - 3600, 0.75, "estimated"),
    ]
    con.executemany(
        "INSERT INTO session_model_usage VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    con.commit()
    con.close()
    monkeypatch.setattr(an, "STATE_DB", db)
    return db


def _analytics(days=30):
    from urllib.parse import urlparse
    from api.nyx_analytics import handle_analytics
    h = _Stub()
    handle_analytics(h, urlparse(f"/api/nyx/analytics?days={days}"))
    return h.payload()


def test_analytics_series_carries_real_per_day_cost(usage_db):
    """`by_day` was declared and read but never written, so every series entry
    reported cost 0.0 no matter what had actually been spent."""
    data = _analytics()
    assert data["series"], "expected at least one day in the series"
    assert sum(p["cost"] for p in data["series"]) > 0
    assert data["spend"] == pytest.approx(sum(p["cost"] for p in data["series"]), rel=1e-3)


def test_analytics_series_days_are_sorted_and_unique(usage_db):
    days = [p["day"] for p in _analytics()["series"]]
    assert days == sorted(days)
    assert len(days) == len(set(days))


def test_analytics_totals_are_consistent(usage_db):
    data = _analytics()
    # fixture: input 4,000,000 / output 3,400 / no cache or reasoning tokens
    assert data["input_tokens"] == 4_000_000
    assert data["output_tokens"] == 3_400
    assert data["tokens"] == 4_003_400
    assert data["calls"] == 9
    assert data["spend"] == pytest.approx(4.50, rel=1e-3)


def test_analytics_every_model_has_share_and_rounded_cost(usage_db):
    """The loop only touched models[:20] while the full list was returned, so
    anything past the 20th came back unrounded and without a `share` key."""
    for m in _analytics()["models"]:
        assert "share" in m
        assert m["cost"] == round(m["cost"], 2)


def test_analytics_days_param_is_clamped(usage_db):
    assert _analytics(days=99999)["days"] == 365
    assert _analytics(days=0)["days"] == 1


def test_analytics_survives_a_garbage_days_param(usage_db):
    from urllib.parse import urlparse
    from api.nyx_analytics import handle_analytics
    h = _Stub()
    handle_analytics(h, urlparse("/api/nyx/analytics?days=abc"))
    assert h.status == 200
    assert h.payload()["days"] == 30


# ── revision_summary: the artifact-list badges (design 4a) ─────────────────


@pytest.fixture
def rev_ws(tmp_path):
    """A workspace with one snapshotted file."""
    ws = tmp_path / "ws"
    (ws).mkdir()
    return ws


def test_revision_summary_none_before_any_snapshot(rev_ws):
    """No snapshot means no badge at all — the client must not show a zero."""
    from api.nyx_revisions import revision_summary
    (rev_ws / "a.py").write_text("x = 1\n")
    assert revision_summary(rev_ws, "a.py") is None


def test_revision_summary_counts_and_reports_clean(rev_ws):
    from api.nyx_revisions import revision_summary, snapshot_revision
    (rev_ws / "a.py").write_text("x = 1\n")
    snapshot_revision(rev_ws, "a.py", "first")
    s = revision_summary(rev_ws, "a.py")
    assert s["count"] == 1
    assert s["unstaged"] is False
    assert s["added"] == 0 and s["removed"] == 0


def test_revision_summary_detects_unstaged_edits_with_a_stat(rev_ws):
    from api.nyx_revisions import revision_summary, snapshot_revision
    f = rev_ws / "a.py"
    f.write_text("a\nb\nc\n")
    snapshot_revision(rev_ws, "a.py", "first")
    f.write_text("a\nB\nc\nd\n")     # one line changed, one added
    s = revision_summary(rev_ws, "a.py")
    assert s["unstaged"] is True
    assert s["added"] == 2 and s["removed"] == 1


def test_revision_summary_counts_multiple_revisions(rev_ws):
    from api.nyx_revisions import revision_summary, snapshot_revision
    f = rev_ws / "a.py"
    for i in range(3):
        f.write_text(f"v{i}\n")
        snapshot_revision(rev_ws, "a.py", f"r{i}")
    s = revision_summary(rev_ws, "a.py")
    assert s["count"] == 3
    assert s["latest_rev"].startswith("r")


def test_revision_summary_binary_reports_no_stat(rev_ws):
    """A binary file gets the chip but no numbers, never a fabricated +0/-0."""
    from api.nyx_revisions import revision_summary, snapshot_revision
    f = rev_ws / "blob.bin"
    f.write_bytes(b"\x00\x01\x02")
    snapshot_revision(rev_ws, "blob.bin", "first")
    f.write_bytes(b"\x00\x01\x03\x04")
    s = revision_summary(rev_ws, "blob.bin")
    assert s["unstaged"] is True
    assert "added" not in s and "removed" not in s


def test_revision_summary_survives_a_deleted_working_file(rev_ws):
    """The file can vanish after being snapshotted; the summary must not raise."""
    from api.nyx_revisions import revision_summary, snapshot_revision
    f = rev_ws / "a.py"
    f.write_text("x\n")
    snapshot_revision(rev_ws, "a.py", "first")
    f.unlink()
    s = revision_summary(rev_ws, "a.py")
    assert s["count"] == 1
    assert s.get("unstaged") is True
