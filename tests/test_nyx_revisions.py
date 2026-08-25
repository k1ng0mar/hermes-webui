"""Tests for the artifact revisions subsystem (design 4b).

Coverage:
  - encode/decode roundtrip on the kind of paths a real agent produces
  - snapshot creates r0, then r1, with monotonic ids
  - snapshot is idempotent: identical content reuses the latest rev
  - list_revisions returns newest-first and respects limit
  - diff_revisions returns unified diff + correct +/- counts
  - diff handles "head" alias for the current working file
  - diff flags binary files (no text diff possible)
  - revert restores the chosen rev and captures the pre-revert state
  - revert is reversible: reverting twice returns to the original
  - FIFO prune caps stored revisions at MAX_REVISIONS_KEPT
  - safe_resolve traversal guard rejects parent escapes
"""
import pytest
from pathlib import Path

from api.nyx_revisions import (
    DEFAULT_REVISION_LIMIT,
    MAX_REVISIONS_KEPT,
    decode_path,
    diff_revisions,
    encode_path,
    file_rev_dir,
    list_revisions,
    revert_to_revision,
    revisions_root,
    snapshot_revision,
)


# ── encode/decode ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("rel", [
    "",
    "foo.py",
    "src/main.py",
    "a/b/c/d.py",
    ".hidden",
    ".git/config",
    "weird__name.py",
    "a=b/c?d",
    "with space and 'quote'/x.md",
    "UPPER/lower.MIXED",
])
def test_encode_decode_roundtrip(rel: str) -> None:
    e = encode_path(rel)
    d = decode_path(e)
    assert d == rel
    # encoded must not contain path separators
    assert "/" not in e
    # encoded must not collide with a different input
    assert e != encode_path(rel + "x")


def test_encode_collision_resistant() -> None:
    """Different inputs produce different encodings."""
    inputs = ["foo.py", "fo_o.py", "f/o.o.py", "foo.p_y", "foo.py/"]
    encs = {encode_path(s) for s in inputs}
    assert len(encs) == len(inputs)


# ── snapshot ────────────────────────────────────────────────────────────────


def test_snapshot_creates_r0(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("print('hello')\n")
    r = snapshot_revision(tmp_path, "foo.py", message="initial")
    assert r["rev"] == "r0"
    assert r["created"] is True
    assert r["size"] > 0
    # content + meta.json on disk
    rev_dir = file_rev_dir(tmp_path, "foo.py") / "r0"
    assert (rev_dir / "content").exists()
    assert (rev_dir / "meta.json").exists()


def test_snapshot_is_monotonic(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("v1\n")
    r1 = snapshot_revision(tmp_path, "foo.py")
    f.write_text("v2\n")
    r2 = snapshot_revision(tmp_path, "foo.py")
    f.write_text("v3\n")
    r3 = snapshot_revision(tmp_path, "foo.py")
    assert (r1["rev"], r2["rev"], r3["rev"]) == ("r0", "r1", "r2")


def test_snapshot_idempotent_on_identical_content(tmp_path: Path) -> None:
    """Two snapshots of the same content reuse the latest rev."""
    f = tmp_path / "foo.py"
    f.write_text("same content\n")
    r1 = snapshot_revision(tmp_path, "foo.py")
    r2 = snapshot_revision(tmp_path, "foo.py")
    assert r1["rev"] == r2["rev"] == "r0"
    assert r1["created"] is True
    assert r2["created"] is False


def test_snapshot_records_parent(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("v1\n")
    snapshot_revision(tmp_path, "foo.py")
    f.write_text("v2\n")
    r2 = snapshot_revision(tmp_path, "foo.py")
    rev_dir = file_rev_dir(tmp_path, "foo.py") / "r1"
    import json
    meta = json.loads((rev_dir / "meta.json").read_text())
    assert meta["parent"] == "r0"
    assert r2["rev"] == "r1"


def test_snapshot_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        snapshot_revision(tmp_path, "nope.py")


# ── list ────────────────────────────────────────────────────────────────────


def test_list_revisions_newest_first(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("a\n")
    snapshot_revision(tmp_path, "foo.py", message="a")
    f.write_text("b\n")
    snapshot_revision(tmp_path, "foo.py", message="b")
    f.write_text("c\n")
    snapshot_revision(tmp_path, "foo.py", message="c")
    revs = list_revisions(tmp_path, "foo.py")
    assert [r["rev"] for r in revs] == ["r2", "r1", "r0"]
    assert [r["message"] for r in revs] == ["c", "b", "a"]


def test_list_revisions_respects_limit(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    for i in range(5):
        f.write_text(f"v{i}\n")
        snapshot_revision(tmp_path, "foo.py")
    revs = list_revisions(tmp_path, "foo.py", limit=2)
    assert len(revs) == 2
    assert [r["rev"] for r in revs] == ["r4", "r3"]


def test_list_revisions_empty_when_no_snapshots(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("just a file, never snapshotted\n")
    assert list_revisions(tmp_path, "foo.py") == []


# ── diff ────────────────────────────────────────────────────────────────────


def test_diff_basic(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("a\nb\nc\nd\n")
    snapshot_revision(tmp_path, "foo.py")
    f.write_text("a\nb modified\nc\nd\n")
    snapshot_revision(tmp_path, "foo.py")
    result = diff_revisions(tmp_path, "foo.py", "r0", "r1")
    assert result["binary"] is False
    assert result["stats"]["removed"] == 1
    assert result["stats"]["added"] == 1
    assert "-b" in result["diff"]
    assert "+b modified" in result["diff"]


def test_diff_against_head(tmp_path: Path) -> None:
    """head alias should read the current working file, not the latest rev."""
    f = tmp_path / "foo.py"
    f.write_text("line 1\nline 2\n")
    snapshot_revision(tmp_path, "foo.py")
    f.write_text("line 1\nline 2 changed\n")
    # Don't snapshot — just diff the rev against the current file
    result = diff_revisions(tmp_path, "foo.py", "r0", "head")
    assert result["stats"]["added"] == 1
    assert result["stats"]["removed"] == 1
    assert "line 2 changed" in result["diff"]


def test_diff_identical_returns_no_changes(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("same\n")
    snapshot_revision(tmp_path, "foo.py")
    f.write_text("same\n")  # unchanged on disk
    snapshot_revision(tmp_path, "foo.py")  # dedupes
    result = diff_revisions(tmp_path, "foo.py", "r0", "r0")
    assert result["stats"]["added"] == 0
    assert result["stats"]["removed"] == 0


def test_diff_binary_file(tmp_path: Path) -> None:
    f = tmp_path / "foo.bin"
    f.write_bytes(b"\x00\x01\x02\xff\xfe")
    snapshot_revision(tmp_path, "foo.bin")
    f.write_bytes(b"\x00\x01\x02\xfd")
    snapshot_revision(tmp_path, "foo.bin")
    result = diff_revisions(tmp_path, "foo.bin", "r0", "r1")
    assert result["binary"] is True
    assert result["diff"] == ""
    assert result["stats"] == {"added": 0, "removed": 0}


# ── revert ──────────────────────────────────────────────────────────────────


def test_revert_restores_content(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("v1\n")
    snapshot_revision(tmp_path, "foo.py")
    f.write_text("v2\n")
    snapshot_revision(tmp_path, "foo.py")
    f.write_text("v3 current\n")
    # Revert to v1
    result = revert_to_revision(tmp_path, "foo.py", "r0")
    assert result["restored_from"] == "r0"
    assert f.read_text() == "v1\n"
    # The pre-revert state (v3 current) was captured as a new rev
    revs = list_revisions(tmp_path, "foo.py")
    assert revs[0]["message"].startswith("pre-revert to r0")


def test_revert_is_reversible(tmp_path: Path) -> None:
    """Reverting twice should return to the original content."""
    f = tmp_path / "foo.py"
    f.write_text("v1\n")
    snapshot_revision(tmp_path, "foo.py")
    f.write_text("v2\n")
    snapshot_revision(tmp_path, "foo.py")
    f.write_text("v3\n")
    # Capture v3 (the current "head") so we can get back to it
    snapshot_revision(tmp_path, "foo.py")  # r2 == v3

    # Revert to r0
    revert_to_revision(tmp_path, "foo.py", "r0")
    assert f.read_text() == "v1\n"
    # Revert to r2 (the pre-revert capture)
    revert_to_revision(tmp_path, "foo.py", "r2")
    assert f.read_text() == "v3\n"


def test_revert_missing_rev_raises(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("v1\n")
    with pytest.raises(FileNotFoundError):
        revert_to_revision(tmp_path, "foo.py", "r0")


def test_revert_missing_file_raises(tmp_path: Path) -> None:
    (file_rev_dir(tmp_path, "nope.py") / "r0").mkdir(parents=True)
    (file_rev_dir(tmp_path, "nope.py") / "r0" / "content").write_text("x")
    with pytest.raises(FileNotFoundError):
        revert_to_revision(tmp_path, "nope.py", "r0")


# ── FIFO prune ──────────────────────────────────────────────────────────────


def test_fifo_prune_caps_storage(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    file_dir = file_rev_dir(tmp_path, "foo.py")
    # Snapshot more than MAX_REVISIONS_KEPT times.
    for i in range(MAX_REVISIONS_KEPT + 10):
        f.write_text(f"v{i}\n")
        snapshot_revision(tmp_path, "foo.py")
    remaining = [c for c in file_dir.iterdir() if c.is_dir()]
    assert len(remaining) == MAX_REVISIONS_KEPT
    # The earliest (lowest rev) is gone; the latest survives.
    surviving = sorted(int(c.name[1:]) for c in remaining)
    assert surviving[0] == 10
    assert surviving[-1] == MAX_REVISIONS_KEPT + 9


# ── safe_resolve guard (path traversal) ─────────────────────────────────────


def test_snapshot_rejects_path_traversal(tmp_path: Path) -> None:
    """A path that escapes the workspace should be refused, not written."""
    (tmp_path / "secret.py").write_text("SECRET")
    with pytest.raises(ValueError):
        snapshot_revision(tmp_path, "../secret.py")


def test_revert_rejects_path_traversal(tmp_path: Path) -> None:
    (tmp_path / "secret.py").write_text("SECRET")
    with pytest.raises(ValueError):
        revert_to_revision(tmp_path, "../secret.py", "r0")
