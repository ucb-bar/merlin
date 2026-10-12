import hashlib
import os

import pytest

from merlin.common import bundle_snapshot as bs


def reference(root):
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != root / bs.MARKER:
            rows.append(
                (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)
            )
    digest = hashlib.sha256()
    for name, identity, size in rows:
        digest.update(name.encode() + b"\0" + str(size).encode() + b"\0" + identity.encode() + b"\n")
    return digest.hexdigest(), len(rows), sum(row[2] for row in rows)


def populate(root):
    (root / "a").mkdir()
    (root / "a" / "file").write_bytes(b"one")
    (root / "a" / bs.MARKER).write_bytes(b"nested payload")
    (root / "a-").write_bytes(b"two")
    (root / bs.MARKER).write_bytes(b"marker")


def test_digest_parts_order_marker_roster_and_tamper(tmp_path):
    populate(tmp_path)
    assert bs.snapshot_content(tmp_path) == reference(tmp_path)
    roster = bs._scanned_roster(bs._scanned_members(tmp_path))
    assert roster == bs._member_roster(tmp_path, sorted(tmp_path.rglob("*")))
    assert [r[0] for r in roster] == ["a", "a/file", "a/snapshot.json", "a-", bs.MARKER]
    before = bs.snapshot_content(tmp_path)
    (tmp_path / "a" / bs.MARKER).write_bytes(b"NESTED PAYLOAD")
    assert bs.snapshot_content(tmp_path) != before
    before = bs.snapshot_content(tmp_path)
    (tmp_path / "a-").write_bytes(b"TWO")
    assert bs.snapshot_content(tmp_path) != before


@pytest.mark.parametrize("change", ["add", "remove", "resize", "marker"])
def test_scoped_roster_and_marker_changes_refuse(tmp_path, change):
    populate(tmp_path)
    issued = bs._Verification()
    issued.root = tmp_path
    issued.marker_bytes = (tmp_path / bs.MARKER).read_bytes()
    issued.members = bs._scanned_roster(bs._scanned_members(tmp_path))
    issued.signature = bs.snapshot_content(tmp_path)
    bs._ISSUED_VERIFICATIONS.add(issued)
    token = bs._ACTIVE_VERIFICATION.set(issued)
    try:
        if change == "add":
            (tmp_path / "new").write_bytes(b"new")
        if change == "remove":
            (tmp_path / "a-").unlink()
        if change == "resize":
            (tmp_path / "a-").write_bytes(b"longer")
        if change == "marker":
            (tmp_path / bs.MARKER).write_bytes(b"MARKER")
        with pytest.raises(RuntimeError, match="changed"):
            bs.snapshot_content(tmp_path)
    finally:
        bs._ACTIVE_VERIFICATION.reset(token)
        bs._ISSUED_VERIFICATIONS.discard(issued)


@pytest.mark.parametrize("kind", ["link", "fifo"])
def test_indirect_and_special_refusal(tmp_path, kind):
    if kind == "link":
        (tmp_path / "bad").symlink_to("/etc/passwd")
    else:
        os.mkfifo(tmp_path / "bad")
    with pytest.raises(RuntimeError, match="indirect|special"):
        bs.snapshot_content(tmp_path)
