"""Shared source discovery preserves Phase 1 and Phase 2 inventory boundaries."""

import os

import pytest

from merlin.common.source_membership import SourceMembershipError, python_members


def test_membership_is_sorted_recursive_and_ignores_only_generated_bytecode(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/b.py").write_text("# b\n")
    (tmp_path / "a.py").write_text("# a\n")
    (tmp_path / "notes.md").write_text("not implementation")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/ignored.py").write_text("generated")
    assert python_members(tmp_path) == {
        "a.py": (tmp_path / "a.py").resolve(),
        "nested/b.py": (tmp_path / "nested/b.py").resolve(),
    }


@pytest.mark.parametrize("kind", ["file-link", "directory-link", "broken-link", "fifo"])
def test_unsafe_nonpython_entries_are_not_silently_omitted(tmp_path, kind):
    entry = tmp_path / "entry"
    if kind == "fifo":
        os.mkfifo(entry, 0o444)
    elif kind == "directory-link":
        entry.symlink_to(tmp_path, target_is_directory=True)
    elif kind == "broken-link":
        entry.symlink_to(tmp_path / "absent")
    else:
        (tmp_path / "target").write_text("value")
        entry.symlink_to(tmp_path / "target")
    with pytest.raises(SourceMembershipError, match="symlinked or nonregular"):
        python_members(tmp_path)


def test_phase1_delegates_with_existing_error_and_source_receipt_identity(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import source_inputs
    from merlin_experiments.spec import SpecError

    absent = tmp_path / "absent"
    with pytest.raises(SpecError, match=f"phase-1 source directory is absent or symlinked: {absent}"):
        source_inputs._python_members(absent, "prefix:")
    calls = []

    def discover(root, *, label):
        calls.append((root, label))
        return {"owner.py": tmp_path / "owner.py"}

    monkeypatch.setattr(source_inputs, "python_members", discover)
    assert source_inputs._python_members(tmp_path, "prefix:") == {"prefix:owner.py": str(tmp_path / "owner.py")}
    assert calls == [(tmp_path, "phase-1")]
