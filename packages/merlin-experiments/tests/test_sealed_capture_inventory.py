"""Exact inventory identities, independent mutation checks and hashing cost."""

import stat

import pytest
from merlin_experiments.capture_execution import sealed_m2m as M
from merlin_experiments.capture_execution import sealed_static as S


def _legacy_identity(root):
    rows = S._tree(root)
    compact = [(name, *sorted(info.items())) for name, info in rows.items()]
    return {"members": len(rows), "bytes": sum(row.get("bytes", 0) for row in rows.values()),
            "sha256": S._digest(S._json(compact))}


def _overlapping_tree(tmp_path):
    root = tmp_path / "runtime"
    venv, base, sibling = root / "opt/capture-venv", root / "usr/local/base", root / "opt/capture-venv-other"
    for path in (venv / "lib", base / "lib", sibling):
        path.mkdir(parents=True)
    (venv / "lib/one.so").write_bytes(b"a" * (2 * 1024 * 1024))
    (base / "lib/two.so").write_bytes(b"b" * (1024 * 1024))
    (sibling / "guard.bin").write_bytes(b"x" * 17)
    return root, venv, base


def test_projected_identities_equal_complete_legacy_inventories(tmp_path):
    root, venv, base = _overlapping_tree(tmp_path)
    rows = S._tree(root)
    for selected in (root, venv, venv / "lib", base, base / "lib"):
        assert M._snapshot_rows(rows, selected.relative_to(root).as_posix()) == _legacy_identity(selected)
    assert M._snapshot_tree(root) == _legacy_identity(root)


def test_complete_file_inventory_retains_legacy_refusal(tmp_path):
    member = tmp_path / "standalone.bin"
    member.write_bytes(b"selected bytes")
    for observe in (M._snapshot_tree, _legacy_identity):
        with pytest.raises(S.SealedCaptureError, match="tree is absent or indirect"):
            observe(member)


@pytest.mark.parametrize("relative", ["", "../runtime", "/runtime", "opt/../opt", "missing", "opt/capture-venv/lib/one.so"])
def test_projection_refuses_invalid_or_non_directory_members(tmp_path, relative):
    root, _, _ = _overlapping_tree(tmp_path)
    with pytest.raises(S.SealedCaptureError):
        M._snapshot_rows(S._tree(root), relative)


@pytest.mark.parametrize("mutation", ["bytes", "mode", "added", "removed"])
def test_subsequent_checkpoint_detects_nested_mutation(tmp_path, mutation):
    root, venv, _ = _overlapping_tree(tmp_path)
    relative = venv.relative_to(root).as_posix()
    before_rows = S._tree(root)
    before = M._snapshot_rows(before_rows, relative)
    member = venv / "lib/one.so"
    if mutation == "bytes":
        member.write_bytes(b"c" * member.stat().st_size)
    elif mutation == "mode":
        member.chmod(stat.S_IMODE(member.stat().st_mode) ^ stat.S_IXUSR)
    elif mutation == "added":
        (venv / "lib/added.bin").write_bytes(b"new")
    else:
        member.unlink()
    after_rows = S._tree(root)
    assert M._snapshot_rows(before_rows, relative) == before
    assert M._snapshot_rows(after_rows, relative) != before
    assert M._snapshot_rows(after_rows, relative) == _legacy_identity(venv)
    assert M._snapshot_rows(after_rows) != M._snapshot_rows(before_rows)


def test_staged_selection_uses_one_checkpoint_and_refuses_next_mutation(tmp_path, monkeypatch):
    source, runtime = tmp_path / "source", tmp_path / "runtime"
    roots = {"venv": runtime / "opt/capture-venv", "base": runtime / "usr/local/base",
             "m2m": source / "m2m-src/m2m", "workload": source / "workload",
             "merlin": source / "merlin-src/merlin"}
    roots["schemas"] = roots["merlin"] / "_data/schemas"
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
        (path / "selected.txt").write_text(path.name)
    plan = {"schema": M.SCHEMA, "base": "/usr/local/base", "merlin_root": "/selected/merlin",
            "schemas_root": "/selected/merlin/_data/schemas",
            "selected_trees": {name: _legacy_identity(path) for name, path in roots.items()}}
    source_rows, runtime_rows = S._tree(source), S._tree(runtime)
    monkeypatch.setattr(M, "_snapshot_tree", lambda *_: pytest.fail("nested tree must not be rehashed"))
    assert M._verify_staged_selection(plan, source, runtime, source_rows=source_rows, runtime_rows=runtime_rows) == plan["selected_trees"]["schemas"]
    (roots["workload"] / "selected.txt").write_text("changed")
    with pytest.raises(M.SealedM2MError, match="staged workload bytes differ"):
        M._verify_staged_selection(plan, source, runtime, source_rows=S._tree(source), runtime_rows=S._tree(runtime))


def test_overlapping_checkpoint_halves_large_file_hashing(tmp_path, monkeypatch):
    root, venv, base = _overlapping_tree(tmp_path)
    observed = []
    original = S._file_digest

    def counted(path):
        observed.append(path.stat().st_size)
        return original(path)

    monkeypatch.setattr(S, "_file_digest", counted)
    expected = [_legacy_identity(path) for path in (root, venv, base)]
    old_bytes, old_reads = sum(observed), len(observed)
    observed.clear()
    rows = S._tree(root)
    actual = [M._snapshot_rows(rows, path.relative_to(root).as_posix()) for path in (root, venv, base)]
    new_bytes, new_reads = sum(observed), len(observed)
    assert actual == expected
    assert (old_bytes, new_bytes) == (6_291_473, 3_145_745)
    assert (old_reads, new_reads) == (5, 3)
    print(f"hashed bytes {old_bytes} -> {new_bytes}; file reads {old_reads} -> {new_reads}")
