"""Relative directory aliases retain one selected target and their exact topology."""

import os

import pytest
from merlin_experiments.capture_execution import readonly_runtime, runtime_store
from merlin_experiments.capture_execution import sealed_m2m as M
from merlin_experiments.capture_execution import sealed_static as S


def _base_tree(root):
    manual = root / "local/share/man/page.1"
    manual.parent.mkdir(parents=True)
    manual.write_bytes(b"selected manual\n")
    (root / "local/man").symlink_to("share/man", target_is_directory=True)
    return root


def test_directory_alias_is_recorded_without_traversing_its_target_twice(tmp_path, monkeypatch):
    root = _base_tree(tmp_path / "base")
    records = []
    original = M._json

    def observe(value):
        records.extend(value)
        return original(value)

    monkeypatch.setattr(M, "_json", observe)
    identity = M._source_tree(root)
    rows = {name: dict(fields) for name, *fields in records}
    assert rows["local/man"] == {"kind": "symlink", "target": "share/man"}
    assert rows["local"]["members"] == ["man", "share"]
    assert not any(name.startswith("local/man/") for name in rows)
    assert identity["bytes"] == len(b"selected manual\n")
    assert identity["members"] == 6


def test_directory_alias_digest_is_root_relative_and_binds_link_text(tmp_path):
    first = _base_tree(tmp_path / "first")
    second = _base_tree(tmp_path / "second")
    assert M._source_tree(first) == M._source_tree(second)
    link = second / "local/man"
    link.unlink()
    link.symlink_to("./share/man", target_is_directory=True)
    assert M._source_tree(first) != M._source_tree(second)
    link.unlink()
    link.symlink_to("share/man", target_is_directory=True)
    (second / "local/share/man/page.1").write_bytes(b"changed manual\n")
    assert M._source_tree(first) != M._source_tree(second)


@pytest.mark.parametrize("kind, message", [
    ("absolute", "absolute directory link"),
    ("escaping", "directory link.*outside"),
    ("dangling", "dangling.*link"),
    ("loop", "unresolvable.*link"),
])
def test_unsafe_directory_aliases_are_refused(tmp_path, kind, message):
    root = _base_tree(tmp_path / "base")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "local/man"
    link.unlink()
    target = {
        "absolute": str(root / "local/share/man"),
        "escaping": "../../outside",
        "dangling": "share/absent",
        "loop": "man",
    }[kind]
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(M.SealedM2MError, match=message):
        M._source_tree(root)
    with pytest.raises(ValueError, match=message):
        M._snapshot_tree(root)


def test_snapshot_inventory_preserves_static_and_file_link_refusals(tmp_path):
    root = _base_tree(tmp_path / "base")
    with pytest.raises(S.SealedCaptureError, match="symlink"):
        S._tree(root)
    assert M._snapshot_tree(root) == M._source_tree(root)
    (root / "file-alias").symlink_to("local/share/man/page.1")
    with pytest.raises(S.SealedCaptureError, match="symlink"):
        M._snapshot_tree(root)


def test_runtime_copy_link_cache_and_readonly_mount_retain_directory_alias(tmp_path, monkeypatch):
    base = _base_tree(tmp_path / "base")
    (base / "bin").mkdir()
    (base / "bin/python3.12").write_bytes(b"selected interpreter")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "lib").mkdir()
    (venv / "lib64").symlink_to("lib", target_is_directory=True)
    (venv / "bin/python").symlink_to(base / "bin/python3.12")
    plan = {
        "venv": str(venv), "base": str(base), "system_libs": [],
        "selected_trees": {"venv": M._source_tree(venv, skip_lib64=True), "base": M._source_tree(base)},
    }
    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    entry = runtime_store.store_entry(plan)
    copied_base = entry / base.relative_to("/")
    assert M._snapshot_tree(copied_base) == plan["selected_trees"]["base"]
    assert os.readlink(copied_base / "local/man") == "share/man"
    token = runtime_store.verified_cached_entry(plan, tmp_path)
    assert token is not None
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runtime_store.link_runtime(plan, runtime, verified_cache=token)
    guest_base = runtime / base.relative_to("/")
    assert os.readlink(guest_base / "local/man") == "share/man"
    assert M._snapshot_tree(guest_base) == plan["selected_trees"]["base"]
    manual = "local/share/man/page.1"
    assert (guest_base / manual).stat().st_ino == (copied_base / manual).stat().st_ino
    assert not (runtime / "opt/capture-venv/bin/python").is_symlink()
    assert not (runtime / "opt/capture-venv/lib64").exists()
    marker = entry.parent / (entry.name + ".complete.json")
    selection = readonly_runtime.select(plan, entry=entry, marker_sha256=S._file_digest(marker), destination=tmp_path)
    rows = readonly_runtime.effective_rows(selection)
    assert rows[(base.relative_to("/") / "local/man").as_posix()] == {"kind": "symlink", "target": "share/man"}
    assert readonly_runtime.verify(plan, selection, tmp_path) == selection
    mounts = readonly_runtime.mounts(selection)
    assert str(entry / base.parts[1]) in mounts


def test_staged_sources_and_input_trees_retain_directory_alias(tmp_path):
    roots = {name: _base_tree(tmp_path / name) for name in ("workload", "merlin", "input")}
    m2m = _base_tree(tmp_path / "selected/m2m")
    schemas = roots["merlin"] / "_data/schemas"
    schemas.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    plan = {
        "schema": M.SCHEMA_V3, "m2m_root": str(m2m.parent), "workload_root": str(roots["workload"]),
        "merlin_root": str(roots["merlin"]), "schemas_root": str(schemas),
        "selected_trees": {"merlin": M._source_tree(roots["merlin"])},
        "selected_inputs": [{"kind": "tree", "source": str(roots["input"]), "guest_member": "model"}],
    }
    M._stage_source(plan, source)
    for original, guest in (
        (m2m, source / "m2m-src/m2m"),
        (roots["merlin"], source / "merlin-src/merlin"),
        (roots["workload"], source / "workload"),
        (roots["input"], source / "inputs/model"),
    ):
        assert os.readlink(guest / "local/man") == "share/man"
        assert M._snapshot_tree(guest) == M._source_tree(original)
