"""Unused dangling file metadata survives snapshots; closure paths remain strict."""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from merlin_experiments.capture_execution import readonly_runtime, runtime_store
from merlin_experiments.capture_execution import sealed_m2m as M
from merlin_experiments.capture_execution import sealed_static as S


def _base_tree(root, target="../tools/docs/kernel-doc"):
    scripts = root / "src/headers/scripts"
    scripts.mkdir(parents=True)
    (scripts / "kernel-doc").symlink_to(target)
    return root, scripts / "kernel-doc"


def _base_inventory(root):
    return M._source_tree(root, allow_dangling_symlinks=True)


@pytest.mark.parametrize("target", [
    "../tools/docs/kernel-doc", "./missing", "../../../../outside", "/absent/merlin-kernel-doc",
])
def test_dangling_file_text_is_recorded_and_copied_exactly(tmp_path, target):
    root, link = _base_tree(tmp_path / "base", target)
    row = {"kind": "dangling_symlink", "target": target}
    if target in {"../../../../outside", "/absent/merlin-kernel-doc"}:
        row = {"kind": "external_symlink", "target": target}
    assert M._tree(root, allow_dangling_symlinks=True)["src/headers/scripts/kernel-doc"] == row
    selected = _base_inventory(root)
    copied = tmp_path / "staged"
    M._copy_source_tree(root, copied)
    staged_link = copied / link.relative_to(root)
    assert os.readlink(staged_link) == target
    assert staged_link.is_symlink() and not staged_link.exists()
    assert M._snapshot_tree(copied, allow_dangling_symlinks=True) == selected
    assert selected["bytes"] == 0
    with pytest.raises(S.SealedCaptureError, match="symlink"):
        S._tree(root)


def test_dangling_file_digest_is_root_relative_and_binds_exact_text(tmp_path):
    first, _ = _base_tree(tmp_path / "first")
    second, link = _base_tree(tmp_path / "second")
    assert _base_inventory(first) == _base_inventory(second)
    link.unlink()
    link.symlink_to("../tools/docs/./kernel-doc")
    assert _base_inventory(first) != _base_inventory(second)
    assert M._snapshot_tree(first, allow_dangling_symlinks=True) != M._snapshot_tree(
        second, allow_dangling_symlinks=True
    )


def test_runtime_selection_keeps_base_metadata_but_refuses_a_dangling_interpreter(tmp_path):
    from merlin_experiments.phase0 import m2m_runtime

    base, _ = _base_tree(tmp_path / "base")
    interpreter = base / "bin/python3.12"
    interpreter.parent.mkdir()
    interpreter.write_bytes(b"selected interpreter")
    interpreter.chmod(0o755)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {base / 'bin'}\n")
    (venv / "bin/python").symlink_to(interpreter)
    assert m2m_runtime._runtime(venv / "bin/python")["base_python"] == _base_inventory(base)
    interpreter.unlink()
    interpreter.symlink_to("absent")
    with pytest.raises(M.SealedM2MError, match="base CPython is absent"):
        m2m_runtime._runtime(venv / "bin/python")


@pytest.mark.parametrize("operation", ["source", "snapshot", "copy", "store", "readonly"])
def test_dangling_file_uses_one_link_text_without_resolving(tmp_path, monkeypatch, operation):
    root, link = _base_tree(tmp_path / "base")
    selected = _base_inventory(root)
    readlink = os.readlink

    def changing_text(path):
        assert Path(path) == link
        target = readlink(path)
        link.unlink()
        link.symlink_to("/different/absent-target")
        return target

    def refuse_resolve(*args, **kwargs):
        pytest.fail("dangling file metadata must not use the filesystem resolver")

    observed = Mock(side_effect=changing_text)
    monkeypatch.setattr(os, "readlink", observed)
    monkeypatch.setattr(Path, "resolve", refuse_resolve)
    if operation == "source":
        assert _base_inventory(root) == selected
    elif operation in {"snapshot", "readonly"}:
        rows = (
            M._tree(root, allow_dangling_symlinks=True)
            if operation == "snapshot"
            else readonly_runtime.effective_rows({"entry": str(root)})
        )
        assert rows[link.relative_to(root).as_posix()] == {
            "kind": "dangling_symlink", "target": "../tools/docs/kernel-doc",
        }
    else:
        destination = tmp_path / "copy"
        copy = M._copy_source_tree if operation == "copy" else runtime_store._link_entry
        copy(root, destination)
        copied = destination / link.relative_to(root)
        assert readlink(copied) == "../tools/docs/kernel-doc"
        assert copied.is_symlink() and not copied.exists()
    observed.assert_called_once_with(link)


@pytest.mark.parametrize("overlay_version", [None, 2, 3])
def test_runtime_store_and_guest_preserve_dangling_file_metadata(tmp_path, monkeypatch, overlay_version):
    base, link = _base_tree(tmp_path / "base")
    venv = tmp_path / "venv"
    venv.mkdir()
    plan = {
        "venv": str(venv), "base": str(base), "system_libs": [],
        "selected_trees": {"venv": M._source_tree(venv), "base": _base_inventory(base)},
    }
    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    entry = runtime_store.store_entry(plan)
    cached = runtime_store.verified_cached_entry(plan, tmp_path)
    assert cached is not None
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runtime_store.link_runtime(plan, runtime, verified_cache=cached)
    for tree in (entry / base.relative_to("/"), runtime / base.relative_to("/")):
        copied = tree / link.relative_to(base)
        assert os.readlink(copied) == "../tools/docs/kernel-doc"
        assert copied.is_symlink() and not copied.exists()
        assert M._snapshot_tree(tree, allow_dangling_symlinks=True) == plan["selected_trees"]["base"]
    marker = entry.parent / (entry.name + ".complete.json")
    options = {}
    if overlay_version is not None:
        predicted = tmp_path / "predicted-venv"
        overlay = predicted / "nested/code.py"
        overlay.parent.mkdir(parents=True)
        overlay.write_text("selected = True\n")
        plan["selected_trees"]["venv"] = M._source_tree(predicted)
        options = {"overlays": {"opt/capture-venv/nested/code.py": overlay}, "overlay_version": overlay_version}
    selection = readonly_runtime.select(
        plan, entry=entry, marker_sha256=S._file_digest(marker), destination=tmp_path, **options,
    )
    assert readonly_runtime.verify(plan, selection, tmp_path) == selection


@pytest.mark.parametrize("member", [
    "bin/python", "pyvenv.cfg", "lib/python3.12/site-packages/startup.pth",
    "package/direct_import.py", "package/dependency.so", "data/weights.bin",
])
def test_closure_tree_refuses_a_member_resolving_through_a_dangling_link(tmp_path, member):
    root = tmp_path / "closure"
    path = root / member
    path.parent.mkdir(parents=True)
    dangling = tmp_path / "dangling"
    dangling.symlink_to("absent")
    path.symlink_to(os.path.relpath(dangling, path.parent))
    with pytest.raises(M.SealedM2MError, match="dangling source link"):
        M._source_tree(root)


@pytest.mark.parametrize("tree", [False, True])
def test_declared_data_path_refuses_dangling_links(tmp_path, tree):
    root, link = _base_tree(tmp_path / "input")
    with pytest.raises(ValueError, match="symlink component|dangling source link"):
        M._input_selection(root if tree else link, "model")


@pytest.mark.parametrize("operation", ["source", "snapshot", "copy", "store"])
def test_explicitly_dangling_directory_link_is_refused(tmp_path, operation):
    # Unix links have no file/directory type; a trailing slash requires a directory.
    root, _ = _base_tree(tmp_path / "base", "../missing-directory/")
    with pytest.raises(ValueError, match="dangling directory link"):
        if operation == "source":
            _base_inventory(root)
        elif operation == "snapshot":
            M._snapshot_tree(root, allow_dangling_symlinks=True)
        else:
            copy = M._copy_source_tree if operation == "copy" else runtime_store._link_entry
            copy(root, tmp_path / "copy")


def test_output_snapshot_refuses_an_unused_dangling_sidecar(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.bin").write_bytes(b"selected output")
    (output / "unused-sidecar").symlink_to("/source/missing-sidecar")
    with pytest.raises(S.SealedCaptureError, match="link"):
        M._snapshot_tree(output)
