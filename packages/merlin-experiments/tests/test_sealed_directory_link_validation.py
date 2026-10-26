"""Directory aliases depend only on one link text and real selected directories."""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from merlin_experiments.capture_execution import readonly_runtime, runtime_store
from merlin_experiments.capture_execution import sealed_m2m as M
from merlin_experiments.capture_execution import sealed_static as S


@pytest.mark.parametrize("inventory", [M._source_tree, M._snapshot_tree])
def test_external_intermediate_cannot_change_staged_alias_bytes(tmp_path, inventory):
    workload = tmp_path / "original/workload"
    (workload / "target").mkdir(parents=True)
    (workload / "target/file").write_text("WORKLOAD")
    bridge = tmp_path / "original/m2m-src"
    bridge.mkdir()
    (bridge / "m2m").symlink_to("../workload")
    (workload / "alias").symlink_to("../m2m-src/m2m/target")
    staged_package = tmp_path / "staged/source/m2m-src/m2m/target"
    staged_package.mkdir(parents=True)
    (staged_package / "file").write_text("M2M")
    assert (workload / "alias/file").read_text() == "WORKLOAD"
    with pytest.raises(ValueError, match="outside"):
        inventory(workload)
    with pytest.raises(ValueError, match="outside"):
        M._copy_source_tree(workload, tmp_path / "staged/source/workload")


@pytest.mark.parametrize("inventory", [M._source_tree, M._snapshot_tree])
def test_retargeted_external_intermediate_is_always_refused(tmp_path, inventory):
    root = tmp_path / "root"
    for name in ("first", "second"):
        (root / name).mkdir(parents=True)
        (root / name / "file").write_text(name)
    outside = tmp_path / "outside"
    outside.mkdir()
    back = outside / "back"
    (root / "alias").symlink_to("../outside/back")
    for name in ("first", "second"):
        back.symlink_to(f"../root/{name}")
        assert (root / "alias/file").read_text() == name
        with pytest.raises(ValueError, match="outside"):
            inventory(root)
        back.unlink()


@pytest.mark.parametrize("target", ["pivot", "pivot/child", "pivot/../target"])
@pytest.mark.parametrize("inventory", [M._source_tree, M._snapshot_tree])
def test_internal_symlink_components_are_refused(tmp_path, target, inventory):
    root = tmp_path / "root"
    (root / "target/child").mkdir(parents=True)
    (root / "pivot").symlink_to("target")
    (root / "alias").symlink_to(target)
    with pytest.raises(ValueError, match="symlink component"):
        inventory(root)


def test_each_lexical_target_component_must_be_a_real_directory(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "file").write_bytes(b"selected")
    for target in ("file", "file/../root"):
        link = root / "alias"
        link.symlink_to(target)
        with pytest.raises(ValueError, match="non-directory"):
            S._directory_symlink(link, root)
        link.unlink()


def test_escaping_text_is_refused_even_if_link_changes_after_read(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "target").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    link = root / "alias"
    link.symlink_to("../outside")
    readlink = os.readlink

    def swap(path):
        text = readlink(path)
        link.unlink()
        link.symlink_to("target")
        return text

    observed = Mock(side_effect=swap)
    monkeypatch.setattr(os, "readlink", observed)
    with pytest.raises(ValueError, match="outside"):
        S._directory_symlink(link, root)
    observed.assert_called_once_with(link)


@pytest.mark.parametrize("operation", ["source", "snapshot", "copy", "store", "readonly"])
def test_one_validated_read_controls_inventory_and_copy(tmp_path, monkeypatch, operation):
    root = tmp_path / "root"
    (root / "local/share/man").mkdir(parents=True)
    (root / "local/share/man/page.1").write_bytes(b"selected manual\n")
    link = root / "local/man"
    link.symlink_to("share/man")
    selected = M._source_tree(root)
    readlink = os.readlink

    def changing_text(path):
        assert Path(path) == link
        if observed.call_count == 1:
            link.unlink()
            link.symlink_to("../../outside")
            return "share/man"
        return "../../outside"

    observed = Mock(side_effect=changing_text)
    monkeypatch.setattr(os, "readlink", observed)
    if operation == "source":
        assert M._source_tree(root) == selected
    elif operation == "snapshot":
        rows = S._tree(root, allow_directory_symlinks=True)
        assert rows["local/man"] == {"kind": "symlink", "target": "share/man"}
    elif operation == "readonly":
        rows = readonly_runtime.effective_rows({"entry": str(root)})
        assert rows["local/man"] == {"kind": "symlink", "target": "share/man"}
    else:
        destination = tmp_path / "copy"
        copy = M._copy_source_tree if operation == "copy" else runtime_store._link_entry
        copy(root, destination)
        assert readlink(destination / "local/man") == "share/man"
        monkeypatch.setattr(os, "readlink", readlink)
        assert M._snapshot_tree(destination) == selected
    observed.assert_called_once_with(link)


def test_valid_alias_never_uses_the_filesystem_resolver(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "local/share/man").mkdir(parents=True)
    link = root / "local/man"
    link.symlink_to("./share/../share/man")

    def refuse_resolve(*args, **kwargs):
        pytest.fail("directory alias validation must be lexical")

    monkeypatch.setattr(Path, "resolve", refuse_resolve)
    assert S._directory_symlink(link, root) == {"kind": "symlink", "target": "./share/../share/man"}
