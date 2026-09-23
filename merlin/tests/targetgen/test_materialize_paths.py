"""Materialization refuses unsafe destinations before changing operator files."""

from pathlib import Path

import pytest

from merlin.common.yaml import write_yaml
from merlin.targetgen import publish as PB


def package(path):
    path.mkdir(parents=True, exist_ok=True)
    write_yaml(path / "manifest.yaml", {"package_id": "fixture"})
    (path / "payload").write_text("preserve me")
    return path


@pytest.mark.parametrize("field", ["target", "package_id"])
@pytest.mark.parametrize("value", ["", ".", "..", "../escape", "a/b", "/absolute"])
def test_invalid_components_refused_without_writes(tmp_path, field, value):
    source = package(tmp_path / "source")
    artifacts = tmp_path / "artifacts"
    args = {"target": "fixture", "package_id": "installed"}
    args[field] = str(tmp_path / "absolute") if value == "/absolute" else value
    with pytest.raises(PB.MaterializeRefused, match="single.*component"):
        PB.materialize_package(source=source, artifacts_root=artifacts, force=True, **args)
    assert not artifacts.exists()
    assert (source / "payload").read_text() == "preserve me"


@pytest.mark.parametrize("relation", ["same", "source_parent", "source_child", "symlink", "ancestor_symlink"])
def test_overlap_and_symlink_refusal_preserves_trees(tmp_path, relation):
    artifacts = tmp_path / "artifacts"
    destination = artifacts / "targets" / "fixture" / "installed"
    if relation == "same":
        source = package(destination)
    elif relation == "source_parent":
        source = package(destination.parent)
    elif relation == "source_child":
        source = package(destination / "source")
    else:
        source = package(tmp_path / "source")
        external = package(tmp_path / "external")
        link = destination if relation == "symlink" else destination.parent
        link.parent.mkdir(parents=True)
        link.symlink_to(external, target_is_directory=True)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(PB.MaterializeRefused, match="overlap|symlink"):
        PB.materialize_package("fixture", source, package_id="installed", artifacts_root=artifacts, force=True)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("failure", ["copy", "manifest", "rename", None])
def test_force_stages_before_replacement_and_retains_recovery(tmp_path, monkeypatch, failure):
    source = package(tmp_path / "source")
    artifacts = tmp_path / "artifacts"
    destination = package(artifacts / "targets" / "fixture" / "installed")
    (destination / "payload").write_text("old package")
    if failure == "copy":

        def fail_copy(*args, **kwargs):
            raise OSError("copy failed")

        monkeypatch.setattr(PB.shutil, "copytree", fail_copy)
    elif failure == "manifest":

        def fail_write(*args, **kwargs):
            raise OSError("manifest failed")

        monkeypatch.setattr(PB.package_records, "write_record", fail_write)
    elif failure == "rename":
        original = Path.rename

        def fail_install(path, target):
            if path.name == "package" and Path(target) == destination:
                raise OSError("rename failed")
            return original(path, target)

        monkeypatch.setattr(Path, "rename", fail_install)
    if failure:
        with pytest.raises(OSError, match=failure):
            PB.materialize_package("fixture", source, package_id="installed", artifacts_root=artifacts, force=True)
        assert (destination / "payload").read_text() == "old package"
    else:
        assert (
            PB.materialize_package("fixture", source, package_id="installed", artifacts_root=artifacts, force=True)
            == destination
        )
        assert (destination / "payload").read_text() == "preserve me"
        backups = list(destination.parent.glob(".materialize-*/previous/payload"))
        assert len(backups) == 1 and backups[0].read_text() == "old package"
        installed = PB.package_records.read_record(destination)
        assert Path(installed["promotion"]["previous_package"]) == backups[0].parent
        assert list(destination.parent.glob("*/manifest.yaml")) == [destination / "manifest.yaml"]
        assert PB.select_champion("fixture", artifacts_root=artifacts).package_dir == destination
    assert (source / "payload").read_text() == "preserve me"


def test_score_inside_replacement_and_non_directory_destination_are_refused(tmp_path):
    source = package(tmp_path / "source")
    artifacts = tmp_path / "artifacts"
    destination = package(artifacts / "targets" / "fixture" / "installed")
    score = destination / "score.json"
    score.write_text("{}")
    with pytest.raises(PB.MaterializeRefused, match="overlap"):
        PB.materialize_package(
            "fixture", source, package_id="installed", artifacts_root=artifacts, score_path=score, force=True
        )
    assert score.read_text() == "{}"
    other = destination.parent / "ordinary-file"
    other.write_text("operator data")
    with pytest.raises(PB.MaterializeRefused, match="not a directory"):
        PB.materialize_package("fixture", source, package_id=other.name, artifacts_root=artifacts, force=True)
    assert other.read_text() == "operator data"


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_first_install_removes_empty_stage_without_misreporting_success(tmp_path, monkeypatch, capsys, cleanup_fails):
    source = package(tmp_path / "source")
    artifacts = tmp_path / "artifacts"
    if cleanup_fails:
        original = Path.rmdir

        def fail_cleanup(path):
            if path.name.startswith(".materialize-"):
                raise OSError("cleanup unavailable")
            return original(path)

        monkeypatch.setattr(Path, "rmdir", fail_cleanup)
    destination = PB.materialize_package("fixture", source, package_id="installed", artifacts_root=artifacts)
    assert (destination / "payload").read_text() == "preserve me"
    containers = list(destination.parent.glob(".materialize-*"))
    assert len(containers) == int(cleanup_fails)
    if cleanup_fails:
        assert not list(containers[0].iterdir())
        assert "package installed" in capsys.readouterr().err
