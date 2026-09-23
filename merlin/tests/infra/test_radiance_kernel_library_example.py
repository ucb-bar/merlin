"""Example routing and synthetic artifact integrity; no upstream or hardware invocation."""

import hashlib
import importlib.util
import io
import subprocess
import tarfile
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root


def load_materializer():
    path = repo_root() / "examples/radiance/phase1/kernel_library/materialize.py"
    spec = importlib.util.spec_from_file_location("radiance_kernel_library_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_bytes_and_pinned_semantics_preserved():
    module = load_materializer()
    payload = module.SELECTION.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == "edbc6a6f7926efd9c21ef742e412a54da3d04d12124eecc75bdfa95e602a6e05"
    selection = yaml.safe_load(payload)
    assert selection["source_commit"] == "399757f6da3b2c75980821b2479e9f1ca732e8ea"
    assert selection["package"] == "out/artifacts/targets/radiance/kernel_library_pr1_v1"


def test_cli_preserves_default_output_without_running_materialization(tmp_path, monkeypatch):
    monkeypatch.delenv("MERLIN_OUT_ROOT", raising=False)
    module = load_materializer()
    observed = []
    monkeypatch.setattr(module, "materialize", lambda checkout, output: observed.append((checkout, output)) or {})

    def forbidden(*args, **kwargs):
        raise AssertionError("routing test must not execute git or generate kernel payloads")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.chdir(tmp_path)
    assert module.main(["--checkout", str(tmp_path)]) == 0
    assert observed == [(tmp_path, repo_root() / "out/artifacts/targets/radiance/kernel_library_pr1_v1")]


def test_cli_preserves_explicit_output(tmp_path, monkeypatch):
    module = load_materializer()
    observed = []
    monkeypatch.setattr(module, "materialize", lambda checkout, output: observed.append((checkout, output)) or {})
    output = tmp_path / "selected-output"
    assert module.main(["--checkout", str(tmp_path), "--output", str(output)]) == 0
    assert observed == [(tmp_path, output)]


def test_cli_honors_output_root(tmp_path, monkeypatch):
    module = load_materializer()
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "products"))
    observed = []
    monkeypatch.setattr(module, "materialize", lambda checkout, output: observed.append(output) or {})
    assert module.main(["--checkout", str(tmp_path)]) == 0
    assert observed == [tmp_path / "products/artifacts/targets/radiance/kernel_library_pr1_v1"]


def test_cli_does_not_hide_output_symlink(tmp_path, monkeypatch):
    module = load_materializer()
    output = tmp_path / "alias"
    output.symlink_to(tmp_path / "missing", target_is_directory=True)
    observed = []
    monkeypatch.setattr(module, "materialize", lambda checkout, output: observed.append(output) or {})
    assert module.main(["--checkout", str(tmp_path), "--output", str(output)]) == 0
    assert observed == [output]


@pytest.fixture
def synthetic_export(tmp_path, monkeypatch):
    module = load_materializer()
    selection = tmp_path / "selection.yaml"
    selection.write_text(
        yaml.safe_dump(
            {
                "source_commit": "a" * 40,
                "source_pin": "synthetic",
                "shared_paths": ["kernel.c"],
            }
        )
    )
    monkeypatch.setattr(module, "SELECTION", selection)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        member = tarfile.TarInfo("kernel.c")
        member.size = len(b"fixture")
        stream.addfile(member, io.BytesIO(b"fixture"))
    monkeypatch.setattr(module, "_git", lambda *args: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=archive.getvalue()))
    output = tmp_path / "export"
    manifest = module.materialize(tmp_path, output)
    return module, output, manifest


def test_unchanged_export_is_reused(synthetic_export, tmp_path):
    module, output, manifest = synthetic_export
    before = (output / "kernel.c").stat().st_ino
    assert module.materialize(tmp_path, output) == manifest
    assert (output / "kernel.c").stat().st_ino == before


@pytest.mark.parametrize("member", ["kernel.c", "selection.yaml", "README.md", "manifest.yaml", "extra.txt"])
def test_changed_export_is_refused_without_overwrite(synthetic_export, tmp_path, member):
    module, output, _ = synthetic_export
    (output / member).write_text("changed")
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        module.materialize(tmp_path, output)
    assert (output / member).read_text() == "changed"


def test_missing_export_member_is_refused(synthetic_export, tmp_path):
    module, output, _ = synthetic_export
    (output / "kernel.c").unlink()
    with pytest.raises(FileExistsError):
        module.materialize(tmp_path, output)
    assert not (output / "kernel.c").exists()


@pytest.mark.parametrize("directory", [False, True])
def test_export_symlinks_are_refused(synthetic_export, tmp_path, directory):
    module, output, _ = synthetic_export
    if directory:
        alias = tmp_path / "alias"
        alias.symlink_to(output, target_is_directory=True)
        output = alias
    else:
        payload = output / "kernel.c"
        external = tmp_path / "external.c"
        payload.rename(external)
        payload.symlink_to(external)
    with pytest.raises(ValueError, match="artifact"):
        module.materialize(tmp_path, output)
