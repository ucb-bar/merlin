"""Example setup routing with all external checks and actions replaced by local doubles."""

import importlib.util
import sys
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def setup_example(tmp_path, monkeypatch):
    path = repo_root() / "examples/atlas/target/setup.py"
    spec = importlib.util.spec_from_file_location("atlas_setup_example", path)
    module = importlib.util.module_from_spec(spec)
    import_path = list(sys.path)
    spec.loader.exec_module(module)
    assert sys.path == import_path
    assert module.ROOT == repo_root()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    hardware = tmp_path / "hardware"
    (hardware / "npu-model/npu_model").mkdir(parents=True)
    monkeypatch.setattr(module, "_resolve_atlas_npu", lambda cli: hardware)
    monkeypatch.setattr(module, "_sha", lambda path: "synthetic")
    monkeypatch.setattr(module, "_CHIPYARD_ATLAS_DEFAULT", tmp_path / "missing")
    monkeypatch.setitem(
        sys.modules, "merlin.targetgen.rtl.mlc_bridge", SimpleNamespace(arc_available=lambda target: True)
    )
    from merlin.targetgen import rtl

    monkeypatch.setattr(rtl, "mlc_bridge", sys.modules["merlin.targetgen.rtl.mlc_bridge"], raising=False)
    from merlin import targetgen
    from merlin.common import paths

    monkeypatch.setattr(paths, "env", lambda name: None)
    calls = []
    monkeypatch.setattr(
        targetgen,
        "capability_manifests",
        SimpleNamespace(
            materialize_generated_target=lambda target, dest: calls.append((target, dest)) or tmp_path / "generated",
        ),
        raising=False,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("status checks must not sync or execute external tools")

    monkeypatch.setattr(module.subprocess, "run", forbidden)
    return module, calls


@pytest.mark.parametrize("args", [[], ["--no-target-package"]])
def test_status_is_read_only_by_default(setup_example, tmp_path, args):
    module, calls = setup_example
    assert module.main(args) == 0
    assert calls == []
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / "generated").exists()


@pytest.mark.parametrize("custom", [False, True])
def test_materialization_requires_explicit_request(setup_example, tmp_path, custom):
    module, calls = setup_example
    dest = tmp_path / "selected"
    args = ["--target-package-dir", str(dest)] if custom else ["--materialize-target-package"]
    assert module.main(args) == 0
    assert calls == [("atlas", dest if custom else None)]


def test_conflicting_generation_options_are_rejected(setup_example):
    module, calls = setup_example
    with pytest.raises(SystemExit) as exc:
        module.main(["--no-target-package", "--materialize-target-package"])
    assert exc.value.code == 2
    assert calls == []


def test_write_env_is_explicit_and_preserves_existing_keys(setup_example, tmp_path):
    module, calls = setup_example
    env = tmp_path / ".env"
    env.write_text("MERLIN_EXT_ATLAS_NPU=/keep/existing\n")
    assert module.main(["--write-env"]) == 0
    contents = env.read_text()
    assert contents.startswith("MERLIN_EXT_ATLAS_NPU=/keep/existing\n")
    assert contents.count("MERLIN_EXT_ATLAS_NPU=") == 1
    assert f"MERLIN_EXT_NPU_MODEL={tmp_path}/hardware/npu-model" in contents
    assert calls == []
    assert module.main(["--write-env"]) == 0
    assert env.read_text() == contents


def test_sync_is_explicit_and_does_not_generate_package(setup_example, monkeypatch, tmp_path):
    module, calls = setup_example
    commands = []
    monkeypatch.setattr(
        module.subprocess, "run", lambda command, **kw: commands.append((command, kw)) or SimpleNamespace(returncode=0)
    )
    assert module.main(["--sync-npu-model"]) == 0
    assert commands == [(["uv", "sync"], {"cwd": str(tmp_path / "hardware/npu-model")})]
    assert calls == []


def test_failed_sync_prevents_requested_materialization(setup_example, monkeypatch):
    module, calls = setup_example
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kw: SimpleNamespace(returncode=1))
    assert module.main(["--sync-npu-model", "--materialize-target-package"]) == 1
    assert calls == []
