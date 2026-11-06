"""Fixture-free registration and fail-closed runner controls, like the Radiance example."""

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import pytest


@pytest.fixture
def runner(monkeypatch):
    root = Path(__file__).resolve().parents[3] / "examples/atlas/model_bringup"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.delitem(__import__("sys").modules, "source_view", raising=False)
    spec = importlib.util.spec_from_file_location("atlas_model_runner_example", root / "run_tests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_owns_every_moved_module(runner):
    modules = runner.source_view.test_modules()
    assert len(modules) == 67
    assert len(set(modules)) == len(modules)


def test_required_skip_is_a_failure(runner):
    class MissingFixture(unittest.TestCase):
        @unittest.skip("fixture unavailable")
        def test_required(self):
            pass

    result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=runner.RequiredResult).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(MissingFixture)
    )
    assert not result.wasSuccessful()
    assert len(result.failures) == 1
    assert not result.skipped
    assert "required check skipped: fixture unavailable" in result.failures[0][1]


def test_canonical_import_escape_is_refused(runner, tmp_path, monkeypatch):
    roots = {"atlas-mlir": str(tmp_path / "compiler"), "merlin": str(tmp_path / "merlin")}
    receipt = {"entries": [{"relative_path": "tools/execution_helper.py"}], "roots": roots}
    monkeypatch.setitem(
        runner.sys.modules,
        "execution_helper",
        SimpleNamespace(__file__=str(tmp_path / "compiler/tools/execution_helper.py")),
    )
    with pytest.raises(AssertionError, match="import escaped execution view"):
        runner.assert_imports(tmp_path / "view", receipt)


def test_full_suite_selects_the_distinct_codec_model(runner, tmp_path, monkeypatch):
    modules = ["test/test_tiny_full_export.py", "test/test_smol_full_phases.py"]
    monkeypatch.setattr(runner.source_view, "test_modules", lambda *args: modules)
    monkeypatch.setattr(runner.source_view, "load_map", lambda: {})
    monkeypatch.setattr(runner.source_view, "verify_view", lambda *args: {})
    monkeypatch.setattr(runner, "configure", lambda *args: None)
    commands = []
    monkeypatch.setattr(
        runner.subprocess, "run", lambda cmd, **kwargs: commands.append(cmd) or SimpleNamespace(returncode=0)
    )
    args = ["run_tests.py", "tests"]
    for key in ("view", "atlas", "merlin", "kernels", "model-root", "codec-model-root"):
        args.extend(["--" + key, str(tmp_path / key)])
    args.extend(["--out", str(tmp_path / "result.json")])
    monkeypatch.setattr(runner.sys, "argv", args)
    assert runner.main() == 0
    assert commands[0][commands[0].index("--model-root") + 1] == str(tmp_path / "codec-model-root")
    assert commands[1][commands[1].index("--model-root") + 1] == str(tmp_path / "model-root")
