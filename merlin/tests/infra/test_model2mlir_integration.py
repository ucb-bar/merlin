"""Configuration and optional-import contracts, without framework capture or downloads."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from merlin.capture import bundle
from merlin.common import paths
from merlin.frontends import quant_ext
from merlin.integrations import model2mlir
from merlin.llvmlower import toolchain
from merlin.targetgen import capsule_source


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    for key in (
        "MERLIN_MODEL2MLIR",
        "MERLIN_M2M_DIR",
        "MERLIN_M2M_PYTHON",
        "MERLIN_M2M_VENV",
        "MERLIN_COMPILER_PYTHON",
        "MERLIN_COMPILER_VENV",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(paths, "_dotenv", lambda: {})
    existing = {name: module for name, module in sys.modules.items() if name == "m2m" or name.startswith("m2m.")}
    for name in existing:
        del sys.modules[name]
    before = list(sys.path)
    yield
    sys.path[:] = before
    for name in tuple(sys.modules):
        if name == "m2m" or name.startswith("m2m."):
            del sys.modules[name]
    sys.modules.update(existing)


@pytest.mark.parametrize(
    ("process", "dotenv", "expected"),
    [
        ({}, {}, "/path/to/model2MLIR"),
        ({"MERLIN_MODEL2MLIR": "A", "MERLIN_M2M_DIR": "B"}, {}, "A"),
        ({"MERLIN_M2M_DIR": "B"}, {"MERLIN_MODEL2MLIR": "A"}, "B"),
        ({}, {"MERLIN_MODEL2MLIR": "A", "MERLIN_M2M_DIR": "B"}, "A"),
        ({}, {"MERLIN_M2M_DIR": "B"}, "B"),
    ],
)
def test_root_keeps_capture_alias_precedence(monkeypatch, process, dotenv, expected):
    monkeypatch.setattr(paths, "_dotenv", lambda: dotenv)
    for key, value in process.items():
        monkeypatch.setenv(key, value)
    assert model2mlir.root() == bundle.model2mlir_root() == Path(expected)


def test_compiler_and_capsule_share_configured_capture_checkout(monkeypatch, tmp_path):
    """The two legacy consumers must not pick a different sibling checkout."""
    selected = tmp_path / "selected"
    monkeypatch.setenv("MERLIN_MODEL2MLIR", str(selected))
    monkeypatch.setenv("MERLIN_M2M_DIR", str(tmp_path / "other"))
    assert toolchain.m2m_dir() == capsule_source._m2m_dir() == selected
    assert toolchain.compiler_python() == selected / ".venv/bin/python"


def _checkout(root: Path, *, broken=False):
    package = root / "m2m"
    quant = package / "ir/quant"
    quant.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "ir/__init__.py").write_text("")
    (quant / "dialect.py").write_text("# declared pure dialect source\n")
    (quant / "__init__.py").write_text(
        "raise ImportError('synthetic broken dependency')\n"
        if broken
        else "class Dialect:\n    name = 'quant_ext'\nQuant = Dialect()\n"
    )
    importlib.invalidate_caches()
    return root


def test_quant_honors_dotenv_and_retries_prior_absence(tmp_path, monkeypatch):
    assert quant_ext.load_dialect() is None
    root = _checkout(tmp_path / "configured")
    monkeypatch.setattr(paths, "_dotenv", lambda: {"MERLIN_M2M_DIR": str(root)})
    before = list(sys.path)
    dialect = quant_ext.load_dialect()
    assert dialect is not None and dialect.name == "quant_ext"
    assert quant_ext.load_dialect() is dialect
    assert sys.path == before
    assert bundle.model2mlir_root() == quant_ext._m2m_dir() == root


def test_quant_failed_import_restores_path_and_can_retry(tmp_path, monkeypatch):
    root = _checkout(tmp_path / "configured", broken=True)
    monkeypatch.setenv("MERLIN_M2M_DIR", str(root))
    before = list(sys.path)
    assert quant_ext.load_dialect() is None
    assert sys.path == before
    (root / "m2m/ir/quant/__init__.py").write_text("Quant = 'repaired dialect'\n")
    importlib.invalidate_caches()
    assert quant_ext.load_dialect() == "repaired dialect"
    assert sys.path == before


def test_quant_refuses_foreign_loaded_graph_without_purging_it(tmp_path, monkeypatch):
    first = _checkout(tmp_path / "first")
    second = _checkout(tmp_path / "second")
    monkeypatch.setenv("MERLIN_M2M_DIR", str(first))
    dialect = quant_ext.load_dialect()
    module = sys.modules["m2m.ir.quant"]
    monkeypatch.setenv("MERLIN_M2M_DIR", str(second))
    before = list(sys.path)
    assert quant_ext.load_dialect() is None
    assert sys.modules["m2m.ir.quant"] is module
    assert module.Quant is dialect
    assert sys.path == before
    monkeypatch.setenv("MERLIN_M2M_DIR", str(first))
    assert quant_ext.load_dialect() is dialect


def test_invalid_preferred_root_does_not_select_alternate(tmp_path, monkeypatch):
    alternate = _checkout(tmp_path / "alternate")
    monkeypatch.setenv("MERLIN_MODEL2MLIR", str(tmp_path / "missing"))
    monkeypatch.setenv("MERLIN_M2M_DIR", str(alternate))
    assert quant_ext.load_dialect() is None
    assert "m2m" not in sys.modules


def test_quant_refuses_foreign_namespace_path(tmp_path, monkeypatch):
    root = _checkout(tmp_path / "configured")
    monkeypatch.setenv("MERLIN_M2M_DIR", str(root))
    assert quant_ext.load_dialect() is not None
    sys.modules["m2m"].__path__.append(str(tmp_path / "foreign"))
    assert quant_ext.load_dialect() is None


def test_preloaded_optional_child_before_parent_is_refused(tmp_path, monkeypatch):
    from types import ModuleType

    root = _checkout(tmp_path / "configured")
    monkeypatch.setenv("MERLIN_M2M_DIR", str(root))
    foreign = ModuleType("m2m.foreign")
    foreign.__spec__ = importlib.util.spec_from_file_location("m2m.foreign", tmp_path / "foreign.py")
    sys.modules["m2m.foreign"] = foreign
    assert quant_ext.load_dialect() is None
    assert sys.modules["m2m.foreign"] is foreign
    assert "m2m" not in sys.modules


def test_same_checkout_symlink_preserves_loaded_dialect_identity(tmp_path, monkeypatch):
    root = _checkout(tmp_path / "configured")
    monkeypatch.setenv("MERLIN_M2M_DIR", str(root))
    dialect = quant_ext.load_dialect()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    monkeypatch.setenv("MERLIN_M2M_DIR", str(alias))
    assert bundle.model2mlir_root() == alias
    assert quant_ext.load_dialect() is dialect


def _python(venv: Path) -> Path:
    path = venv / "bin/python"
    path.parent.mkdir(parents=True)
    path.symlink_to(sys.executable)
    return path


def test_capture_path_selection_is_lazy_but_execution_validation_is_strict(tmp_path):
    checkout = tmp_path / "missing-checkout"
    expected = checkout / ".venv/bin/python"
    assert model2mlir.capture_python_path(checkout=checkout) == expected
    source = capsule_source.PytorchRefSource(m2m_dir=checkout)
    assert source.python == expected
    assert not source.available()
    with pytest.raises(RuntimeError, match="missing or not executable"):
        model2mlir.capture_python(checkout=checkout)


def test_capsule_explicit_checkout_selects_its_own_venv(tmp_path, monkeypatch):
    configured = _checkout(tmp_path / "configured")
    selected = _checkout(tmp_path / "selected")
    _python(configured / ".venv")
    expected = _python(selected / ".venv")
    monkeypatch.setenv("MERLIN_M2M_DIR", str(configured))
    source = capsule_source.PytorchRefSource(m2m_dir=selected)
    assert source.python == expected and source.python != expected.resolve()
    assert source.available()


@pytest.mark.parametrize("setting", ["MERLIN_M2M_PYTHON", "MERLIN_M2M_VENV"])
@pytest.mark.parametrize("dotenv", [False, True])
def test_capsule_uses_shared_capture_configuration(tmp_path, monkeypatch, setting, dotenv):
    checkout = _checkout(tmp_path / "checkout")
    fallback = _python(checkout / ".venv")
    selected = _python(tmp_path / "selected")
    value = str(selected if setting == "MERLIN_M2M_PYTHON" else selected.parent.parent)
    if dotenv:
        monkeypatch.setattr(paths, "_dotenv", lambda: {setting: value})
    else:
        monkeypatch.setenv(setting, value)
    source = capsule_source.PytorchRefSource(m2m_dir=checkout)
    assert source.python == model2mlir.capture_python(checkout=checkout) == selected
    assert source.python != fallback and source.available()


def test_capsule_interpreter_precedence_and_explicit_argument(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path / "checkout")
    dotenv_python = _python(tmp_path / "dotenv")
    process_python = _python(tmp_path / "process")
    argument_python = _python(tmp_path / "argument")
    monkeypatch.setattr(paths, "_dotenv", lambda: {"MERLIN_M2M_PYTHON": str(dotenv_python)})
    monkeypatch.setenv("MERLIN_M2M_VENV", str(process_python.parent.parent))
    assert capsule_source.PytorchRefSource(m2m_dir=checkout).python == dotenv_python
    monkeypatch.setenv("MERLIN_M2M_PYTHON", str(process_python))
    assert capsule_source.PytorchRefSource(m2m_dir=checkout).python == process_python
    assert capsule_source.PytorchRefSource(m2m_dir=checkout, python=argument_python).python == argument_python


@pytest.mark.parametrize("kind", ["missing", "nonexecutable", "directory"])
def test_capsule_bad_explicit_python_never_falls_back(tmp_path, monkeypatch, kind):
    checkout = _checkout(tmp_path / "checkout")
    _python(checkout / ".venv")
    selected = tmp_path / "invalid-python"
    if kind == "nonexecutable":
        selected.write_text("not executable")
    elif kind == "directory":
        selected.mkdir()
    monkeypatch.setenv("MERLIN_M2M_PYTHON", str(selected))
    source = capsule_source.PytorchRefSource(m2m_dir=checkout)
    assert source.python == selected and not source.available()
    with pytest.raises(capsule_source.M2MUnavailable, match="missing"):
        source.capture({"op": "matmul"})


def test_capsule_private_selector_retains_sibling_checkout_default(tmp_path, monkeypatch):
    monkeypatch.setattr(capsule_source, "repo_root", lambda: tmp_path / "merlin")
    assert capsule_source._m2m_python() == tmp_path / "model2MLIR/.venv/bin/python"


def test_capsule_workload_pin_reaches_capture_even_without_shared_python(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path / "checkout")
    workload = checkout / "workloads/example"
    workload.mkdir(parents=True)
    pinned = _python(workload / "custom")
    (workload / "capture.toml").write_text('venv = "custom"\n')
    loader = workload / "loader.py"
    loader.write_text("def get_model_and_inputs(): pass\n")
    monkeypatch.setenv("MERLIN_M2M_DIR", str(checkout))
    source = capsule_source.PytorchRefSource()
    assert not source.available()
    selected = capsule_source.model_capture_python("example")
    assert selected == pinned
    observed = {}

    def capture_worker(*args, **kwargs):
        observed.update(kwargs)
        return "worker invoked"

    monkeypatch.setattr(source, "_run", capture_worker)
    assert source.capture_loader(loader, "f32", python=selected, workdir=tmp_path / "capture") == "worker invoked"
    assert observed["python"] == pinned and observed["src"] == loader.read_text()
    assert capsule_source.model_capture_python(None) is None


def test_shared_capture_interpreter_preserves_venv_path_and_ignores_compiler(tmp_path, monkeypatch):
    capture = _python(tmp_path / "capture")
    monkeypatch.setenv("MERLIN_M2M_VENV", str(capture.parent.parent))
    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", "/missing/compiler/python")
    assert model2mlir.capture_python() == capture
    assert toolchain.compiler_python() == Path("/missing/compiler/python")
    assert model2mlir.capture_python() != capture.resolve()


def test_explicit_capture_python_precedes_venv_and_reads_dotenv(tmp_path, monkeypatch):
    explicit = _python(tmp_path / "explicit")
    fallback = _python(tmp_path / "fallback")
    monkeypatch.setattr(paths, "_dotenv", lambda: {"MERLIN_M2M_PYTHON": str(explicit)})
    monkeypatch.setenv("MERLIN_M2M_VENV", str(fallback.parent.parent))
    assert model2mlir.capture_python() == explicit
    monkeypatch.setenv("MERLIN_M2M_PYTHON", str(fallback))
    assert model2mlir.capture_python() == fallback


@pytest.mark.parametrize("setting", ["MERLIN_M2M_PYTHON", "MERLIN_M2M_VENV"])
def test_bad_explicit_interpreter_never_falls_back(tmp_path, monkeypatch, setting):
    _python(tmp_path / "checkout/.venv")
    monkeypatch.setenv(setting, str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="capture Python is missing or not executable"):
        model2mlir.capture_python(checkout=tmp_path / "checkout")


def test_nonexecutable_capture_choice_is_refused(tmp_path, monkeypatch):
    file = tmp_path / "not_executable"
    file.write_text("not an executable")
    monkeypatch.setenv("MERLIN_M2M_PYTHON", str(file))
    with pytest.raises(RuntimeError, match="not executable"):
        model2mlir.capture_python()


def test_capture_checkout_default_and_workload_policy_remain_distinct(tmp_path, monkeypatch):
    common = _python(tmp_path / "checkout/.venv")
    workload = _python(tmp_path / "checkout/workloads/example/special")
    monkeypatch.setenv("MERLIN_M2M_DIR", str(tmp_path / "checkout"))
    monkeypatch.setattr(bundle, "capture_config", lambda model: {"venv": "special"})
    assert model2mlir.capture_python() == common
    assert bundle.capture_python("example") == workload


def test_gguf_capture_uses_framework_python_not_compiler(tmp_path, monkeypatch):
    from merlin.frontends.adapters import gguf

    capture = _python(tmp_path / "capture")
    source = tmp_path / "model.gguf"
    source.write_bytes(b"synthetic checkpoint; subprocess is observed, not executed")
    monkeypatch.setenv("MERLIN_M2M_PYTHON", str(capture))
    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", "/missing/compiler/python")
    calls = []
    monkeypatch.setattr("subprocess.run", lambda argv, **kwargs: calls.append(argv))
    monkeypatch.setattr(bundle.CaptureBundle, "require", lambda self: self)
    result = gguf.ingest(source, model="fixture", out=tmp_path / "bundle")
    assert result.root == tmp_path / "bundle"
    assert calls[0][:2] == [str(capture), "-c"]
    assert "m2m.frontends.gguf" in calls[0][2]
