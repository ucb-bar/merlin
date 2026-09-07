"""Lazy compiler helpers must be pinned without executing package initializers."""
from __future__ import annotations

import ast
import importlib

import pytest

from merlin.common.paths import repo_root
from merlin.perf.static_imports import imported_attribute_paths, resolve_lazy_export


GETTER = '''
from importlib import import_module
def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(import_module(f".{module}", __name__), name)
'''


def test_literal_comprehension_selects_one_actual_module():
    initializer = '''
_EXPORTS = {name: module for module, names in (
    ("global_plan", ("GlobalPlan", "CycleInterval")),
    ("pipeline", ("execute",)),
) for name in names}
''' + GETTER
    result = resolve_lazy_export(initializer.encode(), package="merlin.lowering", symbol="GlobalPlan")
    assert result.status == "resolved"
    assert result.module == "merlin.lowering.global_plan"
    assert resolve_lazy_export(initializer.encode(), package="merlin.lowering", symbol="*").status == "unresolved"


def test_dynamic_table_is_not_executed(tmp_path):
    marker = tmp_path / "must_not_exist"
    initializer = f'_EXPORTS = __import__("pathlib").Path({str(marker)!r}).touch()\n' + GETTER
    result = resolve_lazy_export(initializer.encode(), package="merlin.runtime", symbol="Tensor")
    assert result.status == "unresolved"
    assert "Call" in result.reason
    assert not marker.exists()


def test_import_alias_and_constant_getattr_are_selected():
    tree = ast.parse('rt.Tensor.zeros(); getattr(rt, "Metrics"); rt.commandbuffer.load()')
    paths = imported_attribute_paths(tree, {"rt": {"merlin.runtime"}})
    assert {"merlin.runtime.Tensor", "merlin.runtime.Metrics", "merlin.runtime.commandbuffer"} <= paths


@pytest.fixture
def closure_environment(tmp_path, monkeypatch):
    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    launcher = importlib.import_module("run_global_perf_experiment")
    merlin_dir = tmp_path / "host"
    root = merlin_dir / "python/merlin"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    runtime = root / "runtime"
    runtime.mkdir()
    (runtime / "__init__.py").write_text(
        '_EXPORTS = {"Tensor": "tensor", "simulate": "simulator", "reference_outputs": "reference"}\n'
        + GETTER + '\nraise RuntimeError("initializer must never execute")\n')
    (runtime / "tensor.py").write_text("class Tensor: pass\n")
    (runtime / "simulator.py").write_text('raise RuntimeError("masked simulator")\n')
    (runtime / "reference.py").write_text('raise RuntimeError("masked reference")\n')
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    monkeypatch.setattr(launcher.PAS, "merlin_dir", lambda: merlin_dir)
    return launcher, candidate, runtime


@pytest.mark.parametrize("statement", [
    "from merlin.runtime import Tensor\n",
    "import merlin.runtime as rt\nvalue = rt.Tensor\n",
    'from merlin import runtime\nvalue = getattr(runtime, "Tensor")\n',
])
def test_selected_leaf_edit_changes_compiler_identity_without_masked_siblings(closure_environment, statement):
    launcher, candidate, runtime = closure_environment
    (candidate / "compiler.py").write_text(statement)
    before = launcher.compiler_dependency_record(candidate)
    assert "runtime/tensor.py" in before["shared_sources"]
    assert "runtime/simulator.py" not in before["shared_sources"]
    assert "runtime/reference.py" not in before["shared_sources"]
    assert before["selected_lazy_exports"] == {"merlin.runtime.Tensor": "merlin.runtime.tensor"}
    (runtime / "simulator.py").write_text("# irrelevant sibling edit\n")
    assert launcher.compiler_dependency_record(candidate) == before
    (runtime / "tensor.py").write_text("class Tensor: dtype = 'changed'\n")
    after = launcher.compiler_dependency_record(candidate)
    assert after["candidate_sha256"] == before["candidate_sha256"]
    assert after["compiler_implementation_sha256"] != before["compiler_implementation_sha256"]


def test_unresolved_lazy_symbol_refuses_to_issue_incomplete_identity(closure_environment):
    launcher, candidate, runtime = closure_environment
    (candidate / "compiler.py").write_text("from merlin.runtime import Tensor\n")
    (runtime / "__init__.py").write_text("_EXPORTS = build_exports()\n" + GETTER)
    with pytest.raises(ValueError, match="unresolved shared lazy import merlin.runtime.Tensor"):
        launcher.compiler_dependency_record(candidate)


def test_actual_global_plan_initializer_maps_to_actual_implementation():
    root = repo_root() / "merlin/python/merlin/xdsl_dialects/lowering"
    result = resolve_lazy_export((root / "__init__.py").read_bytes(),
                                 package="merlin.xdsl_dialects.lowering", symbol="GlobalPlan")
    assert result.module == "merlin.xdsl_dialects.lowering.global_plan"
    assert (root / "global_plan.py").is_file()


def test_actual_global_plan_from_import_enters_production_closure(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"))
    launcher = importlib.import_module("run_global_perf_experiment")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("from merlin.xdsl_dialects.lowering import GlobalPlan\n")
    record = launcher.compiler_dependency_record(candidate)
    assert "xdsl_dialects/lowering/global_plan.py" in record["shared_sources"]
    assert record["selected_lazy_exports"]["merlin.xdsl_dialects.lowering.GlobalPlan"] == (
        "merlin.xdsl_dialects.lowering.global_plan")
