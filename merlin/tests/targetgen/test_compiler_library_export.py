"""Explicit public-library export, isolated imports and fail-closed membership."""

import importlib.util
import json
import subprocess
import sys

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def exporter(monkeypatch):
    directory = repo_root() / "build_tools/scripts"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("export_compiler_library", directory / "export_compiler_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export


def selection():
    return dict(
        schema="merlin.compiler_library_selection.v1",
        review_id="test-review",
        public_modules=["merlin.runtime.packed_float"],
        empty_namespaces=["merlin", "merlin.runtime"],
        sources=[
            dict(path=f"src/merlin/runtime/{name}.py", module=f"merlin.runtime.{name}")
            for name in ("fp8_formats", "packed_float")
        ],
    )


def test_export_is_importable_without_original_namespace_or_reference(exporter, tmp_path):
    contract = exporter(selection(), repo_root(), tmp_path / "library")
    contract.verify(tmp_path / "library")
    assert contract.permits("merlin.runtime.packed_float")
    assert not contract.permits("merlin.runtime")
    script = """
import importlib.util, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from merlin.runtime.packed_float import PackedFloatLayout, pack_codes
assert pack_codes((1, 2), PackedFloatLayout("fp4_e2m1", 2, 1, "lsb-first", "little"))[0] == b"!"
assert importlib.util.find_spec("merlin.runtime.reference") is None
assert all(Path(module.__file__).is_relative_to(root) for name, module in sys.modules.items()
           if name == "merlin" or name.startswith("merlin."))
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path / "library")], check=True, cwd=tmp_path)
    (tmp_path / "library/merlin/runtime/packed_float.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        contract.verify(tmp_path / "library")


@pytest.mark.parametrize("mutation", ["escape", "module", "missing", "parent_grant"])
def test_invalid_selection_never_leaves_a_usable_library(exporter, tmp_path, mutation):
    selected = selection()
    if mutation == "escape":
        selected["sources"][0]["path"] = "../outside.py"
    elif mutation == "module":
        selected["sources"][0]["module"] = "../outside"
    elif mutation == "missing":
        selected["sources"].pop(0)
    else:
        selected["public_modules"] = ["merlin.runtime"]
    with pytest.raises(ValueError):
        exporter(selected, repo_root(), tmp_path / "library")
    assert not (tmp_path / "library").exists()


def test_authoring_library_constructs_schedules_and_outlines_without_host_driver(exporter, tmp_path):
    selected = json.loads((repo_root() / "examples/atlas/shared-library.json").read_text())
    root = tmp_path / "library"
    contract = exporter(selected, repo_root(), root)
    contract.verify(root)
    assert contract.permits("merlin.compile.scheduling.static_schedule")
    assert contract.permits("merlin.compile.scheduling.completion_schedule")
    assert contract.permits("merlin.llvmlower.floating_contractions")
    assert not contract.permits("merlin.llvmlower.floating_matmul_outline")
    script = r'''
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from merlin.compile.scheduling.static_schedule import schedule_fixed_block, materialize_timing_plan
from merlin.compile.scheduling.timed_schedule import TimedEffects
from merlin.compile.scheduling.completion_schedule import make_completion_plan, apply_completion_plan
from merlin.llvmlower.floating_contractions import outline_floating_matmuls, static_tensor_signature, callback_bridge
from merlin.sched.ir.kernel import Call, Kernel, Loop
from merlin.sched.ir.expr import Const
from merlin.compile.scheduling.compact_dense import (
    DenseLoopAdapter, DenseLoopStorage, DenseMemoryView, schedule_compact_dense,
)
from merlin.compile.scheduling.dense_tile_ir import encode_dense_tile, project_dense_tile
from merlin.transforms.dense_planning import DenseProblem, DenseResources
problem = DenseProblem(4, 8, 4, 4, 4, 4, 1, 1, 2)
storage = DenseLoopStorage(DenseMemoryView("lhs", "operands", 4, 0, 4),
                           DenseMemoryView("rhs", "operands", 4, 4, 4),
                           DenseMemoryView("acc", "sums", 8, 0, 4))
adapter = DenseLoopAdapter(encode_dense_tile, project_dense_tile, "test-config", "test-ordered", True, True, 64)
dense = schedule_compact_dense(problem, DenseResources(32, 32, 1, 1, 1), storage, adapter,
                              name="dense", dtypes=("fp8_e4m3", "fp8_e4m3", "bf16"),
                              static_budget=32, external_buffers_disjoint=True)
assert dense is not None and dense.realization["output_tiles"] == 1
dense.revalidate()
kernel = Kernel("k", (), (Call("compute", ()),))
project = lambda c: TimedEffects(c.instr, 4 if c.instr == "compute" else c.arg("cycles").value)
gap = lambda c: 1 if c.instr == "compute" else c.arg("cycles").value
options = dict(project=project, issue_gap=gap, capacities={}, model_identity="selected", clock="core")
plan = schedule_fixed_block(kernel, **options)
result = materialize_timing_plan(kernel, plan, delay=lambda n: (Call("delay", (("cycles", Const(n)),)),), **options)
assert len(result.kernel.body) == 2
async_kernel = Kernel("async", (), (Loop("i", 5, (Call("launch", (), produces="done"),)),))
completed = apply_completion_plan(async_kernel, make_completion_plan(async_kernel),
                                 wait=lambda c: Call("join", (), awaits=(c.produces,)))
assert len(completed.body[0].body) == 2
source = """module { func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
 %z = arith.constant 0.0 : f32
 %init = tensor.splat %z : tensor<2x2xf32>
 %r = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>)
       outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
 return %r : tensor<2x2xf32>
}}"""
host, calls = outline_floating_matmuls(source, entry="main", numerical_policy="explicit-test")
assert len(calls) == 1 and "merlin_matmul_0" in callback_bridge(calls)
assert static_tensor_signature(source, entry="main")["inputs_dtypes"] == ["f32", "f32"]
for missing in ("merlin.runtime.reference", "merlin.llvmlower.floating_matmul_outline", "merlin.llvmlower.pipeline"):
    assert importlib.util.find_spec(missing) is None
root = Path(sys.argv[1])
assert all(Path(module.__file__).is_relative_to(root) for name, module in sys.modules.items()
           if name == "merlin" or name.startswith("merlin."))
'''
    # Candidate imports must satisfy the real admission policy as well as work
    # in Python; package-level compatibility exports are not public leaf APIs.
    from merlin.targetgen.package_runtime import _py_imports_merlin

    assert _py_imports_merlin(script, compiler_library=contract) is None
    subprocess.run([sys.executable, "-I", "-c", script, str(root)], check=True, cwd=tmp_path)
