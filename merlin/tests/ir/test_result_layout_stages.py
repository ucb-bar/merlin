"""Descriptor splitting executes requested rewrites at their buffer anchor."""

import ast
import sys
from types import SimpleNamespace

import pytest

from merlin.llvmlower import result_layout
from merlin.llvmlower.fusion_guard import RUNNER_PRELUDE as FUSION_PRELUDE
from merlin.llvmlower.pipeline import build_rvv_pipeline
from merlin.llvmlower.result_buffers import descriptor_results
from merlin.llvmlower.result_layout import module_attribute
from merlin.llvmlower.selfcopy import RUNNER_PRELUDE


@pytest.mark.parametrize("rvv", [False, True])
def test_descriptor_split_runs_erase_and_mid_once(monkeypatch, rvv):
    events = []
    manager = SimpleNamespace(parse=lambda text, ctx: SimpleNamespace(run=lambda op: events.append(text)))
    monkeypatch.setitem(sys.modules, "torch_mlir.passmanager", SimpleNamespace(PassManager=manager))
    monkeypatch.setitem(sys.modules, "torch_mlir", SimpleNamespace(ir=SimpleNamespace(StringAttr=lambda a: a)))
    namespace = {"sys": SimpleNamespace(argv=[])}
    exec(RUNNER_PRELUDE, namespace)
    split = next(n for n in ast.parse(FUSION_PRELUDE).body if isinstance(n, ast.FunctionDef) and n.name == "_fg_split")
    exec(compile(ast.Module(body=[split], type_ignores=[]), "fusion_split", "exec"), namespace)
    namespace["_erase_self_copies"] = lambda module: events.append("erase") or 1
    monkeypatch.setattr(result_layout, "_fg_split", namespace["_fg_split"], raising=False)
    monkeypatch.setattr(result_layout, "_layout_original_stages", namespace["_run_stages"], raising=False)
    monkeypatch.setattr(result_layout, "_apply_input_layout", lambda m, c: events.append("input-layout"), raising=False)
    monkeypatch.setattr(result_layout, "_apply_result_layout", lambda m, c: events.append("layout"))
    pipeline = (
        build_rvv_pipeline("schedule.mlir")
        if rvv
        else (
            "one-shot-bufferize,bufferization-lower-deallocations,"
            "func.func(buffer-loop-hoisting),canonicalize,cse,convert-linalg-to-loops,finalize-memref-to-llvm"
        )
    )
    pipeline = descriptor_results(
        pipeline,
        "module"
        + module_attribute({})
        + " { func.func @forward(%a: tensor<2xf32>) -> tensor<2xf32> { return %a : tensor<2xf32> } }",
        preserve=True,
    )
    module = SimpleNamespace(
        operation=SimpleNamespace(
            attributes={"m2m.result_layout": SimpleNamespace(value='{"inputs": [], "results": []}')}
        )
    )
    import contextlib

    mid = [("regression", lambda ctx, m: events.append("mid") or 1)]
    result_layout._layout_stages(contextlib.nullcontext(), module, pipeline, True, mid)
    assert events.count("input-layout") == 1
    assert events.index("input-layout") < next(i for i, event in enumerate(events) if "one-shot-bufferize" in event)
    assert events.count("erase") == events.count("mid") == 1
    assert events.index("erase") < events.index("mid")
    llvm = next(i for i, event in enumerate(events) if "finalize-memref-to-llvm" in event)
    assert events.index("mid") < llvm
    assert (events.index("mid") < events.index("layout")) == rvv


@pytest.mark.parametrize("pipeline", ["", "canonicalize,finalize-memref-to-llvm"])
@pytest.mark.parametrize("erase,mid", [(True, ()), (False, (("missing", lambda *_: 1),))])
def test_requested_buffer_rewrites_refuse_missing_anchor(monkeypatch, pipeline, erase, mid):
    monkeypatch.setitem(sys.modules, "torch_mlir.passmanager", SimpleNamespace(PassManager=None))
    namespace = {"sys": SimpleNamespace(argv=[])}
    exec(RUNNER_PRELUDE, namespace)
    with pytest.raises(RuntimeError, match="requested"):
        namespace["_run_stages"](None, None, pipeline, erase, mid)


@pytest.mark.parametrize(
    "pipeline,missing",
    [
        ("one-shot-bufferize,buffer-results-to-out-params,finalize-memref-to-llvm", "buffer-results-to-out-params"),
        ("canonicalize,finalize-memref-to-llvm", "one-shot-bufferize"),
    ],
)
def test_pipeline_cannot_skip_declared_input_layout(monkeypatch, pipeline, missing):
    import contextlib
    import json

    events = []
    contract = dict(inputs=[dict(shape=[2, 3], stride=[6, 2], storage_offset=1)], results=[])
    module = SimpleNamespace(
        operation=SimpleNamespace(attributes={"m2m.result_layout": SimpleNamespace(value=json.dumps(contract))})
    )
    monkeypatch.setitem(sys.modules, "torch_mlir", SimpleNamespace(ir=SimpleNamespace(StringAttr=lambda a: a)))
    monkeypatch.setattr(result_layout, "_fg_split", lambda p: p.split(","), raising=False)
    monkeypatch.setattr(result_layout, "_layout_original_stages", lambda *args: events.append(args), raising=False)
    with pytest.raises(ValueError, match="cannot skip captured input-layout contract.*" + missing):
        result_layout._layout_stages(contextlib.nullcontext(), module, pipeline, False)
    assert not events
    assert "m2m.result_layout" in module.operation.attributes
