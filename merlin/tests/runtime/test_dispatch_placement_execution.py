"""Original placement refuses host replacement at the ordinary runtime caller.

All executors here are diagnostic Python controls. Complete synthetic values
prove routing and refusal behavior, never compiler or hardware semantics.
"""

import ctypes
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import kernel_backend
from merlin.runtime import dispatch_runtime as runtime
from merlin.xdsl_dialects.lowering.compute_groups import Group
from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program
from merlin.xdsl_dialects.lowering.outline import outline_dispatches

COPY = """builtin.module {
  func.func @forward(%x: tensor<2x3xi32>, %dst: tensor<2x3xi32>) -> tensor<2x3xi32> {
    %y = linalg.copy ins(%x : tensor<2x3xi32>) outs(%dst : tensor<2x3xi32>) -> tensor<2x3xi32>
    func.return %y : tensor<2x3xi32>
  }
}"""
MATMUL = """builtin.module {
  func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>, %init: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %y = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x2xf32>)
        outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %y : tensor<2x2xf32>
  }
}"""


def _original(source=COPY, placement="selected_endpoint"):
    module = parse_mlir_text(source)
    root = next(op for op in module.walk() if op.name in ("linalg.copy", "linalg.matmul"))
    groups = (
        None
        if placement is None
        else [Group(0, placement, root, [root], ["movement" if root.name == "linalg.copy" else "contraction"])]
    )
    outlined = outline_dispatches(module, groups=groups)
    build_dispatch_program(outlined)
    return outlined


def _call(outlined):
    driver = next(op for op in outlined.module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    return next(op for op in driver.walk() if op.name == "func.call")


def _function(outlined):
    return next(op for op in outlined.module.walk() if op.name == "func.func" and "$kernel_" in op.sym_name.data)


def _copy_inputs():
    x = np.array([[-17, 31, 51], [109, -8, 4]], np.int32)
    return [x, np.zeros_like(x)]


def _matmul_inputs():
    a = np.array([[1, -2, 3], [-4, 5, 6]], np.float32)
    b = np.array([[7, 8], [-9, 10], [11, -12]], np.float32)
    return [a, b, np.zeros((2, 2), np.float32)]


def _host(monkeypatch, *, matmul=False, mutate=None):
    calls = []

    class DiagnosticHost:
        scalar_result_dtype = None

        def __call__(self, args):
            calls.append("host")
            if matmul:
                a = np.ctypeslib.as_array((ctypes.c_float * 6).from_address(args[0][0])).reshape(2, 3)
                b = np.ctypeslib.as_array((ctypes.c_float * 6).from_address(args[1][0])).reshape(3, 2)
                expected = np.ascontiguousarray(a @ b)
                ctypes.memmove(args[-1][0], expected.ctypes.data, expected.nbytes)
            else:
                ctypes.memmove(args[-1][0], args[0][0], 6 * np.dtype("int32").itemsize)
            if mutate is not None:
                mutate()

    monkeypatch.setattr(kernel_backend, "compile_host", lambda *_a, **_kw: DiagnosticHost())
    return calls


def _mesh(monkeypatch, *, output=False, mutate=None):
    # A fixture transport, not independently admitted target facts or runtime.
    import merlin.compile_cli as compiler

    calls = []
    monkeypatch.setattr(
        runtime,
        "mesh_datapath",
        lambda *_a, **_kw: SimpleNamespace(
            operand_dtype="float32",
            accum_dtype="float32",
            subnormal_operand_flush=False,
            integer=False,
            mlir_dtype=lambda _dt: "f32",
        ),
    )

    def run(_target, a, b, **kw):
        calls.append("diagnostic_device")
        if mutate is not None:
            mutate()
        if output:
            return (np.asarray(a, np.float32) @ np.asarray(b, np.float32)).tolist()
        kw["observed"]["decline"] = "independent controlled device decline"
        return None

    monkeypatch.setattr(compiler, "run_matmul_on_mesh", run)
    return calls


@pytest.mark.parametrize("backend", [None, "xnnpack", "other_host_backend"])
def test_explicit_nonhost_intent_refuses_before_any_host_or_backend_selection(tmp_path, monkeypatch, backend):
    outlined = _original()
    calls = _host(monkeypatch)
    counters = {}
    with pytest.raises(runtime.DispatchRuntimeError, match="explicitly requires placement.*host execution"):
        runtime.execute(outlined, _copy_inputs(), tmp_path, kernel_backend=backend, counters=counters)
    assert calls == [] and not counters.get("dispatch_ledger")


@pytest.mark.parametrize("placement", [None, "host"])
def test_unselected_and_original_host_keep_complete_values_and_order(tmp_path, monkeypatch, placement):
    outlined = _original(placement=placement)
    inputs = _copy_inputs()
    calls = _host(monkeypatch)
    counters = {}
    result = runtime.execute(outlined, inputs, tmp_path, counters=counters)
    assert calls == ["host"] and len(result) == 1
    np.testing.assert_array_equal(result[0], inputs[0])
    assert result[0].shape == (2, 3) and result[0].dtype == np.dtype("int32")
    assert counters["dispatch_ledger"][0]["selected_placement"] == placement
    assert counters["dispatch_ledger"][0]["lane"] == "native_cpu"


@pytest.mark.parametrize("placement", [None, "host", "selected_endpoint"])
def test_device_decline_preserves_fallback_only_when_original_placement_allows_host(tmp_path, monkeypatch, placement):
    outlined = _original(MATMUL, placement)
    device = _mesh(monkeypatch)
    host = _host(monkeypatch, matmul=True)
    counters = {}
    inputs = _matmul_inputs()
    if placement == "selected_endpoint":
        with pytest.raises(runtime.DispatchRuntimeError, match="explicitly requires placement.*host execution"):
            runtime.execute(
                outlined, inputs, tmp_path, kernel_backend="mesh", mesh_target="synthetic", counters=counters
            )
        assert device == ["diagnostic_device"] and host == []
        assert counters["dispatch_ledger"] == []
        assert counters["mesh_fell_back"] == 0 and counters["mesh_ran"] == 0
    else:
        result = runtime.execute(
            outlined, inputs, tmp_path, kernel_backend="mesh", mesh_target="synthetic", counters=counters
        )
        np.testing.assert_array_equal(result[0], inputs[0] @ inputs[1])
        assert host == ["host"] and device == ([] if placement == "host" else ["diagnostic_device"])
        assert counters["dispatch_ledger"][0]["selected_placement"] == placement
        assert counters["dispatch_ledger"][0]["placement"] == "host"


def test_diagnostic_device_result_stays_distinct_from_a_hardware_event_proof(tmp_path, monkeypatch):
    outlined = _original(MATMUL)
    device = _mesh(monkeypatch, output=True)
    host = _host(monkeypatch, matmul=True)
    counters = {}
    inputs = _matmul_inputs()
    result = runtime.execute(
        outlined, inputs, tmp_path, kernel_backend="mesh", mesh_target="synthetic", counters=counters
    )
    np.testing.assert_array_equal(result[0], inputs[0] @ inputs[1])
    assert device == ["diagnostic_device"] and host == []
    row = counters["dispatch_ledger"][0]
    assert row["selected_placement"] == "selected_endpoint" and row["lane"] == "on_mesh"
    assert row["oracle_evidence"] is None and row["trace_check"] is None


def test_cache_does_not_compile_a_host_substitute_for_selected_device_work(tmp_path, monkeypatch):
    outlined = _original(MATMUL)
    device = _mesh(monkeypatch, output=True)

    def unexpected_build(*_a, **_kw):
        pytest.fail("explicit device route must not precompile a host substitute")

    monkeypatch.setattr(runtime, "_compile_kernel_so", unexpected_build)
    monkeypatch.setattr(kernel_backend, "compile_host", unexpected_build)
    result = runtime.execute(
        outlined,
        _matmul_inputs(),
        tmp_path / "ordinary",
        cache_dir=tmp_path / "cache",
        kernel_backend="mesh",
        mesh_target="synthetic",
    )
    assert device == ["diagnostic_device"]
    np.testing.assert_array_equal(result[0], _matmul_inputs()[0] @ _matmul_inputs()[1])
    assert list((tmp_path / "cache").iterdir()) == []


def test_original_host_placement_without_a_group_cannot_be_promoted_by_mesh_classifier(tmp_path, monkeypatch):
    outlined = _original(MATMUL, "host")
    outlined.dispatches[0].group = None
    _function(outlined).attributes.pop("merlin.group")
    build_dispatch_program(outlined)
    device = _mesh(monkeypatch, output=True)
    host = _host(monkeypatch, matmul=True)
    inputs = _matmul_inputs()
    counters = {}
    result = runtime.execute(
        outlined, inputs, tmp_path, kernel_backend="mesh", mesh_target="synthetic", counters=counters
    )
    np.testing.assert_array_equal(result[0], inputs[0] @ inputs[1])
    assert device == [] and host == ["host"]
    assert counters["dispatch_ledger"][0]["selected_placement"] == "host"


@pytest.mark.parametrize("placement", [None, "host"])
def test_original_host_and_unselected_xnnpack_lane_remain_usable(tmp_path, monkeypatch, placement):
    from merlin.runtime.backends import xnnpack_host

    outlined = _original(MATMUL, placement)
    invocations = []
    monkeypatch.setattr(xnnpack_host, "is_available", lambda: True)
    monkeypatch.setattr(xnnpack_host, "classify_matmul_kernel", lambda _fn: {"a": 0, "b": 1})

    def gemm(a, b):
        invocations.append("diagnostic_xnnpack")
        return a @ b

    monkeypatch.setattr(xnnpack_host, "gemm_f32", gemm)
    host = _host(monkeypatch, matmul=True)
    counters = {}
    inputs = _matmul_inputs()
    result = runtime.execute(outlined, inputs, tmp_path, kernel_backend="xnnpack", counters=counters)
    np.testing.assert_array_equal(result[0], inputs[0] @ inputs[1])
    assert invocations == ["diagnostic_xnnpack"] and host == []
    row = counters["dispatch_ledger"][0]
    assert row["selected_placement"] == placement and row["placement"] == "host"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "callee"])
def test_original_call_requires_the_actual_selected_defined_function(tmp_path, monkeypatch, mutation):
    outlined = _original(placement="host")
    function = _function(outlined)
    if mutation == "missing":
        outlined.module.body.block.erase_op(function)
    elif mutation == "duplicate":
        outlined.module.body.block.add_op(function.clone())
    else:
        from xdsl.dialects.builtin import SymbolRefAttr

        _call(outlined).properties["callee"] = SymbolRefAttr("different_endpoint")
    host = _host(monkeypatch)
    with pytest.raises(runtime.DispatchRuntimeError):
        runtime.execute(outlined, _copy_inputs(), tmp_path)
    assert host == []


@pytest.mark.parametrize(
    "mutation",
    ["absent", "extra", "symbol", "index", "bool_index", "operands", "results", "placement", "bool_group"],
)
def test_changed_or_incomplete_original_table_refuses_before_compilation(tmp_path, monkeypatch, mutation):
    outlined = _original(placement="host")
    row = outlined.dispatches[0]
    if mutation == "absent":
        outlined.dispatches.clear()
    elif mutation == "extra":
        outlined.dispatches.append(row)
    else:
        changes = {
            "symbol": {"symbol": "different"},
            "index": {"index": 1},
            "bool_index": {"index": False},
            "operands": {"n_operands": 1},
            "results": {"result_types": ["tensor<3x2xi32>"]},
            "placement": {"placement": None},
            "bool_group": {"group": False},
        }
        outlined.dispatches[0] = replace(row, **changes[mutation])
    host = _host(monkeypatch)
    with pytest.raises(runtime.DispatchRuntimeError):
        runtime.execute(outlined, _copy_inputs(), tmp_path)
    assert host == []


@pytest.mark.parametrize("owner", ["function_attributes", "function_properties", "call_attributes", "call_properties"])
@pytest.mark.parametrize("value", [StringAttr("different_endpoint"), IntegerAttr(0, IntegerType(64))])
def test_conflicting_actual_call_or_function_ownership_refuses(tmp_path, monkeypatch, owner, value):
    outlined = _original(placement="host")
    operation = _function(outlined) if owner.startswith("function") else _call(outlined)
    getattr(operation, owner.split("_")[1])["merlin.placement"] = value
    host = _host(monkeypatch)
    with pytest.raises(runtime.DispatchRuntimeError, match="placement.*actual function|placement.*actual call"):
        runtime.execute(outlined, _copy_inputs(), tmp_path)
    assert host == []


@pytest.mark.parametrize("stage", ["host", "device"])
@pytest.mark.parametrize("owner", ["table", "function", "call"])
def test_actual_executor_cannot_replace_original_selection_before_a_success_ledger(tmp_path, monkeypatch, stage, owner):
    outlined = _original(MATMUL, "host" if stage == "host" else "selected_endpoint")

    def mutate():
        if owner == "table":
            outlined.dispatches[0].placement = None
        elif owner == "function":
            _function(outlined).attributes["merlin.placement"] = StringAttr("different_endpoint")
        else:
            _call(outlined).properties["merlin.placement"] = StringAttr("different_endpoint")

    host = _host(monkeypatch, matmul=True, mutate=mutate if stage == "host" else None)
    device = _mesh(monkeypatch, output=True, mutate=mutate if stage == "device" else None)
    counters = {}
    with pytest.raises(runtime.DispatchRuntimeError):
        runtime.execute(
            outlined, _matmul_inputs(), tmp_path, kernel_backend="mesh", mesh_target="synthetic", counters=counters
        )
    assert host == (["host"] if stage == "host" else [])
    assert device == (["diagnostic_device"] if stage == "device" else [])
    assert counters["dispatch_ledger"] == []
