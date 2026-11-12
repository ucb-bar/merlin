import ctypes

import numpy as np
import pytest

from merlin.compiler.dialects.lowering.compute_groups import Group
from merlin.compiler.dialects.lowering.dispatch_program import build_dispatch_program
from merlin.compiler.dialects.lowering.outline import outline_dispatches
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
from merlin.compiler.lower.model import kernel_backend
from merlin.runtime import dispatch_runtime as runtime

COPY = "builtin.module {\n  func.func @forward(%x: tensor<2x3xi32>, %dst: tensor<2x3xi32>) -> tensor<2x3xi32> {\n    %y = linalg.copy ins(%x : tensor<2x3xi32>) outs(%dst : tensor<2x3xi32>) -> tensor<2x3xi32>\n    func.return %y : tensor<2x3xi32>\n  }\n}"


def _original(source=COPY, placement="selected_endpoint"):
    module = parse_mlir_text(source)
    root = next((op for op in module.walk() if op.name in ("linalg.copy", "linalg.matmul")))
    groups = (
        None
        if placement is None
        else [Group(0, placement, root, [root], ["movement" if root.name == "linalg.copy" else "contraction"])]
    )
    outlined = outline_dispatches(module, groups=groups)
    build_dispatch_program(outlined)
    return outlined


def _copy_inputs():
    x = np.array([[-17, 31, 51], [109, -8, 4]], np.int32)
    return [x, np.zeros_like(x)]


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


@pytest.mark.parametrize("backend", [None])
def test_explicit_nonhost_intent_refuses_before_any_host_or_backend_selection(tmp_path, monkeypatch, backend):
    outlined = _original()
    calls = _host(monkeypatch)
    counters = {}
    with pytest.raises(runtime.DispatchRuntimeError, match="explicitly requires placement.*host execution"):
        runtime.execute(outlined, _copy_inputs(), tmp_path, kernel_backend=backend, counters=counters)
    assert calls == [] and (not counters.get("dispatch_ledger"))
