"""The host dispatch runtimes on a model that calls an external kernel (model2MLIR ``prov.kernel``).

These runtimes execute merlin's OWN outlined dispatches; an external kernel's body is only linked on
the LLVM route (``llvmlower.kernel_link``). They must refuse it by name. Before this, the dispatch
runtime treated the call as an outlined kernel and failed inside kernel extraction, and the program
interpreter would have handed it to ``eval_view`` as if it were glue.
"""

from __future__ import annotations

import numpy as np
import pytest
from mlp_bundles import MLP_BASIC_TRITON, forward_args

from merlin.common.mlir_query import parse
from merlin.llvmlower import toolchain
from merlin.runtime.dispatch_runtime import ExternalKernelNotExecutable
from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program
from merlin.xdsl_dialects.lowering.outline import outline_dispatches

QUALNAME = "triton_exp::fused_mlp2"


def _program():
    return build_dispatch_program(outline_dispatches(parse((MLP_BASIC_TRITON / "model.mlir").read_text())))


@pytest.mark.skipif(not toolchain.available(), reason="clang-23 / model2MLIR venv not available")
def test_execute_refuses_external_kernel(tmp_path):
    from merlin.runtime.dispatch_runtime import execute

    outlined = outline_dispatches(parse((MLP_BASIC_TRITON / "model.mlir").read_text()))
    with pytest.raises(ExternalKernelNotExecutable) as err:
        execute(outlined, forward_args(MLP_BASIC_TRITON), tmp_path)
    assert err.value.qualname == QUALNAME
    assert "kernel_link" in str(err.value)


def test_interpreter_refuses_external_node():
    from merlin.runtime.interpret import run_dispatch_program

    prog = _program()
    inputs = {b.id: np.zeros(b.shape, dtype=np.float32) for b in prog.buffers.values() if b.kind == "arg"}
    with pytest.raises(ExternalKernelNotExecutable) as err:
        run_dispatch_program(
            prog, inputs, invoke_kernel=lambda sym, vals: [vals[0]], eval_view=lambda op, vals, node: [np.zeros(1)]
        )
    assert err.value.qualname == QUALNAME


def test_program_names_external_opcode():
    from merlin.runtime.program import build_program

    prog = build_program(_program())
    assert f"external:{QUALNAME}" in prog.opcodes
    assert f"dispatch:{QUALNAME}" not in prog.opcodes
    assert QUALNAME not in prog.kernels  # not a merlin-compiled kernel
