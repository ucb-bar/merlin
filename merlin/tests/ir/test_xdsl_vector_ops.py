"""Non-matmul ops of a whole model — residual adds, gating multiplies, and pointwise
activations — lower through the SAME staged pipeline as matmuls and execute correctly.

A whole model is not matmuls alone: between the systolic layers run elementwise combines and
activations on the target's vector/scalar lanes. This proves ``build_vector_block``
(``combine(relu(A@W1), A@W2)``) descends input -> contract -> schedule -> interface -> target ->
runtime -> command buffer, emitting the runtime VECTOR_MAP command the engine already models, and
that the result is numerically exact vs numpy on both reference targets. The relu idiom
``max(x, 0)`` is recognized structurally (no regex); a max of two real tensors — which the engine's
vector path does not model — fails closed.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def _lower(combine="add", relu=True, target="toy_npu"):
    from merlin.xdsl_dialects.lowering import lower_module
    from merlin.xdsl_dialects.lowering.input_workload import build_vector_block

    return lower_module(build_vector_block(m=8, k=16, elem="f32", combine=combine, relu=relu),
                        target=target)


def _numpy_ref(A, W1, W2, combine, relu):
    h1 = A @ W1
    if relu:
        h1 = np.maximum(h1, 0)
    h2 = A @ W2
    return h1 + h2 if combine == "add" else h1 * h2


def _run(res):
    from merlin.xdsl_dialects.lowering import execute

    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 16)).astype(np.float32)
    W1 = rng.standard_normal((16, 16)).astype(np.float32)
    W2 = rng.standard_normal((16, 16)).astype(np.float32)
    run = execute(res, {"A0": A.tolist(), "W": W1.tolist(), "W1": W2.tolist()})
    got = np.array(next(iter(run["outputs"].values())), dtype=np.float32)
    return run, got, A, W1, W2


def test_vector_block_every_stage_verifies():
    res = _lower(combine="add", relu=True)
    for mod in res.modules():
        mod.verify()


@pytest.mark.parametrize("combine", ["add", "mul"])
@pytest.mark.parametrize("relu", [True, False])
def test_vector_block_matches_numpy(combine, relu):
    res = _lower(combine=combine, relu=relu)
    run, got, A, W1, W2 = _run(res)
    assert run["correct"] is True
    assert np.allclose(got, _numpy_ref(A, W1, W2, combine, relu), rtol=1e-4, atol=1e-3)


def test_command_buffer_emits_vector_map():
    """The elementwise combine + activation surface as runtime VECTOR_MAP commands, and only the
    final vector result is a declared output (the two matmul commits are intermediates)."""
    res = _lower(combine="add", relu=True)
    cb = res.command_buffer
    vmaps = [c for c in cb["commands"] if c["opcode"] == "VECTOR_MAP"]
    assert len(vmaps) == 2  # the relu (identity) and the residual add
    combines = {c["attributes"]["combine"] for c in vmaps}
    assert combines == {"identity", "add"}
    relu_cmd = next(c for c in vmaps if c["attributes"]["combine"] == "identity")
    assert relu_cmd["attributes"]["activation"] == ["relu"]
    # The final vector_map output is the model result; matmul commits are intermediates.
    assert cb.get("outputs") == ["V1"]
    assert cb["tensors"]["V1"]["role"] == "output"


def test_relu_off_has_one_vector_map():
    """Without relu only the elementwise combine is a vector op (no identity+relu command)."""
    cb = _lower(combine="mul", relu=False).command_buffer
    vmaps = [c for c in cb["commands"] if c["opcode"] == "VECTOR_MAP"]
    assert len(vmaps) == 1
    assert vmaps[0]["attributes"]["combine"] == "mul"
    assert cb.get("outputs") == ["V0"]


def test_vector_block_runs_on_saturn():
    res = _lower(combine="add", relu=True, target="saturn")
    for mod in res.modules():
        mod.verify()
    run, got, A, W1, W2 = _run(res)
    assert run["correct"] is True
    assert np.allclose(got, _numpy_ref(A, W1, W2, "add", True), rtol=1e-4, atol=1e-3)


def test_max_of_two_tensors_fails_closed():
    """A linalg.max of two real tensors is NOT relu; the engine's vector path does not model a
    two-tensor max, so lowering fails closed rather than silently miscompiling."""
    from xdsl.ir import Block, Region
    from xdsl.dialects import tensor as td
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType, f32
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as lo

    from merlin.xdsl_dialects.lowering import lower_module
    from merlin.xdsl_dialects.lowering.interface_lowering import LoweringError

    t = TensorType(f32, [4, 4])
    blk = Block(arg_types=[t, t])
    a, b = blk.args
    e1, e2 = td.EmptyOp((), t), td.EmptyOp((), t)
    mm = lo.MatmulOp(inputs=(a, a), outputs=(e1.tensor,), res=(t,))
    mx = lo.MaxOp(inputs=(mm.results[0], b), outputs=(e2.tensor,), res=(t,))
    blk.add_ops([e1, mm, e2, mx, ReturnOp(mx.results[0])])
    fn = FuncOp("f", FunctionType.from_lists([t, t], [t]), Region([blk]))
    with pytest.raises(LoweringError, match="does not model vector op"):
        lower_module(ModuleOp([fn]))
