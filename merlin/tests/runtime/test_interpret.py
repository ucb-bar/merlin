"""Host interpreter for a dispatch program + running a sliced SECTION standalone (C8).

Uses numpy as the injected kernel — the mechanism (DAG walk, boundary-input binding, result
collection) is what is under test, independent of the real compiled-kernel backend.
"""
from __future__ import annotations

import numpy as np

from merlin.runtime.interpret import run_dispatch_program
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node, slice_program


def _matmul_kernel(symbol, ins):
    a, b = ins
    return [a @ b]


def _chain_program() -> DispatchProgram:
    """x[2,3] @ w0[3,4] -> h[2,4] @ w1[4,5] -> y[2,5]; two provenanced dispatch nodes."""
    buffers = {
        "b0": Buffer(id="b0", shape=[2, 3], dtype="f32", kind="arg", arg_index=0),   # x
        "b1": Buffer(id="b1", shape=[3, 4], dtype="f32", kind="arg", arg_index=1),   # w0
        "b2": Buffer(id="b2", shape=[4, 5], dtype="f32", kind="arg", arg_index=2),   # w1
        "b3": Buffer(id="b3", shape=[2, 4], dtype="f32", kind="intermediate"),       # h
        "b4": Buffer(id="b4", shape=[2, 5], dtype="f32", kind="intermediate"),       # y
    }
    nodes = [
        Node(kind="dispatch", op="forward$kernel_0__rmatmul_0", inputs=["b0", "b1"],
             outputs=["b3"], prov={"prov.region_id": "matmul_0", "prov.fqn": "layers.0.attn.q"}),
        Node(kind="dispatch", op="forward$kernel_1__rmatmul_1", inputs=["b3", "b2"],
             outputs=["b4"], prov={"prov.region_id": "matmul_1", "prov.fqn": "layers.0.mlp.g"}),
    ]
    return DispatchProgram(entry="forward", args=[0, 1, 2], buffers=buffers, nodes=nodes,
                           results=["b4"])


def test_interpret_whole_program():
    rng = np.random.default_rng(0)
    x, w0, w1 = rng.standard_normal((2, 3)), rng.standard_normal((3, 4)), rng.standard_normal((4, 5))
    out = run_dispatch_program(_chain_program(), {"b0": x, "b1": w0, "b2": w1},
                               invoke_kernel=_matmul_kernel)
    np.testing.assert_allclose(out["b4"], x @ w0 @ w1)


def test_interpret_missing_input_fails_closed():
    import pytest
    with pytest.raises(KeyError):
        run_dispatch_program(_chain_program(), {"b0": np.zeros((2, 3))},  # w0/w1 missing
                             invoke_kernel=_matmul_kernel)


def test_run_a_sliced_section_standalone_with_boundary_input():
    """Slice out the second region and run it in isolation: its boundary input (the first region's
    output) is fed in exactly as region_goldens would supply it; the section output matches the
    whole-model intermediate — i.e. we profiled just that layer without running the rest."""
    rng = np.random.default_rng(1)
    x, w0, w1 = rng.standard_normal((2, 3)), rng.standard_normal((3, 4)), rng.standard_normal((4, 5))
    prog = _chain_program()

    whole = run_dispatch_program(prog, {"b0": x, "b1": w0, "b2": w1}, invoke_kernel=_matmul_kernel)
    h_boundary = x @ w0                                   # the region-boundary tensor (region_goldens)

    section = slice_program(prog, {"matmul_1"})
    # the slice's boundary inputs are its arg buffers; bind them from the boundary tensors.
    inputs = {}
    for b in section.buffers.values():
        if b.kind == "arg":
            inputs[b.id] = h_boundary if b.id == "b3" else w1
    sec_out = run_dispatch_program(section, inputs, invoke_kernel=_matmul_kernel)
    # the section, run alone, reproduces the whole-model output for that region.
    np.testing.assert_allclose(sec_out[section.results[0]], whole["b4"])
