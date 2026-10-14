"""merlin's staged passes on a capture that calls an external kernel (model2MLIR ``prov.kernel``).

Until a backend dialect claims it, an external kernel call is a black box whose body only arrives at
link time (``llvmlower.kernel_link``, LLVM route). The staged passes must therefore carry it untouched
and never confuse it with their own outlined dispatches. Before this, on ``mlp_basic_triton``: the
outliner dropped the body-less declaration and failed ``verify()`` on the dangling call; the dispatch
program paired EVERY ``func.call`` with the next outlined dispatch, shifting each later pairing by one;
and the router reported the call as unclassified ``generic`` computation.
"""

from __future__ import annotations

from mlp_bundles import MLP_BASIC_TRITON

from merlin.common.kernel_calls import is_external_decl, is_kernel_call
from merlin.common.mlir_query import parse

QUALNAME = "triton_exp::fused_mlp2"


def _module():
    return parse((MLP_BASIC_TRITON / "model.mlir").read_text())


def test_is_kernel_call_finds_the_one_call():
    calls = [op for op in _module().walk() if is_kernel_call(op)]
    assert len(calls) == 1 and calls[0].callee.root_reference.data == "triton_exp__fused_mlp2"


def test_payload_classes_reports_kernel():
    from merlin.compile_core import payload_classes

    payload = payload_classes(_module())
    assert "kernel" in payload
    assert "matmul" in payload  # fc3 is still ordinary linalg


def test_support_ops_never_absorbs_kernel_call():
    """The call's only consumer is fc3's matmul, which made it look like an accumulator-init-style
    "support" op that materialization may drop — silently deleting fc1+relu+fc2+relu."""
    from merlin.compile_core import _MATMUL_OPS, _entry_block
    from merlin.xdsl_dialects.lowering.interface_lowering import support_ops, unaccounted_ops

    block = _entry_block(_module())
    payload = [op for op in block.ops if op.name in _MATMUL_OPS]
    assert not any(is_kernel_call(op) for op in support_ops(block, set(payload)))
    assert any(is_kernel_call(op) for op in unaccounted_ops(block, payload))


def test_choose_route_sends_kernels_to_llvm():
    from merlin.compile_core import choose_route

    route = choose_route(
        _module(),
        target="t",
        dialect_plan={"lowering": [{"op": "matmul", "to": "t.matmul"}, {"op": "elementwise", "to": "t.ew"}]},
    )
    assert route.kind == "llvm"
    assert "kernel_link" in route.reason and QUALNAME in route.reason


def test_outline_keeps_kernel_call_and_declaration():
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    res = outline_dispatches(_module())  # verify() runs inside
    decls = [op.sym_name.data for op in res.module.walk() if is_external_decl(op)]
    assert decls == ["triton_exp__fused_mlp2"]
    driver = next(op for op in res.module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    assert sum(1 for op in driver.walk() if is_kernel_call(op)) == 1
    assert all("triton_exp" not in d.symbol for d in res.dispatches)


def test_dispatch_program_pairs_after_kernel():
    from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program, verify_program
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    res = outline_dispatches(_module())
    prog = build_dispatch_program(res)
    externals = [n for n in prog.nodes if n.kind == "external"]
    assert [n.op for n in externals] == [QUALNAME]
    dispatched = [n.op for n in prog.nodes if n.kind == "dispatch"]
    assert dispatched == [d.symbol for d in res.dispatches]  # no off-by-one after the kernel
    assert verify_program(prog) == []


def test_prune_keeps_external_node():
    from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program, prune_dead_nodes
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    prog = prune_dead_nodes(build_dispatch_program(outline_dispatches(_module())))
    assert [n.op for n in prog.nodes if n.kind == "external"] == [QUALNAME]


def test_interface_lowering_names_external_kernel():
    import pytest

    from merlin.xdsl_dialects.lowering.interface_lowering import LoweringError, lower_to_interface

    with pytest.raises(LoweringError) as err:
        lower_to_interface(_module())
    assert QUALNAME in str(err.value)
