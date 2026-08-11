"""A target can declare an accelerated elementwise op that Merlin has no way to use.

`interface.py` registers ResidentPack / ResidentEvict / Matmul / AccumulatorCreate / Accumulate /
Commit / AsyncCopy / Await / Fifo* / CommandRegion. There is no `interface.elementwise`, and
`lower_to_interface` materializes matmuls only. So a dialect plan declaring `<target>.elementwise` —
several generated plans do, because their compute units genuinely support it — describes a capability
the compiler cannot reach, and the payload goes down the generic LLVM path instead.

This file exists to settle *whose* gap that is, because the answer decides where the fix goes. The
project's criterion is whether an equivalent non-Triton program hits the same wall. It does: a
hand-authored `linalg.add` is routed identically. So this is a Merlin interface-abstraction gap and
the fix is an `interface.elementwise` every frontend benefits from — never a Triton special case.

Closing it is deliberately not attempted here. It would add an op to the interface dialect, a
materialization path, a target-lowering rule, a runtime opcode, a command-buffer opcode, and matching
simulator and reference semantics — changes running straight through the RTL-certified Gemmini path.
The target that motivates it is Radiance, whose SIMT lowering is not present on this branch, so there
is no way to demonstrate the fix end to end on the hardware that needs it. Evidence first.
"""
from __future__ import annotations

import pytest
import triton_kernels as K

from merlin import compile_core
from merlin.common.paths import repo_root
from merlin.triton import source
from merlin.triton.bridge import to_linalg

GEMMINI_PACKAGE = repo_root() / "out/artifacts/targets/gemmini/hand_v0"

pytestmark = pytest.mark.skipif(not K.HAS_TRITON, reason="the `triton` optional extra is not installed")


def hand_written_add(n: int = 1024):
    """The same computation as the Triton vector add, authored directly in linalg."""
    from xdsl.dialects import tensor as tensor_d
    from xdsl.dialects.builtin import Float32Type, FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as linalg_ops
    from xdsl.ir import Block, Region

    t = TensorType(Float32Type(), [n])
    block = Block(arg_types=[t, t])
    empty = tensor_d.EmptyOp((), t)
    add = linalg_ops.AddOp(inputs=(block.args[0], block.args[1]), outputs=(empty.tensor,), res=(t,))
    block.add_ops([empty, add, ReturnOp(add.results[0])])
    return ModuleOp([FuncOp("add", FunctionType.from_lists([t, t], [t]), Region([block]))])


def elementwise_declaring_targets():
    """Targets whose committed dialect plan claims an accelerated elementwise op."""
    from merlin.targetgen import capability_manifests as cm

    out = []
    for name in ("radiance",):
        try:
            plan = cm.dialect_plan_from_manifest(cm.manifest_for(name))
        except Exception:  # pragma: no cover - manifest absent in a trimmed checkout
            continue
        if "elementwise" in compile_core.plan_interface_ops(plan):
            out.append((name, plan))
    return out


def test_at_least_one_target_declares_elementwise():
    assert elementwise_declaring_targets(), "nothing declares elementwise — the gap is not testable"


def test_a_declared_elementwise_op_is_not_interface_buildable():
    """The gap itself: declared coverage that the interface layer cannot materialize."""
    from merlin.xdsl_dialects import interface

    registered = {op.name for op in interface.INTERFACE_DIALECT.operations}
    assert not any(name.endswith(".elementwise") for name in registered), (
        f"interface.elementwise now exists — this gap is closed and the test should become a "
        f"materialization test: {sorted(registered)}")


@pytest.mark.parametrize("name,_plan", elementwise_declaring_targets())
def test_the_triton_add_is_routed_to_the_generic_path_with_the_reason_stated(name, _plan):
    """Not silent: the route names the declared coverage and what the interface can build."""
    spec = K.vector_add_spec(n=1024)
    bridged = to_linalg(source.make_ttir(spec), spec)
    route = compile_core.choose_route(bridged.module, target=name, dialect_plan=_plan)
    assert route.kind == "llvm"
    assert "elementwise" in route.covered
    assert "elementwise" not in route.materializable
    assert "elementwise" in route.reason or "generic" in route.reason


@pytest.mark.parametrize("name,_plan", elementwise_declaring_targets())
def test_a_hand_written_linalg_add_is_routed_identically(name, _plan):
    """The decisive comparison: no Triton involved, same wall, so the fix is not a Triton fix."""
    spec = K.vector_add_spec(n=1024)
    triton_route = compile_core.choose_route(
        to_linalg(source.make_ttir(spec), spec).module, target=name, dialect_plan=_plan)
    hand_route = compile_core.choose_route(hand_written_add(), target=name, dialect_plan=_plan)
    assert hand_route.kind == triton_route.kind == "llvm"
    assert hand_route.payload == triton_route.payload
    assert hand_route.reason == triton_route.reason


def test_the_same_holds_on_a_target_that_does_not_declare_elementwise():
    """Sanity: an accelerator with no elementwise claim behaves the same, for the same reason."""
    from merlin.targetgen.registry import load_target

    if not GEMMINI_PACKAGE.is_dir():
        pytest.skip("gemmini target package not present")
    package = load_target(GEMMINI_PACKAGE)
    spec = K.vector_add_spec(n=1024)
    triton_route = compile_core.choose_route(
        to_linalg(source.make_ttir(spec), spec).module, target_package=package)
    hand_route = compile_core.choose_route(hand_written_add(), target_package=package)
    assert triton_route.kind == hand_route.kind == "llvm"
    assert triton_route.reason == hand_route.reason
