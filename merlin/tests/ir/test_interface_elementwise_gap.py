"""``interface.elementwise``: the gap, and the evidence that decided where to close it.

For a while `interface.py` registered ResidentPack / ResidentEvict / Matmul / AccumulatorCreate /
Accumulate / Commit / AsyncCopy / Await / Fifo* / CommandRegion and nothing else, while several
generated dialect plans declared coverage for an *elementwise* op — a capability the compiler had no
way to reach, so the payload went down the generic path instead.

Whose gap that was decided where the fix went, and the project's criterion is whether an equivalent
non-Triton program hits the same wall. It did: a hand-authored `linalg.add` was routed identically to
the Triton one. So this was a Merlin interface-abstraction gap, not a Triton gap, and the fix is an
`interface.elementwise` every frontend benefits from rather than a Triton special case.

It is now closed, and this file holds it closed from both sides. The routing tests below still assert
the thing that matters most — that coverage is read per target, so a target whose plan does not
declare elementwise still routes a vector add to the generic path — and the materialization tests
assert the accelerated path actually works for one that does.

Two boundaries were deliberately NOT crossed:

* `linalg.sub` is not expressible. The runtime's VECTOR_MAP implements add and mul; lowering a
  subtract as an add would be a miscompile, so it fails closed.
* A fused matmul-plus-elementwise payload was refused here while the question "does the combine fold
  into the commit epilogue, or become its own dispatch?" was open. It is now ANSWERED, in the
  conservative direction: its own dispatch. The rebuild loop emits the combine as an
  ``interface.vector_map`` threaded through the same value map as the matmuls, and the commit's
  epilogue stays empty — nothing is fused by accident, which is what the refusal was protecting.
  Refusing outright is what made a residual add between two matmul layers — a whole model's actual
  shape — unlowerable, and with it the whole-model-on-mesh and multi-layer-chain tests.
"""
from __future__ import annotations

import numpy as np
import pytest
import triton_kernels as K

from merlin import compile_core
from merlin.common.paths import repo_root
from merlin.triton import source
from merlin.triton.bridge import to_linalg

GEMMINI_PACKAGE = repo_root() / "out/artifacts/targets/gemmini/hand_v0"
RADIANCE_PACKAGE = repo_root() / "out/artifacts/targets/radiance/hand_v0"

pytestmark = pytest.mark.skipif(not K.HAS_TRITON, reason="the `triton` optional extra is not installed")


def _package(path):
    from merlin.targetgen.registry import load_target

    if not path.is_dir():
        pytest.skip(f"target package not present: {path}")
    return load_target(path)


def hand_written(kind: str = "add", n: int = 256):
    """The same computation as the Triton kernel, authored directly in linalg."""
    from xdsl.dialects import tensor as tensor_d
    from xdsl.dialects.builtin import FunctionType, IntegerType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as linalg_ops
    from xdsl.ir import Block, Region

    t = TensorType(IntegerType(32), [n])
    block = Block(arg_types=[t, t])
    empty = tensor_d.EmptyOp((), t)
    cls = {"add": linalg_ops.AddOp, "mul": linalg_ops.MulOp, "sub": linalg_ops.SubOp}[kind]
    combine = cls(inputs=(block.args[0], block.args[1]), outputs=(empty.tensor,), res=(t,))
    block.add_ops([empty, combine, ReturnOp(combine.results[0])])
    return ModuleOp([FuncOp("combine", FunctionType.from_lists([t, t], [t]), Region([block]))])


# --------------------------------------------------------- the op exists, and matches the runtime


def test_the_interface_dialect_now_registers_elementwise():
    from merlin.xdsl_dialects import interface

    registered = {op.name for op in interface.INTERFACE_DIALECT.operations}
    assert "interface.elementwise" in registered


def _vector_map_buffer(combine: str, n: int = 4) -> dict:
    return {
        "abi_version": "0.1", "target": "toy_npu", "backend": "simulator",
        "tensors": {"A0": {"shape": [1, n], "dtype": "i32", "role": "input"},
                    "A1": {"shape": [1, n], "dtype": "i32", "role": "input"},
                    "Y0": {"shape": [1, n], "dtype": "i32", "role": "output"}},
        "commands": [{"opcode": "VECTOR_MAP",
                      "operands": {"lhs": "A0", "rhs": "A1", "dst": "Y0"},
                      "attributes": {"combine": combine}}],
    }


def test_every_accepted_combine_actually_executes():
    """Checked by RUNNING each one, not by reading the runtime's source.

    Accepting a combine the runtime cannot perform would let a module verify at the interface tier
    and then compute nothing — the interface dialect's accepted set has to be answerable to what the
    engine does, so this asks the engine.
    """
    from merlin.runtime import reference_outputs, simulate
    from merlin.xdsl_dialects import interface

    for combine in sorted(interface.KNOWN_COMBINES):
        cb = _vector_map_buffer(combine)
        outputs = simulate(cb)["outputs"]
        assert outputs.get("Y0"), f"VECTOR_MAP combine {combine!r} produced no output"
        assert outputs == reference_outputs(cb), combine


def test_a_combine_outside_the_accepted_set_is_not_silently_accepted():
    """The engine must not treat an unknown combine as a default (it would look like a success)."""
    from merlin.runtime import simulate
    from merlin.xdsl_dialects import interface

    assert "xor" not in interface.KNOWN_COMBINES
    outputs = simulate(_vector_map_buffer("xor"))["outputs"]
    add = simulate(_vector_map_buffer("add"))["outputs"]
    assert outputs.get("Y0") != add.get("Y0") or outputs.get("Y0") is None, (
        "an unknown combine silently behaved like `add`")


def test_a_combine_the_runtime_cannot_do_is_refused():
    """`linalg.sub` has no VECTOR_MAP equivalent, so it must not be silently lowered as an add."""
    from merlin.xdsl_dialects.lowering.interface_lowering import LoweringError, lower_to_interface

    module = hand_written("sub")
    assert compile_core.payload_classes(module) == ("generic",), (
        "linalg.sub is being treated as a materializable elementwise payload")
    with pytest.raises(LoweringError):
        lower_to_interface(module)


# ------------------------------------------------------------- coverage is still read per target


@pytest.mark.parametrize("path", [GEMMINI_PACKAGE])
def test_a_target_whose_plan_omits_elementwise_still_routes_to_the_generic_path(path):
    package = _package(path)
    spec = K.vector_add_i32_spec()
    route = compile_core.choose_route(
        to_linalg(source.make_ttir(spec), spec).module, target_package=package)
    assert route.kind == "llvm"
    assert "elementwise" not in route.covered
    assert "elementwise" in route.reason or "generic" in route.reason


def test_a_hand_written_linalg_add_is_routed_identically():
    """The original decisive comparison, kept: no Triton involved, same decision."""
    package = _package(GEMMINI_PACKAGE)
    spec = K.vector_add_i32_spec()
    triton_route = compile_core.choose_route(
        to_linalg(source.make_ttir(spec), spec).module, target_package=package)
    hand_route = compile_core.choose_route(hand_written("add"), target_package=package)
    assert hand_route.kind == triton_route.kind == "llvm"
    assert hand_route.payload == triton_route.payload == ("elementwise",)
    assert hand_route.reason == triton_route.reason


# ------------------------------------------------- and it materializes where the plan declares it


@pytest.fixture(scope="module")
def descended():
    spec = K.vector_add_i32_spec()
    bridged = to_linalg(source.make_ttir(spec), spec)
    result = compile_core.compile_core_mlir(bridged.module, target_package=_package(RADIANCE_PACKAGE))
    return {"bridged": bridged, "lowered": result.staged, "route": result.route}


def test_it_takes_the_staged_path_on_a_target_that_declares_it(descended):
    assert descended["route"].kind == "staged"
    assert descended["route"].payload == ("elementwise",)
    assert "elementwise" in descended["route"].materializable


def test_every_stage_verifies_and_reaches_the_target_dialect(descended):
    for module in descended["lowered"].modules():
        module.verify()
    from merlin.xdsl_dialects._common import text

    assert "interface.elementwise" in text(descended["lowered"].interface_module)
    assert "elementwise" in text(descended["lowered"].target_module)


def test_the_command_buffer_declares_its_result_and_computes_the_right_answer(descended):
    """The result is a tensor, not a committed accumulator, so it has to be declared as an output.

    A destination missing from the tensor table is the quiet failure here: both engines collect
    vector-family results by role, so the buffer would run, report success and return nothing.
    """
    from merlin.runtime import reference_outputs, simulate
    from merlin.runtime.commandbuffer import materialize_inputs

    cb = descended["lowered"].command_buffer
    assert [c["opcode"] for c in cb["commands"]] == ["VECTOR_MAP"]
    roles = {name: spec["role"] for name, spec in cb["tensors"].items()}
    assert sorted(roles.values()) == ["input", "input", "output"], roles

    outputs = simulate(cb)["outputs"]
    assert outputs and outputs == reference_outputs(cb)

    tensors = materialize_inputs(cb)
    command = cb["commands"][0]["operands"]
    lhs = np.array(tensors[command["lhs"]].to_list(), dtype=np.int64).ravel()
    rhs = np.array(tensors[command["rhs"]].to_list(), dtype=np.int64).ravel()
    got = np.array(outputs[command["dst"]], dtype=np.int64).ravel()
    assert np.array_equal(got, lhs + rhs)


def test_a_fused_matmul_and_elementwise_payload_becomes_its_own_dispatch():
    """Real workload, decision made: the combine is a SEPARATE dispatch, never folded into the commit.

    This replaces an assertion that the same module was refused. The refusal was correct while the
    fusion question was open and the vector lane did not exist on this path; it is wrong now that it
    does, because the shape it refuses — a residual add consuming a matmul result — is what every
    whole model is made of. What the refusal actually guarded against was fusing the combine into the
    commit epilogue by accident, so that is what this asserts instead: the epilogue stays empty and
    the combine is its own VECTOR_MAP."""
    from xdsl.dialects import tensor as tensor_d
    from xdsl.dialects.builtin import FunctionType, IntegerType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as linalg_ops
    from xdsl.ir import Block, Region

    from merlin.xdsl_dialects.lowering.interface_lowering import LoweringError
    from merlin.xdsl_dialects.lowering.pipeline import lower_module

    i32 = IntegerType(32)
    t = TensorType(i32, [8, 8])
    block = Block(arg_types=[t, t, t])
    empty_mm = tensor_d.EmptyOp((), t)
    matmul = linalg_ops.MatmulOp(inputs=(block.args[0], block.args[1]),
                                 outputs=(empty_mm.tensor,), res=(t,))
    empty_add = tensor_d.EmptyOp((), t)
    add = linalg_ops.AddOp(inputs=(matmul.results[0], block.args[2]),
                           outputs=(empty_add.tensor,), res=(t,))
    block.add_ops([empty_mm, matmul, empty_add, add, ReturnOp(add.results[0])])
    module = ModuleOp([FuncOp("fused", FunctionType.from_lists([t, t, t], [t]), Region([block]))])

    assert set(compile_core.payload_classes(module)) >= {"matmul", "elementwise"}
    res = lower_module(module, target="toy_npu")

    names = [op.name for op in res.interface_module.walk()]
    assert "interface.matmul" in names and "interface.vector_map" in names
    # The combine is its OWN dispatch, in order after the commit -- not folded into it.
    assert [c["opcode"] for c in res.command_buffer["commands"]] == [
        "RES_PACK", "MATMUL_RESIDENT", "COMMIT", "VECTOR_MAP", "EVICT"]
    # The guard's real concern: an accidental fusion would show up as a non-empty commit epilogue.
    commits = [op for op in res.interface_module.walk() if op.name == "interface.commit"]
    assert commits and all(len(op.properties["epilogue"]) == 0 for op in commits), \
        "the combine must not have been folded into the commit epilogue"
    assert LoweringError is not None  # imported above; the refusal path is gone, not the error type
