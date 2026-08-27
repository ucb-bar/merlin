"""The generic `lower_module` entry point, the payload-completeness guard, and the payload router.

Three properties, in the order they matter:

1. **Nothing changed.** ``lower_module`` was extracted from ``lower_repeated_rhs_matmul`` so an
   arbitrary generic-MLIR payload can enter the staged pipeline. The STAGE_GOLDENS below were
   captured from the code as it stood immediately before that extraction, so "the refactor preserved
   behavior" is a checkable claim rather than an assertion. A golden that drifts is either a real
   regression or a deliberate change that must be re-captured on purpose.

2. **The pipeline no longer drops payload silently.** ``lower_to_interface`` rebuilds the function
   body as pack/matmul/commit/evict, so anything it does not recognize simply never gets emitted —
   and since every later stage verifies the REBUILT module, a dropped epilogue yields six modules
   that all verify and a command buffer that computes something else. See
   ``test_epilogue_is_rejected_not_dropped``.

3. **Routing is by payload, not by target.** A matmul takes the staged path even on a CPU-class
   target; generic computation takes the LLVM path even on an accelerator.

Targets are DISCOVERED from the registry where a test needs "some reference target", so registering
one does not require editing this file.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from merlin.common.paths import repo_root

pytest.importorskip("xdsl")

STAGES = ("input", "contract", "schedule", "interface", "target", "runtime")

# Stage fingerprints: sha256 of each printed module + the canonically serialized command buffer,
# first 16 hex chars. Do not "fix" a mismatch by re-pasting a new value without establishing WHY the
# IR moved — the whole value of this table is that a drift has to be explained before it is accepted.
#
# RE-CAPTURED 2026-08-25, and the reason is worth writing down because the previous values were stale
# for four months without anyone knowing which change had moved them.
#
# The originals were captured on the target-generalization branch, before the merge `af2a6e39`
# ("Merge feat/target-generalization into the gemmini-eviction stack"). That merge is where they
# stopped matching: measured by bisect, `af2a6e39^2` runs this file 17/17 GREEN and the merge commit
# itself fails 8 — and every commit after it inherits that, so the 400+ commits since were never the
# cause. Two of the merge's inherited behaviours moved the numbers, BOTH from the gemmini-eviction
# side (`af2a6e39^1`), and both verified present on that parent and absent on the other:
#
#   1. The runtime command buffer now DECLARES ITS OUTPUTS. `runtime.command_buffer.create` gained the
#      return values as named tensors with shapes (`Y0 = "64x64:i32"`, ...) plus an explicit
#      `outputs = [...]` list. Before, a consumer could not tell what the buffer produced. This is a
#      straightforward improvement and it moves `runtime` + `command_buffer` for every case.
#   2. A matmul is now ALWAYS staged resident, including at reuse=1. The other parent inferred
#      residency from reuse and emitted a plain non-resident `MATMUL` for a one-shot contraction; this
#      one emits RES_PACK / MATMUL_RESIDENT / COMMIT / EVICT unconditionally. That moves
#      `saturn_reuse1`'s `interface` and `target` too, and it is what
#      `test_toynpu_still_rejects_a_non_resident_rhs` was pinning — see its docstring, which now
#      records the cost this carries rather than pretending the change did not happen.
STAGE_GOLDENS: dict[str, dict[str, str]] = {
    "default_reuse4": {
        "input": "fed6f9dc9c36711a", "contract": "e78889598aefa3fb", "schedule": "18061d93e9f5ac00",
        "interface": "99ba7dd31ddfc6e5", "target": "ad246c1b10e5adcb", "runtime": "1246de5c28434cbe",
        "command_buffer": "c59a7c29773e6785"},
    "toynpu_reuse3": {
        "input": "d9dfdae8ad9f67e0", "contract": "1dd30866cc715105", "schedule": "d381afd0cd889137",
        "interface": "83608464001aad44", "target": "bd935637c2099835", "runtime": "5a861e1eb6d799bf",
        "command_buffer": "6788ed49363552b5"},
    "reuse2_small": {
        "input": "9ee60e58b12e2bce", "contract": "76060b80c611eeab", "schedule": "f995c9e39f8db4e2",
        "interface": "f94dae15b7ac81aa", "target": "11f5613d6b292163", "runtime": "0541f78c1bf66b8b",
        "command_buffer": "4094beaba73f3e99"},
    "saturn_reuse4": {
        "input": "9be380d68092547b", "contract": "d111a6aac70a0f4a", "schedule": "36ad01a37b5b8bd4",
        "interface": "f269988f3f7b3402", "target": "96d674b8dbff0e83", "runtime": "a18dda011616751f",
        "command_buffer": "37a6f443bc86dcc3"},
    "saturn_reuse2_small": {
        "input": "9ee60e58b12e2bce", "contract": "93430b386173dea8", "schedule": "c10a2947720aa0ec",
        "interface": "f94dae15b7ac81aa", "target": "690d7d2882444437", "runtime": "9763c77ae9f6de7b",
        "command_buffer": "3dc672219a0c4f3c"},
    # reuse=1 infers no residency at the SCHEDULE stage — contract and schedule are still the same
    # module — but the interface materializer stages the operand anyway (see note 2 above), so the
    # later stages do differ from a reuse-free lowering.
    "saturn_reuse1": {
        "input": "d71dbd9a059b028e", "contract": "36e4df5c6682d5c9", "schedule": "36e4df5c6682d5c9",
        "interface": "df5fd0bc27873f98", "target": "c38fd2afab854dc9", "runtime": "ffdd37a5f897cfc8",
        "command_buffer": "29216f498926b985"},
}

CASES: dict[str, dict] = {
    "default_reuse4": {},
    "toynpu_reuse3": {"reuse": 3},
    "reuse2_small": {"reuse": 2, "m": 8, "k": 12, "n": 10},
    "saturn_reuse4": {"reuse": 4, "m": 16, "k": 24, "n": 20, "target": "saturn"},
    "saturn_reuse2_small": {"reuse": 2, "m": 8, "k": 12, "n": 10, "target": "saturn"},
    "saturn_reuse1": {"reuse": 1, "m": 16, "k": 16, "n": 16, "target": "saturn"},
}


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _fingerprint(res) -> dict[str, str]:
    from merlin.xdsl_dialects._common import text

    out = {name: _digest(text(m)) for name, m in zip(STAGES, res.modules())}
    out["command_buffer"] = _digest(json.dumps(res.command_buffer, sort_keys=True))
    return out


# --------------------------------------------------------------- 1. the refactor changed nothing


@pytest.mark.parametrize("case", sorted(CASES))
def test_stage_fingerprints_match_pre_refactor(case):
    """Every stage of every configuration is byte-identical to the pre-refactor pipeline."""
    from merlin.xdsl_dialects.lowering import lower_repeated_rhs_matmul

    got = _fingerprint(lower_repeated_rhs_matmul(**CASES[case]))
    for stage in (*STAGES, "command_buffer"):
        assert got[stage] == STAGE_GOLDENS[case][stage], (
            f"{case}/{stage} moved: {STAGE_GOLDENS[case][stage]} -> {got[stage]}")


def test_wrapper_and_generic_entry_agree():
    """lower_repeated_rhs_matmul is now only a payload builder over lower_module."""
    from merlin.xdsl_dialects.lowering import lower_module, lower_repeated_rhs_matmul
    from merlin.xdsl_dialects.lowering.input_workload import build_input_module

    kw = {"reuse": 2, "m": 8, "k": 12, "n": 10}
    via_wrapper = _fingerprint(lower_repeated_rhs_matmul(**kw, target="saturn"))
    via_generic = _fingerprint(lower_module(build_input_module(**kw), target="saturn"))
    assert via_wrapper == via_generic


def test_a_one_shot_matmul_is_staged_resident_on_every_target():
    """What reuse=1 lowers to, and what that costs — the successor to a pin that no longer holds.

    This test used to assert the OPPOSITE: that toy_npu *rejected* a single non-reused matmul, because
    its dialect constrains matmul rhs to `toynpu.resident_tensor` and the lowering inferred residency
    from reuse, so a one-shot contraction descended as a plain non-resident `MATMUL` with nothing to
    satisfy that constraint. Saturn, which has a non-resident matmul, accepted the same shape and
    emitted no pack. The asymmetry was the point: it recorded which targets can execute a one-tile
    kernel without staging.

    That asymmetry is gone. The merge `af2a6e39` took the gemmini-eviction side's lowering, which
    stages EVERY matmul resident regardless of reuse (verified on the parents: `af2a6e39^2` raises
    here and emits no saturn pack; `af2a6e39^1` packs on both). So there is no longer a target that
    refuses this shape, and asserting that one does was pinning a branch that no longer exists.

    Pinned instead: the behaviour that replaced it, INCLUDING its cost. A weight used exactly once is
    packed and evicted anyway — four commands where two would do — and that is a real lowering
    decision with a measurable price, not a detail. It is deliberately left as a passing assertion
    rather than an xfail because the lowering is not wrong, only unconditional: whether a one-shot
    operand should be staged is a residency question that belongs to the schedule stage with a
    measurement behind it, and until something measures it, the honest thing is to record what we
    emit and what it costs.
    """
    from merlin.xdsl_dialects.lowering import lower_repeated_rhs_matmul

    # toy_npu has no non-resident matmul at all, so if anything still inferred residency from reuse
    # this would raise. It does not: the operand is staged, and the dialect constraint is satisfied.
    toy = lower_repeated_rhs_matmul(reuse=1, m=16, k=16, n=16)
    toy_ops = {op.name for op in toy.target_module.walk()}
    assert "toynpu.res_pack" in toy_ops, sorted(toy_ops)

    res = lower_repeated_rhs_matmul(reuse=1, m=16, k=16, n=16, target="saturn")
    assert _fingerprint(res)["command_buffer"] == STAGE_GOLDENS["saturn_reuse1"]["command_buffer"]

    target_ops = {op.name for op in res.target_module.walk()} - {
        "builtin.module", "func.func", "func.return"}
    assert any(name.endswith(".pack") for name in target_ops), target_ops

    # The cost, stated as data rather than left implicit: a pack and an evict for one use.
    opcodes = [c.get("opcode") for c in res.command_buffer.get("commands", [])]
    assert opcodes == ["RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT"], opcodes


def _epilogue_module(m: int = 16, k: int = 16, n: int = 16):
    """matmul -> linalg.transpose -> return. The transpose is a payload op the pipeline cannot build."""
    from xdsl.dialects import arith
    from xdsl.dialects import tensor as tensor_d
    from xdsl.dialects.builtin import DenseArrayBase, FunctionType, IntegerAttr, ModuleOp, TensorType, i8, i32, i64
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as linalg_ops
    from xdsl.ir import Block, Region

    At, Wt, Ot, Tt = (TensorType(i8, [m, k]), TensorType(i8, [k, n]),
                      TensorType(i32, [m, n]), TensorType(i32, [n, m]))
    arg_types = [At, At, Wt]
    blk = Block(arg_types=arg_types)
    a0, a1, w = blk.args
    zp = arith.ConstantOp(IntegerAttr(0, 32))
    ops, outs = [zp], []
    for a in (a0, a1):
        init = tensor_d.EmptyOp((), Ot)
        mm = linalg_ops.QuantizedMatmulOp(inputs=(a, w, zp.result, zp.result),
                                          outputs=(init.tensor,), res=(Ot,))
        tinit = tensor_d.EmptyOp((), Tt)
        tr = linalg_ops.TransposeOp(mm.results[0], tinit.tensor,
                                    permutation=DenseArrayBase.from_list(i64, [1, 0]))
        ops += [init, mm, tinit, tr]
        outs.append(tr.results[0])
    ops.append(ReturnOp(*outs))
    blk.add_ops(ops)
    return ModuleOp([FuncOp("k", FunctionType.from_lists(arg_types, [Tt, Tt]), Region([blk]))])


def test_epilogue_is_rejected_not_dropped():
    """The regression this guard exists for: it used to compile, minus the epilogue."""
    from merlin.xdsl_dialects.lowering import LoweringError, lower_module

    with pytest.raises(LoweringError) as exc:
        lower_module(_epilogue_module(), target="saturn")
    msg = str(exc.value)
    assert "silently drop" in msg and "linalg.transpose" in msg, msg


def test_the_drop_is_caught_twice_over():
    """Neutralize the pre-check and the rebuild loop still refuses — two independent guards.

    This test used to assert the opposite: that without ``_check_payload_complete`` the module
    descends, every stage verifies, a command buffer is emitted, and the epilogue has silently
    vanished. That WAS the state, and it is what made the pre-check load-bearing. It no longer is:
    the rebuild loop grew an explicit fail-closed arm for payload ops it does not lower, so the
    transpose is now named at the loop as well as at the pre-check.

    Keeping the test rather than deleting it keeps the property under watch from the inside. If the
    loop's arm is ever softened back into a silent skip, this goes green-by-way-of-compiling and the
    assertion below is what notices.
    """
    from merlin.xdsl_dialects.lowering import LoweringError, interface_lowering, lower_module

    real = interface_lowering._check_payload_complete
    interface_lowering._check_payload_complete = lambda *a, **k: None
    try:
        with pytest.raises(LoweringError) as exc:
            lower_module(_epilogue_module(), target="saturn")
    finally:
        interface_lowering._check_payload_complete = real
    assert "linalg.transpose" in str(exc.value), str(exc.value)


def test_multiple_functions_are_rejected():
    """Only the first func.func is materialized, so a second one would vanish."""
    from xdsl.dialects.builtin import ModuleOp

    from merlin.xdsl_dialects.lowering import LoweringError, lower_module
    from merlin.xdsl_dialects.lowering.input_workload import build_input_module

    fns = []
    for name in ("a", "b"):
        fn = next(op for op in build_input_module(reuse=2, m=8, k=8, n=8).walk()
                  if op.name == "func.func").clone()
        from xdsl.dialects.builtin import StringAttr
        fn.properties["sym_name"] = StringAttr(name)
        fns.append(fn)
    with pytest.raises(LoweringError) as exc:
        lower_module(ModuleOp(fns), target="saturn")
    assert "func.func" in str(exc.value)


def test_the_reference_payload_still_passes_the_guard():
    """The guard must be inert on legitimate payload: accumulator inits and zero points are consumed."""
    from merlin.xdsl_dialects.lowering import lower_module
    from merlin.xdsl_dialects.lowering.input_workload import build_input_module

    for reuse in (2, 3, 4):
        lower_module(build_input_module(reuse=reuse, m=8, k=12, n=10), target="saturn")


# ----------------------------------------------------------- 3. routing by payload (finding #3)


def _vector_add_module(n: int = 1025):
    from xdsl.dialects import tensor as tensor_d
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType, f32
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as linalg_ops
    from xdsl.ir import Block, Region

    T = TensorType(f32, [n])
    blk = Block(arg_types=[T, T])
    x, y = blk.args
    init = tensor_d.EmptyOp((), T)
    add = linalg_ops.AddOp((x, y), (init.tensor,), res=(T,))
    blk.add_ops([init, add, ReturnOp(add.results[0])])
    return ModuleOp([FuncOp("forward", FunctionType.from_lists([T, T], [T]), Region([blk]))])


def _reference_target() -> str:
    """A curated target that resolves with an in-tree dialect plan, discovered not named."""
    from merlin.targetgen.target_registry import list_targets, resolve

    for name in list_targets():
        try:
            resolve(name).load_dialect_plan()
        except Exception:  # noqa: BLE001 — plan lives out-of-tree; not usable for by-name routing
            continue
        return name
    pytest.skip("no reference target with an in-tree dialect plan")


def test_matmul_payload_routes_to_the_staged_path():
    from merlin.compile_core import choose_route
    from merlin.xdsl_dialects.lowering.input_workload import build_input_module

    route = choose_route(build_input_module(reuse=2, m=8, k=12, n=10), target=_reference_target())
    assert route.kind == "staged" and route.payload == ("matmul",)


def test_an_uncovered_payload_routes_to_llvm_even_on_an_accelerator_target():
    """Routing is by payload AND by what the target's plan covers, not by the target's class.

    The vector add is classified `elementwise` — the interface layer can materialize that shape in
    general — and it still routes to LLVM here, because this target's dialect plan does not declare
    coverage for it. Those are two independent facts and the route needs both: a payload the pipeline
    could build, on a target that never claimed to accelerate it.
    """
    from merlin.compile_core import choose_route

    route = choose_route(_vector_add_module(), target=_reference_target())
    assert route.kind == "llvm" and route.payload == ("elementwise",)
    assert "matmul" in route.materializable        # the target DOES accelerate matmul...
    assert "elementwise" not in route.covered      # ...but never declared this payload
    assert "elementwise" not in route.materializable


def test_unreadable_dialect_plan_fails_closed():
    """A plan that cannot be read must not be read as 'this target accelerates nothing'.

    That would silently demote an accelerator to the generic path and still report success. Uses a
    registered target whose plan lives in its out-of-tree package, if there is one.
    """
    from merlin.compile_core import RoutingError, choose_route
    from merlin.targetgen.target_registry import list_targets, resolve
    from merlin.xdsl_dialects.lowering.input_workload import build_input_module

    victim = None
    for name in list_targets():
        try:
            resolve(name).load_dialect_plan()
        except Exception:  # noqa: BLE001
            victim = name
            break
    if victim is None:
        pytest.skip("every registered target has an in-tree dialect plan")
    with pytest.raises(RoutingError) as exc:
        choose_route(build_input_module(reuse=2, m=8, k=8, n=8), target=victim)
    assert "dialect_plan" in str(exc.value) or "unreadable" in str(exc.value)


def test_out_of_tree_package_routes_to_staged():
    """The isolated, dynamically-loaded target package path (how a generated target is used)."""
    from merlin.compile_core import choose_route
    from merlin.xdsl_dialects.lowering.input_workload import build_input_module

    pkg_dir = repo_root() / "out/artifacts/targets/gemmini/hand_v0"
    if not pkg_dir.is_dir():
        pytest.skip("no out-of-tree target package present")
    from merlin.targetgen.registry import load_target

    route = choose_route(build_input_module(reuse=2, m=16, k=16, n=16),
                         target_package=load_target(pkg_dir))
    assert route.kind == "staged" and "matmul" in route.materializable


def test_plan_interface_ops_reads_both_committed_spellings():
    """Curated plans say `from: interface.matmul`; generated ones say `op: matmul`."""
    from merlin.compile_core import plan_interface_ops

    assert plan_interface_ops({"lowering": [{"from": "interface.matmul", "to": "x.matmul"}]}) == ("matmul",)
    assert plan_interface_ops({"lowering": [{"op": "matmul", "to": "x.matmul"}]}) == ("matmul",)
    assert plan_interface_ops(None) == ()
