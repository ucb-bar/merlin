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

# Captured from the pre-refactor pipeline (sha256 of each printed module + the canonically
# serialized command buffer, first 16 hex chars). Do not "fix" a mismatch by re-pasting a new value
# without establishing WHY the IR moved.
STAGE_GOLDENS: dict[str, dict[str, str]] = {
    "default_reuse4": {
        "input": "fed6f9dc9c36711a", "contract": "e78889598aefa3fb", "schedule": "18061d93e9f5ac00",
        "interface": "99ba7dd31ddfc6e5", "target": "ad246c1b10e5adcb", "runtime": "e138686bf7d29d1f",
        "command_buffer": "bc79d4a332aee311"},
    "toynpu_reuse3": {
        "input": "d9dfdae8ad9f67e0", "contract": "1dd30866cc715105", "schedule": "d381afd0cd889137",
        "interface": "83608464001aad44", "target": "bd935637c2099835", "runtime": "9434bcdd91e8ba9b",
        "command_buffer": "2d59bff01db11bd1"},
    "reuse2_small": {
        "input": "9ee60e58b12e2bce", "contract": "76060b80c611eeab", "schedule": "f995c9e39f8db4e2",
        "interface": "f94dae15b7ac81aa", "target": "11f5613d6b292163", "runtime": "7b45c9c55ef4429f",
        "command_buffer": "d5bce64ddac8462f"},
    "saturn_reuse4": {
        "input": "9be380d68092547b", "contract": "d111a6aac70a0f4a", "schedule": "36ad01a37b5b8bd4",
        "interface": "f269988f3f7b3402", "target": "96d674b8dbff0e83", "runtime": "164ffb1cfa3e317f",
        "command_buffer": "041b38f41e6e595d"},
    "saturn_reuse2_small": {
        "input": "9ee60e58b12e2bce", "contract": "93430b386173dea8", "schedule": "c10a2947720aa0ec",
        "interface": "f94dae15b7ac81aa", "target": "690d7d2882444437", "runtime": "86107f6ef3f6ced9",
        "command_buffer": "c5332c217df9f06e"},
    # reuse=1: no residency is inferred, so contract and schedule are the same module.
    "saturn_reuse1": {
        "input": "d71dbd9a059b028e", "contract": "36e4df5c6682d5c9", "schedule": "36e4df5c6682d5c9",
        "interface": "173f2dab0c244e28", "target": "e907345e1ab52dc9", "runtime": "bc377ef0cd98f4e3",
        "command_buffer": "a4651f55577b8e2d"},
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


def test_toynpu_still_rejects_a_non_resident_rhs():
    """A pre-existing behavior the refactor must not paper over.

    toy_npu's dialect constrains matmul rhs to `toynpu.resident_tensor`, so a single non-reused
    matmul cannot descend — while saturn and gemmini accept exactly that shape. Pinned because the
    one-tile kernel proof depends on knowing which targets allow it.
    """
    from xdsl.utils.exceptions import VerifyException

    from merlin.xdsl_dialects.lowering import lower_repeated_rhs_matmul

    with pytest.raises(VerifyException):
        lower_repeated_rhs_matmul(reuse=1, m=16, k=16, n=16)
    # saturn does allow it, and its command buffer is pinned.
    res = lower_repeated_rhs_matmul(reuse=1, m=16, k=16, n=16, target="saturn")
    assert _fingerprint(res)["command_buffer"] == STAGE_GOLDENS["saturn_reuse1"]["command_buffer"]
    target_ops = {op.name for op in res.target_module.walk()} - {
        "builtin.module", "func.func", "func.return"}
    assert not any(name.endswith(".pack") for name in target_ops), target_ops


# ------------------------------------------------- 2. the payload-completeness guard (finding #2)


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


def test_guard_is_what_rejects_it_and_the_drop_was_real():
    """Neutralize the guard and the SAME module compiles clean with the transpose gone.

    This is the evidence that the guard is load-bearing: without it the module descends, every
    stage verifies, a command buffer is emitted, and the epilogue has vanished.
    """
    from merlin.xdsl_dialects.lowering import interface_lowering, lower_module

    real = interface_lowering._check_payload_complete
    interface_lowering._check_payload_complete = lambda *a, **k: None
    try:
        res = lower_module(_epilogue_module(), target="saturn")
    finally:
        interface_lowering._check_payload_complete = real

    for mod in res.modules():
        mod.verify()                                    # all six verify — that is the danger
    assert res.command_buffer["commands"]               # and a command buffer was produced
    assert "linalg.transpose" not in {op.name for op in res.interface_module.walk()}


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


def test_generic_payload_routes_to_llvm_even_on_an_accelerator_target():
    """Routing is by payload: a target that accelerates matmul still compiles a vector add generically."""
    from merlin.compile_core import choose_route

    route = choose_route(_vector_add_module(), target=_reference_target())
    assert route.kind == "llvm" and route.payload == ("generic",)
    assert "matmul" in route.materializable    # the target DOES accelerate matmul...
    assert "generic" not in route.materializable  # ...just not this payload


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
