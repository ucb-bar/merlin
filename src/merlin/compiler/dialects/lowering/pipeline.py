"""The staged core-dialect lowering pipeline.

linalg input -> contract -> schedule -> interface -> target (toynpu) -> runtime
-> command-buffer dict -> the Python engine (``merlin.runtime``).

Each stage is a plain module->module transform (wrappable as xDSL passes once the IR
stabilizes); every intermediate module is verified and kept on the result so tests and
tools can inspect the whole descent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.compiler.dialects.lowering.analyses import lower_to_contract
from merlin.compiler.dialects.lowering.interface_stage import LoweringError, lower_to_interface, lower_to_schedule
from merlin.compiler.dialects.lowering.runtime_stage import emit_command_buffer, lower_to_runtime
from merlin.compiler.dialects.lowering.target_stage import lower_to_target
from merlin.compiler.dialects.xdsl_context import HAS_XDSL, record_stage
from merlin.compiler.ir import compile_trace
from merlin.compiler.ir.lowering_audit import IrAudit
from merlin.target.capabilities.families import DEFAULT_EXAMPLE_TARGET

# ---- input_workload ----------------------------------------------------------------------------
# Build the linalg-level input module for the MVP workload.
#
# ``repeated_rhs_matmul``: for i in 0..R-1: Y_i = A_i @ W, with one immutable W. Each
# matmul is a real ``linalg.quantized_matmul`` (the standard MLIR idiom for
# i8 x i8 -> i32 accumulation; zero points are 0). Outputs stay i32 accumulations —
# the workload spec carries no epilogue (experiments/benchmarks/semantic_memory/
# repeated_rhs_matmul.yaml: op_sequence [matmul]).


def build_input_module(reuse: int = 4, m: int = 64, k: int = 128, n: int = 64):
    """func @repeated_rhs_matmul(A_0..A_{R-1}: mxk i8, W: kxn i8) -> R x (mxn i32)."""
    if not HAS_XDSL:
        return None
    from xdsl.dialects import arith
    from xdsl.dialects import tensor as tensor_d
    from xdsl.dialects.builtin import FunctionType, IntegerAttr, ModuleOp, TensorType, i8, i32
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as linalg_ops
    from xdsl.ir import Block, Region

    if reuse < 1:
        raise ValueError("reuse must be >= 1")
    At = TensorType(i8, [m, k])
    Wt = TensorType(i8, [k, n])
    Ot = TensorType(i32, [m, n])

    arg_types = [At] * reuse + [Wt]
    blk = Block(arg_types=arg_types)
    a_args, w = list(blk.args[:-1]), blk.args[-1]

    zp = arith.ConstantOp(IntegerAttr(0, 32))
    ops = [zp]
    outs = []
    for a in a_args:
        init = tensor_d.EmptyOp((), Ot)
        mm = linalg_ops.QuantizedMatmulOp(inputs=(a, w, zp.result, zp.result), outputs=(init.tensor,), res=(Ot,))
        ops += [init, mm]
        outs.append(mm.results[0])
    ops.append(ReturnOp(*outs))
    blk.add_ops(ops)
    fn = FuncOp("repeated_rhs_matmul", FunctionType.from_lists(arg_types, [Ot] * reuse), Region([blk]))
    return ModuleOp([fn])


def build_matmul_chain(dims=(8, 16, 12, 6), elem="f32"):
    """func @chain(A, W_1..W_{L}) -> mxn_L : a feed-forward stack of L matmuls where each
    layer's output is the next layer's LHS (the whole-model backbone). ``dims`` = the L+1 shared
    dimensions [m, k_1, k_2, ..., k_L]; ``elem`` selects f32 (plain matmul) or i8/i32 semantics.
    Returns (module, [weight arrays are the caller's to inject by name])."""
    if not HAS_XDSL:
        return None
    from xdsl.dialects import tensor as tensor_d
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType, f16, f32, f64
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as linalg_ops
    from xdsl.ir import Block, Region

    et = {"f16": f16, "f32": f32, "f64": f64}[elem]
    m = dims[0]
    ks = list(dims[1:])
    if len(ks) < 1:
        raise ValueError("a chain needs at least one matmul (dims = [m, k, n, ...])")
    a_t = TensorType(et, [m, ks[0]])
    w_ts = [TensorType(et, [ks[j], ks[j + 1]]) for j in range(len(ks) - 1)]
    blk = Block(arg_types=[a_t] + w_ts)
    cur = blk.args[0]
    ws = list(blk.args[1:])
    ops = []
    rows = m
    for j, w in enumerate(ws):
        out_t = TensorType(et, [rows, ks[j + 1]])
        init = tensor_d.EmptyOp((), out_t)
        mm = linalg_ops.MatmulOp(inputs=(cur, w), outputs=(init.tensor,), res=(out_t,))
        ops += [init, mm]
        cur = mm.results[0]
    ops.append(ReturnOp(cur))
    blk.add_ops(ops)
    fn = FuncOp("chain", FunctionType.from_lists([a_t] + w_ts, [cur.type]), Region([blk]))
    return ModuleOp([fn])


def build_vector_block(m: int = 8, k: int = 16, elem: str = "f32", combine: str = "add", relu: bool = True):
    """func @vecblock(A: m×k, W1: k×k, W2: k×k) -> m×k : ``combine(relu(A@W1), A@W2)``.

    Exercises the non-matmul vector path alongside matmuls: two ``linalg.matmul`` layers, an
    optional relu (``linalg.max`` against a zero fill — the standard relu idiom, which lowers to an
    identity vector_map + relu activation), and an elementwise ``linalg.add``/``linalg.mul`` (a
    residual add or a gating multiply). ``combine`` ∈ {"add", "mul"}."""
    if not HAS_XDSL:
        return None
    from xdsl.dialects import arith
    from xdsl.dialects import tensor as tensor_d
    from xdsl.dialects.builtin import FloatAttr, FunctionType, ModuleOp, TensorType, f16, f32, f64
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as linalg_ops
    from xdsl.ir import Block, Region

    et = {"f16": f16, "f32": f32, "f64": f64}[elem]
    combine_op = {"add": linalg_ops.AddOp, "mul": linalg_ops.MulOp}[combine]
    a_t = TensorType(et, [m, k])
    w_t = TensorType(et, [k, k])
    o_t = TensorType(et, [m, k])
    blk = Block(arg_types=[a_t, w_t, w_t])
    A, W1, W2 = blk.args
    ops = []

    e1 = tensor_d.EmptyOp((), o_t)
    mm1 = linalg_ops.MatmulOp(inputs=(A, W1), outputs=(e1.tensor,), res=(o_t,))
    ops += [e1, mm1]
    cur = mm1.results[0]

    if relu:
        zc = arith.ConstantOp(FloatAttr(0.0, et))
        ze = tensor_d.EmptyOp((), o_t)
        zf = linalg_ops.FillOp(inputs=(zc.result,), outputs=(ze.tensor,), res=(o_t,))
        re = tensor_d.EmptyOp((), o_t)
        rl = linalg_ops.MaxOp(inputs=(cur, zf.results[0]), outputs=(re.tensor,), res=(o_t,))
        ops += [zc, ze, zf, re, rl]
        cur = rl.results[0]

    e2 = tensor_d.EmptyOp((), o_t)
    mm2 = linalg_ops.MatmulOp(inputs=(A, W2), outputs=(e2.tensor,), res=(o_t,))
    ops += [e2, mm2]

    ce = tensor_d.EmptyOp((), o_t)
    cmb = combine_op(inputs=(cur, mm2.results[0]), outputs=(ce.tensor,), res=(o_t,))
    ops += [ce, cmb]

    ops.append(ReturnOp(cmb.results[0]))
    blk.add_ops(ops)
    fn = FuncOp("vecblock", FunctionType.from_lists([a_t, w_t, w_t], [o_t]), Region([blk]))
    return ModuleOp([fn])


def find_matmuls(module):
    """All linalg matmul-family ops in the module (quantized + plain)."""
    from xdsl.dialects.linalg import ops as linalg_ops

    found = []
    for op in module.walk():
        if isinstance(op, (linalg_ops.QuantizedMatmulOp, linalg_ops.MatmulOp)):
            found.append(op)
    return found


def matmul_lhs_rhs(mm):
    """(lhs, rhs) SSA values of a linalg matmul-family op (inputs[0], inputs[1])."""
    return mm.inputs[0], mm.inputs[1]


# linalg named ops that map onto a runtime VECTOR_MAP combine. `linalg.sub` is deliberately absent:
# the runtime implements add and mul only, and lowering a subtract as an add would be a miscompile.
ELEMENTWISE_COMBINES = {"linalg.add": "add", "linalg.mul": "mul"}


def find_elementwise(module):
    """All linalg elementwise ops the interface layer can materialize, as (op, combine) pairs."""
    return [(op, ELEMENTWISE_COMBINES[op.name]) for op in module.walk() if op.name in ELEMENTWISE_COMBINES]


def elementwise_operands(op):
    """(lhs, rhs) SSA values of a linalg elementwise op."""
    return op.inputs[0], op.inputs[1]


# ---- pipeline ----------------------------------------------------------------------------------

#: The stages :func:`lower_module` records, in order (``--stop-after``/``--dump-ir-after`` names them).
STAGES = compile_trace.declare(
    "staged",
    ("input", "contract", "schedule", "interface", "target", "runtime"),
    entry="merlin.compiler.dialects.lowering.lower_module",
    summary="linalg payload -> contract -> schedule -> interface -> target dialect -> runtime (command buffer)",
)


def load_curated_contract(target: str) -> dict:
    """The selected support contract, using the same registry as target lowering.

    FAILS CLOSED on an unknown name. This used to fall back to the default contract whenever the
    in-tree path was missing, which meant asking for *any* out-of-tree or misspelled target quietly
    lowered the whole module for ``toy_npu`` instead — and produced a module that verified at every
    stage, simulated correctly, and emitted a command buffer naming a target nobody asked for.
    Invalid selected support never borrows another contract or a built-in default.
    """
    from merlin.target.registry.identity import resolve

    try:
        selected = resolve(target)
        path = selected.contract_path
        contract = selected.load_contract()
        if not isinstance(contract, dict):
            raise ValueError(f"{path}: expected a target contract mapping")
        return contract
    except Exception as exc:  # noqa: BLE001 — any resolution failure is a hard stop
        raise LoweringError(
            f"no usable selected target contract for {target!r}: "
            f"{type(exc).__name__}: {exc}. An out-of-tree target is reached by pointing MERLIN_TARGET_PATH at "
            f"its package, or by passing target_package=."
        ) from exc


@dataclass
class LoweringResult:
    """All intermediate modules plus the executable command buffer."""

    input_module: Any
    contract_module: Any
    schedule_module: Any
    interface_module: Any
    target_module: Any
    runtime_module: Any
    command_buffer: dict[str, Any] = field(default_factory=dict)

    def modules(self):
        return [
            self.input_module,
            self.contract_module,
            self.schedule_module,
            self.interface_module,
            self.target_module,
            self.runtime_module,
        ]


def lower_module(
    input_module: Any,
    *,
    target: str = DEFAULT_EXAMPLE_TARGET,
    target_contract: dict[str, Any] | None = None,
    dialect_plan: dict[str, Any] | None = None,
    backend: str | None = None,
    target_package: Any | None = None,
    workdir: str | Path | None = None,
    ir_audit: bool | str = False,
    audit_sidecars: tuple[str | Path, ...] = (),
) -> LoweringResult:
    """Lower an ARBITRARY generic-MLIR module end to end; verify every intermediate module.

    With an explicit ``workdir`` and ``ir_audit=True``, retain exact named-stage
    snapshots and failure-prefix evidence. No output is written without a workdir.
    Declared ``audit_sidecars`` are bound in place, never converted or copied.
    ``ir_audit="compact"`` records inspection-only generic xDSL views with large
    dense tensors elided. ``"both"`` also retains the exact stage serialization.

    This is the pipeline's real entry point: the payload is a parameter, so any frontend that can
    produce clean linalg-on-tensors (model2MLIR, a Triton kernel bridge, a hand-authored module)
    descends the same contract -> schedule -> interface -> target -> runtime path. It used to be
    welded to one synthetic workload builder, which meant "compile this kernel" had no way in.

    What the incoming module must look like — the pipeline fails closed rather than silently
    reinterpreting a module it does not understand:

    * one ``func.func`` with a single block (a second function or block would be dropped);
    * matmul-family payload (``linalg.matmul`` / ``linalg.quantized_matmul``) whose operands are
      the function's own block arguments, NOT values produced by memref/bufferization boundary ops
      — residency inference traces to a block argument, and interface materialization maps operands
      through the function arguments;
    * every other op in the block either feeding that payload or being contract/schedule
      decoration (see :func:`interface_lowering.lower_to_interface`'s completeness check).

    This is the whole-model / section compiler entry: ``input_module`` may hold a single matmul,
    the MVP repeated-RHS workload, a chained multi-layer model, or a sliced section — the staged
    descent is the same. ``target_package`` (a loaded target package) lowers
    through an ISOLATED, dynamically-loaded target dialect (no core edits, plug-and-play); built-in
    reference targets still work via ``target``.
    """
    with IrAudit(
        workdir,
        enabled=ir_audit,
        producer="merlin.compiler.dialects.lowering.lower_module",
        source=__file__,
        sidecars=audit_sidecars,
    ) as audit:
        if not HAS_XDSL:
            raise LoweringError("xDSL is required for the lowering pipeline")

        spec = opcodes = None
        if target_package is not None:
            tc = target_contract or target_package.contract or load_curated_contract(target_package.name)
            dialect_plan = dialect_plan or target_package.dialect_plan()
            spec = target_package.spec
            opcodes = target_package.opcode_table
            name = target_package.name
        else:
            tc = target_contract or load_curated_contract(target)
            name = tc["name"]
        from merlin.target.registry.identity import backend_for

        backend = backend or backend_for(name)
        input_module.verify()
        record_stage(audit, "input", input_module)
        contract_module = lower_to_contract(input_module, tc)
        contract_module.verify()
        record_stage(audit, "contract", contract_module)
        schedule_module = lower_to_schedule(contract_module)
        schedule_module.verify()
        record_stage(audit, "schedule", schedule_module)
        interface_module = lower_to_interface(schedule_module)
        interface_module.verify()
        record_stage(audit, "interface", interface_module)
        target_module = lower_to_target(interface_module, dialect_plan, target=name, spec=spec)
        target_module.verify()
        record_stage(audit, "target", target_module)
        runtime_module = lower_to_runtime(target_module, target=name, backend=backend, opcodes=opcodes)
        runtime_module.verify()
        record_stage(audit, "runtime", runtime_module)
        cb = emit_command_buffer(runtime_module)

        return LoweringResult(
            input_module=input_module,
            contract_module=contract_module,
            schedule_module=schedule_module,
            interface_module=interface_module,
            target_module=target_module,
            runtime_module=runtime_module,
            command_buffer=cb,
        )


def lower_repeated_rhs_matmul(
    reuse: int = 4,
    m: int = 64,
    k: int = 128,
    n: int = 64,
    target: str = DEFAULT_EXAMPLE_TARGET,
    target_contract: dict[str, Any] | None = None,
    dialect_plan: dict[str, Any] | None = None,
    backend: str | None = None,
    target_package: Any | None = None,
) -> LoweringResult:
    """Lower the MVP workload (``for i: Y_i = A_i @ W``) end to end.

    A thin wrapper over :func:`lower_module` — it only builds the payload. Kept because it is the
    reference workload every staged-pipeline test is written against.
    """
    return lower_module(
        build_input_module(reuse=reuse, m=m, k=k, n=n),
        target=target,
        target_contract=target_contract,
        dialect_plan=dialect_plan,
        backend=backend,
        target_package=target_package,
    )


def execute(result: LoweringResult, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the lowered command buffer on the engine and assert correctness.

    Returns {outputs, metrics, trace, correct} where ``correct`` is the equality of
    the simulated outputs with the independent reference recomputation.
    """
    from merlin.runtime import outputs_match, reference_outputs, simulate

    res = simulate(result.command_buffer, inputs)
    ref = reference_outputs(result.command_buffer, inputs)
    res["correct"] = outputs_match(res["outputs"], ref)
    return res
