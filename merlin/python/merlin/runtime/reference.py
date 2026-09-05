"""Independent reference recomputation for correctness checking.

The simulator executes the *residency* path (pack a weight once, reuse it). The reference
recomputes each committed output directly from the leaf inputs, bypassing the resident store
entirely. Residency is a performance optimization that must not change results, so the two
must agree -- that equality is the correctness guarantee the execution tests assert.
"""
from __future__ import annotations

from typing import Any

from .commandbuffer import (BIAS_STAGES, apply_pool_stage, bias_tensor_name,
                            conv_im2col, conv_out_dims, materialize_inputs,
                            pool_params)
from .tensor import Tensor


#: Opcodes this reference engine has a definition for. An opcode outside this set (and outside
#: :data:`NO_NUMERIC_EFFECT`) is NOT silently skipped: a buffer whose result depends on it cannot be
#: recomputed here, and dropping it produced an EMPTY output map that downstream read as "the kernel never
#: wrote its output" — indistinguishable from a real dropped store, and unfixable by any submission.
#: Fail closed instead (see :class:`UnmodeledOp`).
MODELED_OPCODES = frozenset({
    "RES_PACK", "MATMUL_RESIDENT", "MATMUL", "COMMIT", "VECTOR_MAP", "VREDUCE", "BIAS_ADD",
    "ATTENTION_QK", "ATTENTION_PV", "CONV2D", "MOVEMENT", "BATCHED_MATMUL",
})

#: Opcodes with NO effect on committed values, correctly ignored here rather than "unmodeled". ``EVICT``
#: frees a resident-store handle and counts an eviction — residency is a performance optimization that by
#: definition does not change results, and this naive path keeps no resident store to free. Listing them
#: explicitly is what lets an opcode the engine genuinely cannot evaluate raise instead of being dropped.
NO_NUMERIC_EFFECT = frozenset({"EVICT"})


class UnmodeledOp(ValueError):
    """The command buffer uses an opcode this integer reference engine has no definition for.

    Raised so a caller can tell "I cannot check this" apart from "the buffer is wrong" — the first is a
    property of THIS engine's op vocabulary (grade the artifact on the hardware oracle instead), the
    second is a defect in the submission. Conflating them fails correct work.
    """

    def __init__(self, opcodes: list[str]) -> None:
        self.opcodes = list(opcodes)
        super().__init__(
            f"the integer reference engine models {sorted(MODELED_OPCODES)} and has no definition for "
            f"{sorted(set(opcodes))}")


# The reference intentionally evaluates whole-op convolution independently of the simulator's command
# dispatch.  Keep its accepted attribute surface explicit and fail closed: silently ignoring one of a
# convolution's geometry knobs produces a plausible, completely wrong tensor.
_CONV2D_ATTRS = frozenset({
    "kernel", "stride", "padding", "dilation", "layout", "epilogue", "output_dtype",
    "acc_scale", "requant_shift", "semantic", "pool_in_dims", "pool_size", "pool_stride",
    "pool_padding", "pool_pad_value",
})

_ATTENTION_ATTRS = frozenset({"epilogue", "output_dtype", "acc_scale", "requant_shift", "semantic"})


def _conv_geom(attrs: dict[str, Any], key: str, arity: int, default: list[int]) -> tuple[int, ...]:
    value = attrs.get(key, default)
    if not isinstance(value, list) or len(value) != arity:
        raise ValueError(f"CONV2D {key} must be a list of {arity} int(s) (got {value!r})")
    return tuple(int(x) for x in value)


def _narrow_int_readout(t: Tensor, dtype: str, op: str) -> Tensor:
    """Saturate an i32 result into its declared integer container.

    This is independently spelled here so the simulator/reference comparison still catches a broken
    command dispatch, while following the ABI's derived-bitwidth rule rather than assuming one target's
    output width. A non-integer token passes through unchanged, matching the golden -- see the longer
    note on :func:`merlin.runtime.simulator._narrow_int_readout` for why raising instead was wrong.
    """
    kind, digits = (dtype[:1], dtype[1:]) if dtype else ("", "")
    if kind not in ("i", "u") or not digits.isdigit():
        return t
    bits, signed = int(digits), kind == "i"
    if bits >= 32:
        return t
    if dtype == "i8":
        return t.to_i8()
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    return Tensor(t.shape, [lo if x < lo else (hi if x > hi else x) for x in t.data], dtype)


def _attention_epilogue(t: Tensor, attrs: dict[str, Any], default_shift: int, op: str) -> Tensor:
    unknown = sorted(set(attrs) - _ATTENTION_ATTRS)
    if unknown:
        raise ValueError(
            f"{op} does not implement attribute(s) {unknown}; it implements "
            f"{sorted(_ATTENTION_ATTRS)}")
    for stage in attrs.get("epilogue", []):
        if stage == "acc_scale":
            t = t.requant_acc_scale(float(attrs.get("acc_scale", 1.0)))
        elif stage == "requant":
            t = t.requant(int(attrs.get("requant_shift", default_shift)))
        elif stage == "relu":
            t = t.relu()
        else:
            raise ValueError(
                f"{op} declares epilogue stage {stage!r}, which this reference engine does not "
                f"implement (implemented: acc_scale, requant, relu)")
    return _narrow_int_readout(t, str(attrs.get("output_dtype", "i32")), op)


def _conv2d(ifm_name: str, ifm: Tensor, weight_name: str, weight: Tensor, dst: str,
            attrs: dict[str, Any], default_shift: int) -> Tensor:
    unknown = sorted(set(attrs) - _CONV2D_ATTRS)
    if unknown:
        raise ValueError(
            f"CONV2D does not implement attribute(s) {unknown}; it implements "
            f"{sorted(_CONV2D_ATTRS)}. Applying a conv with a parameter this engine ignored would be "
            f"a wrong answer that still looks like a tensor")
    layout = str(attrs.get("layout", "nhwc"))
    if layout != "nhwc":
        raise ValueError(f"CONV2D layout {layout!r} unsupported (nhwc only)")
    kernel = attrs.get("kernel")
    if not isinstance(kernel, list) or len(kernel) != 4:
        raise ValueError(f"CONV2D needs kernel = [kh, kw, ci, co]; got {kernel!r}")
    kh, kw, ci, co = (int(v) for v in kernel)
    stride = _conv_geom(attrs, "stride", 2, [1, 1])
    padding = _conv_geom(attrs, "padding", 4, [0, 0, 0, 0])
    dilation = _conv_geom(attrs, "dilation", 2, [1, 1])
    if len(ifm.shape) != 4:
        raise ValueError(f"CONV2D activation {ifm_name}{ifm.shape} is not rank-4 NHWC")
    if ifm.shape[3] != ci:
        raise ValueError(
            f"CONV2D channel mismatch: {ifm_name}{ifm.shape} has C={ifm.shape[3]} but kernel "
            f"declares ci={ci}")
    cols = conv_im2col(ifm, kh=kh, kw=kw, ci=ci, stride=stride, padding=padding,
                       dilation=dilation, layout=layout)
    if weight.shape != (kh * kw * ci, co):
        raise ValueError(
            f"CONV2D weight {weight_name}{weight.shape} is not the im2col-packed "
            f"[Kh*Kw*Ci, Co] = [{kh * kw * ci}, {co}] the kernel attribute declares")
    result = cols.matmul(weight)
    for stage in attrs.get("epilogue", []):
        if stage == "acc_scale":
            result = result.requant_acc_scale(float(attrs.get("acc_scale", 1.0)))
        elif stage == "requant":
            result = result.requant(int(attrs.get("requant_shift", default_shift)))
        elif stage == "relu":
            result = result.relu()
        elif stage == "maxpool":
            ho, wo = conv_out_dims(ifm.shape[1], ifm.shape[2], kh, kw,
                                   stride, padding, dilation)
            pool_in = pool_params(attrs, op=f"CONV2D {dst!r}")["pool_in_dims"]
            if pool_in != (ho, wo):
                raise ValueError(
                    f"CONV2D {dst!r}: pool_in_dims {list(pool_in)} disagrees with the conv's own "
                    f"output extent [{ho}, {wo}]; the pooled image is the conv's output, so the "
                    f"declaration is wrong (or the conv geometry is)")
            result = apply_pool_stage(result, stage, attrs, op=f"CONV2D {dst!r}")
        else:
            raise ValueError(f"CONV2D does not implement epilogue stage {stage!r}")
    return _narrow_int_readout(result, str(attrs.get("output_dtype", "i32")), "CONV2D")


def reference_outputs(cb: dict[str, Any], inputs: dict[str, Any] | None = None) -> dict[str, list]:
    """Recompute committed outputs from leaf inputs via a naive (non-resident) path.

    Raises :class:`UnmodeledOp` when the buffer uses an opcode outside :data:`MODELED_OPCODES`, rather
    than skipping it and returning a silently incomplete output map.
    """
    _known = MODELED_OPCODES | NO_NUMERIC_EFFECT
    unmodeled = [c["opcode"] for c in cb.get("commands", []) if c.get("opcode") not in _known]
    if unmodeled:
        raise UnmodeledOp(unmodeled)
    env: dict[str, Tensor] = materialize_inputs(cb, inputs)
    resident_source: dict[str, str] = {}
    resident_dequant: dict[str, tuple[str, str, int]] = {}   # pack dst -> (i8 src, scale, axis)
    matmul_for: dict[str, dict] = {}
    commits: list[dict] = []

    for cmd in cb.get("commands", []):
        op = cmd["opcode"]
        ops = cmd.get("operands", {})
        attrs = cmd.get("attributes", {})
        if op == "RES_PACK":
            resident_source[ops["dst"]] = ops["src"]
            if "scale" in ops:
                resident_dequant[ops["dst"]] = (ops["src"], ops["scale"],
                                                int(attrs.get("dequant_axis", 1)))
        elif op in ("MATMUL_RESIDENT", "MATMUL"):
            matmul_for[ops["dst"]] = cmd
        elif op == "COMMIT":
            commits.append(cmd)

    default_shift = int(cb.get("params", {}).get("requant_shift", 4))
    outputs: dict[str, list] = {}

    for commit in commits:
        ops = commit.get("operands", {})
        attrs = commit.get("attributes", {})
        mm = matmul_for[ops["src"]]
        mops = mm.get("operands", {})
        lhs = env[mops["lhs"]]
        if mops["rhs"] in resident_dequant:                     # int8 weight-only dequant pack
            src_name, scale_name, axis = resident_dequant[mops["rhs"]]
            rhs = env[src_name].dequant_per_channel(env[scale_name], axis)
        else:
            rhs_name = resident_source.get(mops["rhs"], mops["rhs"])  # resolve through the pack
            rhs = env[rhs_name]
        t = lhs.matmul(rhs)
        shift = int(attrs.get("requant_shift", default_shift))
        for stage in attrs.get("epilogue", []):
            if stage in BIAS_STAGES:
                t = t.add_bias(env[bias_tensor_name(ops, attrs, op=f"COMMIT {ops.get('dst')!r}")])
            elif stage == "requant":
                t = t.requant(shift)
            elif stage == "acc_scale":
                t = t.requant_acc_scale(float(attrs.get("acc_scale", 1.0)))
            elif stage == "relu":
                t = t.relu()
            elif stage == "maxpool":
                # Pooling fused into the accumulator readout, exactly where the store-path ABI puts it
                # (``config_st(..., pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols,
                # upad, lpad)`` immediately before the mvout). At COMMIT the accumulator is ``[M, N]``
                # with no conv geometry in scope, so the window and the spatial extent the rows
                # unflatten to are read from THIS command's attributes -- the same ones the golden reads,
                # through the same parser.
                t = apply_pool_stage(t, stage, attrs, op=f"COMMIT {ops.get('dst')!r}")
            else:
                # FAIL CLOSED, matching the golden engine and the simulator. This loop had no terminal
                # branch, so an epilogue stage the reference did not know was skipped in SILENCE and the
                # buffer's committed value came out un-transformed. That is worse here than anywhere
                # else: the reference is one half of the L0 comparison, so a stage dropped on both sides
                # (golden and reference each skipping it) AGREES, and the capsule passes having proved
                # nothing about the stage it declared. That is precisely how a pooling epilogue could be
                # "covered" by arithmetic no engine performed.
                raise ValueError(
                    f"COMMIT {ops.get('dst')!r} declares epilogue stage {stage!r}, which this reference "
                    f"engine does not implement (implemented: bias_add/bias, requant, acc_scale, relu, "
                    f"maxpool). It is not skipped: a silently dropped stage would make the golden and "
                    f"the reference agree on a value neither of them computed")
        # The readout is the SHARED rule, not an inline i8 test. Two things were wrong here.
        # The default was "i8", so a COMMIT that simply omitted output_dtype had its result
        # CLAMPED to [-128,127] -- while capsule_golden, the authority at L2/L3, defaulted to
        # "i32" and did not narrow at all. The two disagreed from 2026-06-20, and a correct
        # backend omitting the attribute failed L0 on 85 of 130 integer capsules with a
        # message blaming the agent. The exact "== i8" test also could not express i16/i4/u8
        # at all, so any future target narrowing to a non-i8 width was silently mis-graded.
        # Defaulting to i32 is the safe direction: absent means DO NOT narrow, which cannot
        # destroy a wide result, and it matches the engine that decides correctness.
        t = _narrow_int_readout(t, str(attrs.get("output_dtype", "i32")), "COMMIT")
        outputs[ops["dst"]] = t.to_list()
        # Register the committed tensor so a CHAINED consumer (the next layer's matmul lhs, a
        # vector op) resolves it — a whole model's intermediate activations flow through env.
        env[ops["dst"]] = t

    # Vector-family ops: recompute directly (no residency optimization to bypass, so the
    # reference is the same elementwise math — the meaningful gate for this family is
    # merlin == RTL oracle, not the residency-bypass cross-check that matmul has).
    for cmd in cb.get("commands", []):
        op = cmd["opcode"]
        ops = cmd.get("operands", {})
        attrs = cmd.get("attributes", {})
        if op == "VECTOR_MAP":
            combine = attrs.get("combine", "add")
            if combine == "identity":            # data movement: dst is a copy of lhs (layout move)
                a = env[ops["lhs"]]
                t = Tensor(a.shape, list(a.data), a.dtype)
            else:
                a, b = env[ops["lhs"]], env[ops["rhs"]]
                t = a.ew_add(b) if combine == "add" else a.ew_mul(b)
            for stage in attrs.get("activation", []):
                if stage == "relu":
                    t = t.relu()
            env[ops["dst"]] = t
        elif op == "VREDUCE":
            env[ops["dst"]] = env[ops["src"]].reduce_sum()
        elif op == "BIAS_ADD":
            # The UNFUSED half of an epilogue-fusion pair: the same per-column add that a COMMIT
            # runs as a `bias_add` stage, standing alone as its own op so the fused member has
            # something to be compared against. It shares `Tensor.add_bias` with the fused path on
            # purpose -- a separate implementation here could differ from the stage it is supposed
            # to be the unfused equivalent of, and the fusion comparison would then be measuring
            # two different arithmetics rather than one lever.
            t = env[ops["src"]].add_bias(env[bias_tensor_name(ops, attrs, op=f"BIAS_ADD {ops.get('dst')!r}")])
            t = _narrow_int_readout(t, str(attrs.get("output_dtype", "i32")), "BIAS_ADD")
            env[ops["dst"]] = t
            outputs[ops["dst"]] = t.to_list()
        elif op == "ATTENTION_QK":
            # S = Q @ K^T for an attention score block: q is [m, d], k is [n, d] (K stored ROW-per-key,
            # so the contraction is over the trailing head dim of BOTH operands), s is [m, n]. This is
            # the SAME definition the command-buffer simulator executes and the same one the golden
            # engine captures (`q @ k.transpose(-2, -1)`), so the two halves of the L0 comparison agree
            # on one computation.
            #
            # Modeled here because the simulator already models it: with the opcode missing from
            # MODELED_OPCODES the reference raised UnmodeledOp on a conformant buffer, so a capsule
            # declaring `op: attention_qk` had NO expressible command buffer at all — the submission
            # could only be graded by abandoning the opcode the interface grammar defines for it.
            q, k, dst = ops["q"], ops["k"], ops["dst"]
            qt, kt = env[q], env[k]
            (m, d), (n, d2) = qt.shape, kt.shape
            if d != d2:
                raise ValueError(f"ATTENTION_QK head-dim mismatch: {q}{qt.shape} vs {k}{kt.shape}")
            k_t = Tensor((d, n), [kt.data[i * d + j] for j in range(d) for i in range(n)], kt.dtype)
            t = _attention_epilogue(qt.matmul(k_t), attrs, default_shift, f"ATTENTION_QK {dst!r}")
            env[dst] = t
            outputs[dst] = t.to_list()
        elif op == "ATTENTION_PV":
            # P is [m, s] and V is [s, d]. Unlike ATTENTION_QK this operation does not transpose its
            # right operand; that distinction is the native ABI, not an optional attribute.
            p, v, dst = ops["p"], ops["v"], ops["dst"]
            pt, vt = env[p], env[v]
            (m, s), (s2, _) = pt.shape, vt.shape
            if s != s2:
                raise ValueError(f"ATTENTION_PV key-count mismatch: {p}{pt.shape} vs {v}{vt.shape}")
            t = _attention_epilogue(pt.matmul(vt), attrs, default_shift, f"ATTENTION_PV {dst!r}")
            env[dst] = t
            outputs[dst] = t.to_list()
        elif op == "CONV2D":
            ifm_name, weight_operand, dst = ops["ifm"], ops["weight"], ops["dst"]
            if weight_operand in resident_dequant:
                src_name, scale_name, axis = resident_dequant[weight_operand]
                weight = env[src_name].dequant_per_channel(env[scale_name], axis)
            else:
                weight_name = resident_source.get(weight_operand, weight_operand)
                weight = env[weight_name]
            t = _conv2d(ifm_name, env[ifm_name], weight_operand, weight, dst, attrs, default_shift)
            env[dst] = t
            outputs[dst] = t.to_list()
        elif op == "BATCHED_MATMUL":
            # O[b] = A[b] @ W[b] over a batch of INDEPENDENT 2-D contractions. The weight differs per
            # batch, so there is no residency to reuse and no COMMIT to evaluate at: this writes its
            # own output the way the other self-contained operations here do.
            #
            # Modelled because the contract admits rank-3 contractions and the lowering loops the
            # batch around the same 2-D kernel. Leaving it out did not make batched work refuse
            # loudly -- it made this engine raise UnmodeledOp, which reads as "grade it on hardware
            # instead" and silently removed the one tier that compares two independent evaluations.
            a_name, w_name, dst = ops["a"], ops["w"], ops["dst"]
            at, wt = env[a_name], env[w_name]
            if len(at.shape) != 3 or len(wt.shape) != 3:
                raise ValueError(
                    f"BATCHED_MATMUL needs two rank-3 operands: {a_name}{at.shape} @ {w_name}{wt.shape}")
            batch, m, kdim = at.shape
            wbatch, k2, n = wt.shape
            if batch != wbatch or kdim != k2:
                raise ValueError(
                    f"BATCHED_MATMUL operands do not contract: {a_name}{at.shape} @ {w_name}{wt.shape}")
            flat: list = []
            for index in range(batch):
                lhs = Tensor((m, kdim), at.data[index * m * kdim:(index + 1) * m * kdim], at.dtype)
                rhs = Tensor((kdim, n), wt.data[index * kdim * n:(index + 1) * kdim * n], wt.dtype)
                flat.extend(lhs.matmul(rhs).data)
            t = Tensor((batch, m, n), flat, "i32")
            env[dst] = t
            outputs[dst] = t.to_list()
        elif op == "MOVEMENT":
            if "src" not in ops:
                raise ValueError("MOVEMENT needs operands src/dst")
            src, dst = ops["src"], ops["dst"]
            value = env[src]
            moved = Tensor(value.shape, list(value.data), str(attrs.get("output_dtype", value.dtype)))
            env[dst] = moved
            outputs[dst] = moved.to_list()
    for name, spec in cb.get("tensors", {}).items():
        if spec.get("role") == "output" and name in env and name not in outputs:
            outputs[name] = env[name].to_list()

    # When the buffer declares its model outputs, surface EXACTLY those — a chained layer's
    # committed output is an intermediate, not a result.
    declared = cb.get("outputs")
    if declared:
        outputs = {k: v for k, v in outputs.items() if k in set(declared)}
    return outputs


def outputs_match(a: dict[str, list], b: dict[str, list]) -> bool:
    """Exact equality of two output maps."""
    return a == b
