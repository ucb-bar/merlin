"""Independent reference recomputation for correctness checking.

The simulator executes the *residency* path (pack a weight once, reuse it). The reference
recomputes each committed output directly from the leaf inputs, bypassing the resident store
entirely. Residency is a performance optimization that must not change results, so the two
must agree -- that equality is the correctness guarantee the execution tests assert.
"""
from __future__ import annotations

from typing import Any

from .commandbuffer import materialize_inputs
from .tensor import Tensor


#: Opcodes this reference engine has a definition for. An opcode outside this set (and outside
#: :data:`NO_NUMERIC_EFFECT`) is NOT silently skipped: a buffer whose result depends on it cannot be
#: recomputed here, and dropping it produced an EMPTY output map that downstream read as "the kernel never
#: wrote its output" — indistinguishable from a real dropped store, and unfixable by any submission.
#: Fail closed instead (see :class:`UnmodeledOp`).
MODELED_OPCODES = frozenset({
    "RES_PACK", "MATMUL_RESIDENT", "MATMUL", "COMMIT", "VECTOR_MAP", "VREDUCE",
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
            if stage in ("bias_add", "bias"):
                bias_name = ops.get("bias")
                if bias_name is not None:
                    t = t.add_bias(env[bias_name])
            elif stage == "requant":
                t = t.requant(shift)
            elif stage == "acc_scale":
                t = t.requant_acc_scale(float(attrs.get("acc_scale", 1.0)))
            elif stage == "relu":
                t = t.relu()
        if attrs.get("output_dtype", "i8") == "i8":
            t = t.to_i8()
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
