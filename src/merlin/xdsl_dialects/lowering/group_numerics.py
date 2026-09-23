"""The integer arithmetic a closed compute group commits to, read from the group itself.

A fake-quantized layer says, in floating point,

    y = quantize( activation( contract(dequantize(x), dequantize(w)) + bias ), s_out )

and a closed integer group runs it as

    acc = sum x_q * w_q                         (integer accumulator)
    acc = acc + bias_q                          bias_q = roundeven(bias / (s_x * s_w)), computed ONCE
    y   = clamp(roundeven(acc * M), lo, hi)     M = s_x * s_w / s_out, held by the readout
    y   = max(y, 0)                             where the group carries a relu

The two agree up to the rounding of the bias into accumulator units and of one single-precision
product, and they agree on which side of zero a value lands because ``M`` is positive, so applying
the activation after the scale is the same function as applying it before.

This module reads ``s_x``, ``s_w``, ``s_out``, the zero points, the clamp, the activation and
where the bias comes from, out of the group's own operations. It states the result as the
repository's existing epilogue-candidate type, so the capability DERIVED for the target
(:func:`merlin.targetgen.readout_facet.epilogue_capability`) can admit or refuse it with its own
reason codes. What is constant (every per-tensor scale, hence ``M``) is a number here; what is a
stored tensor (the bias, a per-channel scale) is named by the model argument that carries it, so an
offline prepack step knows exactly what to compute and from what.

Nothing here knows a target. A group whose numerics cannot be read is refused with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from merlin.common import mlir_query as mq

from . import compute_groups as CG

SCHEMA = "group_numerics_v1"


class GroupNumericsError(ValueError):
    """The group's integer arithmetic cannot be read from its operations; says what was missing."""


@dataclass(frozen=True)
class ScaleSource:
    """One scale: a compile-time number, or the model argument that stores it."""

    value: float | None = None
    arg_index: int | None = None
    zero_point: int | None = 0

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "arg_index": self.arg_index, "zero_point": self.zero_point}


@dataclass(frozen=True)
class GroupNumerics:
    input: ScaleSource
    weight: ScaleSource
    output: ScaleSource
    #: ``s_x * s_w / s_out`` when all three are compile-time numbers, else ``None``.
    multiplier: float | None
    #: The accumulator-domain bias rule: ``bias_q = roundeven(bias[arg] / divisor)``.
    bias_arg_index: int | None
    bias_divisor: float | None
    activation: str
    clamp: tuple[int, int]
    granularity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "input": self.input.to_dict(),
            "weight": self.weight.to_dict(),
            "output": self.output.to_dict(),
            "multiplier": self.multiplier,
            "bias": (
                None
                if self.bias_arg_index is None
                else {
                    "arg_index": self.bias_arg_index,
                    "divisor": self.bias_divisor,
                    "rule": "roundeven(bias / divisor), in the accumulator's integer domain",
                }
            ),
            "activation": self.activation,
            "clamp": list(self.clamp),
            "granularity": self.granularity,
            "stages": list(stages(self)),
        }


def _constant_of(value) -> float | int | None:
    """The number behind a scalar SSA value: a constant, possibly splatted to a rank-0 tensor."""
    for _ in range(4):
        owner = getattr(value, "owner", None)
        if owner is None or not hasattr(owner, "operands"):
            return None
        name = mq.op_name(owner)
        if name == "arith.constant":
            # Not `a or b`: an integer attribute holding zero is falsy, and a zero point is zero.
            raw = owner.properties.get("value")
            if raw is None:
                raw = owner.attributes.get("value")
            data = getattr(getattr(raw, "value", None), "data", None)
            if isinstance(data, (int, float)) and not isinstance(data, bool):
                return data
            values = getattr(raw, "get_values", None)  # a dense constant of one element
            listed = list(values()) if callable(values) else []
            return listed[0] if len(listed) == 1 and isinstance(listed[0], (int, float)) else None
        if name != "tensor.splat" or not owner.operands:
            return None
        value = owner.operands[0]
    return None


def _arg_index(value) -> int | None:
    from xdsl.ir import BlockArgument

    return int(value.index) if isinstance(value, BlockArgument) else None


def _scale_source(op) -> ScaleSource:
    """Scale and zero point of one quantize or dequantize operation."""
    operands = list(op.operands)
    if len(operands) < 3:
        raise GroupNumericsError(f"{mq.op_name(op)} carries no scale and zero-point operands")
    scale, zero = _constant_of(operands[1]), _constant_of(operands[2])
    if scale is not None:
        if not isinstance(zero, int):
            raise GroupNumericsError(f"{mq.op_name(op)} has a constant scale and no constant zero point")
        return ScaleSource(value=float(scale), zero_point=int(zero))
    index = _arg_index(operands[1])
    if index is None:
        raise GroupNumericsError(
            f"the scale of {mq.op_name(op)} is neither a constant nor a model argument, so nothing can say what it is"
        )
    return ScaleSource(arg_index=index, zero_point=None)


def _int_attr(op, key: str) -> int | None:
    for table in mq._attr_tables(op):
        data = getattr(getattr(table.get(key), "value", None), "data", None)
        if isinstance(data, int):
            return data
    return None


def numerics_of(group: CG.Group) -> GroupNumerics:
    """Read a closed integer group's arithmetic. Raises :class:`GroupNumericsError` otherwise."""
    if group.root is None or group.placement == CG.HOST:
        raise GroupNumericsError("a host region commits to no integer arithmetic")
    if group.window_mean is not None:
        raise GroupNumericsError(
            "a window mean has no stored operand to requantize against; its one multiplier is stated "
            "on the group (Group.window_mean)"
        )
    if group.operand_sum is not None:
        raise GroupNumericsError(
            "an operand sum has no contraction to requantize; its two multipliers and its bound are "
            "stated on the group (Group.operand_sum)"
        )
    kinds = [CG.classify(member) for member in group.members]
    by_kind: dict[str, list[Any]] = {}
    for member, stage in zip(group.members, kinds):
        by_kind.setdefault(stage.kind if stage else "", []).append(member)
    if CG.QUANTIZE not in by_kind:
        raise GroupNumericsError(
            "the group is not closed: no quantize ends it, so its result is "
            "the accumulator and there is no requantization to state"
        )
    dequantizes = by_kind.get(CG.DEQUANTIZE) or []
    if len(dequantizes) != 2:
        raise GroupNumericsError(
            f"expected the two dequantizes behind the contraction's operands, found {len(dequantizes)}"
        )
    chains = [CG._input_chain(operand) for operand in list(group.root.operands)[:2]]
    behind = [chain[1] for chain in chains]
    if any(op is None for op in behind):
        raise GroupNumericsError("an operand of the contraction is not a dequantized integer tensor")
    # By operand position. Which of the two is the stored tensor does not change the arithmetic:
    # the accumulator's unit is the PRODUCT of the two scales, and a first layer's activation is a
    # model argument just as its weight is, so "comes from an argument" does not tell them apart.
    activation_dq, weight_dq = behind
    source_in, source_w = _scale_source(activation_dq), _scale_source(weight_dq)
    quantize = by_kind[CG.QUANTIZE][0]
    source_out = _scale_source(quantize)
    for label, source in (("input", source_in), ("weight", source_w), ("output", source_out)):
        if source.zero_point not in (0, None):
            raise GroupNumericsError(
                f"the {label} zero point is {source.zero_point}; an "
                f"accumulator-domain bias and a single multiplier assume 0"
            )
    low, high = _int_attr(quantize, "quant_min"), _int_attr(quantize, "quant_max")
    if low is None or high is None:
        raise GroupNumericsError("the closing quantize declares no quant_min / quant_max")

    numbers = [s.value for s in (source_in, source_w, source_out)]
    multiplier = divisor = None
    if source_in.value is not None and source_w.value is not None:
        divisor = float(source_in.value) * float(source_w.value)
        if source_out.value is not None:
            multiplier = divisor / float(source_out.value)
    bias_index = None
    for member in by_kind.get(CG.BIAS_ADD) or ():
        indices = [_arg_index(operand) for operand in member.operands]
        bias_index = next((i for i in indices if i is not None), None)
        if bias_index is None:
            raise GroupNumericsError(
                "the bias is not a model argument, so an offline step cannot "
                "be told what to fold into the accumulator's domain"
            )
    return GroupNumerics(
        input=source_in,
        weight=source_w,
        output=source_out,
        multiplier=multiplier,
        bias_arg_index=bias_index,
        bias_divisor=divisor if bias_index is not None else None,
        activation="relu" if CG.RELU in by_kind else "none",
        clamp=(int(low), int(high)),
        granularity=group.scale_granularity or ("tensor" if None not in numbers else "unknown"),
    )


def stages(numerics: GroupNumerics) -> tuple[str, ...]:
    """The group's readout stages in the epilogue contract's vocabulary and order."""
    return (
        (("bias_i32",) if numerics.bias_arg_index is not None else ())
        + ("scale_f32", "round_to_nearest_even", "clamp")
        + (("relu",) if numerics.activation == "relu" else ())
    )


def candidate(numerics: GroupNumerics, capability_name: str, *, bias: tuple[int, ...] = ()):
    """The epilogue candidate a target's derived capability is asked to admit.

    ``bias`` is the accumulator-domain bias an offline step computed (``prepack_bias``); a group
    with a bias and no folded values yet is stated with a single placeholder so the STRUCTURE can
    be checked before the weights are read.
    """
    from merlin.perf.quantization_contract import QuantizedEpilogueCandidate

    if numerics.multiplier is None:
        raise GroupNumericsError(
            "the requantization multiplier is not a compile-time number "
            f"(granularity {numerics.granularity}); a per-axis candidate needs "
            f"the stored scales"
        )
    has_bias = numerics.bias_arg_index is not None
    return QuantizedEpilogueCandidate(
        capability=capability_name,
        scale_granularity="per_tensor",
        scale_axis=None,
        scales=(float(numerics.multiplier),),
        bias_domain="accumulator" if has_bias else "none",
        bias=(tuple(int(v) for v in bias) or (0,)) if has_bias else (),
        output_zero_point=int(numerics.output.zero_point or 0),
        ordered_stages=stages(numerics),
        rounding="round_to_nearest_even",
        saturation=numerics.clamp,
        activation=numerics.activation,
    )


def prepack_bias(numerics: GroupNumerics, bias_values) -> list[int]:
    """The accumulator-domain bias: ``roundeven(b / (s_x * s_w))``. Offline, once per model."""
    if numerics.bias_divisor is None:
        raise GroupNumericsError("the group has no bias, or its divisor is not a compile-time number")
    return [int(round(float(value) / numerics.bias_divisor)) for value in bias_values]


def reference_integer(numerics: GroupNumerics, accumulator, bias_q=None):
    """What the closed group computes from an integer accumulator, as numpy (the contract's order)."""
    import numpy as np

    acc = np.asarray(accumulator, dtype=np.int64)
    if bias_q is not None:
        acc = acc + np.asarray(bias_q, dtype=np.int64)
    scaled = np.float32(numerics.multiplier) * acc.astype(np.float32)
    out = np.clip(np.rint(scaled), numerics.clamp[0], numerics.clamp[1])
    if numerics.activation == "relu":
        out = np.maximum(out, 0)
    return out.astype(np.int64)


def reference_fake_quant(numerics: GroupNumerics, accumulator, bias=None):
    """What the captured floating-point graph computes for the same integers."""
    import numpy as np

    real = np.asarray(accumulator, dtype=np.float64) * (numerics.input.value * numerics.weight.value)
    if bias is not None:
        real = real + np.asarray(bias, dtype=np.float64)
    if numerics.activation == "relu":
        real = np.maximum(real, 0.0)
    return np.clip(np.rint(real / numerics.output.value), numerics.clamp[0], numerics.clamp[1]).astype(np.int64)
