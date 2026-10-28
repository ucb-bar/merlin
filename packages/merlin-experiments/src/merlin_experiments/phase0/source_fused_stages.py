"""Fused readout stages OBSERVED in a written Linalg composition program, behind a structural proof.

A composition capsule states a chain as Linalg source: a contraction, then (optionally) an
accumulator bias add, then a readout. Judged one operation at a time, the bias and readout members
are int32-operand elementwise maps, which no standalone declaration admits. Lowered, they are
epilogue stages of the contraction's command, and an interface program states exactly that: its
commit or contraction command carries an ``epilogue`` list, observed by
``merlin.targetgen.interface_observations._stage_rows`` as fused ``elementwise_map`` rows
``composed_with`` the carrier. The reviewed ``fused_accelerator`` declarations judge those rows.

This module produces the SAME stage rows from a source program, and only where the written Linalg
is proven, op by op, to compute what the stage list declares:

* the producer is an integer multiply-accumulate contraction (i8 x i8 -> i32, zero-initialised
  accumulator, matmul indexing) whose result has exactly one use;
* ``bias_add``: an identity-indexed ``addi`` of that accumulator and an i32 tensor of the same shape,
  whose result has exactly one use (the readout);
* ``relu`` / ``acc_scale``: a readout whose body is exactly ``sitofp``, an optional
  ``maximumf(x, 0.0)``, ``mulf`` by a constant per-column f32 gain, ``roundeven``, a clamp to the
  narrow result type's full range and ``fptosi`` to it -- the selected readout semantics
  (``acc_scale_rounding: half_even``, ``narrowing: saturate_to_declared_dtype``) and nothing else.
  A relu applied before the scale commutes with it only for non-negative gains, so a negative gain
  leaves no observation.

Anything else -- another op, another constant, a second use, fast-math flags, a non-constant gain,
other dtypes or maps -- leaves the member's standalone admission untouched, and that refusal stands.
Every produced row is screened by ``admit_operation_row`` exactly as an interface stage row is; this
module admits nothing itself.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

_MATMUL_MAPS = (
    "affine_map<(d0, d1, d2) -> (d0, d2)>",
    "affine_map<(d0, d1, d2) -> (d2, d1)>",
    "affine_map<(d0, d1, d2) -> (d0, d1)>",
)
_IDENTITY2 = "affine_map<(d0, d1) -> (d0, d1)>"
_COLUMN2 = "affine_map<(d0, d1) -> (d1)>"
_SCALAR2 = "affine_map<(d0, d1) -> (0)>"
_PARALLEL = "#linalg.iterator_type<parallel>"
_REDUCTION = "#linalg.iterator_type<reduction>"
_NARROW = {"i8": (-128.0, 127.0)}


def _name(op) -> str:
    from merlin.common import mlir_query as mq

    return mq.op_name(op)


def _tensor(value) -> tuple[list[int], str] | None:
    from merlin.common import mlir_query as mq

    shape, dtype = mq.type_shape_dtype(value.type)
    if not str(value.type).startswith("tensor<") or not shape or any(type(n) is not int or n < 1 for n in shape):
        return None
    return shape, dtype


def _defining_op(value):
    """The op that defines ``value``; ``None`` for a block argument (a runtime input is no constant)."""
    from xdsl.ir import Operation

    owner = getattr(value, "owner", None)
    return owner if isinstance(owner, Operation) else None


def _uses(value) -> list:
    return [use.operation for use in value.uses]


def _plain(op) -> bool:
    """No fast-math relaxation on a float op (absent or ``none``)."""
    flags = op.properties.get("fastmath") or op.attributes.get("fastmath")
    return flags is None or str(flags) == "#arith.fastmath<none>"


def _float_constant(op) -> float | None:
    if _name(op) != "arith.constant":
        return None
    value = op.properties.get("value") or op.attributes.get("value")
    data = getattr(getattr(value, "value", None), "data", None)
    return float(data) if isinstance(data, float) and str(value.type) == "f32" else None


def _dense_f32(value) -> list[float] | None:
    """The values of an ``arith.constant`` dense f32 vector, else ``None``."""
    owner = _defining_op(value)
    if owner is None or _name(owner) != "arith.constant":
        return None
    attr = owner.properties.get("value") or owner.attributes.get("value")
    typed = _tensor(value)
    if typed is None or typed[1] != "f32" or len(typed[0]) != 1:
        return None
    try:
        from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

        if not isinstance(attr, DenseIntOrFPElementsAttr):
            return None
        values = [float(v) for v in attr.get_values()]
    except Exception:  # noqa: BLE001 - an unreadable constant is no proof
        return None
    if len(values) == 1:
        values = values * typed[0][0]
    if len(values) != typed[0][0] or not all(math.isfinite(v) for v in values):
        return None
    return values


def _dense_int(value) -> list[int] | None:
    """The flat values of an ``arith.constant`` dense integer tensor, else ``None`` (a runtime input)."""
    owner = _defining_op(value)
    if owner is None or _name(owner) != "arith.constant":
        return None
    attr = owner.properties.get("value") or owner.attributes.get("value")
    typed = _tensor(value)
    try:
        from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

        if typed is None or not isinstance(attr, DenseIntOrFPElementsAttr):
            return None
        values = [int(v) for v in attr.get_values()]
    except Exception:  # noqa: BLE001 - an unreadable constant is no bound
        return None
    count = math.prod(typed[0])
    if len(values) == 1:
        values = values * count
    return values if len(values) == count else None


def _accumulator_bound(lhs, rhs, k: int, n: int) -> int:
    """Largest |accumulator| an i8 x i8 contraction can reach: exact for a constant operand, else |i8| <= 128."""
    weights = _dense_int(rhs)
    if weights is not None:
        return max(sum(abs(weights[row * n + col]) for row in range(k)) for col in range(n)) * 128
    activations = _dense_int(lhs)
    if activations is not None:
        m = len(activations) // k
        return max(sum(abs(activations[row * k + col]) for col in range(k)) for row in range(m)) * 128
    return 128 * 128 * k


def _generic(op, maps: tuple[str, ...], iterators: tuple[str, ...]) -> bool:
    return (
        _name(op) == "linalg.generic"
        and tuple(str(m) for m in op.indexing_maps) == maps
        and tuple(str(i) for i in op.iterator_types) == iterators
        and len(op.results) == 1
        and len(op.outputs) == 1
    )


def _body(op) -> list:
    return list(op.body.block.ops)


def _fresh_output(op) -> bool:
    """The destination is a fresh ``tensor.empty`` (the body never reads it)."""
    out = op.outputs[0]
    owner = _defining_op(out)
    return owner is not None and _name(owner) == "tensor.empty" and not list(op.body.block.args[-1].uses)


def contraction(op) -> dict | None:
    """An integer multiply-accumulate matmul into a zero-filled i32 accumulator."""
    if not _generic(op, _MATMUL_MAPS, (_PARALLEL, _PARALLEL, _REDUCTION)) or len(op.inputs) != 2:
        return None
    lhs, rhs, result = _tensor(op.inputs[0]), _tensor(op.inputs[1]), _tensor(op.results[0])
    if None in (lhs, rhs, result) or lhs[1] != "i8" or rhs[1] != "i8" or result[1] != "i32":
        return None
    init = _defining_op(op.outputs[0])
    if init is None or _name(init) != "linalg.fill":
        return None
    zero = _defining_op(init.inputs[0]) if init.inputs else None
    if zero is None or _name(zero) != "arith.constant" or str(getattr(zero, "value", "")) != "0 : i32":
        return None
    a, b, acc = op.body.block.args
    body = _body(op)
    if [_name(item) for item in body] != ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"]:
        return None
    ea, eb, mul, add, yld = body
    if ea.operands[0] is not a or eb.operands[0] is not b:
        return None
    if set(map(id, mul.operands)) != {id(ea.results[0]), id(eb.results[0])}:
        return None
    if set(map(id, add.operands)) != {id(mul.results[0]), id(acc)} or yld.operands[0] is not add.results[0]:
        return None
    return {
        "lhs": {"dtype": "i8", "shape": lhs[0]},
        "rhs": {"dtype": "i8", "shape": rhs[0]},
        "result": result[0],
        "bound": _accumulator_bound(op.inputs[0], op.inputs[1], rhs[0][0], rhs[0][1]),
    }


def _column_values(values: list[int] | None, rows: int, columns: int) -> list[int] | None:
    """One value per column when every row of a row-major ``rows x columns`` constant is the same row."""
    if values is None or len(values) != rows * columns:
        return None
    first = values[:columns]
    return first if all(values[r * columns : (r + 1) * columns] == first for r in range(rows)) else None


def bias_add(op, accumulator) -> dict | None:
    """``accumulator + bias`` elementwise in i32: a per-column bias stage, or a full accumulator seed.

    The fused bias stage adds ONE i32 value per output column, so it is observed only for an
    ``[N]`` vector broadcast along M (map ``(d0, d1) -> (d1)``), or an ``[M, N]`` tensor that is
    provably constant along M (``M == 1``, or a constant whose rows are all equal).

    Any other identity-indexed ``[M, N]`` i32 tensor (a runtime input, or a constant whose rows
    differ) is NOT that stage. It is returned as an accumulator ``seed`` instead: the reviewed
    contraction's int32 accumulator seed (one full-width, unscaled i32 load of the whole tensor into
    the accumulator, which the contraction then adds onto). i32 addition wraps in the source and on
    the device, so ``(A @ B) + S`` and ``S`` seeded then accumulated are equal for every ``S``.
    ``bias_bound`` is ``None`` for a runtime seed.
    """
    if len(op.inputs) != 2 or op.inputs[0] is not accumulator or not _fresh_output(op):
        return None
    if _generic(op, (_IDENTITY2,) * 3, (_PARALLEL, _PARALLEL)):
        broadcast = False
    elif _generic(op, (_IDENTITY2, _COLUMN2, _IDENTITY2), (_PARALLEL, _PARALLEL)):
        broadcast = True
    else:
        return None
    acc, bias, result = _tensor(op.inputs[0]), _tensor(op.inputs[1]), _tensor(op.results[0])
    if None in (acc, bias, result) or acc != result or acc[1] != "i32" or bias[1] != "i32" or len(acc[0]) != 2:
        return None
    rows, columns = acc[0]
    values = _dense_int(op.inputs[1])
    seed = False
    if broadcast:
        if bias[0] != [columns]:
            return None
    elif bias[0] != acc[0]:
        return None
    elif rows != 1:
        column = _column_values(values, rows, columns)
        if column is None:
            seed = True
        else:
            values = column
    x, y, _ = op.body.block.args
    body = _body(op)
    if [_name(item) for item in body] != ["arith.addi", "linalg.yield"]:
        return None
    add, yld = body
    if set(map(id, add.operands)) != {id(x), id(y)} or add.operands[0] is add.operands[1]:
        return None
    if yld.operands[0] is not add.results[0]:
        return None
    bound = max(map(abs, values)) if values else None
    if seed:
        source = "runtime_input" if values is None else "constant"
        return {"seed": {"dtype": "i32", "shape": [rows, columns], "source": source}, "bias_bound": bound}
    return {"bias": {"dtype": "i32", "shape": [columns]}, "bias_bound": bound}


def readout(
    op,
    accumulator,
    *,
    numerics: dict,
    strategy: str | None = None,
    bound: int | None = None,
    source_dtype: str = "i32",
    runtime_scale: int | None = None,
) -> dict | None:
    """The selected readout of an i32 accumulator to a narrow integer: [relu,] scale, round half even, saturate.

    ``source_dtype`` names the integer type read (``i32`` for an accumulator; ``i8`` for an operand
    load requantized by the same body, see ``operand_sum_chain``).

    ``runtime_scale`` is the position of the function argument the capsule declares as its runtime
    readout scale (``None``: none). The gain may then be that argument instead of a constant: one
    ``tensor<1xf32>`` read through ``(d0, d1) -> (0)``, under ``f32_product_rne`` only. Its value is
    not known here; the caller passes it only under the reviewed runtime-scale domain (finite and
    >= 0, ``runtime_scale_domain``), where relu-then-scale and the stage list's scale-then-relu agree
    exactly as they do for a non-negative constant gain.

    The body computes ``f32(acc) * gain`` rounded once: that IS the readout only under the
    ``f32_product_rne`` strategy. Under any other (e.g. an exact rational readout) it is equal only
    when ``sitofp`` is exact (a proven ``|acc| <= 2**24``) and every gain is a power of two, so the
    product is exact too; otherwise there is no observation.
    """
    if strategy != "f32_product_rne" and (bound is None or bound > 2**24):
        return None
    if numerics.get("acc_scale_rounding") != "half_even" or numerics.get("narrowing") != "saturate_to_declared_dtype":
        return None
    runtime = runtime_scale is not None and _generic(op, (_IDENTITY2, _SCALAR2, _IDENTITY2), (_PARALLEL, _PARALLEL))
    if not runtime and not _generic(op, (_IDENTITY2, _COLUMN2, _IDENTITY2), (_PARALLEL, _PARALLEL)):
        return None
    if len(op.inputs) != 2 or op.inputs[0] is not accumulator or not _fresh_output(op):
        return None
    acc, result = _tensor(op.inputs[0]), _tensor(op.results[0])
    if acc is None or result is None or acc[1] != source_dtype or result[1] not in _NARROW or acc[0] != result[0]:
        return None
    if runtime:
        scale = op.inputs[1]
        if (
            strategy != "f32_product_rne"
            or not _program_input(scale)
            or scale.index != runtime_scale
            or _tensor(scale) != ([1], "f32")
        ):
            return None
        gains = None
    else:
        gains = _dense_f32(op.inputs[1])
        if gains is None or len(gains) != result[0][-1]:
            return None
    if strategy != "f32_product_rne" and not all(g > 0 and math.frexp(g)[0] == 0.5 for g in gains):
        return None
    lo_bound, hi_bound = _NARROW[result[1]]
    x, gain, _ = op.body.block.args
    body = _body(op)
    names = [_name(item) for item in body]
    relu = names[:3] == ["arith.sitofp", "arith.constant", "arith.maximumf"]
    expected = ["arith.sitofp"] + (["arith.constant", "arith.maximumf"] if relu else []) + [
        "arith.mulf",
        "math.roundeven",
        "arith.constant",
        "arith.constant",
        "arith.maximumf",
        "arith.minimumf",
        "arith.fptosi",
        "linalg.yield",
    ]
    if names != expected or not all(_plain(item) for item in body if _name(item).startswith(("arith.m", "math."))):
        return None
    it = iter(body)
    to_float = next(it)
    if to_float.operands[0] is not x or str(to_float.results[0].type) != "f32":
        return None
    value = to_float.results[0]
    if relu:
        zero, rectify = next(it), next(it)
        if _float_constant(zero) != 0.0 or math.copysign(1.0, _float_constant(zero)) < 0:
            return None
        if set(map(id, rectify.operands)) != {id(value), id(zero.results[0])}:
            return None
        value = rectify.results[0]
        # relu before the scale commutes with it only for non-negative multipliers.
        if any(g < 0 for g in gains or ()):
            return None
    scale, rounding, lo, hi, floor, ceil, narrow, yld = it
    if set(map(id, scale.operands)) != {id(value), id(gain)} or scale.operands[0] is scale.operands[1]:
        return None
    if rounding.operands[0] is not scale.results[0]:
        return None
    if _float_constant(lo) != lo_bound or _float_constant(hi) != hi_bound:
        return None
    if set(map(id, floor.operands)) != {id(rounding.results[0]), id(lo.results[0])}:
        return None
    if set(map(id, ceil.operands)) != {id(floor.results[0]), id(hi.results[0])}:
        return None
    if narrow.operands[0] is not ceil.results[0] or str(narrow.results[0].type) != result[1]:
        return None
    if yld.operands[0] is not narrow.results[0]:
        return None
    proven = {"relu": relu, "gains": gains, "result": {"dtype": result[1], "shape": result[0]}}
    if runtime:
        proven["scale_source"] = "runtime_input"
    return proven


def runtime_scale_domain(spec: dict) -> dict | None:
    """The reviewed domain of a runtime readout-scale input, when it is finite and non-negative.

    ``quantization.numerics.policies.scale_source.runtime_input`` must be the rtl_derived policy and
    carry a reviewed ``domain`` with ``nonfinite: forbid`` and a finite ``minimum >= 0``.
    """
    policy = (
        (((spec.get("quantization") or {}).get("numerics") or {}).get("policies") or {})
        .get("scale_source", {})
        .get("runtime_input")
        or {}
    )
    domain = policy.get("domain") or {}
    minimum = domain.get("minimum")
    if (
        policy.get("value") != "runtime_input"
        or policy.get("evidence") != "rtl_derived"
        or domain.get("status") != "reviewed"
        or domain.get("nonfinite") != "forbid"
        or type(minimum) not in (int, float)
        or not math.isfinite(minimum)
        or minimum < 0
    ):
        return None
    return {"minimum": float(minimum), "nonfinite": "forbid"}


def _program_input(value) -> bool:
    """A function argument: a runtime tensor the program loads, not a value it computes."""
    from xdsl.ir import BlockArgument

    return isinstance(value, BlockArgument) and _name(value.owner.parent_op()) == "func.func"


def _uniform(values: list[float] | None) -> float | None:
    return values[0] if values and len(set(values)) == 1 else None


def _elementwise_pair(op, lhs_dtype: str, result_dtype: str) -> tuple[list[int], object, object] | None:
    """An identity-indexed two-input map over equal static shapes into a fresh destination."""
    if not _generic(op, (_IDENTITY2,) * 3, (_PARALLEL, _PARALLEL)) or len(op.inputs) != 2 or not _fresh_output(op):
        return None
    lhs, rhs, result = _tensor(op.inputs[0]), _tensor(op.inputs[1]), _tensor(op.results[0])
    if None in (lhs, rhs, result) or not (lhs[0] == rhs[0] == result[0]):
        return None
    if lhs[1] != lhs_dtype or rhs[1] != lhs_dtype or result[1] != result_dtype:
        return None
    if op.inputs[0] is op.inputs[1]:
        return None
    x, y, _ = op.body.block.args
    return result[0], x, y


def integer_operand_sum(op) -> list[int] | None:
    """``extsi(a) + extsi(b)``: two i8 tensors summed exactly in i32 (no rounding, no saturation)."""
    pair = _elementwise_pair(op, "i8", "i32")
    if pair is None:
        return None
    shape, x, y = pair
    body = _body(op)
    if [_name(item) for item in body] != ["arith.extsi", "arith.extsi", "arith.addi", "linalg.yield"]:
        return None
    ea, eb, add, yld = body
    if {id(ea.operands[0]), id(eb.operands[0])} != {id(x), id(y)}:
        return None
    if set(map(id, add.operands)) != {id(ea.results[0]), id(eb.results[0])} or yld.operands[0] is not add.results[0]:
        return None
    return shape


def unit_float_operand_sum(op) -> list[int] | None:
    """``fptosi(roundeven(f32(a) * 1.0 + f32(b) * 1.0))`` of two i8 tensors: exactly ``a + b`` in i32.

    Op-exact: two ``sitofp`` to f32, two constants both exactly 1.0, two ``mulf`` (each one distinct
    ``sitofp`` result times one distinct unit constant), the ``addf`` of the two products, the
    ``roundeven`` of that sum, an ``fptosi`` to i32 and the yield; no fast-math flag anywhere; two
    distinct i8 inputs of one static shape, identity maps, a fresh output. Only UNIT multipliers are
    accepted: then every product and the sum (|a + b| <= 256) are exact in f32, the rounding is the
    identity and the integer result is the exact sum. Any other multiplier rounds, which is a
    different computation this proof does not make.
    """
    pair = _elementwise_pair(op, "i8", "i32")
    if pair is None:
        return None
    shape, x, y = pair
    body = _body(op)
    names = [_name(item) for item in body]
    expected = [
        "arith.sitofp",
        "arith.sitofp",
        "arith.constant",
        "arith.constant",
        "arith.mulf",
        "arith.mulf",
        "arith.addf",
        "math.roundeven",
        "arith.fptosi",
        "linalg.yield",
    ]
    if names != expected or not all(_plain(item) for item in body):
        return None
    fa, fb, c0, c1, m0, m1, add, rounding, narrow, yld = body
    if {id(fa.operands[0]), id(fb.operands[0])} != {id(x), id(y)}:
        return None
    if any(str(item.results[0].type) != "f32" for item in (fa, fb)):
        return None
    if _float_constant(c0) != 1.0 or _float_constant(c1) != 1.0:
        return None
    products, units = [], []
    for mul in (m0, m1):
        operands = list(mul.operands)
        if len(operands) != 2 or operands[0] is operands[1]:
            return None
        factor = next((f for f in (fa, fb) if any(o is f.results[0] for o in operands)), None)
        unit = next((c for c in (c0, c1) if any(o is c.results[0] for o in operands)), None)
        if factor is None or unit is None:
            return None
        if {id(o) for o in operands} != {id(factor.results[0]), id(unit.results[0])}:
            return None
        products.append(factor)
        units.append(unit)
    if products[0] is products[1] or units[0] is units[1]:
        return None
    if len(add.operands) != 2 or {id(o) for o in add.operands} != {id(m0.results[0]), id(m1.results[0])}:
        return None
    if rounding.operands[0] is not add.results[0] or narrow.operands[0] is not rounding.results[0]:
        return None
    if str(narrow.results[0].type) != "i32" or yld.operands[0] is not narrow.results[0]:
        return None
    return shape


def operand_sum_chain(op, users, *, numerics: dict, strategy: str | None) -> dict | None:
    """An operand sum of two program inputs read out once: the fused ``residual_add`` form.

    ``op`` is the i32 sum; ``users(value)`` returns a value's sole top-level user (else ``None``).
    Only the ``per_operand`` form is proven: each of two distinct program inputs of one shape is
    requantized ``sat_i8(RNE(f32(x) * g))`` by ONE constant gain (the readout body with an i8
    source and no relu), each result used only by an exact ``extsi + addi``. The sum's only user
    must be a proven readout (one gain per output column). The returned parameters
    are exactly the source program's: they feed the operand-sum numeric witness.

    The readout's gain may differ by output column (the requant gains may not). ``readout_gains`` is
    always the per-column f32 list (length N); ``readout_scale`` is the one gain when it is uniform,
    else ``None``. A column-varying readout only becomes rows through ``operand_sum_rows``, behind a
    reviewed spec row that declares the column-component screen.

    The one other proven form is the UNIT float sum (``unit_float_operand_sum``, exact at unit
    multipliers): ``single_rounding`` with both scales 1.0 and no requantization members, over two
    distinct program inputs. Its sole user must be the same proven readout; the single-rounding
    witness (``single_rounding_readout_witness``) carries that readout gain, so the gain is screened
    under the contract the single-rounding realization binds. Any other float sum is not observed.
    """
    shape = integer_operand_sum(op)
    if shape is None:
        shape = unit_float_operand_sum(op)
        if shape is None or not all(_program_input(value) for value in op.inputs):
            return None
        rounding, members, scales = "single_rounding", [], [1.0, 1.0]
    else:
        rounding, members, scales, sources = "per_operand", [], [], []
        for value in op.inputs:
            requant = _defining_op(value)
            if requant is None or users(value) is not op:
                return None
            proven = readout(
                requant, requant.inputs[0], numerics=numerics, strategy=strategy, bound=128, source_dtype="i8"
            )
            if proven is None or proven["relu"] or proven["result"]["dtype"] != "i8":
                return None
            gain = _uniform(proven["gains"])
            if gain is None or gain <= 0 or not _program_input(requant.inputs[0]):
                return None
            members.append(requant)
            scales.append(gain)
            sources.append(requant.inputs[0])
        if sources[0] is sources[1]:
            return None
    reader = users(op.results[0])
    if reader is None:
        return None
    proven = readout(reader, op.results[0], numerics=numerics, strategy=strategy, bound=256)
    if proven is None or proven["result"]["dtype"] != "i8":
        return None
    readout_gains = list(proven["gains"])
    if rounding == "single_rounding":
        # The unit sum is screened at ONE tensor gain only (no column components).
        gain = _uniform(readout_gains)
        if gain is None or not gain > 0:
            return None
    return {
        "rounding": rounding,
        "requants": members,
        "reader": reader,
        "lhs_scale": scales[0],
        "rhs_scale": scales[1],
        "readout_scale": _uniform(readout_gains),
        "readout_gains": readout_gains,
        "relu": proven["relu"],
        "shape": shape,
    }


#: Largest i8 digit of a K-extension coefficient: ``c * R == sum(d_i * R)`` with ``0 < d_i <= 127``,
#: so every scaled-identity entry is an ordinary signed int8 weight.
_DIGIT_MAX = 127


def extension_digits(coefficient: int) -> list[int]:
    """Additive int8 digits of a positive integer coefficient: as many 127s as fit, then the rest."""
    count = -(-coefficient // _DIGIT_MAX)
    return [_DIGIT_MAX] * (count - 1) + [coefficient - _DIGIT_MAX * (count - 1)]


def k_extension(op, accumulator, *, max_k: int | None, contraction_k: int) -> dict | None:
    """``acc + c * extsi(R)`` in i32: ``R`` an int8 tensor of the accumulator's shape, ``c`` a positive
    integer CONSTANT (an ``arith.constant`` of i32 -- a float path is no integer coefficient).

    That is exactly the contraction ``[A | R ... R] @ [W ; d_1 I ; ... ; d_n I]`` over the extended
    reduction ``K + n * N`` with int8 digits ``d_i`` (``extension_digits``), provided the extended
    reduction fits the facts' ``k_extension_within.max_k``; otherwise there is no observation.
    """
    if max_k is None:
        return None
    if not _generic(op, (_IDENTITY2,) * 3, (_PARALLEL, _PARALLEL)) or len(op.inputs) != 2:
        return None
    if op.inputs[0] is not accumulator or not _fresh_output(op):
        return None
    acc, residual, result = _tensor(op.inputs[0]), _tensor(op.inputs[1]), _tensor(op.results[0])
    if None in (acc, residual, result) or acc[1] != "i32" or residual[1] != "i8" or result[1] != "i32":
        return None
    if not (acc[0] == residual[0] == result[0]) or len(acc[0]) != 2:
        return None
    x, r, _ = op.body.block.args
    body = _body(op)
    names = [_name(item) for item in body]
    if names == ["arith.extsi", "arith.constant", "arith.muli", "arith.addi", "linalg.yield"]:
        widen, constant, mul, add, yld = body
    elif names == ["arith.constant", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"]:
        constant, widen, mul, add, yld = body
    else:
        return None
    if widen.operands[0] is not r or str(widen.results[0].type) != "i32":
        return None
    value = constant.properties.get("value") or constant.attributes.get("value")
    if str(getattr(value, "type", "")) != "i32" or not isinstance(getattr(getattr(value, "value", None), "data", None), int):
        return None
    coefficient = int(value.value.data)
    if coefficient < 1:
        return None
    if set(map(id, mul.operands)) != {id(widen.results[0]), id(constant.results[0])}:
        return None
    if set(map(id, add.operands)) != {id(x), id(mul.results[0])} or add.operands[0] is add.operands[1]:
        return None
    if yld.operands[0] is not add.results[0]:
        return None
    digits = extension_digits(coefficient)
    extended_k = contraction_k + len(digits) * acc[0][1]
    if extended_k > max_k:
        return None
    return {
        "coefficient": coefficient,
        "digits": digits,
        "extended_k": extended_k,
        "residual": {"dtype": "i8", "shape": residual[0]},
        "bound": coefficient * 128,
    }


#: Store-path pooling register fields (facts ``pool_window_within_register_fields``: 2-bit size and
#: stride, so 1..3) and the pooled grid fields (8 bits).
_POOL_FIELD_MAX = 3
_POOL_GRID_MAX = 255
# Channels are not bounded here: pooling and a tensor-wide readout act per column, so a lowering
# splits them into pooled mvouts of at most one block (pooling_on_readout's columns obligation).


def _ints(attr) -> list[int] | None:
    try:
        return [int(v) for v in attr.get_values()]
    except Exception:  # noqa: BLE001 - an unreadable attribute is no proof
        return None


def _reassociation(attr) -> list[list[int]] | None:
    try:
        return [[int(item.value.data) for item in group.data] for group in attr.data]
    except Exception:  # noqa: BLE001
        return None


def _spatial_source(value, only_user):
    """``value``'s producer viewed as a ``[rows * cols, C]`` pooling grid: the readout result itself
    (unpadded; the grid width is then unknown) or ``collapse(pad(expand(readout)))`` with a constant
    zero-valued static pad of the two spatial axes. Returns (readout_value, rows, cols, low, high)."""
    owner = _defining_op(value)
    if owner is None:
        return None
    if _name(owner) != "tensor.collapse_shape":
        return value, None, None, (0, 0), (0, 0)
    pad = _defining_op(owner.operands[0])
    if pad is None or _name(pad) != "tensor.pad" or only_user(pad.results[0]) is not owner:
        return None
    expand = _defining_op(pad.operands[0])
    if expand is None or _name(expand) != "tensor.expand_shape" or only_user(expand.results[0]) is not pad:
        return None
    if _reassociation(owner.properties.get("reassociation")) != [[0, 1], [2]]:
        return None
    if _reassociation(expand.properties.get("reassociation")) != [[0, 1], [2]]:
        return None
    if len(pad.operands) != 1 or len(pad.regions) != 1:
        return None
    low, high = _ints(pad.properties.get("static_low")), _ints(pad.properties.get("static_high"))
    if low is None or high is None or len(low) != 3 or len(high) != 3 or low[2] or high[2] or min(low + high) < 0:
        return None
    region = list(pad.regions[0].block.ops)
    if [_name(item) for item in region] != ["arith.constant", "tensor.yield"]:
        return None
    pad_value = region[0].properties.get("value") or region[0].attributes.get("value") or getattr(region[0], "value", None)
    if str(getattr(pad_value, "type", "")) != "i8" or getattr(getattr(pad_value, "value", None), "data", None) != 0:
        return None
    if region[1].operands[0] is not region[0].results[0]:
        return None
    grid = _tensor(expand.results[0])
    if grid is None or len(grid[0]) != 3:
        return None
    return expand.operands[0], grid[0][0], grid[0][1], (low[0], low[1]), (high[0], high[1])


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def max_pool(op, *, only_user, producer_nonnegative: bool) -> dict | None:
    """A store-path max pool of a readout: ``out[p, c] = max(-128, max over the window of grid[...])``.

    The access map is checked EXHAUSTIVELY against the pooled-store addressing for one (size,
    stride, grid, pad) assignment with size and stride in 1..3; a window may read the low (upper or
    left) zero pad only when the producer is non-negative (a ReLU readout), never the high pad, and
    every window keeps at least one valid tap."""
    if _name(op) != "linalg.generic" or len(op.inputs) != 2 or len(op.results) != 1 or len(op.outputs) != 1:
        return None
    if tuple(str(i) for i in op.iterator_types) != (_PARALLEL, _PARALLEL, _REDUCTION):
        return None
    source = _spatial_source(op.inputs[0], only_user)
    if source is None:
        return None
    readout_value, rows, cols, low, high = source
    grid_t, window_t, result_t = _tensor(op.inputs[0]), _tensor(op.inputs[1]), _tensor(op.results[0])
    if None in (grid_t, window_t, result_t) or grid_t[1] != "i8" or result_t[1] != "i8" or len(window_t[0]) != 1:
        return None
    if len(grid_t[0]) != 2 or len(result_t[0]) != 2 or grid_t[0][1] != result_t[0][1]:
        return None
    if _dense_int(op.inputs[1]) is None:
        return None
    init = _defining_op(op.outputs[0])
    if init is None or _name(init) != "linalg.fill" or not init.inputs:
        return None
    floor = _defining_op(init.inputs[0])
    floor_value = None if floor is None else (floor.properties.get("value") or floor.attributes.get("value"))
    if floor is None or _name(floor) != "arith.constant" or str(getattr(floor_value, "type", "")) != "i8":
        return None
    if getattr(getattr(floor_value, "value", None), "data", None) != -128:
        return None
    empty = _defining_op(init.outputs[0])
    if empty is None or _name(empty) != "tensor.empty":
        return None
    x, w, acc = op.body.block.args
    body = _body(op)
    if [_name(item) for item in body] != ["arith.maxsi", "linalg.yield"] or list(w.uses):
        return None
    if set(map(id, body[0].operands)) != {id(x), id(acc)} or body[1].operands[0] is not body[0].results[0]:
        return None
    try:
        maps = [m.data for m in op.indexing_maps.data]
    except Exception:  # noqa: BLE001
        return None
    outputs, channels = result_t[0]
    taps = window_t[0][0]
    window = math.isqrt(taps)
    if window * window != taps or not 1 <= window <= _POOL_FIELD_MAX:
        return None
    points = [(p, c, t) for p in range(outputs) for c in range(channels) for t in range(taps)]
    try:
        if any(tuple(maps[1].eval([p, c, t], [])) != (t,) for p, c, t in points):
            return None
        if any(tuple(maps[2].eval([p, c, t], [])) != (p, c) for p, c, t in points):
            return None
        reads = {(p, c, t): tuple(maps[0].eval([p, c, t], [])) for p, c, t in points}
    except Exception:  # noqa: BLE001
        return None
    flat = grid_t[0][0]
    if rows is None:  # unpadded: the readout's [rows * cols, C] grid, width unknown
        grids = [(flat // width, width) for width in _divisors(flat)]
    else:
        grids = [(rows + low[0] + high[0], cols + low[1] + high[1])]
        if grids[0][0] * grids[0][1] != flat:
            return None
    found = []
    for grid_rows, grid_cols in grids:
        for stride in range(1, _POOL_FIELD_MAX + 1):
            for pooled_cols in _divisors(outputs):
                ok = all(
                    reads[(p, c, t)]
                    == (
                        ((p // pooled_cols) * stride + t // window) * grid_cols + (p % pooled_cols) * stride + t % window,
                        c,
                    )
                    for p, c, t in points
                )
                if ok:
                    found.append((grid_rows, grid_cols, stride, pooled_cols))
    if len(found) != 1:
        return None
    grid_rows, grid_cols, stride, pooled_cols = found[0]
    orows, ocols = grid_rows - low[0] - high[0], grid_cols - low[1] - high[1]
    pooled_rows = outputs // pooled_cols
    padded = False
    for p in range(outputs):
        valid = 0
        for t in range(taps):
            r = (p // pooled_cols) * stride + t // window - low[0]
            col = (p % pooled_cols) * stride + t % window - low[1]
            if r >= orows or col >= ocols:
                return None  # a high-pad tap: not the pooled-store addressing
            if r < 0 or col < 0:
                padded = True
            else:
                valid += 1
        if valid == 0:
            return None
    if padded and not producer_nonnegative:
        return None
    if max(orows, ocols, pooled_rows, pooled_cols) > _POOL_GRID_MAX or max(low) > _POOL_FIELD_MAX:
        return None
    return {
        "readout": readout_value,
        "result": {"dtype": "i8", "shape": [outputs, channels]},
        "pooled_store": {
            "pool_size": window,
            "pool_stride": stride,
            "orows": orows,
            "ocols": ocols,
            "porows": pooled_rows,
            "pocols": pooled_cols,
            "upad": low[0],
            "lpad": low[1],
            "padded": padded,
            "padding_value": 0 if padded else None,
        },
    }


def prove_window_chains(top, index, only_user, *, numerics: dict, strategy: str | None) -> list[dict]:
    """Every proven movement -> readout -> max pool chain: an int8 FUNCTION ARGUMENT (a tensor the
    movement loads into the accumulator) read out by the selected readout, then max pooled.

    Only under the ``f32_product_rne`` strategy (the reviewed evidence covers that readout only)."""
    if strategy != "f32_product_rne":
        return []
    chains = []
    for i, op in enumerate(top):
        if _name(op) != "linalg.generic" or not op.inputs:
            continue
        source = op.inputs[0]
        if _defining_op(source) is not None or len(_uses(source)) != 1:
            continue
        proven = readout(op, source, numerics=numerics, strategy=strategy, bound=128, source_dtype="i8")
        if proven is None or len(set(proven["gains"])) != 1 or len(proven["result"]["shape"]) != 2:
            continue
        value = op.results[0]
        user = only_user(value)
        while user is not None and _name(user) in ("tensor.expand_shape", "tensor.pad", "tensor.collapse_shape"):
            user = only_user(user.results[0])
        if user is None:
            continue
        pool = max_pool(user, only_user=only_user, producer_nonnegative=proven["relu"])
        if pool is None or pool["readout"] is not value:
            continue
        chains.append(
            {
                "producer": None,
                "carrier": {"operation": "movement", "source": {"dtype": "i8", "shape": proven["result"]["shape"]}},
                "stages": [
                    {"member": i, "stages": (["relu"] if proven["relu"] else []) + ["acc_scale"], **proven},
                    {
                        "member": index[id(user)],
                        "stages": ["maxpool"],
                        "pooled_store": pool["pooled_store"],
                        "result": pool["result"],
                    },
                ],
            }
        )
    return chains


#: Dataflows whose fused readout needs nothing beyond the program's own arithmetic. An
#: output-stationary contraction keeps its partial sums in the mesh: its epilogue is observed only
#: where ``os_value_bound`` proves the reviewed signed-interval bound (any other dataflow: never).
_OBSERVED_DATAFLOWS = (None, "WS")
_OUTPUT_STATIONARY = "OS"


def os_value_bound(op, mesh_mac: dict | None) -> dict | None:
    """The reviewed output-stationary value bound of contraction ``op``, else ``None``.

    Proven only where the selected spec declares the OS dataflow AND the reviewed signed-interval
    mode of the mesh check (``mesh_mac.os_value_bound``): ``require_value_mac_bound`` must then show,
    from the literal weights, the signed i8 input range and the literal initializer, that every
    partial sum of every column (every prefix, in any order and any K split) lies in
    ``[-2**(bits-1), 2**(bits-1) - 1]``. A runtime weight, a non-literal initializer or a bound
    outside that interval leaves no proof.
    """
    mesh = mesh_mac or {}
    declared = mesh.get("os_value_bound") or {}
    if (
        "OS" not in (mesh.get("dataflows") or ())
        or declared.get("status") != "reviewed"
        or declared.get("mode") != "signed_interval"
        or declared.get("proof") != "constant_weight_all_prefix_intervals"
    ):
        return None
    from merlin.targetgen.native_mac_bound import require_value_mac_bound

    # The bound reads the op's own operands (the same SSA values); only the generator's provenance
    # attributes (``prov.*``), which carry no semantics, are dropped from the copy it inspects.
    probe = None
    try:
        probe = op.clone()
        for key in [key for key in probe.attributes if key.startswith("prov.")]:
            del probe.attributes[key]
        return require_value_mac_bound(probe, mesh_signed_bits=mesh.get("signed_bits"))
    except Exception:  # noqa: BLE001 - any refusal (runtime weight, wrap, unreadable IR) is no proof
        return None
    finally:
        if probe is not None:
            probe.drop_all_references()


def _top_level(program: Path):
    """The single function's top-level ops, their indices and a sole-top-level-user lookup."""
    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir

    normalized, _ = normalize_capture_mlir(Path(program).read_text())
    module = mq.parse(normalized)
    functions = [op for op in module.body.block.ops if mq.op_name(op) == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        return None
    top = [op for op in functions[0].body.block.ops if mq.op_name(op) != "func.return"]
    index = {id(op): i for i, op in enumerate(top)}

    def only_user(value):
        users = _uses(value)
        return users[0] if len(users) == 1 and id(users[0]) in index else None

    return top, index, only_user


def prove_chains(
    program: Path,
    *,
    numerics: dict,
    strategy: str | None = None,
    dataflow: str | None = None,
    max_k: int | None = None,
    pooling_fenced: bool = False,
    mesh_mac: dict | None = None,
    full_tensor_seed: bool = False,
    runtime_scale: int | None = None,
    inputs: list[dict] | None = None,
) -> list[dict]:
    """Every proven contraction -> [k_extension | bias_add | seed] -> [readout] chain, and every proven
    movement -> readout -> max pool chain when its scratchpad-source pooling is fenced, by
    top-level source op index.

    Under OS a contraction chain needs ``os_value_bound`` for its producer and carries no K
    extension and no bias (either would enter the partial sums the bound covers; neither is
    observed there), and no movement window chain is observed.

    A full-tensor bias (``bias_add``'s ``seed``) is observed only with ``full_tensor_seed``, the
    selected spec's reviewed ``accumulator_seed_tensor`` declaration; otherwise it ends the chain.
    """
    if dataflow not in _OBSERVED_DATAFLOWS and dataflow != _OUTPUT_STATIONARY:
        return []
    if dataflow == _OUTPUT_STATIONARY:
        max_k, pooling_fenced = None, False
    parsed = _top_level(program)
    if parsed is None:
        return []
    top, index, only_user = parsed
    if runtime_scale is not None and top:
        # The declared runtime scale is a capsule input by NAME; it is located by position only when
        # the function's arguments are exactly the capsule's declared inputs, in order.
        arguments = [_tensor(arg) for arg in top[0].parent_block().args]
        if arguments != [(list(row.get("shape") or []), row.get("dtype")) for row in inputs or []]:
            runtime_scale = None
    chains = []
    for i, op in enumerate(top):
        carrier = contraction(op)
        if carrier is None:
            continue
        if dataflow == _OUTPUT_STATIONARY:
            mesh_bound = os_value_bound(op, mesh_mac)
            if mesh_bound is None:
                continue
            carrier = {**carrier, "os_value_bound": {key: mesh_bound[key] for key in ("method", "mesh_signed_bits")}}
        stages = []
        value = op.results[0]
        user = only_user(value)
        bound = carrier["bound"]
        extension = None
        if user is not None:
            extension = k_extension(user, value, max_k=max_k, contraction_k=carrier["rhs"]["shape"][0])
            if extension is not None:
                bound = bound + extension["bound"]
                stages.append({"member": index[id(user)], "stages": ["k_extension"], "k_extension": extension})
                value = user.results[0]
                user = only_user(value)
        if user is not None and extension is None:
            bias = bias_add(user, value)
            if bias is not None and dataflow == _OUTPUT_STATIONARY:
                continue
            if bias is not None and "seed" in bias and not full_tensor_seed:
                bias = None
                user = None  # an undeclared full-tensor bias ends the chain: no readout through it
            if bias is not None:
                bound = bound + bias["bias_bound"] if bias["bias_bound"] is not None else None
                if "seed" in bias:
                    stages.append({"member": index[id(user)], "stages": [], "seed": bias["seed"]})
                else:
                    stages.append({"member": index[id(user)], "stages": ["bias_add"], "bias": bias["bias"]})
                value = user.results[0]
                user = only_user(value)
        if user is not None:
            # A runtime scale is observed on an unseeded accumulator only: the realization's runtime
            # readout route (fusion_enumeration runtime v1) refuses a seeded, biased or extended
            # contraction.
            unseeded = not any(key in stage for stage in stages for key in ("seed", "bias", "k_extension"))
            proven = readout(
                user,
                value,
                numerics=numerics,
                strategy=strategy,
                bound=bound,
                runtime_scale=runtime_scale if unseeded else None,
            )
            if proven is not None:
                stages.append(
                    {"member": index[id(user)], "stages": (["relu"] if proven["relu"] else []) + ["acc_scale"], **proven}
                )
        if stages:
            chains.append({"producer": i, "carrier": carrier, "stages": stages})
    if pooling_fenced:
        chains += prove_window_chains(top, index, only_user, numerics=numerics, strategy=strategy)
    return chains


def _extension_rows(index: int, carrier: dict, extension: dict) -> list[dict]:
    """The k_extension member observed as what it is proven to equal: its contraction over the
    extended reduction, a fused stage composed with that contraction (``contraction_shape`` carries
    the extended K, which a declaration bounds with ``shape_bounds``)."""
    from merlin.targetgen.interface_observations import _ROW_MAJOR, _row

    m, n = carrier["result"]
    k = extension["extended_k"]
    return [
        _row(
            index,
            "MATMUL",
            "k_extension",
            "contraction",
            disposition="unclassified",
            operand_dtypes=["i8"],
            ordered_operand_types=[{"dtype": "i8", "shape": [m, k]}, {"dtype": "i8", "shape": [k, n]}],
            ordered_result_types=[{"dtype": "i32", "shape": [m, n]}],
            accumulator_dtypes=["i32"],
            result_dtypes=["i32"],
            contraction_shape={"M": m, "K": k, "N": n},
            layout=_ROW_MAJOR,
            aliasing="disjoint_inputs_outputs",
            tails="zero_pad_valid_window",
            broadcasting="none",
            composed_observation={"epilogues": ["k_extension"], "composed_with": ["contraction"], "scale_granularity": None},
        )
    ]


def stage_rows(chain: dict) -> dict[int, list[dict]]:
    """The interface stage rows each proven member of ``chain`` is observed as (not yet judged)."""
    from merlin.runtime.commandbuffer import EPILOGUE_STAGES
    from merlin.targetgen.interface_observations import _stage_rows

    rows_by_member: dict[int, list[dict]] = {}
    stages = chain["stages"]
    # A movement chain (no producer member) reads an int8 tensor loaded into the accumulator; its
    # stages share the pooled store's destination, as a COMMIT's would.
    movement = chain["producer"] is None
    if movement:
        exit_type = stages[-1]["result"]
    else:
        exit_type = next((s["result"] for s in reversed(stages) if "result" in s), None)
        exit_type = exit_type or {"dtype": "i32", "shape": chain["carrier"]["result"]}
    gains = next((s["gains"] for s in stages if "gains" in s), None)
    # The command attributes an interface program states for this chain: its epilogue (in the
    # order a COMMIT applies it) and its multipliers. The logical output-column partition reads
    # the epilogue to decide which stages a per-column gain composes with. Relu is listed after
    # acc_scale (COMMIT order) although the source applies it first: the readout proof admits that
    # order only for non-negative gains, where the two commute.
    declared = {name for stage in stages for name in stage["stages"]}
    attributes: dict = {"epilogue": [name for name in EPILOGUE_STAGES if name in declared]}
    if gains is not None:
        if len(set(gains)) == 1:
            attributes["acc_scale"] = gains[0]
        else:
            attributes["acc_scale_by_column"] = list(gains)
    if exit_type["dtype"] != "i32":
        attributes["output_dtype"] = exit_type["dtype"]
    carrier = chain["carrier"]
    column_source = None
    if not movement:
        extension = next((s["k_extension"] for s in stages if "k_extension" in s), None)
        lhs, rhs = carrier["lhs"], carrier["rhs"]
        if extension is not None:  # the readout drains the contraction over the extended K
            k = extension["extended_k"]
            lhs = {**lhs, "shape": [lhs["shape"][0], k]}
            rhs = {**rhs, "shape": [k, rhs["shape"][1]]}
        column_source = {
            "operation": "matmul",
            "lhs": lhs,
            "rhs": rhs,
            "operand_names": ["lhs", "rhs"],
            "disjoint": True,
            "result": exit_type,
            "bias": next((s["bias"] for s in stages if "bias" in s), None),
        }
    runtime = any(stage.get("scale_source") == "runtime_input" for stage in stages)
    for stage in stages:
        if "seed" in stage:
            continue  # the contraction's own accumulator seed (``observe``), not an epilogue stage
        if "k_extension" in stage:
            rows = _extension_rows(stage["member"], carrier, stage["k_extension"])
            for row in rows:
                row["_observation_extra"] = {
                    "k_extension": {key: stage["k_extension"][key] for key in ("coefficient", "digits", "extended_k")}
                }
            rows_by_member[stage["member"]] = rows
            continue
        rows_by_member[stage["member"]] = _stage_rows(
            index=stage["member"],
            opcode="MOVEMENT" if movement else "MATMUL",
            stages=stage["stages"],
            producer="movement" if movement else "contraction",
            operand_dtype="i8" if movement else carrier["lhs"]["dtype"],
            result_dtype=exit_type["dtype"],
            accumulator="i32",
            destination_declared=True,
            attributes=attributes,
            column_source=column_source,
            result_type=exit_type,
        )
        if runtime:
            # The scale is the declared runtime input, not a command constant (``readout``).
            for row in rows_by_member[stage["member"]]:
                row["composed_observation"] = {**row["composed_observation"], "scale_source": "runtime_input"}
        if "pooled_store" in stage:
            for row in rows_by_member[stage["member"]]:
                row["_observation_extra"] = {"pooled_store": stage["pooled_store"]}
    return rows_by_member


def prove_operand_sums(program: Path, *, numerics: dict, strategy: str | None = None) -> list[dict]:
    """Every proven operand-sum -> readout chain (``operand_sum_chain``), by top-level source op index."""
    parsed = _top_level(program)
    if parsed is None:
        return []
    top, index, only_user = parsed
    chains = []
    for i, op in enumerate(top):
        proven = operand_sum_chain(op, only_user, numerics=numerics, strategy=strategy)
        if proven is None:
            continue
        chains.append(
            {
                **{key: value for key, value in proven.items() if key not in {"requants", "reader"}},
                "sum_members": sorted([index[id(item)] for item in proven["requants"]] + [i]),
                "readout_member": index[id(proven["reader"])],
            }
        )
    return chains


def _single_rounding_witness(chain: dict, evidence) -> dict | None:
    """The single-rounding witness of a unit float sum, only if it screens exactly the source constants."""
    import numpy as np

    from .program_admission import single_rounding_readout_witness

    if chain["lhs_scale"] != 1.0 or chain["rhs_scale"] != 1.0 or len(chain["sum_members"]) != 1:
        return None
    witness = single_rounding_readout_witness(
        lhs_scale=1.0,
        rhs_scale=1.0,
        readout_scale=chain["readout_scale"],
        relu=chain["relu"],
        evidence=evidence,
    )
    if witness is None or witness.get("status") != "within_bound":
        return None
    parameters = witness.get("parameters") or {}
    with np.errstate(over="ignore"):
        gain = float(np.float32(chain["readout_scale"]))
    if (
        parameters.get("lhs_scale_f32") != 1.0
        or parameters.get("rhs_scale_f32") != 1.0
        or parameters.get("readout_gain_f32") != gain
        or parameters.get("relu") is not chain["relu"]
        or "numerical_contract" in witness
    ):
        return None
    return witness


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


#: The reviewed spec row that declares column-component readout screens.
_COLUMN_COMPONENT_ROW = "elementwise_map_after_per_operand_sum"


def _column_components_declared(evidence) -> bool:
    """Whether the selected spec carries the reviewed per-operand row declaring ``column_component``."""
    from merlin.targetgen.operator_review import review_accepted

    spec = getattr(evidence, "software_spec", None) or {}
    declared = [row for row in spec.get("operations") or () if row.get("id") == _COLUMN_COMPONENT_ROW]
    return (
        len(declared) == 1
        and isinstance(declared[0].get("readout_gain_screens"), list)
        and "column_component" in declared[0]["readout_gain_screens"]
        and review_accepted(declared[0], default="unreviewed")
    )


def _column_runs(gains: list[float]) -> list[tuple[int, int, float]]:
    """``(start, stop, f32 gain)`` for each maximal run of equal gains: disjoint, contiguous, covering every column."""
    runs: list[tuple[int, int, float]] = []
    for column, gain in enumerate(_f32(value) for value in gains):
        if runs and runs[-1][2] == gain:
            runs[-1] = (runs[-1][0], column + 1, gain)
        else:
            runs.append((column, column + 1, gain))
    cursor = 0
    for start, stop, _ in runs:
        assert start == cursor and stop > start, "column components must be contiguous and disjoint"
        cursor = stop
    assert cursor == len(gains), "column components must cover every output column exactly once"
    return runs


def _fused_command_rows(chain: dict, shape: list[int], gain: float, witness: dict) -> list[dict] | None:
    """The rows of ONE synthetic fused RESIDUAL_ADD command over ``shape`` at the scalar readout ``gain``.

    The command is only a vehicle for ``command_rows`` (the interface observation of a fused
    command's stage rows): it is never executed or emitted. Each row carries ``witness`` as its
    ``numeric_screen``, and the carrier/stage ordinals of ``chain``.
    """
    from merlin.targetgen.interface_observations import command_rows

    from merlin.runtime.commandbuffer import EPILOGUE_STAGES

    # Epilogue in the order a command applies it (as ``stage_rows`` states it); every gain screened
    # here is positive (a within-bound witness requires it), so a relu before the scale commutes with it.
    declared = {"acc_scale"} | ({"relu"} if chain["relu"] else set())
    stages = [name for name in EPILOGUE_STAGES if name in declared]
    command = {
        "opcode": "RESIDUAL_ADD",
        "operands": {"lhs": "lhs", "rhs": "rhs", "dst": "out"},
        "attributes": {
            "epilogue": stages,
            "lhs_scale": chain["lhs_scale"],
            "rhs_scale": chain["rhs_scale"],
            "acc_scale": gain,
            "operand_sum_rounding": chain["rounding"],
            "bound_lsb": 0,
            "output_dtype": "i8",
        },
    }
    tensors = {name: {"shape": shape, "dtype": "i8"} for name in ("lhs", "rhs", "out")}
    observed = command_rows({"tensors": tensors, "commands": [command]})
    [carrier] = [row for row in observed if not (row.get("composed_observation") or {}).get("composed_with")]
    # A stage reads what its carrier reads: the fused command's two equal-shape operand tensors in
    # one affine sum (proven above: identity maps, equal static shapes, two distinct inputs).
    operands = {key: carrier[key] for key in ("tensor_operands", "expression", "broadcasting") if key in carrier}
    if operands != {"tensor_operands": 2, "expression": "affine", "broadcasting": "none"}:
        return None
    rows = []
    for row in observed:
        if row is carrier:
            rows.append({**row, "ordinals": chain["sum_members"]})
        else:
            rows.append({**row, **operands, "ordinals": [chain["readout_member"]]})
    return [{**row, "numeric_screen": {"command_index": 0, **witness}} for row in rows]


def operand_sum_rows(chain: dict, *, evidence) -> list[dict] | None:
    """The rows an interface program states for ``chain`` as fused operand-sum command(s).

    The command is the per-operand form (both loads RNE-saturated to i8 at the proven operand
    scales, the exact i32 sum) or the unit single-rounding form (the float sum at unit multipliers,
    exactly the i32 sum; its witness is the single-rounding one carrying the readout gain), then the
    proven readout (relu, scale). Rows come from
    ``command_rows`` (the interface observation itself), each RESIDUAL_ADD row carrying the numeric
    witness computed from the source program's own constants, as ``account_interface_text`` attaches
    one. ``None`` when the selected facts give no witness model.

    A readout with ONE gain is one command over the full shape. A per-operand readout whose gain differs by
    output column is observed as disjoint contiguous column components (maximal runs of equal gain
    covering every column exactly once), each a synthetic command over ``[M, stop - start]`` at that
    run's scalar gain with its own exhaustive witness (the command is never executed) and the
    signature field ``output_column_component``. That needs the selected spec's reviewed
    ``elementwise_map_after_per_operand_sum`` row to declare ``readout_gain_screens: column_component``;
    otherwise, or if any component's witness is not within bound or does not carry exactly the
    run's constants, there is no observation (``None``).
    """
    from .program_admission import per_operand_readout_witness

    shape = list(chain["shape"])
    if chain["rounding"] not in {"per_operand", "single_rounding"}:
        return None
    if chain["readout_scale"] is not None:
        if chain["rounding"] == "single_rounding":
            witness = _single_rounding_witness(chain, evidence)
        else:
            witness = per_operand_readout_witness(
                lhs_scale=chain["lhs_scale"],
                rhs_scale=chain["rhs_scale"],
                readout_scale=chain["readout_scale"],
                relu=chain["relu"],
                evidence=evidence,
            )
        return None if witness is None else _fused_command_rows(chain, shape, chain["readout_scale"], witness)
    if chain["rounding"] != "per_operand":
        return None
    gains = chain.get("readout_gains")
    if not gains or len(gains) != shape[-1] or not _column_components_declared(evidence):
        return None
    lhs, rhs = _f32(chain["lhs_scale"]), _f32(chain["rhs_scale"])
    rows: list[dict] = []
    for start, stop, gain in _column_runs(gains):
        witness = per_operand_readout_witness(
            lhs_scale=chain["lhs_scale"],
            rhs_scale=chain["rhs_scale"],
            readout_scale=gain,
            relu=chain["relu"],
            evidence=evidence,
        )
        if witness is None or witness.get("status") != "within_bound":
            return None
        source = (witness.get("parameters") or {}).get("source") or {}
        if (
            source.get("readout_scale") != gain
            or source.get("lhs_scale") != lhs
            or source.get("rhs_scale") != rhs
            or source.get("relu") != chain["relu"]
        ):
            return None
        component = _fused_command_rows(chain, shape[:-1] + [stop - start], gain, witness)
        if component is None:
            return None
        description = {"start": start, "stop": stop, "columns": shape[-1], "readout_scale": gain}
        rows.extend({**row, "output_column_component": dict(description)} for row in component)
    return rows


def observe(
    program: Path,
    entries: list[dict],
    *,
    evidence,
    readout_strategy: str | None = None,
    dataflow: str | None = None,
    max_k: int | None = None,
    pooling_fenced: bool = False,
    runtime_scale: int | None = None,
    inputs: list[dict] | None = None,
) -> list[dict]:
    """``entries`` with each proven fused member's standalone entry replaced by its stage rows' entries.

    ``runtime_scale`` is the function-argument position of the capsule's declared runtime readout
    scale and ``inputs`` its declared inputs; the scale is observed only when the selected spec
    declares the reviewed runtime-scale domain (``runtime_scale_domain``).

    ``max_k`` is the facts' ``k_extension_within.max_k`` and ``pooling_fenced`` the composition's
    ``scratchpad_source_pooling_fenced``; without them no K extension, respectively no movement
    window chain, is observed.

    A full-tensor seed member (only under the reviewed ``accumulator_seed_tensor`` declaration) is
    judged as part of its contraction: its entry is a copy of the producer's own admitted contraction
    entry, renamed ``accumulator_seed`` and naming the seed it carries."""
    import copy

    spec = getattr(evidence, "software_spec", None) or {}
    numerics = (spec.get("numerical_semantics") or {}).get("readout") or {}
    chains = prove_chains(
        program,
        numerics=numerics,
        strategy=readout_strategy,
        dataflow=dataflow,
        max_k=max_k,
        pooling_fenced=pooling_fenced,
        mesh_mac=((spec.get("numerical_semantics") or {}).get("native_unit_contract") or {}).get("mesh_mac"),
        full_tensor_seed=accumulator_seed_tensor(spec),
        runtime_scale=runtime_scale if runtime_scale_domain(spec) is not None else None,
        inputs=inputs,
    )
    by_member: dict[int, list[dict]] = {}
    placed_producers = {
        index
        for entry in entries
        for index in (entry.get("observed_signature") or {}).get("source_op_indices") or []
        if (entry.get("observed_signature") or {}).get("semantic_family") == "contraction"
        and (entry.get("accelerator_admission") or {}).get("status") == "admitted"
    }
    producer_entries = {
        tuple((entry.get("observed_signature") or {}).get("source_op_indices") or []): entry for entry in entries
    }
    for chain in chains:
        # A movement chain has no producer member: its source is a program input the movement loads.
        if chain["producer"] is not None and chain["producer"] not in placed_producers:
            continue
        for stage in chain["stages"]:
            producer = producer_entries.get((chain["producer"],))
            if "seed" not in stage or producer is None:
                continue
            seeded = copy.deepcopy(producer)
            # Named as the seed load, so no count reads it as a second contraction.
            seeded["operation"] = "accumulator_seed"
            seeded["observed_signature"] = {
                **seeded["observed_signature"],
                "observation": "source_accumulator_seed",
                "accumulator_seed": {**stage["seed"], "producer": chain["producer"]},
            }
            by_member[stage["member"]] = [seeded]
        for member, rows in stage_rows(chain).items():
            extras = [row.pop("_observation_extra", {}) for row in rows]
            by_member[member] = [_admitted(row, spec, evidence) for row in rows]
            for item, extra in zip(by_member[member], extras, strict=True):
                item["observed_signature"].update(extra)
    groups = [({member}, rows) for member, rows in by_member.items()]
    if dataflow in _OBSERVED_DATAFLOWS:
        for chain in prove_operand_sums(program, numerics=numerics, strategy=readout_strategy):
            rows = operand_sum_rows(chain, evidence=evidence)
            if rows is None:
                continue
            for members in (chain["sum_members"], [chain["readout_member"]]):
                own = [row for row in rows if row["ordinals"] == members]
                groups.append((set(members), [_admitted(row, spec, evidence) for row in own]))
    return _replace(entries, groups)


def accumulator_seed_tensor(spec: dict) -> bool:
    """Whether the selected spec reviews a whole [M, N] i32 tensor as a contraction's accumulator seed."""
    declared = (spec.get("numerical_semantics") or {}).get("accumulator_seed_tensor") or {}
    return declared.get("status") == "reviewed" and "full_tensor" in (declared.get("forms") or ())


def _admitted(row: dict, spec: dict, evidence) -> dict:
    from merlin.targetgen.operation_accounting import admit_operation_row

    return {
        "operation": row["operation"],
        "count": 1,
        "observed_signature": {**row, "observation": "source_fused_stage"},
        **admit_operation_row(
            row,
            software_spec=spec or None,
            capability_contract=evidence.contract,
            host_capabilities=getattr(evidence, "host_capabilities", None),
        ),
    }


def _replace(entries: list[dict], groups: list[tuple[set[int], list[dict]]]) -> list[dict]:
    """``entries`` with each group's members' standalone entries replaced by the group's rows.

    A group applies only when every entry naming one of its members names members of that group
    alone (so no unproven op's identity is ever dropped). The group's identity goes on its
    least-admitted row, so its members are placed only when EVERY row they carry is admitted on
    the accelerator.
    """
    if not groups:
        return entries
    from .program_admission import _decision

    def indices(entry):
        return set((entry.get("observed_signature") or {}).get("source_op_indices") or [])

    applied = []
    for members, rows in groups:
        touching = [entry for entry in entries if indices(entry) & members]
        if rows and touching and all(indices(entry) <= members for entry in touching):
            applied.append((members, rows))
    result, emitted = [], set()
    for entry in entries:
        group = next((i for i, (members, _) in enumerate(applied) if indices(entry) & members), None)
        if group is None:
            result.append(entry)
            continue
        if group in emitted:
            continue
        emitted.add(group)
        members, rows = applied[group]
        ranked = sorted(rows, key=lambda item: _decision(item, host_only=False).get("placement") == "accelerator")
        ranked[0]["observed_signature"]["source_op_indices"] = sorted(members)
        result.extend(ranked)
    return result
