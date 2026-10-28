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
from pathlib import Path

_MATMUL_MAPS = (
    "affine_map<(d0, d1, d2) -> (d0, d2)>",
    "affine_map<(d0, d1, d2) -> (d2, d1)>",
    "affine_map<(d0, d1, d2) -> (d0, d1)>",
)
_IDENTITY2 = "affine_map<(d0, d1) -> (d0, d1)>"
_COLUMN2 = "affine_map<(d0, d1) -> (d1)>"
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
    """``accumulator + bias`` elementwise in i32, where the bias is ONE i32 value per output column.

    The fused bias stage adds a per-column bias, so the written bias must be exactly that: an
    ``[N]`` vector broadcast along M (map ``(d0, d1) -> (d1)``), or an ``[M, N]`` tensor that is
    provably constant along M (``M == 1``, or a constant whose rows are all equal). A runtime or
    row-varying ``[M, N]`` bias is a different computation and stays unobserved.
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
    if broadcast:
        if bias[0] != [columns]:
            return None
    elif bias[0] != acc[0]:
        return None
    elif rows != 1:
        values = _column_values(values, rows, columns)
        if values is None:
            return None
    x, y, _ = op.body.block.args
    body = _body(op)
    if [_name(item) for item in body] != ["arith.addi", "linalg.yield"]:
        return None
    add, yld = body
    if set(map(id, add.operands)) != {id(x), id(y)} or add.operands[0] is add.operands[1]:
        return None
    if yld.operands[0] is not add.results[0]:
        return None
    return {"bias": {"dtype": "i32", "shape": [columns]}, "bias_bound": max(map(abs, values)) if values else None}


def readout(
    op,
    accumulator,
    *,
    numerics: dict,
    strategy: str | None = None,
    bound: int | None = None,
    source_dtype: str = "i32",
) -> dict | None:
    """The selected readout of an i32 accumulator to a narrow integer: [relu,] scale, round half even, saturate.

    ``source_dtype`` names the integer type read (``i32`` for an accumulator; ``i8`` for an operand
    load requantized by the same body, see ``operand_sum_chain``).

    The body computes ``f32(acc) * gain`` rounded once: that IS the readout only under the
    ``f32_product_rne`` strategy. Under any other (e.g. an exact rational readout) it is equal only
    when ``sitofp`` is exact (a proven ``|acc| <= 2**24``) and every gain is a power of two, so the
    product is exact too; otherwise there is no observation.
    """
    if strategy != "f32_product_rne" and (bound is None or bound > 2**24):
        return None
    if numerics.get("acc_scale_rounding") != "half_even" or numerics.get("narrowing") != "saturate_to_declared_dtype":
        return None
    if not _generic(op, (_IDENTITY2, _COLUMN2, _IDENTITY2), (_PARALLEL, _PARALLEL)) or len(op.inputs) != 2:
        return None
    if op.inputs[0] is not accumulator or not _fresh_output(op):
        return None
    acc, result = _tensor(op.inputs[0]), _tensor(op.results[0])
    if acc is None or result is None or acc[1] != source_dtype or result[1] not in _NARROW or acc[0] != result[0]:
        return None
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
        if any(g < 0 for g in gains):
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
    return {
        "relu": relu,
        "gains": gains,
        "result": {"dtype": result[1], "shape": result[0]},
    }


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


def operand_sum_chain(op, users, *, numerics: dict, strategy: str | None) -> dict | None:
    """An operand sum of two program inputs read out once: the fused ``residual_add`` form.

    ``op`` is the i32 sum; ``users(value)`` returns a value's sole top-level user (else ``None``).
    Only the ``per_operand`` form is proven: each of two distinct program inputs of one shape is
    requantized ``sat_i8(RNE(f32(x) * g))`` by ONE constant gain (the readout body with an i8
    source and no relu), each result used only by an exact ``extsi + addi``. The sum's only user
    must be a proven readout with ONE constant gain (tensor granularity). The returned parameters
    are exactly the source program's: they feed the operand-sum numeric witness.

    A float (single-rounding) sum is not observed, not even at unit multipliers where it is exact:
    its realization binds the single-rounding contract, whose witness carries no readout gain.
    """
    shape = integer_operand_sum(op)
    if shape is None:
        return None
    members, scales, sources = [], [], []
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
    gain = _uniform(proven["gains"])
    if gain is None:
        return None
    return {
        "rounding": "per_operand",
        "requants": members,
        "reader": reader,
        "lhs_scale": scales[0],
        "rhs_scale": scales[1],
        "readout_scale": gain,
        "relu": proven["relu"],
        "shape": shape,
    }


#: Dataflows whose fused readout needs nothing beyond the program's own arithmetic. An
#: output-stationary contraction keeps its partial sums in the mesh, whose bound Phase 0 does not
#: prove, so its epilogue is not observed (any other or unknown dataflow likewise).
_OBSERVED_DATAFLOWS = (None, "WS")


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
    program: Path, *, numerics: dict, strategy: str | None = None, dataflow: str | None = None
) -> list[dict]:
    """Every proven contraction -> [bias_add] -> [readout] chain, by top-level source op index."""
    if dataflow not in _OBSERVED_DATAFLOWS:
        return []
    parsed = _top_level(program)
    if parsed is None:
        return []
    top, index, only_user = parsed
    chains = []
    for i, op in enumerate(top):
        carrier = contraction(op)
        if carrier is None:
            continue
        stages = []
        value = op.results[0]
        user = only_user(value)
        bound = carrier["bound"]
        if user is not None:
            bias = bias_add(user, value)
            if bias is not None:
                bound = bound + bias["bias_bound"] if bias["bias_bound"] is not None else None
                stages.append({"member": index[id(user)], "stages": ["bias_add"], "bias": bias["bias"]})
                value = user.results[0]
                user = only_user(value)
        if user is not None:
            proven = readout(user, value, numerics=numerics, strategy=strategy, bound=bound)
            if proven is not None:
                stages.append(
                    {"member": index[id(user)], "stages": (["relu"] if proven["relu"] else []) + ["acc_scale"], **proven}
                )
        if stages:
            chains.append({"producer": i, "carrier": carrier, "stages": stages})
    return chains


def stage_rows(chain: dict) -> dict[int, list[dict]]:
    """The interface stage rows each proven member of ``chain`` is observed as (not yet judged)."""
    from merlin.runtime.commandbuffer import EPILOGUE_STAGES
    from merlin.targetgen.interface_observations import _stage_rows

    rows_by_member: dict[int, list[dict]] = {}
    stages = chain["stages"]
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
    bias = next((s["bias"] for s in stages if "bias" in s), None)
    column_source = {
        "operation": "matmul",
        "lhs": carrier["lhs"],
        "rhs": carrier["rhs"],
        "operand_names": ["lhs", "rhs"],
        "disjoint": True,
        "result": exit_type,
        "bias": bias,
    }
    for stage in stages:
        rows_by_member[stage["member"]] = _stage_rows(
            index=stage["member"],
            opcode="MATMUL",
            stages=stage["stages"],
            producer="contraction",
            operand_dtype=carrier["lhs"]["dtype"],
            result_dtype=exit_type["dtype"],
            accumulator="i32",
            destination_declared=True,
            attributes=attributes,
            column_source=column_source,
            result_type=exit_type,
        )
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


def operand_sum_rows(chain: dict, *, evidence) -> list[dict] | None:
    """The rows an interface program states for ``chain`` as ONE fused operand-sum command.

    The command is the per-operand form: both loads RNE-saturated to i8 at the proven operand
    scales, the exact i32 sum, then the proven readout (relu, scale). Rows come from
    ``command_rows`` (the interface observation itself), each RESIDUAL_ADD row carrying the numeric
    witness computed from the source program's own constants, as ``account_interface_text`` attaches
    one. ``None`` when the selected facts give no witness model.
    """
    from merlin.targetgen.interface_observations import command_rows

    from .program_admission import per_operand_readout_witness

    witness = per_operand_readout_witness(
        lhs_scale=chain["lhs_scale"],
        rhs_scale=chain["rhs_scale"],
        readout_scale=chain["readout_scale"],
        relu=chain["relu"],
        evidence=evidence,
    )
    if witness is None:
        return None
    shape = list(chain["shape"])
    from merlin.runtime.commandbuffer import EPILOGUE_STAGES

    # Epilogue in the order a command applies it (as ``stage_rows`` states it); the proven gain is
    # positive, so a relu before the scale commutes with it.
    declared = {"acc_scale"} | ({"relu"} if chain["relu"] else set())
    stages = [name for name in EPILOGUE_STAGES if name in declared]
    command = {
        "opcode": "RESIDUAL_ADD",
        "operands": {"lhs": "lhs", "rhs": "rhs", "dst": "out"},
        "attributes": {
            "epilogue": stages,
            "lhs_scale": chain["lhs_scale"],
            "rhs_scale": chain["rhs_scale"],
            "acc_scale": chain["readout_scale"],
            "operand_sum_rounding": "per_operand",
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


def observe(
    program: Path,
    entries: list[dict],
    *,
    evidence,
    readout_strategy: str | None = None,
    dataflow: str | None = None,
) -> list[dict]:
    """``entries`` with each proven fused member's standalone entry replaced by its stage rows' entries."""
    spec = getattr(evidence, "software_spec", None) or {}
    numerics = (spec.get("numerical_semantics") or {}).get("readout") or {}
    chains = prove_chains(program, numerics=numerics, strategy=readout_strategy, dataflow=dataflow)
    by_member: dict[int, list[dict]] = {}
    placed_producers = {
        index
        for entry in entries
        for index in (entry.get("observed_signature") or {}).get("source_op_indices") or []
        if (entry.get("observed_signature") or {}).get("semantic_family") == "contraction"
        and (entry.get("accelerator_admission") or {}).get("status") == "admitted"
    }
    for chain in chains:
        if chain["producer"] not in placed_producers:
            continue
        for member, rows in stage_rows(chain).items():
            by_member[member] = [_admitted(row, spec, evidence) for row in rows]
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
