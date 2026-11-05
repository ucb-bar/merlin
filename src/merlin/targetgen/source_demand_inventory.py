"""Join routing demands to complete, independently numbered source operations."""

from merlin.common import mlir_query as mq


def indexed_demand_operations(operations):
    """Retain full source indices while omitting only recognized fill initializers.

    The demand reader omits carriers tagged ``fill``. Those are not a license to
    omit arbitrary arithmetic: only a literal constant, its tensor splat, or an
    actual linalg fill can be outside the routing census. Callers must still
    account for *all* operations in source bindings and task ownership.

    Untagged operations remain in this inventory. If the demand reader missed
    one, exact operation identity/count checks must refuse the incomplete join.
    """
    indexed = []
    for index, operation in enumerate(operations):
        name = mq.op_name(operation)
        tag = mq.attr_str(operation, "prov.op")
        if tag == "fill" and name not in ("arith.constant", "tensor.splat", "linalg.fill"):
            raise ValueError(f"source operation {index} ({name}) has an unsupported omitted fill carrier")
        if name == "linalg.fill" or tag == "fill":
            continue
        indexed.append((index, operation))
    return indexed


def _declared_data_operand_format(op) -> str | None:
    """Observe a declared view/quantizer's data role, without granting support.

    Quantizer scales and zero points are metadata operands, not alternate data
    precisions. Unknown operations, malformed declarations and dynamic data
    stay unknown. No requested compile dtype participates in this observation.
    """
    from math import prod

    from xdsl.dialects.builtin import Float32Type, IntegerType, TensorType

    from merlin.common import mlir_query as mq
    from merlin.common import quant_formats as qf

    if op.regions or op.successors or len(op.results) != 1:
        return None

    def static_tensor(value):
        typ = value.type
        return typ if isinstance(typ, TensorType) and all(dim > 0 for dim in typ.get_shape()) else None

    if not op.operands:
        return None
    data, result = static_tensor(op.operands[0]), static_tensor(op.results[0])
    if data is None or result is None:
        return None
    name = mq.op_name(op)
    if name in {"tensor.collapse_shape", "tensor.expand_shape", "tensor.cast"}:
        if (len(op.operands) != 1 or data.element_type != result.element_type
                or prod(data.get_shape()) != prod(result.get_shape())):
            return None
    elif name in {"quant_ext.quantize_per_tensor", "quant_ext.dequantize_per_tensor"}:
        if len(op.operands) != 3 or data.get_shape() != result.get_shape():
            return None
        scale, zero = (static_tensor(value) for value in op.operands[1:])
        if (scale is None or zero is None or scale.get_shape() or zero.get_shape()
                or not isinstance(scale.element_type, Float32Type)
                or not isinstance(zero.element_type, IntegerType) or zero.element_type.width.data != 64):
            return None
        floating, integer = ((data, result) if name == "quant_ext.quantize_per_tensor" else (result, data))
        if not isinstance(floating.element_type, Float32Type) or not isinstance(integer.element_type, IntegerType):
            return None
    else:
        return None
    spelling = str(data.element_type)
    return qf.get(spelling).name if qf.has(spelling) else None


def _observe_declared_data_formats(module, demands):
    """Join only previously absent observations to the exact original carriers."""
    from dataclasses import replace

    from merlin.common import mlir_query as mq
    from .capsule_source import ModelDemandIncomplete

    carriers = [op for op in mq.walk(module)
                if mq.attr_str(op, "prov.op") not in {None, "fill"}]
    if len(carriers) != len(demands):
        raise ModelDemandIncomplete("declared operand observation lacks complete ordered carrier identity")
    observed = []
    for op, demand in zip(carriers, demands, strict=True):
        if (mq.op_name(op) != demand.carrier_op
                or mq.attr_str(op, "prov.region_id") != demand.region_id):
            raise ModelDemandIncomplete("declared operand observation names another source carrier")
        fmt = _declared_data_operand_format(op) if demand.captured_input_formats == () else None
        observed.append(replace(demand, elem_fmt=fmt, captured_input_formats=(fmt,)) if fmt is not None else demand)
    return observed


def model_op_demands_checked(
    linalg_mlir: str, in_fmt: str, weight_fmt: str | None = None, *, source_kind: str = "captured_provenance_v1"
) -> list:
    """Verify complete, ordered contraction identity and shapes.

    Captured mode requires provenance plus independent body/shape agreement.
    Typed mode reads exact initialized integer dots without provenance or admission.
    """
    from .capsule_source import ModelDemandIncomplete, model_op_demands

    if type(source_kind) is not str or source_kind != "captured_provenance_v1":
        from .model_coverage import typed_static_integer_demands

        return typed_static_integer_demands(linalg_mlir, in_fmt, weight_fmt, source_kind=source_kind)
    from collections import Counter
    from math import prod

    from merlin.common import mlir_query as mq
    from merlin.compile.analysis import contraction_shapes as ks
    from merlin.xdsl_dialects.lowering import contraction_coverage as cc

    try:
        module = mq.parse(linalg_mlir)
    except Exception as exc:
        raise ModelDemandIncomplete(f"cannot parse whole-model MLIR for demand completeness: {exc}") from exc

    structural = []
    for ordinal, op in enumerate(mq.walk(module)):
        name = mq.op_name(op)
        if name in cc.MATMUL_OPS:
            structural.append((ordinal, op, name))
        elif name == "linalg.generic":
            try:
                if cc.classify_generic(op) == "contraction":
                    structural.append((ordinal, op, name))
            except Exception as exc:
                raise ModelDemandIncomplete(f"cannot classify linalg.generic at operation {ordinal}: {exc}") from exc

    # The shape observer is independent of the provenance-tag reader. It deliberately may decline a
    # structural contraction whose maps/extent are undecidable; that is an explicit refusal here.
    observed = {id(op): shape for op, shape in ks.observe_contractions(module)}
    expected: Counter = Counter()
    expected_sequence: list[tuple[str, int, int, int, int]] = []
    unshaped: list[str] = []
    untagged: list[str] = []
    for ordinal, op, name in structural:
        marker = f"operation {ordinal} ({name})"
        tag = getattr((op.attributes or {}).get("prov.op"), "data", None)
        if not isinstance(tag, str) or not tag:
            untagged.append(marker)
            continue
        shape = observed.get(id(op))
        if shape is None or len(shape.parallel) < 2 or not shape.reduction:
            unshaped.append(f"{marker}, prov.op={tag!r}")
            continue
        parallel = tuple(int(x) for x in shape.parallel)
        reduction = tuple(int(x) for x in shape.reduction)
        identity = (tag, prod(parallel[:-2]), parallel[-2], prod(reduction), parallel[-1])
        expected[identity] += 1
        expected_sequence.append(identity)

    if untagged or unshaped:
        raise ModelDemandIncomplete(
            f"structural contraction inventory incomplete: {len(structural)} contraction(s); "
            f"untagged={untagged[:8]}, unshaped={unshaped[:8]}"
        )

    demands = _observe_declared_data_formats(module, model_op_demands(linalg_mlir, in_fmt, weight_fmt))
    actual: Counter = Counter()
    actual_sequence: list[tuple[str, int, int, int, int]] = []
    missing_shape: list[str] = []
    for index, demand in enumerate(demands):
        if demand.family != "contraction":
            continue
        if not demand.has_shape or demand.batch is None:
            missing_shape.append(f"demand {index} ({demand.op})")
            continue
        identity = (demand.op, demand.batch, demand.m, demand.k, demand.n)
        actual[identity] += 1
        actual_sequence.append(identity)
    absent = expected - actual
    excess = actual - expected
    out_of_order = [
        {"ordinal": i, "structural": want, "demand": got}
        for i, (want, got) in enumerate(zip(expected_sequence, actual_sequence))
        if want != got
    ]
    if missing_shape or absent or excess or out_of_order:

        def _rows(counter: Counter) -> list[dict]:
            return [
                {"op": op, "batch": batch, "M": m, "K": k, "N": n, "count": count}
                for (op, batch, m, k, n), count in sorted(counter.items())[:8]
            ]

        raise ModelDemandIncomplete(
            f"contraction demands differ from parsed MLIR: structural={sum(expected.values())}, "
            f"demands={sum(actual.values())}, missing_shape={missing_shape[:8]}, "
            f"missing={_rows(absent)}, extra={_rows(excess)}, out_of_order={out_of_order[:8]}"
        )
    return demands
