"""Translate the contractions source family, preserving numerical boundaries."""

from __future__ import annotations

from merlin.compiler.frontend.linalg_composite_math import recognize_static_composite_math_body
from merlin.compiler.frontend.linalg_math_patterns import recognize_static_f32_exp_rsqrt_body
from merlin.compiler.frontend.linalg_native_pointwise import (
    recognize_f32_relu_ordered_select,
    recognize_static_f32_division,
)
from merlin.compiler.frontend.linalg_patterns import (
    InvalidLinalgPattern,
    recognize_signed_i8_i32_matmul,
    recognize_static_projected_pointwise,
)
from merlin.compiler.semantic.linalg_recognition import (
    LinalgBridgeError,
    _linear_affine_expression,
    _maps,
    _named_matmul_maps_are_default,
    _static_f32_square_maps,
    _static_index_copy_map,
    _type,
)
from merlin.compiler.semantic.linalg_state import TranslationState
from merlin.compiler.semantic.model import IndexMap, SemanticNode, pointwise_identity_maps


def translate(state: TranslationState, op, ordinal):
    nodes = state.nodes
    env = state.env
    append_scalar_region = state.append_scalar_region
    if op.name == "linalg.generic" and _type(op.results[0].type).dtype == "f32":
        if len(op.inputs) == 2 and len(op.outputs) == 1:
            try:
                pointwise = recognize_static_projected_pointwise(op)
            except InvalidLinalgPattern as exc:
                try:
                    pointwise = recognize_static_f32_division(op)
                except InvalidLinalgPattern as division_exc:
                    raise LinalgBridgeError(
                        f"f32 linalg.generic needs an admitted semantic decomposition: {exc}; {division_exc}"
                    ) from division_exc
            operation = {
                "arith.addf": "f32_add",
                "arith.subf": "f32_sub",
                "arith.mulf": "f32_mul",
                "arith.divf": "f32_div",
            }.get(pointwise.operation)
            if operation is None:
                raise LinalgBridgeError("f32 linalg.generic needs an admitted semantic decomposition")
            lhs, rhs = op.inputs
            destination = op.outputs[0]
            result_type = _type(op.results[0].type)
            if (
                lhs not in env
                or env[lhs] is None
                or rhs not in env
                or env[rhs] is None
                or destination not in env
                or env[destination] is not None
                or any(
                    _type(value.type).dtype != "f32" or _type(value.type).numerical_policy != "raw-bit-pattern"
                    for value in (lhs, rhs)
                )
                or _type(destination.type) != result_type
            ):
                raise LinalgBridgeError("f32 arithmetic needs defined inputs and an empty output")
            rank = len(result_type.shape)
            maps = tuple(
                IndexMap(
                    rank,
                    tuple(row for row, _ in rows),
                    tuple(offset for _, offset in rows),
                )
                for rows in (
                    tuple(_linear_affine_expression(expression, rank) for expression in mapping.data.results)
                    for mapping in op.get_indexing_maps()
                )
            )
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(
                    node_id,
                    operation,
                    (env[lhs], env[rhs]),
                    result_type,
                    attrs=(("numeric_semantics", f"ieee754-f32-{operation[4:]}-rne-v1"),),
                    index_maps=maps,
                )
            )
            env[op.results[0]] = node_id
        elif len(op.inputs) != 1 or len(op.outputs) != 1:
            raise LinalgBridgeError("f32 linalg.generic needs an admitted semantic decomposition")
        else:
            source_value, destination = op.inputs[0], op.outputs[0]
            if (
                source_value not in env
                or env[source_value] is None
                or destination not in env
                or env[destination] is not None
            ):
                raise LinalgBridgeError("indexed copy needs a defined source and an empty destination")
            source_type, result_type = _type(source_value.type), _type(op.results[0].type)
            if (
                source_type.dtype != "f32"
                or result_type != _type(destination.type)
                or source_type.numerical_policy != result_type.numerical_policy
            ):
                raise LinalgBridgeError("indexed copy changes type or numerical policy")
            try:
                maps = _static_index_copy_map(op, source_type, result_type)
                semantic_op = "gather"
                inputs = (env[source_value],)
                attrs = ()
            except LinalgBridgeError as copy_error:
                try:
                    maps = _static_f32_square_maps(op, source_type, result_type)
                except LinalgBridgeError as square_error:
                    try:
                        checked_source, checked_destination = recognize_f32_relu_ordered_select(op)
                    except InvalidLinalgPattern as relu_error:
                        try:
                            math_source = recognize_static_f32_exp_rsqrt_body(op)
                        except InvalidLinalgPattern as math_error:
                            try:
                                composite = recognize_static_composite_math_body(op)
                            except InvalidLinalgPattern as composite_error:
                                try:
                                    # Preserve the existing closed f32 forms.
                                    # This additional family is the ordered
                                    # literal clamp, not arbitrary scalar SSA.
                                    if tuple(item.name for item in op.regions[0].block.ops) != (
                                        "arith.constant",
                                        "arith.cmpf",
                                        "arith.select",
                                        "linalg.yield",
                                    ):
                                        raise LinalgBridgeError("not an ordered literal comparison/select")
                                    append_scalar_region(op, ordinal)
                                except LinalgBridgeError as scalar_error:
                                    raise LinalgBridgeError(
                                        "f32 linalg.generic needs an admitted semantic decomposition: "
                                        f"{copy_error}; {square_error}; {relu_error}; "
                                        f"{math_error}; {composite_error}; {scalar_error}"
                                    ) from scalar_error
                                return True
                            if composite.operation == "clamp_upper_literal_compare_select":
                                # The shared frontend also recognizes this closed
                                # form. Preserve its exact ordered comparison and
                                # selected bits instead of losing them in an
                                # unqualified composite numerical operation.
                                append_scalar_region(op, ordinal)
                                return True
                            semantic_op = "source_f32_composite_unqualified"
                            maps = pointwise_identity_maps(len(result_type.shape), 1)
                            inputs = (env[source_value],)
                            attrs = (
                                ("numeric_semantics", "source-f32-composite-unqualified-v1"),
                                ("operation", composite.operation),
                                ("literal_bits", ",".join(composite.literal_bits)),
                            )
                        else:
                            kind = math_source.operation.removeprefix("math.")
                            semantic_op = f"source_f32_{kind}_unqualified"
                            maps = pointwise_identity_maps(len(result_type.shape), 1)
                            inputs = (env[source_value],)
                            attrs = (("numeric_semantics", f"source-f32-{kind}-unqualified-v1"),)
                    else:
                        if checked_source is not source_value or checked_destination is not destination:
                            raise LinalgBridgeError("f32 zero select operands differ from the parsed source")
                        semantic_op = "f32_relu_ordered_select"
                        maps = pointwise_identity_maps(len(result_type.shape), 1)
                        inputs = (env[source_value],)
                        attrs = (("numeric_semantics", "f32-ordered-lt-positive-zero-select-v1"),)
                else:
                    semantic_op = "f32_mul"
                    inputs = (env[source_value], env[source_value])
                    attrs = (("numeric_semantics", "ieee754-f32-mul-rne-v1"),)
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(
                    node_id,
                    semantic_op,
                    inputs,
                    result_type,
                    attrs=attrs,
                    index_maps=maps,
                )
            )
            env[op.results[0]] = node_id
        return True

    if op.name in {"linalg.generic", "linalg.matmul"}:
        if op.name == "linalg.generic":
            try:
                lhs, rhs, init = recognize_signed_i8_i32_matmul(op)
            except InvalidLinalgPattern as exc:
                raise LinalgBridgeError(str(exc)) from exc
        else:
            if len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
                raise LinalgBridgeError("named matmul has an unsupported signature")
            if not _named_matmul_maps_are_default(op):
                raise LinalgBridgeError("named matmul has nonstandard indexing or iterators")
            lhs, rhs, init = (*op.inputs, op.outputs[0])
        if any(value not in env or env[value] is None for value in (lhs, rhs, init)):
            raise LinalgBridgeError("matmul reads an undefined or uninitialized operand")
        lhs_type, rhs_type, init_type, result_type = (_type(value.type) for value in (lhs, rhs, init, op.results[0]))
        if lhs_type.dtype != rhs_type.dtype or lhs_type.numerical_policy != rhs_type.numerical_policy:
            raise LinalgBridgeError("matmul input dtypes or numerical policies differ")
        if op.name == "linalg.matmul" and lhs_type.dtype not in {"i32", "f32"}:
            raise LinalgBridgeError("named matmul needs i32 operands or a source-only f32 graph node")
        if (
            len(lhs_type.shape) != 2
            or len(rhs_type.shape) != 2
            or (
                lhs_type.shape[1] != rhs_type.shape[0]
                or result_type.shape != (lhs_type.shape[0], rhs_type.shape[1])
                or init_type != result_type
                or result_type.dtype != ("f32" if lhs_type.dtype == "f32" else "i32")
            )
        ):
            raise LinalgBridgeError("matmul shape or initialized result type differs")
        node_id = f"op{ordinal}r0"
        if op.name == "linalg.matmul" and lhs_type.dtype == "f32":
            from merlin.compiler.quantization.contraction.intake import _check_matmul_body

            if set(op.properties) != {"indexing_maps", "operandSegmentSizes"} or any(
                not key.startswith("prov.") for key in op.attributes
            ):
                raise LinalgBridgeError("source f32 matmul has an unknown semantic attribute")
            _check_matmul_body(op)
            try:
                op.verify()
            except Exception as exc:  # noqa: BLE001 - xDSL owns source verification
                raise LinalgBridgeError(f"source f32 matmul failed verification: {exc}") from exc
            nodes.append(
                SemanticNode(
                    node_id,
                    "source_f32_matmul_unqualified",
                    tuple(env[value] for value in (lhs, rhs, init)),
                    result_type,
                    attrs=(
                        ("reduction_order", "unresolved"),
                        ("numeric_semantics", "source-f32-mul-add-unqualified-v1"),
                    ),
                    index_maps=_maps(),
                )
            )
            env[op.results[0]] = node_id
            return True
        nodes.append(
            SemanticNode(
                node_id,
                "matmul_accumulate",
                tuple(env[value] for value in (lhs, rhs, init)),
                result_type,
                index_maps=_maps(),
            )
        )
        env[op.results[0]] = node_id
        return True

    return False
