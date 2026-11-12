"""Translate the scalar source family, preserving numerical boundaries."""

from __future__ import annotations

from math import prod

from xdsl.dialects.linalg.attrs import IteratorType

from merlin.compiler.frontend.linalg_composite_math import recognize_static_composite_math_body
from merlin.compiler.frontend.linalg_patterns import InvalidLinalgPattern
from merlin.compiler.frontend.linalg_scalar_support import (
    StaticIotaResourceLimit,
    recognize_f32_nonzero,
    recognize_static_iota,
)
from merlin.compiler.semantic.linalg_recognition import LinalgBridgeError, _static_index_copy_map, _type
from merlin.compiler.semantic.linalg_state import TranslationState
from merlin.compiler.semantic.model import ConstantBinding, SemanticNode, pointwise_identity_maps
from merlin.compiler.semantic.reference import TensorValue


def translate(state: TranslationState, op, ordinal):
    nodes = state.nodes
    env = state.env
    constants = state.constants
    constant_bindings = state.constant_bindings
    append_scalar_region = state.append_scalar_region
    if op.name == "linalg.generic" and len(op.results) == len(op.outputs) == 2:
        from merlin.compiler.frontend.linalg_extremum_patterns import recognize_static_i64_argmin

        if len(op.inputs) != 1:
            raise LinalgBridgeError("two-result source reduction needs one admitted input")
        source_type = _type(op.inputs[0].type)
        index_bits = max(2, max(source_type.shape).bit_length() + 1)
        try:
            pattern = recognize_static_i64_argmin(op, index_bits=index_bits)
        except InvalidLinalgPattern as error:
            raise LinalgBridgeError(str(error)) from error
        source_id = env.get(op.inputs[0])
        if source_id is None:
            raise LinalgBridgeError("minimum/first-index reduction reads an uninitialized input")
        # These are pure projections of the same mathematical reduction.
        # A physical multi-result instruction uses the existing descriptor
        # bundle mechanism; separate realizations must count their work.
        for result_index, (value, operation) in enumerate(zip(op.results, ("i64_min_value", "i64_first_min_index"))):
            node = SemanticNode(
                f"op{ordinal}r{result_index}",
                operation,
                (source_id,),
                _type(value.type),
                attrs=(
                    ("axis", pattern.axis),
                    ("minimum_index_bits", index_bits),
                    ("numeric_semantics", "signed-i64-min-first-index-v1"),
                ),
            )
            nodes.append(node)
            env[value] = node.id
        return True

    if op.name == "linalg.generic" and len(op.results) == 1 and len(op.outputs) == 1:
        from merlin.compiler.semantic.constant_indexing import translate_constant_index_read

        indexed = translate_constant_index_read(op, env, nodes, constants, node_id=f"op{ordinal}r0")
        if indexed is not None:
            nodes.append(indexed)
            env[op.results[0]] = indexed.id
            return True
        element_types = tuple(str(getattr(value.type, "element_type", None)) for value in (*op.inputs, *op.results))
        if (
            len(op.inputs) == 1
            and _type(op.inputs[0].type).dtype in {"bf16", "fp8_e4m3", "i8", "i32"}
            and len(op.regions) == 1
            and len(op.regions[0].blocks) == 1
            and tuple(item.name for item in op.regions[0].block.ops) == ("linalg.yield",)
        ):
            value, destination = op.inputs[0], op.outputs[0]
            if env.get(value) is None or env.get(destination, "undefined") is not None:
                raise LinalgBridgeError("indexed copy needs a defined source and an empty destination")
            maps = _static_index_copy_map(op, _type(value.type), _type(op.results[0].type))
            node = SemanticNode(f"op{ordinal}r0", "gather", (env[value],), _type(op.results[0].type), index_maps=maps)
            nodes.append(node)
            env[op.results[0]] = node.id
            return True
        if (
            len(op.inputs) == 1
            and element_types in {("i1", "i64"), ("i64", "i64")}
            and any(item.data == IteratorType.REDUCTION for item in op.iterator_types)
        ):
            from merlin.compiler.frontend.linalg_integer_reductions import recognize_static_integer_reduction

            source_type = _type(op.inputs[0].type)
            index_bits = max(2, max(source_type.shape).bit_length() + 1)
            try:
                pattern = recognize_static_integer_reduction(op, index_bits=index_bits)
            except InvalidLinalgPattern as error:
                raise LinalgBridgeError(str(error)) from error
            source_id = env.get(op.inputs[0])
            if source_id is None:
                raise LinalgBridgeError("integer prefix sum reads an uninitialized input")
            node = SemanticNode(
                f"op{ordinal}r0",
                "i64_cumsum",
                (source_id,),
                _type(op.results[0].type),
                attrs=(
                    ("axis", pattern.axis[0]),
                    ("minimum_index_bits", index_bits),
                    ("numeric_semantics", "signed-i64-prefix-sum-wrap-v1"),
                ),
            )
            nodes.append(node)
            env[op.results[0]] = node.id
            return True
        if element_types == ("f32", "f32", "i64") and any(
            item.data == IteratorType.REDUCTION for item in op.iterator_types
        ):
            from merlin.compiler.semantic.scalar_region import translate_bucketize

            node = translate_bucketize(op, tuple(env.get(value) for value in op.inputs), node_id=f"op{ordinal}r0")
            nodes.append(node)
            env[op.results[0]] = node.id
            return True
        if (
            ("f64" in element_types and not any(item.name == "tensor.extract" for item in op.walk()))
            or (
                element_types[-1] == "f32"
                and len(op.regions) == 1
                and len(op.regions[0].blocks) == 1
                and tuple(item.name for item in op.regions[0].block.ops)
                in {
                    ("math.sin", "linalg.yield"),
                    ("math.cos", "linalg.yield"),
                    ("math.powf", "linalg.yield"),
                }
            )
            or (len(op.inputs) == 3 and all(item.data == IteratorType.PARALLEL for item in op.iterator_types))
            or (
                op.inputs
                and all(dtype == "i64" for dtype in element_types)
                and all(item.data == IteratorType.PARALLEL for item in op.iterator_types)
            )
            or (
                len(op.inputs) == 1
                and (element_types[0], element_types[-1]) in {("bf16", "f32"), ("f32", "bf16"), ("f32", "i64")}
            )
            or (
                element_types == ("i64", "f32")
                and len(op.regions) == 1
                and len(op.regions[0].blocks) == 1
                and tuple(item.name for item in op.regions[0].block.ops) == ("arith.sitofp", "linalg.yield")
            )
            or (
                "bf16" in element_types
                and element_types[-1] == "f32"
                and all(item.data == IteratorType.PARALLEL for item in op.iterator_types)
            )
            or (
                element_types[-1] == "bf16"
                and len(op.inputs) == 2
                and all(dtype == "bf16" for dtype in element_types)
                and all(key.startswith("prov.") for key in op.attributes)
                and all(item.data == IteratorType.PARALLEL for item in op.iterator_types)
            )
            or (
                element_types
                and all(dtype == "i1" for dtype in element_types)
                and all(item.data == IteratorType.PARALLEL for item in op.iterator_types)
            )
        ):
            append_scalar_region(op, ordinal)
            return True

    if (
        op.name == "linalg.generic"
        and len(op.results) == 1
        and (
            (str(getattr(op.results[0].type, "element_type", None)) == "i1" and len(op.inputs) == 2)
            or (
                str(getattr(op.results[0].type, "element_type", None)) == "i64"
                and len(op.inputs) == 1
                and str(getattr(op.inputs[0].type, "element_type", None)) == "i1"
            )
        )
    ):
        from merlin.compiler.semantic.masks import recognize_mask

        try:
            _, maps, attrs = recognize_mask(op)
        except InvalidLinalgPattern as error:
            raise LinalgBridgeError(str(error)) from error
        source_ids = tuple(env.get(value) for value in op.inputs)
        if any(value is None for value in source_ids):
            raise LinalgBridgeError("mask reads an uninitialized input")
        node_id = f"op{ordinal}r0"
        nodes.append(SemanticNode(node_id, "mask", source_ids, _type(op.results[0].type), attrs=attrs, index_maps=maps))
        env[op.results[0]] = node_id
        return True

    if op.name == "linalg.generic" and len(op.results) == 1:
        # A signed reciprocal contains a cast as well as a division. Keep
        # both rounding boundaries instead of treating its i64 input as
        # an embedding index or assuming a fixed mask population.
        if len(op.inputs) == 1 and _type(op.inputs[0].type).dtype == "i64":
            try:
                composite = recognize_static_composite_math_body(op)
            except InvalidLinalgPattern:
                composite = None
            if composite is not None and composite.operation == "reciprocal_signed_i64_to_f32":
                source_id = env.get(op.inputs[0])
                if source_id is None or env.get(op.outputs[0], "undefined") is not None:
                    raise LinalgBridgeError("signed reciprocal needs a defined input and an empty output")
                type_ = _type(op.results[0].type)
                node_id = f"op{ordinal}r0"
                cast_id, one_id = f"{node_id}_cast", f"{node_id}_one"
                maps = pointwise_identity_maps(len(type_.shape), 1)
                nodes.extend(
                    (
                        SemanticNode(
                            cast_id,
                            "signed_i64_to_f32",
                            (source_id,),
                            type_,
                            attrs=(("numeric_semantics", "signed-i64-to-f32-rne-v1"),),
                            index_maps=maps,
                        ),
                        SemanticNode(one_id, "constant", (), type_, effect="constant"),
                        SemanticNode(
                            node_id,
                            "f32_div",
                            (one_id, cast_id),
                            type_,
                            attrs=(("numeric_semantics", "ieee754-f32-div-rne-v1"),),
                            index_maps=pointwise_identity_maps(len(type_.shape), 2),
                        ),
                    )
                )
                count = prod(type_.shape)
                constants[one_id] = TensorValue(type_, (0x3F800000,) * count)
                constant_bindings.append(ConstantBinding(one_id, "f32-le-raw", "0000803f" * count))
                env[op.results[0]] = node_id
                return True
        try:
            generated = recognize_static_iota(op) if not op.inputs else None
            truth_shape = (
                recognize_f32_nonzero(op)
                if len(op.inputs) == 1 and str(getattr(op.results[0].type, "element_type", None)) == "i1"
                else None
            )
        except InvalidLinalgPattern:
            generated, truth_shape = None, None
        except StaticIotaResourceLimit as error:
            from merlin.compiler.semantic.target_binding import NativeCompilationError

            raise NativeCompilationError("resource_limit", str(error)) from error
        if generated is not None:
            node_id = f"op{ordinal}r0"
            type_ = _type(op.results[0].type)
            nodes.append(SemanticNode(node_id, "constant", (), type_, effect="constant"))
            constants[node_id] = TensorValue(type_, generated)
            data = b"".join(value.to_bytes(8, "little", signed=True) for value in generated)
            constant_bindings.append(ConstantBinding(node_id, "i64-le", data.hex()))
            env[op.results[0]] = node_id
            return True
        if truth_shape is not None:
            source_id = env.get(op.inputs[0])
            if source_id is None:
                raise LinalgBridgeError("truth conversion reads an uninitialized tensor")
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(
                    node_id,
                    "f32_nonzero",
                    (source_id,),
                    _type(op.results[0].type),
                    attrs=(("numeric_semantics", "f32-une-zero-v1"),),
                    index_maps=pointwise_identity_maps(len(truth_shape), 1),
                )
            )
            env[op.results[0]] = node_id
            return True

    return False
