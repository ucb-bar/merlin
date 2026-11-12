"""Translate the returns source family, preserving numerical boundaries."""

from __future__ import annotations

from hashlib import sha256

from merlin.compiler.frontend.linalg_patterns import InvalidLinalgPattern
from merlin.compiler.semantic.linalg_recognition import (
    LinalgBridgeError,
    _f32_reduce_maps,
    _recognize_f32_reduce,
    _type,
)
from merlin.compiler.semantic.linalg_state import LinalgTranslation, TranslationState
from merlin.compiler.semantic.model import KernelRequest, SemanticNode


def translate(state: TranslationState, op, ordinal):
    nodes = state.nodes
    env = state.env
    constants = state.constants
    constant_bindings = state.constant_bindings
    boundaries = state.boundaries
    source_operations = state.source_operations
    operations = state.operations
    function = state.function
    boundary_storage = state.boundary_storage
    target_identity = state.target_identity
    lowering_policy = state.lowering_policy
    source = state.source
    if op.name == "func.return":
        if ordinal != len(operations) - 1 or not op.operands:
            raise LinalgBridgeError("func.return must be final and return tensors")
        if tuple(value.type for value in op.operands) != tuple(function.function_type.outputs.data):
            raise LinalgBridgeError("returned tensor types differ from the entry signature")
        outputs = tuple(env.get(value) for value in op.operands)
        if any(output is None for output in outputs):
            raise LinalgBridgeError("function returns an undefined or uninitialized tensor")
        request = KernelRequest(
            nodes=tuple(nodes),
            outputs=outputs,
            output_storages=(boundary_storage,) * len(outputs),
            input_storages=tuple(boundaries),
            target_identity=target_identity,
            lowering_policy=lowering_policy,
            source_identity=sha256(source.encode()).hexdigest(),
            constants=tuple(constant_bindings),
        )
        return LinalgTranslation(request, constants, tuple(source_operations))
        return True

    if op.name == "linalg.reduce" and op.results and _type(op.results[0].type).dtype == "i64":
        from merlin.compiler.frontend.linalg_integer_reductions import recognize_static_integer_reduction

        source_type = _type(op.input.type)
        # This is a residual index-width requirement, not an observation of
        # the target's scalar registers or the host pointer width.
        index_bits = max(2, max(source_type.shape).bit_length() + 1)
        try:
            pattern = recognize_static_integer_reduction(op, index_bits=index_bits)
        except InvalidLinalgPattern as error:
            raise LinalgBridgeError(str(error)) from error
        if len(pattern.axis) != 1:
            raise LinalgBridgeError("integer sum currently needs one axis")
        sources = tuple(env.get(value) for value in (op.input, op.init))
        if any(value is None for value in sources):
            raise LinalgBridgeError("integer sum reads an uninitialized input or seed")
        node_id = f"op{ordinal}r0"
        nodes.append(
            SemanticNode(
                node_id,
                "i64_reduce_add",
                sources,
                _type(op.results[0].type),
                attrs=(
                    ("axis", pattern.axis[0]),
                    ("minimum_index_bits", index_bits),
                    ("numeric_semantics", "i64-add-modulo-2pow64-v1"),
                ),
                index_maps=_f32_reduce_maps(len(source_type.shape), pattern.axis),
            )
        )
        env[op.results[0]] = node_id
        return True

    if op.name == "linalg.reduce" and op.results and _type(op.results[0].type).dtype == "f32":
        source_value, init, axes, order, add_order, scalar_op = _recognize_f32_reduce(op)
        if any(value not in env or env[value] is None for value in (source_value, init)):
            raise LinalgBridgeError("ordered f32 reduction needs a defined input and initializer")
        source_type = _type(source_value.type)
        node_id = f"op{ordinal}r0"
        maximum = scalar_op == "arith.maximumf"
        nodes.append(
            SemanticNode(
                node_id,
                (
                    "source_f32_reduce_max_unqualified"
                    if maximum
                    else "f32_reduce_add"
                    if order == "axis-ascending"
                    else "source_f32_reduce_add_unqualified"
                ),
                (env[source_value], env[init]),
                _type(op.results[0].type),
                attrs=(
                    ("axis", axes[0]) if len(axes) == 1 else ("axes", ",".join(map(str, axes))),
                    ("reduction_order", order),
                    (
                        "numeric_semantics",
                        "source-f32-maximum-unqualified-v1"
                        if maximum
                        else "ieee754-f32-add-rne-v1"
                        if order == "axis-ascending"
                        else "source-f32-add-unqualified-v1",
                    ),
                    *(
                        (("scalar_operand_order" if maximum else "scalar_add_order", add_order),)
                        if order == "unresolved"
                        else ()
                    ),
                ),
                index_maps=_f32_reduce_maps(len(source_type.shape), axes),
            )
        )
        env[op.results[0]] = node_id
        return True

    return False
