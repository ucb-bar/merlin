"""Translate the arithmetic source family, preserving numerical boundaries."""
from __future__ import annotations

from .linalg_recognition import LinalgBridgeError, _embedding_lookup_operands, _oriented_matmul_accumulate_maps, _oriented_matmul_maps, _source_f32_batched_matmul, _type
from .model import SemanticNode, TensorType, pointwise_identity_maps
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.linalg.attrs import IteratorType
from merlin.frontends.linalg_bf16_add import recognize_bf16_f32_chop_add, recognize_bf16_f32_chop_sub
from merlin.frontends.linalg_bf16_mul import recognize_bf16_product_rne
from merlin.frontends.linalg_bf16_relu import recognize_bf16_sign_clamp_relu
from merlin.frontends.linalg_fp8_cast import recognize_bf16_fp8_truncation
from merlin.frontends.linalg_ordered_fp8 import recognize_ordered_fp8_bf16_reset_matmul
from merlin.frontends.linalg_patterns import InvalidLinalgPattern
from .linalg_state import TranslationState

def translate(state: TranslationState, op, ordinal):
    nodes = state.nodes
    env = state.env
    zero_fills = state.zero_fills
    if op.name == "linalg.generic" and len(op.inputs) == 1 and (_type(op.inputs[0].type).dtype == "i64"):
        indices, table, destination = _embedding_lookup_operands(op)
        if (
            indices not in env
            or env[indices] is None
            or table not in env
            or env[table] is None
            or destination not in env
            or env[destination] is not None
        ):
            raise LinalgBridgeError("embedding lookup needs defined indices/table and an empty destination")
        table_type = _type(table.type)
        node_id = f"op{ordinal}r0"
        nodes.append(
            SemanticNode(
                node_id,
                "embedding_lookup",
                (env[indices], env[table]),
                _type(op.results[0].type),
                attrs=(
                    ("index_domain", "nonnegative-below-table-rows"),
                    ("minimum_index_bits", max(2, (table_type.shape[0] - 1).bit_length() + 1)),
                ),
            )
        )
        env[op.results[0]] = node_id
        return True

    if (
        op.name == "linalg.generic"
        and len(op.inputs) == 2
        and ("merlin.numeric_semantics" in op.attributes and "merlin.reduction_order" not in op.attributes)
    ):
        try:
            policy = op.attributes["merlin.numeric_semantics"]
            if not isinstance(policy, StringAttr):
                raise LinalgBridgeError("BF16 binary numerical policy must be a string")
            if policy.data == "selected-f32-rne-bf16-chop-v1":
                body = (
                    tuple(op.regions[0].block.ops)
                    if len(op.regions) == 1 and len(op.regions[0].blocks) == 1
                    else ()
                )
                if len(body) > 2 and body[2].name == "arith.subf":
                    semantic_op = "sub_bf16"
                    lhs, rhs, destination, semantics, domain = recognize_bf16_f32_chop_sub(op)
                else:
                    semantic_op = "add_bf16"
                    lhs, rhs, destination, semantics, domain = recognize_bf16_f32_chop_add(op)
            elif policy.data == "bf16-product-rne-v1":
                semantic_op = "mul_bf16"
                lhs, rhs, destination, semantics, domain = recognize_bf16_product_rne(op)
            else:
                raise LinalgBridgeError(f"unadmitted BF16 binary numerical policy {policy.data!r}")
        except InvalidLinalgPattern as exc:
            raise LinalgBridgeError(str(exc)) from exc
        if any(value not in env or env[value] is None for value in (lhs, rhs)) or (
            destination not in env or env[destination] is not None
        ):
            raise LinalgBridgeError("BF16 binary operation needs two defined sources and an empty destination")
        source_ids = (env[lhs], env[rhs])
        source_nodes = {node.id: node for node in nodes}
        if any(
            source_nodes[source_id].op == "input"
            and dict(source_nodes[source_id].attrs).get("input_domain") != domain
            for source_id in source_ids
        ):
            raise LinalgBridgeError("BF16 binary premise differs from its source boundary")
        result_type = _type(op.results[0].type)
        node_id = f"op{ordinal}r0"
        nodes.append(
            SemanticNode(
                node_id,
                semantic_op,
                source_ids,
                result_type,
                attrs=(("input_domain", domain), ("numeric_semantics", semantics)),
                index_maps=pointwise_identity_maps(2, 2),
            )
        )
        env[op.results[0]] = node_id
        return True

    if op.name == "linalg.generic" and len(op.inputs) == 2 and ("merlin.reduction_order" in op.attributes):
        try:
            lhs, rhs, init, semantics, domain = recognize_ordered_fp8_bf16_reset_matmul(op)
        except InvalidLinalgPattern as exc:
            raise LinalgBridgeError(str(exc)) from exc
        if lhs not in env or rhs not in env:
            raise LinalgBridgeError("ordered FP8 matmul needs declared inputs")
        source_ids = (env[lhs], env[rhs])
        if any(source_id is None for source_id in source_ids):
            raise LinalgBridgeError("ordered FP8 matmul reads uninitialized operands")
        source_nodes = {node.id: node for node in nodes}
        if any(
            (
                source_nodes[source_id].op == "input"
                and dict(source_nodes[source_id].attrs).get("input_domain") != domain
            )
            or (
                source_nodes[source_id].op != "input"
                and not (
                    source_nodes[source_id].op == "pack_fp8"
                    and source_nodes[source_id].type.dtype == "fp8_e4m3"
                    and source_nodes[source_id].type.numerical_policy != "raw-bit-pattern"
                )
            )
            for source_id in source_ids
        ):
            raise LinalgBridgeError("ordered FP8 matmul needs declared boundary inputs or a computed FP8 pack")
        # Preserve a computed pack's policy and def-use edge. The selected
        # target must establish that its result satisfies this matmul's
        # declared input domain; parsing alone cannot prove that closure.
        node_id = f"op{ordinal}r0"
        result_type = _type(op.results[0].type)
        accumulator_id = None if init in zero_fills else env.get(init)
        if init not in zero_fills:
            producer = source_nodes.get(accumulator_id)
            if (
                producer is None
                or producer.op not in {"matmul", "matmul_accumulate"}
                or producer.type != TensorType(result_type.shape, result_type.dtype, semantics)
            ):
                raise LinalgBridgeError("ordered FP8 continuation needs a matching prior ordered accumulator")
        nodes.append(
            SemanticNode(
                node_id,
                "matmul" if accumulator_id is None else "matmul_accumulate",
                source_ids if accumulator_id is None else (*source_ids, accumulator_id),
                TensorType(result_type.shape, result_type.dtype, semantics),
                index_maps=(
                    _oriented_matmul_maps() if accumulator_id is None else _oriented_matmul_accumulate_maps()
                ),
            )
        )
        env[op.results[0]] = node_id
        return True

    if op.name == "linalg.generic" and len(op.inputs) == 1 and (_type(op.results[0].type).dtype == "fp8_e4m3"):
        try:
            source_value, destination, semantics, domain, scale_code = recognize_bf16_fp8_truncation(op)
        except InvalidLinalgPattern as exc:
            raise LinalgBridgeError(str(exc)) from exc
        if (
            source_value not in env
            or env[source_value] is None
            or destination not in env
            or env[destination] is not None
        ):
            raise LinalgBridgeError("BF16-to-FP8 truncation needs a defined source and empty destination")
        source_id = env[source_value]
        source_node = next(node for node in nodes if node.id == source_id)
        if source_node.op == "input" and dict(source_node.attrs).get("input_domain") != domain:
            raise LinalgBridgeError("BF16-to-FP8 input domain differs from its source boundary")
        # A computed producer may discharge the declared input-domain
        # premise through a target rule. Keep its def-use edge intact;
        # the frontend does not infer numerical closure from an op name.
        result_type = _type(op.results[0].type)
        node_id = f"op{ordinal}r0"
        nodes.append(
            SemanticNode(
                node_id,
                "pack_fp8",
                (source_id,),
                TensorType(result_type.shape, result_type.dtype, semantics),
                attrs=(("input_domain", domain), ("scale_code", scale_code)),
                index_maps=pointwise_identity_maps(2, 1),
            )
        )
        env[op.results[0]] = node_id
        return True

    if op.name == "linalg.generic" and len(op.inputs) == 1 and (_type(op.results[0].type).dtype == "bf16"):
        try:
            source_value, destination = recognize_bf16_sign_clamp_relu(op)
        except InvalidLinalgPattern as exc:
            raise LinalgBridgeError(str(exc)) from exc
        if source_value not in env or env[source_value] is None or destination not in env:
            raise LinalgBridgeError("BF16 ReLU reads an undefined or uninitialized tensor")
        source_type = _type(source_value.type)
        destination_type = _type(destination.type)
        result_type = _type(op.results[0].type)
        if (
            source_type.dtype != "bf16"
            or destination_type != result_type
            or result_type.dtype != "bf16"
            or source_type.shape != result_type.shape
        ):
            raise LinalgBridgeError("BF16 ReLU result or destination differs from its source shape and dtype")
        node_id = f"op{ordinal}r0"
        nodes.append(
            SemanticNode(
                node_id,
                "relu",
                (env[source_value],),
                result_type,
                attrs=(("input_domain", "finite-normal-or-zero-bf16"), ("numeric_semantics", "bf16-sign-clamp-v1")),
                index_maps=pointwise_identity_maps(2, 1),
            )
        )
        env[op.results[0]] = node_id
        return True

    if (
        op.name == "linalg.generic"
        and len(op.inputs) == 2
        and len(op.results) == 1
        and _type(op.results[0].type).dtype == "f32"
        and any(iterator.data == IteratorType.REDUCTION for iterator in op.iterator_types.data)
    ):
        lhs, rhs, init, maps, add_order = _source_f32_batched_matmul(op)
        if any(value not in env or env[value] is None for value in (lhs, rhs, init)):
            raise LinalgBridgeError("source f32 batched matmul reads an undefined initializer or operand")
        node_id = f"op{ordinal}r0"
        nodes.append(
            SemanticNode(
                node_id,
                "source_f32_matmul_unqualified",
                tuple(env[value] for value in (lhs, rhs, init)),
                _type(op.results[0].type),
                attrs=(
                    ("reduction_order", "unresolved"),
                    ("numeric_semantics", "source-f32-mul-add-unqualified-v1"),
                    ("scalar_add_order", add_order),
                ),
                index_maps=maps,
            )
        )
        env[op.results[0]] = node_id
        return True

    return False
