"""Render byte-only affine packing for a checked pointwise tile plan."""

from __future__ import annotations

from math import prod

from merlin.compiler.lower.c.pointwise import render_assemble, render_operand_pack
from merlin.compiler.transforms.pointwise import PointwiseTiling


def render_pointwise_c(plan: PointwiseTiling, prefix: str, *, output_element_bytes: int | None = None) -> str:
    """Pack source operands and assemble result tiles with independent byte widths."""
    if (
        not isinstance(plan, PointwiseTiling)
        or not prefix
        or not prefix.isascii()
        or not prefix.isidentifier()
        or not prefix[0].isalpha()
    ):
        raise ValueError("pointwise C packing needs a checked plan and C identifier")
    tm, tn = plan.tile_shape
    width = plan.element_bytes
    output_width = width if output_element_bytes is None else output_element_bytes
    if type(output_width) is not int or output_width <= 0:
        raise ValueError("pointwise output element width must be a positive integer")
    largest = max(plan.packed_bytes, prod(plan.tile_counts) * tm * tn * output_width,
                  *(prod(node.type.shape) * width for node in plan.source.nodes))
    if largest > 2**63 - 1:
        raise ValueError("pointwise buffers exceed the supported signed index domain")
    helpers = [
        "#include <stddef.h>\n#include <stdint.h>\n"
        f'_Static_assert(SIZE_MAX >= {largest}ULL, "pointwise sizes exceed host size_t");'
    ]
    helpers += [render_operand_pack(plan, prefix, operand) for operand in range(len(plan.source.nodes) - 1)]
    helpers.append(render_assemble(plan, prefix, output_width))
    return "\n".join(helpers)


def emit_host_pointwise_c(source, name):
    """Compose one explicit scalar host operation with the shared pointer ABI.

    Selection is by checked source semantics, never model identity. A placement
    provider must select this implementation explicitly. Unsupported operations
    fail here rather than silently entering a framework or device fallback.
    """
    from .float_arithmetic import FLOAT_ARITHMETIC_OPERATIONS, emit_float_arithmetic_c
    from .float_casts import FLOAT_CAST_OPERATIONS, emit_float_cast_c
    from .host_ops.order import ORDER
    from .integer import emit_integer_c
    from .layout import emit_layout_c

    computations = [node for node in source.nodes if node.effect == 'pure']
    if len(computations) != 1:
        raise ValueError('host pointwise needs exactly one source computation')
    operation = computations[0].op
    if operation == 'relu':
        from .host_ops.relu import emit_bf16_relu_c
        record, files = emit_bf16_relu_c(source, name)
        order = ORDER['relu']
    elif operation == 'bf16_batch_norm_inference':
        from .host_ops.batch_norm import emit_bf16_batch_norm_c
        record, files = emit_bf16_batch_norm_c(source, name)
        order = ORDER['batch_norm']
    elif operation == 'bf16_window_max':
        from .host_ops.window_max import emit_bf16_window_max_c
        record, files = emit_bf16_window_max_c(source, name)
        order = ORDER['window_max']
    elif operation == 'bf16_sigmoid':
        from .host_ops.sigmoid import emit_bf16_sigmoid_c
        record, files = emit_bf16_sigmoid_c(source, name)
        order = ORDER['sigmoid']
    elif operation == 'bf16_gelu':
        from .host_ops.gelu import emit_bf16_gelu_c
        record, files = emit_bf16_gelu_c(source, name)
        order = ORDER['gelu_' + dict(computations[0].attrs).get('approximate', '')]
    elif operation == 'bf16_layer_norm':
        from .host_ops.layer_norm import emit_bf16_layer_norm_c
        record, files = emit_bf16_layer_norm_c(source, name)
        order = ORDER['layer_norm']
    elif operation in {'f32_silu', 'source_f32_reduce_add_unqualified', 'source_f32_rsqrt_unqualified'}:
        from .host_ops.f32_math import emit_f32_row_sum_c, emit_f32_rsqrt_c, emit_f32_silu_c
        emit, key = {'f32_silu': (emit_f32_silu_c, 'f32_silu'),
                     'source_f32_reduce_add_unqualified': (emit_f32_row_sum_c, 'f32_row_sum'),
                     'source_f32_rsqrt_unqualified': (emit_f32_rsqrt_c, 'f32_rsqrt')}[operation]
        record, files = emit(source, name)
        order = ORDER[key]
    elif operation == 'source_f32_softmax_unqualified':
        from .softmax_mask import emit_source_softmax_c
        record, files = emit_source_softmax_c(source, name)
        order = ORDER['f32_softmax']
    elif operation == 'bf16_affine_contraction':
        from .host_ops.affine_contraction import emit_bf16_affine_contraction_c
        record, files = emit_bf16_affine_contraction_c(source, name)
        order = ORDER['affine_contraction']
    elif operation in FLOAT_ARITHMETIC_OPERATIONS:
        record, files = emit_float_arithmetic_c(source, name)
        order = ORDER.get(operation, ORDER['neg'] if operation.endswith('_neg') else None)
    elif operation in FLOAT_CAST_OPERATIONS:
        record, files = emit_float_cast_c(source, name)
        order = ORDER['cast']
    elif operation in {'i64_add', 'i64_sub', 'i64_mul', 'signed_i64_to_f32', 'signed_i64_to_f64'}:
        record, files = emit_integer_c(source, name)
        order = ORDER['i64_arithmetic' if operation.startswith('i64_') else 'cast']
    elif operation in {'copy', 'identity', 'reshape'}:
        record, files = emit_layout_c(source, name)
        order = ORDER['reshape' if operation == 'reshape' else 'copy']
    else:
        raise ValueError(f'host pointwise has no qualified scalar emitter for {operation}')
    return {**record, 'operation_order': order}, files
