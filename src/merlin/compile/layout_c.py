"""Compile dense source layout operations without floating-point arithmetic."""

from __future__ import annotations

from math import prod

from merlin.llvmlower.c_runtime import DT_BYTES
from merlin.runtime.fp8_formats import storage_bits
from merlin.semantic_compiler.model import IndexMap, KernelRequest, pointwise_identity_maps

from .c_sources import c_source_pair


def _element_bytes(dtype: str) -> int:
    if dtype in DT_BYTES:
        return DT_BYTES[dtype]
    # Numeric formats have declared exponent/significand widths. Unknown and
    # sub-byte formats cannot be transported through this dense byte ABI.
    try:
        bits = storage_bits(dtype)
    except KeyError as error:
        raise ValueError("layout element type has no declared storage width") from error
    if bits <= 0 or bits % 8:
        raise ValueError("layout element needs a declared whole-byte storage width")
    return bits // 8


def _flat_map(mapping: IndexMap, loops: tuple[int, ...], shape: tuple[int, ...]) -> str:
    """Prove affine read/write bounds before rendering signed C arithmetic."""
    if mapping.loop_rank != len(loops) or len(mapping.coefficients) != len(shape):
        raise ValueError("layout map rank differs from its iteration space or tensor")
    expression = "0"
    for coefficients, offset, extent in zip(mapping.coefficients, mapping.offsets, shape, strict=True):
        low = offset + sum(min(0, c * (n - 1)) for c, n in zip(coefficients, loops, strict=True))
        high = offset + sum(max(0, c * (n - 1)) for c, n in zip(coefficients, loops, strict=True))
        if low < 0 or high >= extent:
            raise ValueError("layout map leaves its declared tensor")
        if max(abs(offset), *(abs(c) for c in coefficients)) > 2**63 - 1 or (
            abs(offset) + sum(abs(c) * (n - 1) for c, n in zip(coefficients, loops, strict=True)) > 2**63 - 1
        ):
            raise ValueError("layout map exceeds the supported signed C index domain")
        terms = " + ".join(f"({c}LL)*(long long)d{i}" for i, c in enumerate(coefficients))
        expression = f"(({expression})*{extent}u + (size_t)({offset}LL + {terms}))"
    return expression


def _indexed_copy(loops, source_shape, source_map, output_shape, output_map, operand, width):
    read = _flat_map(source_map, loops, source_shape)
    write = _flat_map(output_map, loops, output_shape)
    counters = " ".join(f"for (size_t d{i}=0;d{i}<{n}u;++d{i})" for i, n in enumerate(loops))
    return f"""{counters} {{
    __builtin_memcpy((unsigned char *)outputs[0]+{width}u*({write}),
                     (const unsigned char *)inputs[{operand}]+{width}u*({read}),{width}u);
  }}"""


def _emit_indexed_layout_c(source, result, name):
    operands = tuple(source.node(key) for key in result.inputs)
    width = _element_bytes(result.type.dtype)
    if (
        source.outputs != (result.id,)
        or not operands
        or any(
            node.effect not in {"input", "constant"}
            or node.index_maps
            or (node.type.dtype, node.type.numerical_policy) != (result.type.dtype, result.type.numerical_policy)
            for node in operands
        )
        or {node.id for node in source.nodes} != {*result.inputs, result.id}
    ):
        raise ValueError("indexed layout needs complete dense source operands and unchanged element types")
    counts = tuple(prod(node.type.shape) for node in (*operands, result))
    if max(counts) * width > 2**63 - 1:
        raise ValueError("indexed layout exceeds the supported C size domain")
    shape = result.type.shape
    if result.op in {"gather", "transpose"}:
        identity = pointwise_identity_maps(len(shape), 1)[0]
        if len(operands) != 1 or result.attrs or len(result.index_maps) != 2 or result.index_maps[-1] != identity:
            raise ValueError("gather needs one affine source map and an identity output map")
        if result.op == "transpose":
            mapping = result.index_maps[0]
            if (
                len(mapping.coefficients) != len(shape)
                or mapping.offsets != (0,) * len(shape)
                or any(set(row) - {0, 1} or sum(row) != 1 for row in mapping.coefficients)
                or len({row.index(1) for row in mapping.coefficients}) != len(shape)
                or tuple(shape[row.index(1)] for row in mapping.coefficients) != operands[0].type.shape
            ):
                raise ValueError("transpose needs a complete dimension permutation")
        body = _indexed_copy(shape, operands[0].type.shape, result.index_maps[0], shape, identity, 0, width)
    elif result.op == "insert_slice":
        loops = operands[0].type.shape
        identity = pointwise_identity_maps(len(loops), 1)[0]
        if (
            len(operands) != 2
            or operands[1].type != result.type
            or result.attrs
            or len(shape) != len(loops)
            or len(result.index_maps) != 3
            or result.index_maps[0] != identity
            or result.index_maps[1] != result.index_maps[2]
            or len(result.index_maps[1].coefficients) != len(loops)
            or any(
                row[dim] <= 0 or any(c != 0 for axis, c in enumerate(row) if axis != dim)
                for dim, row in enumerate(result.index_maps[1].coefficients)
            )
        ):
            raise ValueError("insert_slice needs an initialized destination and positive diagonal placement")
        body = f"__builtin_memcpy(outputs[0],inputs[1],{counts[-1] * width}u);\n"
        body += _indexed_copy(loops, loops, identity, shape, result.index_maps[1], 0, width)
    elif result.op == "concat":
        axis = dict(result.attrs).get("axis")
        rank = len(shape)
        if (
            len(operands) < 2
            or type(axis) is not int
            or not 0 <= axis < rank
            or result.attrs != (("axis", axis),)
            or result.index_maps
            or any(
                len(node.type.shape) != rank
                or any(n != shape[dim] for dim, n in enumerate(node.type.shape) if dim != axis)
                for node in operands
            )
            or sum(node.type.shape[axis] for node in operands) != shape[axis]
        ):
            raise ValueError("concat axis or source dimensions differ from its output")
        parts, start = [], 0
        identity = pointwise_identity_maps(rank, 1)[0]
        for i, node in enumerate(operands):
            placement = IndexMap(rank, identity.coefficients, tuple(start if dim == axis else 0 for dim in range(rank)))
            parts.append(_indexed_copy(node.type.shape, node.type.shape, identity, shape, placement, i, width))
            start += node.type.shape[axis]
        body = "\n".join(parts)
    else:
        raise ValueError("unknown indexed layout operation")
    checks = [f"!inputs[{i}]" for i in range(len(operands))]
    checks += [f"b>UINTPTR_MAX-{counts[-1] * width}u"]
    for i, count in enumerate(counts[:-1]):
        checks += [
            f"(uintptr_t)inputs[{i}]>UINTPTR_MAX-{count * width}u",
            f"((uintptr_t)inputs[{i}]<b+{counts[-1] * width}u && b<(uintptr_t)inputs[{i}]+{count * width}u)",
        ]
    implementation = f"""#include <stddef.h>
#include <stdint.h>
_Static_assert(SIZE_MAX/{width}u >= {max(counts)}ULL, "layout tensor exceeds host size_t");
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !outputs[0]) return -1;
  uintptr_t b=(uintptr_t)outputs[0];
  if ({" || ".join(checks)}) return -1;
  {body}
  return 0;
}}
"""
    return {
        "source_request_digest": source.digest(),
        "implementation": name,
        "inputs": list(result.inputs),
        "outputs": [result.id],
        "arithmetic": "byte-preserving",
    }, c_source_pair(
        name,
        public_headers={},
        private_headers={},
        sources={"layout.c": implementation},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n",
    )


def _emit_embedding_c(source, result, name):
    if len(result.inputs) != 2:
        raise ValueError("embedding lookup needs indices and a table")
    indices, table = (source.node(key) for key in result.inputs)
    width = _element_bytes(table.type.dtype)
    if (
        source.outputs != (result.id,)
        or result.index_maps
        or indices.type.dtype != "i64"
        or indices.type.numerical_policy != "signed-i64"
        or len(table.type.shape) != 2
        or any(node.effect not in {"input", "constant"} or node.index_maps for node in (indices, table))
        or {node.id for node in source.nodes} != {*result.inputs, result.id}
    ):
        raise ValueError("embedding lookup needs signed-i64 indices and a dense table")
    rows, columns = table.type.shape
    if (
        (result.type.dtype, result.type.numerical_policy) != (table.type.dtype, table.type.numerical_policy)
        or result.type.shape != (*indices.type.shape, columns)
        or dict(result.attrs)
        != {"index_domain": "nonnegative-below-table-rows", "minimum_index_bits": max(2, (rows - 1).bit_length() + 1)}
    ):
        raise ValueError("embedding lookup shape or index conditions differ")
    count = prod(indices.type.shape)
    sizes = (count * 8, rows * columns * width, count * columns * width)
    if max(sizes) > 2**63 - 1:
        raise ValueError("embedding lookup exceeds the supported C size domain")
    checks = [f"!inputs[{i}]" for i in range(2)]
    checks += [f"(uintptr_t)inputs[{i}]>UINTPTR_MAX-{size}u" for i, size in enumerate(sizes[:2])]
    checks += [f"b>UINTPTR_MAX-{sizes[2]}u"]
    checks += [
        f"((uintptr_t)inputs[{i}]<b+{sizes[2]}u && b<(uintptr_t)inputs[{i}]+{size}u)"
        for i, size in enumerate(sizes[:2])
    ]
    implementation = f"""#include <stddef.h>
#include <stdint.h>
_Static_assert(SIZE_MAX >= {max(sizes)}ULL && sizeof(int64_t)==8, "embedding sizes exceed host types");
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !outputs[0]) return -1;
  uintptr_t b=(uintptr_t)outputs[0];
  if ({" || ".join(checks)}) return -1;
  /* Validate all indices before any output write. Read with memcpy so the
   * byte-buffer ABI does not require aligned or aliased int64_t pointers. */
  for (size_t i=0;i<{count}u;++i) {{
    int64_t index; __builtin_memcpy(&index,(const unsigned char *)inputs[0]+8u*i,8u);
    if (index<0 || (uint64_t)index>={rows}ULL) return -2;
  }}
  for (size_t i=0;i<{count}u;++i) {{
    int64_t index; __builtin_memcpy(&index,(const unsigned char *)inputs[0]+8u*i,8u);
    __builtin_memcpy((unsigned char *)outputs[0]+{columns * width}u*i,
                     (const unsigned char *)inputs[1]+{columns * width}u*(size_t)index,{columns * width}u);
  }}
  return 0;
}}
"""
    return {
        "source_request_digest": source.digest(),
        "implementation": name,
        "inputs": list(result.inputs),
        "outputs": [result.id],
        "arithmetic": "byte-preserving",
        "index_dtype": "i64",
        "index_bounds": [0, rows],
        "failure_writes_output": False,
    }, c_source_pair(
        name,
        public_headers={},
        private_headers={},
        sources={"embedding.c": implementation},
        declarations=f"int {name}(void *,const void *const *,void *const *);\n",
    )


def emit_layout_c(source: KernelRequest, name: str) -> tuple[dict, dict[str, str]]:
    if source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("layout C requires dense logical buffers, not target physical representations")
    computations = [node for node in source.nodes if node.effect == "pure"]
    if len(computations) != 1:
        raise ValueError("layout C needs exactly one source computation")
    if computations[0].op == "indexed_insert_slice":
        from .c_indexed_update import emit_indexed_update_c

        return emit_indexed_update_c(source, computations[0], name)
    if computations[0].op == "embedding_lookup":
        return _emit_embedding_c(source, computations[0], name)
    if computations[0].op in {"gather", "insert_slice", "concat"} or (
        computations[0].op == "transpose" and len(computations[0].type.shape) != 2
    ):
        return _emit_indexed_layout_c(source, computations[0], name)
    if len(computations[0].inputs) != 1:
        raise ValueError("layout C needs one single-input source operation")
    result = computations[0]
    operand = source.node(result.inputs[0])
    width = _element_bytes(result.type.dtype)
    if (
        source.outputs != (result.id,)
        or result.attrs
        or operand.type.dtype != result.type.dtype
        or operand.type.numerical_policy != result.type.numerical_policy
        or set(node.id for node in source.nodes) != {operand.id, result.id}
        or prod(operand.type.shape) != prod(result.type.shape)
    ):
        raise ValueError("layout C cannot change type, numerical policy or element count")
    count = prod(result.type.shape)
    if count * width > 2**63 - 1:
        raise ValueError("layout exceeds the supported C size domain")
    if result.op == "reshape" and not result.index_maps:
        body = f"__builtin_memcpy(outputs[0],inputs[0],{count * width}u);"
    elif result.op == "identity" and not result.index_maps and result.type == operand.type:
        body = f"__builtin_memcpy(outputs[0],inputs[0],{count * width}u);"
    elif (
        result.op == "copy"
        and result.type == operand.type
        and result.index_maps
        == pointwise_identity_maps(
            len(result.type.shape),
            1,
        )
    ):
        body = f"__builtin_memcpy(outputs[0],inputs[0],{count * width}u);"
    elif (
        result.op == "transpose"
        and len(operand.type.shape) == 2
        and result.type.shape == operand.type.shape[::-1]
        and result.index_maps == (IndexMap(2, ((0, 1), (1, 0)), (0, 0)), IndexMap(2, ((1, 0), (0, 1)), (0, 0)))
    ):
        rows, columns = result.type.shape
        body = f"""for (size_t row=0; row<{rows}u; ++row) for (size_t col=0; col<{columns}u; ++col)
    __builtin_memcpy((unsigned char *)outputs[0]+{width}u*(row*{columns}u+col),
                     (const unsigned char *)inputs[0]+{width}u*(col*{rows}u+row),{width}u);"""
    else:
        raise ValueError("layout C has no implementation for this operation or index map")
    declarations = f"int {name}(void *, const void *const *, void *const *);\n"
    implementation = f"""#include <stddef.h>
#include <stdint.h>
_Static_assert(SIZE_MAX/{width}u >= {count}ULL, "layout tensor exceeds host size_t");
int {name}(void *context, const void *const *inputs, void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !inputs[0] || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0], b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-{count * width}u || b>UINTPTR_MAX-{count * width}u ||
      (a<b+{count * width}u && b<a+{count * width}u)) return -1;
  {body}
  return 0;
}}
"""
    return {
        "source_request_digest": source.digest(),
        "implementation": name,
        "inputs": [operand.id],
        "outputs": [result.id],
        "arithmetic": "byte-preserving",
    }, c_source_pair(
        name, public_headers={}, private_headers={}, sources={"layout.c": implementation}, declarations=declarations
    )
