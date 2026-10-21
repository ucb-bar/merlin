"""Compile the checked f32 boundary-count reduction as integer-only C."""

from math import prod

from merlin.semantic_compiler.model import IndexMap, TensorType
from merlin.semantic_compiler.predicate_values import ordered_float_compare

from .c_predicates import _float_comparison_c
from .c_sources import c_source_pair


def emit_bucketize_c(source, name):
    """Use binary search after checking sorted, non-NaN boundaries.

    The source counts unordered boundary<input or boundary<=input predicates.
    Sortedness makes binary search equivalent, including duplicate boundaries;
    a NaN query counts every boundary. Invalid buffers or boundaries fail before
    any output write. This is explicitly placed CPU work, not target fallback.
    """
    computations = [node for node in source.nodes if node.effect == "pure"]
    if len(computations) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("bucketize C needs one computation and dense logical buffers")
    node = computations[0]
    if node.op != "bucketize" or len(node.inputs) != 2 or source.outputs != (node.id,):
        raise ValueError("bucketize C needs one boundary-count result")
    values, boundaries = (source.node(key) for key in node.inputs)
    if (
        {item.id for item in source.nodes} != {*node.inputs, node.id}
        or any(item.effect not in {"input", "constant"} or item.index_maps for item in (values, boundaries))
        or not node.type.shape
        or len(boundaries.type.shape) != 1
    ):
        raise ValueError("bucketize C needs complete operand and result accounting")
    rank, count = len(node.type.shape), boundaries.type.shape[0]
    projection = IndexMap(
        rank + 1, tuple(tuple(int(i == j) for j in range(rank + 1)) for i in range(rank)), (0,) * rank
    )
    boundary_map = IndexMap(rank + 1, ((0,) * rank + (1,),), (0,))
    attrs = dict(node.attrs)
    if (
        values.type != TensorType(node.type.shape, "f32", "raw-bit-pattern")
        or boundaries.type != TensorType((count,), "f32", "raw-bit-pattern")
        or node.type != TensorType(node.type.shape, "i64", "signed-i64")
        or node.index_maps != (projection, boundary_map, projection)
        or set(attrs) != {"numeric_semantics", "right", "boundary_count"}
        or attrs["numeric_semantics"] != "f32-unordered-boundary-count-i64-v1"
        or type(attrs["right"]) is not bool
        or type(attrs["boundary_count"]) is not int
        or attrs["boundary_count"] != count
    ):
        raise ValueError("bucketize C has an unsupported type, map or numerical policy")
    elements = prod(node.type.shape)
    sizes, output_bytes = (elements * 4, count * 4), elements * 8
    maximum = max(*sizes, output_bytes)
    if maximum > 2**63 - 1:
        raise ValueError("bucketize C exceeds its signed index or byte-size domain")
    if boundaries.effect == "constant":
        binding = next(item for item in source.constants if item.node_id == boundaries.id)
        data = bytes.fromhex(binding.data_hex)
        limits = [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]
        if any(bits & 0x7FFFFFFF > 0x7F800000 for bits in limits) or any(
            not ordered_float_compare(a, b, 4, "ole") for a, b in zip(limits, limits[1:])
        ):
            raise ValueError("bucketize C needs nondecreasing non-NaN constant boundaries")
    checks = [f"b>UINTPTR_MAX-{output_bytes}u"]
    for i, size in enumerate(sizes):
        checks += [
            f"!inputs[{i}]",
            f"(uintptr_t)inputs[{i}]>UINTPTR_MAX-{size}u",
            f"((uintptr_t)inputs[{i}]<b+{output_bytes}u && b<(uintptr_t)inputs[{i}]+{size}u)",
        ]
    implementation = f"""#include <stddef.h>
#include <stdint.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && SIZE_MAX>={maximum}ULL,"boundary count exceeds host types");
static uint32_t {name}_load(const unsigned char *p) {{
  uint32_t value=0;
  for (unsigned i=0;i<4;++i) value|=(uint32_t)p[i]<<(8u*i);
  return value;
}}
{_float_comparison_c(name, 4)}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !outputs[0]) return -1;
  uintptr_t b=(uintptr_t)outputs[0];
  if ({" || ".join(checks)}) return -1;
  const unsigned char *values=(const unsigned char *)inputs[0];
  const unsigned char *boundaries=(const unsigned char *)inputs[1];
  uint32_t previous=0;
  for (size_t k=0;k<{count}u;++k) {{
    uint32_t current={name}_load(boundaries+4u*k);
    if ((current&UINT32_C(2147483647))>UINT32_C(2139095040) ||
        (k && !{name}_compare(previous,current,3u))) return -2;
    previous=current;
  }}
  for (size_t i=0;i<{elements}u;++i) {{
    uint32_t value={name}_load(values+4u*i);
    size_t low=0,high={count}u;
    if ((value&UINT32_C(2147483647))>UINT32_C(2139095040)) low=high;
    while (low<high) {{
      size_t middle=low+(high-low)/2u;
      uint32_t boundary={name}_load(boundaries+4u*middle);
      if ({name}_compare(boundary,value,{3 if attrs["right"] else 2}u)) low=middle+1u;
      else high=middle;
    }}
    for (unsigned byte=0;byte<8;++byte)
      ((unsigned char *)outputs[0])[8u*i+byte]=(unsigned char)((uint64_t)low>>(8u*byte));
  }}
  return 0;
}}
"""
    return {
        "source_request_digest": source.digest(),
        "implementation": name,
        "inputs": list(node.inputs),
        "outputs": [node.id],
        "arithmetic": "exact-source-boundary-count",
        "algorithm": "checked-binary-search",
        "right": attrs["right"],
        "boundary_count": count,
        "invalid_boundaries": "reject-before-output-writes",
    }, c_source_pair(
        name,
        public_headers={},
        private_headers={},
        sources={"bucketize.c": implementation},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n",
    )
