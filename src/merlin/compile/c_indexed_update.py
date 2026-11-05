"""Emit an exact byte-preserving, bounds-checked single-slice update."""

from math import prod

from .c_sources import c_source_pair


def emit_indexed_update_c(source, result, name):
    if len(result.inputs) != 3 or not name.isascii() or not name.isidentifier() or not name[0].isalpha():
        raise ValueError("indexed update needs three operands and a C identifier")
    update, destination, positions = (source.node(key) for key in result.inputs)
    axis = dict(result.attrs).get("axis")
    shape = result.type.shape
    if (
        type(axis) is not int
        or not 0 <= axis < len(shape)
        or dict(result.attrs) != {"axis": axis, "index_domain": "nonnegative-below-destination-axis"}
        or result.index_maps
        or result.type != destination.type
        or result.type.dtype != "f32"
        or result.type.numerical_policy != "raw-bit-pattern"
        or len(update.type.shape) != len(shape)
        or update.type.dtype != "f32"
        or update.type.numerical_policy != "raw-bit-pattern"
        or update.type.shape[axis] != 1
        or any(update.type.shape[d] != shape[d] for d in range(len(shape)) if d != axis)
        or positions.type.shape != (1,)
        or positions.type.dtype != "i64"
        or positions.type.numerical_policy != "signed-i64"
        or any(n.effect not in {"input", "constant"} or n.index_maps for n in (update, destination, positions))
        or source.outputs != (result.id,)
        or {n.id for n in source.nodes} != {*result.inputs, result.id}
    ):
        raise ValueError("indexed update source types, axis or guard differ")
    sizes = (prod(update.type.shape) * 4, prod(shape) * 4, 8)
    if max(sizes) > (1 << 63) - 1:
        raise ValueError("indexed update exceeds supported host indices")
    inner, outer = prod(shape[axis + 1 :]) * 4, prod(shape[:axis])
    checks = [f"!inputs[{i}] || (uintptr_t)inputs[{i}]>UINTPTR_MAX-{n}u" for i, n in enumerate(sizes)]
    checks += [f"out>UINTPTR_MAX-{sizes[1]}u"]
    checks += [
        f"((uintptr_t)inputs[{i}]<out+{sizes[1]}u && out<(uintptr_t)inputs[{i}]+{n}u)" for i, n in enumerate(sizes)
    ]
    implementation = f"""#include <stddef.h>
#include <stdint.h>
_Static_assert(SIZE_MAX>={max(sizes)}ULL && sizeof(int64_t)==8,"indexed update exceeds host types");
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !outputs[0]) return -1;
  uintptr_t out=(uintptr_t)outputs[0];
  if ({" || ".join(checks)}) return -1;
  int64_t position; __builtin_memcpy(&position,inputs[2],8u);
  if (position<0 || (uint64_t)position>={shape[axis]}ULL) return -2;
  __builtin_memcpy(outputs[0],inputs[1],{sizes[1]}u);
  for (size_t row=0;row<{outer}u;++row) {{
    size_t offset=(row*{shape[axis]}u+(size_t)position)*{inner}u;
    __builtin_memcpy((unsigned char *)outputs[0]+offset,(const unsigned char *)inputs[0]+row*{inner}u,{inner}u);
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
        "index_bounds": [0, shape[axis]],
        "failure_writes_output": False,
    }, c_source_pair(
        name,
        public_headers={},
        private_headers={},
        sources={"indexed_update.c": implementation},
        declarations=f"int {name}(void *,const void *const *,void *const *);\n",
    )
