"""Emit exact signed-i64 arithmetic, scans, reductions and float conversions."""

from math import prod

from merlin.semantic_compiler.model import IndexMap, TensorType, pointwise_identity_maps

from .c_sources import c_source_pair

INTEGER_OPERATIONS = (
    "i64_add",
    "i64_sub",
    "i64_mul",
    "i64_reduce_add",
    "i64_cumsum",
    "signed_i64_to_f32",
    "signed_i64_to_f64",
)


def _body(node, operands):
    shape, attrs = node.type.shape, dict(node.attrs)
    integer = TensorType(shape, "i64", "signed-i64")
    count = prod(shape)
    if node.op in {"i64_reduce_add", "i64_cumsum"}:
        scan = node.op == "i64_cumsum"
        source = operands[0].type
        axis = attrs.get("axis")
        rank = len(source.shape)
        if type(axis) is not int or not 0 <= axis < rank:
            raise ValueError("integer reduction or scan needs a valid axis")
        expected = {
            "axis": axis,
            "minimum_index_bits": max(2, max(source.shape).bit_length() + 1),
            "numeric_semantics": "signed-i64-prefix-sum-wrap-v1" if scan else "i64-add-modulo-2pow64-v1",
        }
        if attrs != expected or node.type != integer:
            raise ValueError("integer reduction or scan has a different numerical policy")
        width = 1 if source.dtype == "i1" else 8
        if source != TensorType(source.shape, "i64", "signed-i64") and not (
            scan and source == TensorType(source.shape, "i1", "boolean")
        ):
            raise ValueError("integer reduction or scan has incompatible input types")
        outer, extent, inner = prod(source.shape[:axis]), source.shape[axis], prod(source.shape[axis + 1 :])
        if scan:
            if len(operands) != 1 or shape != source.shape or node.index_maps:
                raise ValueError("integer scan needs unchanged shape and implicit lane indexing")
            initial = "UINT64_C(0)"
            store = f"{_store('position', 'sum', 8)}"
            final = ""
        else:
            projection = IndexMap(
                rank,
                tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank) if i != axis),
                (0,) * (rank - 1),
            )
            if (
                len(operands) != 2
                or operands[1].type != integer
                or shape != source.shape[:axis] + source.shape[axis + 1 :]
                or node.index_maps != (pointwise_identity_maps(rank, 1)[0], projection, projection)
            ):
                raise ValueError("integer reduction needs its exact seed, shape and indexing maps")
            initial = f"load((const unsigned char *)inputs[1]+8u*(row*{inner}u+col),8u)"
            store = ""
            final = _store(f"row*{inner}u+col", "sum", 8)
        validate = (
            f"for (size_t i=0;i<{prod(source.shape)}u;++i) if (((const unsigned char *)inputs[0])[i]>1u) return -2;"
            if width == 1
            else ""
        )
        return (
            f"""{validate}
  for (size_t row=0;row<{outer}u;++row) for (size_t col=0;col<{inner}u;++col) {{
    uint64_t sum={initial};
    for (size_t k=0;k<{extent}u;++k) {{
      size_t position=(row*{extent}u+k)*{inner}u+col;
      sum+=load((const unsigned char *)inputs[0]+{width}u*position,{width}u);
      {store}
    }}
    {final}
  }}""",
            8,
            [width] + ([] if scan else [8]),
            "",
        )
    if not shape or node.index_maps != pointwise_identity_maps(len(shape), len(operands)):
        raise ValueError("integer pointwise lowering needs equal positive-rank identity maps")
    if any(item.type != integer for item in operands):
        raise ValueError("integer pointwise lowering needs signed-i64 operands of the result shape")
    helper = ""
    if node.op in {"i64_add", "i64_sub", "i64_mul"}:
        arithmetic = node.op[4:]
        if (
            len(operands) != 2
            or node.type != integer
            or attrs != {"numeric_semantics": f"signed-i64-{arithmetic}-wrap-v1"}
        ):
            raise ValueError("integer arithmetic has a different type or wrap policy")
        operator = {"add": "+", "sub": "-", "mul": "*"}[arithmetic]
        expression = (
            f"load((const unsigned char *)inputs[0]+8u*i,8u){operator}load((const unsigned char *)inputs[1]+8u*i,8u)"
        )
        width = 8
    elif node.op in {"signed_i64_to_f32", "signed_i64_to_f64"}:
        dtype = node.op[-3:]
        if (
            len(operands) != 1
            or node.type != TensorType(shape, dtype, "raw-bit-pattern")
            or attrs != {"numeric_semantics": f"signed-i64-to-{dtype}-rne-v1"}
        ):
            raise ValueError("integer conversion has a different type or rounding policy")
        width, fraction, bias = (4, 23, 127) if dtype == "f32" else (8, 52, 1023)
        helper = f"""static uint64_t convert(uint64_t bits) {{
  uint64_t sign=bits>>63, magnitude=sign ? UINT64_C(0)-bits : bits;
  if (!magnitude) return 0;
  unsigned exponent=0;
  for (uint64_t rest=magnitude;rest>1;rest>>=1) ++exponent;
  uint64_t significand;
  if (exponent<={fraction}u) significand=magnitude<<({fraction}u-exponent);
  else {{
    unsigned shift=exponent-{fraction}u;
    significand=magnitude>>shift;
    uint64_t remainder=magnitude-(significand<<shift), halfway=UINT64_C(1)<<(shift-1u);
    if (remainder>halfway || (remainder==halfway && (significand&1u))) ++significand;
    if (significand==(UINT64_C(1)<<{fraction + 1}u)) {{ significand>>=1; ++exponent; }}
  }}
  return (sign<<{width * 8 - 1}u)|((uint64_t)(exponent+{bias}u)<<{fraction}u)|
         (significand&((UINT64_C(1)<<{fraction}u)-1u));
}}
"""
        expression = "convert(load((const unsigned char *)inputs[0]+8u*i,8u))"
    else:
        raise ValueError("source operation has no compiled integer implementation")
    return f"for (size_t i=0;i<{count}u;++i) {{ {_store('i', expression, width)} }}", width, [8] * len(operands), helper


def _store(index, value, width):
    return f"store((unsigned char *)outputs[0]+{width}u*({index}),{value},{width}u);"


def emit_integer_c(source, name):
    """Compile one explicitly CPU-placed operation with disjoint dense buffers.

    Unsigned carriers implement modulo-2^64 arithmetic without signed overflow.
    Integer-only IEEE encoding gives RNE conversion independently of the host
    floating environment. The runtime never calls a semantic evaluator.
    """
    computations = [node for node in source.nodes if node.effect == "pure"]
    if len(computations) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("integer C needs one computation and dense logical buffers")
    node = computations[0]
    operands = tuple(source.node(key) for key in node.inputs)
    if (
        node.op not in INTEGER_OPERATIONS
        or source.outputs != (node.id,)
        or not operands
        or {item.id for item in source.nodes} != {*node.inputs, node.id}
        or any(item.effect not in {"input", "constant"} or item.index_maps for item in operands)
    ):
        raise ValueError("integer C needs complete supported operand and result accounting")
    body, width, widths, helper = _body(node, operands)
    sizes = [prod(item.type.shape) * w for item, w in zip(operands, widths, strict=True)]
    output_bytes = prod(node.type.shape) * width
    maximum = max(*sizes, output_bytes)
    if maximum > 2**63 - 1:
        raise ValueError("integer C exceeds its signed index or byte-size domain")
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
_Static_assert(CHAR_BIT==8 && SIZE_MAX>={maximum}ULL,"integer tensor exceeds host types");
static uint64_t load(const unsigned char *p,unsigned width) {{
  uint64_t value=0;
  for (unsigned i=0;i<width;++i) value|=(uint64_t)p[i]<<(8u*i);
  return value;
}}
static void store(unsigned char *p,uint64_t value,unsigned width) {{
  for (unsigned i=0;i<width;++i) p[i]=(unsigned char)(value>>(8u*i));
}}
{helper}
"""
    # Helpers remain distinct when application export combines several regions.
    for symbol in ("load", "store", "convert"):
        implementation = implementation.replace(symbol + "(", name + "_" + symbol + "(")
        body = body.replace(symbol + "(", name + "_" + symbol + "(")
    implementation += f"""int {name}(void *context,const void *const *inputs,void *const *outputs) {{
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
        "inputs": list(node.inputs),
        "outputs": [node.id],
        "arithmetic": "exact-source-integer",
    }, c_source_pair(
        name,
        public_headers={},
        private_headers={},
        sources={"integer.c": implementation},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n",
    )
