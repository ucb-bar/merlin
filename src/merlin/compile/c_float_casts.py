"""Compile explicitly CPU-placed source float casts using integer bit carriers."""

from math import prod

from merlin.semantic_compiler.model import TensorType, pointwise_identity_maps

from .c_sources import c_source_pair

_CASTS = {
    "bf16_to_f32": ("bf16", "f32", 2, 4, "bf16-to-f32-exact-v1"),
    "f32_to_bf16": ("f32", "bf16", 4, 2, "f32-to-bf16-rne-v1"),
    "f32_to_f64": ("f32", "f64", 4, 8, "f32-to-f64-exact-v1"),
    "f64_to_f32": ("f64", "f32", 8, 4, "f64-to-f32-rne-v1"),
}
FLOAT_CAST_OPERATIONS = tuple(_CASTS)


def _convert_c(operation, name):
    if operation == "bf16_to_f32":
        return "return bits<<16;"
    if operation == "f32_to_bf16":
        return """uint64_t high=bits>>16, low=bits&65535u;
  return high+(low>32768u || (low==32768u && (high&1u)));"""
    if operation == "f32_to_f64":
        return """uint64_t sign=(bits&UINT64_C(0x80000000))<<32;
  unsigned exponent=(unsigned)((bits>>23)&255u);
  uint64_t significand=bits&UINT64_C(0x7fffff);
  if (exponent==255u) return sign|UINT64_C(0x7ff0000000000000);
  if (!exponent && !significand) return sign;
  int power=(int)exponent-127;
  if (exponent) significand|=UINT64_C(0x800000);
  else { power=-126; while (!(significand&UINT64_C(0x800000))) { significand<<=1; --power; } }
  return sign|((uint64_t)(power+1023)<<52)|((significand&UINT64_C(0x7fffff))<<29);"""
    return f"""uint64_t sign=(bits>>32)&UINT64_C(0x80000000);
  unsigned exponent=(unsigned)((bits>>52)&2047u);
  if (exponent==2047u) return sign|UINT64_C(0x7f800000);
  if (!exponent) return sign; /* Every binary64 subnormal is below binary32 half-minimum. */
  int power=(int)exponent-1023;
  if (power>127) return sign|UINT64_C(0x7f800000);
  uint64_t significand=(bits&UINT64_C(0xfffffffffffff))|UINT64_C(0x10000000000000);
  if (power < -126) {{
    unsigned shift=(unsigned)(-power-97);
    return shift>53u ? sign : sign|{name}_round_right(significand,shift);
  }}
  uint64_t rounded={name}_round_right(significand,29u);
  if (rounded==UINT64_C(0x1000000)) {{ rounded>>=1; ++power; }}
  if (power>127) return sign|UINT64_C(0x7f800000);
  return sign|((uint64_t)(power+127)<<23)|(rounded&UINT64_C(0x7fffff));"""


def emit_float_cast_c(source, name):
    """Preserve declared IEEE rounding, signed zeros, subnormals and infinities.

    Source NaN payload semantics remain unqualified. Reject any NaN before
    writing output. Dense byte buffers may be unaligned, but must be disjoint.
    No floating instructions or reference evaluator enter the runtime.
    """
    computations = [node for node in source.nodes if node.effect == "pure"]
    if len(computations) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("float cast C needs one computation and dense logical buffers")
    node = computations[0]
    if node.op not in _CASTS or len(node.inputs) != 1 or source.outputs != (node.id,):
        raise ValueError("float cast C needs one supported conversion and one result")
    operand = source.node(node.inputs[0])
    dtype, output_dtype, width, output_width, policy = _CASTS[node.op]
    if (
        {item.id for item in source.nodes} != {operand.id, node.id}
        or operand.effect not in {"input", "constant"} or operand.index_maps
        or not node.type.shape or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
        or operand.type != TensorType(node.type.shape, dtype, "raw-bit-pattern")
        or node.type != TensorType(node.type.shape, output_dtype, "raw-bit-pattern")
        or dict(node.attrs) != {"numeric_semantics": policy}
    ):
        raise ValueError("float cast C has different source types, indexing or rounding")
    count = prod(node.type.shape)
    input_bytes, output_bytes = count * width, count * output_width
    if max(input_bytes, output_bytes) > 2**63 - 1:
        raise ValueError("float cast C exceeds its signed index or byte-size domain")
    fraction_bits = {2: 7, 4: 23, 8: 52}[width]
    magnitude = (1 << (8 * width - 1)) - 1
    infinity = magnitude - ((1 << fraction_bits) - 1)
    if operand.effect == "constant":
        binding = next(item for item in source.constants if item.node_id == operand.id)
        if binding.encoding != f"{dtype}-le-raw":
            raise ValueError("float cast constant needs its exact source encoding")
        data = bytes.fromhex(binding.data_hex)
        if any(int.from_bytes(data[i:i + width], "little") & magnitude > infinity
               for i in range(0, len(data), width)):
            raise ValueError("float cast constant NaN payload is unqualified")
    rounding = f"""static uint64_t {name}_round_right(uint64_t value,unsigned shift) {{
  uint64_t high=value>>shift, low=value&((UINT64_C(1)<<shift)-1u);
  uint64_t half=UINT64_C(1)<<(shift-1u);
  return high+(low>half || (low==half && (high&1u)));
}}
""" if node.op == "f64_to_f32" else ""
    implementation = f"""#include <stddef.h>
#include <stdint.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && SIZE_MAX>={max(input_bytes, output_bytes)}ULL,"float tensor exceeds host types");
static uint64_t {name}_load(const unsigned char *p) {{
  uint64_t bits=0; for (unsigned i=0;i<{width}u;++i) bits|=(uint64_t)p[i]<<(8u*i); return bits;
}}
{rounding}
static uint64_t {name}_convert(uint64_t bits) {{ {_convert_c(node.op, name)} }}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !inputs[0] || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0], b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-{input_bytes}u || b>UINTPTR_MAX-{output_bytes}u ||
      (a<b+{output_bytes}u && b<a+{input_bytes}u)) return -1;
  for (size_t i=0;i<{count}u;++i) {{
    uint64_t bits={name}_load((const unsigned char *)inputs[0]+{width}u*i);
    if ((bits&UINT64_C({magnitude}))>UINT64_C({infinity})) return -2;
  }}
  for (size_t i=0;i<{count}u;++i) {{
    uint64_t bits={name}_convert({name}_load((const unsigned char *)inputs[0]+{width}u*i));
    for (unsigned j=0;j<{output_width}u;++j)
      ((unsigned char *)outputs[0])[{output_width}u*i+j]=(unsigned char)(bits>>(8u*j));
  }}
  return 0;
}}
"""
    return {
        "source_request_digest": source.digest(), "implementation": name,
        "inputs": list(node.inputs), "outputs": [node.id],
        "arithmetic": "exact-source-float-conversion", "numeric_semantics": policy,
        "domain": "all non-NaN source encodings; reject before output writes",
    }, c_source_pair(
        name, public_headers={}, private_headers={}, sources={"float_cast.c": implementation},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n",
    )
