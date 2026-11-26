"""Checked host BF16 contraction over one affine loop domain, such as an unfolded convolution.

out[o(L)] = BF16_RNE(RN(sum + bias[b(L)])), where sum starts at +0 and adds
RN(f32(x[a(L)]) * f32(w[v(L)])) in ascending lexicographic order of the
reduction loops, one binary32 RNE rounding per product and per addition: the
host BF16 linear policy, with the gathered operand read through its affine map.
The source leaves the reduction order unresolved; this order is the declared
choice. Before any store, every operand must be finite and a binary64 bound
K*max|x|*max|w| + max|bias| must stay below 2^127, so no binary32 product,
partial sum or BF16 result can overflow.

lower/c/bf16_linear.py implements the same policy for the host_bf16_linear route.
"""

from math import prod

from merlin.compiler.semantic.linalg.source_compositions import CONTRACTION_SEMANTICS, FINITE_BF16
from merlin.compiler.semantic.model import TensorType

from ..dispatch import c_source_pair
from .gelu import ENVIRONMENT_CHECK

POLICY = "bf16-exact-product-f32-ltr-rne-bf16-rne-v1"


def _unit_rows(mapping):
    """Loop index of each result dimension when every row is a unit vector, else None."""
    axes = []
    for row, offset in zip(mapping.coefficients, mapping.offsets):
        if offset or sorted(row) != [0] * (len(row) - 1) + [1]:
            return None
        axes.append(row.index(1))
    return axes


def emit_bf16_affine_contraction_c(source, name):
    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("BF16 affine contraction needs one dense logical computation")
    node = pure[0]
    attrs = dict(node.attrs)
    try:
        loops = tuple(int(v) for v in str(attrs.get("loop_extents", "")).split(","))
        reduced = tuple(int(v) for v in str(attrs.get("reduction_loops", "")).split(","))
    except ValueError as error:
        raise ValueError("BF16 affine contraction loop attributes differ") from error
    biased = attrs.get("affine_operands") == "bias"
    if (
        node.op != "bf16_affine_contraction" or source.outputs != (node.id,)
        or set(attrs) != {"affine_operands", "input_domain", "loop_extents", "numeric_semantics",
                          "reduction_loops", "reduction_order"}
        or attrs["affine_operands"] not in {"", "bias"} or attrs["input_domain"] != FINITE_BF16
        or attrs["numeric_semantics"] != CONTRACTION_SEMANTICS or attrs["reduction_order"] != "unresolved"
        or len(node.inputs) != 2 + biased or len(set(node.inputs)) != len(node.inputs)
        or len(node.index_maps) != len(node.inputs) + 1
        or any(extent <= 0 for extent in loops) or not reduced or len(set(reduced)) != len(reduced)
        or any(not 0 <= axis < len(loops) for axis in reduced)
        or node.type != TensorType(node.type.shape, "bf16", "raw-bit-pattern")
    ):
        raise ValueError("BF16 affine contraction source type, operands, or policy differs")
    operands = [source.node(key) for key in node.inputs]
    if any(value.effect != "input" or value.attrs or value.index_maps or value.type.dtype != "bf16"
           or value.type.numerical_policy != "raw-bit-pattern" for value in operands) or source.constants:
        raise ValueError("BF16 affine contraction operands must be BF16 inputs")
    width = len(loops)
    if any(mapping.loop_rank != width for mapping in node.index_maps):
        raise ValueError("BF16 affine contraction maps differ from the loop domain")
    *operand_maps, output_map = node.index_maps
    kept = [axis for axis in range(width) if axis not in reduced]
    output_axes = _unit_rows(output_map)
    weight_axes = _unit_rows(operand_maps[1])
    bias_axes = _unit_rows(operand_maps[2]) if biased else []
    if (output_axes is None or sorted(output_axes) != kept
            or tuple(loops[a] for a in output_axes) != node.type.shape
            or weight_axes is None or len(set(weight_axes)) != len(weight_axes)
            or not set(reduced) <= set(weight_axes)
            or tuple(loops[a] for a in weight_axes) != operands[1].type.shape
            or bias_axes is None or not set(bias_axes) <= set(kept)
            or biased and tuple(loops[a] for a in bias_axes) != operands[2].type.shape):
        raise ValueError("BF16 affine contraction output, weight, or bias indexing differs")
    gathered = operand_maps[0]
    if len(gathered.coefficients) != len(operands[0].type.shape):
        raise ValueError("BF16 affine contraction gathered map rank differs")
    for row, offset, extent in zip(gathered.coefficients, gathered.offsets, operands[0].type.shape):
        low = offset + sum(c * (n - 1) for c, n in zip(row, loops) if c < 0)
        high = offset + sum(c * (n - 1) for c, n in zip(row, loops) if c > 0)
        if low < 0 or high >= extent:
            raise ValueError("BF16 affine contraction gathered map leaves its source bounds")
    count = prod(loops[a] for a in reduced)
    sizes = [2 * prod(value.type.shape) for value in operands]
    output_size = 2 * prod(node.type.shape)
    if count >= 1 << 20 or max(*sizes, output_size) > 2**62:
        raise ValueError("BF16 affine contraction exceeds its checked overflow or size domain")

    def flat(rows, offsets, shape):
        expression = "0"
        for row, offset, extent in zip(rows, offsets, shape):
            terms = "".join(f"+({c})*(long long)l{i}" for i, c in enumerate(row) if c)
            expression = f"(({expression})*{extent}LL+({offset}LL{terms}))"
        return expression

    index = [flat(m.coefficients, m.offsets, value.type.shape) for m, value in zip(operand_maps, operands)]
    out_index = flat(output_map.coefficients, output_map.offsets, node.type.shape)
    outer = "".join(f"for (size_t l{a}=0;l{a}<{loops[a]}u;++l{a}) " for a in kept)
    inner = "".join(f"for (size_t l{a}=0;l{a}<{loops[a]}u;++l{a}) " for a in sorted(reduced))
    bias_add = f"volatile float biased=sum+{name}_value(b,{index[2]}); sum=biased;" if biased else ""
    scans = "".join(f"\n  if (!{name}_scan((const unsigned char *)inputs[{i}],{sizes[i] // 2}u,&maximum[{i}])) return -2;"
                    for i in range(len(operands)))
    overlap = "".join(
        f"\n  {{ uintptr_t t=(uintptr_t)inputs[{i}]; if (!t || t>UINTPTR_MAX-{sizes[i]}u || "
        f"(t<o+{output_size}u && o<t+{sizes[i]}u)) return -1; }}" for i in range(len(operands)))
    code = f"""#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#if defined(__FAST_MATH__)
#error BF16 affine contraction requires strict floating point
#endif
_Static_assert(CHAR_BIT==8 && sizeof(float)==4 && FLT_MANT_DIG==24 && FLT_RADIX==2 && FLT_MAX_EXP==128 &&
               SIZE_MAX>={max(*sizes, output_size)}ULL,"binary32 host representation required");
static float {name}_value(const unsigned char *p,long long i) {{
  uint32_t bits=((uint32_t)p[2*i]|((uint32_t)p[2*i+1]<<8))<<16; float v; memcpy(&v,&bits,4); return v;
}}
/* Refuse non-finite codes; return the largest magnitude as an exact binary64. */
static int {name}_scan(const unsigned char *p,size_t n,double *largest) {{
  *largest=0.0;
  for (size_t i=0;i<n;++i) {{
    unsigned bits=(unsigned)p[2*i]|((unsigned)p[2*i+1]<<8);
    if ((bits&32640u)==32640u) return 0;
    double v={name}_value(p,(long long)i); if (v<0) v=-v; if (v>*largest) *largest=v;
  }}
  return 1;
}}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !outputs[0]) return -1;
  uintptr_t o=(uintptr_t)outputs[0];
  if (o>UINTPTR_MAX-{output_size}u) return -1;{overlap}
{ENVIRONMENT_CHECK}
  double maximum[3]={{0.0,0.0,0.0}};{scans}
  if ({count}.0*maximum[0]*maximum[1]+maximum[2]>=0x1p127) return -2;
  const unsigned char *x=(const unsigned char *)inputs[0],*w=(const unsigned char *)inputs[1];
  {"const unsigned char *b=(const unsigned char *)inputs[2];" if biased else ""}
  unsigned char *y=(unsigned char *)outputs[0];
  {outer}{{
    float sum=0.0f;
    {inner}{{
      volatile float product={name}_value(x,{index[0]})*{name}_value(w,{index[1]});
      volatile float total=sum+product; sum=total;
    }}
    {bias_add}
    uint32_t bits; memcpy(&bits,&sum,4);
    bits=(bits+0x7fffU+((bits>>16)&1U))>>16;
    long long at={out_index};
    y[2*at]=(unsigned char)bits; y[2*at+1]=(unsigned char)(bits>>8);
  }}
  return 0;
}}
"""
    record = dict(
        source_request_digest=source.digest(), implementation=name, inputs=list(node.inputs), outputs=[node.id],
        numeric_semantics=POLICY, source_semantics=CONTRACTION_SEMANTICS, placement="host",
        selected_reduction_order="ascending lexicographic over the reduction loops, from +0",
        domain="finite BF16 operands; K*max|x|*max|w|+max|bias| < 2^127 checked before stores",
        weight_codes_or_exponents_changed=False, workspace_bytes=0,
        environment="binary32 RNE and gradual underflow; no fast-math",
        qualification="requires-exact-receipt",
    )
    return record, c_source_pair(
        name, public_headers={}, private_headers={}, sources={"affine_contraction.c": code},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n")
