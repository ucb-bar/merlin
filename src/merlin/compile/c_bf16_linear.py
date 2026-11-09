"""Explicit host BF16 dot arithmetic preserving every source weight bit.

The source policy accepts either a direct unresolved BF16 contraction or its
closed BF16-to-f32/contraction/f32-to-BF16 source chain. The latter keeps the
original f32 seed, source digest, and output identity; it does not manufacture a
BF16 contraction. Intermediates cannot escape a closed emitted region.

The existing ``bf16_f32acc`` host lane is a separate xDSL/package lowering path,
not a semantic KernelRequest emitter. Its widening rewrite also permits
contraction, so it cannot establish this policy's separate, ascending-K binary32
operations. Reuse this C emitter to keep the arithmetic contract explicit.

The default POLICY remains for historical saved-boundary arithmetic and its raw
word regression tests. Source-f32 selection must opt into SOURCE_POLICY; the
historical product rounding and tile chop cannot stand in for that selection.
"""
from math import prod

from merlin.semantic_compiler.model import IndexMap, TensorType, pointwise_identity_maps
from .c_sources import c_source_pair

# Historical explicit host arithmetic used by the saved-boundary head path.
POLICY = "bf16-product-rne-tree32-f32-rne-bf16-rne-tiles-ltr-chop-v1"
SOURCE_POLICY = "bf16-exact-product-f32-ltr-rne-bf16-rne-v1"


def emit_c(source, name, *, policy=POLICY):
    """Emit a dense positive-zero-seeded BF16 matmul under an explicit policy.

    SOURCE_POLICY widens BF16 operands exactly, multiplies in binary32, adds in
    ascending K order with one binary32 rounding per step, and rounds once to
    BF16. BF16 products are exact in binary32 when representable; binary32
    underflow follows RNE. POLICY retains the historical BF16 product/tree/tile
    arithmetic for existing callers. This is host placement, with no FP8 recoding.
    The caller must disable contraction and fast math.
    """
    if policy not in {POLICY, SOURCE_POLICY}:
        raise ValueError("unknown host BF16 matmul arithmetic policy")
    pure = [node for node in source.nodes if node.effect == "pure"]
    if source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("host BF16 matmul needs one dense logical computation")
    if len(source.outputs) != 1:
        raise ValueError("host BF16 matmul operation or ordered inputs differ")
    output = source.node(source.outputs[0])
    attrs_of = lambda node: {key: value for key, value in node.attrs if not key.startswith("prov.")}
    widened = policy == SOURCE_POLICY and output.op == "f32_to_bf16"
    if widened:
        if len(pure) != 4 or len(output.inputs) != 1:
            raise ValueError("host BF16 matmul needs a closed widened source chain")
        node = source.node(output.inputs[0])
        if node.op != "source_f32_matmul_unqualified" or len(node.inputs) != 3:
            raise ValueError("host BF16 matmul widened contraction differs")
        casts = tuple(source.node(key) for key in node.inputs[:2])
        for cast in casts:
            if (cast.op != "bf16_to_f32" or len(cast.inputs) != 1
                    or attrs_of(cast) != {"numeric_semantics": "bf16-to-f32-exact-v1"}
                    or cast.index_maps != pointwise_identity_maps(2, 1)):
                raise ValueError("host BF16 matmul widened operand conversion differs")
        lhs, rhs = (source.node(cast.inputs[0]) for cast in casts)
        seed = source.node(node.inputs[2])
        if (attrs_of(output) != {"numeric_semantics": "f32-to-bf16-rne-v1"}
                or output.index_maps != pointwise_identity_maps(2, 1)):
            raise ValueError("host BF16 matmul final conversion differs")
        allowed_ids = {lhs.id, rhs.id, seed.id, *(cast.id for cast in casts), node.id, output.id}
        arithmetic_dtype = "f32"
    else:
        if len(pure) != 1:
            raise ValueError("host BF16 matmul needs one dense logical computation")
        node = output
        if node.op != "source_bf16_matmul_unqualified" or len(node.inputs) != 3:
            raise ValueError("host BF16 matmul operation or ordered inputs differ")
        lhs, rhs, seed = (source.node(key) for key in node.inputs)
        allowed_ids = {lhs.id, rhs.id, seed.id, node.id}
        arithmetic_dtype = "bf16"
    if len(lhs.type.shape) != 2 or len(rhs.type.shape) != 2 or lhs.type.shape[1] != rhs.type.shape[0]:
        raise ValueError("host BF16 matmul requires [M,K] and [K,N]")
    m, k = lhs.type.shape
    n = rhs.type.shape[1]
    if (output.type != TensorType((m, n), "bf16", "raw-bit-pattern")
            or node.type != TensorType((m, n), arithmetic_dtype, "raw-bit-pattern")
            or seed.type != node.type or any(value.type.dtype != "bf16" or value.type.numerical_policy != "raw-bit-pattern"
                                           for value in (lhs, rhs))
            or any(value.effect not in {"input", "constant"} or value.inputs or value.index_maps for value in (lhs, rhs, seed))
            or any(key != "input_retention" or value not in {"preserve", "reusable"}
                   for operand in (lhs, rhs, seed) for key, value in operand.attrs if not key.startswith("prov."))
            or {item.id for item in source.nodes} != allowed_ids):
        raise ValueError("host BF16 matmul source types or effects differ")
    if widened and any(cast.type != TensorType(value.type.shape, "f32", "raw-bit-pattern")
                       for cast, value in zip(casts, (lhs, rhs))):
        raise ValueError("host BF16 matmul widened operand types differ")
    expected_maps = (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )
    arithmetic_attrs = attrs_of(node)
    if arithmetic_attrs.get("scalar_add_order") in {"accumulator-product", "product-accumulator"}:
        arithmetic_attrs.pop("scalar_add_order")
    if node.index_maps != expected_maps or arithmetic_attrs != {
            "numeric_semantics": f"source-{arithmetic_dtype}-mul-add-unqualified-v1", "reduction_order": "unresolved"}:
        raise ValueError("host BF16 matmul indexing or source arithmetic differs")
    seed_width = 4 if widened else 2
    sizes = (2 * m * k, 2 * k * n, seed_width * m * n)
    output_size = 2 * m * n
    if max(sizes) > 2**63 - 1:
        raise ValueError("host BF16 matmul exceeds the signed byte-size domain")
    for binding in source.constants:
        operand = source.node(binding.node_id)
        encoding = "f32-le-raw" if operand == seed and widened else "bf16-le-raw"
        if operand not in (lhs, rhs, seed) or binding.encoding != encoding:
            raise ValueError("host BF16 matmul constant encoding differs")
        raw = bytes.fromhex(binding.data_hex)
        width = seed_width if operand == seed else 2
        if len(raw) != prod(operand.type.shape) * width:
            raise ValueError("host BF16 matmul constant byte count differs")
        if operand == seed and any(raw):
            raise ValueError("host BF16 matmul seed must be positive zero")
    implementation = r"""#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <fenv.h>
#ifdef __FAST_MATH__
#error host_BF16_matmul_forbids_fast_math
#endif
_Static_assert(CHAR_BIT==8 && sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24 && FLT_MAX_EXP==128 && FLT_EVAL_METHOD==0,"host BF16 needs IEEE binary32");
_Static_assert(SIZE_MAX>=MAXSIZEULL,"host BF16 tensor exceeds size_t");
static uint16_t PREFIX_load(const unsigned char *p) { return (uint16_t)((unsigned)p[0]|((unsigned)p[1]<<8)); }
static uint32_t PREFIX_bits(float value) { uint32_t out;memcpy(&out,&value,4);return out; }
static float PREFIX_float(uint16_t bits) { uint32_t raw=(uint32_t)bits<<16;float value;memcpy(&value,&raw,4);return value; }
static int PREFIX_finite(uint32_t bits) { return (bits&0x7f800000u)!=0x7f800000u; }
static uint16_t PREFIX_rne(uint32_t bits) { return (uint16_t)((bits+0x7fffu+((bits>>16)&1u))>>16); }
int PREFIX(void *context,const void *const *inputs,void *const *outputs) {
  (void)context;
  if (!inputs || !outputs || !outputs[0] || fegetround()!=FE_TONEAREST) return -1;
  volatile float tiny=FLT_MIN,half=0.5f,twice=2.0f;
  volatile float sub=tiny*half,restored=sub*twice;
  if (sub==0.0f || restored!=FLT_MIN) return -1;
  const size_t sizes[3]={ASIZEu,BSIZEu,SEEDSIZEu};
  uintptr_t destination=(uintptr_t)outputs[0];
  if (destination>UINTPTR_MAX-CSIZEu) return -1;
  for (unsigned p=0;p<3;++p) {
    uintptr_t address=(uintptr_t)inputs[p];
    if (!address || address>UINTPTR_MAX-sizes[p] || (address<destination+CSIZEu && destination<address+sizes[p])) return -1;
  }
  for (unsigned p=0;p<2;++p) for (size_t i=0;i<sizes[p];i+=2) {
    uint16_t code=PREFIX_load((const unsigned char *)inputs[p]+i);
    if ((code&0x7fffu)>=0x7f80u) return -2;
  }
  for (size_t i=0;i<SEEDSIZEu;++i) if (((const unsigned char *)inputs[2])[i]) return -2;
  unsigned char *temporary=(unsigned char *)malloc(CSIZEu);
  if (!temporary) return -3;
  const unsigned char *a=(const unsigned char *)inputs[0],*b=(const unsigned char *)inputs[1];
  for (size_t row=0;row<MDIMu;++row) for (size_t column=0;column<NDIMu;++column) {
    uint16_t total=0;
    for (size_t begin=0;begin<KDIMu;begin+=32u) {
      float tree[32];
      for (size_t lane=0;lane<32u;++lane) {
        uint16_t left=0,right=0;
        if (begin+lane<KDIMu) { left=PREFIX_load(a+2u*(row*KDIMu+begin+lane));right=PREFIX_load(b+2u*((begin+lane)*NDIMu+column)); }
        volatile float product=PREFIX_float(left)*PREFIX_float(right);
        uint32_t raw=PREFIX_bits(product);
        if (!PREFIX_finite(raw)) { free(temporary);return -2; }
        uint16_t rounded=PREFIX_rne(raw);
        if ((rounded&0x7fffu)>=0x7f80u) { free(temporary);return -2; }
        tree[lane]=PREFIX_float(rounded);
      }
      for (size_t width=32u;width>1u;width/=2u) for (size_t i=0;i<width/2u;++i) {
        volatile float sum=tree[2u*i]+tree[2u*i+1u];
        if (!PREFIX_finite(PREFIX_bits(sum))) { free(temporary);return -2; }
        tree[i]=sum;
      }
      uint16_t partial=PREFIX_rne(PREFIX_bits(tree[0]));
      if ((partial&0x7fffu)>=0x7f80u) { free(temporary);return -2; }
      volatile float sum=PREFIX_float(total)+PREFIX_float(partial);
      uint32_t raw=PREFIX_bits(sum);
      if (!PREFIX_finite(raw)) { free(temporary);return -2; }
      total=(uint16_t)(raw>>16);
    }
    size_t at=2u*(row*NDIMu+column);temporary[at]=(unsigned char)total;temporary[at+1u]=(unsigned char)(total>>8);
  }
  memcpy(outputs[0],temporary,CSIZEu);free(temporary);return 0;
}
"""
    if policy == SOURCE_POLICY:
        begin = implementation.index("    uint16_t total=0;")
        end = implementation.index("    size_t at=", begin)
        implementation = implementation[:begin] + r"""    float accumulator=0.0f;
    for (size_t reduction=0;reduction<KDIMu;++reduction) {
      uint16_t left=PREFIX_load(a+2u*(row*KDIMu+reduction));
      uint16_t right=PREFIX_load(b+2u*(reduction*NDIMu+column));
      volatile float product=PREFIX_float(left)*PREFIX_float(right);
      if (!PREFIX_finite(PREFIX_bits(product))) { free(temporary);return -2; }
      volatile float sum=accumulator+product;
      if (!PREFIX_finite(PREFIX_bits(sum))) { free(temporary);return -2; }
      accumulator=sum;
    }
    uint16_t total=PREFIX_rne(PREFIX_bits(accumulator));
    if ((total&0x7fffu)>=0x7f80u) { free(temporary);return -2; }
""" + implementation[end:]
    replacements = {"PREFIX": name, "MAXSIZE": str(max(sizes)), "ASIZE": str(sizes[0]), "BSIZE": str(sizes[1]),
                    "SEEDSIZE": str(sizes[2]), "CSIZE": str(output_size), "MDIM": str(m), "KDIM": str(k), "NDIM": str(n)}
    for token, value in replacements.items():
        implementation = implementation.replace(token, value)
    return {"source_request_digest": source.digest(), "implementation": name,
            "inputs": [lhs.id, rhs.id, seed.id], "outputs": [output.id], "arithmetic": "selected-host-bf16-matmul",
            "numeric_semantics": policy, "placement": "host",
            "required_c_flags": ["-frounding-math", "-ffp-contract=off"], "required_link_flags": ["-lm"],
            "domain": "finite BF16 operands, intermediates and result; positive-zero seed; IEEE binary32 RNE with gradual underflow",
            "weight_codes_or_exponents_changed": False}, c_source_pair(
                name, public_headers={}, private_headers={}, sources={"bf16_linear.c": implementation},
                declarations=f"int {name}(void *,const void *const *,void *const *);\n")
