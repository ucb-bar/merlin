"""Explicit host BF16 dot arithmetic preserving every source weight bit."""
from math import prod

from merlin.semantic_compiler.model import IndexMap, TensorType
from .c_sources import c_source_pair

POLICY = "bf16-product-rne-tree32-f32-rne-bf16-rne-tiles-ltr-chop-v1"


def emit_c(source, name):
    """Emit a dense seeded BF16 matmul under the selected VPU arithmetic.

    This is a host placement. It does not pack FP8, cast stored parameters, or
    claim device execution. The caller must disable contraction and fast math.
    """
    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("host BF16 matmul needs one dense logical computation")
    node = pure[0]
    if node.op != "source_bf16_matmul_unqualified" or len(node.inputs) != 3 or source.outputs != (node.id,):
        raise ValueError("host BF16 matmul operation or ordered inputs differ")
    lhs, rhs, seed = (source.node(key) for key in node.inputs)
    if len(lhs.type.shape) != 2 or len(rhs.type.shape) != 2 or lhs.type.shape[1] != rhs.type.shape[0]:
        raise ValueError("host BF16 matmul requires [M,K] and [K,N]")
    m, k = lhs.type.shape
    n = rhs.type.shape[1]
    if (node.type != TensorType((m, n), "bf16", "raw-bit-pattern")
            or seed.type != node.type or any(value.type.dtype != "bf16" or value.type.numerical_policy != "raw-bit-pattern"
                                           for value in (lhs, rhs))
            or any(value.effect not in {"input", "constant"} or value.inputs or value.index_maps for value in (lhs, rhs, seed))
            or any(key != "input_retention" or value not in {"preserve", "reusable"}
                   for operand in (lhs, rhs, seed) for key, value in operand.attrs if not key.startswith("prov."))
            or {item.id for item in source.nodes} != {lhs.id, rhs.id, seed.id, node.id}):
        raise ValueError("host BF16 matmul source types or effects differ")
    expected_maps = (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )
    attrs = {key: value for key, value in node.attrs if not key.startswith("prov.")}
    if node.index_maps != expected_maps or attrs != {"numeric_semantics": "source-bf16-mul-add-unqualified-v1", "reduction_order": "unresolved"}:
        raise ValueError("host BF16 matmul indexing or source arithmetic differs")
    sizes = (2 * m * k, 2 * k * n, 2 * m * n)
    if max(sizes) > 2**63 - 1:
        raise ValueError("host BF16 matmul exceeds the signed byte-size domain")
    for binding in source.constants:
        operand = source.node(binding.node_id)
        if operand not in (lhs, rhs, seed) or binding.encoding != "bf16-le-raw":
            raise ValueError("host BF16 matmul constant encoding differs")
        raw = bytes.fromhex(binding.data_hex)
        if len(raw) != prod(operand.type.shape) * 2:
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
  const size_t sizes[3]={ASIZEu,BSIZEu,CSIZEu};
  uintptr_t destination=(uintptr_t)outputs[0];
  if (destination>UINTPTR_MAX-CSIZEu) return -1;
  for (unsigned p=0;p<3;++p) {
    uintptr_t address=(uintptr_t)inputs[p];
    if (!address || address>UINTPTR_MAX-sizes[p] || (address<destination+CSIZEu && destination<address+sizes[p])) return -1;
  }
  for (unsigned p=0;p<3;++p) for (size_t i=0;i<sizes[p];i+=2) {
    uint16_t code=PREFIX_load((const unsigned char *)inputs[p]+i);
    if ((code&0x7fffu)>=0x7f80u || (p==2 && code)) return -2;
  }
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
    replacements = {"PREFIX": name, "MAXSIZE": str(max(sizes)), "ASIZE": str(sizes[0]), "BSIZE": str(sizes[1]),
                    "CSIZE": str(sizes[2]), "MDIM": str(m), "KDIM": str(k), "NDIM": str(n)}
    for token, value in replacements.items():
        implementation = implementation.replace(token, value)
    return {"source_request_digest": source.digest(), "implementation": name,
            "inputs": list(node.inputs), "outputs": [node.id], "arithmetic": "selected-host-bf16-matmul",
            "numeric_semantics": POLICY, "placement": "host",
            "required_c_flags": ["-frounding-math", "-ffp-contract=off"], "required_link_flags": ["-lm"],
            "domain": "finite BF16 operands and intermediates; positive-zero seed; IEEE binary32 RNE with gradual underflow",
            "weight_codes_or_exponents_changed": False}, c_source_pair(
                name, public_headers={}, private_headers={}, sources={"bf16_linear.c": implementation},
                declarations=f"int {name}(void *,const void *const *,void *const *);\n")
