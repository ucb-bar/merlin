"""Checked BF16 sigmoid using the pinned binary32 exponential supplier.

Source order is expf(-x), binary32 addition, binary32 division, then BF16 RNE.
Supplier identity is recorded; independent framework qualification remains a
caller obligation. NaN and unsupported FP environments refuse before stores.
"""

from math import prod

from merlin.compiler.frontend.patterns.linalg_bf16_sigmoid import INPUT_DOMAIN, NUMERIC_SEMANTICS
from merlin.compiler.semantic.model import TensorType, pointwise_identity_maps

from ..dispatch import c_source_pair


def emit_bf16_sigmoid_c(source, name):
    from merlin.runtime import host_math_supplier
    from merlin.runtime.host_math_supplier.embedded import embedded_host_math

    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("BF16 sigmoid needs one dense logical computation")
    node = pure[0]
    if node.op != "bf16_sigmoid" or len(node.inputs) != 1 or source.outputs != (node.id,):
        raise ValueError("BF16 sigmoid needs one declared source activation")
    operand = source.node(node.inputs[0])
    if (
        not node.type.shape
        or node.type != TensorType(node.type.shape, "bf16", "raw-bit-pattern")
        or operand.type != node.type
        or operand.effect != "input"
        or operand.attrs
        or operand.index_maps
        or len(source.nodes) != 2
        or source.constants
        or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
        or dict(node.attrs) != {"numeric_semantics": NUMERIC_SEMANTICS, "input_domain": INPUT_DOMAIN}
    ):
        raise ValueError("BF16 sigmoid type, operand, indexing, or semantics differ")
    count = prod(node.type.shape)
    if count * 2 > 2**63 - 1:
        raise ValueError("BF16 sigmoid exceeds signed byte-size domain")
    header, policy = embedded_host_math(host_math_supplier, ("expf",))
    symbol = policy["functions"]["expf"]["symbol"]
    code = f"""#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#if defined(__FAST_MATH__)
#error BF16 sigmoid requires strict floating point
#endif
_Static_assert(CHAR_BIT==8 && sizeof(float)==4 && FLT_MANT_DIG==24 && FLT_RADIX==2 &&
               FLT_MAX_EXP==128 && SIZE_MAX>={count * 2}ULL,"binary32 host representation required");
{header}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !inputs[0] || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0],b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-{count * 2}ULL || b>UINTPTR_MAX-{count * 2}ULL ||
      (a<b+{count * 2}ULL && b<a+{count * 2}ULL)) return -1;
#if defined(__riscv)
  unsigned rm; __asm__ volatile("frrm %0" : "=r"(rm));
  if (rm) return -3;
#elif defined(__x86_64__) || defined(__i386__)
  unsigned mxcsr; __asm__ volatile("stmxcsr %0" : "=m"(mxcsr));
  if (mxcsr & 0xe040U) return -3;
#else
#error BF16 sigmoid needs an admitted floating-point environment check
#endif
  const unsigned char *input=(const unsigned char *)inputs[0];
  unsigned char *output=(unsigned char *)outputs[0];
  for (size_t i=0;i<{count}ULL;++i) {{
    unsigned bits=(unsigned)input[2*i]|((unsigned)input[2*i+1]<<8);
    if ((bits&32767u)>32640u) return -2;
  }}
  for (size_t i=0;i<{count}ULL;++i) {{
    uint32_t bits=((uint32_t)input[2*i]|((uint32_t)input[2*i+1]<<8))<<16;
    float x; memcpy(&x,&bits,4);
    volatile float e={symbol}(-x);
    volatile float d=1.0f+e;
    volatile float y=1.0f/d;
    float result=y; memcpy(&bits,&result,4);
    bits=(bits+0x7fffU+((bits>>16)&1U))>>16;
    output[2*i]=(unsigned char)bits;
    output[2*i+1]=(unsigned char)(bits>>8);
  }}
  return 0;
}}
"""
    record = dict(
        source_request_digest=source.digest(),
        implementation=name,
        inputs=[operand.id],
        outputs=[node.id],
        numeric_semantics=NUMERIC_SEMANTICS,
        input_domain=INPUT_DOMAIN,
        host_math_policy_sha256=host_math_supplier.sha256(host_math_supplier.POLICY_PATH),
        host_math_symbols=[symbol],
        host_math_supplier="embedded-pinned-sources",
        library=policy["library"],
        qualification="requires-exact-receipt",
    )
    return record, c_source_pair(
        name,
        public_headers={},
        private_headers={},
        sources={"sigmoid.c": code},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n",
    )
