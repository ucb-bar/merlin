"""Checked BF16 GELU in the source's binary32 opmath order.

tanh form: x*x, *x, 0.044715*cube, x+that, sqrt(2/pi)*sum, tanhf, 1+t,
0.5*x, product. erf form: x*(1/sqrt(2)), erff, 1+e, 0.5*x, product. Each step
rounds once to binary32 nearest even; tanhf and erff come from the pinned
correctly rounded host-math supplier. One final BF16 RNE. Non-finite inputs
and unsupported FP environments refuse before any store.
"""

from math import prod
from struct import pack, unpack

from merlin.compiler.semantic.linalg.source_compositions import FINITE_BF16, GELU_SEMANTICS
from merlin.compiler.semantic.model import TensorType, pointwise_identity_maps

from ..dispatch import c_source_pair

_CONSTANTS = {"half": 0x3F000000, "one": 0x3F800000, "cubic": 0x3D372713, "root": 0x3F4C422A, "inverse": 0x3F3504F3}
ENVIRONMENT_CHECK = r"""#if defined(__riscv)
  unsigned rm; __asm__ volatile("frrm %0" : "=r"(rm));
  if (rm) return -3;
#elif defined(__x86_64__) || defined(__i386__)
  unsigned mxcsr; __asm__ volatile("stmxcsr %0" : "=m"(mxcsr));
  if (mxcsr & 0xe040U) return -3;
#else
#error host binary32 opmath needs an admitted floating-point environment check
#endif"""


def literal(bits):
    """Exact C hexadecimal binary32 literal for finite ``bits``."""
    return unpack("<f", pack("<I", bits))[0].hex() + "f"


def emit_bf16_gelu_c(source, name):
    from merlin.runtime import host_math_supplier
    from merlin.runtime.host_math_supplier.embedded import embedded_host_math

    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("BF16 GELU needs one dense logical computation")
    node = pure[0]
    attrs = dict(node.attrs)
    if node.op != "bf16_gelu" or len(node.inputs) != 1 or source.outputs != (node.id,):
        raise ValueError("BF16 GELU needs one declared source activation")
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
        or set(attrs) != {"approximate", "input_domain", "numeric_semantics"}
        or attrs["approximate"] not in {"tanh", "none"}
        or attrs["input_domain"] != FINITE_BF16
        or attrs["numeric_semantics"] != GELU_SEMANTICS
    ):
        raise ValueError("BF16 GELU type, operand, indexing, or semantics differ")
    count = prod(node.type.shape)
    if count * 2 > 2**63 - 1:
        raise ValueError("BF16 GELU exceeds signed byte-size domain")
    function = "tanhf" if attrs["approximate"] == "tanh" else "erff"
    header, policy = embedded_host_math(host_math_supplier, (function,))
    symbol = policy["functions"][function]["symbol"]
    c = {key: literal(bits) for key, bits in _CONSTANTS.items()}
    if function == "tanhf":
        body = f"""    volatile float square=x*x;
    volatile float cube=square*x;
    volatile float scaled={c["cubic"]}*cube;
    volatile float sum=x+scaled;
    volatile float inner={c["root"]}*sum;
    volatile float t={symbol}(inner);"""
    else:
        body = f"""    volatile float scaled=x*{c["inverse"]};
    volatile float t={symbol}(scaled);"""
    code = f"""#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#if defined(__FAST_MATH__)
#error BF16 GELU requires strict floating point
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
{ENVIRONMENT_CHECK}
  const unsigned char *input=(const unsigned char *)inputs[0];
  unsigned char *output=(unsigned char *)outputs[0];
  for (size_t i=0;i<{count}ULL;++i) {{
    unsigned bits=(unsigned)input[2*i]|((unsigned)input[2*i+1]<<8);
    if ((bits&32640u)==32640u) return -2;
  }}
  for (size_t i=0;i<{count}ULL;++i) {{
    uint32_t bits=((uint32_t)input[2*i]|((uint32_t)input[2*i+1]<<8))<<16;
    float x; memcpy(&x,&bits,4);
{body}
    volatile float onep={c["one"]}+t;
    volatile float half=x*{c["half"]};
    volatile float y=half*onep;
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
        numeric_semantics=GELU_SEMANTICS,
        approximate=attrs["approximate"],
        input_domain=FINITE_BF16,
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
        sources={"gelu.c": code},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n",
    )
