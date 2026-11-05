"""Compile explicitly CPU-placed IEEE arithmetic using integer bit carriers."""

from math import prod

from merlin.semantic_compiler.model import TensorType, pointwise_identity_maps

from .c_sources import c_source_pair

FLOAT_ARITHMETIC_OPERATIONS = ("f64_mul", "bf16_add", "bf16_sub")

# BF16 finite values are integer multiples of 2^-133. Five 64-bit limbs
# preserve the entire exact sum, including cancellation across all exponents.
# Round once after the signed addition; no host floating environment is used.
_BF16_HELPERS = r"""
static uint64_t load(const unsigned char *p) { return (uint64_t)p[0]|((uint64_t)p[1]<<8); }
static int valid(uint64_t a,uint64_t b) {
  b^=SUBTRACT_SIGN;
  uint64_t am=a&32767u,bm=b&32767u;
  return am<=32640u && bm<=32640u && !(am==32640u && bm==32640u && ((a^b)&32768u));
}
static void expand(uint64_t bits,uint64_t out[5]) {
  unsigned exponent=(unsigned)(bits>>7),shift=exponent?exponent-1u:0u;
  uint64_t significand=(bits&127u)|(exponent?128u:0u);
  unsigned limb=shift/64u,offset=shift%64u;
  for (unsigned i=0;i<5;++i) out[i]=0;
  out[limb]=significand<<offset;
  if (offset>56u) out[limb+1u]=significand>>(64u-offset);
}
static unsigned bit(const uint64_t value[5],unsigned index) {
  return (unsigned)((value[index/64u]>>(index%64u))&1u);
}
static uint64_t compute(uint64_t a,uint64_t b) {
  b^=SUBTRACT_SIGN;
  uint64_t am=a&32767u,bm=b&32767u;
  if (am==32640u) return a;
  if (bm==32640u) return b;
  if (!am && !bm) return (a&b)&32768u;
  if (am<bm) { uint64_t swap=a;a=b;b=swap;swap=am;am=bm;bm=swap; }
  uint64_t x[5],y[5],carry=0;
  expand(am,x);expand(bm,y);
  for (unsigned i=0;i<5;++i) {
    if (!((a^b)&32768u)) {
      uint64_t sum=x[i]+y[i],next=sum+carry;
      carry=(sum<x[i])|(next<sum);x[i]=next;
    } else {
      uint64_t difference=x[i]-y[i],next=difference-carry;
      carry=(x[i]<y[i])|(difference<carry);x[i]=next;
    }
  }
  unsigned top=320u;
  while (top && !bit(x,top-1u)) --top;
  if (!top) return 0; /* Exact cancellation under RNE gives positive zero. */
  --top;
  unsigned shift=top>7u?top-7u:0u;
  uint64_t rounded=0;
  for (unsigned i=0;i<8;++i) rounded|=(uint64_t)bit(x,shift+i)<<i;
  if (shift) {
    unsigned sticky=0;
    for (unsigned i=0;i+1u<shift;++i) sticky|=bit(x,i);
    rounded+=bit(x,shift-1u) && (sticky || (rounded&1u));
  }
  if (rounded==256u) { rounded>>=1;++shift; }
  uint64_t sign=a&32768u;
  if (rounded<128u) return sign|rounded;
  if (shift>=254u) return sign|32640u;
  return sign|((uint64_t)(shift+1u)<<7)|(rounded-128u);
}
"""

_HELPERS = r"""
static uint64_t load(const unsigned char *p) {
  uint64_t bits=0; for (unsigned i=0;i<8;++i) bits|=(uint64_t)p[i]<<(8u*i); return bits;
}
static int valid(uint64_t a,uint64_t b) {
  uint64_t am=a&UINT64_C(0x7fffffffffffffff), bm=b&UINT64_C(0x7fffffffffffffff);
  const uint64_t inf=UINT64_C(0x7ff0000000000000);
  return am<=inf && bm<=inf && !(am==inf && !bm) && !(bm==inf && !am);
}
static uint64_t compute(uint64_t a,uint64_t b) {
  uint64_t sign=(a^b)&UINT64_C(0x8000000000000000);
  uint64_t am=a&UINT64_C(0x7fffffffffffffff), bm=b&UINT64_C(0x7fffffffffffffff);
  const uint64_t inf=UINT64_C(0x7ff0000000000000), hidden=UINT64_C(1)<<52;
  if (am==inf || bm==inf) return sign|inf;
  if (!am || !bm) return sign;
  unsigned ae=(unsigned)(am>>52), be=(unsigned)(bm>>52);
  uint64_t as=(am&(hidden-1u))|(ae?hidden:0), bs=(bm&(hidden-1u))|(be?hidden:0);
  int power=(int)(ae?ae:1)+(int)(be?be:1)-2150;
  __uint128_t product=(__uint128_t)as*bs;
  unsigned top=0;
  for (__uint128_t rest=product;rest>1;rest>>=1) ++top;
  int highest=(int)top+power;
  if (highest>1023) return sign|inf;
  int quantum=highest-52>-1074 ? highest-52 : -1074;
  int shift=quantum-power;
  uint64_t rounded;
  if (shift<=0) rounded=(uint64_t)(product<<(unsigned)-shift);
  else if (shift>=128) rounded=0;
  else {
    __uint128_t high=product>>(unsigned)shift;
    __uint128_t low=product- (high<<(unsigned)shift), half=(__uint128_t)1<<(unsigned)(shift-1);
    rounded=(uint64_t)high+(low>half || (low==half && (high&1u)));
  }
  if (rounded==(hidden<<1)) { rounded>>=1; ++quantum; }
  if (rounded<hidden) return sign|rounded;
  int exponent=quantum+1075;
  if (exponent>=2047) return sign|inf;
  return sign|((uint64_t)exponent<<52)|(rounded-hidden);
}
"""


def emit_float_arithmetic_c(source, name):
    """Emit one checked logical operation, without a floating runtime or FENV use.

    Preserve complete significands before the one declared rounding step.
    Reject unqualified NaNs and invalid infinity operations across the entire
    tensor before any output write. BF16 source add/sub is not truncated.
    """
    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("float arithmetic needs one logical computation and dense buffers")
    node = pure[0]
    if node.op not in FLOAT_ARITHMETIC_OPERATIONS or len(node.inputs) != 2 or source.outputs != (node.id,):
        raise ValueError("float arithmetic needs one supported operation and result")
    keys = tuple(dict.fromkeys(node.inputs))
    bf16 = node.op.startswith("bf16_")
    dtype, width = ("bf16", 2) if bf16 else ("f64", 8)
    policy = f"ieee754-{dtype}-{node.op[len(dtype) + 1:]}-rne-v1"
    type_ = TensorType(node.type.shape, dtype, "raw-bit-pattern")
    if (
        not type_.shape
        or node.type != type_
        or dict(node.attrs) != {"numeric_semantics": policy}
        or node.index_maps != pointwise_identity_maps(len(type_.shape), 2)
        or {item.id for item in source.nodes} != {*keys, node.id}
        or any(
            source.node(key).type != type_
            or source.node(key).effect not in {"input", "constant"}
            or source.node(key).index_maps
            for key in keys
        )
    ):
        raise ValueError("float arithmetic source types, indexing or numerical policy differ")
    count = prod(type_.shape)
    size = count * width
    if size > 2**63 - 1:
        raise ValueError("float arithmetic exceeds its signed byte-size domain")
    constants = {}
    for binding in source.constants:
        if binding.encoding != dtype + "-le-raw":
            raise ValueError("float arithmetic constant needs its exact source encoding")
        data = bytes.fromhex(binding.data_hex)
        constants[binding.node_id] = tuple(
            int.from_bytes(data[i:i + width], "little") for i in range(0, len(data), width)
        )
        magnitude, infinity = (32767, 32640) if bf16 else (0x7FFFFFFFFFFFFFFF, 0x7FF0000000000000)
        if any(bits & magnitude > infinity for bits in constants[binding.node_id]):
            raise ValueError("float arithmetic constant NaN payload is unqualified")
    if all(key in constants for key in node.inputs):
        for left, right in zip(*(constants[key] for key in node.inputs), strict=True):
            a, b = left & magnitude, right & magnitude
            if bf16:
                right ^= 32768 if node.op == "bf16_sub" else 0
                if a == b == infinity and (left ^ right) & 32768:
                    raise ValueError("constant infinity cancellation is outside the source arithmetic domain")
            elif a == infinity and b == 0 or b == infinity and a == 0:
                raise ValueError("constant zero-times-infinity is outside the source arithmetic domain")
    checks = [
        f"!inputs[{i}] || (uintptr_t)inputs[{i}]>UINTPTR_MAX-{size}u || "
        f"((uintptr_t)inputs[{i}]<b+{size}u && b<(uintptr_t)inputs[{i}]+{size}u)"
        for i in range(len(keys))
    ]
    a, b = (keys.index(key) for key in node.inputs)
    helpers = (_BF16_HELPERS.replace("SUBTRACT_SIGN", "32768u" if node.op == "bf16_sub" else "0u")
               if bf16 else _HELPERS)
    for symbol in ("load", "valid", "compute", "expand", "bit"):
        helpers = helpers.replace(symbol + "(", name + "_" + symbol + "(")
    implementation = f"""#include <stddef.h>
#include <stdint.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && SIZE_MAX>={size}ULL,"float arithmetic host types");
{helpers}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !outputs[0]) return -1;
  uintptr_t b=(uintptr_t)outputs[0];
  if (b>UINTPTR_MAX-{size}u || {" || ".join(checks)}) return -1;
  for (size_t i=0;i<{count}u;++i)
    if (!{name}_valid({name}_load((const unsigned char *)inputs[{a}]+{width}u*i),
                     {name}_load((const unsigned char *)inputs[{b}]+{width}u*i))) return -2;
  for (size_t i=0;i<{count}u;++i) {{
    uint64_t bits={name}_compute({name}_load((const unsigned char *)inputs[{a}]+{width}u*i),
                               {name}_load((const unsigned char *)inputs[{b}]+{width}u*i));
    for (unsigned j=0;j<{width}u;++j) ((unsigned char *)outputs[0])[{width}u*i+j]=(unsigned char)(bits>>(8u*j));
  }}
  return 0;
}}
"""
    return {
        "source_request_digest": source.digest(),
        "implementation": name,
        "inputs": list(keys),
        "outputs": [node.id],
        "arithmetic": f"exact-source-{node.op.replace('_', '-')}" if bf16 else "exact-source-f64-multiply",
        "numeric_semantics": policy,
        "domain": ("non-NaN operands, excluding infinity cancellation; reject before output writes" if bf16
                   else "non-NaN operands, excluding zero times infinity; reject before output writes"),
    }, c_source_pair(
        name,
        public_headers={},
        private_headers={},
        sources={"float_arithmetic.c": implementation},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n",
    )
