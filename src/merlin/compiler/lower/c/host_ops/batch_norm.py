"""Scalar BF16 inference batch norm with exact binary32 steps on integer carriers.

Per channel: alpha = RN(RN(1 / RN(sqrt(RN(var + eps)))) * weight) and beta = RN(bias - mean * alpha)
(fused). Each element computes RN_bf16(RN(x * alpha + beta)) with one fused binary32 rounding, then
BF16 round to nearest even. The Python helpers are exact rationals; the C uses integer carriers.
Non-finite input or terms return -2 before any output write.
"""

from fractions import Fraction
from math import isqrt, prod

from merlin.compiler.frontend.patterns.linalg_batch_norm import INPUT_DOMAIN, NUMERIC_SEMANTICS
from merlin.compiler.semantic.constant_folding import _components, _encode
from merlin.compiler.semantic.model import IndexMap, TensorType, pointwise_identity_maps

from ..dispatch import c_source_pair

ROLES = ("mean", "var", "weight", "bias")


def _value(bits: int, dtype: str) -> Fraction:
    sign, significand, exponent = _components(bits, dtype)
    if significand is None:
        raise ValueError("batch norm term is not finite")
    return (-1) ** sign * Fraction(significand) * Fraction(2) ** exponent


def _round(value: Fraction, zero_sign: int = 0) -> int:
    return _encode(value.numerator, value.denominator, 0, "f32", zero_sign)


def _sqrt(bits: int) -> int:
    """Correctly rounded binary32 square root of a positive finite binary32 value."""
    value = _value(bits, "f32")
    if value <= 0:
        raise ValueError("batch norm variance plus epsilon must be positive")
    scale = 200  # even power of two leaving at least 26 exact root bits, even for the smallest subnormal
    scaled = value * Fraction(2) ** scale
    integer = scaled.numerator // scaled.denominator
    root = isqrt(integer)
    sticky = root * root != integer or scaled.denominator != 1 or integer != scaled
    # Two extra low bits make the sticky flag strictly below every rounding boundary.
    return _encode(4 * root + (1 if sticky else 0), 1, -scale // 2 - 2, "f32")


def _negative(bits: int, dtype: str) -> int:
    return bits >> (15 if dtype == "bf16" else 31)


def _fused(a: tuple[int, str], b: tuple[int, str], c: tuple[int, str] | None) -> int:
    """RN32(a * b + c) with one rounding, or RN32(a * b) when c is None. IEEE signs for exact zeros."""
    product = _value(*a) * _value(*b)
    product_negative = _negative(*a) ^ _negative(*b)
    if c is None:
        return _round(product, product_negative)
    addend = _value(*c)
    zero_sign = product_negative & _negative(*c) if not product and not addend else 0
    return _round(product + addend, zero_sign)


def channel_terms(operands: dict[str, tuple[int, str]], epsilon_bits: int) -> tuple[int, int]:
    """Return binary32 (alpha, beta) bits from one channel's raw operand bits and dtypes."""
    total = _fused((0x3F800000, "f32"), operands["var"], (epsilon_bits, "f32"))
    inverse = _round(1 / _value(_sqrt(total), "f32"))
    alpha = _fused((inverse, "f32"), operands["weight"], None) if "weight" in operands else inverse
    mean_bits, mean_dtype = operands["mean"]
    negated = (mean_bits ^ (0x8000 if mean_dtype == "bf16" else 0x80000000), mean_dtype)
    beta = _fused(negated, (alpha, "f32"), operands.get("bias", (0, "f32")))
    for bits in (alpha, beta):
        _value(bits, "f32")  # refuse infinite terms
    return alpha, beta


def element(bits: int, alpha: int, beta: int) -> int:
    """RN_bf16(RN32(x * alpha + beta)) for a finite BF16 x."""
    wide = _fused((bits, "bf16"), (alpha, "f32"), (beta, "f32"))
    lower, upper = wide & 0xFFFF, wide >> 16
    if (wide & 0x7F800000) != 0x7F800000 and (lower > 0x8000 or lower == 0x8000 and upper & 1):
        upper += 1
    return upper


def _channel_maps(rank):
    channel = IndexMap(rank, (tuple(int(axis == 1) for axis in range(rank)),), (0,))
    identity = pointwise_identity_maps(rank, 1)[0]
    return identity, channel


_C_ARITHMETIC = r"""
typedef unsigned __int128 wide_t;
typedef struct { unsigned neg; uint64_t sig; int32_t exp; } exact_t; /* (-1)^neg * sig * 2^exp */
static int bits_of(wide_t v) {
  uint64_t hi=(uint64_t)(v>>64),lo=(uint64_t)v;
  return hi?128-__builtin_clzll(hi):lo?64-__builtin_clzll(lo):0;
}
/* Round sign*R*2^E once to binary32 with ties to even, including subnormals and overflow. */
static uint32_t round32(unsigned sign,wide_t r,int32_t e) {
  uint32_t s=(uint32_t)sign<<31;
  if (!r) return s;
  int32_t top=e+bits_of(r)-1,quantum=top-23<-149?-149:top-23,shift=quantum-e;
  uint64_t sig;
  if (shift<=0) sig=(uint64_t)(r<<-shift);
  else if (shift>=127) sig=0;
  else {
    wide_t rest=r&(((wide_t)1<<shift)-1),half=(wide_t)1<<(shift-1);
    sig=(uint64_t)(r>>shift);
    if (rest>half || (rest==half && (sig&1u))) ++sig;
  }
  if (sig==(1ull<<24)) { sig>>=1; ++quantum; }
  if (sig<(1ull<<23)) return s|(uint32_t)sig;
  int32_t biased=quantum+150;
  if (biased>=255) return s|0x7f800000u;
  return s|((uint32_t)biased<<23)|((uint32_t)sig-(1u<<23));
}
static exact_t decode(uint32_t bits,int bf16) {
  exact_t v;
  unsigned exponent=bf16?(bits>>7)&255u:(bits>>23)&255u;
  v.neg=bf16?(bits>>15)&1u:bits>>31;
  v.sig=bf16?((bits&127u)|(exponent?128u:0u)):((bits&0x7fffffu)|(exponent?0x800000u:0u));
  v.exp=(int32_t)(exponent?exponent:1u)-(bf16?134:150);
  return v;
}
static int finite(uint32_t bits,int bf16) { return bf16?((bits>>7)&255u)!=255u:((bits>>23)&255u)!=255u; }
static exact_t product(exact_t a,exact_t b) { exact_t p={a.neg^b.neg,a.sig*b.sig,a.exp+b.exp}; return p; }
/* RN32(a+b) for exact terms of at most 64 significant bits, with IEEE signs for exact zeros. */
static uint32_t sum32(exact_t a,exact_t b) {
  if (!a.sig && !b.sig) return (a.neg&&b.neg)?0x80000000u:0u;
  if (!a.sig) return round32(b.neg,b.sig,b.exp);
  if (!b.sig) return round32(a.neg,a.sig,a.exp);
  int32_t ta=a.exp+bits_of(a.sig),tb=b.exp+bits_of(b.sig),e=(ta>tb?ta:tb)-100;
  wide_t va,vb;
  int32_t sa=a.exp-e,sb=b.exp-e;
  va=sa>=0?(wide_t)a.sig<<sa:sa<=-64?1:((wide_t)(a.sig>>-sa)|((a.sig&((1ull<<-sa)-1))!=0));
  vb=sb>=0?(wide_t)b.sig<<sb:sb<=-64?1:((wide_t)(b.sig>>-sb)|((b.sig&((1ull<<-sb)-1))!=0));
  if (a.neg==b.neg) return round32(a.neg,va+vb,e);
  if (va==vb) return 0u; /* exact cancellation rounds to +0 */
  return va>vb?round32(a.neg,va-vb,e):round32(b.neg,vb-va,e);
}
static uint32_t sqrt32(uint32_t bits) { /* positive finite binary32 */
  exact_t v=decode(bits,0);
  int32_t normalize=__builtin_clzll(v.sig)-40; /* subnormals get a 24-bit significand: at least 32 root bits */
  v.sig<<=normalize; v.exp-=normalize;
  if (v.exp&1) { v.sig<<=1; --v.exp; }
  wide_t n=(wide_t)v.sig<<40,root=0,bit=(wide_t)1<<66;
  while (bit>n) bit>>=2;
  while (bit) {
    if (n>=root+bit) { n-=root+bit; root=(root>>1)+bit; } else root>>=1;
    bit>>=2;
  }
  return round32(0,(root<<2)|(n!=0),(v.exp-40)/2-2);
}
static uint32_t reciprocal32(uint32_t bits) { /* positive finite binary32 */
  exact_t v=decode(bits,0);
  wide_t q=((wide_t)1<<60)/v.sig,r=((wide_t)1<<60)%v.sig;
  return round32(0,(q<<1)|(r!=0),-v.exp-61);
}
"""


def emit_bf16_batch_norm_c(source, name):
    """Emit runtime per-channel terms and element updates; statistics may be inputs or constants."""
    pure = [n for n in source.nodes if n.effect == "pure"]
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("BF16 batch norm needs one dense logical computation")
    node = pure[0]
    attrs = dict(node.attrs)
    roles = tuple(str(attrs.get("channel_operands", "")).split(","))
    if (
        node.op != "bf16_batch_norm_inference"
        or source.outputs != (node.id,)
        or attrs.get("numeric_semantics") != NUMERIC_SEMANTICS
        or attrs.get("input_domain") != INPUT_DOMAIN
        or set(attrs) != {"numeric_semantics", "input_domain", "epsilon_f32_bits", "channel_operands"}
        or roles[:2] != ROLES[:2]
        or not set(roles) <= set(ROLES)
        or len(set(roles)) != len(roles)
        or len(node.inputs) != 1 + len(roles)
        or len(set(node.inputs)) != len(node.inputs)
        or len(node.type.shape) < 2
        or node.type != TensorType(node.type.shape, "bf16", "raw-bit-pattern")
    ):
        raise ValueError("BF16 batch norm source type, operands, or policy differs")
    identity, channel = _channel_maps(len(node.type.shape))
    if node.index_maps != (identity, *(channel,) * len(roles), identity):
        raise ValueError("BF16 batch norm indexing differs from activation and channel maps")
    x = source.node(node.inputs[0])
    if x.type != node.type or x.effect != "input" or x.attrs or x.index_maps:
        raise ValueError("BF16 batch norm activation operand differs")
    epsilon = attrs["epsilon_f32_bits"]
    if type(epsilon) is not int or not 0 <= epsilon < 0x7F800000:
        raise ValueError("BF16 batch norm epsilon must be a finite nonnegative binary32")
    constants = {binding.node_id: binding for binding in source.constants}
    channels = node.type.shape[1]
    inputs, terms = [x.id], {}
    statics = []
    for role, key in zip(roles, node.inputs[1:]):
        term = source.node(key)
        if term.type not in {TensorType((channels,), dtype, "raw-bit-pattern") for dtype in ("bf16", "f32")}:
            raise ValueError("BF16 batch norm statistics and affine terms must be per-channel BF16 or f32")
        if term.effect == "constant":
            binding = constants.get(key)
            if binding is None or binding.encoding != f"{term.type.dtype}-le-raw":
                raise ValueError("BF16 batch norm constant term encoding differs")
            initializer = ",".join(str(b) for b in bytes.fromhex(binding.data_hex))
            statics.append(f"static const unsigned char {role}_bytes[]={{{initializer}}};")
            terms[role] = (f"{role}_bytes", term.type.dtype)
        elif term.effect == "input" and not term.attrs and not term.index_maps:
            terms[role] = (f"((const unsigned char *)inputs[{len(inputs)}])", term.type.dtype)
            inputs.append(term.id)
        else:
            raise ValueError("BF16 batch norm term must be an input or a constant")
    count, inner = prod(node.type.shape), prod(node.type.shape[2:])
    size = 2 * count
    if size > 2**63 - 1 or channels * 4 > 2**63 - 1:
        raise ValueError("BF16 batch norm exceeds the signed byte-size domain")

    def load(role):
        pointer, dtype = terms[role]
        if dtype == "bf16":
            return f"(uint32_t)({pointer}[2*c]|({pointer}[2*c+1]<<8))", 1
        return (
            f"((uint32_t){pointer}[4*c]|((uint32_t){pointer}[4*c+1]<<8)|((uint32_t){pointer}[4*c+2]<<16)|((uint32_t){pointer}[4*c+3]<<24))",
            0,
        )

    loads = "\n".join(
        f"    uint32_t {role}={load(role)[0]};\n    if (!finite({role},{load(role)[1]})) return -2;" for role in roles
    )
    weight = (
        (
            "exact_t scale=product(decode(inverse,0),decode(weight,{}));\n"
            "    *alpha=round32(scale.neg,scale.sig,scale.exp);"
        ).format(load("weight")[1])
        if "weight" in roles
        else "*alpha=inverse;"
    )
    bias = f"decode(bias,{load('bias')[1]})" if "bias" in roles else "(exact_t){0,0,0}"
    checks = " || ".join(f"!inputs[{i}]" for i in range(len(inputs)))
    overlap_checks = "\n".join(
        f"  uintptr_t term{i}=(uintptr_t)inputs[{i}];\n"
        f"  if (term{i}>UINTPTR_MAX-{channels * (2 if terms[role][1] == 'bf16' else 4)}u || "
        f"(term{i}<b+{size}u && b<term{i}+{channels * (2 if terms[role][1] == 'bf16' else 4)}u)) return -1;"
        for i, role in enumerate((role for role in roles if terms[role][0].startswith("((")), start=1)
    )
    code = f"""#include <stddef.h>
#include <stdint.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && SIZE_MAX>={size}ULL,"BF16 batch norm host types");
{_C_ARITHMETIC}
{chr(10).join(statics)}
static int {name}_channel_terms(const void *const *inputs,size_t c,uint32_t *alpha,uint32_t *beta) {{
{loads}
    exact_t epsilon=decode({epsilon}u,0);
    uint32_t total=sum32(decode(var,{load("var")[1]}),epsilon);
    if (total>=0x7f800000u || !(total&0x7fffffffu)) return -2;
    uint32_t root=sqrt32(total),inverse=reciprocal32(root);
    {weight}
    exact_t mean_value=decode(mean,{load("mean")[1]});
    mean_value.neg^=1u;
    *beta=sum32(product(mean_value,decode(*alpha,0)),{bias});
    return (!finite(*alpha,0) || !finite(*beta,0))?-2:0;
}}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || {checks} || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0],b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-{size}u || b>UINTPTR_MAX-{size}u || (a<b+{size}u && b<a+{size}u)) return -1;
{overlap_checks}
  const unsigned char *x=(const unsigned char *)inputs[0];
  unsigned char *y=(unsigned char *)outputs[0];
  uint32_t alpha,beta;
  for (size_t i=0;i<{size}u;i+=2u) if ((x[i+1]&127u)==127u && (x[i]&128u)) return -2;
  for (size_t c=0;c<{channels}u;++c) {{
    if ({name}_channel_terms(inputs,c,&alpha,&beta)) return -2;
  }}
  /* All source operands are disjoint from output. Recompute checked channel
   * terms in constant space; no shape-dependent scratch or shared state. */
  size_t previous={channels}u;
  for (size_t i=0;i<{count}u;++i) {{
    size_t c=(i/{inner}u)%{channels}u;
    if (c!=previous) {{ (void){name}_channel_terms(inputs,c,&alpha,&beta); previous=c; }}
    uint32_t wide=sum32(product(decode((uint32_t)x[2*i]|((uint32_t)x[2*i+1]<<8),1),decode(alpha,0)),decode(beta,0));
    unsigned upper=wide>>16,lower=wide&65535u;
    if ((wide&0x7f800000u)!=0x7f800000u && (lower>32768u || (lower==32768u && (upper&1u)))) ++upper;
    y[2*i]=(unsigned char)(upper&255u);
    y[2*i+1]=(unsigned char)(upper>>8);
  }}
  return 0;
}}
"""
    # Generated regions share one translation unit; give every file-scope helper the region's name.
    identifiers = (
        "wide_t",
        "exact_t",
        "bits_of",
        "round32",
        "decode",
        "finite",
        "product",
        "sum32",
        "sqrt32",
        "reciprocal32",
        "alpha",
        "beta",
        *(f"{role}_bytes" for role in roles),
    )
    tokens, at = [], 0
    while at < len(code):
        end = at + 1
        if code[at].isascii() and (code[at].isalpha() or code[at] == "_"):
            while end < len(code) and code[end].isascii() and (code[end].isalnum() or code[end] == "_"):
                end += 1
        token = code[at:end]
        tokens.append(f"{name}_{token}" if token in identifiers else token)
        at = end
    code = "".join(tokens)
    record = dict(
        source_request_digest=source.digest(),
        implementation=name,
        inputs=inputs,
        outputs=[node.id],
        numeric_semantics=NUMERIC_SEMANTICS,
        input_domain=INPUT_DOMAIN,
        scratch_policy="constant-space-channel-term-recomputation",
        workspace_bytes=0,
    )
    return record, c_source_pair(
        name,
        public_headers={},
        private_headers={},
        sources={"batch_norm.c": code},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n",
    )
