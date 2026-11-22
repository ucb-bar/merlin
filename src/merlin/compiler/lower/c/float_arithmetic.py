"""Compile explicitly CPU-placed IEEE arithmetic using integer bit carriers."""

from math import prod

from merlin.compiler.semantic.model import TensorType, pointwise_identity_maps

from merlin.compiler.lower.c.dispatch import c_source_pair

ORDER = {'f32_add': 'Project operands, exact signed sum, one binary32 RNE rounding.', 'f32_sub': 'Project operands, flip right sign, exact signed sum, one binary32 RNE rounding.', 'f32_mul': 'Project operands, exact significand product, one binary32 RNE rounding.', 'f32_div': 'Project operands, exact significand ratio, one binary32 RNE rounding.', 'f64_mul': 'Project operands, exact 106-bit significand product, one binary64 RNE rounding.', 'bf16_add': 'Project operands, exact signed sum after BF16 readout, one BF16 RNE rounding.', 'bf16_sub': 'Project operands, flip right sign, exact signed sum, one BF16 RNE rounding.'}


FLOAT_ARITHMETIC_OPERATIONS = ("f32_add", "f32_sub", "f32_div", "f32_mul", "f64_mul", "bf16_add", "bf16_sub", "bf16_neg", "f32_neg", "f64_neg")


def _emit_float_negation_c(source, name):
    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("float sign flip needs one dense logical computation")
    node = pure[0]
    dtype = node.op.removesuffix("_neg")
    policy = f"{dtype}-sign-bit-flip-v1"
    if (dtype not in {"bf16", "f32", "f64"} or len(node.inputs) != 1 or source.outputs != (node.id,)
            or node.type.dtype != dtype or node.type.numerical_policy != "raw-bit-pattern"
            or dict(node.attrs) != {"numeric_semantics": policy}
            or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
            or len(source.nodes) != 2 or source.node(node.inputs[0]).type != node.type
            or source.node(node.inputs[0]).effect not in {"input", "constant"}
            or source.node(node.inputs[0]).attrs or source.node(node.inputs[0]).index_maps):
        raise ValueError("float sign flip source types, indexing, or policy differ")
    width = {"bf16": 2, "f32": 4, "f64": 8}[dtype]
    size = prod(node.type.shape) * width
    if not node.type.shape or size > 2**63 - 1:
        raise ValueError("float sign flip exceeds its byte-size domain")
    for binding in source.constants:
        if binding.node_id != node.inputs[0] or binding.encoding != dtype + "-le-raw" or len(bytes.fromhex(binding.data_hex)) != size:
            raise ValueError("float sign flip constant encoding differs")
    implementation = f"""#include <stddef.h>
#include <stdint.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && SIZE_MAX>={size}ULL,"float sign flip host types");
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !inputs[0] || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0],b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-{size}u || b>UINTPTR_MAX-{size}u || (a<b+{size}u && b<a+{size}u)) return -1;
  const unsigned char *input=(const unsigned char *)inputs[0];
  unsigned char *output=(unsigned char *)outputs[0];
  for (size_t i=0;i<{size}u;++i) output[i]=input[i]^((i%{width}u=={width-1}u)?128u:0u);
  return 0;
}}
"""
    return {
        "source_request_digest": source.digest(), "implementation": name,
        "inputs": list(node.inputs), "outputs": [node.id], "arithmetic": "exact-source-" + node.op.replace("_", "-"),
        "numeric_semantics": policy, "domain": "all bit patterns, including signed zero and NaN payloads",
    }, c_source_pair(name, public_headers={}, private_headers={}, sources={"float_negation.c": implementation},
                     declarations=f"int {name}(void *, const void *const *, void *const *);\n")

# Exact sums use five limbs for both eight-exponent-bit source formats.
# Format constants determine every field boundary and rounding threshold.
def _add_helpers(dtype, subtract):
    width, fraction = {"bf16": (2, 7), "f32": (4, 23)}[dtype]
    sign = 1 << (8 * width - 1)
    hidden = 1 << fraction
    infinity = 255 << fraction
    subtract_sign = sign if subtract else 0
    return f"""
static uint64_t load(const unsigned char *p) {{
  uint64_t bits=0; for (unsigned i=0;i<{width};++i) bits|=(uint64_t)p[i]<<(8u*i); return bits;
}}
static int valid(uint64_t a,uint64_t b) {{
  b^={subtract_sign}u;
  uint64_t am=a&{sign-1}u,bm=b&{sign-1}u;
  return am<={infinity}u && bm<={infinity}u && !(am=={infinity}u && bm=={infinity}u && ((a^b)&{sign}u));
}}
static void expand(uint64_t bits,uint64_t out[5]) {{
  unsigned exponent=(unsigned)(bits>>{fraction}),shift=exponent?exponent-1u:0u;
  uint64_t significand=(bits&{hidden-1}u)|(exponent?{hidden}u:0u);
  unsigned limb=shift/64u,offset=shift%64u;
  for (unsigned i=0;i<5;++i) out[i]=0;
  out[limb]=significand<<offset;
  if (offset>{63-fraction}u) out[limb+1u]=significand>>(64u-offset);
}}
static unsigned bit(const uint64_t value[5],unsigned index) {{
  return (unsigned)((value[index/64u]>>(index%64u))&1u);
}}
static uint64_t compute(uint64_t a,uint64_t b) {{
  b^={subtract_sign}u;
  uint64_t am=a&{sign-1}u,bm=b&{sign-1}u;
  if (am=={infinity}u) return a;
  if (bm=={infinity}u) return b;
  if (!am && !bm) return (a&b)&{sign}u;
  if (am<bm) {{ uint64_t swap=a;a=b;b=swap;swap=am;am=bm;bm=swap; }}
  uint64_t x[5],y[5],carry=0;
  expand(am,x);expand(bm,y);
  for (unsigned i=0;i<5;++i) {{
    if (!((a^b)&{sign}u)) {{
      uint64_t sum=x[i]+y[i],next=sum+carry;
      carry=(sum<x[i])|(next<sum);x[i]=next;
    }} else {{
      uint64_t difference=x[i]-y[i],next=difference-carry;
      carry=(x[i]<y[i])|(difference<carry);x[i]=next;
    }}
  }}
  unsigned top=320u;
  while (top && !bit(x,top-1u)) --top;
  if (!top) return 0; /* Exact cancellation under RNE gives positive zero. */
  --top;
  unsigned shift=top>{fraction}u?top-{fraction}u:0u;
  uint64_t rounded=0;
  for (unsigned i=0;i<{fraction+1};++i) rounded|=(uint64_t)bit(x,shift+i)<<i;
  if (shift) {{
    unsigned sticky=0;
    for (unsigned i=0;i+1u<shift;++i) sticky|=bit(x,i);
    rounded+=bit(x,shift-1u) && (sticky || (rounded&1u));
  }}
  if (rounded=={hidden*2}u) {{ rounded>>=1;++shift; }}
  uint64_t sign=a&{sign}u;
  if (rounded<{hidden}u) return sign|rounded;
  if (shift>=254u) return sign|{infinity}u;
  return sign|((uint64_t)(shift+1u)<<{fraction})|(rounded-{hidden}u);
}}
"""

def _multiply_helpers(dtype):
    width, fraction, bias, product_type, product_width = {
        "f32": (4, 23, 127, "uint64_t", 64),
        "f64": (8, 52, 1023, "__uint128_t", 128),
    }[dtype]
    sign = 1 << (8 * width - 1)
    infinity = ((1 << (8 * width - fraction - 1)) - 1) << fraction
    minimum = 1 - bias - fraction
    return f"""
static uint64_t load(const unsigned char *p) {{
  uint64_t bits=0; for (unsigned i=0;i<{width};++i) bits|=(uint64_t)p[i]<<(8u*i); return bits;
}}
static int valid(uint64_t a,uint64_t b) {{
  uint64_t am=a&UINT64_C({sign-1}), bm=b&UINT64_C({sign-1});
  const uint64_t inf=UINT64_C({infinity});
  return am<=inf && bm<=inf && !(am==inf && !bm) && !(bm==inf && !am);
}}
static uint64_t compute(uint64_t a,uint64_t b) {{
  uint64_t sign=(a^b)&UINT64_C({sign});
  uint64_t am=a&UINT64_C({sign-1}), bm=b&UINT64_C({sign-1});
  const uint64_t inf=UINT64_C({infinity}), hidden=UINT64_C(1)<<{fraction};
  if (am==inf || bm==inf) return sign|inf;
  if (!am || !bm) return sign;
  unsigned ae=(unsigned)(am>>{fraction}), be=(unsigned)(bm>>{fraction});
  uint64_t as=(am&(hidden-1u))|(ae?hidden:0), bs=(bm&(hidden-1u))|(be?hidden:0);
  int power=(int)(ae?ae:1)+(int)(be?be:1)-{2*(bias+fraction)};
  {product_type} product=({product_type})as*bs;
  unsigned top=0;
  for ({product_type} rest=product;rest>1;rest>>=1) ++top;
  int highest=(int)top+power;
  if (highest>{bias}) return sign|inf;
  int quantum=highest-{fraction}>{minimum} ? highest-{fraction} : {minimum};
  int shift=quantum-power;
  uint64_t rounded;
  if (shift<=0) rounded=(uint64_t)(product<<(unsigned)-shift);
  else if (shift>={product_width}) rounded=0;
  else {{
    {product_type} high=product>>(unsigned)shift;
    {product_type} low=product- (high<<(unsigned)shift), half=({product_type})1<<(unsigned)(shift-1);
    rounded=(uint64_t)high+(low>half || (low==half && (high&1u)));
  }}
  if (rounded==(hidden<<1)) {{ rounded>>=1; ++quantum; }}
  if (rounded<hidden) return sign|rounded;
  int exponent=quantum+{bias+fraction};
  if (exponent>={(infinity>>fraction)}) return sign|inf;
  return sign|((uint64_t)exponent<<{fraction})|(rounded-hidden);
}}
"""

_F32_DIV_HELPERS = r"""
static uint64_t load(const unsigned char *p) {
  uint64_t bits=0; for (unsigned j=0;j<4;++j) bits|=(uint64_t)p[j]<<(8*j); return bits;
}
static int valid(uint64_t a,uint64_t b) {
  a&=0x7fffffffu;b&=0x7fffffffu;
  return a<=0x7f800000u && b<=0x7f800000u && !(a==0x7f800000u && b==a) && !(a==0 && b==0);
}
static uint64_t compute(uint64_t a,uint64_t b) {
  uint64_t sign=(a^b)&0x80000000u,am=a&0x7fffffffu,bm=b&0x7fffffffu;
  if (am==0x7f800000u || !bm) return sign|0x7f800000u;
  if (!am || bm==0x7f800000u) return sign;
  int ae=(int)(am>>23),be=(int)(bm>>23);
  uint64_t x=(am&0x7fffffu)|(ae?0x800000u:0),y=(bm&0x7fffffu)|(be?0x800000u:0);
  ae=ae?ae-127:-126;be=be?be-127:-126;
  while (x<0x800000u) { x<<=1;--ae; }
  while (y<0x800000u) { y<<=1;--be; }
  int highest=ae-be-(x<y);
  if (highest>127) return sign|0x7f800000u;
  if (highest < -150) return sign;
  int quantum=highest-23 > -149 ? highest-23 : -149;
  /* Normalized significands are 24 bits. The retained exponent range
     bounds shift to [-1,24], so numerator and doubled remainder fit uint64. */
  int shift=ae-be-quantum;
  if (shift>=0) x<<=(unsigned)shift; else y<<=(unsigned)-shift;
  uint64_t q=x/y,r=x%y;
  q+=(r*2>y || (r*2==y && (q&1)));
  if (q==0x1000000u) { q>>=1;++quantum; }
  if (q<0x800000u) return sign|q;
  if (quantum+150>=255) return sign|0x7f800000u;
  return sign|((uint64_t)(quantum+150)<<23)|(q-0x800000u);
}
"""


def emit_float_arithmetic_c(source, name):
    """Scalar exact arithmetic with static singleton projection, including bias."""
    if any(node.op in {"bf16_neg", "f32_neg", "f64_neg"} for node in source.nodes):
        return _emit_float_negation_c(source, name)
    pure = [n for n in source.nodes if n.effect == 'pure']
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError('float arithmetic needs one dense logical computation')
    node = pure[0]
    if node.op not in FLOAT_ARITHMETIC_OPERATIONS:
        raise ValueError('float arithmetic operation differs')
    dtype, action = node.op.split('_', 1)
    width = {'bf16': 2, 'f32': 4, 'f64': 8}.get(dtype)
    if width is None:
        raise ValueError('float arithmetic dtype differs')
    sign = 1 << (8 * width - 1)
    magnitude, infinity = {'bf16': (32767, 32640), 'f32': (0x7fffffff, 0x7f800000),
                           'f64': (0x7fffffffffffffff, 0x7ff0000000000000)}[dtype]
    policy = f'ieee754-{dtype}-{action}-rne-v1'
    keys = tuple(dict.fromkeys(node.inputs))
    if (node.op not in FLOAT_ARITHMETIC_OPERATIONS or len(node.inputs) != 2
            or source.outputs != (node.id,) or not node.type.shape
            or node.type != TensorType(node.type.shape, dtype, 'raw-bit-pattern')
            or dict(node.attrs) != {'numeric_semantics': policy}
            or {n.id for n in source.nodes} != {*keys, node.id}
            or len(node.index_maps) != 3
            or node.index_maps[-1] != pointwise_identity_maps(len(node.type.shape), 1)[0]):
        raise ValueError('float arithmetic source type, operation, or policy differs')
    count = prod(node.type.shape)
    sizes = {key: prod(source.node(key).type.shape)*width for key in keys}
    if max(count*width, *sizes.values()) > 2**63-1:
        raise ValueError('float arithmetic exceeds signed byte-size domain')
    indices = []
    for key, mapping in zip(node.inputs, node.index_maps[:-1], strict=True):
        operand = source.node(key)
        if (operand.effect not in {'input', 'constant'} or operand.index_maps
                or operand.type.dtype != dtype or operand.type.numerical_policy != 'raw-bit-pattern'
                or mapping.loop_rank != len(node.type.shape)
                or len(mapping.coefficients) != len(operand.type.shape)):
            raise ValueError('float arithmetic operand differs')
        index, last = '0', -1
        for extent, row, offset in zip(operand.type.shape, mapping.coefficients, mapping.offsets, strict=True):
            axis = next((i for i, c in enumerate(row) if c), None)
            if axis is None and extent == 1 and offset == 0:
                coordinate = '0'
            elif (axis is not None and axis > last and offset == 0
                    and row == tuple(int(i == axis) for i in range(len(row)))
                    and extent == node.type.shape[axis]):
                coordinate = f'((i/{prod(node.type.shape[axis+1:])}u)%{extent}u)'
                last = axis
            else:
                raise ValueError('float arithmetic requires singleton projection maps')
            index = f'(({index})*{extent}u+{coordinate})'
        indices.append(index)
    for binding in source.constants:
        if binding.node_id not in keys or binding.encoding != dtype + '-le-raw':
            raise ValueError('float arithmetic constant encoding differs')
        data = bytes.fromhex(binding.data_hex)
        if len(data) != sizes[binding.node_id] or any(
                int.from_bytes(data[i:i+width], 'little') & magnitude > infinity for i in range(0,len(data),width)):
            raise ValueError('float arithmetic constant NaN payload or size is unqualified')
    constant_data = {b.node_id: bytes.fromhex(b.data_hex) for b in source.constants}
    if all(key in constant_data for key in node.inputs):
        from itertools import product
        for coordinate in product(*(range(n) for n in node.type.shape)):
            bits = []
            for key, mapping in zip(node.inputs, node.index_maps[:-1], strict=True):
                at = 0
                for extent, index in zip(source.node(key).type.shape, mapping.apply(coordinate), strict=True):
                    at = at * extent + index
                bits.append(int.from_bytes(constant_data[key][width*at:width*at+width], 'little'))
            left, right = bits
            a, b = left & magnitude, right & magnitude
            if action == 'mul' and ((a == 0 and b == infinity) or (b == 0 and a == infinity)):
                raise ValueError('constant zero-times-infinity is outside the source arithmetic domain')
            if action == 'div' and ((a == b == 0) or (a == b == infinity)):
                raise ValueError('constant invalid division is outside the source arithmetic domain')
            if action in {'add', 'sub'} and a == b == infinity and (
                    (left ^ right ^ (sign if action == 'sub' else 0)) & sign):
                raise ValueError('constant infinity cancellation is outside the source arithmetic domain')
    if action in {'add', 'sub'}:
        helpers = _add_helpers(dtype, action == 'sub')
    elif action == 'mul':
        helpers = _multiply_helpers(dtype)
    else:
        helpers = _F32_DIV_HELPERS
    # Regions with the same operation and dtype share one copy of the helpers per translation unit,
    # named and guarded by a hash of their text.
    from hashlib import sha256
    prefix = 'merlin_float_' + sha256(helpers.encode()).hexdigest()[:16]
    for symbol in ('load','valid','compute','expand','bit'):
        helpers = helpers.replace(symbol+'(',prefix+'_'+symbol+'(')
    helpers = f'#ifndef {prefix.upper()}\n#define {prefix.upper()}\n{helpers}#endif\n'
    reads = [f'{prefix}_load((const unsigned char *)inputs[{keys.index(key)}]+{width}u*{index})'
             for key,index in zip(node.inputs,indices,strict=True)]
    checks = [f'!inputs[{i}] || (uintptr_t)inputs[{i}]>UINTPTR_MAX-{sizes[key]}u || '
              f'((uintptr_t)inputs[{i}]<b+{count*width}u && b<(uintptr_t)inputs[{i}]+{sizes[key]}u)'
              for i,key in enumerate(keys)]
    implementation = f'''#include <stdint.h>
#include <stddef.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && SIZE_MAX>={max(count*width,*sizes.values())}ULL,"float arithmetic host types");
{helpers}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !outputs[0]) return -1;
  uintptr_t b=(uintptr_t)outputs[0];
  if (b>UINTPTR_MAX-{count*width}u || {' || '.join(checks)}) return -1;
  for (size_t i=0;i<{count}u;++i) if (!{prefix}_valid({reads[0]},{reads[1]})) return -2;
  for (size_t i=0;i<{count}u;++i) {{
    uint64_t bits={prefix}_compute({reads[0]},{reads[1]});
    for (unsigned j=0;j<{width};++j) ((unsigned char *)outputs[0])[{width}*i+j]=(unsigned char)(bits>>(8*j));
  }}
  return 0;
}}
'''
    return dict(source_request_digest=source.digest(), implementation=name, inputs=list(keys),
                outputs=[node.id], arithmetic='exact-source-'+node.op.replace('_','-'),
                numeric_semantics=policy, operation_order=ORDER[node.op],
                domain='non-NaN operands, excluding invalid infinity operations; reject invalid lanes before writes'), c_source_pair(
        name, public_headers={}, private_headers={}, sources={'float_arithmetic.c':implementation},
        declarations=f'int {name}(void *, const void *const *, void *const *);\n')
