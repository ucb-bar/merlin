"""Render checked affine additive or Boolean masks for a selected softmax policy."""

from math import prod
import math
import struct

from merlin.compiler.lower.c.conversions import render_f32_array_c, render_bf16_codes_c
from merlin.compiler.semantic.linalg.guarded_softmax_source import GuardedMaskedRowSoftmaxSource


def render_bf16_row_span_preflight(rows, columns, prefix, *, maximum_lower_bound,
                                   maximum_upper_bound, maximum_span, masked=False, empty_rows_zero=False):
    """Check exact row extrema and span before device I/O on encoded BF16.

    Integer limbs represent finite BF16 magnitudes in units of 2^-149.
    Their exact difference is compared to the provider's binary32 span,
    including tiny contributions that host float subtraction could discard.
    Numerical bounds are supplied by the selected provider, never inferred.
    """
    bounds = (maximum_lower_bound, maximum_upper_bound, maximum_span)
    if (type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0
            or not prefix.isascii() or not prefix.isidentifier() or not prefix[0].isalpha()
            or not all(math.isfinite(x) and struct.unpack('<f', struct.pack('<f', x))[0] == x for x in bounds)
            or maximum_lower_bound > maximum_upper_bound or maximum_span <= 0
            or empty_rows_zero and not masked):
        raise ValueError("BF16 row-span preflight needs finite binary32 bounds and nonempty shape")
    low, high = (float(x).hex() + 'f' for x in bounds[:2])
    numerator, denominator = float(maximum_span).as_integer_ratio()
    span = numerator * (1 << 149) // denominator
    limbs = ','.join(f'{(span >> (64*i)) & ((1 << 64)-1)}ULL' for i in range(5))
    return f"""
#include <float.h>
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24 && FLT_MAX_EXP==128,
               "Exact row-span preflight requires IEEE binary32");
static void {prefix}_magnitude(unsigned code,uint64_t out[5]) {{
  unsigned exponent=(code>>7)&255u,significand=(code&127u)|(exponent?128u:0u);
  unsigned shift=exponent?exponent+15u:16u,word=shift/64u,bit=shift%64u;
  for (size_t i=0;i<5u;++i) out[i]=0;
  out[word]=(uint64_t)significand<<bit;
  if (bit>56u) out[word+1u]=(uint64_t)significand>>(64u-bit);
}}
static int {prefix}_span_exceeded(unsigned maximum,unsigned minimum) {{
  uint64_t a[5],b[5],difference[5],carry=0;
  static const uint64_t limit[5]={{{limbs}}};
  {prefix}_magnitude(maximum,a); {prefix}_magnitude(minimum,b);
  for (size_t i=0;i<5u;++i) {{
    if ((maximum^minimum)&0x8000u) {{
      uint64_t sum=a[i]+b[i],overflow=sum<a[i];
      difference[i]=sum+carry; carry=overflow || difference[i]<sum;
    }} else {{
      uint64_t left=(maximum&0x8000u)?b[i]:a[i],right=(maximum&0x8000u)?a[i]:b[i];
      uint64_t delta=left-right,borrow=left<right;
      difference[i]=delta-carry; carry=borrow || delta<carry;
    }}
  }}
  for (size_t i=5u;i>0u;--i) {{
    if (difference[i-1u]>limit[i-1u]) return 1;
    if (difference[i-1u]<limit[i-1u]) return 0;
  }}
  return 0;
}}
static int {prefix}_row_span(const uint8_t *in,const uint8_t *keep) {{
  (void)keep;
  for (size_t row=0;row<{rows}u;++row) {{
    float minimum=0.0f,maximum=0.0f; unsigned any=0,min_code=0,max_code=0;
    for (size_t column=0;column<{columns}u;++column) {{
      size_t at=(row*{columns}u+column)*2u;
      {"unsigned active=(unsigned)keep[at]|((unsigned)keep[at+1u]<<8); if (active!=0u && active!=0x3f80u) return 1; if (!active) continue;" if masked else ""}
      unsigned code=(unsigned)in[at]|((unsigned)in[at+1u]<<8);
      uint32_t bits=(uint32_t)code<<16;
      if ((bits&0x7f800000u)==0x7f800000u) return 1;
      float value; __builtin_memcpy(&value,&bits,4u);
      if (!any || value<minimum) {{ minimum=value; min_code=code; }}
      if (!any || value>maximum) {{ maximum=value; max_code=code; }}
      any=1;
    }}
    if (!any) {{ if ({int(empty_rows_zero)}u) continue; return 1; }}
    if (maximum<{low} || maximum>{high}) return 1;
    if ({prefix}_span_exceeded(max_code,min_code)) return 1;
  }}
  return 0;
}}
"""


def render_masked_softmax_inputs(region, mask_shape, spec, pad_element, prefix, *, finite_sentinel_bits=None, score_dtype="f32"):
    """Convert additive or Boolean exclusion masks to sanitized scores and keeps.

    This is an explicit selected representation change. The device still
    computes the maximum, exponential, denominator and normalization.
    """
    shape = region.shape
    guarded = isinstance(region, GuardedMaskedRowSoftmaxSource)
    count = prod(shape)
    mapping = region.mask_map
    if (
        not prefix.isascii()
        or not prefix.isidentifier()
        or not prefix[0].isalpha()
        or score_dtype not in {"f32", "bf16"}
        or spec.format != "bf16"
        or spec.direction != "encode"
        or len(pad_element) != 2
        or len(mask_shape) != len(mapping.coefficients)
        or count > 1 << 24
        or prod(mask_shape) > 1 << 24
    ):
        raise ValueError("masked softmax packing needs bounded dense BF16 output")
    strides, stride = [], 1
    for extent in reversed(mask_shape):
        strides.insert(0, stride)
        stride *= extent
    coefficients = tuple(
        sum(row[d] * s for row, s in zip(mapping.coefficients, strides, strict=True)) for d in range(len(shape))
    )
    offset = sum(o * s for o, s in zip(mapping.offsets, strides, strict=True))
    low = offset + sum(min(0, c * (n - 1)) for c, n in zip(coefficients, shape, strict=True))
    high = offset + sum(max(0, c * (n - 1)) for c, n in zip(coefficients, shape, strict=True))
    if low < 0 or high >= prod(mask_shape):
        raise ValueError("masked softmax affine map leaves its source")
    mask_domain = "bits>1u" if guarded else "bits!=0u && bits!=0xff800000u"
    score_preflight = ""
    score_type = "uint8_t" if score_dtype == "bf16" else "float"
    read_score = ("uint32_t bits=((uint32_t)scores[2*i]|((uint32_t)scores[2*i+1]<<8))<<16;"
                  if score_dtype == "bf16" else "uint32_t bits; __builtin_memcpy(&bits,scores+i,4u);")
    if finite_sentinel_bits is not None:
        if guarded or type(finite_sentinel_bits) is not int or not 0 <= finite_sentinel_bits <= 0xffffffff:
            raise ValueError("finite additive sentinel needs an explicit binary32 bit pattern")
        sentinel = struct.unpack("<f", struct.pack("<I", finite_sentinel_bits))[0]
        ranges = spec.admitted_code_ranges
        if not ranges or any(lo // 0x8000 != hi // 0x8000 for lo, hi in ranges):
            raise ValueError("finite additive sentinel needs bounded score encodings")
        largest = max(value & 0x7fff for pair in ranges for value in pair)
        bound = struct.unpack("<f", struct.pack("<I", largest << 16))[0]
        half_ulp = math.ldexp(1.0, ((finite_sentinel_bits >> 23) & 255) - 127 - 24)
        # Every admitted score and row maximum must leave the addend unchanged
        # under binary32 RNE. Its exponential then rounds to zero in binary32.
        if (not math.isfinite(sentinel) or sentinel >= 0 or not math.isfinite(bound) or bound <= 0
                or 2 * bound >= half_ulp or sentinel + bound >= -128):
            raise ValueError("finite additive sentinel lacks exact addition and exponential-underflow proof")
        bound_bits = struct.unpack("<I", struct.pack("<f", bound))[0]
        mask_domain = f"bits!=0u && bits!={finite_sentinel_bits}u"
        score_preflight = f"""  for (size_t i=0;i<{count}u;++i) {{
    {read_score}
    if ((bits&0x7fffffffu)>{bound_bits}u) return 1;
  }}
"""
    steps = []
    for dimension in reversed(range(len(shape))):
        steps.append(f"  size_t d{dimension}=index%{shape[dimension]}u; index/={shape[dimension]}u;")
    expression = str(offset) + "".join(f"+({c}LL)*(int64_t)d{d}" for d, c in enumerate(coefficients))
    pad = int.from_bytes(pad_element, "little")
    load = (
        "uint32_t bits=((const uint8_t *)mask)[at];"
        if guarded
        else "uint32_t bits; __builtin_memcpy(&bits,(const uint8_t *)mask+4u*at,4u);"
    )
    return (
        (render_bf16_codes_c if score_dtype == "bf16" else render_f32_array_c)(spec, count, prefix + "_encode")
        + f"""
static size_t {prefix}_mask_index(size_t index) {{
{chr(10).join(steps)}
  return (size_t)({expression});
}}
static int {prefix}_prepare(const {score_type} *scores,const void *mask,uint8_t *encoded,uint8_t *keep) {{
{score_preflight}
  for (size_t i=0; i<{prod(mask_shape)}u; ++i) {{
    size_t at=i; {load}
    if ({mask_domain}) return 1;
  }}
  for (size_t row=0; row<{prod(shape[:-1])}u; ++row) {{
    unsigned any=0;
    for (size_t col=0; col<{shape[-1]}u; ++col) {{
      size_t at={prefix}_mask_index(row*{shape[-1]}u+col); {load}
      any|=bits==0u;
    }}
    if (!any && {int(not guarded)}u) return 1;
  }}
  if ({prefix}_encode(scores,encoded)) return 1;
  for (size_t i=0; i<{count}u; ++i) {{
    size_t at={prefix}_mask_index(i); {load}
    uint16_t code=bits ? 0u : 0x3f80u;
    keep[i*2u]=(uint8_t)code; keep[i*2u+1u]=(uint8_t)(code>>8);
    if (bits) {{ encoded[i*2u]={pad & 255}u; encoded[i*2u+1u]={pad >> 8}u; }}
  }}
  return 0;
}}
"""
    )


def emit_host_softmax_c(region, name, *, host_math, policy_path, contract, policy,
                        mask_shape=()):
    """Package scalar f32 C only for a bound, completely recognized source.

    Scores and optional masks are disjoint from the output. Context is unused;
    exponentials are stored in the output and normalized in place. The pinned
    CORE-MATH expf source is embedded, so the region links no math library.
    """
    from merlin.runtime.host_math_supplier.embedded import embedded_host_math
    from merlin.compiler.lower.c.host_ops.softmax_precision import (
        HostSoftmaxContract, ORDER_RECORD, admit_host_math, validate_policy)
    from merlin.compiler.lower.c.dispatch import c_source_pair
    if not isinstance(contract, HostSoftmaxContract):
        raise ValueError("recognized source contract required")
    validate_policy(policy)
    contract.validate_region(region)
    if not name.isascii() or not name.isidentifier() or not name[0].isalpha():
        raise ValueError("host softmax implementation name must be a C identifier")
    mask_kind = contract.mask_kind
    fill_bits, keep_when = contract.fill_bits, contract.keep_when
    shape = tuple(region.shape)
    if (not shape or any(type(n) is not int or n <= 0 for n in shape)
            or prod(shape) > (2**63-1)//4 or mask_kind not in {'none', 'additive', 'select'}):
        raise ValueError('host softmax needs bounded dense f32 rows and an explicit mask kind')
    count, columns = prod(shape), shape[-1]
    mask_shape = tuple(mask_shape)
    if mask_kind != 'none' and mask_shape != tuple(region.mask_shape):
        raise ValueError('mask shape differs from bound source contract')
    mapping = getattr(region, 'mask_map', None)
    mask_index = 'return 0;'
    mask_count = 0
    if mask_kind != 'none':
        if (any(type(n) is not int or n <= 0 for n in mask_shape)
                or prod(mask_shape) > (2**63-1)//4 or mapping is None
                or len(mapping.coefficients) != len(mask_shape)
                or len(mapping.offsets) != len(mask_shape)
                or any(len(row) != len(shape) for row in mapping.coefficients)):
            raise ValueError('host softmax requires a checked affine mask map')
        for row, origin, extent in zip(mapping.coefficients, mapping.offsets, mask_shape):
            lo = origin + sum(min(0, c*(n-1)) for c, n in zip(row, shape))
            hi = origin + sum(max(0, c*(n-1)) for c, n in zip(row, shape))
            if lo < 0 or hi >= extent:
                raise ValueError('host softmax mask map leaves its source')
        strides, stride = [], 1
        for extent in reversed(mask_shape):
            strides.insert(0, stride)
            stride *= extent
        coefficients = [sum(row[d]*s for row, s in zip(mapping.coefficients, strides)) for d in range(len(shape))]
        offset = sum(o*s for o, s in zip(mapping.offsets, strides))
        low = offset + sum(min(0, c*(n-1)) for c, n in zip(coefficients, shape))
        high = offset + sum(max(0, c*(n-1)) for c, n in zip(coefficients, shape))
        if low < 0 or high >= prod(mask_shape):
            raise ValueError('host softmax mask map leaves its source')
        # Size-one axes contribute zero and need no C integer literal.
        coefficients = [0 if n == 1 else c for c, n in zip(coefficients, shape)]
        # Signed intermediate products must also fit, even when they cancel.
        if abs(offset) + sum(abs(c)*(n-1) for c, n in zip(coefficients, shape)) > 2**63-1:
            raise ValueError('host softmax mask index exceeds int64')
        steps = [f'size_t d{d}=index%{shape[d]}u; index/={shape[d]}u;' for d in reversed(range(len(shape)))]
        expression = str(offset) + ''.join(f'+({c}LL)*(int64_t)d{d}' for d, c in enumerate(coefficients))
        mask_index = ' '.join(steps) + f' return (size_t)({expression});'
        mask_count = prod(mask_shape)
    supplier = admit_host_math(host_math, policy_path)
    if host_math.sha256(policy_path) != host_math.sha256(host_math.POLICY_PATH):
        raise ValueError('host softmax embeds the supplier policy it admits')
    embedded, _ = embedded_host_math(host_math, ('expf',))
    size = count*4
    mask_size = mask_count*(1 if mask_kind == 'select' else 4)
    score = (f'if (mask[{name}_mask_index(i)]!={int(keep_when)}u) return {name}_from({fill_bits}u);'
             if mask_kind == 'select' else
             f'x={name}_round(x+{name}_load(mask+4u*{name}_mask_index(i)));'
             if mask_kind == 'additive' else '(void)mask;')
    implementation = f'''#include <stdint.h>
#include <stddef.h>
#include <float.h>
#include <limits.h>
#include <string.h>
{embedded}
#if defined(__FAST_MATH__)
#error host softmax forbids fast math
#endif
#if defined(__riscv) && (__riscv_xlen!=64 || !defined(__riscv_float_abi_double))
#error host softmax requires RV64 lp64d with enabled FPU
#endif
#if defined(__FLT_EVAL_METHOD__) && __FLT_EVAL_METHOD__!=0
#error host softmax requires binary32 evaluation
#endif
_Static_assert(CHAR_BIT==8 && sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24 && FLT_MAX_EXP==128,"binary32 required");
static float {name}_from(uint32_t b) {{ float x; memcpy(&x,&b,4); return x; }}
static uint32_t {name}_bits(float x) {{ uint32_t b; memcpy(&b,&x,4); return b; }}
static float {name}_load(const unsigned char *p) {{ uint32_t b=0; for(unsigned j=0;j<4;++j) b|=(uint32_t)p[j]<<(8*j); return {name}_from(b); }}
static float {name}_round(float x) {{ volatile float y=x; return y; }}
static float {name}_canon(float x) {{ return ({name}_bits(x)&0x7fffffffu)>0x7f800000u ? {name}_from(0x7fc00000u) : x; }}
static void {name}_store(unsigned char *p,float x) {{ uint32_t b={name}_bits({name}_canon(x)); for(unsigned j=0;j<4;++j) p[j]=(unsigned char)(b>>(8*j)); }}
static inline size_t {name}_mask_index(size_t index) {{ (void)index; {mask_index} }}
static float {name}_score(const unsigned char *scores,const unsigned char *mask,size_t i) {{
  float x={name}_load(scores+4u*i); {score} return {name}_canon(x);
}}
static int {name}_overlap(const void *a,size_t na,const void *b,size_t nb) {{
  uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
  return x>UINTPTR_MAX-na || y>UINTPTR_MAX-nb || (x<y+nb && y<x+na);
}}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  if (!inputs || !outputs || !inputs[0] || !outputs[0]{' || !inputs[1]' if mask_count else ''}) return -1;
  const unsigned char *scores=(const unsigned char *)inputs[0];
  const unsigned char *mask={'(const unsigned char *)inputs[1]' if mask_count else 'NULL'};
  (void)context;
  unsigned char *out=(unsigned char *)outputs[0];
  if ({name}_overlap(scores,{size}u,out,{size}u)) return -1;
  {'if ('+name+'_overlap(mask,'+str(mask_size)+'u,out,'+str(size)+'u)) return -1;' if mask_count else ''}
#if defined(__riscv)
  unsigned rounding; __asm__ volatile("frrm %0" : "=r"(rounding)); if (rounding) return -3;
#elif defined(__x86_64__)
  /* Host EXACT execution: RNE with gradual underflow (no FTZ or DAZ). */
  unsigned mxcsr; __asm__ volatile("stmxcsr %0" : "=m"(mxcsr)); if (mxcsr & 0xe040u) return -3;
#else
#error host softmax needs an RNE query for this target
#endif
  {'for(size_t i=0;i<'+str(mask_count)+'u;++i) if(mask[i]>1u) return -2;' if mask_kind == 'select' else ''}
  for(size_t row=0;row<{count//columns}u;++row) {{
    size_t base=row*{columns}u; float maximum={name}_from(0xff800000u);
    for(size_t c=0;c<{columns}u;++c) {{
      float x={name}_score(scores,mask,base+c);
      if (({name}_bits(x)&0x7fffffffu)>0x7f800000u) maximum={name}_from(0x7fc00000u);
      else if(x>maximum) maximum=x;
    }}
    float total=0.0f;
    for(size_t c=0;c<{columns}u;++c) {{
      float shifted={name}_round({name}_score(scores,mask,base+c)-maximum);
      float e=merlin_host_expf({name}_canon(shifted));
      {name}_store(out+4u*(base+c),e); total={name}_round(total+e);
    }}
    for(size_t c=0;c<{columns}u;++c)
      {name}_store(out+4u*(base+c),{name}_round({name}_load(out+4u*(base+c))/total));
  }}
  return 0;
}}
'''
    metadata = {
        'source_request_digest': region.source_digest,
        'implementation': name, 'inputs': [region.scores] + ([region.mask] if mask_count else []),
        'outputs': [region.result], 'numeric_order': dict(ORDER_RECORD),
        'shape': list(shape), 'workspace_bytes': 0, 'source_contract': contract.record(), 'host_isa': 'rv64gc', 'host_abi': 'lp64d',
        'required_cflags': ['-fno-fast-math', '-ffp-contract=off', '-fno-tree-vectorize'],
        'host_math_revision': supplier['revision'],
        'host_math_policy_sha256': host_math.sha256(policy_path),
        'host_math_supplier': 'embedded-pinned-sources', 'host_math_symbols': ['merlin_host_expf'],
        'host_math_sources_sha256': {supplier['functions']['expf']['source']: supplier['functions']['expf']['sha256']},
        'qualification': 'unqualified-until-independent-spike-exact',
    }
    return metadata, c_source_pair(name, public_headers={}, private_headers={},
        sources={'host_softmax.c': implementation},
        declarations=f'int {name}(void *, const void *const *, void *const *);\n')
