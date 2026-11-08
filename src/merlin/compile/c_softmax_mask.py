"""Render checked affine additive or Boolean masks for a selected softmax policy."""

from math import prod
import math
import struct

from merlin.compile.c_conversions import render_f32_array_c
from merlin.semantic_compiler.guarded_softmax_source import GuardedMaskedRowSoftmaxSource


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


def render_masked_softmax_inputs(region, mask_shape, spec, pad_element, prefix, *, finite_sentinel_bits=None):
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
    uint32_t bits; __builtin_memcpy(&bits,scores+i,4u);
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
        render_f32_array_c(spec, count, prefix + "_encode")
        + f"""
static size_t {prefix}_mask_index(size_t index) {{
{chr(10).join(steps)}
  return (size_t)({expression});
}}
static int {prefix}_prepare(const float *scores,const void *mask,uint8_t *encoded,uint8_t *keep) {{
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
