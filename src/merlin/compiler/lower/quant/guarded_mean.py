"""Exact integer-sum fast path for a scalar Q/DQ mean.

Every possible signed-i8 sum has a certificate. Rational bounds enclose the
original f32 multiply, serial additions, divide, and final multiply. Ambiguous
sums replay those original operations in their original order. No sample-derived
input bound or floating-point reassociation is used.
"""

import math
from fractions import Fraction

from merlin.compiler.lower.quant.requantization import f32


def _round_even(value):
    whole = value.numerator // value.denominator
    remainder = value - whole
    return whole + int(remainder > Fraction(1, 2) or remainder == Fraction(1, 2) and whole & 1)


def _quantized(value):
    return max(-128, min(127, _round_even(value)))


def derive(count, input_scale, output_scale):
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError("static reduction count1..128 required")
    try:
        input_scale, output_scale = f32(input_scale), f32(output_scale)
    except (OverflowError, TypeError) as error:
        raise ValueError("representable f32 scales required") from error
    if any(not math.isfinite(x) or x <= 0 for x in (input_scale, output_scale)):
        raise ValueError("positive finite f32 scales required")
    try:
        reciprocal = f32(1.0 / output_scale)
    except OverflowError as error:
        raise ValueError("finite f32 reciprocal required") from error
    if not math.isfinite(reciprocal) or reciprocal <= 0:
        raise ValueError("finite nonzero f32 reciprocal required")
    scale, reciprocal_exact = Fraction(input_scale), Fraction(reciprocal)
    # For finite IEEE binary32 RN-even, including gradual underflow:
    # |fl(t)-t| <= 2^-24 |t| + 2^-150. Track absolute error so cancellation
    # never justifies an invalid relative-error bound.
    unit, tiny = Fraction(1, 2**24), Fraction(1, 2**150)
    maximum = 128 * scale
    product_error = unit * maximum + tiny
    rounded_product_max = maximum + product_error
    add_error = Fraction(0)
    for step in range(1, count + 1):
        add_error = (1 + unit) * add_error + unit * step * rounded_product_max + tiny
    sum_error = count * product_error + add_error
    mean_error = sum_error / count + unit * (count * maximum + sum_error) / count + tiny
    scaled_error = reciprocal_exact * mean_error + unit * reciprocal_exact * (maximum + mean_error) + tiny
    float_max = Fraction(float.fromhex("0x1.fffffep127"))
    if any(
        x >= float_max
        for x in (
            rounded_product_max,
            count * maximum + sum_error,
            maximum + mean_error,
            reciprocal_exact * (maximum + mean_error) + scaled_error,
        )
    ):
        raise ValueError("source arithmetic may overflow binary32")
    table = []
    for total in range(-128 * count, 127 * count + 1):
        center = Fraction(total, count) * scale * reciprocal_exact
        low, high = _quantized(center - scaled_error), _quantized(center + scaled_error)
        table.append(low if low == high else -129)
    return dict(
        schema="guarded_quantized_mean_v1",
        count=count,
        input_scale=input_scale,
        output_scale=output_scale,
        reciprocal=reciprocal,
        sum_min=-128 * count,
        sum_max=127 * count,
        error_bound_rational=str(scaled_error),
        error_bound_float=float(scaled_error),
        table=table,
        ambiguous_sums=[i - 128 * count for i, value in enumerate(table) if value == -129],
        arithmetic="binary32 RN-even with gradual underflow; separate multiply/add; increasing reduction index; zero initial accumulator",
        domain="all signed-i8 input arrays of the stated static count",
    )


def _source_replay(proof, stride=1):
    count = proof["count"]
    access = "row[k]" if stride == 1 else f"row[k*{stride}]"
    return f"""  if (code==-129) {{
   float sum=0.0f;
   for (int k=0; k<{count}; ++k) {{
    volatile float value=(float){access}*{proof["input_scale"].hex()}f;
    sum=value+sum;
   }}
   float mean=sum/{float(count).hex()}f;
   float scaled=mean*{proof["reciprocal"].hex()}f;
   if (scaled<=-128.0f) code=-128;
   else if (scaled>=127.0f) code=127;
   else {{
    code=(int32_t)scaled;
    float delta=scaled-(float)code;
    float absolute=delta<0.0f?-delta:delta;
    if (absolute>0.5f || (absolute==0.5f && (code&1)))
     code+=scaled<0.0f?-1:1;
   }}
  }}
"""


def emit_kernel(proof, symbol, channels):
    if proof != derive(proof["count"], proof["input_scale"], proof["output_scale"]):
        raise ValueError("mean certificate changed")
    if type(channels) is not int or channels <= 0:
        raise ValueError("positive static channel count required")
    if not symbol.isidentifier():
        raise ValueError("C identifier required")
    count = proof["count"]
    values = ",".join(map(str, proof["table"]))
    return f"""#include <stdint.h>
static const int16_t {symbol}_table[{len(proof["table"])}] __attribute__((aligned(64)))={{{values}}};
void {symbol}(const int8_t *input, int8_t *output) {{
 for (int channel=0; channel<{channels}; ++channel) {{
  const int8_t *row=input+(int64_t)channel*{count};
  int32_t total=0;
  for (int k=0; k<{count}; ++k) total+=(int32_t)row[k];
  int32_t code={symbol}_table[total+{128 * count}];
{_source_replay(proof)}  output[channel]=(int8_t)code;
 }}
}}
"""


def emit_packed_nhwc_kernel(proof, symbol, batches, channels):
    """Consume contiguous NHWC without materializing a BCHW transpose.

    Bias signed bytes by128, widen alternate bytes to unsigned16 lanes and
    accumulate. Count<=128 bounds every lane by32640, so no carry can cross
    lanes. Removing the bias gives the exact original signed integer sums.
    Unaligned input uses a scalar fallback; little-endian word layout is checked.
    """
    if proof != derive(proof["count"], proof["input_scale"], proof["output_scale"]):
        raise ValueError("mean certificate changed")
    if (
        type(batches) is not int
        or batches <= 0
        or type(channels) is not int
        or channels <= 0
        or channels % 8
        or not symbol.isidentifier()
    ):
        raise ValueError("positive batches, channels divisible by8 and C identifier required")
    count = proof["count"]
    values = ",".join(map(str, proof["table"]))
    replay = _source_replay(proof, stride=channels)
    return f"""#include <stdint.h>
_Static_assert(__BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__,"little-endian packed mean required");
typedef uint64_t {symbol}_alias_word __attribute__((may_alias));
static const int16_t {symbol}_table[{len(proof["table"])}] __attribute__((aligned(64)))={{{values}}};
static int8_t {symbol}_finish(int32_t total,const int8_t *row) {{
 int32_t code={symbol}_table[total+{128 * count}];
{replay}
 return (int8_t)code;
}}
void {symbol}(const int8_t *input,int8_t *output) {{
 if ((uintptr_t)input&7) {{
  for(int batch=0;batch<{batches};++batch) for(int channel=0;channel<{channels};++channel) {{
   const int8_t *row=input+(int64_t)batch*{count * channels}+channel;
   int32_t total=0;
   for(int k=0;k<{count};++k) total+=(int32_t)row[k*{channels}];
   output[batch*{channels}+channel]={symbol}_finish(total,row);
  }}
  return;
 }}
 for(int batch=0;batch<{batches};++batch) for(int channel=0;channel<{channels};channel+=8) {{
  const int8_t *row=input+(int64_t)batch*{count * channels}+channel;
  uint64_t even=0,odd=0;
  for(int k=0;k<{count};++k) {{
   uint64_t word=*(const {symbol}_alias_word*)(row+k*{channels});
   word^=UINT64_C(0x8080808080808080);
   even+=word&UINT64_C(0x00ff00ff00ff00ff);
   odd+=(word>>8)&UINT64_C(0x00ff00ff00ff00ff);
  }}
  for(int lane=0;lane<8;++lane) {{
   uint64_t sum=lane&1?odd:even;
   int32_t total=(int32_t)((sum>>(16*(lane/2)))&65535)-{128 * count};
   output[batch*{channels}+channel+lane]={symbol}_finish(total,row+lane);
  }}
 }}
}}
"""


def emit_integer_sum_finish(proof, symbol, batches, channels, *, input_layout):
    """Finish externally computed exact integer sums under the source proof.

    The producer must sum every original signed-i8 value with zero init and no
    integer overflow. ``sums`` is dense [batches, channels] i32. Original input
    storage is immutable until finishing, with channel-major [B,C,count] or
    channel-minor [B,count,C] layout. Both buffers and the fully written i8
    output are disjoint. An accelerator is one possible sum producer; this
    shared routine makes no target or DMA assumptions.

    Ambiguous sums retain the original ordered binary32 arithmetic. The sum
    range guard prevents out-of-bounds lookup if an external caller violates
    the producer contract; it does not establish that the supplied sum is exact.
    Compile with separate multiply/add and IEEE binary32 RN-even semantics.
    """
    if proof != derive(proof["count"], proof["input_scale"], proof["output_scale"]):
        raise ValueError("mean certificate changed")
    if (
        type(batches) is not int
        or batches <= 0
        or type(channels) is not int
        or channels <= 0
        or not symbol.isidentifier()
    ):
        raise ValueError("positive static batches/channels and C identifier required")
    if input_layout not in ("channel_major", "channel_minor"):
        raise ValueError("explicit channel-major or channel-minor input required")
    count = proof["count"]
    # This follows from the complete source domain, never a captured range.
    if 128 * count > (1 << 31) - 1:
        raise ValueError("signed-i8 reduction may overflow i32")
    offset = f"channel*{count}" if input_layout == "channel_major" else "channel"
    stride = 1 if input_layout == "channel_major" else channels
    values = ",".join(map(str, proof["table"]))
    return f"""#include <stdint.h>
static const int16_t {symbol}_table[{len(proof["table"])}] __attribute__((aligned(64)))={{{values}}};
void {symbol}(const int8_t *input,const int32_t *sums,int8_t *output) {{
 for(int batch=0;batch<{batches};++batch) for(int channel=0;channel<{channels};++channel) {{
  const int8_t *row=input+(int64_t)batch*{count * channels}+{offset};
  int32_t total=sums[(int64_t)batch*{channels}+channel];
  if(total<{proof["sum_min"]} || total>{proof["sum_max"]}) __builtin_trap();
  int32_t code={symbol}_table[total+{128 * count}];
{_source_replay(proof, stride=stride)}  output[(int64_t)batch*{channels}+channel]=(int8_t)code;
 }}
}}
"""
