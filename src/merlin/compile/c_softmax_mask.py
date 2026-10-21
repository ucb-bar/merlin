"""Render checked affine additive or Boolean masks for a selected softmax policy."""

from math import prod

from merlin.compile.c_conversions import render_f32_array_c
from merlin.semantic_compiler.guarded_softmax_source import GuardedMaskedRowSoftmaxSource


def render_masked_softmax_inputs(region, mask_shape, spec, pad_element, prefix):
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
  for (size_t i=0; i<{prod(mask_shape)}u; ++i) {{
    size_t at=i; {load}
    if ({"bits>1u" if guarded else "bits!=0u && bits!=0xff800000u"}) return 1;
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
