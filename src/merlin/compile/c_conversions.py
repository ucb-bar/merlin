"""Render checked source-tensor conversions for compiled region calls."""

from __future__ import annotations

from merlin.runtime.fp8_formats import float_format_params
from merlin.semantic_compiler.precision_codegen import finite_f32_codebook, render_finite_f32_codebook_c
from merlin.semantic_compiler.precision_transform import BoundaryConversion
from merlin.semantic_compiler.precision_values import FloatBoundarySpec


def render_f32_array_c(spec: FloatBoundarySpec, count: int, prefix: str) -> str:
    """Encode a contiguous C span; its logical rank stays in the caller's IR."""
    from merlin.semantic_compiler.model import TensorType
    from merlin.semantic_compiler.precision_intake import SourceValueRef

    if type(count) is not int or count <= 0:
        raise ValueError("array conversion needs a positive element count")
    conversion = BoundaryConversion(
        "array",
        SourceValueRef("argument", 0),
        TensorType((1, count), "f32", "raw-bit-pattern"),
        TensorType((1, count), spec.format, "raw-bit-pattern"),
        (0, 1),
        spec.policy,
        spec,
    )
    return render_f32_operand_c(conversion, prefix)


def render_bf16_codes_c(spec: FloatBoundarySpec, count: int, prefix: str) -> str:
    """Check and copy little-endian BF16 codes without a numerical conversion.

    The selected code ranges remain explicit. Reject transformations that
    would change already encoded values. As with the f32 encoder, callers
    provide valid disjoint buffers and the complete domain is checked first.
    """
    if (type(count) is not int or count <= 0 or not prefix or not prefix.isascii()
            or not prefix.isidentifier() or not prefix[0].isalpha()
            or spec.direction != "encode" or spec.format != "bf16"
            or spec.pre_encode_scale_exponent or spec.flush_below_normal
            or spec.canonicalize_negative_zero):
        raise ValueError("raw BF16 input needs a code-domain policy without numerical conversion")
    eb, mb, _, scheme = float_format_params(spec.format)
    if (eb, mb, scheme) != (8, 7, "ieee"):
        raise ValueError("raw BF16 input requires its IEEE exponent layout")
    maximum = (1 << eb) - 2 if spec.max_normal_exponent_code is None else spec.max_normal_exponent_code
    limit = (maximum << mb) | ((1 << mb) - 1)
    admitted = " || ".join(
        f"(code >= {low}u && code <= {high}u)" if low else f"code <= {high}u"
        for low, high in spec.admitted_code_ranges
    ) or "1"
    return f"""#include <stddef.h>
#include <stdint.h>
_Static_assert(SIZE_MAX/2 >= {count}ULL, "source tensor exceeds host size_t");
static int {prefix}(const uint8_t *source,uint8_t *output) {{
  if (!source || !output) return 1;
  for (size_t i=0;i<{count}u;++i) {{
    uint32_t code=(uint32_t)source[2*i]|((uint32_t)source[2*i+1]<<8);
    if ((code&0x7fffu)>{limit}u || !({admitted})) return 2;
  }}
  __builtin_memcpy(output,source,{count * 2}u);
  return 0;
}}
"""


def render_bf16_decode_c(spec: FloatBoundarySpec, prefix: str) -> str:
    """Return binary32 bits after BF16 decode and a constant binary scale.

    Integer operations implement round-to-nearest-even at subnormal boundaries
    and overflow to signed infinity. NaN payloads and signed zeros are preserved;
    the host floating environment and optional math library are not consulted.
    """
    if (
        spec.direction != "decode"
        or spec.format != "bf16"
        or not prefix
        or not prefix.isascii()
        or not prefix.isidentifier()
        or not prefix[0].isalpha()
    ):
        raise ValueError("BF16 decoder requires a decode specification and an ASCII C identifier")
    return f"""#include <stdint.h>
static uint32_t {prefix}(uint16_t code) {{
  uint32_t bits=(uint32_t)code<<16, sign=bits&0x80000000u;
  int exponent=(int)((bits>>23)&255u);
  uint32_t significand=bits&0x7fffffu;
  if (exponent==255 || (exponent==0 && !significand)) return bits;
  if (exponent) significand|=0x800000u;
  else {{
    exponent=1;
    while (significand<0x800000u) {{ significand<<=1; --exponent; }}
  }}
  exponent+={spec.post_decode_scale_exponent};
  if (exponent>=255) return sign|0x7f800000u;
  if (exponent>0) return sign|((uint32_t)exponent<<23)|(significand&0x7fffffu);
  unsigned shift=(unsigned)(1-exponent);
  if (shift>24) return sign;
  uint32_t rounded=(significand+((1u<<(shift-1))-1u)+((significand>>shift)&1u))>>shift;
  return sign|rounded;
}}
"""


def render_f32_operand_c(conversion: BoundaryConversion, prefix: str) -> str:
    """Encode a complete rank-two source window in the recorded axis order.

    The caller supplies valid disjoint buffers. This helper checks the entire
    numerical input domain before writing output. Byte formats use the shared
    finite_float_boundary.h encoder. BF16 uses integer-only RNE and code-range
    checks directly; its pre-encode scaling is currently unsupported.
    """
    if not prefix or not prefix.isascii() or not prefix.isidentifier() or not prefix[0].isalpha():
        raise ValueError("conversion prefix must begin with an ASCII letter")
    spec = conversion.spec
    if (
        conversion.source_type.dtype != "f32"
        or len(conversion.source_type.shape) != 2
        or conversion.source_batch_index
        or conversion.source_offsets != (0, 0)
        or conversion.target_offsets != (0, 0)
        or conversion.window_shape != conversion.source_type.shape
        or conversion.target_type.shape
        != tuple(conversion.source_type.shape[axis] for axis in conversion.target_axes_from_source)
        or spec is None
        or spec.direction != "encode"
        or spec.policy != conversion.conversion_policy
        or conversion.target_type.dtype != spec.format
    ):
        raise ValueError("C operand conversion requires one complete rank-two f32 window")
    rows, cols = conversion.target_type.shape
    source_cols = conversion.source_type.shape[1]
    source_index = (
        f"row*{source_cols}u+col" if conversion.target_axes_from_source == (0, 1) else (f"col*{source_cols}u+row")
    )
    if spec.format == "bf16":
        if spec.pre_encode_scale_exponent:
            raise ValueError("C BF16 conversion does not support a pre-encode scale")
        return _render_bf16_operand_c(conversion, prefix, source_index)
    codebook = prefix + "_codebook"
    table = "\n".join(render_finite_f32_codebook_c(codebook, finite_f32_codebook(spec)))
    return f"""#include <float.h>
#include <stddef.h>
#include <stdint.h>
_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "source conversion requires binary32 float");
_Static_assert(SIZE_MAX/4 >= {rows * cols}ULL, "source tensor exceeds host size_t");
{table}
static int {prefix}(const float *source, uint8_t *output) {{
  if (!source || !output) return MERLIN_F32_U8_INVALID;
  for (unsigned pass = 0; pass < 2; ++pass) {{
    for (size_t row = 0; row < {rows}u; ++row) {{
      for (size_t col = 0; col < {cols}u; ++col) {{
        uint32_t bits; uint8_t code;
        __builtin_memcpy(&bits,source+({source_index}),4u);
        int status = merlin_f32_u8_encode_checked_table(bits,&{codebook},&code);
        if (status != MERLIN_F32_U8_OK) return status;
        if (pass) output[row*{cols}u+col] = code;
      }}
    }}
  }}
  return MERLIN_F32_U8_OK;
}}
"""


def _render_bf16_operand_c(conversion: BoundaryConversion, prefix: str, source_index: str) -> str:
    """Integer-only BF16 RNE with the same bounds as FloatBoundarySpec.

    Return zero on success, one for invalid/nonfinite input and two for an
    excluded code. This helper assumes valid disjoint buffers, as the byte
    encoder does; the generated application entry checks those ranges.
    """
    spec = conversion.spec
    assert spec is not None
    eb, mb, _bias, scheme = float_format_params(spec.format)
    if (eb, mb, scheme) != (8, 7, "ieee"):
        raise ValueError("C BF16 conversion requires its IEEE exponent layout")
    rows, cols = conversion.target_type.shape
    maximum = 254 if spec.max_normal_exponent_code is None else spec.max_normal_exponent_code
    max_bits = ((maximum << mb) | ((1 << mb) - 1)) << 16
    min_bits = (spec.min_normal_exponent_code << mb) << 16
    admitted = (
        " || ".join(
            f"(code >= {low}u && code <= {high}u)" if low else f"code <= {high}u"
            for low, high in spec.admitted_code_ranges
        )
        or "1"
    )
    flush = f"if (magnitude < 0x{min_bits:08x}u) magnitude=0;" if spec.flush_below_normal else ""
    canonical = "if (code==0x8000u) code=0;" if spec.canonicalize_negative_zero else ""
    return f"""#include <float.h>
#include <stddef.h>
#include <stdint.h>
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24 && FLT_MAX_EXP==128,
               "source conversion requires binary32 float");
_Static_assert(SIZE_MAX/4 >= {rows * cols}ULL, "source tensor exceeds host size_t");
static int {prefix}(const float *source, uint8_t *output) {{
  if (!source || !output) return 1;
  for (unsigned pass=0; pass<2; ++pass) {{
    for (size_t row=0; row<{rows}u; ++row) for (size_t col=0; col<{cols}u; ++col) {{
      uint32_t bits;
      __builtin_memcpy(&bits,source+({source_index}),4u);
      uint32_t magnitude=bits&0x7fffffffu;
      if (magnitude>=0x7f800000u) return 1;
      if (magnitude>0x{max_bits:08x}u) magnitude=0x{max_bits:08x}u;
      {flush}
      uint32_t code=(magnitude+0x7fffu+((magnitude>>16)&1u))>>16;
      code|=(bits>>16)&0x8000u;
      {canonical}
      if (!({admitted})) return 2;
      if (pass) {{
        size_t at=(row*{cols}u+col)*2u;
        output[at]=(uint8_t)code; output[at+1]=(uint8_t)(code>>8);
      }}
    }}
  }}
  return 0;
}}
"""
