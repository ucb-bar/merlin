"""Render byte-only affine packing for a checked pointwise tile plan."""

from __future__ import annotations

from math import prod

from merlin.compiler.transforms.pointwise import PointwiseTiling


def render_pointwise_c(plan: PointwiseTiling, prefix: str) -> str:
    if (
        not isinstance(plan, PointwiseTiling)
        or not prefix
        or not prefix.isascii()
        or not prefix.isidentifier()
        or not prefix[0].isalpha()
    ):
        raise ValueError("pointwise C packing needs a checked plan and C identifier")
    width = plan.element_bytes
    largest = max(plan.packed_bytes, *(prod(node.type.shape) * width for node in plan.source.nodes))
    if largest > 2**63 - 1:
        raise ValueError("pointwise buffers exceed the supported signed index domain")
    helpers = [
        "#include <stddef.h>\n#include <stdint.h>\n"
        f'_Static_assert(SIZE_MAX >= {largest}ULL, "pointwise sizes exceed host size_t");'
    ]
    helpers += [render_operand_pack(plan, prefix, operand) for operand in range(len(plan.source.nodes) - 1)]
    helpers.append(render_assemble(plan, prefix, width))
    return "\n".join(helpers)


def _contiguous_rows(plan: PointwiseTiling, operand: int):
    """(base, row_step) when every in-range packed element (row, col) reads source element base + row_step*row + col.

    Derived from the operand's affine map: the C index is the sum of each source dimension times its stride. With a
    two-dimensional logical shape it is affine in row and col directly. Otherwise each logical coordinate is digit i of
    L = row*n + col, which stays below the logical element count because the packed shape never exceeds the matrix
    shape, and the index is affine in L when every coordinate's weight equals the last weight times its digit place;
    an extent-1 coordinate is always 0, so its weight never matters. With one packed row the row weight never matters
    either. The last packed element must lie in the source, so every emitted constant is below the source element
    count. None when a row is not one run of the source.
    """
    node = plan.source.nodes[operand]
    mapping = plan.input_map(operand)
    logical, n = plan.logical_shape, plan.shape[1]
    packed_m, packed_n = plan.input_shape(operand)
    strides = [prod(node.type.shape[d + 1 :]) for d in range(len(node.type.shape))]
    base = sum(offset * stride for offset, stride in zip(mapping.offsets, strides, strict=True))
    weights = [sum(row[i] * stride for row, stride in zip(mapping.coefficients, strides, strict=True))
               for i in range(len(logical))]
    if len(logical) == 2:
        row_step, col_step = weights
    elif all(extent == 1 or weight == weights[-1] * prod(logical[i + 1 :])
             for i, (weight, extent) in enumerate(zip(weights, logical, strict=True))):
        row_step, col_step = weights[-1] * n, weights[-1]
    else:
        return None
    if packed_m == 1:
        row_step = 0
    inside = 0 <= base and 0 <= row_step and base + row_step * (packed_m - 1) + packed_n - 1 < prod(node.type.shape)
    return (base, row_step) if col_step == 1 and inside else None


def render_operand_pack(plan: PointwiseTiling, prefix: str, operand: int) -> str:
    """Pack one operand into row-major tiles, padding out-of-range lanes with the operand's pad element."""
    node = plan.source.nodes[operand]
    n = plan.shape[1]
    tm, tn = plan.tile_shape
    width = plan.element_bytes
    packed_m, packed_n = plan.input_shape(operand)
    pad = ",".join(str(byte) for byte in plan.input_pad(operand))
    # The original index expression and its integer-domain checks come first, whichever loop is emitted.
    mapping = plan.input_map(operand)
    expression = "0"
    for coefficients, offset, extent in zip(mapping.coefficients, mapping.offsets, node.type.shape, strict=True):
        if max(abs(offset), *(abs(c) for c in coefficients)) > 2**63 - 1 or (
            abs(offset) + sum(abs(c) * (n - 1) for c, n in zip(coefficients, plan.logical_shape, strict=True))
            > 2**63 - 1
        ):
            raise ValueError("affine map exceeds the supported signed C index domain")
        if len(plan.logical_shape) == 2:
            dimension = (
                f"({offset}LL + ({coefficients[0]}LL)*(long long)row + ({coefficients[1]}LL)*(long long)col)"
            )
        else:
            coordinates = (
                f"(((row*{n}u+col)/{prod(plan.logical_shape[i + 1 :])}u)%{extent}u)"
                for i, extent in enumerate(plan.logical_shape)
            )
            terms = " + ".join(
                f"({c}LL)*(long long){index}" for c, index in zip(coefficients, coordinates, strict=True)
            )
            dimension = f"({offset}LL + {terms})"
        expression = f"(({expression})*{extent}u + (size_t){dimension})"
    rows = _contiguous_rows(plan, operand)
    if rows is not None:
        base, row_step = rows
        return f"""static void {prefix}_pack_{operand}(uint8_t *out,const uint8_t *source) {{
  const uint8_t pad[{width}]={{{pad}}}; size_t written=0;
  for (size_t mr=0;mr<{packed_m}u;mr+={tm}u) for (size_t nc=0;nc<{packed_n}u;nc+={tn}u)
    for (size_t r=0;r<{tm}u;++r) {{
      size_t row=mr+r,c=0;
      /* The affine map makes each packed row's in-range elements one source run. */
      if (row<{packed_m}u) {{ c={packed_n}u-nc<{tn}u?{packed_n}u-nc:{tn}u; __builtin_memcpy(out+written,source+({base}u+{row_step}u*row+nc)*{width}u,c*{width}u); }}
      for (;c<{tn}u;++c) __builtin_memcpy(out+written+c*{width}u,pad,{width}u);
      written+={tn * width}u;
    }}
}}
"""
    return f"""static void {prefix}_pack_{operand}(uint8_t *out,const uint8_t *source) {{
  const uint8_t pad[{width}]={{{pad}}}; size_t written=0;
  for (size_t mr=0;mr<{packed_m}u;mr+={tm}u) for (size_t nc=0;nc<{packed_n}u;nc+={tn}u)
    for (size_t r=0;r<{tm}u;++r) for (size_t c=0;c<{tn}u;++c) {{
      size_t row=mr+r,col=nc+c;
      if (row<{packed_m}u && col<{packed_n}u) __builtin_memcpy(out+written,source+({expression})*{width}u,{width}u);
      else __builtin_memcpy(out+written,pad,{width}u);
      written+={width}u;
    }}
}}
"""


def render_assemble(plan: PointwiseTiling, prefix: str, width: int) -> str:
    """Copy result tiles back to row-major order, one run of up to a tile width per tile row."""
    m, n = plan.shape
    tm, tn = plan.tile_shape
    return f"""static void {prefix}_assemble(uint8_t *out,const uint8_t *tiles) {{
  for (size_t row=0;row<{m}u;++row) for (size_t col=0;col<{n}u;col+={tn}u) {{
    size_t at=(((row/{tm}u)*{plan.tile_counts[1]}u+col/{tn}u)*{tm * tn}u+(row%{tm}u)*{tn}u)*{width}u;
    __builtin_memcpy(out+(row*{n}u+col)*{width}u,tiles+at,({n}u-col<{tn}u?{n}u-col:{tn}u)*{width}u);
  }}
}}
"""
