"""Render byte-only affine packing for a checked pointwise tile plan."""

from __future__ import annotations

from math import prod

from merlin.semantic_compiler.pointwise_tiling import PointwiseTiling


def render_pointwise_c(plan: PointwiseTiling, prefix: str) -> str:
    if (
        not isinstance(plan, PointwiseTiling)
        or not prefix
        or not prefix.isascii()
        or not prefix.isidentifier()
        or not prefix[0].isalpha()
    ):
        raise ValueError("pointwise C packing needs a checked plan and C identifier")
    m, n = plan.shape
    tm, tn = plan.tile_shape
    width = plan.element_bytes
    largest = max(plan.packed_bytes, *(prod(node.type.shape) * width for node in plan.source.nodes))
    if largest > 2**63 - 1:
        raise ValueError("pointwise buffers exceed the supported signed index domain")
    helpers = [
        "#include <stddef.h>\n#include <stdint.h>\n"
        f'_Static_assert(SIZE_MAX >= {largest}ULL, "pointwise sizes exceed host size_t");'
    ]
    for operand, node in enumerate(plan.source.nodes[:-1]):
        packed_m, packed_n = plan.input_shape(operand)
        pad = ",".join(str(byte) for byte in plan.input_pad(operand))
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
        helpers.append(f"""static void {prefix}_pack_{operand}(uint8_t *out,const uint8_t *source) {{
  const uint8_t pad[{width}]={{{pad}}}; size_t written=0;
  for (size_t mr=0;mr<{packed_m}u;mr+={tm}u) for (size_t nc=0;nc<{packed_n}u;nc+={tn}u)
    for (size_t r=0;r<{tm}u;++r) for (size_t c=0;c<{tn}u;++c) {{
      size_t row=mr+r,col=nc+c;
      if (row<{packed_m}u && col<{packed_n}u) __builtin_memcpy(out+written,source+({expression})*{width}u,{width}u);
      else __builtin_memcpy(out+written,pad,{width}u);
      written+={width}u;
    }}
}}
""")
    helpers.append(f"""static void {prefix}_assemble(uint8_t *out,const uint8_t *tiles) {{
  for (size_t row=0;row<{m}u;++row) for (size_t col=0;col<{n}u;++col) {{
    size_t at=(((row/{tm}u)*{plan.tile_counts[1]}u+col/{tn}u)*{tm * tn}u+(row%{tm}u)*{tn}u+col%{tn}u)*{width}u;
    __builtin_memcpy(out+(row*{n}u+col)*{width}u,tiles+at,{width}u);
  }}
}}
""")
    return "\n".join(helpers)
