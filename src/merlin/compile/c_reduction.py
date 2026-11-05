"""Render byte-only axis permutation and packing for seeded reductions."""

from __future__ import annotations

from math import prod

from merlin.semantic_compiler.reduction_tiling import ReductionTiling

from .c_pointwise import render_pointwise_c


def render_reduction_c(plan: ReductionTiling, prefix: str) -> str:
    if not isinstance(plan, ReductionTiling):
        raise ValueError("reduction C packing needs a checked reduction plan")
    seed = render_pointwise_c(plan.seed_plan(), prefix + "_seed")
    shape = plan.source.nodes[0].type.shape
    m, n = plan.matrix_shape
    tm, tn = plan.tile_shape
    width = plan.element_bytes
    if max(plan.packed_sizes) > 2**63 - 1:
        raise ValueError("reduction buffers exceed the supported signed index domain")
    coordinates = {}
    for axes, index in ((plan.kept_axes, "row"), (plan.axes, "col")):
        for i, axis in enumerate(axes):
            stride = prod(shape[a] for a in axes[i + 1 :])
            coordinates[axis] = f"(({index}/{stride}u)%{shape[axis]}u)"
    at = "0"
    for axis, extent in enumerate(shape):
        at = f"(({at})*{extent}u+{coordinates[axis]})"
    pad = ",".join(str(byte) for byte in plan.pad_element)
    return (
        seed
        + f"""
_Static_assert(SIZE_MAX >= {max(plan.packed_sizes)}ULL, "reduction sizes exceed host size_t");
static void {prefix}_pack_0(uint8_t *out,const uint8_t *source) {{
  const uint8_t pad[{width}]={{{pad}}}; size_t written=0;
  for (size_t mr=0;mr<{m}u;mr+={tm}u) for (size_t nc=0;nc<{n}u;nc+={tn}u)
    for (size_t r=0;r<{tm}u;++r) for (size_t c=0;c<{tn}u;++c) {{
      size_t row=mr+r,col=nc+c;
      if (row<{m}u && col<{n}u) __builtin_memcpy(out+written,source+({at})*{width}u,{width}u);
      else __builtin_memcpy(out+written,pad,{width}u);
      written+={width}u;
    }}
}}
static void {prefix}_pack_1(uint8_t *out,const uint8_t *source) {{
  {prefix}_seed_pack_0(out,source);
}}
static void {prefix}_assemble(uint8_t *out,const uint8_t *tiles) {{
  {prefix}_seed_assemble(out,tiles);
}}
"""
    )
