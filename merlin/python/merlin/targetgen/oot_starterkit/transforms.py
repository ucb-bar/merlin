"""Generic, target-AGNOSTIC compiler transforms the agent calls. NOT target-specific lowerings.

These are textbook transforms every matmul-capable accelerator backend needs — the same math whether the
target is Gemmini, a TPU, or a toy NPU. Per the Q2 ruling (agent-callable + generalizable ⇒ legitimate),
they live in the shared kit, not as a Gemmini answer. They reduce a problem to a 2D matmul / tile it; the
agent still maps the resulting matmul to ITS target's instructions (the target-specific work).

  * im2col(...)     — conv (NHWC + KHWC weights) -> (im2col matrix, packed weight) shapes + a recipe so a
                      conv becomes a 2D matmul. Pure shape/layout algebra; no target opcodes.
  * tile_to_dim(...) — split an MxK x KxN matmul into DIMxDIM tiles (the standard systolic tiling).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Im2colPlan:
    """Shapes + recipe to turn a conv into a 2D matmul (im2col matrix @ packed weights)."""
    im2col_shape: tuple[int, int]          # [out_h*out_w, kh*kw*cin]
    weight_2d_shape: tuple[int, int]       # [kh*kw*cin, cout]
    out_shape: tuple[int, int, int, int]   # [n, out_h, out_w, cout]
    recipe: dict[str, Any] = field(default_factory=dict)


def im2col(ifm_nhwc: tuple[int, int, int, int], weight_khwc: tuple[int, int, int, int],
           stride=(1, 1), padding=(0, 0, 0, 0), dilation=(1, 1)) -> Im2colPlan:
    """Generic conv->matmul reduction (NHWC input, weight [kh,kw,cin,cout]). Returns the matmul shapes +
    a recipe (the same recipe schema the contract's `params.im2col_recipes` expects). No target specifics.
    """
    n, h, w, cin = ifm_nhwc
    kh, kw, wcin, cout = weight_khwc
    if wcin != cin:
        raise ValueError(f"channel mismatch: ifm cin={cin} vs weight cin={wcin}")
    sh, sw = stride
    pt, pb, pl, pr = padding
    dh, dw = dilation
    out_h = (h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    out_w = (w + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    k = kh * kw * cin
    return Im2colPlan(
        im2col_shape=(out_h * out_w, k),
        weight_2d_shape=(k, cout),
        out_shape=(n, out_h, out_w, cout),
        recipe={"kh": kh, "kw": kw, "ci": cin, "stride": list(stride), "padding": list(padding),
                "dilation": list(dilation), "layout": "nhwc"})


@dataclass
class Tile:
    m0: int; n0: int; k0: int; m1: int; n1: int; k1: int   # tile bounds [m0:m1, n0:n1, k0:k1]


def tile_to_dim(m: int, n: int, k: int, dim: int) -> list[Tile]:
    """Standard systolic tiling of an (MxK)·(KxN) matmul into DIMxDIM·DIMxDIM tiles (row-major, k-inner).
    Generic — the agent maps each tile to its target's load/preload/compute/store."""
    tiles: list[Tile] = []
    for mo in range(0, m, dim):
        for no in range(0, n, dim):
            for ko in range(0, k, dim):
                tiles.append(Tile(mo, no, ko, min(mo + dim, m), min(no + dim, n), min(ko + dim, k)))
    return tiles
