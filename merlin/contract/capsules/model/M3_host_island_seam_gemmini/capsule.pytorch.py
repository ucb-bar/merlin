"""The host-island seam, on its own: accelerator -> host -> accelerator, and nothing else.

WHY THIS CAPSULE EXISTS. ``A->H->A`` -- an accelerator region, a host island, another accelerator
region -- is the expensive seam and the one a placement decision gets wrong: the intermediate has to
leave the accelerator, be computed somewhere the accelerator cannot compute it, and come back. The
composition axis (:mod:`merlin.targetgen.boundary`) reported that shape as covered, but covered
INCIDENTALLY: every capsule containing it is a whole model named for a different shape (all four
classify as ``routing``), so the shape was exercised by accident rather than proven. A capsule that is
ABOUT the seam is the difference between "some large capsule happens to contain one" and "the seam is
under test": when a routing capsule fails, nothing says whether the seam or one of its 155 other
regions was the cause.

WHAT IS ON EACH SIDE IS DERIVED, NOT CHOSEN. The two outer regions are int8 contractions because
``contraction/int8`` is the only family-and-dtype this target's capability manifest admits on its mesh.
The island is a LayerNorm because ``normalization`` is a family real captures contain 249 regions of
and this target declares no capability for at all -- the hardware has no reduction unit reachable
standalone, so a normalization has nowhere but the host lane to go. That makes the island honest: it is
not host work by an author's decision, it is host work because the accelerator cannot do it.

WHY IT IS EXACTLY THIS SHAPE AND NOT ONE REGION LONGER. ``classify_sequence`` names a sequence
``routing`` as soon as it has two accelerator RUNS and two host RUNS, so a single float region after
the last GEMM -- a dequantize, a bias, a reshape that lowers to a generic -- would retitle this capsule
``routing`` and put it right back in the incidental bucket beside the whole models. So the model
returns the second GEMM's int8 accumulator directly, and the input arrives already int8: there is no
host work before the first accelerator region and none after the second. Everything between them (the
widening cast, the LayerNorm, the requantize) is float and lands in ONE host run.

WHY THE ARITHMETIC LOOKS LIKE THIS. torch's int8 matmul accumulates in int8, so an int8 x int8 GEMM is
faithful to a mesh (which accumulates wide and requantizes on the readout) only while no partial sum
leaves int8 range. That is true here BY CONSTRUCTION rather than by luck, exactly as in the microvit
capsule beside this one: every weight is ternary (|w| <= 1) and every activation enters a GEMM bounded
by a saturating quantizer, so |accumulator| <= K * amp is a static property of the graph.

    GEMM         K   amp   bound        against int8's 127
    first       32    3      96
    second      32    3      96

The quantizer is ``tanh`` before the narrowing cast because a truncating quantizer scaled for the WORST
case annihilates the typical one (LayerNorm output is bounded by sqrt(C) but is typically far smaller),
and a capsule that grades a network of zeros grades nothing.

SIZED FOR THE CYCLE-ACCURATE TIER. Two 16x32 by 32x32 GEMMs and one 16x32 LayerNorm: every extent is a
multiple of the target's own tile edge, so the model is minimal for THIS geometry rather than minimal
for one hard-coded shape, and the whole thing is three tiles of arithmetic. Weights are random-init
from a fixed seed; the golden checks lowering exactness, not accuracy. The network is defined ENTIRELY
in this file -- no external model checkout, so a clean clone can rebuild it.
"""
from __future__ import annotations

import torch
from torch import nn

#: The target's tile edge. Every extent below is a multiple of it, so the model is minimal for THIS
#: geometry rather than minimal for one hard-coded shape.
TILE = 16

M = TILE              # 16 rows of activations
C = 2 * TILE          # 32 channels -- the contraction depth on both sides of the island
AMP = 3               # the quantizer's amplitude; |accumulator| <= C * AMP = 96 against int8's 127


def _ternary(g, *shape):
    """A ternary int8 weight. |w| <= 1 is what makes the accumulator bound above a static fact."""
    return torch.randint(0, 3, shape, generator=g, dtype=torch.int8) - 1


def _q(x, gain: float, amp: int = AMP):
    """Saturating quantizer: fp32 -> int8 with |q| <= amp, whatever x is. tanh is the bound."""
    return (torch.tanh(x * gain) * float(amp)).to(torch.int8)


class HostIslandSeam(nn.Module):
    """int8 GEMM -> LayerNorm on the host lane -> int8 GEMM. One seam, nothing else."""

    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(20260830)
        self.register_buffer("w_in", _ternary(g, C, C))
        self.ln = nn.LayerNorm(C)
        self.register_buffer("w_out", _ternary(g, C, C))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- accelerator region 1: int8 contraction on the mesh -------------------------------
        acc = torch.matmul(x, self.w_in)
        # --- host island: a normalization. This target's manifest declares NO capability for the
        # normalization family, so the widening cast, the LayerNorm and the requantize all fall to the
        # host/scalar lane -- one contiguous host run between the two accelerator runs.
        h = self.ln(acc.to(torch.float32))
        q = _q(h, 1.1)
        # --- accelerator region 2: the value comes BACK to the mesh. Returned as the int8 accumulator
        # itself: a dequantize here would be a second host run and would retitle the capsule `routing`.
        return torch.matmul(q, self.w_out)


def get_model_and_inputs():
    g = torch.Generator().manual_seed(11)
    x = torch.randint(-AMP, AMP + 1, (M, C), generator=g, dtype=torch.int8)
    return HostIslandSeam().eval(), (x,)
