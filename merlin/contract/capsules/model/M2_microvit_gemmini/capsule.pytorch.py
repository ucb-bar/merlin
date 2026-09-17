"""MicroViT — the smallest whole network that still exercises every family this target declares.

WHY THIS CAPSULE EXISTS. The corpus's whole-model capsules are the strongest thing it asserts (the
compiler takes a real network end to end) and the one thing no hardware ever checked: they are far too
large for the cycle-accurate tier, so the claim rests on a functional oracle whose descriptor says
``derived_from_rtl: false``. This model is sized to be affordable AT that tier instead: same shape as
the reference vision+recurrent control net, two orders of magnitude less data.

WHAT IT CONTAINS IS DERIVED, NOT CHOSEN. :mod:`merlin.targetgen.micro_model` composes the inventory from
three evidence sources — the accelerator families the target's capability manifest ADMITS, the families
real captures CONTAIN that it does not admit (which therefore MUST run on the host lane), and the op
spelling those captures actually use — and interleaves them so the host work sits BETWEEN accelerator
work. Every layer below discharges one row of that inventory:

    contraction / i8   accelerator   ``matmul``      the 12 int8 GEMMs (patch embed, q/kv, scores,
                                                     context, projection, MixFFN, decoder, LSTM gates,
                                                     command head)
    movement / i8      accelerator   ``transpose``   the K transpose feeding the attention scores, and
                                                     the accumulator fills every GEMM begins with
    elementwise_map/i8 accelerator   ``generic``     the int8 residual add on the projection output
                                                     (this target declares elementwise_map reachable
                                                     ONLY fused with a contraction, so it is authored
                                                     as an epilogue on a GEMM result, never standalone)
    normalization      HOST          ``generic``     the two LayerNorms and the attention softmax
    reduction          ACCEL/HOST    ``reduce``      the attention's 2x2 max pool consumes the bounded
                                                     int8 token grid and reaches the target's admitted
                                                     ``reduction/i8`` cell. The decoder's 4x4 pool
                                                     consumes the float host grid, preserving a real
                                                     host island in the model composition.

STRUCTURE (shrunk from vitfly's LSTMNetVIT, https://github.com/anish-bhattacharya/vitfly):
overlapping patch-merge convolution -> LayerNorm -> efficient self-attention with a pooled spatial
reduction -> int8 residual -> LayerNorm -> MixFFN (expand, depthwise conv, GELU, contract) -> residual
-> PixelShuffle up / max-pool down fusion -> linear decoder -> concat with the sensor vector -> an LSTM
cell step from a learned state -> linear command head. Every extent is a multiple of the target's own
tile edge (16) at the derived working extent (32); the reference's 60x90 image, 4608->512 decoder and
517->128x3 LSTM shrink to 16x16, 32->32 and 48->16.

WHY THE ARITHMETIC LOOKS LIKE THIS. torch's int8 matmul accumulates in int8, so an int8xint8 GEMM is
only faithful to a mesh (which accumulates wide and requantizes on the way out) while no partial sum
leaves int8 range. That is guaranteed here BY CONSTRUCTION rather than by luck: every weight is ternary
(|w| <= 1) and every activation enters a GEMM through a saturating quantizer whose output is bounded by
its amplitude, so |accumulator| <= K * amp is a static property of the graph. The largest bound in the
model is 120 (the context GEMM), against int8's 127:

    GEMM              K    amp   bound        GEMM              K    amp   bound
    patch embed       36    3     108         MixFFN expand     32    3      96
    query             32    3      96         MixFFN contract   48    2      96
    key/value         32    3      96         decoder           32    3      96
    scores  Q@K^T     32    3      96         LSTM  W_ih @ x    48    2      96
    context P@V        4  10x3    120         LSTM  W_hh @ h    16    3      48
    projection        32    3      96         command head      16    7     112

The quantizer is ``tanh`` before the narrowing cast because a truncating quantizer scaled for the WORST
case annihilates the typical one: LayerNorm output is bounded by sqrt(C)=5.66 but is typically 0.8, so a
worst-case-safe linear scale sends almost every activation to zero and the capsule grades a network of
zeros. Where the source is already bounded (a softmax probability, an LSTM output) the linear scale is
used instead, and the bound is stated above.

Weights are random-init from a fixed seed, as the reference capsule's are: the golden checks lowering
exactness, not control accuracy. The network is defined ENTIRELY in this file — no external model
checkout, no ``nn.LSTM`` (whose ``_flat_weights`` torch.export refuses), nothing to install.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

#: The target's tile edge. Every extent below is a multiple of it, so the model is minimal for THIS
#: geometry rather than minimal for one hard-coded shape.
TILE = 16

IMG = TILE            # 16x16 depth image (reference: 60x90)
C = 2 * TILE          # 32 embed channels, the derived working extent (reference: 32 -> 64)
NTOK = TILE           # 16 tokens after patch merge, a 4x4 grid
GRID = 4
HID = 3 * TILE        # 48 MixFFN hidden width (reference: 4x expansion)
HST = TILE            # 16 recurrent hidden units (reference: 128, three layers)
SENS = TILE           # 16 sensor scalars (reference: 1 desired velocity + 4 quaternion)
OUT = TILE            # 16 command outputs (reference: 3-DoF command)
PK, PS, PP = 6, 4, 1  # overlapping patch window -> 4x4 tokens, im2col depth 36


def _ternary(g, *shape):
    """A ternary int8 weight. |w| <= 1 is what makes every accumulator bound above a static fact."""
    return torch.randint(0, 3, shape, generator=g, dtype=torch.int8) - 1


def _q(x, gain: float, amp: int):
    """Saturating quantizer: fp32 -> int8 with |q| <= amp, whatever x is. tanh is the bound."""
    return (torch.tanh(x * gain) * float(amp)).to(torch.int8)


def _qi(acc, gain: float, amp: int):
    """Requantize a GEMM accumulator: the mesh's output-path rescale, |q| <= amp by the same argument."""
    return _q(acc.to(torch.float32), gain, amp)


def _lin(x, amp: int):
    """Linear quantizer, for a source whose range is already known (softmax probs, an LSTM output)."""
    return (x * float(amp)).to(torch.int8)


def _dq(acc, k: float):
    """Dequantize a GEMM accumulator back to the host lane's float domain."""
    return acc.to(torch.float32) * k


class MicroViT(nn.Module):
    """Vision + recurrent control net, shrunk to the target's tile edge."""

    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(20260830)
        # --- OverlapPatchMerging -------------------------------------------------------------
        self.register_buffer("w_patch", _ternary(g, PK * PK, C))
        self.ln1 = nn.LayerNorm(C)
        # --- EfficientSelfAttention (one kv projection over the reduced sequence, as upstream) --
        self.register_buffer("w_q", _ternary(g, C, C))
        self.register_buffer("w_kv", _ternary(g, C, 2 * C))
        self.register_buffer("w_proj", _ternary(g, C, C))
        self.ln2 = nn.LayerNorm(C)
        # --- MixFFN ---------------------------------------------------------------------------
        self.register_buffer("w_e", _ternary(g, C, HID))
        self.dw = nn.Conv2d(HID, HID, 3, padding=1, groups=HID)
        with torch.no_grad():
            self.dw.weight.copy_(torch.randn(self.dw.weight.shape, generator=g) * 0.5)
            self.dw.bias.copy_(torch.randn(self.dw.bias.shape, generator=g) * 0.1)
        self.register_buffer("w_c", _ternary(g, HID, C))
        # --- decoder + recurrent head -----------------------------------------------------------
        self.register_buffer("w_dec", _ternary(g, C, C))
        self.register_buffer("w_ih", _ternary(g, C + SENS, 4 * HST))
        self.register_buffer("w_hh", _ternary(g, HST, 4 * HST))
        self.register_buffer("b_g", torch.randn(4 * HST, generator=g) * 0.3)
        # The recurrent state the step starts from. Held quantized (h) and float (c) because that is
        # what the cell consumes; a NON-ZERO state is the point -- a step from zero multiplies the
        # recurrent weight by nothing and proves no recurrence at all.
        self.register_buffer("h0q", torch.randint(-3, 4, (1, HST), generator=g).to(torch.int8))
        self.register_buffer("c0", torch.randn(1, HST, generator=g) * 0.5)
        self.register_buffer("w_out", _ternary(g, HST, OUT))

    def forward(self, depth: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        # --- OverlapPatchMerging: im2col + int8 GEMM, the same lowering the capture pipeline gives a
        # strided convolution. Windows OVERLAP (6 wide, stride 4), as upstream's do.
        cols = F.unfold(depth, kernel_size=PK, stride=PS, padding=PP)          # (1, 36, 16)
        cols = cols.transpose(1, 2).reshape(NTOK, PK * PK)
        t = self.ln1(torch.matmul(_q(cols, 0.8, 3), self.w_patch).to(torch.float32))

        # --- EfficientSelfAttention. The spatial reduction is a 2x2 max pool over the bounded int8
        # token grid. This witnesses the target's admitted reduction/int8 cell; the later decoder pool
        # remains float host work, so the model still proves an accelerator -> host -> accelerator seam.
        qa = _q(t, 1.1, 3)
        grid = qa.reshape(1, GRID, GRID, C).permute(0, 3, 1, 2)                # (1, 32, 4, 4)
        red = grid.reshape(1, C, GRID // 2, 2, GRID // 2, 2).amax(dim=5).amax(dim=3)
        red = red.permute(0, 2, 3, 1).reshape(GRID, C)                         # (4, 32)
        ra = _q(red.to(torch.float32), 0.9, 3)
        qh = _qi(torch.matmul(qa, self.w_q), 0.15, 3)
        kv = _qi(torch.matmul(ra, self.w_kv), 0.15, 3)
        kh, vh = kv[:, :C], kv[:, C:]
        # kh.transpose is an int8 movement region on the accelerator, not a host copy.
        scores = _dq(torch.matmul(qh, kh.transpose(0, 1)), 1.0 / math.sqrt(float(C)))
        prob = torch.softmax(scores, dim=-1)                                   # host normalization
        ctx = _qi(torch.matmul(_lin(prob, 10), vh), 0.04, 3)
        proj = _qi(torch.matmul(ctx, self.w_proj), 0.15, 3)
        # int8 residual add: the elementwise_map/i8 cell, authored as an epilogue on a GEMM result
        # because this target declares that family reachable only fused with a contraction.
        t2 = self.ln2((qa + proj).to(torch.float32))

        # --- MixFFN: expand -> depthwise conv -> GELU -> contract -------------------------------
        hid = _dq(torch.matmul(_q(t2, 1.1, 3), self.w_e), 0.1)                 # (16, 48)
        hid = hid.reshape(1, GRID, GRID, HID).permute(0, 3, 1, 2)
        hid = F.gelu(self.dw(hid)).permute(0, 2, 3, 1).reshape(NTOK, HID)
        t3 = t2 + _dq(torch.matmul(_q(hid, 4.0, 2), self.w_c), 0.1)

        # --- PixelShuffle up / max-pool down fusion ---------------------------------------------
        gr = t3.reshape(1, GRID, GRID, C).permute(0, 3, 1, 2)                  # (1, 32, 4, 4)
        up = F.pixel_shuffle(gr, 2)                                            # (1, 8, 8, 8)
        dn = up.reshape(1, C // 4, 2, 4, 2, 4).amax(dim=5).amax(dim=3).reshape(1, C)
        dec = _dq(torch.matmul(_q(dn, 0.6, 3), self.w_dec), 0.1)               # (1, 32)

        # --- recurrent head: one LSTM cell step, written out as gates (no nn.LSTM) ---------------
        z = torch.cat([dec, sensor], dim=1)                                    # (1, 48)
        gates = ((torch.matmul(_q(z, 1.2, 2), self.w_ih).to(torch.float32)
                  + torch.matmul(self.h0q, self.w_hh).to(torch.float32)) * 0.12 + self.b_g)
        gi_, gf_, gg_, go_ = gates.chunk(4, dim=1)
        c1 = torch.sigmoid(gf_) * self.c0 + torch.sigmoid(gi_) * torch.tanh(gg_)
        h1 = torch.sigmoid(go_) * torch.tanh(c1)
        return _dq(torch.matmul(_lin(h1, 7), self.w_out), 0.12)


def get_model_and_inputs():
    g = torch.Generator().manual_seed(11)
    depth = torch.randn(1, 1, IMG, IMG, generator=g)
    sensor = torch.randn(1, SENS, generator=g) * 0.8
    return MicroViT().eval(), (depth, sensor)
