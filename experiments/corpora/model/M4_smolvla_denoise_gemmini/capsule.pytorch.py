"""SmolVLA denoise step — one flow-matching denoising step of a vision-language-action policy,
at the smallest extents that still carry the op surface the real capture contains.

WHY THIS CAPSULE EXISTS. ``smolvla`` is named in this repo's target descriptors as a perf-bench
workload and was captured five times (``out/artifacts/recaptures/smolvla_*``), but no capsule ever
graded it: the corpus's whole-model capstones are a decoder LLM (``small_llama``), a vision+recurrent
control net (``lstmnetvit`` / ``microvit``) and a classifier (``resnet50``). None of them is a VLA
policy, so the compiler's reach over the *action-expert* shape -- a frozen vision-language prefix
that a separate denoising expert cross-attends into -- was asserted by nothing.

WHY IT IS THIS SMALL. The real denoise-step capture is 1.2 GB of weights over 6335 linalg regions with
a 512x512 image; a whole-model capsule on this scale is not schedulable at the cycle-accurate tier at
all (a sibling model capsule on another target was blocked outright by a 125 MB generated ``main.c`` of
element-wise tensor initialisers). So this capsule keeps the *shape* and drops the *scale*: same six
inputs in the same roles, same action-chunk width, two orders of magnitude less data.

WHAT IT CONTAINS IS DERIVED FROM THE CAPTURE, NOT CHOSEN. The op inventory below is the ``prov.op`` /
``prov.family`` census of ``out/artifacts/recaptures/smolvla_denoise_step_fp32_app/model.mlir`` -- the
real thing, exported by the same model2MLIR that exports this one. Every family that census reports
with a non-trivial count has a layer here that carries it:

    census (real capture)              this model
    contraction   507   matmul  494    patch-embed conv, q/k/v/o of two attention stacks, the
                                       cross-attention into the prefix, both FFNs, the action head
    attention     228   sdpa    228    prefix self-attention, expert self-attention, expert
                                       cross-attention (Q from the action tokens, K/V from the prefix).
                                       MEASURED CAVEAT: the prefix attention is authored fused
                                       (F.scaled_dot_product_attention, what the real policy calls) and
                                       m2m DECOMPOSES it on export, so this capsule's interface carries
                                       0 `sdpa` regions and 33 `softmax` ones. The `attention` family is
                                       therefore absent from the emitted linalg even though the source
                                       calls the fused op -- a property of the exporter, not of the
                                       model, and it is stated here rather than left to be discovered
    normalization 927   layer_norm 575 the four RMSNorms (pow / reduce_mean / rsqrt / mul), and the
                        softmax  352   three softmaxes inside the attentions
    elementwise  1506   mul     569    SwiGLU (silu) in the prefix FFN, GELU in the expert FFN, the
                        silu     33    residual adds, the timestep-conditioned scale
                        gelu     12
    cast          406   dtype_cast     the two boolean masks (img_mask, lang_masks) entering the
                                       additive attention mask
    concat         96   cat       96   the prefix is a concatenation of vision + language + state
                                       tokens -- the defining structure of a VLA prefix
    iota           63   arange    62   RoPE position indices and the flow-matching timestep embedding
                        sin/cos  57/57
    layout       1482   view/unsqueeze the head split/merge every attention does
    reduce        471   reduce_mean    the RMSNorm means and the softmax denominators
    embedding       2   embedding      the language token embedding

Families the census reports that are deliberately NOT here: ``arg_reduce`` / ``search`` /
``gather_scatter`` / ``scan`` (56 + 8 + 18 + 12 regions) belong to the real capture's tokenizer-side
bucketize/cumsum mask plumbing, which is host-only bookkeeping around the network rather than part of
the denoising computation, and ``slice_scatter`` (112) is the real capture's KV-cache write-back, which
a single step with no cache does not have.

EXTENTS ARE A MULTIPLE OF THE TARGET'S OWN TILE EDGE, so this capsule is minimal *for this target*
rather than minimal for one geometry. At edge 16: the model width is two edges (32), each attention
head is exactly one edge (16), the fused prefix is exactly one edge of tokens (4 vision + 11 language
+ 1 state = 16), and the action chunk is one edge (16). The action dimension (32) and the state
dimension (32) are the REAL capture's, unchanged -- they are the policy's interface to its robot, not
a size knob. The real capture's 512x512x3 image, 48 language tokens and 50-step action chunk shrink to
32x32x3, 11 and 16.

INTERFACE. Positionally identical to the real capture's ``input_order``
(``{img: 0, img_mask: 1, lang_tokens: 2, lang_masks: 3, state: 4, noise: 5}``), so a submission
compiled against this capsule is compiled against the real policy's calling convention:

    img          f32 [1, 3, 32, 32]   the observation frame
    img_mask     bool[1]              whether the frame is present (a real VLA runs with cameras absent)
    lang_tokens  i64 [1, 11]          the instruction
    lang_masks   bool[1, 11]          which instruction tokens are real
    state        f32 [1, 32]          proprioceptive state
    noise        f32 [1, 16, 32]      the noised action chunk this step denoises
    ->           f32 [1, 16, 32]      the denoised action chunk

The flow-matching TIMESTEP is a graph constant, not an input -- exactly as in the real capture, whose
exported ABI also has six inputs and no timestep tensor (the step is traced at a fixed t).
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

# Every extent below is a multiple of this target's tile edge; see the module docstring.
TILE = 16
DM = 2 * TILE            # model width (32)
HEADS = 2
HD = DM // HEADS         # head dim == one tile edge (16)
IMG = 2 * TILE           # 32x32 frame
PATCH = TILE             # 16x16 patches -> a 2x2 vision-token grid
NIMG = (IMG // PATCH) ** 2   # 4 vision tokens
NLANG = 11               # so the fused prefix is exactly one tile edge of tokens
NPRE = NIMG + NLANG + 1  # 16 prefix tokens (vision + language + state)
VOCAB = 256
STATE = 2 * TILE         # 32 -- the real capture's state width
ACT = 2 * TILE           # 32 -- the real capture's action width
CHUNK = TILE             # 16 action steps (real: 50)
FFN = 4 * TILE           # 64
TIMESTEP = 0.5           # the traced denoising time (a graph constant, as in the real capture)


class RMSNorm(nn.Module):
    """The census's dominant normalization, written out (pow / reduce_mean / rsqrt / mul)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.w


def _rope(x, pos):
    """Rotary position embedding over [B, H, T, HD] -- the capture's arange/sin/cos/mul regions."""
    half = x.shape[-1] // 2
    freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos[:, None].to(torch.float32) * freq[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None, None]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def _heads(x):
    b, t, _ = x.shape
    return x.view(b, t, HEADS, HD).transpose(1, 2)


def _merge(x):
    b, h, t, d = x.shape
    return x.transpose(1, 2).reshape(b, t, h * d)


class Attention(nn.Module):
    """One attention stack. ``cross`` takes K/V from a separate context (the prefix)."""

    def __init__(self, d_ctx: int = DM):
        super().__init__()
        self.q = nn.Linear(DM, DM, bias=False)
        self.k = nn.Linear(d_ctx, DM, bias=False)
        self.v = nn.Linear(d_ctx, DM, bias=False)
        self.o = nn.Linear(DM, DM, bias=False)

    def forward(self, x, ctx=None, *, rope_pos=None, add_mask=None, fused=False):
        c = x if ctx is None else ctx
        q, k, v = _heads(self.q(x)), _heads(self.k(c)), _heads(self.v(c))
        if rope_pos is not None:
            q = _rope(q, rope_pos)
            k = _rope(k, rope_pos)
        if fused:
            # The real policy's frozen VLM calls the fused op, so the prefix attention does too. It does
            # NOT survive to the interface: m2m decomposes sdpa on export (measured -- 0 `sdpa` regions
            # in the emitted linalg), so this is fidelity to the SOURCE, not a fused region the compiler
            # gets to see.
            ctx_out = F.scaled_dot_product_attention(q, k, v, attn_mask=add_mask)
        else:
            att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(HD))
            if add_mask is not None:
                att = att + add_mask
            ctx_out = torch.matmul(torch.softmax(att, dim=-1), v)
        return self.o(_merge(ctx_out))


class SwiGLU(nn.Module):
    """The prefix FFN -- the capture's ``silu`` regions."""

    def __init__(self):
        super().__init__()
        self.g = nn.Linear(DM, FFN, bias=False)
        self.u = nn.Linear(DM, FFN, bias=False)
        self.d = nn.Linear(FFN, DM, bias=False)

    def forward(self, x):
        return self.d(F.silu(self.g(x)) * self.u(x))


class GeluFFN(nn.Module):
    """The expert FFN -- the capture's ``gelu`` regions."""

    def __init__(self):
        super().__init__()
        self.up = nn.Linear(DM, FFN, bias=False)
        self.dn = nn.Linear(FFN, DM, bias=False)

    def forward(self, x):
        return self.dn(F.gelu(self.up(x)))


class SmolVlaDenoiseStep(nn.Module):
    """Vision-language prefix -> action expert cross-attending into it -> one denoised chunk."""

    def __init__(self):
        super().__init__()
        # --- vision-language prefix ---------------------------------------------------------
        self.patch = nn.Conv2d(3, DM, kernel_size=PATCH, stride=PATCH, bias=False)
        self.tok = nn.Embedding(VOCAB, DM)
        self.state_in = nn.Linear(STATE, DM, bias=False)
        self.n_pre1 = RMSNorm(DM)
        self.pre_attn = Attention()
        self.n_pre2 = RMSNorm(DM)
        self.pre_ffn = SwiGLU()
        # --- action expert ------------------------------------------------------------------
        self.act_in = nn.Linear(ACT, DM, bias=False)
        self.t_in = nn.Linear(DM, DM, bias=False)
        self.n_a1 = RMSNorm(DM)
        self.a_self = Attention()
        self.n_a2 = RMSNorm(DM)
        self.a_cross = Attention()
        self.n_a3 = RMSNorm(DM)
        self.a_ffn = GeluFFN()
        self.head = nn.Linear(DM, ACT, bias=False)

    def _timestep(self):
        """Sinusoidal flow-matching timestep embedding (arange / sin / cos), then a projection."""
        half = DM // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / half)
        a = TIMESTEP * f
        return self.t_in(torch.cat([a.sin(), a.cos()], -1)[None])       # [1, DM]

    def forward(self, img, img_mask, lang_tokens, lang_masks, state, noise):
        # --- prefix: vision + language + state tokens, concatenated -------------------------
        vis = self.patch(img).flatten(2).transpose(1, 2)                # [1, NIMG, DM]
        vm = img_mask.to(torch.float32)[:, None, None]                  # cast: bool -> f32
        vis = vis * vm
        lang = self.tok(lang_tokens)                                    # [1, NLANG, DM]
        st = self.state_in(state)[:, None, :]                           # [1, 1, DM]
        prefix = torch.cat([vis, lang, st], dim=1)                      # [1, NPRE, DM]

        # key-padding mask over the prefix, built from the two boolean inputs
        keep = torch.cat([img_mask.to(torch.bool)[:, None].expand(-1, NIMG),
                          lang_masks.to(torch.bool),
                          torch.ones_like(img_mask, dtype=torch.bool)[:, None]], dim=1)
        pad = (1.0 - keep.to(torch.float32)) * (-1.0e4)                 # [1, NPRE]
        pre_mask = pad[:, None, None, :]                                # broadcast over heads/queries

        pos = torch.arange(NPRE)
        prefix = prefix + self.pre_attn(self.n_pre1(prefix), rope_pos=pos, add_mask=pre_mask,
                                        fused=True)
        prefix = prefix + self.pre_ffn(self.n_pre2(prefix))

        # --- action expert: denoise the chunk, cross-attending into the prefix ---------------
        a = self.act_in(noise) + self._timestep()[:, None, :]           # [1, CHUNK, DM]
        causal = torch.full((CHUNK, CHUNK), float(-1.0e4)).triu(1)[None, None]
        a = a + self.a_self(self.n_a1(a), rope_pos=torch.arange(CHUNK), add_mask=causal)
        a = a + self.a_cross(self.n_a2(a), ctx=prefix, add_mask=pre_mask)
        a = a + self.a_ffn(self.n_a3(a))
        return self.head(a)                                             # [1, CHUNK, ACT]


def get_model_and_inputs():
    """Deterministic inputs (the capture worker seeds torch before calling this).

    The inputs are a SEEDED SYNTHETIC stream, not a robot episode: this capsule grades whether the
    compiled program reproduces the reference the same loader produced, and says nothing about the
    policy's success rate. ``img_mask`` is True and one language token is masked off, so both boolean
    inputs are exercised with a value that changes the answer.
    """
    g = torch.Generator().manual_seed(50)
    img = torch.randn(1, 3, IMG, IMG, generator=g) * 0.5
    img_mask = torch.ones(1, dtype=torch.bool)
    lang_tokens = torch.randint(0, VOCAB, (1, NLANG), generator=g, dtype=torch.long)
    lang_masks = torch.ones(1, NLANG, dtype=torch.bool)
    lang_masks[0, NLANG - 1] = False
    state = torch.randn(1, STATE, generator=g) * 0.6
    noise = torch.randn(1, CHUNK, ACT, generator=g)
    return SmolVlaDenoiseStep().eval(), (img, img_mask, lang_tokens, lang_masks, state, noise)
