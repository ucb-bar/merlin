"""pi0.5 loop-preserving capture wrapper (P21-S1).

Mirrors smolVLA's flow-matching pattern: the K-step Euler denoise loop is wrapped
in a single ``torch.while_loop(i < K, body, (i, x_t))`` so K is a captured IR
constant, the loop body is the structural repeated region, and the action latent
``x_t`` is an explicit loop-carried iter_arg. The SigLIP+Gemma prefix pass + its
KV cache are computed ONCE before the loop and closed over (invariant
additional_inputs) -- the expert reads them read-only every step (use_cache=False,
so the cache never grows: it is invariant, not carried, unlike an AR decode).

Faithful to openpi's ``PI0Pytorch.sample_actions``:
    dt = -1/num_steps ; x_t = noise ; time = 1
    while time >= -dt/2:  v = denoise_step(state, prefix, kv, x_t, time.expand(b))
                          x_t = x_t + dt*v ; time += dt
The integer counter ``i`` replaces the float ``time`` (recomputed time = 1 + i*dt)
per the while_loop "hoist in-body constants / recompute from counter" rule.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/agustin/projects/model2MLIR/workloads/pi05")

import math

import torch
import torch.nn.functional as F
from torch import nn

import loader as pi05_loader
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

K = 10  # num_steps (matches sample_actions default)


def suffix_embs_pi05(m, x_t, timestep, scaling_factor):
    """The x_t/timestep-dependent part of ``embed_suffix`` (pi05 path):
    action embedding + adaRMS time MLP. The constant masks and the sinusoidal
    BASIS (``scaling_factor``, a torch.linspace constant that must be hoisted out
    of the loop) are computed once outside and passed in. Returns
    (suffix_embs, adarms_cond)."""
    sin_input = scaling_factor[None, :] * timestep[:, None]
    time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    time_emb = time_emb.type(dtype=timestep.dtype)
    action_emb = m.action_in_proj(x_t)            # (b, ah, width)
    x = m.time_mlp_in(time_emb)
    x = F.silu(x)
    x = m.time_mlp_out(x)
    adarms_cond = F.silu(x)
    return action_emb, adarms_cond


def denoise_core(m, past_key_values, suffix_embs, full_att_2d_masks_4d, position_ids, adarms_cond):
    """``denoise_step`` MINUS embed_suffix and the attention-mask/position
    computation (both x_t-independent -> hoisted outside the loop; the mask uses
    a torch.where(-2.38e38) scalar literal that would otherwise become an in-body
    get_attr constant) and MINUS the in-body config mutation (set once outside)."""
    outputs_embeds, _ = m.paligemma_with_expert.forward(
        attention_mask=full_att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    suffix_out = outputs_embeds[1]
    suffix_out = suffix_out[:, -m.config.action_horizon:]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return m.action_out_proj(suffix_out)


class Pi05WhileLoopSampler(nn.Module):
    """Prefix once + a torch.while_loop over the K Euler denoise steps."""

    def __init__(self, model, num_steps: int = K) -> None:
        super().__init__()
        self.model = model
        self.num_steps = num_steps

    def forward(self, i0, i1, i2, m0, m1, m2, lang_tokens, lang_masks, state, noise):
        m = self.model
        images, img_masks = [i0, i1, i2], [m0, m1, m2]

        # --- prefix (SigLIP + Gemma) pass + KV cache: computed ONCE ---
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = m._prepare_attention_masks_4d(prefix_att_2d)
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = m.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # set attn impl ONCE, outside the loop body (denoise_core omits it)
        m.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        bsize = state.shape[0]
        dt = -1.0 / self.num_steps
        K_local = self.num_steps
        ah = m.config.action_horizon
        dev = noise.device

        # --- hoist ALL in-body constants out of the loop ---
        # sinusoidal time-embedding basis (a torch.linspace constant)
        dim = m.action_in_proj.out_features
        fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64, device=dev)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        scaling_factor = (1.0 / period * 2 * math.pi)            # (dim/2,), constant
        # constant suffix masks (shape-only; pi05 suffix = action tokens)
        suffix_pad_masks = torch.ones(bsize, ah, dtype=torch.bool, device=dev)
        att_row = torch.cat([torch.ones(1, dtype=noise.dtype, device=dev),
                             torch.zeros(ah - 1, dtype=noise.dtype, device=dev)])
        suffix_att_masks = att_row[None, :].expand(bsize, ah)
        # full attention mask (4d) + position_ids: x_t-independent -> hoist out
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, ah, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        full_att_2d_masks_4d = m._prepare_attention_masks_4d(full_att_2d)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        def cond_fn(i, x_t):
            return i < K_local

        def body_fn(i, x_t):
            # recompute time from the integer counter (no in-body float carry)
            time = 1.0 + i.to(torch.float32) * dt
            expanded_time = time.reshape(1).expand(bsize)
            s_embs, adarms_cond = suffix_embs_pi05(m, x_t, expanded_time, scaling_factor)
            v_t = denoise_core(
                m, past_key_values, s_embs, full_att_2d_masks_4d, position_ids, adarms_cond,
            )
            x_next = x_t + dt * v_t
            return i + 1, x_next

        i0_ = torch.zeros((), dtype=torch.long)
        _, x_final = torch.while_loop(cond_fn, body_fn, (i0_, noise))
        return x_final


def build():
    step_mod, inputs = pi05_loader.get_model_and_inputs()
    model = step_mod.model
    return Pi05WhileLoopSampler(model).eval(), inputs


def ref_unrolled(model, inputs, num_steps=K):
    """Reference: the same loop, unrolled eagerly (to check the wrapper matches)."""
    m = model
    i0, i1, i2, m0, m1, m2, lang_tokens, lang_masks, state, noise = inputs
    images, img_masks = [i0, i1, i2], [m0, m1, m2]
    prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_4d = m._prepare_attention_masks_4d(prefix_att_2d)
    m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, pkv = m.paligemma_with_expert.forward(
        attention_mask=prefix_att_4d, position_ids=prefix_pos, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=True)
    bsize = state.shape[0]
    dt = -1.0 / num_steps
    x_t = noise
    for i in range(num_steps):
        time = 1.0 + i * dt
        et = torch.tensor(time, dtype=torch.float32).reshape(1).expand(bsize)
        v_t = m.denoise_step(state, prefix_pad_masks, pkv, x_t, et)
        x_t = x_t + dt * v_t
    return x_t


if __name__ == "__main__":
    wrapper, inputs = build()
    with torch.no_grad():
        out = wrapper(*inputs)
        ref = ref_unrolled(wrapper.model, inputs)
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    print("out shape:", tuple(out.shape), "cos vs unrolled:", float(cos))
