"""rdt2 loop-preserving capture wrapper (P21-S1).

RDT2 (thu-ml/RDT2) is an RDT-style DiT action expert that flow-matches an action
chunk by cross-attending to a frozen Qwen2.5-VL KV cache. Inference is a K-step
Euler ODE solve (``RDTRunner.conditional_sample``):

    timestep = 0.0 ; step_size = 1/K ; x = noisy_action
    for _ in range(K):
        action_traj  = act_adaptor(x)                 # re-embed (20 -> 1024)
        model_output = model(action_traj, timestep, **cond)   # velocity (.., 20)
        x            = model_output * step_size + x    # 1-order Euler
        timestep    += step_size
    return x

Same flow-matching family as smolVLA / pi0.5: carry ONLY the evolving action latent
``x`` (the un-adapted ``noisy_action`` (B, horizon, action_dim)) + the integer
counter; close over the VLM KV cache, the adapted state token, and all weights
(invariant additional_inputs). The body is the real per-step pipeline so the carried
latent round-trips shape-invariantly: ``act_adaptor`` re-embeds (1,24,20)->(1,24,1024),
the model returns (1,24,20), the Euler update keeps it (1,24,20).

HAZARD handled (diffusion): the only in-body tensor constant is the sinusoidal time
basis in ``TimestepEmbedder.timestep_embedding`` (a ``torch.arange`` -> ``torch.exp``
``freqs`` vector). It is hoisted out of the loop; the per-step embedding is recomputed
from the closed-over ``freqs`` and the timestep derived from the integer counter
(``timestep = i * step_size``), per the "recompute from counter / hoist constants" rule.

The loader (model2MLIR/workloads/rdt2/loader.py) captures only ``RDT.forward`` and
its input ``x`` is ALREADY adapted (1,24,1024). The faithful loop carries the
un-adapted (1,24,20) latent and re-adapts each step, so this wrapper builds the
mlp3x_silu ``act_adaptor`` (configs/rdt/post_train.yaml) itself (random init -- this
is a structural / random-init capture, not a checkpointed run).
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, "/scratch/agustin/projects/RDT2")
sys.path.insert(0, "/scratch/agustin/projects/model2MLIR/workloads/rdt2")

import torch
import torch.nn as nn
import torch.nn.functional as F

import loader as rdt2_loader

K = 5  # num_inference_timesteps (configs/rdt/post_train.yaml)


def _build_act_adaptor(action_dim: int, hidden_size: int) -> nn.Module:
    """mlp3x_silu adaptor, mirroring RDTRunner.build_condition_adapter."""
    return nn.Sequential(
        nn.Linear(action_dim, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
    )


def _rdt_forward_with_temb(model, x, t_emb, lang_c_kv, state_c):
    """RDT.forward (models/rdt/model.py, from line 162) but taking the ALREADY
    time-embedded ``t_emb`` (output of ``t_embedder``) instead of the raw timestep,
    so the in-body sinusoidal ``arange`` constant is hoisted out. No img branch,
    no lang_mask (matches the loader's deployment path: lang enters via KV cache)."""
    t = t_emb
    if t.shape[0] == 1:
        t = t.expand(x.shape[0], -1)

    state_c = state_c + model.state_pos_emb
    t = torch.cat([t.unsqueeze(1), state_c], dim=1).reshape(x.shape[0], model.hidden_size * 2)

    r = model.register_tokens.expand(x.shape[0], -1, -1)
    x = torch.cat([x, r], dim=1)
    x = x + model.x_pos_emb

    # conds = [lang_c_kv]; single condition (num_conds == 1, no img branch)
    for i, block in enumerate(model.blocks):
        ck, cv = lang_c_kv[i]
        ck = ck.transpose(1, 2)  # (bs, n_kv_heads, seq_len, head_size)
        cv = cv.transpose(1, 2)
        x = block(x, t, None, ck, cv, mask=None)
    x = model.final_layer(x, t)
    x = x[:, : -model.num_register_tokens]
    return x


class RDT2WhileLoopSampler(nn.Module):
    """act_adaptor + a torch.while_loop over the K Euler denoise steps."""

    def __init__(self, rdt_model: nn.Module, act_adaptor: nn.Module, num_steps: int = K) -> None:
        super().__init__()
        self.model = rdt_model
        self.act_adaptor = act_adaptor
        self.num_steps = num_steps

    def forward(self, noisy_action, state_c, *kv):
        m = self.model
        # Rebuild the per-block (k, v) list (one pair per RDT block) -- invariant.
        lang_c_kv = [(kv[2 * i], kv[2 * i + 1]) for i in range(len(kv) // 2)]

        K_local = self.num_steps
        step_size = 1.0 / K_local

        # --- hoist the ONLY in-body tensor constant: the sinusoidal time basis ---
        # (TimestepEmbedder.timestep_embedding: arange -> exp -> freqs)
        half = m.t_embedder.frequency_embedding_size // 2  # 128
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )  # (128,), constant -> closed over

        def cond_fn(i, x):
            return i < K_local

        def body_fn(i, x):
            # recompute the timestep from the integer counter (no in-body float carry)
            timestep = i.to(torch.float32) * step_size            # scalar tensor ()
            t1 = timestep.reshape(1)                               # (1,)
            # sinusoidal time embedding from the closed-over freqs (no in-body arange)
            args = t1[:, None] * freqs[None]                      # (1, 128)
            t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (1, 256)
            t_emb = m.t_embedder.mlp(t_freq.to(m.t_embedder.dtype))         # (1, D)

            action_traj = self.act_adaptor(x)                    # (1,24,20)->(1,24,1024)
            model_output = _rdt_forward_with_temb(m, action_traj, t_emb, lang_c_kv, state_c)
            x_next = model_output * step_size + x                 # 1-order Euler
            return i + 1, x_next

        i0 = torch.zeros((), dtype=torch.long)
        _, x_final = torch.while_loop(cond_fn, body_fn, (i0, noisy_action))
        return x_final


def build():
    """Return (wrapper, inputs). inputs = (noisy_action (1,24,20), state_c (1,1,1024), *kv)."""
    step_mod, loader_inputs = rdt2_loader.get_model_and_inputs()
    rdt_model = step_mod.model
    x_adapted, t, state_c, *kv = loader_inputs

    action_dim = 20  # loader output_size
    hidden_size = rdt_model.hidden_size
    act_adaptor = _build_act_adaptor(action_dim, hidden_size).eval()

    # The carried latent is the UN-adapted noisy action (1, horizon, action_dim).
    b, horizon = x_adapted.shape[0], x_adapted.shape[1]
    noisy_action = torch.randn(b, horizon, action_dim, dtype=x_adapted.dtype)

    inputs = (noisy_action, state_c, *kv)
    return RDT2WhileLoopSampler(rdt_model, act_adaptor).eval(), inputs


def ref_unrolled(wrapper, inputs, num_steps=K):
    """Reference: the same Euler loop, unrolled eagerly (to check the wrapper matches)."""
    m = wrapper.model
    noisy_action, state_c, *kv = inputs
    lang_c_kv = [(kv[2 * i], kv[2 * i + 1]) for i in range(len(kv) // 2)]
    step_size = 1.0 / num_steps
    x = noisy_action
    timestep = torch.tensor([0.0], dtype=noisy_action.dtype)
    for _ in range(num_steps):
        action_traj = wrapper.act_adaptor(x)
        model_output = m(
            x=action_traj, t=timestep, lang_c=None, lang_c_kv=lang_c_kv,
            img_c=None, state_c=state_c, lang_mask=None, img_mask=None,
        )
        x = model_output * step_size + x
        timestep = timestep + step_size
    return x


if __name__ == "__main__":
    wrapper, inputs = build()
    with torch.no_grad():
        out = wrapper(*inputs)
        ref = ref_unrolled(wrapper, inputs)
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    print("out shape:", tuple(out.shape), "cos vs unrolled:", float(cos))
