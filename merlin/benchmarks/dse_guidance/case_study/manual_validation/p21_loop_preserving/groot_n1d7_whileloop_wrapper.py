"""GR00T N1.5/N1.7 loop-preserving capture wrapper (P21).

NVIDIA Isaac-GR00T's diffusion action head (``Gr00tN1d7ActionHead``) is a
flow-matching DiT. ``get_action_with_features`` runs a K-step Euler denoise loop
(``gr00t_n1d7.py`` line 383):

    dt = 1/N ; actions = noise
    for t in range(N):                       # N = num_inference_timesteps = 4
        t_disc = int((t/N) * num_timestep_buckets)
        ts = full((B,), t_disc)
        af  = action_encoder(actions, ts, embodiment_id) (+ pos_embed)
        sa  = cat([state_features, af], dim=1)
        out = model(sa, vl_embeds, timestep=ts, image_mask, backbone_attn_mask)
        v   = action_decoder(out, embodiment_id)[:, -ah:]
        actions = actions + dt * v * vel_strength

This mirrors smolVLA/pi0.5 flow-matching: wrap the K steps in a single
``torch.while_loop(i < K, body, (i, actions))`` so K is a captured IR constant,
the body is the structural repeated region, and the action latent ``actions`` is
the shape-invariant loop-carried iter_arg ([B, action_horizon, action_dim]).

HOISTED / closed-over invariants (computed ONCE before the loop):
  - vl_embeds = vlln(backbone_features) (a real LayerNorm; vl_self_attention=Identity)
  - state_features = state_encoder(state.view(B,1,-1), embodiment_id)
  - the action position-embedding table ``pos_embs`` (built from torch.arange — an
    in-body get_attr constant if left inside; hoisted per the while_loop rule)
  - vel_strength = ones_like(actions) (non-RTC path; invariant)

RECOMPUTED FROM THE COUNTER (no in-body float carry, no precomputed table index):
  - t_disc(i) = (i * num_timestep_buckets) // N     -> timesteps_tensor = full((B,), t_disc)

The ``image_mask & backbone_attention_mask`` masks the DiT builds internally are
derived from closed-over invariant inputs (not in-body constants) so they are fine.
The sinusoidal basis inside action_encoder/timestep_encoder is built from the
carried ``timesteps`` (data-dependent on i); m2m's _lower_while_loop imports the
body recursively so these recompute per-step inside the scf.for region.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/agustin/projects/model2MLIR/workloads/groot_n1d7")

import torch
from torch import nn

from loader import get_model_and_inputs  # noqa: E402
from torch._higher_order_ops.while_loop import while_loop  # noqa: E402


K = 4  # num_inference_timesteps (registry / config constant)


class Gr00tDenoiseLoop(nn.Module):
    """Full K-step flow-matching denoise as a single torch.while_loop.

    Carried state = (i, actions):
      i       : int64 scalar step counter, shape ()                 [shape-invariant]
      actions : action latent [B, action_horizon, action_dim]       [shape-invariant; v same shape]
    """

    def __init__(self, head) -> None:
        super().__init__()
        self.head = head
        self.K = int(head.num_inference_timesteps)            # 4
        self.num_buckets = int(head.num_timestep_buckets)     # 1000
        self.dt = 1.0 / self.K

    def forward(
        self,
        backbone_features,
        state,
        actions,
        embodiment_id,
        backbone_attention_mask,
        image_mask,
        timesteps,  # unused (recomputed from counter); kept for loader-input ABI parity
    ):
        h = self.head
        K_local = self.K
        dt = self.dt
        num_buckets = self.num_buckets
        bsize = backbone_features.shape[0]
        dev = backbone_features.device

        # ---------- conditioning: computed ONCE, closed over ----------
        vl_embeds = h.vlln(backbone_features)
        vl_embeds = h.vl_self_attention(vl_embeds)

        state_r = state.view(state.shape[0], 1, -1)
        state_features = h.state_encoder(state_r, embodiment_id)

        # non-RTC velocity strength is just ones (invariant) — close over it
        vel_strength = torch.ones_like(actions)

        # ---------- hoist the action position-embedding table out of the body ----------
        add_pos = bool(h.config.add_pos_embed)
        if add_pos:
            pos_ids = torch.arange(h.action_horizon, dtype=torch.long, device=dev)
            pos_embs = h.position_embedding(pos_ids).unsqueeze(0)   # (1, ah, w)

        # ---------- hoist the action_encoder sinusoidal FREQUENCY BASIS out of the body ----------
        # SinusoidalPositionalEncoding.forward builds, every call,
        #   exponent = -arange(half_dim) * (log(tensor(10000.)) / half_dim)
        # the ``torch.log(torch.tensor(10000.0))`` literal becomes a get_attr tensor
        # constant inside the while_loop HOP subgraph (SpecViolationError). It depends
        # ONLY on half_dim (a python int), not on the carried timesteps, so compute the
        # basis ``exp(exponent)`` ONCE here and have the body multiply the per-step
        # timesteps by this closed-over invariant (no in-body constant).
        import math as _math
        pe = h.action_encoder.pos_encoding
        half_dim = pe.embedding_dim // 2
        freq_basis = torch.exp(
            -torch.arange(half_dim, dtype=torch.float, device=dev)
            * (_math.log(10000.0) / half_dim)
        )  # (half_dim,) constant, created OUTSIDE the loop

        def _pos_encoding_hoisted(timesteps_bt):
            ts = timesteps_bt.float()
            freqs = ts.unsqueeze(-1) * freq_basis  # (B, T, half_dim)
            return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)

        # ---------- carried state ----------
        i0 = torch.zeros((), dtype=torch.int64)
        x0 = actions

        def cond(i, x_t):
            return i < K_local

        def body(i, x_t):
            # recompute the discretized timestep bucket from the integer counter
            #   t_disc = int((i/N) * num_buckets) == (i * num_buckets) // N
            t_disc = (i * num_buckets) // K_local                  # int64 scalar ()
            timesteps_tensor = t_disc.reshape(1).expand(bsize)     # (B,) int64

            # ---- inlined MultiEmbodimentActionEncoder.forward with hoisted basis ----
            ae = h.action_encoder
            B, T, _ = x_t.shape
            ts_bt = timesteps_tensor.unsqueeze(1).expand(-1, T)    # (B, T)
            a_emb = ae.W1(x_t, embodiment_id)
            tau_emb = _pos_encoding_hoisted(ts_bt).to(dtype=a_emb.dtype)
            ae_x = torch.cat([a_emb, tau_emb], dim=-1)
            ae_x = torch.nn.functional.silu(ae.W2(ae_x, embodiment_id))
            action_features = ae.W3(ae_x, embodiment_id)
            if add_pos:
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1)

            if h.config.use_alternate_vl_dit:
                model_output = h.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=image_mask,
                    backbone_attention_mask=backbone_attention_mask,
                )
            else:
                model_output = h.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = h.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -h.action_horizon:]

            x_next = x_t + dt * pred_velocity * vel_strength
            return (i + 1, x_next)

        i_out, x_out = while_loop(cond, body, (i0, x0))
        return x_out


def build():
    head_step, inputs = get_model_and_inputs()
    return Gr00tDenoiseLoop(head_step.head).eval(), inputs


def ref_unrolled(head, inputs, num_steps=K):
    """Reference: the same K-step loop, unrolled eagerly (no while_loop)."""
    (backbone_features, state, actions, embodiment_id,
     backbone_attention_mask, image_mask, timesteps) = inputs
    h = head
    dt = 1.0 / num_steps
    num_buckets = int(h.num_timestep_buckets)
    bsize = backbone_features.shape[0]
    dev = backbone_features.device

    vl_embeds = h.vl_self_attention(h.vlln(backbone_features))
    state_features = h.state_encoder(state.view(state.shape[0], 1, -1), embodiment_id)
    vel_strength = torch.ones_like(actions)
    add_pos = bool(h.config.add_pos_embed)
    if add_pos:
        pos_ids = torch.arange(h.action_horizon, dtype=torch.long, device=dev)
        pos_embs = h.position_embedding(pos_ids).unsqueeze(0)

    x_t = actions
    for t in range(num_steps):
        t_disc = int((t / float(num_steps)) * num_buckets)
        ts = torch.full((bsize,), t_disc, dtype=torch.long, device=dev)
        af = h.action_encoder(x_t, ts, embodiment_id)
        if add_pos:
            af = af + pos_embs
        sa = torch.cat((state_features, af), dim=1)
        out = h.model(
            hidden_states=sa, encoder_hidden_states=vl_embeds, timestep=ts,
            image_mask=image_mask, backbone_attention_mask=backbone_attention_mask,
        )
        pred = h.action_decoder(out, embodiment_id)
        x_t = x_t + dt * pred[:, -h.action_horizon:] * vel_strength
    return x_t


if __name__ == "__main__":
    torch.manual_seed(0)
    wrapper, inputs = build()
    with torch.no_grad():
        out_loop = wrapper(*inputs)
        ref = ref_unrolled(wrapper.head, inputs)
    print("eager while_loop OK, out shape:", tuple(out_loop.shape), out_loop.dtype)
    cos = torch.nn.functional.cosine_similarity(out_loop.flatten(), ref.flatten(), dim=0).item()
    maxabs = (out_loop - ref).abs().max().item()
    print(f"vs unrolled reference: cos={cos:.7f} maxabs={maxabs:.3e}")
