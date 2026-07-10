"""XR0 loop-preserving capture wrapper (P21-S1).

XR0 (Xiaomi-Robotics-0) is a DiT / rectified-flow VLA: a Qwen3-VL backbone
encodes vision+language into a KV-cache, and a DiT head decodes an action chunk
via Euler integration of a rectified flow over ``num_steps`` (K) denoise steps.

Mirrors smolVLA / pi0.5's flow-matching pattern: the K-step Euler denoise loop is
wrapped in a single ``torch.while_loop(i < K, body, (i, z))`` so K is a captured
IR constant, the loop body is the structural repeated region, and the action
latent ``z`` is an explicit shape-invariant loop-carried iter_arg. The VLM
KV-cache, RoPE (cos, sin), attention mask, action_mask and projected state are
computed/closed over ONCE (invariant additional_inputs); the DiT reads them
read-only every step.

Faithful to ``XR0._flow_generate`` (xr0/mibot/models/VLA/XR0.py:546):
    dt = 1/num_steps ; z = x0.clone()
    for step in range(num_steps):
        t = ones(B,1,1) * step / num_steps
        v = dit_forward(z, t, **dit_kwargs)
        z = z + v * dt
The integer counter ``i`` replaces ``step`` (recomputed t = i/num_steps) per the
while_loop "hoist in-body constants / recompute from counter" rule.

KEY WRAPPER LESSON (this model): the ONLY in-body tensor constant is the
sinusoidal frequency basis built inside ``TimestepEmbedder.timestep_embedding``
(``torch.arange(half)`` -> ``freqs``, XR0.py:146). Materializing it in the loop
body triggers the ``aten.lift_fresh_copy`` / FunctionalTensor failure. We HOIST
``freqs`` out of the loop (it is timestep-independent) and recompute only the
data-dependent ``args = t * freqs`` + sin/cos inside the body. dit_forward's
``prefix_length`` is the Python constant 0, so the masked-assignment branch
(non-exportable in-place write) is excluded as designed (same as the loader).
"""

from __future__ import annotations

import math
import sys

import torch
from torch import nn

sys.path.insert(0, "/scratch/agustin/projects/model2MLIR/workloads/xr0")

import loader as xr0_loader  # noqa: E402
from torch._higher_order_ops.while_loop import while_loop  # noqa: E402

def _repo_root():
    from pathlib import Path as _P
    p = _P(__file__).resolve()
    while p != p.parent and not (p / "merlin" / "python").is_dir():
        p = p.parent
    return p
_ROOT = _repo_root()

K = 5  # num_steps (XR0 default; registry K=5)


class XR0DenoiseLoop(nn.Module):
    """Full K-step rectified-flow Euler denoise as a single torch.while_loop.

    Carried state = (i, z):
      i : int64 scalar step counter, shape ()                    [shape-invariant]
      z : action latent, shape (B, action_len, action_dim)       [shape-invariant; v same shape]

    Everything else (VLM KV-cache, RoPE cos/sin, attn_mask, action_mask, the
    projected state embedding, and the *hoisted* sinusoidal frequency basis) is
    computed ONCE before the loop and closed over. The body contains NO tensor
    constants -> functionalizes cleanly.
    """

    def __init__(self, model: nn.Module, num_steps: int = K) -> None:
        super().__init__()
        self.model = model
        self.num_steps = num_steps

    def forward(self, noisy_action, t, action_mask, state, cos, sin, attn_mask, *kv):
        m = self.model.model  # XR0DenoiseStep wraps the DiT head in `.model`
        K_local = self.num_steps
        dt = 1.0 / K_local
        bsize = state.shape[0]
        dev = noisy_action.device

        # ---------- conditioning computed ONCE, closed over ----------
        # Reassemble flat (k0,v0,k1,v1,...) into per-layer [(k,v),...].
        past_key_values = [(kv[2 * i], kv[2 * i + 1]) for i in range(len(kv) // 2)]
        state_embed = m.state_projector(state)
        position_embeds = (cos, sin)
        # DiT layer/KV alignment constant (max() hoisted OUT of the loop body).
        start_idx0 = max(0, len(past_key_values) - m.dit.layer_num)

        # ---------- hoist the in-body sinusoidal frequency basis ----------
        # TimestepEmbedder.timestep_embedding builds torch.arange(half) -> freqs
        # (XR0.py:146) inside every call; it is timestep-independent, so hoist it.
        te = m.t_embedder
        dim = te.frequency_embedding_size
        max_period = 10000
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=dev)
            / half
        )  # (half,), constant -> closed over

        def embed_time(t_scalar):
            """timestep_embedding + t_embedder MLP, body-safe (freqs hoisted)."""
            # t_scalar: (B,)  ; freqs: (half,)
            args = t_scalar[:, None].float() * freqs[None]              # (B, half)
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
            emb = emb.to(te.dtype)
            t_freq = emb
            t_emb = te.mlp(t_freq)
            return t_emb[:, None]                                       # (B, 1, D)

        def dit_step(z, t_bxx):
            """XR0.dit_forward (prefix_length==0 path) with hoisted time basis.

            Matches the loader's capture-faithful copy: the only deviation from
            upstream is t[:, 0, 0] -> reshape(B) (rank-reducing-select lowering
            bug), preserved here for the same reason.
            """
            t_scalar = t_bxx.reshape(t_bxx.shape[0]) * 1000               # (B,)
            t_embeds = embed_time(t_scalar)                               # (B, 1, D)
            t_embeds = m.t_projector(t_embeds).view(t_embeds.shape[0], 6, -1)

            na = z * action_mask
            na = m.action_projector(na)

            sink = m.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
            hidden_states = torch.cat([sink, state_embed, na], dim=1).contiguous()

            # Inline DiT.forward's layer loop. The upstream DiT.forward begins with
            #   start_idx = max(0, len(past_key_values) - self.layer_num)
            # The Python-level max() over a closed-over list of additional_input
            # tensors triggers export's `_tensor_min_max` -> a hard while_loop graph
            # break. start_idx is the Python constant `start_idx0` (computed once
            # outside the loop), so we index past_key_values directly per layer and
            # drive the layers explicitly -> no max() inside the traced body.
            for li, layer in enumerate(m.dit.layers):
                hidden_states = layer(
                    hidden_states,
                    past_key_values[start_idx0 + li],
                    position_embeds,
                    t_embeds,
                    attn_mask=attn_mask,
                )
            hidden_states = hidden_states[:, -na.shape[1]:, :]
            return m.action_output_layer(hidden_states)

        # ---------- carried state ----------
        i0 = torch.zeros((), dtype=torch.int64)
        z0 = noisy_action

        def cond_fn(i, z):
            return i < K_local

        def body_fn(i, z):
            # recompute t = i / num_steps from the integer counter
            t_val = i.to(torch.float32) * dt                             # scalar ()
            t_bxx = t_val.reshape(1, 1, 1).expand(bsize, 1, 1)           # (B,1,1)
            v = dit_step(z, t_bxx)
            z_next = z + v * dt
            return (i + 1, z_next)

        _, z_final = while_loop(cond_fn, body_fn, (i0, z0))
        return z_final


def build():
    step_mod, inputs = xr0_loader.get_model_and_inputs()
    return XR0DenoiseLoop(step_mod).eval(), inputs


def ref_unrolled(model, inputs, num_steps=K):
    """Reference: the same loop, unrolled eagerly via the REAL XR0._flow_generate
    semantics (calls the loader's capture-faithful dit_forward)."""
    step_mod = model
    m = step_mod.model
    noisy_action, t, action_mask, state, cos, sin, attn_mask, *kv = inputs
    past_key_values = [(kv[2 * i], kv[2 * i + 1]) for i in range(len(kv) // 2)]
    state_embed = m.state_projector(state)
    dt = 1.0 / num_steps
    z = noisy_action.clone()
    for step in range(num_steps):
        t_step = torch.ones((z.shape[0], 1, 1), device=z.device, dtype=z.dtype) * step / num_steps
        v = m.dit_forward(
            noisy_action=z,
            t=t_step,
            action_mask=action_mask,
            state_embed=state_embed,
            position_embeds=(cos, sin),
            past_key_values=past_key_values,
            attn_mask=attn_mask,
            prefix_length=0,
        )
        z = z + v * dt
    return z


if __name__ == "__main__":
    torch.manual_seed(0)
    wrapper, inputs = build()

    with torch.no_grad():
        out = wrapper(*inputs)
        ref = ref_unrolled(wrapper.model, inputs)
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    maxabs = (out - ref).abs().max().item()
    print("out shape:", tuple(out.shape), out.dtype)
    print(f"vs unrolled reference (real dit_forward): cos={cos:.7f} maxabs={maxabs:.3e}")

    print("\n=== torch.export.export(strict=False) ===")
    ep = torch.export.export(wrapper, inputs, strict=False)
    gm = ep.graph_module
    wl = [n for n in gm.graph.nodes
          if n.op == "call_function" and "while_loop" in str(n.target)]
    print("export OK; while_loop nodes:", len(wl),
          "| subgraphs:", [k for k in dir(gm) if "while_loop" in k])

    print("\n=== m2m.convert(linalg-on-tensors, fx_importer) ===")
    import os
    import m2m
    res = m2m.convert(wrapper, inputs, output_type="linalg-on-tensors",
                      quantization=None, level="linalg-on-tensors", backend="fx_importer")
    s = res.mlir_text or ""
    print("path_taken:", res.path_taken, "| MLIR length:", len(s))
    print("scf.for:", s.count("scf.for"), "| scf.yield:", s.count("scf.yield"))
    mod = getattr(res, "module", None)
    if mod is not None:
        mod.verify()
        print("module.verify() OK")
    out = f"{_ROOT}/merlin/benchmarks/dse_guidance/recaptures_loop/xr0/model.mlir"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if s:
        open(out, "w").write(s)
        print("WROTE", out, "bytes", len(s))
