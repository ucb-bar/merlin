"""Scratch: torch.while_loop wrapper for smolVLA flow-matching denoise loop.

Goal: capture the K-step Euler denoise loop as a single torch.while_loop (NOT unrolled),
with shape-invariant carried state, so torch.export + m2m.convert yield a loop-preserving
MLIR instead of K unrolled denoise_step copies.

Source facts (lerobot modeling_smolvla.py):
  - sample_actions loop: line 837  `for step in range(num_steps)`
  - num_steps = 10 (configuration_smolvla.py:66), dt = -1/num_steps (line 834)
  - per-step: time = 1.0 + step*dt; v_t = denoise_step(x_t, t); x_t = x_t + dt*v_t (line 865)
  - denoise_step signature (line 872): (prefix_pad_masks, past_key_values, x_t, timestep)
  - v_t shape == x_t shape == (b, chunk_size=50, max_action_dim=32)  -> shape-invariant
  - conditioning (prefix_pad_masks, past_key_values) computed once, closed over (not carried)

KEY BLOCKER FOUND (torch 2.10): functionalization of a torch.while_loop BODY fails with
  "Attempting to use FunctionalTensor on its own"
whenever the body materializes a *tensor constant* (-> aten.lift_fresh_copy). smolVLA's body
hits this via embed_suffix line 759 `torch.tensor([1]*chunk_size)` and via
create_sinusoidal_pos_embedding line 91 `torch.linspace(...)`. FIX: hoist every in-body
tensor-constant OUT of the loop and close over it. Because the time schedule
time_k = 1 + k*dt is a Python-known constant for k in range(K), we precompute the full
(K, expert_hidden) sinusoidal time-embedding table outside the loop and index it by the
carried step counter. The suffix attention/pad masks + position ids are data-independent
(depend only on shapes) so they too are precomputed once and closed over.
"""

from __future__ import annotations
import os
import sys
import torch
from torch import nn

sys.path.insert(0, "/path/to/model2MLIR/workloads/smolvla")
from loader import get_model_and_inputs  # noqa: E402

from lerobot.policies.smolvla.modeling_smolvla import (  # noqa: E402
    make_att_2d_masks,
    create_sinusoidal_pos_embedding,
)
from torch._higher_order_ops.while_loop import while_loop  # noqa: E402


class SmolVLADenoiseLoop(nn.Module):
    """Full K-step flow-matching denoise as a single torch.while_loop.

    Carried state = (i, x_t):
      i   : int64 scalar step counter, shape ()            [shape-invariant]
      x_t : action latent, shape (b, chunk_size, action)   [shape-invariant; v_t same shape]

    Everything else (prefix KV cache, all masks/position ids, and the *precomputed*
    per-step time-embedding table) is computed ONCE before the loop and closed over.
    The body therefore contains NO tensor constants -> functionalizes cleanly.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.K = int(model.config.num_steps)            # Python int constant = 10
        self.dt = -1.0 / self.K

    def forward(self, img, img_mask, lang_tokens, lang_masks, state, noise):
        m = self.model
        cfg = m.config
        bsize = state.shape[0]
        K = self.K
        dt = self.dt

        # ---------- prefix / conditioning: computed ONCE, closed over ----------
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            [img], [img_mask], lang_tokens, lang_masks, state=state
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = m.vlm_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=cfg.use_cache,
            fill_kv_cache=True,
        )

        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = cfg.chunk_size  # embed_suffix emits exactly chunk_size action tokens

        # ---------- suffix masks / positions: data-independent, computed ONCE ----------
        # mirrors embed_suffix (line 752-760) + denoise_step (line 882-891), const-free
        suffix_pad_masks = torch.ones(bsize, suffix_len, dtype=torch.bool, device=state.device)
        suffix_att_1d = torch.ones(bsize, suffix_len, dtype=prefix_embs.dtype, device=state.device)
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_1d)
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # ---------- sinusoidal frequency vector: const-free in body ----------
        # create_sinusoidal_pos_embedding builds linspace (a constant) in-body; hoist it.
        # time = 1.0 + i*dt is computed from the carried counter (no data-dependent index).
        dim = m.vlm_with_expert.expert_hidden_size
        import math
        fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64, device=state.device)
        period = cfg.min_period * (cfg.max_period / cfg.min_period) ** fraction
        scaling_factor = (1.0 / period * 2 * math.pi).to(dtype=torch.float32)     # (dim//2,)

        # ---------- carried state ----------
        i0 = torch.zeros((), dtype=torch.int64)
        x0 = noise

        def cond(i, x_t):
            return i < K

        def body(i, x_t):
            # ----- embed_suffix, const-free (uses closed-over scaling_factor & masks) -----
            action_emb = m.action_in_proj(x_t)                                   # (b, S, hidden)
            # time = 1.0 + i*dt  (float scalar from carried counter; no data-dependent index)
            time = 1.0 + i.to(torch.float32) * dt                                # scalar ()
            sin_input = scaling_factor * time                                    # (dim//2,)
            time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=0)  # (dim,)
            time_emb = time_emb.to(dtype=action_emb.dtype)
            time_emb = time_emb[None, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)
            action_time_emb = m.action_time_mlp_in(action_time_emb)
            action_time_emb = torch.nn.functional.silu(action_time_emb)
            suffix_embs = m.action_time_mlp_out(action_time_emb)

            # ----- expert pass against the cached prefix (denoise_step lines 893-904) -----
            outputs_embeds, _ = m.vlm_with_expert.forward(
                attention_mask=full_att_2d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=cfg.use_cache,
                fill_kv_cache=False,
            )
            suffix_out = outputs_embeds[1][:, -cfg.chunk_size:].to(dtype=torch.float32)
            v_t = m.action_out_proj(suffix_out)                                  # same shape as x_t

            x_next = x_t + dt * v_t
            return (i + 1, x_next)

        i_out, x_out = while_loop(cond, body, (i0, x0))
        return x_out


def build():
    base, inputs = get_model_and_inputs()
    return SmolVLADenoiseLoop(base.model).eval(), inputs


if __name__ == "__main__":
    torch.manual_seed(0)
    wrapper, inputs = build()

    with torch.no_grad():
        out_loop = wrapper(*inputs)
    print("eager while_loop forward OK, out shape:", tuple(out_loop.shape), out_loop.dtype)

    # reference: replicate sample_actions' unrolled loop (calls the REAL denoise_step)
    m = wrapper.model
    img, img_mask, lang_tokens, lang_masks, state, noise = inputs
    with torch.no_grad():
        pe, ppm, pam = m.embed_prefix([img], [img_mask], lang_tokens, lang_masks, state=state)
        a2d = make_att_2d_masks(ppm, pam); pos = torch.cumsum(ppm, dim=1) - 1
        _, pkv = m.vlm_with_expert.forward(
            attention_mask=a2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[pe, None], use_cache=m.config.use_cache, fill_kv_cache=True)
        x_t = noise
        for step in range(wrapper.K):
            tt = torch.tensor(1.0 + step * wrapper.dt, dtype=torch.float32).expand(state.shape[0])
            v_t = m.denoise_step(ppm, pkv, x_t, tt)
            x_t = x_t + wrapper.dt * v_t
        ref = x_t
    cos = torch.nn.functional.cosine_similarity(out_loop.flatten(), ref.flatten(), dim=0).item()
    maxabs = (out_loop - ref).abs().max().item()
    print(f"vs unrolled reference (real denoise_step): cos={cos:.7f} maxabs={maxabs:.3e}")

    print("\n=== torch.export.export(strict=False) ===")
    ep = torch.export.export(wrapper, inputs, strict=False)
    gm = ep.graph_module
    wl_nodes = [n for n in gm.graph.nodes
                if n.op == "call_function" and "while_loop" in str(n.target)]
    print("export OK; while_loop call_function nodes:", len(wl_nodes),
          "| subgraphs:", [k for k in dir(gm) if "while_loop" in k])

    print("\n=== run_decompositions (functionalize) ===")
    try:
        ep2 = ep.run_decompositions({})
        nwl = sum(1 for n in ep2.graph_module.graph.nodes
                  if n.op == "call_function" and "while_loop" in str(n.target))
        print("run_decompositions OK; while_loop nodes after:", nwl)
    except Exception as e:
        print("run_decompositions FAIL:", type(e).__name__, str(e).splitlines()[0])

    # ---- raw torch-dialect import proves loop preservation (before the failing pass) ----
    print("\n=== torch-mlir raw import (loop-preservation proof) ===")
    from torch_mlir import fx as tmfx
    mraw = tmfx.export_and_import(ep, output_type="raw", func_name="forward")
    raw = mraw.operation.get_asm(large_elements_limit=8, large_resource_limit=8,
                                 enable_debug_info=False)
    print("torch.prim.Loop ops (1 op => +1 .condition):", raw.count("torch.prim.Loop"))

    import m2m
    out_dir = "/tmp/claude-2621/-scratch-agustin-projects-merlin/7d66f954-67c6-438f-8e72-ebd11bf2d229/scratchpad"
    for ot, be in (("torch", "auto"), ("linalg-on-tensors", "auto"),
                   ("linalg-on-tensors", "fx_importer")):
        print(f"\n=== m2m.convert(output_type={ot!r}, backend={be!r}) ===")
        res = m2m.convert(wrapper, inputs, output_type=ot, backend=be)
        s = res.mlir_text or ""
        print("path_taken:", res.path_taken, "| frontend:", getattr(res, "frontend", "?"),
              "| MLIR length:", len(s))
        print("  scf.while:", s.count("scf.while"), "| scf.for:", s.count("scf.for"),
              "| torch.prim.Loop:", s.count("torch.prim.Loop"),
              "| @while_loop refs:", s.count("while_loop"))
        if res.diagnostics:
            print("  diag[-1]:", str(res.diagnostics[-1]).splitlines()[0][:220])
        if s:
            p = f"{out_dir}/smolvla_wl_{ot.replace('-', '_')}_{be}.mlir"
            open(p, "w").write(s); print("  wrote", p)
