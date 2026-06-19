"""Scratch: torch.while_loop wrapper for openVLA's K=7 autoregressive action-token
decode, with a FIXED-SIZE (static) KV cache as shape-invariant carried state.

Carried state (all plain tensors, shape-invariant across the K steps):
    i        : scalar int64  -- decode step counter (0..K)
    cur_tok  : (B,1) int64   -- token fed at this step
    out_toks : (B,K) int64   -- collected action tokens (written at slot i)
    k_cache  : (L, B, Hkv, S, Dh) f32   -- key cache, written in-place at position pos+i
    v_cache  : (L, B, Hkv, S, Dh) f32   -- value cache
where S = prompt_len + K (static).  The prompt KV is pre-filled into the first
prompt_len positions before the loop; the body writes one new K/V per layer at
position (prompt_len + i) each step -> shape never changes -> exportable.

The body is a *manual* Llama decode step (RMSNorm -> QKV -> RoPE -> static-cache
write -> full-causal attention over the static window -> MLP -> final norm ->
lm_head -> argmax).  We avoid the HF Cache object because torch.while_loop iter_args
must be plain tensors.
"""
from __future__ import annotations

import os

os.environ.setdefault("M2M_OPENVLA_LLM_LAYERS", "2")
os.environ.setdefault("M2M_OPENVLA_VIT_LAYERS", "2")
os.environ.setdefault("M2M_OPENVLA_VOCAB", "512")

import torch
from torch import nn

import sys
sys.path.insert(0, "/scratch/agustin/projects/model2MLIR/workloads/openvla")
from loader import get_model_and_inputs  # noqa: E402

from transformers.models.llama.modeling_llama import (  # noqa: E402
    apply_rotary_pos_emb,
)

K = 7  # openVLA action dimension (predict_action -> max_new_tokens=7)


def rms_norm(x, weight, eps):
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return weight * x


class StaticKVDecodeWrapper(nn.Module):
    """forward(prompt_embeds) -> (out_toks, k_cache, v_cache).

    Runs the prefill once (eagerly, inside forward, traced) to fill the static
    KV cache for the prompt positions and produce the first decode token, then
    runs K manual decode steps as a single torch.while_loop.
    """

    def __init__(self, vla):
        super().__init__()
        self.lm = vla.language_model
        m = self.lm.model
        self.layers = m.layers
        self.final_norm = m.norm
        self.embed = m.embed_tokens
        self.rotary = m.rotary_emb
        self.lm_head = self.lm.lm_head
        cfg = vla.config.text_config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        self.eps = cfg.rms_norm_eps
        self.hidden = cfg.hidden_size
        self.K = K

    # --- one transformer pass over a (B, T, H) hidden, writing KV at [pos:pos+T] ---
    def _layers_pass(self, hidden, k_cache, v_cache, pos, seqlen_filled, position_ids):
        # position_embeddings for the queries at this pass
        cos, sin = self.rotary(hidden, position_ids)
        B, T, _ = hidden.shape
        S = k_cache.shape[3]
        new_k = []
        new_v = []
        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            residual = hidden
            hs = rms_norm(hidden, layer.input_layernorm.weight, self.eps)
            q = attn.q_proj(hs).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(hs).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            v = attn.v_proj(hs).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            # write the new k/v into the static cache at [pos : pos+T]
            kc = k_cache[li].index_copy(2, pos + torch.arange(T, device=hidden.device), k)
            vc = v_cache[li].index_copy(2, pos + torch.arange(T, device=hidden.device), v)
            new_k.append(kc)
            new_v.append(vc)
            # attention over the full static window, masked to "filled" positions
            attn_w = torch.matmul(q, kc.transpose(2, 3)) / (self.head_dim ** 0.5)
            # mask: key position j is valid if j < seqlen_filled+T and (causal) j <= pos+t
            key_pos = torch.arange(S, device=hidden.device)
            qpos = (pos + torch.arange(T, device=hidden.device)).unsqueeze(1)  # (T,1)
            causal = key_pos.unsqueeze(0) <= qpos                              # (T,S)
            mask = causal.unsqueeze(0).unsqueeze(0)                            # (1,1,T,S)
            attn_w = attn_w.masked_fill(~mask, float("-inf"))
            attn_w = torch.softmax(attn_w, dim=-1)
            out = torch.matmul(attn_w, vc)                                    # (B,Hkv,T,Dh)
            out = out.transpose(1, 2).reshape(B, T, -1)
            out = attn.o_proj(out)
            hidden = residual + out
            # MLP
            residual = hidden
            hs = rms_norm(hidden, layer.post_attention_layernorm.weight, self.eps)
            mlp = layer.mlp
            hs = mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(hs)) * mlp.up_proj(hs))
            hidden = residual + hs
        k_cache = torch.stack(new_k, 0)
        v_cache = torch.stack(new_v, 0)
        return hidden, k_cache, v_cache

    def forward(self, prompt_embeds):
        # prompt_embeds: (B, P, H)  -- the multimodal prefix embeddings.
        B, P, H = prompt_embeds.shape
        device = prompt_embeds.device
        S = P + self.K  # static cache length
        k_cache = torch.zeros(self.n_layers, B, self.n_kv, S, self.head_dim, device=device)
        v_cache = torch.zeros(self.n_layers, B, self.n_kv, S, self.head_dim, device=device)

        # --- prefill: run the prompt through all layers, fill cache[0:P] ---
        pos0 = torch.zeros((), dtype=torch.long, device=device)
        ppos = torch.arange(P, device=device).unsqueeze(0)
        hidden, k_cache, v_cache = self._layers_pass(
            prompt_embeds, k_cache, v_cache, pos0, P, ppos
        )
        last = self.final_norm(hidden[:, -1:, :])
        logits = self.lm_head(last)            # (B,1,V)
        first_tok = logits[:, -1, :].argmax(-1, keepdim=True)  # (B,1)

        # --- carried state ---
        i0 = torch.zeros((), dtype=torch.long, device=device)
        out_toks0 = torch.zeros(B, self.K, dtype=torch.long, device=device)
        P_t = torch.tensor(P, dtype=torch.long, device=device)

        def cond(i, cur_tok, out_toks, k_cache, v_cache):
            return i < self.K

        def body(i, cur_tok, out_toks, k_cache, v_cache):
            # record the token at slot i
            out_toks = out_toks.index_copy(1, i.unsqueeze(0), cur_tok)
            pos = P_t + i
            emb = self.embed(cur_tok)                      # (B,1,H)
            position_ids = pos.view(1, 1)
            hidden, k_cache, v_cache = self._layers_pass(
                emb, k_cache, v_cache, pos, pos + 1, position_ids
            )
            last = self.final_norm(hidden)                 # (B,1,H)
            logits = self.lm_head(last)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            return (i + 1, nxt, out_toks, k_cache, v_cache)

        i, cur_tok, out_toks, k_cache, v_cache = torch.while_loop(
            cond, body, (i0, first_tok, out_toks0, k_cache, v_cache)
        )
        return out_toks


def build():
    m, inp = get_model_and_inputs()
    vla = m.vla
    input_ids, pixel_values = inp
    # Build the multimodal prefix embeddings the same way the VLM forward does.
    with torch.no_grad():
        patch_features = vla.vision_backbone(pixel_values)
        projected = vla.projector(patch_features)
        input_embeddings = vla.get_input_embeddings()(input_ids)
        prompt_embeds = torch.cat(
            [input_embeddings[:, :1, :], projected, input_embeddings[:, 1:, :]], dim=1
        )
    wrapper = StaticKVDecodeWrapper(vla).eval()
    return wrapper, (prompt_embeds,), vla


def ref_generate(vla, prompt_embeds):
    """Reference: HF greedy generate of K tokens from the same prompt embeds."""
    lm = vla.language_model
    with torch.no_grad():
        gen = lm.generate(
            inputs_embeds=prompt_embeds,
            max_new_tokens=K,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    return gen[:, -K:]


if __name__ == "__main__":
    torch.manual_seed(0)
    wrapper, inputs, vla = build()
    print("prompt_embeds shape:", inputs[0].shape, "K =", K)
    with torch.no_grad():
        out = wrapper(*inputs)
    print("eager decode out_toks:", out)

    # --- numerical check vs HF generate ---
    try:
        ref = ref_generate(vla, inputs[0])
        print("ref HF generate  :", ref)
        print("MATCH:", bool(torch.equal(out, ref)))
    except Exception as e:
        print("ref generate failed:", repr(e)[:200])

    # --- torch.export ---
    print("\n=== torch.export.export(strict=False) ===")
    ep = torch.export.export(wrapper, inputs, strict=False)
    txt = str(ep.graph_module.graph)
    has_wl = "while_loop" in txt
    print("export OK; while_loop node present:", has_wl)
    # count while_loop call nodes
    n_wl = sum(1 for n in ep.graph_module.graph.nodes
               if "while_loop" in str(getattr(n, "target", "")))
    print("while_loop call_function nodes:", n_wl)
    # any dynamic/symint dims in carried state?
    print("--- graph (head) ---")
    print("\n".join(txt.splitlines()[:40]))

    # --- m2m.convert ---
    print("\n=== m2m.convert (module) ===")
    import m2m
    res = m2m.convert(wrapper, inputs, output_type="linalg-on-tensors", backend="fx_importer")
    print("path_taken:", res.path_taken, "ok:", res.ok)
    mt = res.mlir_text or ""
    print("scf.while count       :", mt.count("scf.while"))
    print("func.call @while_loop :", mt.count("while_loop"))
    print("diagnostics:", res.diagnostics[:5])
    # dump a snippet
    with open("/tmp/claude-2621/-scratch-agustin-projects-oscar-merlin/7d66f954-67c6-438f-8e72-ebd11bf2d229/scratchpad/openvla_decode.mlir", "w") as f:
        f.write(mt)
    print("mlir len:", len(mt))
