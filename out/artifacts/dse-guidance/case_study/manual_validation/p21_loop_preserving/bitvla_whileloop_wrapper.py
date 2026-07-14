"""torch.while_loop wrapper for **bitvla** (BitNet W1.58-A8 VLA) K-step autoregressive
action-token decode, with a FIXED-SIZE (static) KV cache as shape-invariant carried state.

Same decode family as openVLA (see openvla_whileloop_decode_wrapper.py); the only
differences are BitNet-specific layer internals:
  * q/k/v/o projections are BitLinear (ActQuant.apply(input) + WeightQuant.apply(weight)
    in the DEFAULT f32 fake-quant path -- NO `.item()` in the traced body; the native
    int2 path (BITVLA_NATIVE_QUANT=1) is a SEPARATE track and is NOT used here).
  * an extra `attn_sub_norm` RMSNorm after the attention reshape, before o_proj.
  * MLP uses squared_relu (relu(x)**2), an `ffn_sub_norm` RMSNorm before down_proj, and
    BitLinear gate/up/down.
  * BitNet RoPE: `attn.rotary_emb(v, seq_len=S)` returns the full cos/sin tables which
    `apply_rotary_pos_emb(q, k, cos, sin, position_ids)` then indexes by position_ids.

Carried state (all plain, shape-invariant tensors across the K steps):
    i        : scalar int64                          -- decode step counter (0..K)
    cur_tok  : (B,1) int64                           -- token fed at this step
    out_toks : (B,K) int64                           -- collected action tokens (slot i)
    k_cache  : (L, B, Hkv, S, Dh) f32               -- key cache, written in-place at pos
    v_cache  : (L, B, Hkv, S, Dh) f32               -- value cache
where S = prompt_len + K (static). The prompt KV is pre-filled into the first prompt_len
positions before the loop; the body writes one new K/V per layer at position (prompt_len+i)
each step -> shapes never change -> exportable -> scf.for.

NOTE on the reference: BitVLA's real `predict_action` is a single bidirectional forward,
not an autoregressive loop. This wrapper deliberately builds a *causal greedy decode* in
the openVLA decode-family shape (the task's purpose: prove the while_loop->scf.for capture
on the BitNet datapath). The numeric check compares this wrapper's eager run against an
eager-unrolled run of the very same body (same math, K times), not against predict_action.
"""
from __future__ import annotations

import os

os.environ.setdefault("BITVLA_LLM_LAYERS", "2")
os.environ.setdefault("BITVLA_SEQ", "32")
# Force the DEFAULT f32 fake-quant path (absmean WeightQuant) -- do NOT set
# BITVLA_NATIVE_QUANT (the packed-int2 path calls .item() and graph-breaks export).
os.environ.pop("BITVLA_NATIVE_QUANT", None)

import sys

sys.path.insert(0, "/path/to/model2MLIR/workloads/bitvla")

import torch
from torch import nn

from loader import get_model_and_inputs  # noqa: E402

# BitNet's own apply_rotary_pos_emb (signature differs from Llama's: takes position_ids).
from transformers.models.llava.modeling_bitnet import apply_rotary_pos_emb  # noqa: E402

def _repo_root():
    from pathlib import Path as _P
    p = _P(__file__).resolve()
    while p != p.parent and not (p / "merlin" / "python").is_dir():
        p = p.parent
    return p
_ROOT = _repo_root()

K = 7  # action-token chunk to decode (IR loop constant)


def rms_norm(x, weight, eps):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return weight * x.to(input_dtype)


def squared_relu(x):
    return torch.nn.functional.relu(x) ** 2


class StaticKVDecodeWrapper(nn.Module):
    """forward(prompt_embeds) -> out_toks (B,K).

    Prefill the prompt into the static KV cache (eager, traced) then run K manual BitNet
    decode steps as a single torch.while_loop.
    """

    def __init__(self, vla):
        super().__init__()
        lm = vla.language_model
        m = lm.model
        self.layers = m.layers
        self.final_norm = m.norm
        self.embed = m.embed_tokens
        self.lm_head = lm.lm_head
        cfg = vla.config.text_config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.eps = cfg.rms_norm_eps
        self.hidden = cfg.hidden_size
        self.K = K

    def _repeat_kv(self, x, T, S):
        # x: (B, Hkv, S, Dh) -> (B, Hkv*n_rep, S, Dh)
        if self.n_rep == 1:
            return x
        B = x.shape[0]
        x = x[:, :, None, :, :].expand(B, self.n_kv, self.n_rep, S, self.head_dim)
        return x.reshape(B, self.n_kv * self.n_rep, S, self.head_dim)

    def _layers_pass(self, hidden, k_cache, v_cache, pos, position_ids, idx_T, idx_S):
        """One BitNet pass over (B,T,H), writing KV at [pos:pos+T] of the static cache.

        idx_T: precomputed torch.arange(T) (hoisted constant, passed in -- no in-body arange).
        idx_S: precomputed torch.arange(S) key positions (hoisted constant, passed in).
        position_ids: (1,T) long -- absolute positions for RoPE / causal masking.
        """
        B, T, _ = hidden.shape
        S = k_cache.shape[3]
        new_k = []
        new_v = []
        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            residual = hidden
            hs = layer.input_layernorm(hidden)
            q = attn.q_proj(hs).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(hs).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            v = attn.v_proj(hs).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            # BitNet RoPE: full tables sized to S, indexed inside by position_ids.
            cos, sin = attn.rotary_emb(v, seq_len=S)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            # write new k/v into the static cache at [pos : pos+T]
            kc = k_cache[li].index_copy(2, pos + idx_T, k)
            vc = v_cache[li].index_copy(2, pos + idx_T, v)
            new_k.append(kc)
            new_v.append(vc)
            kc_r = self._repeat_kv(kc, T, S)
            vc_r = self._repeat_kv(vc, T, S)
            attn_w = torch.matmul(q, kc_r.transpose(2, 3)) / (self.head_dim ** 0.5)
            # causal mask over the static window: key j valid iff j <= query absolute pos
            qpos = (pos + idx_T).unsqueeze(1)  # (T,1)
            causal = idx_S.unsqueeze(0) <= qpos  # (T,S)
            mask = causal.unsqueeze(0).unsqueeze(0)  # (1,1,T,S)
            attn_w = attn_w.masked_fill(~mask, float("-inf"))
            attn_w = torch.softmax(attn_w, dim=-1)
            out = torch.matmul(attn_w, vc_r)  # (B,H,T,Dh)
            out = out.transpose(1, 2).reshape(B, T, self.hidden)
            out = attn.attn_sub_norm(out)
            out = attn.o_proj(out)
            hidden = residual + out
            # MLP (squared_relu + ffn_sub_norm)
            residual = hidden
            hs = layer.post_attention_layernorm(hidden)
            mlp = layer.mlp
            gated = squared_relu(mlp.gate_proj(hs)) * mlp.up_proj(hs)
            hs = mlp.down_proj(mlp.ffn_sub_norm(gated))
            hidden = residual + hs
        k_cache = torch.stack(new_k, 0)
        v_cache = torch.stack(new_v, 0)
        return hidden, k_cache, v_cache

    def forward(self, prompt_embeds):
        B, P, H = prompt_embeds.shape
        device = prompt_embeds.device
        S = P + self.K
        k_cache = torch.zeros(self.n_layers, B, self.n_kv, S, self.head_dim, device=device)
        v_cache = torch.zeros(self.n_layers, B, self.n_kv, S, self.head_dim, device=device)

        # --- HOISTED tensor constants (must not appear inside the while_loop body) ---
        idx_S = torch.arange(S, device=device)                 # (S,) key positions
        idx_prefill = torch.arange(P, device=device)           # (P,) prefill positions
        idx_one = torch.arange(1, device=device)               # (1,) one-step positions

        # --- prefill: run the prompt through all layers, fill cache[0:P] ---
        pos0 = torch.zeros((), dtype=torch.long, device=device)
        ppos = idx_prefill.unsqueeze(0)                        # (1,P)
        hidden, k_cache, v_cache = self._layers_pass(
            prompt_embeds, k_cache, v_cache, pos0, ppos, idx_prefill, idx_S
        )
        last = self.final_norm(hidden[:, -1:, :])
        logits = self.lm_head(last)
        first_tok = logits[:, -1, :].argmax(-1, keepdim=True)  # (B,1)

        # --- carried state ---
        i0 = torch.zeros((), dtype=torch.long, device=device)
        out_toks0 = torch.zeros(B, self.K, dtype=torch.long, device=device)
        P_t = torch.tensor(P, dtype=torch.long, device=device)

        def cond(i, cur_tok, out_toks, k_cache, v_cache):
            return i < self.K

        def body(i, cur_tok, out_toks, k_cache, v_cache):
            out_toks = out_toks.index_copy(1, i.unsqueeze(0), cur_tok)
            pos = P_t + i
            emb = self.embed(cur_tok)                          # (B,1,H)
            position_ids = pos.view(1, 1)
            hidden, k_cache, v_cache = self._layers_pass(
                emb, k_cache, v_cache, pos, position_ids, idx_one, idx_S
            )
            last = self.final_norm(hidden)
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
    inputs_embeds, _attn_mask = inp
    wrapper = StaticKVDecodeWrapper(vla).eval()
    return wrapper, (inputs_embeds,), vla


def ref_unrolled(wrapper, prompt_embeds):
    """Eager reference: same body math, unrolled K times in Python (no while_loop)."""
    B, P, H = prompt_embeds.shape
    device = prompt_embeds.device
    S = P + wrapper.K
    idx_S = torch.arange(S, device=device)
    k_cache = torch.zeros(wrapper.n_layers, B, wrapper.n_kv, S, wrapper.head_dim, device=device)
    v_cache = torch.zeros(wrapper.n_layers, B, wrapper.n_kv, S, wrapper.head_dim, device=device)
    idx_prefill = torch.arange(P, device=device)
    idx_one = torch.arange(1, device=device)
    pos0 = torch.zeros((), dtype=torch.long, device=device)
    hidden, k_cache, v_cache = wrapper._layers_pass(
        prompt_embeds, k_cache, v_cache, pos0, idx_prefill.unsqueeze(0), idx_prefill, idx_S
    )
    last = wrapper.final_norm(hidden[:, -1:, :])
    cur = wrapper.lm_head(last)[:, -1, :].argmax(-1, keepdim=True)
    out = torch.zeros(B, wrapper.K, dtype=torch.long, device=device)
    P_t = torch.tensor(P, dtype=torch.long, device=device)
    for step in range(wrapper.K):
        i = torch.tensor(step, dtype=torch.long, device=device)
        out = out.index_copy(1, i.unsqueeze(0), cur)
        pos = P_t + i
        emb = wrapper.embed(cur)
        hidden, k_cache, v_cache = wrapper._layers_pass(
            emb, k_cache, v_cache, pos, pos.view(1, 1), idx_one, idx_S
        )
        last = wrapper.final_norm(hidden)
        cur = wrapper.lm_head(last)[:, -1, :].argmax(-1, keepdim=True)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    wrapper, inputs, vla = build()
    print("prompt_embeds shape:", inputs[0].shape, "K =", K)

    with torch.no_grad():
        out = wrapper(*inputs)
    print("eager while-loop wrapper out_toks:", out)

    with torch.no_grad():
        ref = ref_unrolled(wrapper, inputs[0])
    print("eager unrolled ref         :", ref)
    print("NUMERIC MATCH:", bool(torch.equal(out, ref)))

    print("\n=== torch.export.export(strict=False) ===")
    ep = torch.export.export(wrapper, inputs, strict=False)
    txt = str(ep.graph_module.graph)
    n_wl = sum(1 for n in ep.graph_module.graph.nodes
               if "while_loop" in str(getattr(n, "target", "")))
    print("while_loop call_function nodes:", n_wl)

    print("\n=== m2m.convert (linalg-on-tensors) ===")
    import m2m
    res = m2m.convert(wrapper, inputs, output_type="linalg-on-tensors",
                      backend="fx_importer", quantization=None)
    print("path_taken:", res.path_taken, "ok:", res.ok)
    mt = res.mlir_text or ""
    print("scf.for count:", mt.count("scf.for"))
    print("func.call @while_loop count:", mt.count("while_loop"))
    out_dir = f"{_ROOT}/merlin/benchmarks/dse_guidance/recaptures_loop/bitvla"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model.mlir"), "w") as f:
        f.write(mt)
    print("saved:", os.path.join(out_dir, "model.mlir"), "len:", len(mt))
