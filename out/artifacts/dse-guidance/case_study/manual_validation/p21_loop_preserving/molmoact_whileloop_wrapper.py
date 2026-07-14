"""torch.while_loop wrapper for MolmoAct's K=8 autoregressive action-token decode,
with a FIXED-SIZE (static) KV cache as shape-invariant carried state.

MolmoAct is a Molmo/OLMo-lineage VLA: SigLIP2 ViT + adapter + a Qwen2-style LLM
decoder (MolmoActLlm) that autoregressively emits reasoning/depth/action tokens.
We capture the *action-token decode loop* of the LLM decoder exactly the way the
proven openVLA wrapper does (same autoregressive family, static KV cache as a
shape-invariant iter_arg).

Carried state (all plain tensors, shape-invariant across the K steps):
    i        : scalar int64                 -- decode step counter (0..K)
    cur_tok  : (B,1) int64                  -- token fed at this step
    out_toks : (B,K) int64                  -- collected action tokens (written at slot i)
    k_cache  : (L, B, Hkv, S, Dh) f32       -- key cache, written in-place at pos+i
    v_cache  : (L, B, Hkv, S, Dh) f32       -- value cache
where S = prompt_len + K (static).  The prompt KV is pre-filled into the first
prompt_len positions before the loop; the body writes one new K/V per layer at
position (prompt_len + i) each step -> shape never changes -> exportable.

Architecture deltas vs openVLA's Llama (handled below):
  * fused QKV: a single `att_proj` (with qkv_bias) split by fused_dims, not q/k/v_proj.
  * norm names: `attn_norm` (pre-attention) and `ff_norm` (pre-MLP); final `ln_f`.
  * MLP: `ff_proj` -> chunk(2) -> act(gate)*x -> `ff_out`  (not gate/up/down_proj).
  * GQA: num_attention_heads=28 vs num_key_value_heads=4 -> repeat_kv before attention.
  * custom token embedding: MolmoActEmbedding (embedding ++ new_embedding) reused directly.
  * pre-norm decoder (norm_after=False, the loader default) and use_qk_norm=False.

K=8 from the dse_guidance registry (autoregressive_vla/action_token_decode).
"""
from merlin.common.paths import repo_root as _RR
from __future__ import annotations

import os
import sys

# Match the loader's smoke config so the captured graph is tractable.
os.environ.setdefault("M2M_MOLMOACT_LAYERS", "4")
os.environ.setdefault("M2M_MOLMOACT_VOCAB", "4096")
os.environ.setdefault("M2M_SEQ", "8")

import torch
from torch import nn

sys.path.insert(0, "/path/to/model2MLIR/workloads/molmoact")
from loader import get_model_and_inputs  # noqa: E402

_MOLMOACT_REPO = os.environ.get("MOLMOACT_REPO", "/path/to/molmoact")
if _MOLMOACT_REPO not in sys.path:
    sys.path.insert(0, _MOLMOACT_REPO)
from olmo.hf_model.molmoact.modeling_molmoact import (  # noqa: E402
    apply_rotary_pos_emb,
    repeat_kv,
)

K = 8  # MolmoAct action dimension (registry: autoregressive_vla/action_token_decode)


def rms_norm(x, weight, eps):
    # MolmoActRMSNorm in float32, then scale by weight (matches modeling_molmoact).
    og = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    x = x.to(og)
    return weight * x


class StaticKVDecodeWrapper(nn.Module):
    """forward(input_ids) -> out_toks  (the K decoded action tokens).

    Prefill the prompt once (eagerly, traced) into the static KV cache and produce
    the first decode token, then run K manual decode steps as one torch.while_loop.
    """

    def __init__(self, causal_lm):
        super().__init__()
        m = causal_lm.model            # MolmoActLlm
        self.blocks = m.blocks
        self.final_norm = m.ln_f
        self.embed = m.wte             # MolmoActEmbedding (or nn.Embedding)
        self.rotary = m.rotary_emb
        self.lm_head = causal_lm.lm_head
        cfg = causal_lm.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv
        self.head_dim = cfg.head_dim
        self.eps = cfg.layer_norm_eps
        self.hidden = cfg.hidden_size
        self.norm_after = cfg.norm_after
        # fused QKV split sizes (query | key | value), in feature units
        self.fused_dims = (
            cfg.hidden_size,
            cfg.head_dim * cfg.num_key_value_heads,
            cfg.head_dim * cfg.num_key_value_heads,
        )
        self.K = K

    # --- one transformer pass over a (B, T, H) hidden, writing KV at [pos:pos+T] ---
    def _layers_pass(self, hidden, k_cache, v_cache, pos, position_ids):
        cos, sin = self.rotary(hidden, position_ids)
        B, T, _ = hidden.shape
        S = k_cache.shape[3]
        scaling = self.head_dim ** -0.5
        new_k = []
        new_v = []
        key_pos = torch.arange(S, device=hidden.device)
        qpos = (pos + torch.arange(T, device=hidden.device)).unsqueeze(1)   # (T,1)
        causal = key_pos.unsqueeze(0) <= qpos                               # (T,S)
        mask = causal.unsqueeze(0).unsqueeze(0)                             # (1,1,T,S)
        for li, block in enumerate(self.blocks):
            attn = block.self_attn
            residual = hidden
            hs = rms_norm(hidden, block.attn_norm.weight, self.eps)
            qkv = attn.att_proj(hs)
            q, k, v = torch.split(qkv, list(self.fused_dims), dim=-1)
            q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            # write new k/v into the static cache at [pos : pos+T]
            idx = pos + torch.arange(T, device=hidden.device)
            kc = k_cache[li].index_copy(2, idx, k)
            vc = v_cache[li].index_copy(2, idx, v)
            new_k.append(kc)
            new_v.append(vc)
            # GQA: repeat kv heads to query-head count
            kc_r = repeat_kv(kc, self.n_rep)
            vc_r = repeat_kv(vc, self.n_rep)
            attn_w = torch.matmul(q, kc_r.transpose(2, 3)) * scaling
            attn_w = attn_w.masked_fill(~mask, float("-inf"))
            attn_w = torch.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(attn_w, vc_r)                               # (B,H,T,Dh)
            out = out.transpose(1, 2).reshape(B, T, -1)
            out = attn.attn_out(out)
            if self.norm_after:
                out = rms_norm(out, block.attn_norm.weight, self.eps)
            hidden = residual + out
            # MLP
            residual = hidden
            mlp_in = hidden if self.norm_after else rms_norm(hidden, block.ff_norm.weight, self.eps)
            ff = block.mlp.ff_proj(mlp_in)
            ff_x, gate = torch.chunk(ff, 2, dim=-1)
            ff_x = block.mlp.act(gate) * ff_x
            ff_x = block.mlp.ff_out(ff_x)
            if self.norm_after:
                ff_x = rms_norm(ff_x, block.ff_norm.weight, self.eps)
            hidden = residual + ff_x
        k_cache = torch.stack(new_k, 0)
        v_cache = torch.stack(new_v, 0)
        return hidden, k_cache, v_cache

    def forward(self, input_ids):
        B, P = input_ids.shape
        device = input_ids.device
        S = P + self.K
        k_cache = torch.zeros(self.n_layers, B, self.n_kv, S, self.head_dim, device=device)
        v_cache = torch.zeros(self.n_layers, B, self.n_kv, S, self.head_dim, device=device)

        # --- prefill: run the prompt through all layers, fill cache[0:P] ---
        safe_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        prompt_embeds = self.embed(safe_ids)
        pos0 = torch.zeros((), dtype=torch.long, device=device)
        ppos = torch.arange(P, device=device).unsqueeze(0)
        hidden, k_cache, v_cache = self._layers_pass(prompt_embeds, k_cache, v_cache, pos0, ppos)
        last = self.final_norm(hidden[:, -1:, :])
        logits = self.lm_head(last)
        first_tok = logits[:, -1, :].argmax(-1, keepdim=True)             # (B,1)

        # --- carried state ---
        i0 = torch.zeros((), dtype=torch.long, device=device)
        out_toks0 = torch.zeros(B, self.K, dtype=torch.long, device=device)
        P_t = torch.tensor(P, dtype=torch.long, device=device)

        def cond(i, cur_tok, out_toks, k_cache, v_cache):
            return i < self.K

        def body(i, cur_tok, out_toks, k_cache, v_cache):
            out_toks = out_toks.index_copy(1, i.unsqueeze(0), cur_tok)
            pos = P_t + i
            emb = self.embed(cur_tok)                                     # (B,1,H)
            position_ids = pos.view(1, 1)
            hidden, k_cache, v_cache = self._layers_pass(emb, k_cache, v_cache, pos, position_ids)
            last = self.final_norm(hidden)
            logits = self.lm_head(last)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            return (i + 1, nxt, out_toks, k_cache, v_cache)

        i, cur_tok, out_toks, k_cache, v_cache = torch.while_loop(
            cond, body, (i0, first_tok, out_toks0, k_cache, v_cache)
        )
        return out_toks


def build():
    m, inp = get_model_and_inputs()       # m is _LogitsOnly(MolmoActForCausalLM)
    causal_lm = m.lm
    (input_ids,) = inp
    wrapper = StaticKVDecodeWrapper(causal_lm).eval()
    return wrapper, (input_ids,), causal_lm


def ref_unrolled_decode(wrapper, input_ids):
    """Reference: eager greedy decode of K tokens using the same manual decode path,
    but unrolled in Python (no while_loop). Numeric oracle for the loop body."""
    with torch.no_grad():
        B, P = input_ids.shape
        device = input_ids.device
        S = P + wrapper.K
        k_cache = torch.zeros(wrapper.n_layers, B, wrapper.n_kv, S, wrapper.head_dim, device=device)
        v_cache = torch.zeros(wrapper.n_layers, B, wrapper.n_kv, S, wrapper.head_dim, device=device)
        safe_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        prompt_embeds = wrapper.embed(safe_ids)
        pos0 = torch.zeros((), dtype=torch.long, device=device)
        ppos = torch.arange(P, device=device).unsqueeze(0)
        hidden, k_cache, v_cache = wrapper._layers_pass(prompt_embeds, k_cache, v_cache, pos0, ppos)
        last = wrapper.final_norm(hidden[:, -1:, :])
        cur = wrapper.lm_head(last)[:, -1, :].argmax(-1, keepdim=True)
        out = torch.zeros(B, wrapper.K, dtype=torch.long, device=device)
        for i in range(wrapper.K):
            out[:, i:i + 1] = cur
            pos = torch.tensor(P + i, dtype=torch.long, device=device)
            emb = wrapper.embed(cur)
            hidden, k_cache, v_cache = wrapper._layers_pass(emb, k_cache, v_cache, pos, pos.view(1, 1))
            last = wrapper.final_norm(hidden)
            cur = wrapper.lm_head(last)[:, -1, :].argmax(-1, keepdim=True)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    wrapper, inputs, causal_lm = build()
    print("input_ids shape:", inputs[0].shape, "K =", K)
    with torch.no_grad():
        out = wrapper(*inputs)
    print("eager while_loop out_toks:", out)

    # --- numeric check vs unrolled eager decode ---
    ref = ref_unrolled_decode(wrapper, inputs[0])
    print("ref unrolled out_toks    :", ref)
    print("MATCH:", bool(torch.equal(out, ref)))

    # --- torch.export ---
    print("\n=== torch.export.export(strict=False) ===")
    ep = torch.export.export(wrapper, inputs, strict=False)
    txt = str(ep.graph_module.graph)
    n_wl = sum(1 for n in ep.graph_module.graph.nodes
               if "while_loop" in str(getattr(n, "target", "")))
    print("while_loop call_function nodes:", n_wl)

    # --- m2m.convert ---
    print("\n=== m2m.convert (linalg-on-tensors) ===")
    import m2m
    res = m2m.convert(wrapper, inputs, output_type="linalg-on-tensors", backend="fx_importer")
    print("path_taken:", res.path_taken, "ok:", res.ok)
    mt = res.mlir_text or ""
    print("scf.for count:", mt.count("scf.for"))
    print("diagnostics:", res.diagnostics[:5])

    out_path = str(_RR() / "merlin/benchmarks/dse_guidance/recaptures_loop/molmoact/model.mlir")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(mt)
    print("saved MLIR ->", out_path, "len:", len(mt))
