"""torch.while_loop wrapper for TinyLlama's K-step autoregressive token-decode
loop, with a FIXED-SIZE (static) KV cache as shape-invariant carried state.

This mirrors the proven openVLA reference (same LLaMA decoder family) but for
TinyLlama, whose loader hands us `input_ids` (a prompt) rather than multimodal
prefix embeddings.

Carried state (all plain tensors, shape-invariant across the K steps):
    i        : scalar int64  -- decode step counter (0..K)
    cur_tok  : (B,1) int64   -- token fed at this step
    out_toks : (B,K) int64   -- collected decoded tokens (written at slot i)
    k_cache  : (L, B, Hkv, S, Dh) f32   -- key cache, written in-place at position pos+i
    v_cache  : (L, B, Hkv, S, Dh) f32   -- value cache
where S = prompt_len + K (static).  The prompt KV is pre-filled into the first
prompt_len positions before the loop; the body writes one new K/V per layer at
position (prompt_len + i) each step -> shape never changes -> exportable.

The body is a *manual* Llama decode step (RMSNorm -> QKV -> RoPE -> static-cache
write -> full-causal attention over the static window -> MLP -> final norm ->
lm_head -> argmax).  We avoid the HF Cache object because torch.while_loop iter_args
must be plain tensors.

WRAPPER RULES followed (from STATUS.md / openVLA reference):
  1. carry only evolving state; close over invariants (lm weights, P, etc).
  2. carried state is shape-invariant (static KV written in-place at pos+i).
  3. no out-of-scope mutation inside the body.
  4. every in-body tensor constant hoisted: position recomputed from carried i;
     no linspace/arange-of-constants/torch.tensor literals that become get_attr.
  5. cond returns i < K (K an int constant); body returns the new carry tuple.
"""
from merlin.common.paths import repo_root as _RR
from __future__ import annotations

import os

# Truncate layers for a tractable capture (real Llama arch, fewer layers).
os.environ.setdefault("M2M_LLAMA_LAYERS", "2")
os.environ.setdefault("M2M_SEQ", "8")

import torch
from torch import nn

import sys
sys.path.insert(0, "/path/to/model2MLIR/workloads/tiny_llama")
from loader import get_model_and_inputs  # noqa: E402

from transformers.models.llama.modeling_llama import (  # noqa: E402
    apply_rotary_pos_emb,
)

K = 7  # number of decode tokens to preserve as the loop bound


def rms_norm(x, weight, eps):
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return weight * x


class StaticKVDecodeWrapper(nn.Module):
    """forward(input_ids) -> out_toks (B,K).

    Runs prefill once (eagerly, traced) to fill the static KV cache for the prompt
    positions and produce the first decode token, then runs K manual decode steps
    as a single torch.while_loop.
    """

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        m = lm.model
        # IMPORTANT: do NOT call nn.Module.__call__ inside the while_loop body.
        # Under torch.export(strict=False) the body is traced by proxy_tensor,
        # whose call_module uses a WeakKeyDictionary keyed on the (already-tracked)
        # wrapper module -> AssertionError. So we hold the raw weight TENSORS and
        # use functional ops (F.linear / F.embedding) instead of submodules.
        # Register all weights as a single nn.ParameterDict at the TOP level (lifted
        # as flat inputs), and index them by name -- never as nested submodules.
        params = {}

        def reg(name, t):
            params[name] = nn.Parameter(t.detach().clone(), requires_grad=False)

        reg("embed", m.embed_tokens.weight)
        reg("final_norm", m.norm.weight)
        reg("lm_head", lm.lm_head.weight)
        self.lm_head_bias = lm.lm_head.bias is not None
        if self.lm_head_bias:
            reg("lm_head_bias", lm.lm_head.bias)
        for li, layer in enumerate(m.layers):
            a = layer.self_attn
            mlp = layer.mlp
            reg(f"l{li}_in_ln", layer.input_layernorm.weight)
            reg(f"l{li}_post_ln", layer.post_attention_layernorm.weight)
            reg(f"l{li}_q", a.q_proj.weight)
            reg(f"l{li}_k", a.k_proj.weight)
            reg(f"l{li}_v", a.v_proj.weight)
            reg(f"l{li}_o", a.o_proj.weight)
            reg(f"l{li}_gate", mlp.gate_proj.weight)
            reg(f"l{li}_up", mlp.up_proj.weight)
            reg(f"l{li}_down", mlp.down_proj.weight)
        self.p = nn.ParameterDict(params)
        # Close over inv_freq directly (a buffer -> lifts as additional_input).
        # cos/sin computed inline -- m.rotary_emb.forward has a @dynamic_rope_update
        # weakref cache decorator that dynamo cannot trace inside the HOP body.
        self.inv_freq = nn.Parameter(m.rotary_emb.inv_freq.detach().clone(), requires_grad=False)
        self.attention_scaling = float(getattr(m.rotary_emb, "attention_scaling", 1.0))
        cfg = lm.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        self.eps = cfg.rms_norm_eps
        self.hidden = cfg.hidden_size
        self.n_rep = self.n_heads // self.n_kv
        self.K = K

    def _repeat_kv(self, x):
        # x: (B, Hkv, S, Dh) -> (B, Hkv*n_rep, S, Dh)
        B, Hkv, S, Dh = x.shape
        x = x[:, :, None, :, :].expand(B, Hkv, self.n_rep, S, Dh)
        return x.reshape(B, Hkv * self.n_rep, S, Dh)

    def _rope_cos_sin(self, position_ids, dtype, device):
        # Inline LlamaRotaryEmbedding.forward (no weakref cache decorator).
        inv_freq = self.inv_freq.to(device)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    # --- one transformer pass over (B, T, H), writing KV at [pos:pos+T] ---
    # Pure functional (F.linear over closed-over weight tensors) -- no submodule
    # __call__, so the proxy_tensor module-tracking weakref is never hit.
    def _layers_pass(self, hidden, k_cache, v_cache, pos, position_ids):
        F = torch.nn.functional
        p = self.p
        cos, sin = self._rope_cos_sin(position_ids, hidden.dtype, hidden.device)
        B, T, _ = hidden.shape
        S = k_cache.shape[3]
        new_k = []
        new_v = []
        for li in range(self.n_layers):
            residual = hidden
            hs = rms_norm(hidden, p[f"l{li}_in_ln"], self.eps)
            q = F.linear(hs, p[f"l{li}_q"]).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = F.linear(hs, p[f"l{li}_k"]).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            v = F.linear(hs, p[f"l{li}_v"]).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            # write new k/v into the static cache at [pos : pos+T]
            write_idx = pos + torch.arange(T, device=hidden.device)
            kc = k_cache[li].index_copy(2, write_idx, k)
            vc = v_cache[li].index_copy(2, write_idx, v)
            new_k.append(kc)
            new_v.append(vc)
            # GQA expand then attention over the full static window
            kc_e = self._repeat_kv(kc)
            vc_e = self._repeat_kv(vc)
            attn_w = torch.matmul(q, kc_e.transpose(2, 3)) / (self.head_dim ** 0.5)
            key_pos = torch.arange(S, device=hidden.device)
            qpos = (pos + torch.arange(T, device=hidden.device)).unsqueeze(1)  # (T,1)
            causal = key_pos.unsqueeze(0) <= qpos                              # (T,S)
            mask = causal.unsqueeze(0).unsqueeze(0)                            # (1,1,T,S)
            attn_w = attn_w.masked_fill(~mask, float("-inf"))
            attn_w = torch.softmax(attn_w, dim=-1)
            out = torch.matmul(attn_w, vc_e)                                  # (B,Hq,T,Dh)
            out = out.transpose(1, 2).reshape(B, T, -1)
            out = F.linear(out, p[f"l{li}_o"])
            hidden = residual + out
            # MLP
            residual = hidden
            hs = rms_norm(hidden, p[f"l{li}_post_ln"], self.eps)
            hs = F.linear(F.silu(F.linear(hs, p[f"l{li}_gate"])) * F.linear(hs, p[f"l{li}_up"]),
                          p[f"l{li}_down"])
            hidden = residual + hs
        k_cache = torch.stack(new_k, 0)
        v_cache = torch.stack(new_v, 0)
        return hidden, k_cache, v_cache

    def _embed(self, ids):
        return torch.nn.functional.embedding(ids, self.p["embed"])

    def _final_norm(self, x):
        return rms_norm(x, self.p["final_norm"], self.eps)

    def _lm_head(self, x):
        b = self.p["lm_head_bias"] if self.lm_head_bias else None
        return torch.nn.functional.linear(x, self.p["lm_head"], b)

    def forward(self, input_ids):
        B, P = input_ids.shape
        device = input_ids.device
        S = P + self.K  # static cache length
        k_cache = torch.zeros(self.n_layers, B, self.n_kv, S, self.head_dim, device=device)
        v_cache = torch.zeros(self.n_layers, B, self.n_kv, S, self.head_dim, device=device)

        # --- prefill: run the prompt through all layers, fill cache[0:P] ---
        prompt_embeds = self._embed(input_ids)
        pos0 = torch.zeros((), dtype=torch.long, device=device)
        ppos = torch.arange(P, device=device).unsqueeze(0)
        hidden, k_cache, v_cache = self._layers_pass(prompt_embeds, k_cache, v_cache, pos0, ppos)
        last = self._final_norm(hidden[:, -1:, :])
        logits = self._lm_head(last)                             # (B,1,V)
        first_tok = logits[:, -1, :].argmax(-1, keepdim=True)    # (B,1)

        # --- carried state ---
        i0 = torch.zeros((), dtype=torch.long, device=device)
        out_toks0 = torch.zeros(B, self.K, dtype=torch.long, device=device)
        P_t = torch.tensor(P, dtype=torch.long, device=device)  # closed-over invariant scalar

        def cond(i, cur_tok, out_toks, k_cache, v_cache):
            return i < self.K

        def body(i, cur_tok, out_toks, k_cache, v_cache):
            out_toks = out_toks.index_copy(1, i.unsqueeze(0), cur_tok)
            pos = P_t + i
            emb = self._embed(cur_tok)                           # (B,1,H)
            position_ids = pos.view(1, 1)
            hidden, k_cache, v_cache = self._layers_pass(emb, k_cache, v_cache, pos, position_ids)
            last = self._final_norm(hidden)                      # (B,1,H)
            logits = self._lm_head(last)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            return (i + 1, nxt, out_toks, k_cache, v_cache)

        i, cur_tok, out_toks, k_cache, v_cache = torch.while_loop(
            cond, body, (i0, first_tok, out_toks0, k_cache, v_cache)
        )
        return out_toks


def build():
    m, inp = get_model_and_inputs()
    lm = m.lm  # the underlying LlamaForCausalLM (m is the _LogitsOnly wrapper)
    wrapper = StaticKVDecodeWrapper(lm).eval()
    return wrapper, inp, lm


def ref_unrolled(wrapper, input_ids):
    """Eager K-step decode by directly running body K times (the unrolled loop)."""
    with torch.no_grad():
        B, P = input_ids.shape
        device = input_ids.device
        S = P + wrapper.K
        k_cache = torch.zeros(wrapper.n_layers, B, wrapper.n_kv, S, wrapper.head_dim, device=device)
        v_cache = torch.zeros(wrapper.n_layers, B, wrapper.n_kv, S, wrapper.head_dim, device=device)
        prompt_embeds = wrapper._embed(input_ids)
        pos0 = torch.zeros((), dtype=torch.long, device=device)
        ppos = torch.arange(P, device=device).unsqueeze(0)
        hidden, k_cache, v_cache = wrapper._layers_pass(prompt_embeds, k_cache, v_cache, pos0, ppos)
        last = wrapper._final_norm(hidden[:, -1:, :])
        cur = wrapper._lm_head(last)[:, -1, :].argmax(-1, keepdim=True)
        out = torch.zeros(B, wrapper.K, dtype=torch.long, device=device)
        P_t = torch.tensor(P, dtype=torch.long, device=device)
        for step in range(wrapper.K):
            out[:, step:step + 1] = cur
            pos = P_t + torch.tensor(step, dtype=torch.long, device=device)
            emb = wrapper._embed(cur)
            position_ids = pos.view(1, 1)
            hidden, k_cache, v_cache = wrapper._layers_pass(emb, k_cache, v_cache, pos, position_ids)
            last = wrapper._final_norm(hidden)
            cur = wrapper._lm_head(last)[:, -1, :].argmax(-1, keepdim=True)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    wrapper, inputs, lm = build()
    print("input_ids shape:", inputs[0].shape, "K =", K, "n_layers =", wrapper.n_layers)
    with torch.no_grad():
        out = wrapper(*inputs)
    print("eager (while_loop) out_toks:", out)

    ref = ref_unrolled(wrapper, inputs[0])
    print("ref unrolled out_toks      :", ref)
    print("TOKEN MATCH:", bool(torch.equal(out, ref)))

    print("\n=== torch.export.export(strict=False) ===")
    ep = torch.export.export(wrapper, inputs, strict=False)
    txt = str(ep.graph_module.graph)
    n_wl = sum(1 for n in ep.graph_module.graph.nodes
               if "while_loop" in str(getattr(n, "target", "")))
    print("while_loop call_function nodes:", n_wl)

    print("\n=== m2m.convert ===")
    import m2m
    res = m2m.convert(wrapper, inputs, backend="fx_importer",
                      quantization=None, level="linalg-on-tensors")
    mt = res.mlir_text or ""
    print("ok:", getattr(res, "ok", "?"))
    print("scf.for count        :", mt.count("scf.for"))
    print("func.call @while_loop:", mt.count("while_loop"))
    out_path = str(_RR() / "merlin/benchmarks/dse_guidance/recaptures_loop/tiny_llama/model.mlir")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(mt)
    print("saved:", out_path, "len:", len(mt))
