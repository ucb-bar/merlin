"""torch.while_loop wrapper for small_llama's autoregressive token-decode loop.

Extends the PROVEN P21 loop-preserving technique (smolVLA / openVLA / pi0.5) to
the model2MLIR `small_llama` workload — a self-contained tiny LLaMA-style decoder
(RMSNorm + RoPE + full-causal attention + SwiGLU MLP + lm_head), see
`/path/to/model2MLIR/workloads/small_llama/loader.py`.

This mirrors `openvla_whileloop_decode_wrapper.py` (same decoder-LLM family):
prefill once -> static, fixed-size KV cache -> K greedy-decode steps as a single
`torch.while_loop`, KV written IN-PLACE at position (prompt_len + i) each step so
the carried state is SHAPE-INVARIANT and the loop exports (not unrolled) and
lowers to `scf.for` via m2m's `_lower_while_loop`.

Carried state (all plain, shape-invariant tensors):
    i        : scalar int64  ()          -- decode step counter (0..K)
    cur_tok  : (B,1) int64               -- token fed at this step
    out_toks : (B,K) int64               -- collected tokens (written at slot i)
    k_cache  : (L, B, H, S, Dh) f32      -- key cache, written in-place at pos P+i
    v_cache  : (L, B, H, S, Dh) f32      -- value cache
where S = prompt_len + K (static).

RULES followed (STATUS.md, empirically required):
  1. carry only evolving state; close over invariants (weights, tables).
  2. carried state shape-invariant: static fixed-size KV cache, in-place index_copy.
  3. no out-of-scope mutation in the body.
  4. HOIST every in-body tensor constant — small_llama recomputes `torch.arange`
     (rope freq + pos) and `torch.full(...).triu(1)` (causal mask) INSIDE its
     forward; those become invalid get_attr constants in the HOP subgraph. We
     precompute the full cos/sin RoPE tables over all S positions and the key-pos
     index vector ONCE outside the loop and recompute the per-step position from
     the carried counter `i` (don't index by python step).
  5. cond_fn: i < K ; body_fn returns the new carry tuple.
"""
from __future__ import annotations

import math
import sys

import torch
from torch import nn

sys.path.insert(0, "/path/to/model2MLIR/workloads/small_llama")
from loader import get_model_and_inputs  # noqa: E402

from torch._higher_order_ops.while_loop import while_loop  # noqa: E402

K = 7  # number of tokens to decode (IR constant -> scf.for bound)


def _rms_norm(x, w, eps):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w


def _build_rope_tables(S, dh, base=10000.0):
    """Full (S, dh) cos/sin RoPE tables for absolute positions 0..S-1.

    Mirrors loader.rope: half-split, [ang.cos(),ang.cos()] / [ang.sin(),ang.sin()].
    Computed ONCE outside the loop and closed over (rule 4).
    """
    half = dh // 2
    freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))  # (half,)
    pos = torch.arange(S, dtype=torch.float32)                                  # (S,)
    ang = pos[:, None] * freq[None, :]                                          # (S, half)
    cos = torch.cat([ang.cos(), ang.cos()], -1)                                 # (S, dh)
    sin = torch.cat([ang.sin(), ang.sin()], -1)                                 # (S, dh)
    return cos, sin


def _apply_rope_at(x, cos_rows, sin_rows, half):
    """x: (B,H,T,Dh); cos_rows/sin_rows: (T,Dh) for the T query positions."""
    cos = cos_rows[None, None]   # (1,1,T,Dh)
    sin = sin_rows[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


def _layers_pass(hidden, k_cache, v_cache, qpos_rows, cos_tab, sin_tab, key_pos,
                 W, n_layers, n_heads, head_dim, eps):
    """One transformer pass over (B,T,D) hidden, writing KV at the query positions
    of the static cache and attending over the full static window (causal mask).

    Module-FREE (plain tensors + python scalars only) so it can run inside the
    torch.while_loop body — dynamo cannot track an nn.Module across the body, so
    the body must not reach through `self`. Reproduces small_llama's Block exactly.
    """
    B, T, D = hidden.shape
    H, Dh = n_heads, head_dim
    half = Dh // 2
    cos_rows = cos_tab.index_select(0, qpos_rows)   # (T,Dh)
    sin_rows = sin_tab.index_select(0, qpos_rows)
    new_k, new_v = [], []
    for li in range(n_layers):
        residual = hidden
        hs = _rms_norm(hidden, W["n1"][li], eps)
        q = torch.matmul(hs, W["q"][li].t()).view(B, T, H, Dh).transpose(1, 2)
        k = torch.matmul(hs, W["k"][li].t()).view(B, T, H, Dh).transpose(1, 2)
        v = torch.matmul(hs, W["v"][li].t()).view(B, T, H, Dh).transpose(1, 2)
        q = _apply_rope_at(q, cos_rows, sin_rows, half)
        k = _apply_rope_at(k, cos_rows, sin_rows, half)
        # write new k/v into the static cache at the query positions
        kc = k_cache[li].index_copy(2, qpos_rows, k)       # (B,H,S,Dh)
        vc = v_cache[li].index_copy(2, qpos_rows, v)
        new_k.append(kc)
        new_v.append(vc)
        # attention over the full static window, causal-masked to filled positions
        att = torch.matmul(q, kc.transpose(2, 3)) / math.sqrt(Dh)   # (B,H,T,S)
        qpos_col = qpos_rows.unsqueeze(1)                  # (T,1)
        causal = key_pos.unsqueeze(0) <= qpos_col          # (T,S) bool
        mask = causal[None, None]                          # (1,1,T,S)
        att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, vc)                        # (B,H,T,Dh)
        out = out.transpose(1, 2).reshape(B, T, D)
        hidden = residual + torch.matmul(out, W["o"][li].t())
        # SwiGLU MLP
        residual = hidden
        hs = _rms_norm(hidden, W["n2"][li], eps)
        g = torch.matmul(hs, W["g"][li].t())
        u = torch.matmul(hs, W["u"][li].t())
        hidden = residual + torch.matmul(torch.nn.functional.silu(g) * u,
                                         W["dn"][li].t())
    k_cache = torch.stack(new_k, 0)
    v_cache = torch.stack(new_v, 0)
    return hidden, k_cache, v_cache


class StaticKVDecodeWrapper(nn.Module):
    """forward(ids) -> out_toks (B,K).

    Prefill the prompt `ids` through all blocks (filling the static KV cache and
    producing the first decode token), then run K manual decode steps as a single
    torch.while_loop. The manual decode step reproduces small_llama's Block exactly
    (RMSNorm -> QKV -> RoPE -> static-cache write -> full-causal attention -> o_proj
    -> SwiGLU MLP), so it is numerically identical to a real K-step eager decode.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.blocks = model.blocks
        self.norm = model.norm
        self.emb = model.emb
        self.lm = model.lm
        attn0 = self.blocks[0].attn
        self.n_heads = attn0.h
        self.head_dim = attn0.dh
        self.n_layers = len(self.blocks)
        self.eps = self.blocks[0].n1.eps
        self.K = K

    def _pack_weights(self):
        """Extract every parameter into plain-tensor lists so the loop body
        closes over TENSORS, not nn.Modules (dynamo cannot track an nn.Module
        across a while_loop body). Returns a dict of per-layer weight lists."""
        w = {k: [] for k in
             ("n1", "q", "k", "v", "o", "n2", "g", "u", "dn")}
        for blk in self.blocks:
            w["n1"].append(blk.n1.w)
            w["q"].append(blk.attn.q.weight)
            w["k"].append(blk.attn.k.weight)
            w["v"].append(blk.attn.v.weight)
            w["o"].append(blk.attn.o.weight)
            w["n2"].append(blk.n2.w)
            w["g"].append(blk.mlp.g.weight)
            w["u"].append(blk.mlp.u.weight)
            w["dn"].append(blk.mlp.dn.weight)
        return w

    def forward(self, ids):
        B, P = ids.shape
        device = ids.device
        H, Dh, L = self.n_heads, self.head_dim, self.n_layers
        S = P + self.K

        # --- hoisted invariants (rule 1 + 4): closed over by the loop body ---
        cos_tab, sin_tab = _build_rope_tables(S, Dh)          # (S,Dh) each
        key_pos = torch.arange(S, device=device)              # (S,)
        P_t = torch.tensor(P, dtype=torch.long, device=device)
        W = self._pack_weights()                              # plain tensors
        emb_w = self.emb.weight                               # (V,D)
        norm_w = self.norm.w                                  # (D,)
        lm_w = self.lm.weight                                 # (V,D)
        eps = self.eps
        n_layers, n_heads, head_dim = L, H, Dh                # plain ints (closed over)

        k_cache = torch.zeros(L, B, H, S, Dh, device=device)
        v_cache = torch.zeros(L, B, H, S, Dh, device=device)

        # --- prefill: run the prompt, fill cache[0:P], get the first token ---
        ppos = torch.arange(P, device=device)                 # (P,)
        hidden = torch.nn.functional.embedding(ids, emb_w)    # (B,P,D)
        hidden, k_cache, v_cache = _layers_pass(
            hidden, k_cache, v_cache, ppos, cos_tab, sin_tab, key_pos,
            W, n_layers, n_heads, head_dim, eps
        )
        last = _rms_norm(hidden[:, -1:, :], norm_w, eps)      # (B,1,D)
        logits = torch.matmul(last, lm_w.t())                 # (B,1,V)
        first_tok = logits[:, -1, :].argmax(-1, keepdim=True) # (B,1)

        # --- carried state ---
        Kc = self.K                                           # plain int (closed over)
        i0 = torch.zeros((), dtype=torch.long, device=device)
        out_toks0 = torch.zeros(B, Kc, dtype=torch.long, device=device)

        def cond(i, cur_tok, out_toks, k_cache, v_cache):
            return i < Kc

        def body(i, cur_tok, out_toks, k_cache, v_cache):
            out_toks = out_toks.index_copy(1, i.unsqueeze(0), cur_tok)
            pos = P_t + i
            qpos_rows = pos.unsqueeze(0)                       # (1,) absolute pos
            emb = torch.nn.functional.embedding(cur_tok, emb_w)  # (B,1,D)
            hidden, k_cache, v_cache = _layers_pass(
                emb, k_cache, v_cache, qpos_rows, cos_tab, sin_tab, key_pos,
                W, n_layers, n_heads, head_dim, eps
            )
            last = _rms_norm(hidden, norm_w, eps)              # (B,1,D)
            logits = torch.matmul(last, lm_w.t())
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            return (i + 1, nxt, out_toks, k_cache, v_cache)

        i, cur_tok, out_toks, k_cache, v_cache = while_loop(
            cond, body, (i0, first_tok, out_toks0, k_cache, v_cache)
        )
        return out_toks


def build():
    m, inp = get_model_and_inputs()
    wrapper = StaticKVDecodeWrapper(m).eval()
    return wrapper, inp, m


def ref_unrolled_decode(model, ids, k=K):
    """Reference: a plain python K-step greedy decode reusing the model's own
    forward over the GROWING sequence (recomputes full attention each step).
    This is the ground truth the while_loop must match bit-exactly."""
    with torch.no_grad():
        seq = ids
        out = []
        # first token from the prompt
        logits = model(seq)
        tok = logits[:, -1, :].argmax(-1, keepdim=True)
        for _ in range(k):
            out.append(tok)
            seq = torch.cat([seq, tok], dim=1)
            logits = model(seq)
            tok = logits[:, -1, :].argmax(-1, keepdim=True)
        return torch.cat(out, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    wrapper, inputs, model = build()
    ids = inputs[0]
    print("ids shape:", ids.shape, "K =", K)

    with torch.no_grad():
        out = wrapper(*inputs)
    print("while_loop decode out_toks:", out)

    ref = ref_unrolled_decode(model, ids)
    print("eager unrolled decode     :", ref)
    print("NUMERIC MATCH:", bool(torch.equal(out, ref)))

    print("\n=== torch.export.export(strict=False) ===")
    ep = torch.export.export(wrapper, inputs, strict=False)
    txt = str(ep.graph_module.graph)
    n_wl = sum(1 for n in ep.graph_module.graph.nodes
               if "while_loop" in str(getattr(n, "target", "")))
    print("while_loop call_function nodes:", n_wl, "(==1 means NOT unrolled)")

    print("\n=== m2m.convert ===")
    import m2m
    res = m2m.convert(wrapper, inputs, backend="fx_importer",
                      quantization=None, level="linalg-on-tensors")
    print("ok:", res.ok, "path_taken:", getattr(res, "path_taken", "?"))
    mt = res.mlir_text or ""
    print("scf.for count        :", mt.count("scf.for"))
    print("while_loop prov count:", mt.count('prov.op = "while_loop"'))
    print("mlir len:", len(mt))
