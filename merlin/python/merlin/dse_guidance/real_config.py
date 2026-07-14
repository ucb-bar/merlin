"""Real-config (deployment) magnitudes + KV sizing  (P21 S2 + S3).

The recaptures are small/random configs (e.g. openVLA's LM is reduced to 2 layers /
hidden 128). But MACs and bytes are determined by SHAPES, not weight VALUES, and
transformer stacks are layer-identical — so the deployment-real magnitude is EXACT
as an analytical composition from the published/extracted config:

    total = embed + lm_head + per_layer(stack) x n_layers(stack)   (summed over stacks)

This module holds the real deployment geometry for the loop-preserving workloads
(pi0.5 / smolVLA extracted from the instantiated model config; openVLA = Llama-2-7B
published) and computes, with NO dependence on weight values:

  * S2 — per-layer & full-depth GEMM weight bytes + GEMM MACs-per-token, at real depth.
  * S3 — KV-cache bytes = 2 * kv_heads * head_dim * seq * n_layers * dtype, per the
         decode stack; cross-checked against the IR-recovered KV iter_arg shape.

Evidence label: recovered_from_model_config (config-exact; weight values irrelevant).
No timing / speedup / cycles — structural magnitudes only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_DTYPE_BYTES = {"f32": 4, "bf16": 2, "fp16": 2, "int8": 1, "int4": 0.5}


@dataclass
class Stack:
    """One transformer stack (identical-layer); role in the runtime topology."""
    name: str
    n_layers: int
    hidden: int
    interm: int
    heads: int
    kv_heads: int
    head_dim: int
    role: str                     # prefix_once | repeated_head | decode_lm
    carries_decode_kv: bool = False
    # For NON-standard layers (e.g. a DiT block = self-attn + cross-attn + FFN + adaLN) where the
    # standard q/k/v/o+gate/up/down formula does not apply, set raw_layer_gemm_params to the per-layer
    # GEMM param count READ from the real model's nn.Linear shapes (not a formula/guess). The standard
    # path is used when this is None.
    raw_layer_gemm_params: int | None = None

    # ---- per-LAYER GEMM parameter counts (== MACs per token for that GEMM) ----
    def gemm_params(self) -> dict[str, int]:
        if self.raw_layer_gemm_params is not None:
            return {"dit_layer_gemms": self.raw_layer_gemm_params}   # read from real Linear shapes
        qd = self.heads * self.head_dim
        kvd = self.kv_heads * self.head_dim
        return {
            "q_proj": self.hidden * qd,
            "k_proj": self.hidden * kvd,
            "v_proj": self.hidden * kvd,
            "o_proj": qd * self.hidden,
            "gate_proj": self.hidden * self.interm,
            "up_proj": self.hidden * self.interm,
            "down_proj": self.interm * self.hidden,
        }

    def layer_params(self) -> int:
        return sum(self.gemm_params().values())


@dataclass
class RealGeometry:
    workload: str
    source: str                   # provenance of the numbers
    stacks: list[Stack]
    vocab: int
    embed_hidden: int
    decode_seq: int               # S used for KV sizing (prompt + decode tokens)
    K: int
    tied_embeddings: bool = True
    note: str = ""

    def embed_params(self) -> int:
        e = self.vocab * self.embed_hidden
        return e if self.tied_embeddings else 2 * e

    def total_layer_params(self) -> int:
        return sum(s.layer_params() * s.n_layers for s in self.stacks)

    def total_params(self) -> int:
        return self.embed_params() + self.total_layer_params()

    def decode_stack(self) -> Stack | None:
        for s in self.stacks:
            if s.carries_decode_kv:
                return s
        return None

    def kv_cache_elems(self) -> int | None:
        s = self.decode_stack()
        if s is None:
            return None
        # K and V, per kv-head, per layer, over the sequence
        return 2 * s.kv_heads * s.head_dim * self.decode_seq * s.n_layers

    def kv_cache_bytes(self, dtype: str = "bf16") -> int | None:
        e = self.kv_cache_elems()
        return int(e * _DTYPE_BYTES[dtype]) if e is not None else None


# Deployment geometry. pi05/smolvla extracted from the instantiated model config
# (this session); openvla LM = published Llama-2-7B (the recapture reduces it to 2
# layers / hidden 128 — the deployment magnitude is the composition below).
REAL_GEOMETRY: dict[str, RealGeometry] = {
    "pi05": RealGeometry(
        "pi05", "extracted from Pi0Config(pi05=True) instantiated model",
        stacks=[
            Stack("paligemma_lm_gemma_2b", 18, 2048, 16384, 8, 1, 256, "prefix_once",
                  carries_decode_kv=True),    # prefix KV reused (invariant) by the expert
            Stack("action_expert_gemma_300m", 18, 1024, 4096, 8, 1, 256, "repeated_head"),
        ],
        vocab=257152, embed_hidden=2048, decode_seq=200, K=10,
        note="flow-matching: expert runs K=10x reading the prefix KV (invariant)."),
    "smolvla": RealGeometry(
        "smolvla", "extracted from SmolVLA instantiated model config",
        stacks=[
            Stack("vlm_vision", 12, 768, 3072, 12, 12, 64, "prefix_once"),
            Stack("vlm_text_smollm2", 32, 960, 2560, 15, 5, 64, "prefix_once",
                  carries_decode_kv=True),
            Stack("action_expert", 16, 720, 2048, 15, 5, 64, "repeated_head"),
        ],
        vocab=49280, embed_hidden=960, decode_seq=50, K=10,
        note="flow-matching: action expert runs K=10x; VLM prefix once."),
    "openvla": RealGeometry(
        "openvla", "published Llama-2-7B LM (recapture reduces to 2L/128)",
        stacks=[
            Stack("llama2_7b_lm", 32, 4096, 11008, 32, 32, 128, "decode_lm",
                  carries_decode_kv=True),
        ],
        vocab=32064, embed_hidden=4096, decode_seq=263, K=7,
        tied_embeddings=False,
        note="autoregressive: 7 action tokens decoded; prompt ~256 vision+text tokens."),
    # --- P22 GAP-A: standard decoder-LLM stacks, geometry extracted from the real config object
    # (no guessed fields). Only models the standard q/k/v/o+gate/up/down Stack represents exactly are
    # added; DiT/diffusion (rdt/rdt2/groot/xr0) and bitvla (real deployment config not sourceable) are
    # intentionally omitted rather than misrepresented. decode_seq = real max_position_embeddings.
    "tiny_llama": RealGeometry(
        "tiny_llama", "extracted from AutoConfig TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        stacks=[Stack("tinyllama_lm", 22, 2048, 5632, 32, 4, 64, "decode_lm", carries_decode_kv=True)],
        vocab=32000, embed_hidden=2048, decode_seq=2048, K=7, tied_embeddings=False,
        note="TinyLlama-1.1B decoder; decode_seq = max_position_embeddings (2048)."),
    "molmoact": RealGeometry(
        "molmoact", "extracted from MolmoActLlmConfig() defaults",
        stacks=[Stack("molmoact_lm", 48, 3584, 18944, 28, 4, 128, "decode_lm", carries_decode_kv=True)],
        vocab=152064, embed_hidden=3584, decode_seq=4096, K=8, tied_embeddings=False,
        note="MolmoAct LLM; decode_seq = max_position_embeddings (4096)."),
    "small_llama": RealGeometry(
        "small_llama", "synthetic toy config (fully known; not a deployment-scale model)",
        stacks=[Stack("small_llama_lm", 2, 128, 344, 4, 4, 32, "decode_lm", carries_decode_kv=True)],
        vocab=256, embed_hidden=128, decode_seq=64, K=7, tied_embeddings=False,
        note="SYNTHETIC toy decoder (random init); included for completeness, not deployment-real."),
    # P24: the one DiT/diffusion model whose per-layer config (incl FFN) is fully sourced from the real
    # config (RDT2 post_train.yaml: hidden 1024, multiple_of 256 -> SwiGLU 2816; depth 14). Per-layer
    # GEMM params READ from the real model's nn.Linear shapes (self+cross-attn + SwiGLU FFN + adaLN);
    # 33,816,576/layer. Action-space head -> no vocab embedding; cross-attn KV is prefix-invariant (not
    # a carried decode cache). rdt / groot_n1d7 / xr0 are OMITTED from deployment magnitudes: their real
    # FFN dim is not published in a composable config (the loaders use a reduced/default ratio) -> would
    # require a guess (see threats_to_validity T1/T9). bitvla omitted (real config not in repo).
    "rdt2": RealGeometry(
        "rdt2", "RDT2 post_train.yaml (hidden 1024, multiple_of 256, depth 14); per-layer GEMM params "
                "read from the real instantiated nn.Linear shapes (no formula guess)",
        stacks=[Stack("rdt2_dit", 14, 1024, 2816, 8, 4, 128, "repeated_head",
                      carries_decode_kv=False, raw_layer_gemm_params=33_816_576)],
        vocab=0, embed_hidden=0, decode_seq=24, K=5, tied_embeddings=True,
        note="DiT flow-matching action head (self+cross-attn+SwiGLU+adaLN); action-space (no vocab)."),
}

_MAGNITUDE_COLS = ["workload", "source", "n_stacks", "total_layers", "per_layer_gemm_macs_top_stack",
                   "total_gemm_params", "weight_bytes_bf16", "weight_bytes_int8",
                   "gemm_macs_per_token", "kv_cache_bytes_bf16", "kv_seq", "K", "evidence"]


def _top_stack(g: RealGeometry) -> Stack:
    return max(g.stacks, key=lambda s: s.layer_params() * s.n_layers)


def magnitude_rows() -> list[dict]:
    rows = []
    for w, g in sorted(REAL_GEOMETRY.items()):
        top = _top_stack(g)
        total_layers = sum(s.n_layers for s in g.stacks)
        params = g.total_params()
        # GEMM MACs per decoded/denoised token = sum over the repeated/decode stacks of
        # (per-layer params x n_layers) — each weight is one MAC per token.
        per_tok = sum(s.layer_params() * s.n_layers for s in g.stacks
                      if s.role in ("repeated_head", "decode_lm"))
        rows.append({
            "workload": w, "source": g.source, "n_stacks": len(g.stacks),
            "total_layers": total_layers,
            "per_layer_gemm_macs_top_stack": top.layer_params(),
            "total_gemm_params": params,
            "weight_bytes_bf16": int(params * 2),
            "weight_bytes_int8": int(params * 1),
            "gemm_macs_per_token": per_tok,
            "kv_cache_bytes_bf16": g.kv_cache_bytes("bf16"),
            "kv_seq": g.decode_seq, "K": g.K,
            "evidence": "recovered_from_model_config",
        })
    return rows


def magnitudes_csv() -> str:
    from merlin.dse_guidance.corpus import _csv  # reuse the shared CSV writer
    return _csv(magnitude_rows(), _MAGNITUDE_COLS)


# ----------------------------------------------------------------- S3: KV sizing
_KV_COLS = ["workload", "decode_stack", "n_kv_heads", "head_dim", "seq", "n_layers",
            "kv_bytes_f32", "kv_bytes_bf16", "kv_bytes_int8", "loop_carried_in_ir",
            "ir_formula_check", "evidence"]


def kv_sizing_rows(loop_dir=None) -> list[dict]:
    """Per-workload KV-cache capacity from real config geometry. Where a loop-preserving
    capture exists, also report that the KV is an IR-recovered loop-carried iter_arg and
    cross-check the byte formula (2*kv_heads*head_dim*seq*n_layers*dtype) against the
    captured KV iter_arg shape (validates the formula on the small config, then applies
    it at deployment scale)."""
    rows = []
    lr_by_w = {}
    from pathlib import Path

    from merlin.dse_guidance.loop_recovery import recover_loop

    def _mp(w):
        # Explicit root (tests/temp dirs) wins; otherwise resolve per-workload through the corpus
        # accessor so the out/artifacts/recaptures/ overflow models (pi05/smolvla/groot) are included
        # — a single committed root would silently drop them.
        if loop_dir is not None:
            return Path(loop_dir) / w / "model.mlir"
        from merlin.dse_guidance.corpus import _recap_dir_in
        return _recap_dir_in(w, "recaptures_loop") / "model.mlir"

    for w in REAL_GEOMETRY:
        mp = _mp(w)
        if mp.is_file():
            lr = recover_loop(mp, w)
            if lr.present:
                lr_by_w[w] = lr
    for w, g in sorted(REAL_GEOMETRY.items()):
        s = g.decode_stack()
        if s is None:
            continue
        elems = g.kv_cache_elems()
        lr = lr_by_w.get(w)
        # IR cross-check: recompute the CAPTURED KV bytes from the recovered iter_arg
        # shapes and confirm the same 2*prod(shape)*dtype formula reproduces them.
        check = "n/a (no loop capture)"
        carried = False
        if lr is not None:
            kvs = [c for c in lr.carried_state if c.role == "kv_cache"]
            carried = bool(kvs)
            if kvs:
                recomputed = sum(int(_prod(c.shape)) * _DTYPE_BYTES.get(c.dtype, 4) for c in kvs)
                check = ("formula matches IR iter_arg "
                         f"({recomputed}=={lr.kv_cache_bytes})") if recomputed == lr.kv_cache_bytes \
                    else f"MISMATCH ({recomputed} vs {lr.kv_cache_bytes})"
        rows.append({
            "workload": w, "decode_stack": s.name, "n_kv_heads": s.kv_heads,
            "head_dim": s.head_dim, "seq": g.decode_seq, "n_layers": s.n_layers,
            "kv_bytes_f32": int(elems * 4), "kv_bytes_bf16": int(elems * 2),
            "kv_bytes_int8": int(elems * 1),
            "loop_carried_in_ir": carried, "ir_formula_check": check,
            "evidence": "recovered_from_model_config" + ("+recovered_from_ir" if carried else ""),
        })
    return rows


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= max(int(x), 1)
    return n


def kv_sizing_csv(loop_dir=None) -> str:
    from merlin.dse_guidance.corpus import _csv
    return _csv(kv_sizing_rows(loop_dir), _KV_COLS)
