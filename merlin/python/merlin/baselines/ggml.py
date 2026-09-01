"""ggml / llama.cpp baseline arm — build llama.cpp with RVV and run OUR LLM subset on the K1.

ggml/llama.cpp is the natural fit for the LLM + BitNet-ternary subset of the corpus
(``tiny_llama``, ``small_llama``, ``bitvla``). Its own CPU backend ships hand-written RVV kernels
(``ggml_gemm_q4_K_16x1_q8_K``, ``ggml_vec_dot_*``, tiled flash-attention) for riscv64; we cross-compile
it for ``rv64gcv`` with the SpacemiT GCC toolchain (the toolchain file llama.cpp itself ships,
``cmake/riscv64-spacemit-linux-gnu-gcc.cmake``), push the binaries to the board, and run
``llama-bench`` on real silicon. The RVV vector kernels live in ``libggml-cpu.so``; we
:func:`rvv_audit.audit_binary` it to record which ops are genuinely vectorized vs scalar.

Pipeline (per model)::

    HF checkpoint / captured config
      -> convert_hf_to_gguf.py            : GGUF (f16), optionally -> llama-quantize (q4_K / tq)
      -> cross-compiled llama.cpp (rv64gcv, SpacemiT gcc)  [the RVV artifact: libggml-cpu.so]
      -> rvv_audit.audit_binary(libggml-cpu.so)  : mechanical RVV%/scalar-fallback honesty
      -> on-board (board_lock): llama-bench -> MERLIN_E2E / MERLIN_REGION -> profile
      -> correctness: UNCOMPARABLE to our golden.npy (see below) -> cos=None + explicit gap note

Correctness gating — the honest limitation
-------------------------------------------
Our capture bundles are NOT the real HF checkpoints: ``tiny_llama`` is the real Llama arch but
**truncated to 2 layers with random init** (``M2M_LLAMA_LAYERS=2``), ``small_llama`` is a bespoke
toy transformer (vocab 256, random init — no GGUF converter exists for it), and ``bitvla`` is a small
random BitNet+SigLIP config whose capture unit is a forward on ``inputs_embeds`` (not token ids).
llama.cpp/ggml, by contrast, ingests a REAL GGUF checkpoint and runs a full token-in / logits-out
model with its own tokenizer. There is therefore NO way to reproduce our exact captured forward
(2-layer random weights, seeded random-token input, or an embeds-in forward) through ggml, and
``cos``/``rel`` vs our ``golden.npy`` is genuinely **not comparable**. Per ``not_run_is_not_pass`` we
record ``cos=None`` with a clear ``notes``/``gap`` explaining the correctness-gating limitation rather
than claim a pass we cannot verify. The arm's honest value is the *RVV-coverage audit + real E2E
latency on real K1 silicon* for a genuinely-vectorized foreign LLM runtime; whole-model output
fidelity is out of reach for this comparison and is labeled as such.

VLA models — precise per-model scope boundary
---------------------------------------------
The VLA models are recorded as explicit ``not_built`` gaps, but each with a PRECISE reason
(``_VLA_SCOPE_REASON``) that honestly separates *backbone-arch support* from *captured-forward
reproducibility*:

* ``openvla`` — LLM backbone IS Llama-2 (a supported ggml arch), but the captured unit is the
  *multimodal* ``forward(input_ids, pixel_values)->logits`` (DINOv2+SigLIP ViT + projector + image-
  token splice feed the LM); llama.cpp has no vision/projector/splice path, so the captured forward
  is not reproducible (only a text-only LM sub-forward would be — not the captured unit).
* ``molmoact`` — the captured unit IS a clean causal-LM ``input_ids->logits`` forward with a Qwen2-
  style backbone (a supported family), but it is a custom ``MolmoActForCausalLM`` with no
  ``convert_hf_to_gguf.py`` architecture entry (and random-init weights → no correlated golden).
* ``rdt``/``rdt2``/``xr0``/``pi05``/``smolvla``/``groot_n1d7`` — NOT causal LMs at all: diffusion /
  flow-matching action heads taking ``noisy_action``/``x``/``state``/``noise``/cached-``kv_*`` inputs
  and emitting small action tensors, with no token-in path and no vocab-logits output.

``bitvla`` sits between: its LLM backbone is a genuine ``BitNetForCausalLM`` (a supported ggml arch,
whose ternary TQ RVV kernels are ggml's most-vectorized path), but its captured unit is an
``inputs_embeds`` forward with bi-directional attention returning hidden states — none of which
llama.cpp's token-in / causal / vocab-logits API can reproduce (see the ``bitvla`` branch below).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from merlin.baselines import bundle as _bundle
from merlin.baselines import k1_exec, profile, rvv_audit
from merlin.baselines.contract import BaselineResult, RegionProfile, ScalarFallback
from merlin.common import artifacts
from merlin.common.artifacts import cache_dir
from merlin.common.paths import build_dir, repo_root
from merlin.mining import k1

FRAMEWORK = "ggml"

# --- llama.cpp build layout (build tree gitignored; built by this arm) --------------------------
_BUILD_ROOT = build_dir() / "baselines" / "ggml"
_LLAMA_SRC = repo_root() / "third_party" / "baselines" / "llama.cpp"
_GGUF_DIR = _BUILD_ROOT / "gguf"

# The rv64gcv march llama.cpp's own SpacemiT toolchain file selects (VLEN=256 K1 X60). Zfh/Zvfh are
# the K1's half-precision vector extensions; Zicbop = cache-block prefetch. rv64gcv is the RVV core.
GGML_MARCH = "rv64gcv_zfh_zvfh_zicbop_zihintpause"

# The ggml CPU backend variant .so that actually carries the RVV compute kernels (gemm/gemv/vec_dot/
# flash-attn). This — not the llama-bench driver ELF — is the RVV artifact we audit.
_GGML_CPU_SO_GLOBS = ("libggml-cpu.so*", "libggml-cpu-*.so*")

# ggml compute-kernel symbol prefixes — the "did the WORK vectorize" denominator, distinct from the
# whole-.so overall (which is diluted by C++ glue / plt / metadata that was never vectorizable).
_GGML_KERNEL_TOKENS = ("ggml_gemm", "ggml_gemv", "ggml_vec_dot", "ggml_compute_forward",
                       "ggml_vec_", "quantize_row", "dequantize_row")


def llama_build_dir() -> Path:
    """llama.cpp CMake build tree. Override with ``MERLIN_GGML_BUILD``."""
    v = os.environ.get("MERLIN_GGML_BUILD")
    return Path(v) if v else _BUILD_ROOT / "build"


def _bin(name: str) -> Path | None:
    p = llama_build_dir() / "bin" / name
    return p if p.is_file() else None


def llama_bench() -> Path | None:
    return _bin("llama-bench")


def llama_quantize() -> Path | None:
    return _bin("llama-quantize")


def ggml_cpu_so() -> Path | None:
    """The cross-built ggml CPU-backend shared object (holds the RVV kernels), or None."""
    bindir = llama_build_dir() / "bin"
    if not bindir.is_dir():
        return None
    # Prefer the concrete versioned file (libggml-cpu.so.0.15.3) over the symlinks.
    cands: list[Path] = []
    for g in _GGML_CPU_SO_GLOBS:
        cands += [p for p in bindir.glob(g) if not p.is_symlink()]
    if not cands:
        for g in _GGML_CPU_SO_GLOBS:
            cands += list(bindir.glob(g))
    return sorted(cands, key=lambda p: len(p.name))[-1] if cands else None


def ggml_available() -> bool:
    """True iff llama-bench + the ggml CPU RVV .so are cross-built for rv64gcv."""
    return llama_bench() is not None and ggml_cpu_so() is not None


def ggml_commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(_LLAMA_SRC), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


class GgmlError(RuntimeError):
    pass


# --- logits dumper (the correctness gate) -------------------------------------------------------

# A tiny rv64gcv helper: load a GGUF, decode an EXPLICIT list of token ids (no tokenizer, no BOS
# injection — the bundle's seeded input_ids verbatim), and dump the full [seq, n_vocab] logits as
# raw float32. This is what makes ggml's tiny_llama correctness COMPARABLE to our golden.npy: same
# real TinyLlama-1.1B checkpoint, same input_ids, so cos(logits, golden) is meaningful (not None).
_MERLIN_LOGITS_CPP = r'''// generated by merlin.baselines.ggml — do not edit. Dumps GGUF logits for explicit token ids.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.gguf> <out.f32> <id0> [id1 ...]\n", argv[0]); return 2; }
    std::vector<llama_token> ids;
    for (int i = 3; i < argc; ++i) ids.push_back((llama_token) atoi(argv[i]));
    const int seq = (int) ids.size();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "FAIL load %s\n", argv[1]); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = seq + 8; cp.n_batch = seq + 8; cp.n_threads = 8; cp.n_threads_batch = 8;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "FAIL ctx\n"); return 1; }
    llama_batch batch = llama_batch_init(seq, 0, 1);
    for (int i = 0; i < seq; ++i) {
        batch.token[i] = ids[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0; batch.logits[i] = 1;
    }
    batch.n_tokens = seq;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL decode\n"); return 1; }
    FILE * f = fopen(argv[2], "wb");
    if (!f) { fprintf(stderr, "FAIL out %s\n", argv[2]); return 1; }
    int32_t hv = n_vocab, hs = seq; fwrite(&hv, 4, 1, f); fwrite(&hs, 4, 1, f);
    for (int i = 0; i < seq; ++i) {
        const float * lg = llama_get_logits_ith(ctx, i);
        if (!lg) { fprintf(stderr, "FAIL logits %d\n", i); fclose(f); return 1; }
        fwrite(lg, sizeof(float), (size_t) n_vocab, f);
    }
    fclose(f);
    fprintf(stdout, "OK n_vocab=%d seq=%d\n", n_vocab, seq);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
'''


def logits_dumper() -> Path | None:
    """Path to the built rv64gcv logits dumper, or None if not yet built."""
    p = _BUILD_ROOT / "merlin_logits"
    return p if p.is_file() else None


def build_logits_dumper() -> Path:
    """Cross-compile the logits dumper against the cross-built llama.cpp libs (idempotent)."""
    out = _BUILD_ROOT / "merlin_logits"
    src = _BUILD_ROOT / "merlin_logits.cpp"
    if out.is_file():
        return out
    root = k1._toolchain_root()
    if root is None:
        raise GgmlError("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")
    gxx = root / "bin" / "riscv64-unknown-linux-gnu-g++"
    if not gxx.is_file():
        raise GgmlError(f"no riscv g++ at {gxx}")
    src.write_text(_MERLIN_LOGITS_CPP)
    bindir = llama_build_dir() / "bin"
    r = subprocess.run([str(gxx), "-std=c++17", "-O2", f"-march={GGML_MARCH}", "-mabi=lp64d",
                        f"-I{_LLAMA_SRC / 'include'}", f"-I{_LLAMA_SRC / 'ggml' / 'include'}",
                        str(src), f"-L{bindir}", f"-Wl,-rpath-link,{bindir}",
                        "-lllama", "-lggml", "-lggml-base", "-lggml-cpu", "-o", str(out)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not out.is_file():
        raise GgmlError(f"logits-dumper build failed: {r.stderr[-400:]}")
    return out


# --- model coverage -----------------------------------------------------------------------------

# The subset ggml can actually ingest as a causal LM, and how each maps to a GGUF.
#   tiny_llama -> real TinyLlama-1.1B HF checkpoint (standard Llama arch, GGUF-convertible).
#   small_llama -> bespoke random toy transformer: NO GGUF converter (not a HF arch).
#   bitvla -> small random BitNet+SigLIP; capture unit is an inputs_embeds forward: NOT GGUF-shaped.
LLM_SUBSET = ("tiny_llama", "small_llama", "bitvla")

# The HF checkpoint id / local snapshot each convertible model maps to. Only tiny_llama has a real
# public GGUF-convertible checkpoint; the others are documented gaps below.
_HF_CHECKPOINTS: dict[str, str] = {
    "tiny_llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# VLA-specific models: no runnable llama.cpp GGUF for the CAPTURED forward. Explicit gaps, never
# forced. The reasons are per-model and PRECISE about the scope boundary — specifically whether the
# LLM backbone is a supported ggml arch, and (if so) exactly why the captured unit still isn't a
# token-in / vocab-logits-out causal forward llama.cpp can reproduce.
VLA_OUT_OF_SCOPE = ("openvla", "rdt", "rdt2", "molmoact", "groot_n1d7", "xr0", "pi05", "smolvla")

_VLA_SCOPE_REASON: dict[str, str] = {
    # LLM backbone IS a supported arch, but the CAPTURED forward is multimodal / not token-in-only:
    "openvla": ("openvla's LLM backbone is Llama-2 (a supported ggml arch), but the captured unit is "
                "the MULTIMODAL forward(input_ids, pixel_values)->logits: a fused DINOv2+SigLIP ViT + "
                "MLP projector + image-token splice feed the LM, and the golden [1,200,32064] is that "
                "vision-conditioned output. llama.cpp has no vision tower / projector / image-splice "
                "path, so this forward is not reproducible (only a text-only LM sub-forward would be, "
                "and that is not the captured unit; weights are random-init)"),
    "molmoact": ("molmoact's captured unit IS a clean causal-LM input_ids->logits forward and its "
                 "backbone is Qwen2-style (a supported ggml arch family), but it is a custom "
                 "MolmoActForCausalLM (Molmo/OLMo lineage) with no convert_hf_to_gguf.py architecture "
                 "entry and non-standard module naming — no GGUF converter path exists, and the "
                 "weights are random-init so a hand-built GGUF would have no correlated golden"),
    # NOT causal LMs at all — diffusion / flow-matching action heads (no token-in, no vocab logits):
    "rdt":  ("rdt is an RDT diffusion action expert: the captured forward takes (x, freq, t, lang_c, "
             "img_c, lang_mask) and returns an action tensor [1,64,128] — a denoising step, not a "
             "token-in causal LM; no llama.cpp arch and no vocab-logits output"),
    "rdt2": ("rdt2 is an RDT2 flow-matching action head over cached cross-attention KV (inputs "
             "x,t,state_c,kv_0..27 -> action [1,24,20]) — a diffusion/flow step, not a causal LM; "
             "no llama.cpp arch / no token-in / no vocab logits"),
    "xr0":  ("xr0 is a diffusion action head (noisy_action,t,state,cos,sin,attn_mask,kv_0..31 -> "
             "[1,30,32]) — a denoising step over cached KV, not a token-in causal LM; no ggml arch"),
    "pi05": ("pi05 is a pi-0.5 flow-matching VLA denoiser (multi-image + lang_tokens + state + noise "
             "-> action [1,50,32]) — a diffusion step, not a token-in / vocab-logits causal LM; no "
             "ggml arch"),
    "smolvla": ("smolvla is a SmolVLA flow-matching denoiser (img + lang_tokens + state + noise -> "
                "action [1,50,32]) — a diffusion step, not a token-in causal LM; no ggml arch"),
    "groot_n1d7": ("groot_n1d7 is a GR00T-N1 action head over precomputed backbone_features (+ state, "
                   "actions, embodiment_id, timesteps -> [1,40,132]) — a flow-matching action model, "
                   "not a token-in causal LM; no ggml arch / no vocab logits"),
}

DEFAULT_MODELS = LLM_SUBSET + VLA_OUT_OF_SCOPE


def _hf_snapshot(model: str) -> Path | None:
    """Best-effort local HF snapshot dir for a convertible model (from the HF hub cache)."""
    repo = _HF_CHECKPOINTS.get(model)
    if not repo:
        return None
    org, name = repo.split("/", 1)
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    base = cache / f"models--{org}--{name}" / "snapshots"
    if not base.is_dir():
        return None
    snaps = sorted(base.glob("*/"))
    return snaps[-1] if snaps else None


# --- GGUF conversion (host-side) ----------------------------------------------------------------

def gguf_path(model: str, quant: str = "f16") -> Path:
    return _GGUF_DIR / f"{model}-{quant}.gguf"


def convert_to_gguf(model: str, *, outtype: str = "f16", timeout: int = 1200) -> Path:
    """Convert a real HF checkpoint to GGUF via llama.cpp's ``convert_hf_to_gguf.py``.

    Uses the model2MLIR venv python (it carries torch/transformers) with llama.cpp's bundled
    ``gguf-py`` on PYTHONPATH. Raises :class:`GgmlError` (never fabricates) if the model has no
    real GGUF-convertible checkpoint or the converter fails.
    """
    snap = _hf_snapshot(model)
    if snap is None:
        raise GgmlError(f"no local HF checkpoint for {model!r} to convert to GGUF "
                        f"(only {sorted(_HF_CHECKPOINTS)} are GGUF-convertible)")
    out = gguf_path(model, outtype)
    if out.is_file() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    py = os.environ.get("MERLIN_GGUF_PYTHON",
                        "/path/to/model2MLIR/.venv/bin/python")
    if not Path(py).is_file():
        raise GgmlError(f"GGUF-conversion python not found: {py} (set MERLIN_GGUF_PYTHON)")
    conv = _LLAMA_SRC / "convert_hf_to_gguf.py"
    env = dict(os.environ, PYTHONPATH=f"{_LLAMA_SRC / 'gguf-py'}:{os.environ.get('PYTHONPATH','')}")
    r = subprocess.run([py, str(conv), str(snap), "--outfile", str(out), "--outtype", outtype],
                       capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0 or not out.is_file():
        raise GgmlError(f"convert_hf_to_gguf failed for {model}: {r.stderr[-400:]}")
    return out


# --- small_llama: direct GGUF builder (no HF checkpoint / no convert_hf_to_gguf converter) --------
#
# small_llama is a bespoke random-init toy transformer (module names ``blocks.N.attn.{q,k,v,o}`` /
# ``mlp.{g,u,dn}`` / ``n1/n2/norm``, vocab 256, d=128, h=4, head_dim=32, ffn=344, 2 layers) — NOT a
# ``LlamaForCausalLM`` HF checkpoint, so ``convert_hf_to_gguf.py`` has no architecture for it and
# refuses. BUT its op surface is byte-for-byte a standard Llama block: RMSNorm (eps=1e-5) + separate
# no-bias Q/K/V/O attention + rotate-half NeoX RoPE (theta=1e4) + SwiGLU (silu) MLP + untied lm_head.
# Every one of those maps onto llama.cpp's ``llama`` arch, so we build the GGUF *directly* with
# llama.cpp's own ``gguf-py`` writer (the same library ``convert_hf_to_gguf.py`` uses) and run it as a
# real token-in / logits-out forward — giving small_llama a COMPARABLE cos vs the fp32 golden.
#
# The one subtlety: small_llama uses NeoX-style rotate-half RoPE (``rot=cat([-x2,x1])``) whereas the
# ``llama`` arch applies NORM interleaved-pair RoPE. We reconcile them with the *exact* HF-Llama Q/K
# weight permutation (rows (head,dh)->(head,2,dh/2)->(head,dh/2,2)); with that permutation, NORM rope
# on the permuted weights == NeoX rope on the originals. Verified numerically to cos=1.0 vs golden
# (host reproduction), so the GGUF forward reproduces the captured graph exactly (not_run_is_not_pass).

_SMALL_LLAMA_HP = dict(vocab=256, d=128, h=4, dh=32, ffn=344, layers=2, eps=1e-5, rope_theta=1e4)


def _permute_qk(w, n_head: int, dh: int):
    """HF-Llama Q/K permute so llama.cpp NORM rope == small_llama's NeoX rotate-half rope.

    ``w`` is a ``[n_head*dh, in]`` linear weight; per head, reshape rows ``(2, dh//2)`` and transpose
    to ``(dh//2, 2)``. This is the inverse of the convention gap between rotate-half and interleaved
    rope (identical to the permute in ``convert_hf_to_gguf`` / the original Llama HF conversion).
    """
    import numpy as np
    o, i = w.shape
    return (w.reshape(n_head, 2, dh // 2, i).swapaxes(1, 2).reshape(o, i)).astype(np.float32)


def _load_safetensors_f32(path: Path) -> dict:
    """Minimal float32 safetensors reader (no torch dep) for the small_llama fp32 bundle."""
    import json as _json
    import struct

    import numpy as np
    out: dict = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = _json.loads(f.read(n))
        base = f.tell()
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            if v["dtype"] != "F32":
                continue  # fp32 bundle only; quant handled on-board by llama-quantize
            s, e = v["data_offsets"]
            f.seek(base + s)
            out[k] = np.frombuffer(f.read(e - s), dtype=np.float32).reshape(v["shape"]).copy()
    return out


def build_small_llama_gguf() -> Path:
    """Build a GGUF for small_llama directly from its fp32 capture bundle (idempotent).

    Uses llama.cpp's bundled ``gguf-py`` (added to ``sys.path``) to emit a ``general.architecture=llama``
    GGUF with ``tokenizer.ggml.model="none"`` (dummy 256-token vocab — our logits dumper feeds explicit
    ids, no tokenizer needed) and the HF-permuted Q/K weights. Raises :class:`GgmlError` on any gap.
    """
    import sys

    import numpy as np
    out = gguf_path("small_llama", "f16")
    if out.is_file() and out.stat().st_size > 0:
        return out
    from merlin.baselines import bundle as _b
    bnd = _b.resolve("small_llama", "fp32")
    if not bnd.weights.is_file():
        raise GgmlError(f"small_llama fp32 capture weights not found at {bnd.weights}")
    W = _load_safetensors_f32(bnd.weights)
    hp = _SMALL_LLAMA_HP
    ggpy = str(_LLAMA_SRC / "gguf-py")
    if ggpy not in sys.path:
        sys.path.insert(0, ggpy)
    try:
        import gguf  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise GgmlError(f"cannot import gguf-py from {ggpy}: {e}") from e
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".gguf.tmp")
    w = gguf.GGUFWriter(str(tmp), "llama")
    w.add_name("small_llama")
    w.add_context_length(2048)
    w.add_embedding_length(hp["d"])
    w.add_block_count(hp["layers"])
    w.add_feed_forward_length(hp["ffn"])
    w.add_head_count(hp["h"])
    w.add_head_count_kv(hp["h"])
    w.add_rope_dimension_count(hp["dh"])
    w.add_rope_freq_base(hp["rope_theta"])
    w.add_layer_norm_rms_eps(hp["eps"])
    w.add_vocab_size(hp["vocab"])
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    # 'none' vocab: dummy token list; ids fed explicitly by merlin_logits (no tokenizer path taken).
    w.add_tokenizer_model("none")
    w.add_token_list([f"<{i}>" for i in range(hp["vocab"])])
    w.add_token_types([gguf.TokenType.NORMAL] * hp["vocab"])

    def T(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr.astype(np.float32)))

    T("token_embd.weight", W["emb.weight"])
    T("output_norm.weight", W["norm.w"])
    T("output.weight", W["lm.weight"])
    for L in range(hp["layers"]):
        p = f"blocks.{L}."
        b = f"blk.{L}."
        T(b + "attn_norm.weight", W[p + "n1.w"])
        T(b + "attn_q.weight", _permute_qk(W[p + "attn.q.weight"], hp["h"], hp["dh"]))
        T(b + "attn_k.weight", _permute_qk(W[p + "attn.k.weight"], hp["h"], hp["dh"]))
        T(b + "attn_v.weight", W[p + "attn.v.weight"])
        T(b + "attn_output.weight", W[p + "attn.o.weight"])
        T(b + "ffn_norm.weight", W[p + "n2.w"])
        T(b + "ffn_gate.weight", W[p + "mlp.g.weight"])
        T(b + "ffn_up.weight", W[p + "mlp.u.weight"])
        T(b + "ffn_down.weight", W[p + "mlp.dn.weight"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    tmp.replace(out)
    if not out.is_file() or out.stat().st_size == 0:
        raise GgmlError("small_llama GGUF write produced no file")
    return out


# --- RVV audit ----------------------------------------------------------------------------------

def _region_of_symbol(sym: str) -> str:
    """Map a ggml compute symbol to a REGIONS bucket by name."""
    s = sym.lower()
    if "gemm" in s or "gemv" in s or "vec_dot" in s or "mul_mat" in s:
        return "gemm"
    if "flash_attn" in s or "attn" in s or "softmax" in s:
        return "attention"
    if "norm" in s or "rms" in s or "rsqrt" in s:
        return "norm"
    if any(t in s for t in ("_add", "_mul", "gelu", "silu", "relu", "_exp", "quantize", "dequantize",
                            "rope", "_scale", "_cpy")):
        return "elementwise"
    return "other"


def _is_kernel(sym: str) -> bool:
    return any(t in sym for t in _GGML_KERNEL_TOKENS)


# The inner SIMD math kernels for the q4_K/q8_K path a q4_K_M model actually runs (this is the
# fairest "did the benchmarked compute vectorize" number — the q4_K_M GEMM/GEMV inner kernels).
_ACTIVE_QUANT_KERNELS = ("q4_K", "q8_K", "q4_k", "q8_k")

# Per-quant inner-kernel token sets, so we can report the RVV% of the SPECIFIC path a given GGUF
# runs. int8 (Q8_0) activations flow through the q8_0 + q8_K reduction dot-products; the K-quant
# weight paths (q4_K/q8_K) share the q8_K activation dot; the BitNet/ternary path is TQ1_0/TQ2_0.
_QUANT_KERNEL_TOKENS: dict[str, tuple[str, ...]] = {
    "q8_0": ("q8_0", "q8_K"),          # int8: Q8_0 weights + q8_K activation dot
    "q8_K": ("q8_K",),
    "q4_K": ("q4_K", "q8_K"),          # q4_K weights dot against q8_K-quantized activations
    "tq1_0": ("tq1_0",),               # BitNet 1.58-bit ternary (1.6875 bpw)
    "tq2_0": ("tq2_0",),               # BitNet ternary (2.0625 bpw)
}


def _quant_path_coverage(report, quant: str) -> float | None:
    """RVV% over the inner gemm/gemv/vec_dot kernels of a specific quant path (int8/ternary/etc.)."""
    toks = _QUANT_KERNEL_TOKENS.get(quant.lower().replace("_m", "").replace("_s", ""))
    if not toks:
        # normalize e.g. "q4_K_M" -> "q4_K", "Q8_0" -> "q8_0"
        base = quant.lower()
        for k in _QUANT_KERNEL_TOKENS:
            if base.startswith(k.lower()):
                toks = _QUANT_KERNEL_TOKENS[k]
                break
    if not toks:
        return None
    v = s = 0
    for name, sc in report.by_symbol.items():
        if name.endswith("@plt"):
            continue
        if any(t in name for t in ("ggml_gemm", "ggml_gemv", "ggml_vec_dot")) and any(t in name for t in toks):
            v += sc.vector
            s += sc.scalar_compute
    return (v / (v + s)) if (v + s) else None


@dataclass
class RvvAudit:
    coverage_overall: float | None = None          # whole-.so compute-insn coverage (diluted by glue)
    coverage_kernels: float | None = None          # coverage over ggml COMPUTE kernels only
    coverage_active_quant: float | None = None      # inner gemm/gemv/vec_dot for the q4_K/q8_K path
    coverage_int8: float | None = None              # Q8_0 (int8) inner-kernel path (q8_0 + q8_K)
    coverage_ternary: float | None = None           # BitNet TQ2_0 ternary inner-kernel path
    fallbacks: list[ScalarFallback] = None          # compute kernels that stayed fully scalar
    region_coverage: dict[str, float] = None        # per-REGIONS-bucket RVV coverage over kernels


def audit_cpu_so(so: Path) -> RvvAudit:
    """RVV-audit the ggml CPU backend .so. Reports whole-.so, compute-kernel, and active-quant cov.

    ggml's ``.so`` is mostly C++ dispatch/metadata/plt that was never vectorizable, so the whole-.so
    overall coverage understates the vectorization of the actual compute; we additionally compute a
    coverage over just the ``ggml_gemm/gemv/vec_dot/compute_forward`` kernels, and — most
    representatively — over the inner q4_K/q8_K GEMM/GEMV/vec_dot SIMD kernels a q4_K_M model
    actually executes (the "did the benchmarked math vectorize" number), plus per-region coverage.
    Scalar glue is never averaged away; every fully-scalar compute kernel is labeled.
    """
    report = rvv_audit.audit_binary(so)
    # kernel-only tallies
    kv = ks = 0
    aqv = aqs = 0          # active-quant (q4_K/q8_K) inner SIMD kernels
    region_v: dict[str, int] = {}
    region_s: dict[str, int] = {}
    fallbacks: list[ScalarFallback] = []
    for name, sc in sorted(report.by_symbol.items()):
        if name.endswith("@plt") or not _is_kernel(name):
            continue
        kv += sc.vector
        ks += sc.scalar_compute
        reg = _region_of_symbol(name)
        region_v[reg] = region_v.get(reg, 0) + sc.vector
        region_s[reg] = region_s.get(reg, 0) + sc.scalar_compute
        if (any(t in name for t in ("ggml_gemm", "ggml_gemv", "ggml_vec_dot"))
                and any(q in name for q in _ACTIVE_QUANT_KERNELS)):
            aqv += sc.vector
            aqs += sc.scalar_compute
        if sc.is_scalar_fallback:
            fallbacks.append(ScalarFallback(symbol=name[:80],
                                            reason="ggml compute kernel emitted fully scalar (no RVV)",
                                            region=reg))
    cov_kernels = (kv / (kv + ks)) if (kv + ks) else None
    cov_aq = (aqv / (aqv + aqs)) if (aqv + aqs) else None
    region_cov = {r: (region_v[r] / (region_v[r] + region_s.get(r, 0)))
                  for r in region_v if (region_v[r] + region_s.get(r, 0))}
    return RvvAudit(coverage_overall=report.coverage_overall, coverage_kernels=cov_kernels,
                    coverage_active_quant=cov_aq,
                    coverage_int8=_quant_path_coverage(report, "q8_0"),
                    coverage_ternary=_quant_path_coverage(report, "tq2_0"),
                    fallbacks=fallbacks, region_coverage=region_cov)


# --- board run ----------------------------------------------------------------------------------

_BOARD_DIR = "/root/ggml_bench"


def _ssh_opts() -> list[str]:
    return ["-i", k1_exec.K1_SSH_KEY, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]


def _board_sh(cmd: str, *, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *_ssh_opts(), k1_exec.K1_HOST, cmd],
                          capture_output=True, text=True, timeout=timeout)


def deploy_runtime() -> None:
    """Push the cross-built llama.cpp libs + tools to the board (idempotent). Fail-closed if down."""
    if not k1_exec.board_available():
        raise k1_exec.BoardUnavailable("K1 board unavailable for ggml runtime deploy")
    bindir = llama_build_dir() / "bin"
    _board_sh(f"mkdir -p {_BOARD_DIR}/lib", timeout=60)
    libs = [p for p in bindir.glob("*.so*")]
    tools = [bindir / t for t in ("llama-bench", "llama-quantize", "llama-completion")
             if (bindir / t).is_file()]
    for p in libs:
        subprocess.run(["scp", *_ssh_opts(), str(p), f"{k1_exec.K1_HOST}:{_BOARD_DIR}/lib/{p.name}"],
                       check=True, capture_output=True, timeout=300)
    for p in tools:
        subprocess.run(["scp", *_ssh_opts(), str(p), f"{k1_exec.K1_HOST}:{_BOARD_DIR}/{p.name}"],
                       check=True, capture_output=True, timeout=300)
    _board_sh(f"chmod +x {_BOARD_DIR}/llama-*", timeout=60)


def _parse_llama_bench(stdout: str) -> dict[str, float]:
    """Parse llama-bench's markdown/CSV-ish output for tokens/s (pp + tg) and any timing.

    llama-bench prints a table; we extract the ``t/s`` column per test. We record pp (prompt/prefill,
    GEMM-bound) and tg (token-gen, GEMV/attention-bound) throughput as the two comparable regions.
    """
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        low = line.lower()
        # Markdown row: | model | ... | test | t/s |   with test like "pp512" / "tg128".
        if "|" in line and ("pp" in low or "tg" in low) and "t/s" not in low:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) < 2:
                continue
            test = next((c for c in cells if c.startswith(("pp", "tg"))), None)
            # t/s is usually the last numeric-ish cell (may carry a ± stddev).
            val = None
            for c in reversed(cells):
                tok = c.split("±")[0].strip().replace(",", "")
                try:
                    val = float(tok)
                    break
                except ValueError:
                    continue
            if test and val is not None:
                out[test] = val
    return out


def run_on_board(gguf_remote: str, *, n_threads: int = 8, pp: int = 64, tg: int = 32,
                 timeout: int = 1200) -> tuple[dict, str]:
    """Run llama-bench on the board for a GGUF, timed with rdtime brackets. Returns (parsed, raw).

    We wrap llama-bench in a shell that reads rdtime before/after via /dev/... — but the board
    kernel traps userspace rdcycle, and rdtime needs an asm helper we don't have in shell. So we
    time with wall-clock (``date +%s%N``) for the E2E marker and derive tokens/s from llama-bench's
    own reporting (its internal timing is the authoritative throughput). Cycles stay estimated.
    """
    ld = f"LD_LIBRARY_PATH={_BOARD_DIR}/lib:/usr/lib/riscv64-linux-gnu"
    # -r 3 repetitions; single prompt+gen size so the table is compact.
    bench = (f"{ld} {_BOARD_DIR}/llama-bench -m {gguf_remote} -t {n_threads} "
             f"-p {pp} -n {tg} -r 3")
    wrapped = f"S=$(date +%s%N); {bench}; E=$(date +%s%N); echo MERLIN_WALL_NS=$((E-S))"
    proc = _board_sh(wrapped, timeout=timeout)
    raw = proc.stdout + "\n" + proc.stderr
    parsed = _parse_llama_bench(proc.stdout)
    wall_ns = None
    for line in raw.splitlines():
        if "MERLIN_WALL_NS=" in line:
            try:
                wall_ns = int(line.split("=", 1)[1].strip())
            except ValueError:
                pass
    return {"tps": parsed, "wall_ns": wall_ns, "rc": proc.returncode}, raw


# --- the runner ---------------------------------------------------------------------------------

# int8 first (Q8_0 = faithful int8, ggml's home turf), then the 4-bit K-quant. Both exercise the
# vectorized RVV integer dot-products (q8_K / q4_K); Q8_0 is the int8 comparison the coordinator asked
# for, Q4_K_M is the standard deployment quant.
DEFAULT_QUANTS: tuple[str, ...] = ("Q8_0", "q4_K_M")


def run_model(model: str, variant: str = "fp32", *, quant: str | None = None,
              quants: tuple[str, ...] | None = None,
              write: bool = True, run_board: bool | None = None) -> BaselineResult:
    """Run one model through the ggml arm end-to-end and return a BaselineResult.

    Benchmarks each quant in ``quants`` (default int8 ``Q8_0`` first, then ``q4_K_M``) on the board;
    the int8 path is the headline E2E. Re-runnable and fail-closed: builds/audits happen regardless
    of the board; the on-board llama-bench is the only board-gated step. Correctness vs our golden is
    UNCOMPARABLE (see module docstring) so ``cos`` stays None with an explicit note.
    """
    if quants is None:
        quants = (quant,) if quant else DEFAULT_QUANTS
    cos_thr, rel_thr = _bundle.tolerance(model)
    res = BaselineResult(framework=FRAMEWORK, model=model, variant=variant,
                         substrate="k1_spacemit", cos_threshold=cos_thr, rel_threshold=rel_thr,
                         march=GGML_MARCH, toolchain="llama.cpp+ggml(RVV) / spacemit-gcc-14.3",
                         framework_commit=ggml_commit(), timestamp=artifacts.utc_stamp())

    # WHICH bundle this measurement is on. ggml resolves its bundle LATER and per-variant (its
    # correctness reference may come from a different variant than the timed run -- see
    # `_correctness_bundle`), so record the cell's own resolve here rather than borrowing that one.
    # A ratio taken across two different bundles is not a speedup; see
    # compare.executorch_column.bundle_mismatch_reason.
    res.bundle_id = _bundle.resolve(model, variant).root.name

    # VLA models are out of ggml scope — explicit not_built gap with a PER-MODEL precise reason
    # (backbone-arch support vs captured-forward reproducibility), never forced.
    if model in VLA_OUT_OF_SCOPE:
        res.gap_reason = _VLA_SCOPE_REASON.get(
            model, f"{model}: VLA/diffusion/multimodal graph with no llama.cpp arch — out of ggml scope")
        return _finish(res, model, variant, write)

    if not ggml_available():
        res.gap_reason = ("llama.cpp not cross-built for rv64gcv under build/baselines/ggml/build "
                          "(need llama-bench + libggml-cpu.so; see AGENT.md)")
        return _finish(res, model, variant, write)

    # RVV audit of the ggml CPU backend .so — mechanical honesty, board-independent.
    so = ggml_cpu_so()
    try:
        rvv_audit.enforce_rvv_march(GGML_MARCH)
        aud = audit_cpu_so(so)
        # Headline coverage = the q4_K/q8_K inner GEMM/GEMV/vec_dot kernels the benchmarked model
        # actually runs (the fairest "did the benchmarked math vectorize"), falling back to the
        # broader kernel coverage if the active-quant kernels weren't found.
        res.rvv_coverage_overall = aud.coverage_active_quant or aud.coverage_kernels
        res.scalar_fallbacks = aud.fallbacks
        res.notes += (f" rvv_int8_cov(Q8_0:q8_0+q8_K)={aud.coverage_int8} "
                      f"rvv_ternary_cov(TQ2_0)={aud.coverage_ternary} "
                      f"rvv_active_quant_cov(q4_K/q8_K)={aud.coverage_active_quant} "
                      f"rvv_all_kernels_cov={aud.coverage_kernels} rvv_whole_so={aud.coverage_overall} "
                      f"region_cov={aud.region_coverage} so={so.name};")
    except Exception as e:  # noqa: BLE001
        res.notes += f" rvv-audit failed: {str(e)[:150]};"

    # GGUF — tiny_llama has a real HF-convertible checkpoint; small_llama we build DIRECTLY from its
    # capture bundle (its op surface IS a Llama block — see build_small_llama_gguf). Others are gaps.
    if model == "small_llama":
        try:
            gguf_f16 = build_small_llama_gguf()
        except GgmlError as e:
            res.gap_reason = f"small_llama GGUF build failed: {str(e)[:250]}"
            return _finish(res, model, variant, write)
        res.built = True
        res.notes += (f" gguf_f16={gguf_f16.name}({gguf_f16.stat().st_size//1024}KB, direct "
                      f"gguf-py llama-arch build from capture bundle, HF-permuted Q/K);")
        res.cos = None
        do_board = k1_exec.board_available() if run_board is None else run_board
        if do_board:
            try:
                _run_bench_on_board(res, model, gguf_f16, quants)
            except k1_exec.BoardUnavailable as e:
                res.gap_reason = res.gap_reason or f"K1 board run failed: {str(e)[:200]}"
            except Exception as e:  # noqa: BLE001
                res.gap_reason = res.gap_reason or f"K1 ggml bench error: {str(e)[:200]}"
        else:
            res.gap_reason = res.gap_reason or "K1 board unavailable (MERLIN_K1_HOST unset)"
        res.board_vlenb = k1_exec.board_vlenb()
        return _finish(res, model, variant, write)

    if model not in _HF_CHECKPOINTS:
        if model == "bitvla":
            # PRECISE scope boundary. bitvla's LLM backbone IS a genuine BitNet arch — the capture
            # weights carry `vla.language_model.model.layers.N.self_attn.{q,k,v,o}_proj`,
            # `mlp.{gate,up,down}_proj` AND BitNet's `attn_sub_norm`/`ffn_sub_norm` sub-norms, i.e.
            # exactly `BitNetForCausalLM`, which llama.cpp DOES support (LLM_ARCH_BITNET) and whose
            # ternary TQ2_0/TQ1_0 RVV kernels are ggml's MOST-vectorized path (~51%/39% in our audit).
            # But the CAPTURED FORWARD is not expressible through llama.cpp's token-in/logits-out
            # causal API, for three concrete reasons: (1) the capture unit is
            # `forward(inputs_embeds=[1,32,256])` — raw embeddings, not token ids — and llama.cpp has
            # NO public inputs_embeds injection path (llama_decode/merlin_logits take ids only);
            # (2) it runs with `use_bi_attn=True` (BI-directional attention over the assembled
            # multimodal prefix), whereas llama.cpp's bitnet is strictly CAUSAL; (3) the golden is
            # `[1,32,1024]` HIDDEN STATES, not vocab logits (1024 != the 256 vocab), so even the
            # output tensor llama.cpp exposes (logits) is not comparable. The weights are also
            # random-init (no pretrained correlation). So: BitNet ARCH supported by ggml, but THIS
            # embeds-in / bidirectional / hidden-states-out VLA forward is not reproducible.
            reason = ("bitvla's LLM backbone is a genuine BitNetForCausalLM (llama.cpp supports the "
                      "BitNet arch + ternary TQ RVV kernels), but the captured unit is an "
                      "inputs_embeds forward with bi-directional attention returning hidden states "
                      "[1,32,1024] — llama.cpp has no inputs_embeds injection path, is causal-only, "
                      "and exposes vocab logits (256) not hidden states, so this exact forward is "
                      "not reproducible through ggml (arch supported; captured forward is not)")
        else:
            reason = f"{model}: no GGUF-convertible checkpoint"
        # built stays False: we have the RVV artifact but no runnable GGUF for this model.
        res.gap_reason = reason
        return _finish(res, model, variant, write)

    try:
        gguf_f16 = convert_to_gguf(model, outtype="f16")
    except GgmlError as e:
        res.gap_reason = f"GGUF conversion failed: {str(e)[:250]}"
        return _finish(res, model, variant, write)
    res.built = True
    res.notes += (f" gguf_f16={gguf_f16.name}({gguf_f16.stat().st_size//(1024*1024)}MB);")

    # Correctness: for tiny_llama the recapture is now the REAL full TinyLlama-1.1B (int8_full
    # bundle), so cos IS comparable — the on-board step below dumps logits for the SAME seeded
    # input_ids and gates cos/rel vs golden.npy. cos starts None and is filled by that gate; if no
    # real-checkpoint bundle exists for a model, cos stays None with the uncomparable note here.
    res.cos = None
    if _correctness_bundle(model) is None:
        res.notes += (" correctness: cos=None UNCOMPARABLE — no real-checkpoint capture bundle "
                      "whose golden a ggml GGUF forward can reproduce (not_run_is_not_pass);")

    # On-board llama-bench — the only board-gated step. Fail-closed when the board is down.
    do_board = k1_exec.board_available() if run_board is None else run_board
    if do_board:
        try:
            _run_bench_on_board(res, model, gguf_f16, quants)
        except k1_exec.BoardUnavailable as e:
            res.gap_reason = res.gap_reason or f"K1 board run failed: {str(e)[:200]}"
        except Exception as e:  # noqa: BLE001
            res.gap_reason = res.gap_reason or f"K1 ggml bench error: {str(e)[:200]}"
    else:
        res.gap_reason = res.gap_reason or "K1 board unavailable (MERLIN_K1_HOST unset)"

    res.board_vlenb = k1_exec.board_vlenb()
    return _finish(res, model, variant, write)


def _quant_path_cov_for(so: Path | None, quant: str) -> float | None:
    """RVV% of the inner-kernel path a given quant runs (int8 Q8_0, q4_K, ternary...)."""
    if so is None:
        return None
    try:
        rep = rvv_audit.audit_binary(so)
        return _quant_path_coverage(rep, quant)
    except Exception:  # noqa: BLE001
        return None


# Correctness bundles that ARE the real HF checkpoint (so cos vs golden is comparable). tiny_llama's
# full recapture is under the int8 variant dir (torchao int8-weight-only of the real TinyLlama-1.1B);
# the GGUF we run is converted from the SAME checkpoint, so the comparison is apples-to-apples.
_CORRECTNESS_BUNDLE = {
    "tiny_llama": ("int8",),   # golden is the real full TinyLlama-1.1B (int8-weight-only capture)
    # small_llama's GGUF is built from the SAME fp32 capture weights, so its fp32 golden is exactly
    # reproducible by a ggml forward; we gate the (Q8_0/f16) GGUF logits vs the fp32 golden (== "how
    # much did ggml's on-board quant cost" vs the shared fp32 reference).
    "small_llama": ("fp32", "int8"),
}


def _correctness_bundle(model: str):
    """Return a resolved capture bundle whose golden is the REAL checkpoint (or None if none)."""
    for v in _CORRECTNESS_BUNDLE.get(model, ()):
        b = _bundle.resolve(model, v)
        if b.golden.is_file() and b.inputs.is_file():
            return b
    return None


def _read_logits_f32(path: Path) -> "tuple[int, int, object]":
    """Read the (n_vocab, seq, float32[seq*n_vocab]) blob written by the logits dumper."""
    import struct

    import numpy as np
    with open(path, "rb") as f:
        nv, sq = struct.unpack("ii", f.read(8))
        data = np.fromfile(f, dtype=np.float32)
    return nv, sq, data


def _compare_logits_to_golden(logits_path: Path, golden_path: Path) -> "tuple[float | None, float | None]":
    """cos + rel of dumped ggml logits vs golden.npy (both flattened, truncated to common length)."""
    import numpy as np
    nv, sq, got = _read_logits_f32(logits_path)
    gold = np.load(golden_path).astype(np.float64).ravel()
    got = got.astype(np.float64).ravel()
    n = min(got.size, gold.size)
    if n == 0:
        return None, None
    a, g = got[:n], gold[:n]
    denom = (np.linalg.norm(a) * np.linalg.norm(g)) or 1.0
    cos = float(np.dot(a, g) / denom)
    rel = float(np.linalg.norm(a - g) / (np.linalg.norm(g) or 1.0))
    return cos, rel


def _run_bench_on_board(res: BaselineResult, model: str, gguf_f16: Path,
                        quants: tuple[str, ...]) -> None:
    """Deploy once, then quantize + llama-bench EACH quant under one board lock.

    Runs all quants inside a single ``board_lock()`` acquisition (fair to the shared board — one
    hold, not one-per-quant). int8 ``Q8_0`` is benchmarked first and becomes the headline E2E; each
    quant's per-phase throughput + its specific int8/ternary RVV% are recorded. When a real-checkpoint
    correctness bundle exists (tiny_llama), it ALSO dumps logits for the SAME seeded input_ids and
    gates ``cos``/``rel`` vs golden.npy — the first comparable ggml correctness number.
    """
    ld = f"LD_LIBRARY_PATH={_BOARD_DIR}/lib:/usr/lib/riscv64-linux-gnu"
    so = ggml_cpu_so()
    per_quant: dict[str, dict] = {}
    with k1_exec.board_lock():
        deploy_runtime()
        # Push the f16 GGUF (idempotent: skip if already present with matching size).
        remote_f16 = f"{_BOARD_DIR}/{gguf_f16.name}"
        want = gguf_f16.stat().st_size
        chk = _board_sh(f"stat -c %s {remote_f16} 2>/dev/null || echo 0", timeout=60)
        have = int((chk.stdout.strip() or "0").splitlines()[-1]) if chk.stdout.strip() else 0
        if have != want:
            subprocess.run(["scp", *_ssh_opts(), str(gguf_f16), f"{k1_exec.K1_HOST}:{remote_f16}"],
                           check=True, capture_output=True, timeout=1800)
        for quant in quants:
            remote_q = f"{_BOARD_DIR}/{model}-{quant}.gguf"
            run_gguf = remote_q
            if "yes" not in _board_sh(f"test -s {remote_q} && echo yes || echo no", timeout=60).stdout:
                qp = _board_sh(f"{ld} {_BOARD_DIR}/llama-quantize {remote_f16} {remote_q} {quant} 2>&1",
                               timeout=1800)
                if "yes" not in _board_sh(f"test -s {remote_q} && echo yes || echo no").stdout:
                    # Surface the precise llama-quantize reason (e.g. a tensor ncols not divisible by
                    # the quant block size — a real dim constraint, not a runner bug).
                    tail = qp.stdout.strip().splitlines()
                    # Prefer the actionable dim constraint ("ncols not divisible by N") over the
                    # generic trailing "failed to quantize" line.
                    why = next((ln.strip() for ln in tail if "not divisible" in ln), "") or \
                          next((ln.strip() for ln in reversed(tail)
                                if "no tensor type fallback" in ln or "failed to quantize" in ln), "")
                    per_quant[quant] = {"tps": {}, "wall_ns": None, "rc": 1,
                                        "note": f"on-board quantize->{quant} unsupported: {why[:160]}"}
                    continue
            bench, raw = run_on_board(run_gguf, n_threads=k1.K1_OMP_THREADS)
            bench["cov"] = _quant_path_cov_for(so, quant)
            bench["gguf"] = Path(run_gguf).name
            per_quant[quant] = bench

        # --- correctness gate (still under the SAME board lock) -------------------------------
        # If a real-checkpoint bundle exists, dump logits for its exact seeded input_ids and gate
        # cos/rel vs golden.npy. We prefer the int8 (Q8_0) GGUF since the golden is itself an
        # int8-weight-only capture, but fall back to the f16 GGUF if Q8_0 wasn't produced.
        cbundle = _correctness_bundle(model)
        if cbundle is not None:
            try:
                import numpy as np
                dumper = build_logits_dumper()
                subprocess.run(["scp", *_ssh_opts(), str(dumper),
                                f"{k1_exec.K1_HOST}:{_BOARD_DIR}/merlin_logits"],
                               check=True, capture_output=True, timeout=120)
                _board_sh(f"chmod +x {_BOARD_DIR}/merlin_logits")
                ids = np.load(cbundle.inputs)[np.load(cbundle.inputs).files[0]].ravel().tolist()
                idstr = " ".join(str(int(i)) for i in ids)
                # pick the int8 GGUF if present, else the first successful quant, else f16.
                corr_gguf = None
                for q in ("Q8_0", *quants):
                    cand = f"{_BOARD_DIR}/{model}-{q}.gguf"
                    if "yes" in _board_sh(f"test -s {cand} && echo yes || echo no").stdout:
                        corr_gguf = cand
                        corr_quant = q
                        break
                if corr_gguf is None:
                    corr_gguf, corr_quant = remote_f16, "f16"
                out_remote = f"{_BOARD_DIR}/logits_{model}_{corr_quant}.f32"
                r = _board_sh(f"{ld} {_BOARD_DIR}/merlin_logits {corr_gguf} {out_remote} {idstr}",
                              timeout=900)
                loc = cache_dir("baselines") / f"logits_{model}_{corr_quant}.f32"
                subprocess.run(["scp", *_ssh_opts(), f"{k1_exec.K1_HOST}:{out_remote}", str(loc)],
                               capture_output=True, timeout=120)
                if loc.is_file() and loc.stat().st_size > 8:
                    cos, rel = _compare_logits_to_golden(loc, cbundle.golden)
                    res.cos, res.rel = cos, rel
                    src = ("real full TinyLlama-1.1B checkpoint" if model == "tiny_llama" else
                           "GGUF built directly from the same capture-bundle weights"
                           if model == "small_llama" else "same-checkpoint GGUF")
                    res.notes += (f" correctness: COMPARABLE cos={cos:.6f} rel={rel:.6f} vs "
                                  f"{cbundle.root.name}/golden.npy via {corr_quant} logits on the "
                                  f"SAME seeded input_ids ({src});")
                else:
                    res.notes += f" correctness dump produced no logits: {r.stdout.strip()[-100:]} {r.stderr.strip()[-100:]};"
            except Exception as e:  # noqa: BLE001
                res.notes += f" correctness gate failed (cos stays None): {str(e)[:150]};"

    # Headline = the FIRST quant that actually produced throughput (Q8_0 int8 by default).
    ran_any = False
    regions: list[RegionProfile] = []
    tps_summ: list[str] = []
    for quant in quants:
        b = per_quant.get(quant, {})
        if b.get("rc") == 0 and b.get("tps"):
            ran_any = True
            cov = b.get("cov")
            tps_summ.append(f"{quant}[" + ", ".join(f"{t}={v}tok/s" for t, v in b["tps"].items())
                            + f", rvv={cov}]")
            # first successful quant fills the E2E + region profile
            if res.e2e_wall_ns is None and b.get("wall_ns"):
                res.e2e_wall_ns = int(b["wall_ns"])
            if not regions:
                for test, v in b["tps"].items():
                    reg = "gemm" if test.startswith("pp") else "attention"
                    regions.append(RegionProfile(name=reg, rvv_coverage=cov, calls=None,
                        note=f"{quant} llama-bench {test}: {v} tokens/s "
                             f"(int8 path rvv={cov}; rdtime cycles n/a — llama-bench times internally)"))
                # headline RVV% = the benchmarked int8 path's inner-kernel coverage
                if cov is not None:
                    res.rvv_coverage_overall = cov
        elif b.get("note"):
            tps_summ.append(f"{quant}[{b['note']}]")

    res.regions = regions
    res.ran = ran_any
    if not ran_any:
        res.gap_reason = res.gap_reason or f"llama-bench produced no throughput for any of {quants}"
        return
    res.notes += " llama_bench_throughput{" + " | ".join(tps_summ) + "};"


def _finish(res: BaselineResult, model: str, variant: str, write: bool) -> BaselineResult:
    res.validate()
    if write:
        m = artifacts.new_measurement("k1_spacemit", model, "cross_framework")
        res.write(m.path)
    return res


def run_all(models=DEFAULT_MODELS, variant: str = "fp32", *, write: bool = True) -> list[BaselineResult]:
    """Run the ggml arm over the model set (LLM subset first, then VLA out-of-scope gaps)."""
    out = []
    for m in models:
        try:
            out.append(run_model(m, variant, write=write))
        except Exception as e:  # noqa: BLE001 - one model must never sink the batch
            r = BaselineResult(framework=FRAMEWORK, model=m, variant=variant,
                               gap_reason=f"runner exception: {str(e)[:200]}",
                               timestamp=artifacts.utc_stamp())
            if write:
                try:
                    md = artifacts.new_measurement("k1_spacemit", m, "cross_framework")
                    r.write(md.path)
                except Exception:  # noqa: BLE001
                    pass
            out.append(r)
    return out


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="ggml / llama.cpp K1-RVV baseline arm")
    ap.add_argument("models", nargs="*", default=list(DEFAULT_MODELS),
                    help="models to run (default: LLM subset then VLA out-of-scope gaps)")
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--quants", default=",".join(DEFAULT_QUANTS),
                    help="comma-separated on-board quants, int8 first (default Q8_0,q4_K_M)")
    ap.add_argument("--no-write", action="store_true", help="do not write BaselineResult artifacts")
    args = ap.parse_args(argv)
    quants = tuple(q.strip() for q in args.quants.split(",") if q.strip())
    results = []
    for m in (args.models or list(DEFAULT_MODELS)):
        results.append(run_model(m, args.variant, quants=quants, write=not args.no_write))
    for r in results:
        cov = f"{100*r.rvv_coverage_overall:.0f}%RVV" if r.rvv_coverage_overall is not None else "?RVV"
        print(f"{r.model}/{r.variant}: {r.status():10s} {cov} "
              f"fallbacks={len(r.scalar_fallbacks)} {r.gap_reason}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
