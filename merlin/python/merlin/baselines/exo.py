"""EXO baseline arm — forced whole-model on the K1 via EXO RVV kernels + a C glue runtime.

EXO is a kernel DSL + scheduler/autotuner: it emits individual kernels, not a model runtime. Per
the locked decision we force it to *whole-model* by (1) authoring the dominant op — the nn.Linear
GEMM — as an EXO RVV schedule (``exo_kernels/gemm.py`` -> ``gemm_nt_ref``, 8-wide for K1 VLEN=256),
(2) lowering it to C with ``exocc``, and (3) driving the FULL TinyLlama forward from a hand C glue
runtime (``exo_kernels/llama_glue.c``) that calls that kernel for every Linear. Everything the EXO
kernel does NOT cover — RMSNorm, RoPE, GQA softmax attention, SwiGLU/SiLU, residual adds, the token
embedding gather — is **scalar C glue**, recorded here as an explicit :class:`ScalarFallback`
(reason ``"no EXO RVV kernel — scalar glue"``), never hidden.

Two naming schemas are supported so the same decoder-only glue runs multiple Llama-family captures
(``_schema_names``): "hf" (HuggingFace TinyLlama ``lm.model.layers.N...``) and "terse" (the
synthetic small_llama toy ``blocks.N...``). int8 is WEIGHT-ONLY (W8A16): this capture's int8
golden.npy is a weight-only int8 reference (per-output-channel int8 weights dequantized to f32,
activations f32 — verified: all zero-points 0, numpy repro matches golden at cos 1.000000), so the
int8 glue dequantizes the native int8 weight on the fly inside a transpose-free k-reduction f32 dot
(``fgemm_nk.c`` -> ``fgemm_nk_dot_i8``) — NOT a lossy W8A8 activation-quant path (that gates against
a W8A8 golden this full-fidelity capture does not carry).

Honesty mechanics (contract + AGENT.md invariants):
  * ``rvv_audit.enforce_rvv_march`` on the build flags, then ``rvv_audit.audit_binary`` on the
    linked ELF -> ``rvv_coverage_overall`` + the scalar-fallback table.
  * two-level profiling: ``MERLIN_E2E`` + per-region ``MERLIN_REGION`` markers parsed by
    ``profile.parse_profile``.
  * ``not_run_is_not_pass``: a build/run/board failure yields a ``not_built``/``not_run`` cell with
    a specific ``gap_reason``; correctness is gated cos/rel vs ``golden.npy``.

This is EXO-kernels-in-a-glue-runtime, NOT EXO-as-a-whole-model-compiler — stated in the result
``notes``. Results land under ``new_measurement("k1_spacemit", model, "cross_framework")``.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

from merlin.baselines import bundle, k1_exec, profile, rvv_audit
from merlin.baselines.contract import BaselineResult, RegionProfile, ScalarFallback
from merlin.common.artifacts import new_measurement
from merlin.common.paths import repo_root
from merlin.rvvgen import k1

FRAMEWORK = "exo"
MARCH = "rv64gcv"
MABI = "lp64d"

# The scalar-glue ops (no EXO RVV kernel). Recorded verbatim as ScalarFallbacks — honest labeling.
_SCALAR_GLUE = [
    ("rmsnorm", "norm"),
    ("rope", "elementwise"),
    ("attention_softmax", "attention"),
    ("swiglu_silu", "elementwise"),
    ("residual_add", "elementwise"),
    ("embed_gather", "elementwise"),
]

# TinyLlama-1.1B config (matches the tiny_llama capture: 22 layers, GQA 32/4 heads, HD=64).
_CFG = dict(NL=22, H=2048, NH=32, NKV=4, HD=64, FF=5632, V=32000, EPS=1e-5, THETA=10000.0)

# All model2MLIR models. int8 first (coordinator priority): ~4x smaller int8 weights (fit the K1
# faster). This capture's int8 golden is a WEIGHT-ONLY int8 reference (per-channel int8 weights
# dequantized to f32, activations f32), so the int8 glue dequantizes and runs the EXO f32 GEMM.
ALL_MODELS: tuple[str, ...] = (
    "tiny_llama", "small_llama", "molmoact", "bitvla", "openvla",
    "rdt", "rdt2", "groot_n1d7", "pi05", "smolvla", "xr0",
)
VARIANT_ORDER: tuple[str, ...] = ("int8", "fp32")

# Models the C glue's Llama-family forward CANNOT run yet, with a SPECIFIC reason (never omitted /
# fabricated). The glue is a decoder-only Llama forward (embed -> N x {RMSNorm, GQA-attn+RoPE,
# SwiGLU} -> norm -> lm_head) supporting two naming schemas ("hf" TinyLlama + "terse" small_llama,
# see ``_schema_names``). These captures do not fit that decoder-only shape at all:
_GLUE_GAP: dict[str, str] = {
    "molmoact": "OLMo-style backbone (lm.model.blocks.N with attn_norm/ff_norm, mlp.ff_proj/ff_out "
                "fused-gate MLP + wte.embedding) — not the Llama q/k/v/o + gate/up/down decoder the "
                "glue runs",
    "bitvla": "multimodal VLA (vla.vision_tower + vla.multi_modal_projector + vla.language_model) — "
              "glue has no vision encoder / cross-modal path (decoder-only text stack only)",
    "openvla": "multimodal VLA (vla.vision_backbone + vla.projector + vla.language_model) — glue "
               "has no vision backbone / projector path",
    "rdt": "diffusion transformer action head (model.blocks + freq/pos embedders + final_layer, no "
           "vocab embed/lm_head) — not a decoder-only LM the glue can run",
    "rdt2": "diffusion transformer action head (model.blocks + register_tokens + t_embedder) — not "
            "a decoder-only LM",
    "groot_n1d7": "action-head model (head.action_encoder/decoder + state_encoder + diffusion) — "
                  "not a decoder-only LM",
    "pi05": "PaliGemma-with-expert + action/time MLPs (model.paligemma_with_expert, "
            "action_in/out_proj) — multimodal, not the Llama decoder the glue runs",
    "smolvla": "VLA with action/time MLPs + state proj (model.action_*_proj, action_time_mlp) — not "
               "a decoder-only LM",
    "xr0": "diffusion action head (model.dit + action_projector/output_layer + state_projector) — "
           "not a decoder-only LM",
}


def _exo_dir() -> Path:
    return repo_root() / "merlin/python/merlin/baselines/exo_kernels"


def _venv_python() -> Path | None:
    p = repo_root() / "build/baselines/exo/venv/bin/python"
    return p if p.is_file() else None


def _exocc() -> Path | None:
    p = repo_root() / "build/baselines/exo/venv/bin/exocc"
    return p if p.is_file() else None


def _llvm_objdump() -> str | None:
    """llvm-objdump from the SpacemiT toolchain (decodes RVV in linked ELFs; GNU objdump does not)."""
    root = k1._toolchain_root()
    if root and (root / "bin" / "llvm-objdump").is_file():
        return str(root / "bin" / "llvm-objdump")
    import shutil
    return shutil.which("llvm-objdump")


def compile_exo_gemm(out_dir: Path) -> Path:
    """Lower the EXO GEMM schedule to C via exocc. Returns the generated exo_gemm.c path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exocc = _exocc()
    if exocc is None:
        raise RuntimeError("EXO venv/exocc missing — run: build/baselines/exo/venv pip install -e "
                           "third_party/baselines/exo")
    lib = out_dir / "exo_gemm_lib.py"
    lib.write_text("from __future__ import annotations\n"
                   "from merlin.baselines.exo_kernels.gemm import gemm_nt_rvv\n")
    env = dict(os.environ, PYTHONPATH=str(repo_root() / "merlin/python"))
    r = subprocess.run([str(exocc), "-o", str(out_dir), "-p", str(repo_root() / "merlin/python"),
                        "-s", "exo_gemm", str(lib)],
                       capture_output=True, text=True, env=env, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"exocc failed: {r.stdout[-500:]}\n{r.stderr[-500:]}")
    c = out_dir / "exo_gemm.c"
    if not c.is_file():
        raise RuntimeError(f"exocc produced no {c}")
    return c


def compile_glue_ops(out_dir: Path) -> Path:
    """Lower the elementwise glue RVV kernels (residual add, ewise mul) to C via exocc.

    Emits exo_glue_ops.c defining ``residual_add_ref`` (vfadd.vv) + ``ewise_mul_ref`` (vfmul.vv),
    which the int8 glue calls to vectorise its residual-add and SwiGLU product."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exocc = _exocc()
    if exocc is None:
        raise RuntimeError("EXO venv/exocc missing")
    lib = out_dir / "exo_glue_ops_lib.py"
    lib.write_text("from __future__ import annotations\n"
                   "from merlin.baselines.exo_kernels.glue_ops import residual_add_rvv, ewise_mul_rvv\n")
    env = dict(os.environ, PYTHONPATH=str(repo_root() / "merlin/python"))
    r = subprocess.run([str(exocc), "-o", str(out_dir), "-p", str(repo_root() / "merlin/python"),
                        "-s", "exo_glue_ops", str(lib)],
                       capture_output=True, text=True, env=env, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"exocc (glue_ops) failed: {r.stdout[-500:]}\n{r.stderr[-500:]}")
    c = out_dir / "exo_glue_ops.c"
    if not c.is_file():
        raise RuntimeError(f"exocc produced no {c}")
    return c


def _safetensors_offsets(weights: Path) -> tuple[int, dict]:
    with open(weights, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return 8 + n, hdr


def compile_exo_igemm(out_dir: Path, ku: int = 1, u: int = 1) -> Path:
    """Lower the int8 EXO GEMM (vwmacc) to C via exocc with the given schedule knobs.

    ``u`` = output-register blocking (U 16-wide accumulators sharing one A[m,k] load per k — the RVV
    ceiling lever); ``ku`` = k-unroll (only meaningful for u=1). The emitted C function is always
    ``igemm_nt_ref`` (the glue's declared symbol), so the autotuner can swap knobs transparently."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exocc = _exocc()
    if exocc is None:
        raise RuntimeError("EXO venv/exocc missing")
    lib = out_dir / "exo_igemm_lib.py"
    # build_igemm schedules the proc named 'igemm_nt_ref' (scheduling preserves the name), so the
    # emitted C function is 'igemm_nt_ref' for any (ku,u) — the glue's declared symbol.
    lib.write_text("from __future__ import annotations\n"
                   "from merlin.baselines.exo_kernels.igemm import build_igemm\n"
                   f"igemm_nt_ref = build_igemm({int(ku)}, {int(u)})\n")
    env = dict(os.environ, PYTHONPATH=str(repo_root() / "merlin/python"))
    r = subprocess.run([str(exocc), "-o", str(out_dir), "-p", str(repo_root() / "merlin/python"),
                        "-s", "exo_igemm", str(lib)],
                       capture_output=True, text=True, env=env, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"exocc (int8) failed: {r.stdout[-500:]}\n{r.stderr[-500:]}")
    c = out_dir / "exo_igemm.c"
    if not c.is_file():
        raise RuntimeError(f"exocc produced no {c}")
    return c


# Tensor-naming schemas the glue understands. A schema maps the glue's canonical per-layer roles
# (input_ln/q/k/v/o/post_ln/gate/up/down) + globals (embed/final_norm/lm_head) onto a capture's
# actual tensor names, so the SAME decoder-only glue runs multiple Llama-family captures without
# per-model C. Two are supported:
#   * "hf"    — HuggingFace TinyLlama-1.1B layout (lm.model.layers.N.self_attn.q_proj / mlp.gate_proj
#               / input_layernorm; lm.model.embed_tokens / lm.model.norm / lm.lm_head), rope inv_freq
#               from the extra.npz buffer.
#   * "terse" — the synthetic small_llama toy (blocks.N.attn.{q,k,v,o} / mlp.{g,u,dn} / n1,n2;
#               emb / norm / lm), MHA (NKV==NH), rope inv_freq SYNTHESIZED (theta=10000, no buffer).
# Detection is by tensor-name presence so the glue only claims captures it actually reproduces.
def _schema_names(schema: str) -> dict:
    if schema == "terse":
        return dict(embed="emb.weight", final_norm="norm.w", lm_head="lm",
                    layer_prefix="blocks.{L}", input_ln="{p}.n1.w", post_ln="{p}.n2.w",
                    q="{p}.attn.q", k="{p}.attn.k", v="{p}.attn.v", o="{p}.attn.o",
                    gate="{p}.mlp.g", up="{p}.mlp.u", down="{p}.mlp.dn")
    return dict(embed="lm.model.embed_tokens.weight", final_norm="lm.model.norm.weight",
                lm_head="lm.lm_head", layer_prefix="lm.model.layers.{L}",
                input_ln="{p}.input_layernorm.weight", post_ln="{p}.post_attention_layernorm.weight",
                q="{p}.self_attn.q_proj", k="{p}.self_attn.k_proj", v="{p}.self_attn.v_proj",
                o="{p}.self_attn.o_proj", gate="{p}.mlp.gate_proj", up="{p}.mlp.up_proj",
                down="{p}.mlp.down_proj")


def detect_llama_config(hdr: dict, variant: str) -> dict | None:
    """Derive (NL, H, NH, NKV, HD, FF, V, schema, NH_hint) from the safetensors header.

    Tries each supported naming schema ("hf" then "terse"); returns the config for the first that
    matches, or None if the capture is NOT a runnable llama decoder (missing embed/norm/lm_head or
    the per-layer q/k/v/o + gate/up/down set) — the caller then records an honest not_built gap.
    """
    q8 = variant == "int8"
    suffix = ".parametrizations.weight.original0" if q8 else ".weight"

    def shp(name: str):
        return hdr[name]["shape"] if name in hdr else None

    for schema in ("hf", "terse"):
        nm = _schema_names(schema)
        emb = shp(nm["embed"])
        if emb is None:
            continue
        V, H = int(emb[0]), int(emb[1])
        NL = 0
        while True:
            p = nm["layer_prefix"].format(L=NL)
            need = [nm[r].format(p=p) + suffix for r in ("q", "k", "v", "o", "gate", "up", "down")]
            need += [nm["input_ln"].format(p=p), nm["post_ln"].format(p=p)]
            if not all(n in hdr for n in need):
                break
            NL += 1
        if NL == 0 or nm["final_norm"] not in hdr:
            continue
        p0 = nm["layer_prefix"].format(L=0)
        kv_out = int(shp(nm["k"].format(p=p0) + suffix)[0])       # [NKV*HD, H]
        FF = int(shp(nm["gate"].format(p=p0) + suffix)[0])        # [FF, H]
        # NH: for the terse toy, q_proj is [H,H] (MHA) so NH is not derivable from shapes; carry a
        # hint (small_llama uses h=4). For hf, NH = H/HD is derived from the rope inv_freq length.
        return dict(NL=NL, H=H, V=V, FF=FF, KV_OUT=kv_out, schema=schema,
                    NH_hint=(4 if schema == "terse" else None))
    return None


def emit_weights_header(bundle_root: Path, out_dir: Path, variant: str = "fp32") -> Path:
    """Emit llama_weights.h — config + safetensors offset table + input ids + rope inv_freq.

    Auto-detects NL/dims + the naming schema ("hf" TinyLlama or "terse" small_llama) from the header,
    and for int8 emits the (i8 weight offset, f32 scale offset) pair per Linear so the int8 glue can
    dequantize per-channel (weight-only int8). The rope inv_freq is read from the extra.npz buffer
    when present, else synthesized (theta=10000, HD=H/NH) for captures that compute rope inline.
    Returns the header path; raises if the capture is not a runnable llama stack.
    """
    data0, hdr = _safetensors_offsets(bundle_root / "weights.safetensors")
    cfg = detect_llama_config(hdr, variant)
    if cfg is None:
        raise RuntimeError("capture is not a runnable Llama-family decoder (glue supports "
                           "decoder-only q/k/v/o + gate/up/down stacks)")

    def off(name: str) -> int:
        return int(hdr[name]["data_offsets"][0])

    nm = _schema_names(cfg["schema"])
    ids = np.load(bundle_root / "inputs.npz")["in0"][0].astype(np.int64)
    S = int(ids.shape[0])
    NL, H, V, FF, KV_OUT = cfg["NL"], cfg["H"], cfg["V"], cfg["FF"], cfg["KV_OUT"]

    # rope inv_freq: prefer the captured buffer (hf); else SYNTHESIZE it (the terse toy computes rope
    # inline with theta=10000, HD = H/NH). HD then fixes NH/NKV.
    extra = np.load(bundle_root / "extra.npz")
    ivkey = "buf::lm.model.rotary_emb.inv_freq"
    if ivkey in extra.files:
        inv_freq = extra[ivkey].astype(np.float32)
        HD = 2 * int(inv_freq.shape[0])
    else:
        NH = int(cfg["NH_hint"] or (H // 64))     # toy: NH from the loader hint (small_llama h=4)
        HD = H // NH
        half = HD // 2
        inv_freq = (1.0 / (_CFG["THETA"] ** (np.arange(0, half, dtype=np.float32) / half))
                    ).astype(np.float32)
    NH, NKV = H // HD, KV_OUT // HD
    q8 = variant == "int8"
    sfx = ".parametrizations.weight.original0" if q8 else ".weight"
    scl = ".parametrizations.weight.original1"

    def lm_head_name(part: str) -> str:
        return nm["lm_head"] + part

    lines = ["/* generated by merlin.baselines.exo.emit_weights_header — do not edit. */",
             "#pragma once", "#include <stddef.h>", "#include <stdint.h>",
             f"#define NL {NL}", f"#define H {H}", f"#define NH {NH}", f"#define NKV {NKV}",
             f"#define HD {HD}", f"#define FF {FF}", f"#define V {V}", f"#define S {S}",
             f"#define EPS {_CFG['EPS']}f", f"#define WDATA0 {data0}UL",
             f"#define WOFF_EMBED {off(nm['embed'])}UL",
             f"#define WOFF_FINAL_NORM {off(nm['final_norm'])}UL",
             "static const long INPUT_IDS[S] = {" + ",".join(str(int(i)) for i in ids) + "};",
             "static const float INV_FREQ[HD/2] = {" + ",".join(repr(float(x)) for x in inv_freq) + "};"]
    if q8:
        lines += [f"#define WOFF_LM_HEAD_W {off(lm_head_name(sfx))}UL",
                  f"#define WOFF_LM_HEAD_S {off(lm_head_name(scl))}UL",
                  "struct layer_off { size_t input_ln, q_proj_w,q_proj_s, k_proj_w,k_proj_s,"
                  " v_proj_w,v_proj_s, o_proj_w,o_proj_s, post_ln,"
                  " gate_w,gate_s, up_w,up_s, down_w,down_s; };",
                  "static const struct layer_off LAYERS[NL] = {"]
        for L in range(NL):
            p = nm["layer_prefix"].format(L=L)
            fields = [nm["input_ln"].format(p=p)]
            for role in ("q", "k", "v", "o"):
                fields += [nm[role].format(p=p) + sfx, nm[role].format(p=p) + scl]
            fields += [nm["post_ln"].format(p=p)]
            for role in ("gate", "up", "down"):
                fields += [nm[role].format(p=p) + sfx, nm[role].format(p=p) + scl]
            lines.append("  {" + ",".join(str(off(f)) + "UL" for f in fields) + "},")
    else:
        lines += [f"#define WOFF_LM_HEAD {off(lm_head_name('.weight'))}UL",
                  "struct layer_off { size_t input_ln, q_proj, k_proj, v_proj, o_proj,"
                  " post_ln, gate_proj, up_proj, down_proj; };",
                  "static const struct layer_off LAYERS[NL] = {"]
        for L in range(NL):
            p = nm["layer_prefix"].format(L=L)
            roles = ("input_ln", "q", "k", "v", "o", "post_ln", "gate", "up", "down")
            names = [nm[r].format(p=p) + ("" if r in ("input_ln", "post_ln") else ".weight")
                     for r in roles]
            lines.append("  {" + ",".join(str(off(n)) + "UL" for n in names) + "},")
    lines += ["};", ""]
    h = out_dir / "llama_weights.h"
    h.write_text("\n".join(lines))
    return h


def build_glue_binary(bundle_root: Path, work: Path, variant: str = "fp32",
                      ku: int = 1, u: int = 1) -> tuple[Path, list[str]]:
    """Compile the EXO GEMM + C glue into a K1 rv64gcv ELF. Returns (elf, march_flags).

    fp32 -> the f32 vfmacc GEMM (gemm.py) + llama_glue.c. int8 -> the vwmacc widening GEMM
    (igemm.py; output-register-blocked by ``u``, k-unrolled by ``ku``) + llama_glue_int8.c. Raises
    with a specific reason if the capture is not a runnable llama decoder (a not_built gap)."""
    work.mkdir(parents=True, exist_ok=True)
    cc = k1.toolchain_cc()
    if cc is None:
        raise RuntimeError("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")
    emit_weights_header(bundle_root, work, variant)   # raises if not a llama stack
    flags = [f"-march={MARCH}", f"-mabi={MABI}"]
    rvv_audit.enforce_rvv_march(MARCH)
    extra_c: list[str] = []
    if variant == "int8":
        # WEIGHT-ONLY int8 (W8A16): this capture's int8 golden.npy is a weight-only int8 reference
        # (int8 per-output-channel weights dequantized to f32, activations kept f32 — verified: all
        # 155 zero-points are exactly 0, and a numpy weight-only repro matches golden.npy at cos
        # 1.000000 / rel 0.0000, whereas full W8A8 activation-quant gives cos 0.949 / rel 0.958
        # against it). So the glue dequant+transposes each int8 weight and runs the EXO *f32* RVV
        # GEMM (gemm_nt_ref) — the same kernel/reference the passing ExecuTorch/TVM int8 cells use.
        # The EXO int8 vwmacc GEMM (igemm_nt_ref) is still compiled+linked below for the RVV audit
        # (EXO-authored-kernel story), but it is NOT on the weight-only glue's hot path.
        kernel_c = compile_exo_gemm(work)                        # EXO f32 vfmacc GEMM (compiled+audited)
        extra_c = [str(compile_glue_ops(work)),                  # residual-add + ewise-mul RVV
                   str(_exo_dir() / "fgemm_nk.c"),               # transpose-free weight-only dot (hot path)
                   str(compile_exo_igemm(work, ku, u))]          # EXO int8 vwmacc GEMM (audit only)
        glue_c = _exo_dir() / "llama_glue_int8.c"
        elf = work / "exo_llama_int8_k1"
    else:
        kernel_c = compile_exo_gemm(work)
        glue_c = _exo_dir() / "llama_glue.c"
        elf = work / "exo_llama_k1"
    cmd = [str(cc), "--target=riscv64-unknown-linux-gnu", *flags, "-O3", "-ffast-math",
           f"-I{work}", str(glue_c), str(kernel_c), *extra_c, "-lm", "-lpthread", "-o", str(elf)]
    try:
        _run(cmd + ["-static"])
    except RuntimeError:
        _run(cmd)
    if not elf.is_file():
        raise RuntimeError("glue link produced no binary")
    return elf, flags


# Bounded output-register-blocking (U) search for the int8 vwmacc GEMM autotune (VLEN=256). U is
# the RVV-ceiling lever: U 16-wide i32 accumulators share one scalar A[m,k] load per k. Candidates
# kept small so the shared K1 isn't hogged (one quick micro-bench run each under board_lock). All U
# here divide N/16 for every Linear (smallest N=256 -> N/16=16, divisible by 1,2,4,8).
IGEMM_U_CANDIDATES: tuple[int, ...] = (1, 2, 4, 8)
IGEMM_KU_CANDIDATES: tuple[int, ...] = IGEMM_U_CANDIDATES  # back-comat alias (tests/history)


def autotune_igemm_ku(work: Path, *, shape=(8, 4096, 2048),
                      candidates=IGEMM_U_CANDIDATES) -> tuple[int, dict[int, int]]:
    """Bounded on-board search for the best output-register-blocking ``U`` of the int8 vwmacc GEMM.

    For each U candidate: build_igemm(1, U) -> exocc C, cross-compile the standalone micro-bench
    (igemm_bench.c) at the representative ``shape``, run once on the board, parse BENCH_TICKS.
    Returns (best_U, {U: ticks}). MUST be called inside a ``k1_exec.board_lock()``. Fail-closed: a
    candidate that won't build/run is skipped; if none run, returns (1, {})."""
    cc = k1.toolchain_cc()
    if cc is None:
        raise RuntimeError("SpacemiT toolchain not found")
    bm, bn, bk = shape
    bench_c = _exo_dir() / "igemm_bench.c"
    ticks: dict[int, int] = {}
    for u in candidates:
        d = work / f"u{u}"
        try:
            kernel_c = compile_exo_igemm(d, 1, u)
            elf = d / "igemm_bench"
            _run([str(cc), "--target=riscv64-unknown-linux-gnu", f"-march={MARCH}", f"-mabi={MABI}",
                  "-O3", "-ffast-math", f"-DBM={bm}", f"-DBN={bn}", f"-DBK={bk}", f"-I{d}",
                  str(bench_c), str(kernel_c), "-lm", "-o", str(elf)] + (["-static"]))
            remote = k1_exec.push(elf, f"/tmp/igemm_bench_u{u}")
            k1_exec.run([f"chmod +x {remote}"], timeout=30)
            proc = k1_exec.run([remote], timeout=300)
            k1_exec.run([f"rm -f {remote}"], timeout=30)
            for line in (proc.stdout + proc.stderr).splitlines():
                if line.startswith("BENCH_TICKS "):
                    ticks[u] = int(line.split()[1])
                    break
        except Exception:  # noqa: BLE001 - skip a candidate that won't build/run (fail-closed)
            continue
    if not ticks:
        return 1, {}
    best = min(ticks, key=ticks.get)
    return best, ticks


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")


def _parse_out(stdout: str) -> np.ndarray | None:
    """Parse the ASCII OUT fallback stream (small-V only). Robust to a truncated final token: an
    SSH drop mid-stream leaves garbage, so we keep only clean integer tokens and drop the rest
    rather than crash (the caller then sees a short array and records an honest gap)."""
    for line in stdout.splitlines():
        if line.startswith("OUT "):
            parts = line.split()[2:]
            vals = []
            for tok in parts:
                if tok.isdigit():
                    vals.append(int(tok))
                else:
                    break  # truncated / connection-noise token: stop cleanly
            if not vals:
                return None
            return np.array(vals, dtype=np.uint32).view(np.float32)
    return None


def _parse_outfile(stdout: str) -> tuple[str, int] | None:
    """Parse the OUTFILE marker: 'OUTFILE <remote_path> <n_floats>'."""
    for line in stdout.splitlines():
        if line.startswith("OUTFILE "):
            p = line.split()
            if len(p) >= 3 and p[2].isdigit():
                return p[1], int(p[2])
    return None


def _scalar_fallbacks(variant: str = "fp32") -> list[ScalarFallback]:
    """The ops still on the scalar path (honest labeling). For int8, residual-add + the SwiGLU
    product are now RVV (glue_ops vfadd/vfmul), so they drop off; only the SiLU sigmoid exp stays
    scalar. fp32 keeps all elementwise ops scalar (only its GEMM is an EXO kernel)."""
    if variant == "int8":
        # weight-only int8: residual-add + the SwiGLU product are RVV (glue_ops vfadd/vfmul); only
        # the SiLU sigmoid exp stays scalar. NO activation-quant/requant (activations stay f32).
        ops = [("rmsnorm", "norm"), ("rope", "elementwise"),
               ("attention_softmax", "attention"),
               ("silu_sigmoid_exp", "elementwise"),   # exp is scalar; the *u product is RVV
               ("embed_gather", "elementwise")]
    else:
        ops = list(_SCALAR_GLUE)
    return [ScalarFallback(symbol=sym, reason="no EXO RVV kernel — scalar glue", region=reg)
            for sym, reg in ops]


def run(model: str = "tiny_llama", variant: str = "fp32", *, autotune: bool = True) -> BaselineResult:
    """Build + (board-locked) run the EXO-glue whole-model on the K1; return a BaselineResult."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = _submodule_sha()
    res = BaselineResult(framework=FRAMEWORK, model=model, variant=variant, substrate="k1_spacemit",
                         march=MARCH, toolchain="spacemit-clang-19+exo-1.0.0",
                         framework_commit=commit, timestamp=ts, cycle_accurate=False)
    kdesc = ("WEIGHT-ONLY int8 (W8A16), TRANSPOSE-FREE: native [OUT,IN] int8 weights dequantized "
             "on the fly inside a k-reduction f32 dot GEMM (fgemm_nk_dot_i8, hand-RVV vfmacc.vv + "
             "vfredusum) — matches the capture's weight-only int8 golden (activations stay f32). The "
             "EXO f32 GEMM (gemm_nt_ref) + EXO int8 vwmacc GEMM (igemm_nt_ref) are compiled + "
             "RVV-audited (EXO-authored-kernel story) but off the hot path"
             if variant == "int8" else "f32 vfmacc.vf GEMM (gemm_nt_ref, EXO)")
    if variant == "int8":
        glue_note = ("RVV kernels: transpose-free weight-only int8 dot GEMM (fgemm_nk_dot_i8, "
                     "hand-RVV) + residual-add (vfadd, EXO) + SwiGLU product (vfmul, EXO); scalar "
                     "glue: RMSNorm/RoPE/attention-softmax/SiLU-sigmoid-exp/embed (labeled). "
                     "Activations are NOT quantized (weight-only int8, matching golden.npy — full "
                     "W8A8 would gate against a W8A8 golden this full-fidelity capture doesn't "
                     "carry: cos 0.949 vs weight-only's cos 1.0).")
    else:
        glue_note = ("Only the nn.Linear GEMM is an EXO kernel; RMSNorm/RoPE/attention-softmax/"
                     "SwiGLU/residual/embed are scalar C glue (labeled).")
    res.notes = (f"whole-model = RVV GEMM kernel [{kdesc}] + hand C glue runtime; EXO is a "
                 f"kernel DSL/scheduler, NOT a whole-model compiler. {glue_note}")
    res.scalar_fallbacks = _scalar_fallbacks(variant)

    # Honest not_built gap for captures the Llama-family glue cannot run (specific reason each).
    if model in _GLUE_GAP:
        res.gap_reason = (f"EXO glue runtime is a Llama-family decoder; {model} does not fit: "
                          f"{_GLUE_GAP[model]}")
        return res.validate()
    # int8/W8A8 gates at the repo's integer tier (cos>0.999, rel<1e-2), matching
    # zephyr_model/buddy/tvm; fp32 uses the strict per-model tolerance.
    if variant == "int8":
        cos_t, rel_t = 0.999, 1e-2
    else:
        cos_t, rel_t = bundle.tolerance(model)
    res.cos_threshold, res.rel_threshold = cos_t, rel_t

    # fp32 tiny_llama capture is the legacy-named 'tiny_consistent' dir; resolve then fall back.
    b = bundle.resolve(model, variant)
    if not b.mlir.is_file() and model == "tiny_llama" and variant == "fp32":
        from merlin.common.artifacts import recaptures_dir
        alt = recaptures_dir() / "tiny_consistent"
        if (alt / "golden.npy").is_file():
            b = bundle.CaptureBundle(model=model, variant=variant, root=alt)
    try:
        b.require()
    except FileNotFoundError as e:
        res.gap_reason = f"capture bundle missing: {e}"
        return res.validate()

    work = repo_root() / "build/baselines/exo/work" / f"{model}_{variant}"
    # int8: bounded on-board autotune of the EXO vwmacc GEMM's output-register-blocking U. NOTE: the
    # whole-model glue now calls the TRANSPOSE-FREE dot GEMM (igemm_nk_dot), which measured ~32x
    # faster end-to-end than transpose+EXO-vwmacc (K1 strided loads are catastrophic; the dot is
    # contiguous, M tiny). The autotune still runs to pick + report the best-measured U for the
    # EXO-authored kernel (compiled + RVV-audited), but that kernel is not on the glue's hot path.
    u = 8
    u_ticks: dict[int, int] = {}
    if variant == "int8" and autotune and k1_exec.board_available():
        try:
            emit_weights_header(b.root, work, variant)  # ensure work dir + header exist first
            with k1_exec.board_lock():
                u, u_ticks = autotune_igemm_ku(work / "autotune")
            res.notes += (f" | EXO-igemm U autotune (audit only; glue uses dot GEMM): "
                          f"U_ticks={u_ticks} -> best U={u}"
                          if u_ticks else " | EXO-igemm autotune: no candidate ran, U=8")
        except Exception as e:  # noqa: BLE001
            res.notes += f" | autotune skipped: {str(e)[:80]}"
            u = 8
    try:
        elf, flags = build_glue_binary(b.root, work, variant, u=u)
        res.built = True
    except Exception as e:  # noqa: BLE001
        res.gap_reason = f"build failed: {type(e).__name__}: {str(e)[:300]}"
        return res.validate()

    # RVV audit of the linked ELF (honesty). Use llvm-objdump: the SpacemiT GNU objdump emits
    # ".insn" for RVV in a *linked* ELF (binutils relaxation quirk) and would under-report to 0;
    # llvm-objdump decodes vset/vle/vse/vfmacc reliably. The GEMM (gemm_nt_ref) is the only EXO
    # kernel; the whole-ELF coverage is dominated by static libc, so we also record the kernel's
    # own coverage in notes.
    try:
        rep = rvv_audit.audit_binary(elf, objdump=_llvm_objdump())
        res.rvv_coverage_overall = rep.coverage_overall
        # int8: audit BOTH the runtime dot GEMM (igemm_nk_dot, the hot path) and the EXO vwmacc
        # kernel (igemm_nt_ref, kept for the EXO-authored story). fp32: the EXO f32 GEMM.
        ksyms = (("fgemm_nk_dot_i8", "gemm_nt_ref", "igemm_nt_ref") if variant == "int8"
                 else ("gemm_nt_ref",))
        for ksym in ksyms:
            gk = rep.by_symbol.get(ksym)
            if gk is not None and gk.coverage is not None:
                res.notes += (f" | {ksym} RVV coverage={gk.coverage:.2f} "
                              f"(vec={gk.vector} scalar={gk.scalar_compute})")
        res.notes += f"; whole-ELF coverage libc-dominated ({res.rvv_coverage_overall:.3f})."
    except Exception as e:  # noqa: BLE001
        res.notes += f" | rvv_audit unavailable: {e}"

    if not k1_exec.board_available():
        res.gap_reason = "K1 board unavailable (built + RVV-audited OK; on-board run fail-closed)"
        return res.validate()
    res.board_vlenb = k1_exec.board_vlenb()

    # On-board run — serialize behind the board lock (single physical K1 shared with 4 agents).
    # The fp32 weights blob is multi-GB; scp is slow, so it is cached on the board's rootfs (real
    # flash, not /tmp tmpfs) and only re-pushed when absent or size-mismatched. The binary is small.
    try:
        with k1_exec.board_lock():
            remote = k1_exec.push(elf, f"/tmp/{model}_{variant}_exo_llama_k1")
            remote_w = f"{k1.K1_REMOTE_DIR}/{model}_{variant}.weights.bin"
            want = b.weights.stat().st_size
            have = k1_exec.run([f"stat -c %s {remote_w} 2>/dev/null || echo 0"], timeout=60)
            if have.stdout.strip() != str(want):
                # generous timeout for the multi-GB blob (slow, contended board network); retry
                # once on a stall since scp cannot resume (a killed/hung transfer leaves a partial).
                for attempt in range(2):
                    try:
                        k1_exec.push(b.weights, remote_w, timeout=3600)
                        chk = k1_exec.run([f"stat -c %s {remote_w} 2>/dev/null || echo 0"], timeout=60)
                        if chk.stdout.strip() == str(want):
                            break
                    except Exception:  # noqa: BLE001
                        if attempt == 1:
                            raise
            k1_exec.run([f"chmod +x {remote}"], timeout=60)
            # Write the S*V logits to a binary file on-board (robust vs streaming ~2MB ASCII over a
            # contended SSH link); scp it back (to the board's rootfs, NOT /tmp tmpfs). A partial/
            # failed scp must NOT be mistaken for a stale prior file, so we delete any stale
            # local_out first and verify the scp succeeded + the returned file has the EXACT
            # expected size before gating (else -> honest not_run gap).
            remote_out = f"{k1.K1_REMOTE_DIR}/{model}_{variant}_exo_out.bin"
            local_out = work / "exo_out.bin"
            if local_out.exists():
                local_out.unlink()
            proc = k1_exec.run(
                [f"MERLIN_WEIGHTS={remote_w} MERLIN_OUTFILE={remote_out} {remote}"], timeout=1800)
            import time as _time
            _time.sleep(1)  # let the on-board file flush before scp
            scp_ok = False
            for _att in range(2):
                r = subprocess.run(["scp", *k1_exec._SSH_OPTS, f"{k1.K1_HOST}:{remote_out}",
                                    str(local_out)], capture_output=True, timeout=300)
                if r.returncode == 0 and local_out.is_file():
                    scp_ok = True
                    break
                if local_out.exists():
                    local_out.unlink()
            try:
                k1_exec.run([f"rm -f {remote} {remote_out}"], timeout=60)  # keep cached weights
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        res.gap_reason = f"K1 run failed: {type(e).__name__}: {str(e)[:300]}"
        return res.validate()

    stdout = proc.stdout + proc.stderr
    # prefer the binary output file (robust); require the EXACT expected element count so a
    # truncated/failed transfer becomes an honest gap, never a false-low cos on partial/stale data.
    out = None
    of = _parse_outfile(stdout)
    if of is not None and scp_ok and local_out.is_file():
        raw = np.fromfile(local_out, dtype=np.float32)
        if raw.size == of[1]:            # exact match only (not >=; guards stale/partial files)
            out = raw
        else:
            res.gap_reason = (f"output file truncated: got {raw.size} floats, expected {of[1]} "
                              f"(scp_ok={scp_ok}) — transfer incomplete, not gating on partial data")
            return res.validate()
    if out is None:
        out = _parse_out(stdout)
    if out is None:
        res.gap_reason = (f"no usable output (rc={proc.returncode}); OUTFILE={of} scp_ok={scp_ok}; "
                          f"console tail: {stdout[-200:]}")
        return res.validate()
    res.ran = True

    e2e, regions = profile.parse_profile(stdout)
    res.e2e_rdtime_ticks, res.e2e_cycles, res.e2e_wall_ns = e2e.rdtime_ticks, e2e.cycles, e2e.wall_ns
    res.regions = regions

    # int8 gates against the W8A8 golden (the math the integer datapath intends) when present,
    # matching the repo convention (buddy/tvm _golden_for); fp32 uses golden.npy.
    gold_path = b.golden
    if variant == "int8" and (b.root / "golden_w8a8.npy").is_file():
        gold_path = b.root / "golden_w8a8.npy"
    res.notes += f" | gold={gold_path.name}"
    gold = np.load(gold_path).astype(np.float32).ravel()
    pred = out.astype(np.float32).ravel()[:gold.size]
    if pred.size == gold.size:
        cos = float(pred @ gold / (np.linalg.norm(pred) * np.linalg.norm(gold) + 1e-30))
        rel = float(np.abs(pred - gold).max() / (np.abs(gold).max() + 1e-9))
        res.cos, res.rel = cos, rel
        # for int8, also record cos vs golden.npy (weight-only) for context.
        if variant == "int8" and b.golden.is_file():
            g2 = np.load(b.golden).astype(np.float32).ravel()
            if g2.size == pred.size:
                res.notes += (f" | vs golden.npy(weight-only) cos="
                              f"{float(pred @ g2 / (np.linalg.norm(pred)*np.linalg.norm(g2)+1e-30)):.4f}")
    else:
        res.gap_reason = f"output size {pred.size} != golden {gold.size}"
    return res.validate()


def _submodule_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           cwd=repo_root() / "third_party/baselines/exo",
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def run_and_record(model: str = "tiny_llama", variant: str = "fp32") -> Path:
    """Run the arm and write baseline_result.json into a cross_framework measurement dir."""
    res = run(model, variant)
    mdir = new_measurement("k1_spacemit", model, "cross_framework")
    res.write(mdir.path)
    mdir.write_manifest()
    return mdir.path


def run_all(models: tuple[str, ...] = ALL_MODELS, variant: str = "int8") -> list[BaselineResult]:
    """Run the EXO arm over the model set for one variant (int8 first). Writes each result.

    Every model produces a BaselineResult: llama-family captures run end-to-end; the rest are
    explicit not_built gaps with a specific reason (never omitted). One model's failure never sinks
    the batch."""
    out: list[BaselineResult] = []
    for m in models:
        try:
            res = run(m, variant)
        except Exception as e:  # noqa: BLE001
            res = BaselineResult(framework=FRAMEWORK, model=m, variant=variant,
                                 gap_reason=f"runner exception: {str(e)[:200]}",
                                 timestamp=_dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        try:
            mdir = new_measurement("k1_spacemit", m, "cross_framework")
            res.write(mdir.path)
            mdir.write_manifest()
        except Exception:  # noqa: BLE001
            pass
        out.append(res.validate())
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        variant = sys.argv[2] if len(sys.argv) > 2 else "int8"
        for r in run_all(variant=variant):
            print(f"{r.model:12s} {r.variant:4s} {r.status():10s} "
                  f"cos={r.cos} rvv={r.rvv_coverage_overall} {r.gap_reason[:60]}")
    else:
        m = sys.argv[1] if len(sys.argv) > 1 else "tiny_llama"
        v = sys.argv[2] if len(sys.argv) > 2 else "fp32"
        d = run_and_record(m, v)
        print(f"wrote {d}")
