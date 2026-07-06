"""EXO baseline arm — forced whole-model on the K1 via EXO RVV kernels + a C glue runtime.

EXO is a kernel DSL + scheduler/autotuner: it emits individual kernels, not a model runtime. Per
the locked decision we force it to *whole-model* by (1) authoring the dominant op — the nn.Linear
GEMM — as an EXO RVV schedule (``exo_kernels/gemm.py`` -> ``gemm_nt_ref``, 8-wide for K1 VLEN=256),
(2) lowering it to C with ``exocc``, and (3) driving the FULL TinyLlama forward from a hand C glue
runtime (``exo_kernels/llama_glue.c``) that calls that kernel for every Linear. Everything the EXO
kernel does NOT cover — RMSNorm, RoPE, GQA softmax attention, SwiGLU/SiLU, residual adds, the token
embedding gather — is **scalar C glue**, recorded here as an explicit :class:`ScalarFallback`
(reason ``"no EXO RVV kernel — scalar glue"``), never hidden.

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

# All model2MLIR models. int8 first (coordinator priority): ~4x smaller weights (fits the K1
# faster) and the real integer RVV vwmacc datapath.
ALL_MODELS: tuple[str, ...] = (
    "tiny_llama", "small_llama", "molmoact", "bitvla", "openvla",
    "rdt", "rdt2", "groot_n1d7", "pi05", "smolvla", "xr0",
)
VARIANT_ORDER: tuple[str, ...] = ("int8", "fp32")

# Models the C glue's Llama-family forward CANNOT run yet, with a SPECIFIC reason (never omitted /
# fabricated). The glue is a decoder-only Llama forward (embed -> N x {RMSNorm, GQA-attn+RoPE,
# SwiGLU} -> norm -> lm_head) keyed on the ``lm.model.layers.N.self_attn.{q,k,v,o}_proj`` +
# ``mlp.{gate,up,down}_proj`` naming. These captures do not fit that shape:
_GLUE_GAP: dict[str, str] = {
    "small_llama": "renamed Llama (blocks.N.attn.{q,k,v,o}, emb/norm.w) and no rotary_emb.inv_freq "
                   "in extra.npz — glue needs the lm.model.layers naming + RoPE inv_freq buffer",
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


def _safetensors_offsets(weights: Path) -> tuple[int, dict]:
    with open(weights, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return 8 + n, hdr


def compile_exo_igemm(out_dir: Path) -> Path:
    """Lower the int8 EXO GEMM (vwmacc) to C via exocc. Returns the generated exo_igemm.c path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exocc = _exocc()
    if exocc is None:
        raise RuntimeError("EXO venv/exocc missing")
    lib = out_dir / "exo_igemm_lib.py"
    lib.write_text("from __future__ import annotations\n"
                   "from merlin.baselines.exo_kernels.igemm import igemm_nt_rvv\n")
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


# A Llama-family capture the glue can run: a decoder-only stack with these per-layer weights. Detect
# by tensor-name presence so the glue only claims models it actually reproduces (else honest gap).
def detect_llama_config(hdr: dict, variant: str) -> dict | None:
    """Derive (NL, H, NH, NKV, HD, FF, V) from the safetensors header for a Llama-family capture.

    Returns None if the capture is NOT a runnable llama decoder (missing embed/norm/lm_head or the
    per-layer q/k/v/o + gate/up/down set) — the caller then records an honest not_built gap.
    """
    q8 = variant == "int8"
    suffix = ".parametrizations.weight.original0" if q8 else ".weight"

    def shp(name: str):
        return hdr[name]["shape"] if name in hdr else None

    emb = shp("lm.model.embed_tokens.weight")
    if emb is None:
        return None
    V, H = int(emb[0]), int(emb[1])
    # count contiguous layers 0..NL-1 that have the full attention+mlp weight set.
    NL = 0
    while True:
        p = f"lm.model.layers.{NL}"
        need = [f"{p}.self_attn.q_proj{suffix}", f"{p}.self_attn.k_proj{suffix}",
                f"{p}.self_attn.v_proj{suffix}", f"{p}.self_attn.o_proj{suffix}",
                f"{p}.mlp.gate_proj{suffix}", f"{p}.mlp.up_proj{suffix}",
                f"{p}.mlp.down_proj{suffix}", f"{p}.input_layernorm.weight",
                f"{p}.post_attention_layernorm.weight"]
        if not all(n in hdr for n in need):
            break
        NL += 1
    if NL == 0 or "lm.model.norm.weight" not in hdr:
        return None
    kv = shp(f"lm.model.layers.0.self_attn.k_proj{suffix}")   # [NKV*HD, H]
    ff = shp(f"lm.model.layers.0.mlp.gate_proj{suffix}")       # [FF, H]
    kv_out, FF = int(kv[0]), int(ff[0])
    # head_dim from rope inv_freq length (HD = 2*len(inv_freq)); NH = H/HD; NKV = kv_out/HD.
    return dict(NL=NL, H=H, V=V, FF=FF, KV_OUT=kv_out)


def emit_weights_header(bundle_root: Path, out_dir: Path, variant: str = "fp32") -> Path:
    """Emit llama_weights.h — config + safetensors offset table + input ids + rope inv_freq.

    Auto-detects NL/dims from the header (captures are truncated to a few layers), and for int8
    emits the (i8 weight offset, f32 scale offset) pair per Linear so the int8 glue can run the
    W8A8 vwmacc path. Returns the header path; raises if the capture is not a runnable llama stack.
    """
    data0, hdr = _safetensors_offsets(bundle_root / "weights.safetensors")
    cfg = detect_llama_config(hdr, variant)
    if cfg is None:
        raise RuntimeError("capture is not a runnable Llama-family decoder (glue supports "
                           "decoder-only q/k/v/o + gate/up/down stacks)")

    def off(name: str) -> int:
        return int(hdr[name]["data_offsets"][0])

    ids = np.load(bundle_root / "inputs.npz")["in0"][0].astype(np.int64)
    inv_freq = np.load(bundle_root / "extra.npz")["buf::lm.model.rotary_emb.inv_freq"].astype(np.float32)
    S = int(ids.shape[0])
    HD = 2 * int(inv_freq.shape[0])
    NL, H, V, FF, KV_OUT = cfg["NL"], cfg["H"], cfg["V"], cfg["FF"], cfg["KV_OUT"]
    NH, NKV = H // HD, KV_OUT // HD
    q8 = variant == "int8"
    sfx = ".parametrizations.weight.original0" if q8 else ".weight"
    scl = ".parametrizations.weight.original1"

    lines = ["/* generated by merlin.baselines.exo.emit_weights_header — do not edit. */",
             "#pragma once", "#include <stddef.h>", "#include <stdint.h>",
             f"#define NL {NL}", f"#define H {H}", f"#define NH {NH}", f"#define NKV {NKV}",
             f"#define HD {HD}", f"#define FF {FF}", f"#define V {V}", f"#define S {S}",
             f"#define EPS {_CFG['EPS']}f", f"#define WDATA0 {data0}UL",
             f"#define WOFF_EMBED {off('lm.model.embed_tokens.weight')}UL",
             f"#define WOFF_FINAL_NORM {off('lm.model.norm.weight')}UL",
             "static const long INPUT_IDS[S] = {" + ",".join(str(int(i)) for i in ids) + "};",
             "static const float INV_FREQ[HD/2] = {" + ",".join(repr(float(x)) for x in inv_freq) + "};"]
    if q8:
        lines += [f"#define WOFF_LM_HEAD_W {off('lm.lm_head' + sfx)}UL",
                  f"#define WOFF_LM_HEAD_S {off('lm.lm_head' + scl)}UL",
                  "struct layer_off { size_t input_ln, q_proj_w,q_proj_s, k_proj_w,k_proj_s,"
                  " v_proj_w,v_proj_s, o_proj_w,o_proj_s, post_ln,"
                  " gate_w,gate_s, up_w,up_s, down_w,down_s; };",
                  "static const struct layer_off LAYERS[NL] = {"]
        for L in range(NL):
            p = f"lm.model.layers.{L}"
            fields = [f"{p}.input_layernorm.weight"]
            for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.o_proj"):
                fields += [f"{p}.{proj}{sfx}", f"{p}.{proj}{scl}"]
            fields += [f"{p}.post_attention_layernorm.weight"]
            for proj in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                fields += [f"{p}.{proj}{sfx}", f"{p}.{proj}{scl}"]
            lines.append("  {" + ",".join(str(off(f)) + "UL" for f in fields) + "},")
    else:
        lines += [f"#define WOFF_LM_HEAD {off('lm.lm_head.weight')}UL",
                  "struct layer_off { size_t input_ln, q_proj, k_proj, v_proj, o_proj,"
                  " post_ln, gate_proj, up_proj, down_proj; };",
                  "static const struct layer_off LAYERS[NL] = {"]
        for L in range(NL):
            p = f"lm.model.layers.{L}"
            lines.append("  {" + ",".join(str(off(f"{p}.{s}")) + "UL" for s in (
                "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                "mlp.down_proj.weight")) + "},")
    lines += ["};", ""]
    h = out_dir / "llama_weights.h"
    h.write_text("\n".join(lines))
    return h


def build_glue_binary(bundle_root: Path, work: Path, variant: str = "fp32") -> tuple[Path, list[str]]:
    """Compile the EXO GEMM + C glue into a K1 rv64gcv ELF. Returns (elf, march_flags).

    fp32 -> the f32 vfmacc GEMM (gemm.py) + llama_glue.c. int8 -> the vwmacc widening GEMM
    (igemm.py) + llama_glue_int8.c. Raises with a specific reason if the capture is not a runnable
    llama decoder (surfaced by the caller as a not_built gap)."""
    work.mkdir(parents=True, exist_ok=True)
    cc = k1.toolchain_cc()
    if cc is None:
        raise RuntimeError("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")
    emit_weights_header(bundle_root, work, variant)   # raises if not a llama stack
    flags = [f"-march={MARCH}", f"-mabi={MABI}"]
    rvv_audit.enforce_rvv_march(MARCH)
    if variant == "int8":
        kernel_c = compile_exo_igemm(work)
        glue_c = _exo_dir() / "llama_glue_int8.c"
        elf = work / "exo_llama_int8_k1"
    else:
        kernel_c = compile_exo_gemm(work)
        glue_c = _exo_dir() / "llama_glue.c"
        elf = work / "exo_llama_k1"
    cmd = [str(cc), "--target=riscv64-unknown-linux-gnu", *flags, "-O3", "-ffast-math",
           f"-I{work}", str(glue_c), str(kernel_c), "-lm", "-lpthread", "-o", str(elf)]
    try:
        _run(cmd + ["-static"])
    except RuntimeError:
        _run(cmd)
    if not elf.is_file():
        raise RuntimeError("glue link produced no binary")
    return elf, flags


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")


def _parse_out(stdout: str) -> np.ndarray | None:
    for line in stdout.splitlines():
        if line.startswith("OUT "):
            parts = line.split()
            k = int(parts[1])
            bits = np.array([int(x) for x in parts[2:2 + k]], dtype=np.uint32)
            return bits.view(np.float32)
    return None


def _scalar_fallbacks() -> list[ScalarFallback]:
    return [ScalarFallback(symbol=sym, reason="no EXO RVV kernel — scalar glue", region=reg)
            for sym, reg in _SCALAR_GLUE]


def run(model: str = "tiny_llama", variant: str = "fp32") -> BaselineResult:
    """Build + (board-locked) run the EXO-glue whole-model on the K1; return a BaselineResult."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = _submodule_sha()
    res = BaselineResult(framework=FRAMEWORK, model=model, variant=variant, substrate="k1_spacemit",
                         march=MARCH, toolchain="spacemit-clang-19+exo-1.0.0",
                         framework_commit=commit, timestamp=ts, cycle_accurate=False)
    kdesc = ("int8 vwmacc i16xi16->i32 widening MAC (igemm_nt_ref)" if variant == "int8"
             else "f32 vfmacc.vf GEMM (gemm_nt_ref)")
    res.notes = (f"whole-model = EXO RVV GEMM kernel [{kdesc}] + hand C glue runtime; EXO is a "
                 "kernel DSL/scheduler, NOT a whole-model compiler. Only the nn.Linear GEMM is an "
                 "EXO kernel; RMSNorm/RoPE/attention-softmax/SwiGLU/residual/embed + (int8) "
                 "activation-quant/requant are scalar C glue (labeled).")
    res.scalar_fallbacks = _scalar_fallbacks()
    if variant == "int8":
        res.scalar_fallbacks.append(
            ScalarFallback("activation_quant_requant", "no EXO RVV kernel — scalar glue", "other"))

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
    try:
        elf, flags = build_glue_binary(b.root, work, variant)
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
        ksym = "igemm_nt_ref" if variant == "int8" else "gemm_nt_ref"
        gk = rep.by_symbol.get(ksym)
        if gk is not None and gk.coverage is not None:
            res.notes += (f" | EXO-GEMM kernel RVV coverage={gk.coverage:.2f} "
                          f"(vec={gk.vector} scalar={gk.scalar_compute}); whole-ELF coverage is "
                          f"libc-dominated ({res.rvv_coverage_overall:.3f}).")
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
            proc = k1_exec.run([f"MERLIN_WEIGHTS={remote_w} {remote}"], timeout=1800)
            try:
                k1_exec.run([f"rm -f {remote}"], timeout=60)  # keep cached weights for reruns
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        res.gap_reason = f"K1 run failed: {type(e).__name__}: {str(e)[:300]}"
        return res.validate()

    stdout = proc.stdout + proc.stderr
    out = _parse_out(stdout)
    if out is None:
        res.gap_reason = f"no OUT marker in board console (rc={proc.returncode}): {stdout[-300:]}"
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
