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


def emit_weights_header(bundle_root: Path, out_dir: Path) -> Path:
    """Emit llama_weights.h — config + safetensors byte-offset table + input ids + rope inv_freq."""
    data0, hdr = _safetensors_offsets(bundle_root / "weights.safetensors")

    def off(name: str) -> int:
        return int(hdr[name]["data_offsets"][0])

    ids = np.load(bundle_root / "inputs.npz")["in0"][0].astype(np.int64)
    inv_freq = np.load(bundle_root / "extra.npz")["buf::lm.model.rotary_emb.inv_freq"].astype(np.float32)
    S = int(ids.shape[0])
    c = _CFG

    lines = ["/* generated by merlin.baselines.exo.emit_weights_header — do not edit. */",
             "#pragma once", "#include <stddef.h>", "#include <stdint.h>",
             f"#define NL {c['NL']}", f"#define H {c['H']}", f"#define NH {c['NH']}",
             f"#define NKV {c['NKV']}", f"#define HD {c['HD']}", f"#define FF {c['FF']}",
             f"#define V {c['V']}", f"#define S {S}",
             f"#define EPS {c['EPS']}f", f"#define THETA {c['THETA']}f",
             f"#define WDATA0 {data0}UL",
             f"#define WOFF_EMBED {off('lm.model.embed_tokens.weight')}UL",
             f"#define WOFF_FINAL_NORM {off('lm.model.norm.weight')}UL",
             f"#define WOFF_LM_HEAD {off('lm.lm_head.weight')}UL",
             "static const long INPUT_IDS[S] = {" + ",".join(str(int(i)) for i in ids) + "};",
             "static const float INV_FREQ[HD/2] = {" + ",".join(repr(float(x)) for x in inv_freq) + "};",
             "struct layer_off { size_t input_ln, q_proj, k_proj, v_proj, o_proj,"
             " post_ln, gate_proj, up_proj, down_proj; };",
             "static const struct layer_off LAYERS[NL] = {"]
    for L in range(c["NL"]):
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


def build_glue_binary(bundle_root: Path, work: Path) -> tuple[Path, list[str]]:
    """Compile the EXO GEMM + C glue into a K1 rv64gcv ELF. Returns (elf, march_flags)."""
    work.mkdir(parents=True, exist_ok=True)
    cc = k1.toolchain_cc()
    if cc is None:
        raise RuntimeError("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")
    gemm_c = compile_exo_gemm(work)
    emit_weights_header(bundle_root, work)
    glue_c = _exo_dir() / "llama_glue.c"
    flags = [f"-march={MARCH}", f"-mabi={MABI}"]
    rvv_audit.enforce_rvv_march(MARCH)  # refuse a non-+v build up front
    elf = work / "exo_llama_k1"
    cmd = [str(cc), "--target=riscv64-unknown-linux-gnu", *flags, "-O3", "-ffast-math",
           f"-I{work}", str(glue_c), str(gemm_c), "-lm", "-lpthread", "-o", str(elf)]
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
    res.notes = ("whole-model = EXO RVV GEMM kernel + hand C glue runtime; EXO is a kernel DSL/"
                 "scheduler, NOT a whole-model compiler. Only the nn.Linear GEMM is an EXO kernel; "
                 "RMSNorm/RoPE/attention-softmax/SwiGLU/residual/embed are scalar C glue (labeled).")
    res.scalar_fallbacks = _scalar_fallbacks()
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
        elf, flags = build_glue_binary(b.root, work)
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
        gk = rep.by_symbol.get("gemm_nt_ref")
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
    try:
        with k1_exec.board_lock():
            remote = k1_exec.push(elf, f"/tmp/{model}_{variant}_exo_llama_k1")
            remote_w = f"{k1.K1_REMOTE_DIR}/{model}_{variant}.weights.bin"
            k1_exec.push(b.weights, remote_w)
            k1_exec.run([f"chmod +x {remote}"], timeout=60)
            proc = k1_exec.run([f"MERLIN_WEIGHTS={remote_w} {remote}"], timeout=900)
            try:
                k1_exec.run([f"rm -f {remote} {remote_w}"], timeout=60)
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

    gold = np.load(b.golden).astype(np.float32).ravel()
    pred = out.astype(np.float32).ravel()[:gold.size]
    if pred.size == gold.size:
        cos = float(pred @ gold / (np.linalg.norm(pred) * np.linalg.norm(gold) + 1e-30))
        rel = float(np.abs(pred - gold).max() / (np.abs(gold).max() + 1e-9))
        res.cos, res.rel = cos, rel
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


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "tiny_llama"
    v = sys.argv[2] if len(sys.argv) > 2 else "fp32"
    d = run_and_record(m, v)
    print(f"wrote {d}")
