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

VLA models are OUT of ggml scope
--------------------------------
The VLA-specific models (``openvla``, ``rdt``, ``rdt2``, ``molmoact``, ``groot_n1d7``, ``xr0``,
``pi05``, ``smolvla``) have no GGUF converter / no llama.cpp architecture — they are diffusion /
action-head / multi-modal graphs, not causal LMs. They are recorded as explicit ``not_built`` gaps
with that reason, never forced or omitted.
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
from merlin.common.paths import repo_root
from merlin.rvvgen import k1

FRAMEWORK = "ggml"

# --- llama.cpp build layout (build tree gitignored; built by this arm) --------------------------
_BUILD_ROOT = repo_root() / "build" / "baselines" / "ggml"
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

# VLA-specific models: no llama.cpp architecture / no GGUF converter. Explicit gaps, never forced.
VLA_OUT_OF_SCOPE = ("openvla", "rdt", "rdt2", "molmoact", "groot_n1d7", "xr0", "pi05", "smolvla")

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
                        "/scratch/agustin/projects/model2MLIR/.venv/bin/python")
    if not Path(py).is_file():
        raise GgmlError(f"GGUF-conversion python not found: {py} (set MERLIN_GGUF_PYTHON)")
    conv = _LLAMA_SRC / "convert_hf_to_gguf.py"
    env = dict(os.environ, PYTHONPATH=f"{_LLAMA_SRC / 'gguf-py'}:{os.environ.get('PYTHONPATH','')}")
    r = subprocess.run([py, str(conv), str(snap), "--outfile", str(out), "--outtype", outtype],
                       capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0 or not out.is_file():
        raise GgmlError(f"convert_hf_to_gguf failed for {model}: {r.stderr[-400:]}")
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


@dataclass
class RvvAudit:
    coverage_overall: float | None = None          # whole-.so compute-insn coverage (diluted by glue)
    coverage_kernels: float | None = None          # coverage over ggml COMPUTE kernels only
    coverage_active_quant: float | None = None      # inner gemm/gemv/vec_dot for the q4_K/q8_K path
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
                    coverage_active_quant=cov_aq, fallbacks=fallbacks, region_coverage=region_cov)


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

def run_model(model: str, variant: str = "fp32", *, quant: str = "q4_K_M",
              write: bool = True, run_board: bool | None = None) -> BaselineResult:
    """Run one model through the ggml arm end-to-end and return a BaselineResult.

    Re-runnable and fail-closed: builds/audits happen regardless of the board; the on-board
    llama-bench is the only board-gated step. Correctness vs our golden is UNCOMPARABLE (see module
    docstring) so ``cos`` stays None with an explicit note — never a fabricated pass.
    """
    cos_thr, rel_thr = _bundle.tolerance(model)
    res = BaselineResult(framework=FRAMEWORK, model=model, variant=variant,
                         substrate="k1_spacemit", cos_threshold=cos_thr, rel_threshold=rel_thr,
                         march=GGML_MARCH, toolchain="llama.cpp+ggml(RVV) / spacemit-gcc-14.3",
                         framework_commit=ggml_commit(), timestamp=artifacts.utc_stamp())

    # VLA models are out of ggml scope — explicit not_built gap, never forced.
    if model in VLA_OUT_OF_SCOPE:
        res.gap_reason = (f"{model} is a VLA/diffusion/multimodal graph with no llama.cpp "
                          f"architecture / no GGUF converter — out of ggml scope")
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
        res.notes += (f" rvv_active_quant_cov(q4_K/q8_K)={aud.coverage_active_quant} "
                      f"rvv_all_kernels_cov={aud.coverage_kernels} rvv_whole_so={aud.coverage_overall} "
                      f"region_cov={aud.region_coverage} so={so.name};")
    except Exception as e:  # noqa: BLE001
        res.notes += f" rvv-audit failed: {str(e)[:150]};"

    # GGUF — only tiny_llama has a real GGUF-convertible checkpoint. The others are honest gaps.
    if model not in _HF_CHECKPOINTS:
        if model == "small_llama":
            reason = ("small_llama is a bespoke random-init toy transformer (vocab 256, 2 layers) — "
                      "no HF/GGUF architecture; llama.cpp cannot ingest it")
        elif model == "bitvla":
            reason = ("bitvla capture unit is an inputs_embeds forward of a small random "
                      "BitNet+SigLIP config — not a token-in GGUF causal LM; no llama.cpp arch")
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

    # Correctness is UNCOMPARABLE to our golden (real full checkpoint vs our 2-layer random-init
    # capture; ggml tokenizer-driven token-in forward vs our seeded-random-token/embeds capture).
    res.cos = None
    res.notes += (" correctness: cos=None UNCOMPARABLE — ggml runs the REAL full TinyLlama-1.1B "
                  "checkpoint (token-in, own tokenizer) while our golden.npy is a 2-layer "
                  "random-init capture on seeded random tokens; whole-model output fidelity vs our "
                  "golden is not verifiable through ggml (not_run_is_not_pass: no fabricated pass);")

    # On-board llama-bench — the only board-gated step. Fail-closed when the board is down.
    do_board = k1_exec.board_available() if run_board is None else run_board
    if do_board:
        try:
            _run_bench_on_board(res, model, gguf_f16, quant)
        except k1_exec.BoardUnavailable as e:
            res.gap_reason = res.gap_reason or f"K1 board run failed: {str(e)[:200]}"
        except Exception as e:  # noqa: BLE001
            res.gap_reason = res.gap_reason or f"K1 ggml bench error: {str(e)[:200]}"
    else:
        res.gap_reason = res.gap_reason or "K1 board unavailable (MERLIN_K1_HOST unset)"

    res.board_vlenb = k1_exec.board_vlenb()
    return _finish(res, model, variant, write)


def _run_bench_on_board(res: BaselineResult, model: str, gguf_f16: Path, quant: str) -> None:
    """Deploy + quantize (on-board) + llama-bench under the board lock; fill E2E + region profile."""
    with k1_exec.board_lock():
        deploy_runtime()
        # Push the f16 GGUF (idempotent-ish: skip if already present with matching size).
        remote_f16 = f"{_BOARD_DIR}/{gguf_f16.name}"
        want = gguf_f16.stat().st_size
        chk = _board_sh(f"stat -c %s {remote_f16} 2>/dev/null || echo 0", timeout=60)
        have = int((chk.stdout.strip() or "0").splitlines()[-1]) if chk.stdout.strip() else 0
        if have != want:
            subprocess.run(["scp", *_ssh_opts(), str(gguf_f16), f"{k1_exec.K1_HOST}:{remote_f16}"],
                           check=True, capture_output=True, timeout=1800)
        # Quantize on-board to a q4_K model (exercises the well-vectorized q4_K RVV GEMM kernels)
        # so the benchmark stays comfortably within the 3.4G board RAM. q4_K also matches ggml's
        # BitNet-adjacent low-bit story for the (out-of-scope) ternary models.
        remote_q = f"{_BOARD_DIR}/{model}-{quant}.gguf"
        ld = f"LD_LIBRARY_PATH={_BOARD_DIR}/lib:/usr/lib/riscv64-linux-gnu"
        qchk = _board_sh(f"test -s {remote_q} && echo yes || echo no", timeout=60)
        run_gguf = remote_q
        if "yes" not in qchk.stdout:
            qz = _board_sh(f"{ld} {_BOARD_DIR}/llama-quantize {remote_f16} {remote_q} {quant}",
                           timeout=1800)
            if "yes" not in _board_sh(f"test -s {remote_q} && echo yes || echo no").stdout:
                res.notes += f" on-board quantize->{quant} failed, benchmarking f16;"
                run_gguf = remote_f16
        bench, raw = run_on_board(run_gguf, n_threads=k1.K1_OMP_THREADS)

    res.notes += f" bench_gguf={Path(run_gguf).name};"
    res.ran = bench["rc"] == 0 and bool(bench["tps"])
    if not res.ran:
        res.gap_reason = res.gap_reason or f"llama-bench produced no throughput (rc={bench['rc']}): {raw[-200:]}"
        return
    if bench.get("wall_ns"):
        res.e2e_wall_ns = int(bench["wall_ns"])
    # Region profile from llama-bench's two phases: prompt-processing (pp, GEMM-bound) and
    # token-generation (tg, GEMV/attention-bound). We record throughput (tokens/s) as the region
    # metric via note (contract has no tps field); rdtime cycles stay None (llama-bench times
    # internally; we do not have an rdtime asm bracket around it — honest, not fabricated).
    kcov = None
    for tok in res.notes.split():
        if tok.startswith("rvv_kernels_cov="):
            v = tok.split("=", 1)[1].rstrip(";")
            kcov = float(v) if v not in ("None", "") else None
    regions = []
    for test, tps in bench["tps"].items():
        reg = "gemm" if test.startswith("pp") else "attention"
        regions.append(RegionProfile(name=reg, rvv_coverage=kcov, calls=None,
                                     note=f"llama-bench {test}: {tps} tokens/s (rdtime cycles n/a — "
                                          f"llama-bench times internally)"))
    res.regions = regions
    tps_str = ", ".join(f"{t}={v}tok/s" for t, v in bench["tps"].items())
    res.notes += f" llama_bench_throughput[{tps_str}];"


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
    ap.add_argument("--quant", default="q4_K_M", help="on-board quantization (default q4_K_M)")
    ap.add_argument("--no-write", action="store_true", help="do not write BaselineResult artifacts")
    args = ap.parse_args(argv)
    results = []
    for m in (args.models or list(DEFAULT_MODELS)):
        results.append(run_model(m, args.variant, quant=args.quant, write=not args.no_write))
    for r in results:
        cov = f"{100*r.rvv_coverage_overall:.0f}%RVV" if r.rvv_coverage_overall is not None else "?RVV"
        print(f"{r.model}/{r.variant}: {r.status():10s} {cov} "
              f"fallbacks={len(r.scalar_fallbacks)} {r.gap_reason}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
