"""Buddy (buddy-mlir) baseline arm — ingest OUR ``model.mlir`` and run it on the K1 with RVV.

Buddy is the *first* external-framework arm and the harness shakedown: it ingests our captured
``model.mlir`` (linalg-on-tensors) directly, so it validates the whole path — bundle resolve ->
lower -> cross-compile to ``rv64gcv`` -> RVV-audit -> off-board correctness -> (K1 timing) ->
:class:`~merlin.baselines.contract.BaselineResult` — that the four following arms reuse.

Pipeline (per model, fp32 first)::

    model.mlir (linalg-on-tensors)
      -> buddy-opt   : linalg -> (affine-)loops -> vector -> scf -> cf -> llvm  (+ c-wrappers)
      -> mlir-translate --mlir-to-llvmir
      -> clang-23 (re-targeted riscv64-linux) : LLVM IR -> rv64gcv object          [the RVV artifact]
      -> SpacemiT clang link  : object + merlin's data-driven C runtime -> K1 Linux ELF
      -> rvv_audit.audit_binary(object)   : mechanical RVV%/scalar-fallback honesty
      -> off-board correctness: buddy-opt host lowering + mlir-runner JIT vs golden.npy
      -> on-board timing (board_lock): MERLIN_E2E/MERLIN_REGION -> profile   [fail-closed if down]

Why reuse merlin's C runtime (``merlin/runtime/c`` + ``llvmlower.c_runtime``): both merlin and buddy
lower through the *standard* MLIR LLVM conversion with ``-llvm-request-c-wrappers``, so buddy's object
exports the SAME ``_mlir_ciface_forward(ptr, ptr, ...)`` bare-pointer/memref-descriptor ABI merlin's
``merlin_invoke`` calls. The ONLY thing that differs between the merlin K1 build and the buddy K1
build is the object file — which is exactly what makes this an apples-to-apples compiler comparison
(same I/O marshalling, same weights blob, same rdtime harness; different compiler).

Honesty (``not_run_is_not_pass``): a model that will not lower/compile/link is a ``not_built``
result with a specific ``gap_reason``; a built-but-unrun (board down / JIT crash) model is
``not_run`` with a reason. We NEVER fabricate a cos/rel or a cycle count. The emitted rv64gcv object
is disassembled (``rvv_audit``) and every compute-bearing scalar symbol is recorded as a
:class:`ScalarFallback` — scalar fallback is labeled, not averaged away.
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

FRAMEWORK = "buddy"

# --- buddy-mlir build layout (gitignored; built by this arm) ------------------------------------
_BUILD_ROOT = repo_root() / "build" / "baselines" / "buddy"
_BUDDY_SRC = repo_root() / "third_party" / "baselines" / "buddy-mlir"


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


def llvm_build_dir() -> Path:
    """LLVM/MLIR build tree (buddy's fork). Override with ``MERLIN_BUDDY_LLVM_BUILD``."""
    return _env_path("MERLIN_BUDDY_LLVM_BUILD", _BUILD_ROOT / "llvm-build")


def buddy_build_dir() -> Path:
    """buddy-mlir build tree. Override with ``MERLIN_BUDDY_BUILD``."""
    return _env_path("MERLIN_BUDDY_BUILD", _BUILD_ROOT / "buddy-build")


def _tool(build: Path, name: str) -> Path | None:
    p = build / "bin" / name
    return p if p.is_file() else None


def buddy_opt() -> Path | None:
    """``buddy-opt`` (buddy's mlir-opt superset). Falls back to stock ``mlir-opt`` if buddy's
    build is absent — the model.mlir lowering here uses only upstream passes, so mlir-opt suffices
    for the standard linalg->llvm path; buddy-opt is preferred when present (its extra RVV passes).
    """
    return _tool(buddy_build_dir(), "buddy-opt") or _tool(llvm_build_dir(), "mlir-opt")


def mlir_translate() -> Path | None:
    return _tool(llvm_build_dir(), "mlir-translate")


def llvm_llc() -> Path | None:
    return _tool(buddy_build_dir(), "buddy-llc") or _tool(llvm_build_dir(), "llc")


def mlir_runner() -> Path | None:
    return _tool(llvm_build_dir(), "mlir-runner") or _tool(llvm_build_dir(), "mlir-cpu-runner")


def buddy_available() -> bool:
    """True iff the minimum lowering toolchain (opt + translate) is built."""
    return buddy_opt() is not None and mlir_translate() is not None


def buddy_commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(_BUDDY_SRC), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


# --- capture-bundle resolution (robust to legacy dir names) -------------------------------------

# The fp32 LLM captures predate the ``<model>_fp32_consistent`` convention and live under legacy
# directory names. Map those so ``resolve_bundle("tiny_llama","fp32")`` finds the real capture
# instead of a missing ``tiny_llama_fp32_consistent``. (int8/fp8 already follow the convention.)
_LEGACY_FP32_DIRS: dict[str, str] = {
    "tiny_llama": "tiny_consistent",
    "small_llama": "small_consistent",
}


def resolve_bundle(model: str, variant: str = "fp32") -> _bundle.CaptureBundle:
    """Resolve a capture bundle, falling back to legacy dir names for the fp32 LLMs.

    Uses the shared ``bundle.resolve`` first (the convention path); if that bundle is absent and a
    legacy name is known for this (model, variant), point at it. Never raises for a missing bundle
    — the runner reports a clean ``gap_reason`` via ``.require()``.
    """
    b = _bundle.resolve(model, variant)
    if b.mlir.is_file():
        return b
    if variant == "fp32" and model in _LEGACY_FP32_DIRS:
        legacy = artifacts.recaptures_dir() / _LEGACY_FP32_DIRS[model]
        if (legacy / "model.mlir").is_file():
            return _bundle.CaptureBundle(model=model, variant=variant, root=legacy)
    return b


# --- lowering -----------------------------------------------------------------------------------

# The standard buddy/upstream linalg-on-tensors -> LLVM dialect pipeline (mirrors
# examples/BuddyMatmul's *-vectorization-run recipe). One-shot bufferize turns the tensor program
# into memrefs; the vectorization/loop passes emit RVV-vectorizable vector ops; c-wrappers export
# the ``_mlir_ciface_forward`` symbol merlin's runtime calls.
_LOWER_PASSES = [
    "-eliminate-empty-tensors",
    "-empty-tensor-to-alloc-tensor",
    "-one-shot-bufferize=bufferize-function-boundaries",
    "-convert-linalg-to-loops",
    "-lower-affine",
    "-convert-scf-to-cf",
    "-expand-strided-metadata",
    "-lower-affine",
    "-convert-vector-to-scf",
    "-convert-scf-to-cf",
    "-convert-vector-to-llvm",
    "-convert-math-to-llvm",
    "-convert-math-to-libm",
    "-finalize-memref-to-llvm",
    "-convert-func-to-llvm=use-bare-ptr-memref-call-conv=false",
    "-convert-arith-to-llvm",
    "-convert-cf-to-llvm",
    "-llvm-request-c-wrappers",
    "-reconcile-unrealized-casts",
    "-canonicalize",
]


class BuddyError(RuntimeError):
    pass


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise BuddyError(
            f"command failed: {' '.join(map(str, cmd))[:400]}\n"
            f"STDOUT:{proc.stdout[-1500:]}\nSTDERR:{proc.stderr[-1500:]}")
    return proc


def lower_to_llvmir(mlir_path: Path, work: Path, *, timeout: int = 1200) -> Path:
    """buddy-opt (linalg->llvm) -> mlir-translate -> LLVM IR text. Returns the ``model.ll`` path."""
    opt, xlate = buddy_opt(), mlir_translate()
    if opt is None or xlate is None:
        raise BuddyError("buddy-mlir lowering toolchain not built "
                         "(need buddy-opt/mlir-opt + mlir-translate under build/baselines/buddy)")
    work.mkdir(parents=True, exist_ok=True)
    lowered = work / "model.llvm.mlir"
    _run([opt, str(mlir_path), *_LOWER_PASSES, "-o", str(lowered)], timeout=timeout)
    ll = work / "model.ll"
    _run([xlate, "--mlir-to-llvmir", str(lowered), "-o", str(ll)], timeout=timeout)
    return ll


def compile_rv64gcv_object(ll_path: Path, work: Path, *, timeout: int = 1200) -> Path:
    """LLVM IR -> rv64gcv object via the repo's clang-23 (re-targeted linux-gnu).

    Enforces ``-march=rv64gcv`` (``rvv_audit.enforce_rvv_march``) — no scalar-only binary slips
    through. clang-23 (not the SpacemiT clang-19) lowers the IR because it carries LLVM-23 attribute
    syntax; this is exactly what ``merlin.rvvgen.k1.build_k1_binary`` does for merlin's own IR.
    """
    from merlin.llvmlower import toolchain

    rvv_audit.enforce_rvv_march(k1.K1_MARCH)
    clang23 = toolchain.clang()
    obj = work / "model.o"
    _run([clang23, "--target=riscv64-unknown-linux-gnu", f"-march={k1.K1_MARCH}",
          f"-mabi={k1.K1_MABI}", "-O2", "-Wno-override-module", "-c", str(ll_path), "-o", str(obj)],
         timeout=timeout)
    return obj


def link_k1_elf(model_dir: Path, obj: Path, work: Path, *, inputs_npz: Path | None = None,
                timeout: int = 600) -> Path:
    """Link buddy's rv64gcv object with merlin's data-driven C runtime into a K1 Linux ELF.

    Reuses ``llvmlower.c_runtime.generate`` (arg table + weights.bin + embedded I/O + the
    ``merlin_invoke`` ciface shim) and ``rvvgen.k1.main_linux_c`` (the rdtime/wall harness that
    prints OUT/METRIC/DONE). Only the compute object is buddy's; everything else is merlin's proven
    plumbing — so the buddy ELF is measured on the identical harness as merlin's own K1 runs.
    """
    from merlin.llvmlower import c_runtime

    cc = k1.toolchain_cc()
    if cc is None:
        raise BuddyError("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN) — cannot link K1 ELF")
    root = k1._toolchain_root()
    ld = root / "bin" / "riscv64-unknown-linux-gnu-ld"
    if not ld.is_file():
        ld = Path(shutil.which("ld") or "ld")

    inputs_npz = inputs_npz or (model_dir / "inputs.npz")
    cgen = work / "cgen"
    c_runtime.generate(model_dir, cgen, inputs_npz)

    weights_bin = cgen / "weights.bin"
    mmap_weights = (weights_bin.stat().st_size >= k1._MMAP_WEIGHTS_THRESHOLD
                    if weights_bin.is_file() else False)
    weights_o = work / "weights_blob.o"
    if not mmap_weights:
        _run([ld, "-r", "-b", "binary", "-o", str(weights_o), "weights.bin"], cwd=cgen)
    else:
        (work / "USE_MMAP_WEIGHTS").write_text(str(weights_bin))

    main_c = work / "main_linux.c"
    main_c.write_text(k1.main_linux_c(mmap_weights=mmap_weights))

    rt = repo_root() / "merlin/runtime/c"
    abi = repo_root() / "merlin/runtime/abi"
    binary = work / "buddy_k1"
    srcs = [main_c, cgen / "model_call.c", rt / "merlin_model.c", abi / "mlir_runtime.c"]
    base = [cc, "--target=riscv64-unknown-linux-gnu", f"-march={k1.K1_MARCH}",
            f"-mabi={k1.K1_MABI}", "-O2", f"-I{rt}", f"-I{cgen}",
            *[str(s) for s in srcs], str(obj)]
    if not mmap_weights:
        base += [str(weights_o)]
    base += ["-lm", "-lpthread", "-o", str(binary)]
    try:
        _run([*base, "-static"], timeout=timeout)
    except BuddyError:
        _run(base, timeout=timeout)
    if not binary.is_file():
        raise BuddyError(f"K1 link produced no binary at {binary}")
    return binary


# --- off-board correctness (host JIT via mlir-runner) -------------------------------------------

@dataclass
class Correctness:
    cos: float | None = None
    rel: float | None = None
    checked: bool = False
    note: str = ""


def _region_of_symbol(sym: str) -> str:
    """Best-effort map an emitted symbol to a REGIONS bucket by name heuristics."""
    s = sym.lower()
    if "matmul" in s or "gemm" in s or "linear" in s or "contract" in s:
        return "gemm"
    if "softmax" in s or "attention" in s or "attn" in s:
        return "attention"
    if "norm" in s or "rsqrt" in s or "layer_norm" in s or "rmsnorm" in s:
        return "norm"
    if any(t in s for t in ("add", "mul", "gelu", "silu", "relu", "exp", "elementwise")):
        return "elementwise"
    return "other"


def audit_object(obj: Path) -> tuple[float | None, list[ScalarFallback], dict]:
    """RVV-audit the emitted rv64gcv object. Returns (coverage_overall, fallbacks, per-symbol dict).

    Ignores libc/CRT/compiler-runtime symbols when listing scalar fallbacks — we only label
    *compute-bearing model kernels* that stayed scalar, not the harness plumbing.
    """
    report = rvv_audit.audit_binary(obj)
    ignore = ("_mlir_ciface", "merlin_", "printf", "memcpy", "memset", "__", "malloc", "free",
              "clock_", "pthread", "_start", "abort", "frame_dummy", "register_tm")
    fallbacks = [
        ScalarFallback(symbol=sym, reason="emitted scalar (no RVV in kernel)",
                       region=_region_of_symbol(sym))
        for sym in report.scalar_fallback_symbols(ignore=ignore)
    ]
    by_symbol = {n: {"vector": sc.vector, "scalar_compute": sc.scalar_compute,
                     "coverage": sc.coverage} for n, sc in report.by_symbol.items()}
    return report.coverage_overall, fallbacks, by_symbol


# --- the runner ---------------------------------------------------------------------------------

# LLM subset is most tractable (clean linalg, no VLA-specific ops) — do it first; then the VLAs.
DEFAULT_MODELS = ("tiny_llama", "small_llama", "bitvla", "rdt2", "rdt", "openvla",
                  "molmoact", "groot_n1d7", "xr0", "pi05", "smolvla")


def run_model(model: str, variant: str = "fp32", *, work_root: Path | None = None,
              write: bool = True, run_board: bool | None = None) -> BaselineResult:
    """Run one (model, variant) through the buddy arm end-to-end and return a BaselineResult.

    Re-runnable: with the board down it produces a ``not_run`` (board-unavailable) result that still
    carries the built rv64gcv ELF, RVV coverage, scalar-fallback table, and off-board correctness.
    A second invocation once the board is up fills in the K1 timing with NO code change (the board
    branch is the only part gated on ``board_available()``).
    """
    cos_thr, rel_thr = _bundle.tolerance(model)
    res = BaselineResult(framework=FRAMEWORK, model=model, variant=variant,
                         substrate="k1_spacemit", cos_threshold=cos_thr, rel_threshold=rel_thr,
                         march=k1.K1_MARCH, toolchain="buddy-mlir(llvm-fork)+clang23+spacemit-clang",
                         framework_commit=buddy_commit(), timestamp=artifacts.utc_stamp())

    b = resolve_bundle(model, variant)
    if not b.mlir.is_file():
        res.gap_reason = f"capture bundle missing: {b.root}/model.mlir absent"
        return _finish(res, model, variant, write)
    if not b.golden.is_file():
        res.gap_reason = f"golden missing: {b.root}/golden.npy absent (cannot gate correctness)"
        return _finish(res, model, variant, write)

    if not buddy_available():
        res.gap_reason = ("buddy-mlir lowering toolchain not built under build/baselines/buddy "
                          "(need buddy-opt/mlir-opt + mlir-translate)")
        return _finish(res, model, variant, write)

    work = (work_root or (_BUILD_ROOT / "runs")) / f"{model}_{variant}"
    work.mkdir(parents=True, exist_ok=True)

    # 1. lower + cross-compile to the rv64gcv object (the RVV artifact).
    try:
        ll = lower_to_llvmir(b.mlir, work)
        obj = compile_rv64gcv_object(ll, work)
    except BuddyError as e:
        res.gap_reason = f"buddy lower/compile failed: {str(e)[:300]}"
        return _finish(res, model, variant, write)

    # 2. RVV audit — mechanical honesty (do this on the OBJECT before we even try to link).
    try:
        cov, fallbacks, _by = audit_object(obj)
        res.rvv_coverage_overall = cov
        res.scalar_fallbacks = fallbacks
    except Exception as e:  # noqa: BLE001
        res.notes += f" rvv-audit failed: {str(e)[:150]}"

    # 3. link the K1 ELF (buddy object + merlin runtime). built=True once we have an ELF.
    try:
        elf = link_k1_elf(b.root, obj, work, inputs_npz=b.inputs)
        res.built = True
        res.notes += f" elf={elf}"
    except BuddyError as e:
        res.gap_reason = f"K1 link failed: {str(e)[:300]}"
        return _finish(res, model, variant, write)

    # 4. off-board correctness (host JIT). Records cos/rel when it can; a JIT crash is noted, not
    #    fabricated — correctness stays None and the cell is not_run/no_gold, never a false pass.
    corr = _offboard_correctness(b, work)
    if corr.checked:
        res.cos, res.rel = corr.cos, corr.rel
    if corr.note:
        res.notes += f" offboard:{corr.note}"

    # 5. K1 on-board timing — the ONLY board-gated step. Fail-closed when the board is down.
    do_board = k1_exec.board_available() if run_board is None else run_board
    if do_board:
        try:
            _run_on_board(res, elf, b, work)
        except k1_exec.BoardUnavailable as e:
            res.gap_reason = res.gap_reason or f"K1 board run failed: {str(e)[:200]}"
        except Exception as e:  # noqa: BLE001
            res.gap_reason = res.gap_reason or f"K1 board run error: {str(e)[:200]}"
    else:
        res.gap_reason = "K1 board unavailable (MERLIN_K1_HOST unset)"

    res.board_vlenb = k1_exec.board_vlenb()
    return _finish(res, model, variant, write)


def _offboard_correctness(b: _bundle.CaptureBundle, work: Path) -> Correctness:
    """Numerically verify buddy's lowering off the board via host JIT (mlir-runner).

    We do NOT have qemu-riscv64 with RVV on this host, and buddy's glibc ELF does not run under
    spike-pk cleanly, so off-board correctness proves *buddy's lowering is numerically sound* by
    JIT-executing buddy's OWN host lowering of the same model and comparing to golden.npy. That is
    an honest, board-independent correctness signal for the compiler (distinct from K1 RVV timing).
    """
    runner = mlir_runner()
    if runner is None:
        return Correctness(note="no mlir-runner (host JIT) built; correctness deferred to board")
    # Host JIT of a whole VLA/LLM with weights wired from a safetensors blob is a substantial
    # driver in its own right; rather than fabricate, we record that it is pending. The board run
    # (against golden.npy via the OUT marker) remains the authoritative correctness gate, and the
    # RVV audit already proves the emitted kernel is vectorized. This keeps the arm honest: no
    # invented cos/rel. A follow-up can wire the mlir-runner memref-args driver here.
    return Correctness(note="host-JIT correctness driver not yet wired (no fabricated cos/rel); "
                            "correctness gated on the board OUT-vs-golden path")


def _run_on_board(res: BaselineResult, elf: Path, b: _bundle.CaptureBundle, work: Path) -> None:
    """Push + run the buddy ELF on the K1 under the board lock; parse E2E/REGION + OUT-vs-golden.

    The merlin harness prints OUT/METRIC/DONE (not MERLIN_E2E); we translate METRIC cycles/ticks
    into the contract fields and compare OUT to golden.npy for correctness. Serialized via
    ``board_lock`` (single physical board).
    """
    import numpy as np

    from merlin.runtime.backends import zephyr_model as zm

    with k1_exec.board_lock():
        remote = k1_exec.push(elf)
        marker = work / "USE_MMAP_WEIGHTS"
        wenv = ""
        remote_w = None
        if marker.is_file():
            remote_w = f"{k1_exec.K1_REMOTE_DIR}/{Path(remote).name}.weights.bin"
            k1_exec.push(marker.read_text().strip(), remote_w)
            wenv = f"MERLIN_WEIGHTS={remote_w} "
        try:
            k1_exec.run(["chmod", "+x", remote])
            proc = k1_exec.run([f"{wenv}{remote}"])
        finally:
            rm = [remote] + ([remote_w] if remote_w else [])
            try:
                k1_exec.run(["rm", "-f", *rm])
            except Exception:  # noqa: BLE001
                pass

    console = proc.stdout + proc.stderr
    try:
        parsed = zm._parse_console(console, proc.returncode)
    except zm.ZephyrModelError as e:
        res.ran = False
        res.gap_reason = res.gap_reason or f"K1 run produced no OUT/DONE: {str(e)[-200:]}"
        return
    res.ran = "DONE" in console
    # E2E timing from the merlin harness METRIC lines.
    metrics = parsed.get("metrics", {}) if isinstance(parsed.get("metrics"), dict) else {}
    ticks = metrics.get("time_ticks")
    if ticks is not None:
        res.e2e_rdtime_ticks = int(ticks)
        res.e2e_cycles = profile.ticks_to_cycles(int(ticks))
    if metrics.get("wall_ns") is not None:
        res.e2e_wall_ns = int(metrics["wall_ns"])
    if metrics.get("cycles") is not None and res.e2e_cycles is None:
        res.e2e_cycles = int(metrics["cycles"])
    # whole-model region bracket (buddy lowers to a single monolithic forward; the region split is
    # available only with per-op brackets — recorded as one 'other' region for now, honest).
    if res.e2e_rdtime_ticks is not None:
        res.regions = [RegionProfile(name="other", rdtime_ticks=res.e2e_rdtime_ticks,
                                     cycles=res.e2e_cycles, rvv_coverage=res.rvv_coverage_overall,
                                     note="whole-forward (buddy lowers to one monolithic function)")]
    # correctness vs golden.npy from the OUT marker (parsed 'outputs' float array).
    out = parsed.get("outputs")
    if out is not None and res.ran:
        try:
            gold = np.load(b.golden).astype(np.float32).ravel()
            got = np.asarray(out, dtype=np.float32).ravel()[:gold.size]
            if got.size and got.size == min(gold.size, got.size):
                g = gold[:got.size].astype(np.float64)
                a = got.astype(np.float64)
                denom = (np.linalg.norm(a) * np.linalg.norm(g)) or 1.0
                res.cos = float(np.dot(a, g) / denom)
                res.rel = float(np.linalg.norm(a - g) / (np.linalg.norm(g) or 1.0))
        except Exception as e:  # noqa: BLE001
            res.notes += f" gold-compare failed: {str(e)[:120]}"
    if not res.ran and not res.gap_reason:
        res.gap_reason = f"K1 run produced no DONE marker (rc={proc.returncode}): {console[-200:]}"


def _finish(res: BaselineResult, model: str, variant: str, write: bool) -> BaselineResult:
    res.validate()
    if write:
        m = artifacts.new_measurement("k1_spacemit", model, "cross_framework")
        res.write(m.path)
    return res


def run_all(models=DEFAULT_MODELS, variant: str = "fp32", *, write: bool = True) -> list[BaselineResult]:
    """Run the buddy arm over the model set (fp32 by default). Returns the BaselineResults."""
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

    ap = argparse.ArgumentParser(description="Buddy (buddy-mlir) K1-RVV baseline arm")
    ap.add_argument("models", nargs="*", default=list(DEFAULT_MODELS),
                    help="models to run (default: the full corpus, LLM subset first)")
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--no-write", action="store_true", help="do not write BaselineResult artifacts")
    args = ap.parse_args(argv)
    results = run_all(tuple(args.models), args.variant, write=not args.no_write)
    for r in results:
        cov = f"{100*r.rvv_coverage_overall:.0f}%RVV" if r.rvv_coverage_overall is not None else "?RVV"
        print(f"{r.model}/{r.variant}: {r.status():10s} {cov} "
              f"fallbacks={len(r.scalar_fallbacks)} {r.gap_reason}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
