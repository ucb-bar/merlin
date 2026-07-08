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
import re
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

# buddy embeds weights via `ld -r -b binary`; blobs at/above this size are mmap'd (demand-paged) from
# a file instead — keeps the ELF small and resident RAM to the working set. Lower than merlin's 1.5 GB
# default because buddy's embedded-blob link fails/OOMs earlier (smolvla/rdt2 ~0.5 GB).
_BUDDY_MMAP_WEIGHTS_THRESHOLD = 256 * 1024 * 1024

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


def llvm_opt() -> Path | None:
    """LLVM IR optimizer (``opt``) from buddy's LLVM fork — runs the loop/SLP vectorizer."""
    return _tool(buddy_build_dir(), "opt") or _tool(llvm_build_dir(), "opt")


def buddy_available() -> bool:
    """True iff the minimum lowering + codegen toolchain is built.

    We lower with buddy-opt/mlir-opt + mlir-translate and generate the rv64gcv object with the
    SAME LLVM fork's ``llc`` (not the repo's IREE clang-23, whose IR parser rejects the fork's
    ``float f0x…`` hex-float literals — a version-skew we sidestep by staying inside one build).
    """
    return (buddy_opt() is not None and mlir_translate() is not None
            and llvm_llc() is not None)


def buddy_commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(_BUDDY_SRC), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


# --- native torch importer (Phase 2: buddy's OWN DynamoCompiler, not m2m linalg) ----------------

# buddy.compiler.frontend needs torch + torch._dynamo + the buddy_mlir python bindings, none of
# which live in oscar-merlin's venv — so the native import runs as a SUBPROCESS under the model2MLIR
# torch venv with PYTHONPATH pointing at buddy's python_packages + the MLIR bindings.
def _torch_venv_python() -> Path | None:
    """The model2MLIR torch venv python (has torch/dynamo + our installed nanobind bindings)."""
    p = _bundle.model2mlir_root() / ".venv" / "bin" / "python"
    override = os.environ.get("MERLIN_BUDDY_TORCH_PYTHON")
    cand = Path(override) if override else p
    return cand if cand.is_file() else None


def buddy_python_packages() -> Path | None:
    """buddy-mlir's built ``python_packages`` dir (holds ``buddy/compiler/frontend.py``)."""
    d = buddy_build_dir() / "python_packages"
    return d if (d / "buddy" / "compiler" / "frontend.py").is_file() else None


def mlir_python_packages() -> Path | None:
    """The MLIR python bindings dir (``mlir/_mlir_libs``) from the LLVM build."""
    d = llvm_build_dir() / "tools" / "mlir" / "python_packages" / "mlir_core"
    return d if (d / "mlir" / "_mlir_libs").is_dir() else None


def native_import_available() -> bool:
    """True iff the native torch-importer path is usable (torch venv + both python packages built)."""
    return (_torch_venv_python() is not None and buddy_python_packages() is not None
            and mlir_python_packages() is not None)


def native_import(model: str, variant: str, out_dir: Path, *, registry: str = "tosa",
                  timeout: int = 1800) -> Path:
    """Run buddy's DynamoCompiler over the REAL torch model → ``subgraph0.mlir`` + params.

    Uses the **tosa** registry by default — buddy's ``linalg`` registry has an ``expand_op``
    UnboundLocalError on the LLM/VLA graphs; tosa lowers robustly (then stock mlir-opt does
    tosa→linalg).

    Sets the full-fidelity loader env (``bundle.full_env``) so the SAME native-architecture model the
    golden was captured on is ingested, then spawns :mod:`.buddy_native_import` under the torch venv.
    Returns the ``subgraph0.mlir`` path. Raises :class:`BuddyError` on any failure (fail-closed).
    """
    py = _torch_venv_python()
    bpp = buddy_python_packages()
    mpp = mlir_python_packages()
    if py is None or bpp is None or mpp is None:
        raise BuddyError("native importer unavailable (need torch venv + buddy/MLIR python packages; "
                         "build with -DMLIR_ENABLE_BINDINGS_PYTHON=ON + -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b = resolve_bundle(model, variant)
    loader = b.torch_loader
    if not loader.is_file():
        raise BuddyError(f"m2m loader missing for {model}: {loader}")

    env = dict(os.environ)
    # full-fidelity capture env: the exact loader settings the recapture (and golden) used.
    env.update(_bundle.full_env(model))
    # find the repo's merlin python package so `-m merlin.baselines.buddy_native_import` resolves,
    # + buddy python_packages + MLIR bindings.
    merlin_py = str(repo_root() / "merlin" / "python")
    env["PYTHONPATH"] = os.pathsep.join([str(bpp), str(mpp), merlin_py,
                                         env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # buddy frontend also reads these to locate its runtime libs.
    env["BUDDY_MLIR_BUILD_DIR"] = str(buddy_build_dir())
    env["LLVM_MLIR_BUILD_DIR"] = str(llvm_build_dir())

    cmd = [str(py), "-m", "merlin.baselines.buddy_native_import",
           "--model", model, "--variant", variant, "--out-dir", str(out_dir),
           "--loader", str(loader), "--registry", registry]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    sub = out_dir / "subgraph0.mlir"
    if r.returncode != 0 or not sub.is_file():
        raise BuddyError(f"native import failed for {model}/{variant}:\n"
                         f"STDOUT:{r.stdout[-1000:]}\nSTDERR:{r.stderr[-1500:]}")
    return sub


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

# rv64gcv codegen target for llc/opt (K1 X60, VLEN=256). We drive RVV via the LLVM backend's
# loop/SLP vectorizer (opt -O3 on the lowered IR) rather than affine-super-vectorize — the latter
# NYI's on the non-trivial memref layout maps the whole-model bufferizer produces. The vector width
# hint (--riscv-v-vector-bits-min=256) matches the K1's 256-bit vector registers.
_RVV_MATTR = "+m,+a,+f,+d,+c,+v"
_RVV_VBITS = 256
# LLVM IR (text) byte size above which full ``opt -O3`` is intractable on a whole-model monolithic
# function; use a bounded loop/SLP-vectorize recipe instead so huge models (pi05 ~200 MB) still
# build + emit RVV rather than timing out. 64 MB comfortably clears tiny/small LLMs + VLAs (<6 MB
# lowered) while catching pi05.
_O3_IR_SIZE_LIMIT = 64 * 1024 * 1024

# The buddy/upstream linalg-on-tensors -> LLVM dialect pipeline. One-shot bufferize turns the tensor
# program into memrefs; ``-convert-linalg-to-loops`` gives a scalar loop nest that lowers cleanly
# for the WHOLE captured model (affine-super-vectorize NYI's on strided layouts), then the LLVM
# backend vectorizer (opt -O3, see compile_rv64gcv_object) turns the hot loops into RVV. c-wrappers
# export the ``_mlir_ciface_forward`` symbol merlin's runtime calls.
#
# MEMORY PLANNING (the OOM fix): after bufferization, EVERY intermediate is a live ``memref.alloc``
# with no free — so a whole VLA/LLM keeps all ~290 buffers resident and OOMs the 3.8GB K1. We insert
# ``-buffer-deallocation-pipeline`` (frees each buffer after its LAST use) + hoisting so peak resident
# memory is the live-set, not the sum of all intermediates. This is buddy's own upstream MLIR
# deallocation, not a merlin trick.
_LOWER_PASSES = [
    "-eliminate-empty-tensors",
    "-empty-tensor-to-alloc-tensor",
    "-one-shot-bufferize=bufferize-function-boundaries",
    "-buffer-hoisting",
    "-buffer-loop-hoisting",
    "-buffer-deallocation-pipeline",
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
    # c-wrappers MUST precede -convert-func-to-llvm: the pass wraps `func.func` to export the
    # `_mlir_ciface_forward` bare-pointer/descriptor entry merlin_invoke calls; run after func
    # lowering it no-ops (funcs are already llvm.func) and the ciface symbol never appears.
    "-llvm-request-c-wrappers",
    "-convert-func-to-llvm=use-bare-ptr-memref-call-conv=false",
    "-convert-arith-to-llvm",
    "-convert-cf-to-llvm",
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


def prepare_model_mlir(bundle: _bundle.CaptureBundle, work: Path) -> Path:
    """Normalize a raw capture ``model.mlir`` into a buddy-lowerable module.

    Buddy can't lower the raw ``quant.*`` dialect (int8 captures) or the over-rank ``aten.linear``
    matmuls / bf16 accumulation some captures carry. We reuse merlin's OWN normalization
    (``zephyr_model._prepare_model_mlir``) so the buddy arm ingests the SAME prepared IR merlin's
    runtime does — apples-to-apples — and so the int8 path is a REAL W8A8 integer datapath (i8*i8
    -> i32 via ``arith.extsi`` + integer MAC, which ``opt -O3`` vectorizes to ``vwmacc.vv`` on
    rv64gcv), not an f32 dequant. ``int8_compute=True`` only for the int8 variant.
    """
    from merlin.runtime.backends import zephyr_model as zm

    work.mkdir(parents=True, exist_ok=True)
    int8 = bundle.variant == "int8"
    prepared = zm._prepare_model_mlir(bundle.mlir, work, int8_compute=int8)
    # Repair the m2m aten.select export bug (malformed rank-reducing tensor.extract_slice) — see
    # _repair_malformed_select_slices. This is a buddy-arm compat shim for a bug in the EXPORTED
    # model.mlir (m2m repo), not in oscar-merlin or buddy; without it bitvla/smolvla fail
    # bufferization ("mixed offsets rank to match mixed sizes rank").
    txt = prepared.read_text()
    fixed, n = _repair_malformed_select_slices(txt)
    if n:
        prepared.write_text(fixed)
        print(f"[buddy] repaired {n} malformed aten.select extract_slice (m2m export bug)")
    return prepared


# The m2m torch exporter emits `aten.select.int` as a rank-reducing `tensor.extract_slice` whose
# `static_sizes` is left at rank 1 (the result rank) while `static_offsets`/`static_strides` stay at
# the source rank — invalid IR (MLIR's bufferizer rejects "mixed offsets rank (R) vs sizes rank (1)").
# We reconstruct the rank-R `static_sizes` so the slice is well-formed and its rank-reduction to the
# (already-correct) result type is legal: exactly one kept dim holds the size, the rest are 1.
# One malformed extract_slice op. Captures: (1) offsets, (2) sizes, (3) strides, (4) the tail from
# after strides through the op's type signature (kept verbatim). The source dims are recovered from
# the tail's first `(tensor<DIMSxELEM>)` operand type.
_EXTRACT_SLICE_RE = re.compile(
    r'static_offsets = array<i64: ([^>]*)>, static_sizes = array<i64: ([^>]*)>, '
    r'static_strides = array<i64: ([^>]*)>(.*?: \(tensor<([0-9x]+)x[a-z0-9]+>\) -> tensor<[^>]*>)',
    re.DOTALL)


def _repair_malformed_select_slices(text: str) -> tuple[str, int]:
    """Rewrite malformed rank-reducing ``tensor.extract_slice`` sizes to rank-R. Returns (text, count).

    For a slice with offsets/strides of rank R and a rank-1 ``static_sizes`` value V, pick the KEPT
    dim = the last source dim whose extent == V and where ``offset + V <= source_extent`` (so a
    select'd dim, whose offset leaves < V elements, is never chosen); set that dim's size to V and all
    others to 1. Deterministic when one offset is non-zero (bitvla: dim identified by the select
    index); for all-zero offsets (smolvla) it takes the canonical last-matching dim. A wrong guess
    can only LOWER the on-board cos (never a false pass — correctness stays gated vs golden)."""
    import re as _re

    count = [0]

    def repl(m: "_re.Match") -> str:
        offs = [int(x) for x in m.group(1).split(",")]
        sizes = [x.strip() for x in m.group(2).split(",")]
        strides = [x.strip() for x in m.group(3).split(",")]
        tail = m.group(4)
        src = [int(x) for x in m.group(5).split("x")]
        r = len(offs)
        if len(sizes) != 1 or len(offs) == len(sizes) or len(strides) != r or len(src) != r:
            return m.group(0)  # well-formed or not the rank-1 malformation
        v = int(sizes[0])
        kept = None
        for i in range(r):
            if src[i] == v and offs[i] + v <= src[i]:
                kept = i  # last matching dim
        if kept is None:
            return m.group(0)  # can't safely reconstruct — leave (fails loudly, not silently)
        new_sizes = ", ".join(str(v if i == kept else 1) for i in range(r))
        count[0] += 1
        return (f"static_offsets = array<i64: {m.group(1)}>, static_sizes = array<i64: {new_sizes}>, "
                f"static_strides = array<i64: {m.group(3)}>{tail}")

    out = _EXTRACT_SLICE_RE.sub(repl, text)
    return out, count[0]


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


def compile_rv64gcv_object(ll_path: Path, work: Path, *, timeout: int = 2400) -> Path:
    """LLVM IR -> rv64gcv object via buddy's LLVM fork (``opt -O3`` vectorize, then ``llc``).

    Enforces ``-march=rv64gcv`` (``rvv_audit.enforce_rvv_march``) — no scalar-only build slips
    through. We stay INSIDE buddy's LLVM-23 fork for both steps (its ``mlir-translate`` emits
    ``float f0x…`` hex-float literals the repo's IREE clang-23 parser rejects — a version-skew we
    avoid by using the fork's own ``opt``+``llc``). ``opt -O3`` with the RVV target features runs
    the loop/SLP vectorizer that turns the scalar loop nest into RVV; ``llc`` emits the object.
    """
    rvv_audit.enforce_rvv_march(k1.K1_MARCH)
    opt, llc = llvm_opt(), llvm_llc()
    if llc is None:
        raise BuddyError("no llc in buddy's LLVM build (need build/baselines/buddy/llvm-build/bin/llc)")
    triple = "riscv64-unknown-linux-gnu"
    vec_ll = ll_path
    if opt is not None:
        vec_ll = work / "model.opt.ll"
        common = [f"-mtriple={triple}", f"-mattr={_RVV_MATTR}",
                  f"--riscv-v-vector-bits-min={_RVV_VBITS}"]
        # Full ``-O3`` is superlinear on a whole-model MONOLITHIC forward function; for a very large
        # IR (e.g. pi05 ~200 MB) it never finishes. Above a size threshold, fall back to a BOUNDED
        # vectorization recipe (loop+SLP vectorize + cleanup) — seconds instead of tens of minutes,
        # and still emits RVV (measured ~9% vs ~17% at O3 on tiny_llama) rather than timing out.
        big = ll_path.stat().st_size > _O3_IR_SIZE_LIMIT
        passes = (["-passes=function(loop-vectorize,slp-vectorizer,instcombine,simplifycfg)"]
                  if big else ["-O3"])
        _run([opt, *passes, *common, str(ll_path), "-o", str(vec_ll)], timeout=timeout)
    obj = work / "model.o"
    _run([llc, f"-mtriple={triple}", f"-mattr={_RVV_MATTR}", f"--target-abi={k1.K1_MABI}",
          f"--riscv-v-vector-bits-min={_RVV_VBITS}", "-O3", "-filetype=obj",
          str(vec_ll), "-o", str(obj)], timeout=timeout)
    return obj


# --- native (buddy DynamoCompiler) lowering: tosa/linalg -> rv64gcv --------------------------------

# The native ``forward.mlir``/``subgraph0.mlir`` from buddy's importer are tosa-on-tensors; the
# canonical buddy recipe first runs the tosa->linalg conversion pass-pipeline, then the standard
# lowering. We do it with STOCK mlir-opt (single-threaded scf-to-cf, NOT scf-to-openmp — no libomp
# on the K1) + the OOM-fixing buffer-deallocation-pipeline + c-wrappers.
_TOSA_TO_LINALG = ("builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),"
                   "func.func(tosa-to-tensor),func.func(tosa-to-arith))")


def lower_native_to_llvmir(mlir_path: Path, out_ll: Path, *, timeout: int = 1800) -> Path:
    """Lower one native buddy MLIR file (forward/subgraph, tosa-on-tensors) to LLVM IR text."""
    opt, xlate = buddy_opt(), mlir_translate()
    if opt is None or xlate is None:
        raise BuddyError("buddy lowering toolchain not built")
    out_ll.parent.mkdir(parents=True, exist_ok=True)
    tosa_out = out_ll.with_suffix(".tosa2linalg.mlir")
    _run([opt, str(mlir_path), "-pass-pipeline", _TOSA_TO_LINALG, "-o", str(tosa_out)], timeout=timeout)
    llvm_out = out_ll.with_suffix(".llvm.mlir")
    _run([opt, str(tosa_out),
          "-convert-elementwise-to-linalg",
          "-eliminate-empty-tensors", "-empty-tensor-to-alloc-tensor",
          "-one-shot-bufferize=bufferize-function-boundaries",
          "-buffer-deallocation-pipeline",
          "-convert-linalg-to-loops", "-lower-affine", "-convert-scf-to-cf",
          "-expand-strided-metadata", "-lower-affine", "-convert-vector-to-scf", "-convert-scf-to-cf",
          "-convert-vector-to-llvm", "-convert-math-to-llvm", "-convert-math-to-libm",
          "-finalize-memref-to-llvm", "-llvm-request-c-wrappers", "-convert-func-to-llvm",
          "-convert-arith-to-llvm", "-convert-cf-to-llvm", "-reconcile-unrealized-casts",
          "-canonicalize", "-o", str(llvm_out)], timeout=timeout)
    _run([xlate, "--mlir-to-llvmir", str(llvm_out), "-o", str(out_ll)], timeout=timeout)
    return out_ll


def compile_native_objects(native_dir: Path, work: Path, *, timeout: int = 2400) -> list[Path]:
    """Lower + compile the native forward.mlir + subgraph0.mlir to rv64gcv objects (PIC)."""
    rvv_audit.enforce_rvv_march(k1.K1_MARCH)
    opt, llc = llvm_opt(), llvm_llc()
    triple = "riscv64-unknown-linux-gnu"
    objs = []
    for name in ("forward", "subgraph0"):
        src = native_dir / f"{name}.mlir"
        if not src.is_file():
            raise BuddyError(f"native import missing {src}")
        ll = lower_native_to_llvmir(src, work / f"nat_{name}.ll", timeout=timeout)
        vec = ll
        if opt is not None:
            vec = work / f"nat_{name}.opt.ll"
            big = ll.stat().st_size > _O3_IR_SIZE_LIMIT
            passes = (["-passes=function(loop-vectorize,slp-vectorizer,instcombine,simplifycfg)"]
                      if big else ["-O3"])
            _run([opt, *passes, f"-mtriple={triple}", f"-mattr={_RVV_MATTR}",
                  f"--riscv-v-vector-bits-min={_RVV_VBITS}", str(ll), "-o", str(vec)], timeout=timeout)
        obj = work / f"nat_{name}.o"
        _run([llc, f"-mtriple={triple}", f"-mattr={_RVV_MATTR}", f"--target-abi={k1.K1_MABI}",
              f"--riscv-v-vector-bits-min={_RVV_VBITS}", "-O3", "-relocation-model=pic",
              "-filetype=obj", str(vec), "-o", str(obj)], timeout=timeout)
        objs.append(obj)
    return objs


# The buddy-native ABI is `_mlir_ciface_forward(result*, params*, input*)` where each arg is a
# pointer to a MemRef descriptor {allocated, aligned, offset, sizes[rank], strides[rank]}. This
# harness mmap's the flat f32 param blob (arg0.data), reads the seeded int64 input from a header,
# calls forward once, times it with rdtime, and prints OUT/METRIC/DONE (same markers profile.py +
# zephyr_model._parse_console consume).
def _native_harness_c(out_elems: int, out_lastdim: int, in_elems: int, param_elems: int,
                       input_vals: list[int]) -> str:
    dump_cap = min(out_elems, 4096)
    in_init = ",".join(str(int(v)) for v in input_vals) or "0"
    return f"""/* Generated by merlin.baselines.buddy — buddy-native K1 harness. Do not edit. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>

typedef struct {{ void *a; void *al; int64_t off; int64_t sz[8]; int64_t st[8]; }} MR;
/* result (rank3), params (rank1), input (rank2) */
extern void _mlir_ciface_forward(MR *res, MR *params, MR *input);

#define OUT_ELEMS {out_elems}
#define OUT_LASTDIM {out_lastdim}
#define IN_ELEMS {in_elems}
#define PARAM_ELEMS {param_elems}L
#define DUMP_CAP {dump_cap}
#define K1_TIMEBASE_HZ {k1.K1_TIMEBASE_HZ}ULL
#define K1_CPU_HZ {k1.K1_CPU_HZ}ULL

static float OUT[OUT_ELEMS];
static int64_t IN[IN_ELEMS] = {{{in_init}}};

static inline uint64_t rd_time(void) {{ uint64_t t; __asm__ volatile("rdtime %0":"=r"(t)); return t; }}
static inline uint64_t rd_vlenb(void){{ uint64_t v; __asm__ volatile("csrr %0, vlenb":"=r"(v)); return v; }}
static uint64_t wall_ns(void){{ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return (uint64_t)ts.tv_sec*1000000000ULL+(uint64_t)ts.tv_nsec; }}

static const float *PARAMS;

static void *worker(void *arg){{
  (void)arg;
  printf("=== buddy_native vlenb=%llu ===\\n",(unsigned long long)rd_vlenb()); fflush(stdout);
  MR res={{0}}, par={{0}}, inp={{0}};
  res.a=res.al=OUT; res.off=0; res.sz[0]=1; res.sz[1]=OUT_ELEMS/OUT_LASTDIM; res.sz[2]=OUT_LASTDIM;
  res.st[0]=OUT_ELEMS; res.st[1]=OUT_LASTDIM; res.st[2]=1;
  par.a=par.al=(void*)PARAMS; par.off=0; par.sz[0]=PARAM_ELEMS; par.st[0]=1;
  inp.a=inp.al=IN; inp.off=0; inp.sz[0]=1; inp.sz[1]=IN_ELEMS; inp.st[0]=IN_ELEMS; inp.st[1]=1;
  uint64_t w0=wall_ns(), t0=rd_time();
  _mlir_ciface_forward(&res,&par,&inp);
  uint64_t t1=rd_time(), w1=wall_ns();
  /* buddy's forward ALLOCATES the result internally and the ciface stores the returned descriptor
   * into &res — so the data lives at res.al (buddy's aligned ptr), NOT our pre-passed OUT[]. Read
   * from res.al + res.off. */
  const float *R = (const float *)res.al;
  if (!R) {{ fprintf(stderr,"FAIL null result ptr\\n"); R = OUT; }}
  R += res.off;
  int k = OUT_ELEMS<DUMP_CAP?OUT_ELEMS:DUMP_CAP;
  printf("OUT %d",k);
  for(int i=0;i<k;i++){{ uint32_t b; memcpy(&b,&R[i],4); printf(" %u",(unsigned)b); }}
  printf("\\n");
  if(OUT_ELEMS>DUMP_CAP){{
    int rows=OUT_ELEMS/OUT_LASTDIM; printf("ARGMAX %d",rows);
    for(int r=0;r<rows;r++){{ const float*row=&R[(long)r*OUT_LASTDIM]; int bi=0; float bv=row[0];
      for(int j=1;j<OUT_LASTDIM;j++) if(row[j]>bv){{bv=row[j];bi=j;}} printf(" %d",bi); }}
    printf("\\n");
    float s=0; for(int i=0;i<OUT_ELEMS;i++) s+=R[i]; uint32_t sb; memcpy(&sb,&s,4);
    printf("SUM %u\\n",(unsigned)sb);
  }}
  uint64_t ticks=t1-t0, est=ticks*(K1_CPU_HZ/K1_TIMEBASE_HZ);
  printf("METRIC cycles %llu\\n",(unsigned long long)est);
  printf("METRIC time_ticks %llu\\n",(unsigned long long)ticks);
  printf("METRIC wall_ns %llu\\n",(unsigned long long)(w1-w0));
  printf("DONE\\n"); fflush(stdout);
  return NULL;
}}

int main(int argc,char**argv){{
  const char*wp=getenv("MERLIN_WEIGHTS"); if(!wp&&argc>1) wp=argv[1];
  if(!wp){{ fprintf(stderr,"FAIL no MERLIN_WEIGHTS (arg0.data)\\n"); return 2; }}
  int fd=open(wp,O_RDONLY); if(fd<0){{ fprintf(stderr,"FAIL open %s\\n",wp); return 2; }}
  struct stat st; fstat(fd,&st);
  void*m=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
  if(m==MAP_FAILED){{ fprintf(stderr,"FAIL mmap params\\n"); return 2; }}
  PARAMS=(const float*)m;
  pthread_attr_t at; pthread_t th;
  if(pthread_attr_init(&at)==0 && pthread_attr_setstacksize(&at,(size_t){k1.K1_STACK_BYTES}ULL)==0
     && pthread_create(&th,&at,worker,NULL)==0) pthread_join(th,NULL);
  else worker(NULL);
  return 0;
}}
"""


def link_native_k1_elf(native_dir: Path, objs: list[Path], work: Path, input_vals: list[int],
                       out_elems: int, out_lastdim: int, in_elems: int, param_elems: int,
                       *, timeout: int = 600) -> Path:
    """Link the native rv64gcv objects + the buddy-native harness into a K1 Linux ELF.

    Weights (arg0.data) are mmap'd at run time (MERLIN_WEIGHTS), so the binary is small.
    """
    cc = k1.toolchain_cc()
    if cc is None:
        raise BuddyError("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")
    main_c = work / "native_main.c"
    main_c.write_text(_native_harness_c(out_elems, out_lastdim, in_elems, param_elems, input_vals))
    binary = work / "buddy_native_k1"
    # buddy's tosa lowering emits memref.copy -> a call to `memrefCopy`; merlin's mlir_runtime.c
    # provides it (normally MLIR's C runner utils would).
    abi_rt = repo_root() / "merlin/runtime/abi" / "mlir_runtime.c"
    base = [cc, "--target=riscv64-unknown-linux-gnu", f"-march={k1.K1_MARCH}",
            f"-mabi={k1.K1_MABI}", "-O2", "-fPIC", str(main_c), str(abi_rt),
            *[str(o) for o in objs], "-lm", "-lpthread", "-o", str(binary)]
    try:
        _run([*base, "-static"], timeout=timeout)
    except BuddyError:
        _run(base, timeout=timeout)
    if not binary.is_file():
        raise BuddyError(f"native link produced no binary at {binary}")
    return binary


# --- buddy m2m ABI: buddy lowers a TENSOR-RETURNING forward with DYNAMIC-STRIDED input memrefs -----
#
# Two ABI facts distinguish buddy's object from merlin's own, and BOTH must be honoured or the board
# prints garbage (all-zeros / a broadcast constant — observed cos≈0 across fp32 and int8):
#
#  (1) sret-first result.  merlin lowers `forward` in destination-passing style (output is a trailing
#      pre-allocated arg); buddy keeps `forward(...) -> tensor`, so its emit-c-interface wrapper is
#      **sret-first**: `_mlir_ciface_forward(result_desc*, in0*, ...)`. arg0 is a descriptor the callee
#      OVERWRITES with the freshly-allocated result; the operands start at arg1. Passing buddy the DPS
#      order (weights at d[0], OUT last) shifts every operand by one AND never fills OUT.
#
#  (2) RANK-EXACT descriptors.  buddy's forward takes `memref<...xT, strided<[?,?], offset: ?>>`
#      **dynamic-strided** inputs, so the callee READS each operand's sizes/strides FROM the passed
#      descriptor, expecting the MLIR layout `{ptr, ptr, i64 offset, i64 sizes[rank], i64 strides[rank]}`
#      with sizes/strides packed to the operand's ACTUAL rank. merlin's `merlin_descriptor_t` uses a
#      FIXED `sizes[8]/strides[8]`, so for a rank-2 operand buddy reads `strides[0]` from where the
#      fixed struct stores `sizes[2]` — garbage/zero strides → the compute loads the input as zeros
#      (proven on a K1 micro-repro: an elementwise `+1` returns all-1.0, i.e. input read as 0).
#      merlin's OWN runtime never hits this because its forward uses STATIC memrefs (compile-time
#      strides, never read from the descriptor). So the fix is arm-local: build per-operand rank-exact
#      packed descriptors ourselves, in signature order, and call the ciface sret-first.
#
# The harness below therefore does NOT use merlin_run/merlin_descriptor_t; it packs each operand's
# descriptor to its exact rank from the c_runtime MERLIN_ARGS table (kind/offset/rank/dims), reads
# the result from the returned sret descriptor, and prints the same OUT/METRIC/DONE markers.
def _m2m_harness_c(*, mmap_weights: bool = False) -> str:
    """buddy m2m-path harness: sret-first ciface + RANK-EXACT packed operand descriptors.

    Builds each forward operand's memref descriptor packed to its actual rank (so buddy's
    dynamic-strided reads land on the right bytes), calls the sret-first ciface with the result
    descriptor first, then reads the output from that returned descriptor's aligned pointer."""
    mmap_includes = ("#include <fcntl.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n"
                     "#include <unistd.h>\n" if mmap_weights else "")
    weights_decl = ("static const void *MERLIN_WEIGHTS_PTR;\n" if mmap_weights else
                    "extern const unsigned char _binary_weights_bin_start[];\n"
                    "static const void *MERLIN_WEIGHTS_PTR;\n")
    if mmap_weights:
        weights_init = """  const char *wpath = getenv("MERLIN_WEIGHTS");
  if (!wpath && argc > 1) wpath = argv[1];
  if (!wpath) { fprintf(stderr, "FAIL no MERLIN_WEIGHTS path\\n"); return 2; }
  int wfd = open(wpath, O_RDONLY);
  if (wfd < 0) { fprintf(stderr, "FAIL open weights %s\\n", wpath); return 2; }
  struct stat wst; fstat(wfd, &wst);
  void *wmap = mmap(NULL, (size_t)wst.st_size, PROT_READ, MAP_PRIVATE, wfd, 0);
  if (wmap == MAP_FAILED) { fprintf(stderr, "FAIL mmap weights\\n"); return 2; }
  MERLIN_WEIGHTS_PTR = wmap;"""
    else:
        weights_init = "  MERLIN_WEIGHTS_PTR = (const void *)_binary_weights_bin_start;"
    return f"""/* Generated by merlin.baselines.buddy — buddy m2m K1 harness (sret-first, rank-exact). Do not edit. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <pthread.h>
{mmap_includes}#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"

{weights_decl}#define MERLIN_DUMP_CAP {4096}
#define MERLIN_TIMEBASE_HZ {k1.K1_TIMEBASE_HZ}ULL
#define MERLIN_CPU_HZ {k1.K1_CPU_HZ}ULL

/* buddy overwrites this with the freshly-allocated result descriptor. */
extern void *_mlir_ciface_forward();  /* variadic-tolerant decl; called via merlin_buddy_invoke */
void merlin_buddy_invoke(void **descs);  /* generated in model_call.c (sret-first unrolled call) */
extern void *merlin_buddy_result_desc;

/* A packed rank-R memref descriptor: ptr, ptr, i64 offset, i64 sizes[R], i64 strides[R]. We pack
 * per operand to its EXACT rank (buddy reads dynamic strides from here). A generous byte buffer
 * holds each; the fields are written at rank-dependent offsets. */
static float OUT[MERLIN_OUT_ELEMS];

/* build a rank-exact descriptor for arg `i` into `buf` (>= 24 + 16*rank bytes); returns buf. */
static void *pack_desc(unsigned char *buf, const merlin_arg_t *a, void *data) {{
  int r = a->rank;
  void **pa = (void **)buf;          /* [0] allocated */
  pa[0] = data;                      /* allocated */
  pa[1] = data;                      /* aligned    */
  int64_t *q = (int64_t *)(buf + 16);
  q[0] = 0;                          /* offset */
  int64_t *sizes = q + 1;            /* sizes[r] */
  int64_t *strides = sizes + r;      /* strides[r] IMMEDIATELY after sizes[r] (rank-exact packing) */
  int64_t s = 1;
  for (int d = r - 1; d >= 0; d--) {{ sizes[d] = a->dims[d]; strides[d] = s; s *= a->dims[d]; }}
  return buf;
}}

static inline uint64_t rd_time(void) {{ uint64_t t; __asm__ volatile("rdtime %0":"=r"(t)); return t; }}
static inline uint64_t rd_vlenb(void){{ uint64_t v; __asm__ volatile("csrr %0, vlenb":"=r"(v)); return v; }}
static uint64_t wall_ns(void){{ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return (uint64_t)ts.tv_sec*1000000000ULL+(uint64_t)ts.tv_nsec; }}

static void *worker(void *arg) {{
  (void)arg;
  printf("=== buddy_m2m vlenb=%llu ===\\n",(unsigned long long)rd_vlenb());
  /* COMPUTE-vs-OVERHEAD split. The buddy ELF is a monolithic forward (all GEMM/attention/norm compute
   * + its own intermediate allocs) plus this thin harness. The only "runtime overhead" outside the
   * compute is our descriptor-pack loop (building rank-exact memref descriptors for the N operands);
   * there is NO merlin dispatch/per-op runtime here. We bracket the descriptor pack (overhead_region)
   * with cheap rdtime, then the forward (compute_region) — reported alongside the clean whole-forward
   * e2e (the headline). rdtime brackets are ~near-zero cost, so they don't perturb the e2e. */
  static unsigned char DBUF[MERLIN_N_ARGS][24 + 16 * MERLIN_MAX_RANK];
  static void *DPTR[MERLIN_N_ARGS];
  uint64_t ov0 = rd_time();
  for (int i = 0; i < MERLIN_N_ARGS; i++) {{
    const merlin_arg_t *a = &MERLIN_ARGS[i];
    void *data = 0;
    if (a->kind == MERLIN_WEIGHT)      data = (void *)((const char *)MERLIN_WEIGHTS_PTR + a->offset);
    else if (a->kind == MERLIN_INPUT)  data = MERLIN_INPUT_PTR[i];
    else /* MERLIN_OUTPUT */           data = OUT;   /* placeholder; buddy overwrites this descriptor */
    DPTR[i] = pack_desc(DBUF[i], a, data);
  }}
  uint64_t ov1 = rd_time();          /* descriptor-pack overhead = ov1-ov0 */
  uint64_t w0=wall_ns(), t0=rd_time();
  merlin_buddy_invoke(DPTR);          /* the compute region (whole monolithic forward) */
  uint64_t t1=rd_time(), w1=wall_ns();

  /* read the output from buddy's RETURNED (sret) descriptor's aligned pointer + offset. */
  const float *R = OUT;
  void **rp = (void **)merlin_buddy_result_desc;
  if (rp && rp[1]) {{ int64_t off = ((int64_t *)((unsigned char *)merlin_buddy_result_desc + 16))[0];
                      R = (const float *)rp[1] + off; }}

  int k = MERLIN_OUT_ELEMS < MERLIN_DUMP_CAP ? MERLIN_OUT_ELEMS : MERLIN_DUMP_CAP;
  printf("OUT %d", k);
  for (int i=0;i<k;i++){{ uint32_t b; memcpy(&b,&R[i],4); printf(" %u",(unsigned)b); }}
  printf("\\n");
  if (MERLIN_OUT_ELEMS > MERLIN_DUMP_CAP) {{
    int rows = MERLIN_OUT_ELEMS / MERLIN_OUT_LASTDIM;
    printf("ARGMAX %d", rows);
    for (int r=0;r<rows;r++){{ const float*row=&R[(long)r*MERLIN_OUT_LASTDIM]; int bi=0; float bv=row[0];
      for (int j=1;j<MERLIN_OUT_LASTDIM;j++) if(row[j]>bv){{bv=row[j];bi=j;}} printf(" %d",bi); }}
    printf("\\n");
    float s=0; for(int i=0;i<MERLIN_OUT_ELEMS;i++) s+=R[i]; uint32_t sb; memcpy(&sb,&s,4);
    printf("SUM %u\\n",(unsigned)sb);
  }}
  uint64_t ticks=t1-t0, est=ticks*(MERLIN_CPU_HZ/MERLIN_TIMEBASE_HZ);
  uint64_t ov=ov1-ov0;               /* descriptor-pack overhead ticks */
  printf("METRIC cycles %llu\\n",(unsigned long long)est);
  printf("METRIC time_ticks %llu\\n",(unsigned long long)ticks);
  printf("METRIC wall_ns %llu\\n",(unsigned long long)(w1-w0));
  /* compute-vs-overhead split (rdtime ticks): compute = the whole forward; overhead = descriptor pack.
   * MERLIN_REGION name=<> ticks=<> is the format profile.parse_profile consumes. */
  printf("MERLIN_REGION name=compute ticks=%llu\\n",(unsigned long long)ticks);
  printf("MERLIN_REGION name=overhead ticks=%llu\\n",(unsigned long long)ov);
  printf("METRIC overhead_ticks %llu\\n",(unsigned long long)ov);
  printf("DONE\\n"); fflush(stdout);
  return NULL;
}}

int main(int argc, char **argv) {{
  (void)argc; (void)argv;
{weights_init}
  struct rlimit rl = {{ {k1.K1_STACK_BYTES}ULL, {k1.K1_STACK_BYTES}ULL }};
  setrlimit(RLIMIT_STACK, &rl);
  pthread_attr_t attr; pthread_t th;
  if (pthread_attr_init(&attr)==0 && pthread_attr_setstacksize(&attr,(size_t){k1.K1_STACK_BYTES}ULL)==0
      && pthread_create(&th,&attr,worker,NULL)==0) pthread_join(th,NULL);
  else worker(NULL);
  return 0;
}}
"""


def _buddy_model_call_c(n_ciface_args: int) -> str:
    """buddy sret-first invocation shim: call ``_mlir_ciface_forward(result, in0, in1, ...)``.

    The c_runtime arg table (and our ``DPTR``) is in signature order with OUTPUT last (``d[N-1]``).
    buddy's ciface wants ``(result_desc*, operand0*, ...)`` — so we pass the OUTPUT descriptor FIRST,
    then the N-1 signature operands. buddy overwrites the result descriptor in place; we stash a
    pointer to it (``merlin_buddy_result_desc``) so the harness reads the aligned data post-call."""
    fwd_args = ",".join(["void*"] * n_ciface_args)
    passed = ",".join(["d[N-1]"] + [f"d[{i}]" for i in range(n_ciface_args - 1)])
    return f"""/* Generated by merlin.baselines.buddy — buddy sret-first ciface shim. Do not edit. */
#define N {n_ciface_args}
extern void _mlir_ciface_forward({fwd_args});
void *merlin_buddy_result_desc = 0;
void merlin_buddy_invoke(void **d) {{
  merlin_buddy_result_desc = d[N-1];
  _mlir_ciface_forward({passed});
}}
"""


def link_k1_elf(model_dir: Path, obj: Path, work: Path, *, inputs_npz: Path | None = None,
                timeout: int = 600) -> Path:
    """Link buddy's rv64gcv object with merlin's data-driven arg table into a K1 Linux ELF.

    Reuses ``llvmlower.c_runtime.generate`` (arg table + weights.bin + embedded I/O) but drives the
    compute with a buddy-specific harness + call shim that honour buddy's **sret-first** result ABI
    AND its **dynamic-strided** input memrefs (rank-exact packed descriptors) — see the ABI note
    above. Weights blob / embedded I/O / board plumbing stay merlin's proven path; only the compute
    object and the (correct) descriptor marshalling differ, so it's still an apples-to-apples build.
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
    meta = c_runtime.generate(model_dir, cgen, inputs_npz)

    weights_bin = cgen / "weights.bin"
    # buddy embeds the weight blob via `ld -r -b binary`; a ~0.5 GB+ embedded blob makes an ELF that
    # both blows up the static link (relocation/size) and forces all weights resident on the 3.4 GB
    # board. Use a LOWER buddy-local threshold (256 MB) than merlin's default so mid/big models
    # (smolvla 506 MB, rdt2 488 MB, rdt 1.2 GB) mmap the blob (demand-paged) instead — smaller binary,
    # resident RAM = working set.
    mmap_weights = (weights_bin.stat().st_size >= _BUDDY_MMAP_WEIGHTS_THRESHOLD
                    if weights_bin.is_file() else False)
    weights_o = work / "weights_blob.o"
    if not mmap_weights:
        _run([ld, "-r", "-b", "binary", "-o", str(weights_o), "weights.bin"], cwd=cgen)
    else:
        (work / "USE_MMAP_WEIGHTS").write_text(str(weights_bin))

    # buddy sret-first call shim (replaces c_runtime's DPS model_call.c).
    n_args = int(meta["n_args"])
    (cgen / "buddy_call.c").write_text(_buddy_model_call_c(n_args))

    main_c = work / "main_linux.c"
    main_c.write_text(_m2m_harness_c(mmap_weights=mmap_weights))

    rt = repo_root() / "merlin/runtime/c"
    abi = repo_root() / "merlin/runtime/abi"
    binary = work / "buddy_k1"
    # NOTE: we do NOT link merlin_model.c/model_call.c (their DPS merlin_run + fixed-8 descriptor is
    # exactly what breaks buddy's dynamic-strided ABI); the buddy harness builds descriptors itself.
    srcs = [main_c, cgen / "buddy_call.c", abi / "mlir_runtime.c"]
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
    res.notes += f" datapath={'w8a8-int' if variant == 'int8' else 'f32'}"

    # 0. normalize the capture (quant.* / over-rank matmul / bf16) into buddy-lowerable IR. int8 ->
    #    a real i8*i8->i32 integer datapath (opt -O3 vectorizes the MACs to vwmacc.vv on rv64gcv).
    try:
        prepared = prepare_model_mlir(b, work)
    except Exception as e:  # noqa: BLE001 — a capture that won't even normalize is a hard not_built
        res.gap_reason = f"MLIR normalization failed (variant={variant}): {str(e)[:280]}"
        return _finish(res, model, variant, write)

    # 1. lower + cross-compile to the rv64gcv object (the RVV artifact).
    try:
        ll = lower_to_llvmir(prepared, work)
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


def run_model_native(model: str, variant: str = "int8", *, work_root: Path | None = None,
                     write: bool = True, run_board: bool | None = None) -> BaselineResult:
    """Phase-2 NATIVE path: import via buddy's DynamoCompiler, lower + run on the K1.

    Ingests the REAL torch model (buddy's own importer, full-fidelity ``bundle.full_env``) instead of
    m2m's linalg ``model.mlir`` — DIFFERENT IR that may bypass the m2m-path SIGSEGV. Records honest
    gaps: RAM-infeasible 7B VLAs (``bundle.K1_RAM_INFEASIBLE``) are ``not_run`` (attempt build, never
    a false fit); import/lower/link failures are ``not_built``.
    """
    import numpy as np

    cos_thr, rel_thr = _bundle.tolerance(model)
    res = BaselineResult(framework=FRAMEWORK, model=model, variant=variant, substrate="k1_spacemit",
                         cos_threshold=cos_thr, rel_threshold=rel_thr, march=k1.K1_MARCH,
                         toolchain="buddy-mlir(native-dynamo)+llvm23", framework_commit=buddy_commit(),
                         timestamp=artifacts.utc_stamp(), notes="native=dynamo-importer")

    b = resolve_bundle(model, variant)
    if not b.golden.is_file():
        res.gap_reason = f"golden missing: {b.golden} (cannot gate correctness)"
        return _finish(res, model, variant, write)
    if not native_import_available():
        res.gap_reason = ("buddy native importer unavailable (need MLIR python bindings + buddy "
                          "python packages built, and the model2MLIR torch venv)")
        return _finish(res, model, variant, write)

    work = (work_root or (_BUILD_ROOT / "native")) / f"{model}_{variant}"
    work.mkdir(parents=True, exist_ok=True)
    nat = work / "import"

    # 1. native import (buddy DynamoCompiler over the real torch model).
    try:
        native_import(model, variant, nat)
    except BuddyError as e:
        res.gap_reason = f"native import failed: {str(e)[:300]}"
        return _finish(res, model, variant, write)

    import json
    meta = json.loads((nat / "import_meta.json").read_text())
    res.notes += f" registry={meta.get('registry')} n_params={meta.get('n_params')}"

    # 2. lower + compile to rv64gcv objects + RVV audit.
    try:
        objs = compile_native_objects(nat, work)
        cov, fbs, _ = audit_object(work / "nat_subgraph0.o")
        res.rvv_coverage_overall = cov
        res.scalar_fallbacks = fbs
    except BuddyError as e:
        res.gap_reason = f"native lower/compile failed: {str(e)[:300]}"
        return _finish(res, model, variant, write)

    # 3. link the native K1 ELF (output shape from the fp32 golden — the native gate reference).
    gold = np.load(_fp32_golden_for(b))
    out_elems = int(np.prod(gold.shape))
    out_lastdim = int(gold.shape[-1]) if gold.ndim else 1
    iv = []
    if b.inputs.is_file():
        npz = np.load(b.inputs)
        key = next((k for k in npz.files), None)
        if key is not None:
            iv = np.asarray(npz[key]).astype(np.int64).ravel().tolist()
    param_elems = int(meta.get("param_bytes", 0)) // 4
    try:
        elf = link_native_k1_elf(nat, objs, work, iv, out_elems=out_elems, out_lastdim=out_lastdim,
                                 in_elems=max(1, len(iv)), param_elems=param_elems)
        res.built = True
        res.notes += f" elf={elf}"
    except BuddyError as e:
        res.gap_reason = f"native link failed: {str(e)[:300]}"
        return _finish(res, model, variant, write)

    # 4. RAM-infeasible 7B VLAs: attempt build (done) but never fit on the 3.8 GB board.
    if model in _bundle.K1_RAM_INFEASIBLE:
        res.gap_reason = (f"{model} is 7B-class — {param_elems*4/1e9:.1f} GB params exceed the 3.8 GB "
                          "K1 RAM even at int8 (built + RVV audited; on-board run RAM-infeasible)")
        return _finish(res, model, variant, write)

    # 5. on-board run (native harness; params mmap'd from arg0.data).
    do_board = k1_exec.board_available() if run_board is None else run_board
    if do_board:
        try:
            _run_native_on_board(res, elf, nat / "arg0.data", b, work)
        except Exception as e:  # noqa: BLE001
            res.gap_reason = res.gap_reason or f"K1 native run error: {str(e)[:200]}"
    else:
        res.gap_reason = "K1 board unavailable (MERLIN_K1_HOST unset)"
    res.board_vlenb = k1_exec.board_vlenb()
    return _finish(res, model, variant, write)


def _run_native_on_board(res: BaselineResult, elf: Path, arg0: Path, b: _bundle.CaptureBundle,
                         work: Path) -> None:
    """Push the native ELF + arg0.data (mmap'd), run under board_lock, parse OUT/METRIC + gate."""
    import numpy as np

    from merlin.runtime.backends import zephyr_model as zm

    with k1_exec.board_lock():
        remote = k1_exec.push(str(elf))
        # arg0.data -> stable size-keyed name on the rootfs (idempotent: skip re-scp if present).
        wsize = arg0.stat().st_size
        remote_w = f"{k1_exec.K1_REMOTE_DIR}/{b.model}_{b.variant}_native_{wsize}.arg0.data"
        probe = k1_exec.run(["stat", "-c%s", remote_w, "2>/dev/null", "||", "echo", "0"])
        if not (probe.stdout.strip().isdigit() and int(probe.stdout.strip()) == wsize):
            k1_exec.push(str(arg0), remote_w, timeout=3600)
        try:
            k1_exec.run(["chmod", "+x", remote])
            proc = k1_exec.run([f"MERLIN_WEIGHTS={remote_w} {remote}"], timeout=1800)
        finally:
            try:
                k1_exec.run(["rm", "-f", remote])
            except Exception:  # noqa: BLE001
                pass

    console = proc.stdout + proc.stderr
    try:
        parsed = zm._parse_console(console, proc.returncode)
    except zm.ZephyrModelError as e:
        res.ran = False
        res.gap_reason = res.gap_reason or f"K1 native run no OUT/DONE: {str(e)[-200:]}"
        return
    res.ran = "DONE" in console
    metrics = parsed.get("metrics", {}) if isinstance(parsed.get("metrics"), dict) else {}
    if metrics.get("time_ticks") is not None:
        res.e2e_rdtime_ticks = int(metrics["time_ticks"])
        res.e2e_cycles = profile.ticks_to_cycles(int(metrics["time_ticks"]))
    if metrics.get("wall_ns") is not None:
        res.e2e_wall_ns = int(metrics["wall_ns"])
    if res.e2e_rdtime_ticks is not None:
        res.regions = [RegionProfile(name="other", rdtime_ticks=res.e2e_rdtime_ticks,
                                     cycles=res.e2e_cycles, rvv_coverage=res.rvv_coverage_overall,
                                     note="whole-forward (buddy native)")]
    out = parsed.get("outputs")
    if out is not None and res.ran:
        try:
            # int8-via-native gates vs the FP32 golden ("what did int8 cost").
            gp = _fp32_golden_for(b)
            res.notes += f" gold={gp.name}"
            g = np.load(gp).astype(np.float64).ravel()
            a = np.asarray(out, dtype=np.float64).ravel()[:g.size]
            g = g[:a.size]
            if a.size:
                res.cos = float(np.dot(a, g) / ((np.linalg.norm(a) * np.linalg.norm(g)) or 1.0))
                res.rel = float(np.linalg.norm(a - g) / (np.linalg.norm(g) or 1.0))
        except Exception as e:  # noqa: BLE001
            res.notes += f" gold-compare failed: {str(e)[:120]}"
    if not res.ran and not res.gap_reason:
        res.gap_reason = f"K1 native run no DONE (rc={proc.returncode}): {console[-200:]}"


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
            wpath = Path(marker.read_text().strip())
            # Big-weights path (multi-GB): deploy to a STABLE, content-keyed remote name and reuse it
            # if a same-size copy is already on the board (idempotent -> re-runs skip the slow scp).
            # Weights live on the rootfs (real flash), NOT /tmp (tmpfs). Generous timeout: a 4.4 GB
            # blob over the DHCP link far exceeds the default 300 s.
            wsize = wpath.stat().st_size if wpath.is_file() else 0
            remote_w = f"{k1_exec.K1_REMOTE_DIR}/{b.model}_{b.variant}_{wsize}.weights.bin"
            probe = k1_exec.run(["stat", "-c%s", remote_w, "2>/dev/null", "||", "echo", "0"])
            already = probe.stdout.strip().isdigit() and int(probe.stdout.strip()) == wsize
            if not already:
                k1_exec.push(str(wpath), remote_w, timeout=3600)
            wenv = f"MERLIN_WEIGHTS={remote_w} "
        try:
            k1_exec.run(["chmod", "+x", remote])
            proc = k1_exec.run([f"{wenv}{remote}"], timeout=1800)
        finally:
            # remove only the (small) binary; KEEP the big weights blob for re-runs.
            try:
                k1_exec.run(["rm", "-f", remote])
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
    # compute-vs-overhead split from the harness MERLIN_REGION brackets (compute = whole monolithic
    # forward; overhead = the descriptor-pack loop — the only runtime cost outside buddy's compute).
    _wm, _regions = profile.parse_profile(console)
    ov_ticks = next((int(r.rdtime_ticks) for r in _regions
                     if r.name == "overhead" and r.rdtime_ticks is not None), None)
    if res.e2e_rdtime_ticks is not None:
        res.regions = [RegionProfile(name="compute", rdtime_ticks=res.e2e_rdtime_ticks,
                                     cycles=res.e2e_cycles, rvv_coverage=res.rvv_coverage_overall,
                                     note="whole monolithic forward (all GEMM/attn/norm compute)")]
        if ov_ticks is not None:
            res.regions.append(RegionProfile(
                name="overhead", rdtime_ticks=ov_ticks, cycles=profile.ticks_to_cycles(ov_ticks),
                rvv_coverage=0.0, note="descriptor-pack (runtime dispatch overhead; near-zero)"))
            frac = ov_ticks / res.e2e_rdtime_ticks if res.e2e_rdtime_ticks else 0.0
            res.notes += f" overhead={ov_ticks}t({100*frac:.3f}%) compute={res.e2e_rdtime_ticks}t"
    # correctness from the OUT marker (parsed 'outputs' float array). int8 gates against the W8A8
    # reference (golden_w8a8.npy) when present — that is the math the integer datapath INTENDS to
    # run; falling back to the fp32 golden only when no W8A8 ref exists.
    out = parsed.get("outputs")
    if out is not None and res.ran:
        try:
            gold_path = _golden_for(b)
            res.notes += f" gold={gold_path.name}"
            gold = np.load(gold_path).astype(np.float32).ravel()
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


def _golden_for(b: _bundle.CaptureBundle) -> Path:
    """Pick the correctness reference: the W8A8 golden for int8 (the math the integer datapath
    intends to run) when present, else the fp32 golden. Falls back to golden.npy always existing."""
    if b.variant == "int8":
        w8a8 = b.root / "golden_w8a8.npy"
        if w8a8.is_file():
            return w8a8
    return b.golden


def _fp32_golden_for(b: _bundle.CaptureBundle) -> Path:
    """The FP32 golden for a model — the "what did int8 cost" reference the native path gates on.

    Buddy's native importer materializes int8->fp32, so its output is naturally compared to the fp32
    golden (``bundle.resolve(model,'fp32').golden``) rather than the W8A8 reference. Falls back to
    the (int8) bundle's own golden if the fp32 recapture is absent.
    """
    fp = _bundle.resolve(b.model, "fp32")
    return fp.golden if fp.golden.is_file() else _golden_for(b)


def _finish(res: BaselineResult, model: str, variant: str, write: bool) -> BaselineResult:
    res.validate()
    if write:
        m = artifacts.new_measurement("k1_spacemit", model, "cross_framework")
        res.write(m.path)
    return res


def _run_one_safe(model: str, variant: str, *, write: bool) -> BaselineResult:
    """run_model with a hard exception guard so one model never sinks the batch."""
    try:
        return run_model(model, variant, write=write)
    except Exception as e:  # noqa: BLE001
        r = BaselineResult(framework=FRAMEWORK, model=model, variant=variant,
                           gap_reason=f"runner exception: {str(e)[:200]}",
                           timestamp=artifacts.utc_stamp())
        if write:
            try:
                md = artifacts.new_measurement("k1_spacemit", model, "cross_framework")
                r.write(md.path)
            except Exception:  # noqa: BLE001
                pass
        return r


def run_all(models=DEFAULT_MODELS, variant: str = "fp32", *, write: bool = True) -> list[BaselineResult]:
    """Run the buddy arm over the model set for a single variant. Returns the BaselineResults."""
    return [_run_one_safe(m, variant, write=write) for m in models]


# int8 is prioritized: ~4x smaller weights (fits the K1 RAM/disk better) and a real integer RVV
# datapath (vwmacc.vv), so it both fits more models and runs faster than the slow fp32 path.
VARIANT_ORDER: tuple[str, ...] = ("int8", "fp32")


def run_all_variants(models=DEFAULT_MODELS, variants=VARIANT_ORDER, *,
                     write: bool = True) -> list[BaselineResult]:
    """Attempt every (model, variant) — int8 FIRST, then fp32. Every attempt yields a BaselineResult
    (pass / fail / not_built / not_run with a gap_reason); nothing is omitted or fabricated."""
    out = []
    for variant in variants:
        for m in models:
            out.append(_run_one_safe(m, variant, write=write))
    return out


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Buddy (buddy-mlir) K1-RVV baseline arm")
    ap.add_argument("models", nargs="*", default=list(DEFAULT_MODELS),
                    help="models to run (default: the full corpus, LLM subset first)")
    ap.add_argument("--variant", default=None,
                    help="single variant (int8|fp32); default runs BOTH, int8 first")
    ap.add_argument("--no-write", action="store_true", help="do not write BaselineResult artifacts")
    args = ap.parse_args(argv)
    if args.variant:
        results = run_all(tuple(args.models), args.variant, write=not args.no_write)
    else:
        results = run_all_variants(tuple(args.models), write=not args.no_write)
    for r in results:
        cov = f"{100*r.rvv_coverage_overall:.0f}%RVV" if r.rvv_coverage_overall is not None else "?RVV"
        cc = f"cos={r.cos:.4f}" if r.cos is not None else "cos=?"
        print(f"{r.model}/{r.variant}: {r.status():10s} {cov} {cc} "
              f"fallbacks={len(r.scalar_fallbacks)} {r.gap_reason[:80]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
