"""Upstream MLIR lowering + LLVM IR translation, run inside the model2MLIR venv.

The torch-mlir wheel carries the full upstream pass registry (LLVM 23) and the
``translate_module_to_llvmir`` bridge; the lowering executes via a subprocess
in that venv.

Pipeline (whole module, in one shot):
linalg-on-tensors → (fuse elementwise, generalize) → one-shot-bufferize
→ buffer-results-to-out-params{modify-public-functions hoist-static-allocs}
→ convert-linalg-to-loops → scf->cf → llvm-dialect → translate -> .ll

NB: ``buffer-results-to-out-params`` MUST run with ``modify-public-functions`` —
the default silently skips public functions and the entry retains a returned
memref descriptor (heap-allocated). Discovered the hard way; covered by tests.

Vectorization strategy: rely on LLVM's RISC-V auto-vectorizer (clang -O2 -march=rv64gcv)
to vectorize scalar loops — verified to emit vsetvli/RVV instructions. A scalable
vector path (linalg tile→vectorize at vscale) can be layered later without changing
this driver's interface.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .toolchain import m2m_python

UPSTREAM_PIPELINE = ",".join([
    "canonicalize", "cse",
    "func.func(linalg-fuse-elementwise-ops)",
    "func.func(linalg-generalize-named-ops)",
    "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
    "buffer-results-to-out-params{modify-public-functions hoist-static-allocs}",
    "func.func(buffer-hoisting,buffer-loop-hoisting)",
    "func.func(convert-linalg-to-loops)",
    "convert-scf-to-cf",
    # memref.expand/collapse_shape/subview must be expanded before LLVM conversion.
    "expand-strided-metadata",
    "lower-affine",
    # transcendentals (erf/exp/tanh/...) have no LLVM intrinsic -> libm calls
    # (host libm; riscv newlib libm). Remaining math (fma/abs/...) -> intrinsics.
    "convert-math-to-libm",
    "convert-math-to-llvm",
    "convert-index-to-llvm",
    "convert-arith-to-llvm",
    "finalize-memref-to-llvm",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    "reconcile-unrealized-casts",
    "canonicalize", "cse", "symbol-dce",
])


def _parallel_pipeline() -> str:
    """The scalar pipeline, re-targeted for **multicore** via OpenMP.

    Identical to :data:`UPSTREAM_PIPELINE` except the loop-generation stage: parallel
    iterator dimensions of each linalg op become ``scf.parallel`` (``convert-linalg-to-
    parallel-loops``) which ``convert-scf-to-openmp`` wraps in an ``omp.parallel`` +
    ``omp.wsloop`` (work-shared across threads); reduction dims stay sequential
    ``scf.for`` (correctness preserved). ``mlir-translate`` then emits ``__kmpc_*`` runtime
    calls, satisfied at link time by the cross-built ``libomp.a`` (build_tools/k1_openmp).
    Set ``OMP_NUM_THREADS`` on the board to fan out across the 8 cores.

    This is the K1 big-model path: the diffusion-transformer / large VLAs that run on the
    *scalar* fallback (no fixed-width vectorize) otherwise execute on one core and time out.
    Gated — never on the default flow (``UPSTREAM_PIPELINE`` is untouched)."""
    return ",".join([
        "canonicalize", "cse",
        "func.func(linalg-fuse-elementwise-ops)",
        "func.func(linalg-generalize-named-ops)",
        "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
        "buffer-results-to-out-params{modify-public-functions}",  # heap intermediates (K1)
        "func.func(buffer-hoisting,buffer-loop-hoisting)",
        # parallel iterator dims -> scf.parallel; reduction dims stay scf.for (sequential).
        "func.func(convert-linalg-to-parallel-loops)",
        "convert-scf-to-openmp",          # scf.parallel -> omp.parallel{omp.wsloop}
        "canonicalize",
        "convert-scf-to-cf",              # remaining (reduction) scf.for -> cf
        "expand-strided-metadata",
        "lower-affine",
        "convert-openmp-to-llvm",         # omp region operands -> llvm
        "convert-math-to-libm",
        "convert-math-to-llvm",
        "convert-index-to-llvm",
        "convert-arith-to-llvm",
        "finalize-memref-to-llvm",
        "convert-func-to-llvm",
        "convert-cf-to-llvm",
        "reconcile-unrealized-casts",
        "canonicalize", "cse", "symbol-dce",
    ])


# --- Native RVV (fixed-width) vectorization path --------------------------------------
# Instead of relying on clang's RISC-V auto-vectorizer of scalar loops (the scalar path
# above), we bake vector ops into the IR: a transform-dialect schedule vectorizes every
# static-shaped linalg op to fixed-width `vector<Nxf32>` and lowers the structured vector
# ops (contraction/transpose/shape_cast/masked-transfers) to LLVM-convertible primitives.
# clang -march=rv64gcv then emits real RVV (vsetvli/vle32.v/vfmacc.vv) for these vectors
# (verified end-to-end). Ops the vectorizer skips (dynamic shapes) fall through
# `convert-linalg-to-loops` to scalar code, so correctness never depends on every op
# vectorizing. Fixed-width (not scalable `vector<[n]>`) is deliberate: scalable
# transpose/shape_cast lowering is incomplete in this LLVM-23 build; fixed-width lowers
# cleanly and the RISC-V backend still targets RVV.
# We vectorize the CONTRACTION ops — `linalg.matmul` (FFN + projections) and
# `linalg.batch_matmul` (attention QKᵀ / attn·V) — that is where ~all the compute cycles live,
# and they vectorize cleanly. We do NOT blanket-vectorize: a `vectorize_children` on a whole
# transformer scalarizes the transposes/reduces/elementwise generics into tens of thousands of
# `vector.extract`s that don't lower in this LLVM-23 build. The named contraction ops are
# recovered first by `linalg-specialize-generic-ops` (the capture emits attention as
# `linalg.generic`); everything non-contraction stays linalg and goes through
# `convert-linalg-to-loops` (scalar). Tile + vectorize at fixed sizes (matmul [M4,N8,K1];
# batch_matmul [B1,M4,N8,K1]) — uniform sizes that fit any shape (smaller dims masked); proven
# to emit real RVV (vsetvli/vle32.v/vfmacc) and lower end-to-end with no extract explosion.
RVV_TRANSFORM_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %t vector_sizes [4, 8, 1] : !transform.any_op
    %bm = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt, %bl:4 = transform.structured.tile_using_for %bm tile_sizes [1, 4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt vector_sizes [1, 4, 8, 1] : !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_transpose
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


def build_rvv_pipeline(sched_path: "str | Path", hoist_static_allocs: bool = True,
                       features: "frozenset[str]" = frozenset()) -> str:
    """Whole-module pipeline with the transform vectorization stage spliced in after
    named-op generalization (vectorize on tensors) and before bufferization, plus the
    vector-lowering passes needed to reach LLVM. ``sched_path`` is the preloaded schedule.

    ``hoist_static_allocs=False`` drops the ``hoist-static-allocs`` option of
    buffer-results-to-out-params so static intermediate buffers stay HEAP (memref.alloc) instead
    of being promoted to stack ``alloca``. Needed for the K1 Linux target: big models' lowered
    intermediates otherwise overflow even a multi-GB stack; on the heap they use the board RAM.
    The bare-metal spike/Zephyr targets keep hoisting (default True) — their big linker-reserved
    stack tolerates it and their malloc arena is bounded."""
    brop = ("buffer-results-to-out-params{modify-public-functions hoist-static-allocs}"
            if hoist_static_allocs
            else "buffer-results-to-out-params{modify-public-functions}")
    late_hoist: list[str] = []
    passes = [
        "canonicalize", "cse",
        # Recover named contraction ops (matmul/batch_matmul) from the capture's generics so
        # the schedule can match them, THEN vectorize. Do NOT run linalg-fuse-elementwise-ops
        # before this — it folds matmuls into fused generics and the `ops{["linalg.matmul"]}`
        # match then finds nothing (silent 0-vectorization).
        "func.func(linalg-specialize-generic-ops)",
        f"transform-preload-library{{transform-library-paths={sched_path}}}",
        "transform-interpreter{entry-point=__transform_main}",
        "canonicalize", "cse",
        # POST-matmul elementwise fusion (env-gated, default OFF -> baseline byte-identical): after the
        # schedule has tiled/vectorized the matmuls (so the `ops{["linalg.matmul"]}` match already ran),
        # it is now SAFE to fuse the remaining elementwise generics — collapsing the non-matmul producer
        # ->consumer chains so they don't each materialize an intermediate to DRAM (the dispatch-overhead
        # bucket that loses openvla/rdt2 to XNNPACK). cos-gated before any claim.
        *( ["func.func(linalg-fuse-elementwise-ops)", "canonicalize", "cse"]
           if __import__("os").environ.get("MERLIN_FUSE_POST") else [] ),
        "func.func(linalg-generalize-named-ops)",   # remaining (non-vectorized) ops -> loops
        "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
        brop,
        "func.func(buffer-hoisting,buffer-loop-hoisting)",
        "func.func(lower-vector-multi-reduction)",
        "func.func(lower-vector-mask)",
        "convert-vector-to-scf",
        "func.func(convert-linalg-to-loops)",   # fallback: any op the vectorizer skipped
        *late_hoist,                            # K1: hoist loop-body alloca temps out of loops
        "convert-scf-to-cf",
        "expand-strided-metadata",
        "lower-affine",
        "convert-vector-to-llvm",
        "convert-ub-to-llvm",
        "convert-math-to-libm",
        "convert-math-to-llvm",
        "convert-index-to-llvm",
        "convert-arith-to-llvm",
        "finalize-memref-to-llvm",
        "convert-func-to-llvm",
        "convert-cf-to-llvm",
        "reconcile-unrealized-casts",
        "canonicalize", "cse", "symbol-dce",
    ]
    if features:                       # default-off impr-fork hooks; empty -> byte-identical
        from .impr_features import apply_pipeline
        passes = apply_pipeline(passes, features)
    return ",".join(passes)


_RUNNER = r'''
import sys

from torch_mlir import ir
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects import llvm

src_path, out_path, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]
ctx = ir.Context()
with open(src_path) as f:
    module = ir.Module.parse(f.read(), ctx)
PassManager.parse("builtin.module(" + pipeline + ")", ctx).run(module.operation)
with open(out_path, "w") as f:
    f.write(str(llvm.translate_module_to_llvmir(module.operation)))
print("OK")
'''


# Runner variant for the vectorized_transcendental_activation feature: BEFORE the pass manager runs,
# rewrite every math.exp/erf/tanh into its inline arith polynomial (so the subsequent transform
# schedule vectorizes the activation generic's polynomial to vfmacc chains, instead of
# convert-math-to-libm scalarizing the math op to a libm call loop). The rewriter source is spliced
# in from act_poly.rewrite_source(); `_ir` is the alias the rewriter uses for the MLIR ir module.
# Default-off: only used when the feature is enabled (the baseline keeps the plain _RUNNER above).
_RUNNER_ACT_POLY_HEAD = r'''
import sys

from torch_mlir import ir
from torch_mlir import ir as _ir
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects import llvm
'''

_RUNNER_ACT_POLY_TAIL = r'''
src_path, out_path, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]
ctx = ir.Context()
with open(src_path) as f:
    module = ir.Module.parse(f.read(), ctx)
with ctx, ir.Location.unknown():
    _n = apply_activation_polynomial(module, ctx)
PassManager.parse("builtin.module(" + pipeline + ")", ctx).run(module.operation)
with open(out_path, "w") as f:
    f.write(str(llvm.translate_module_to_llvmir(module.operation)))
print("OK act_poly rewrote", _n)
'''


def _accum_microkernel_v3_features() -> frozenset[str]:
    """Names of the accumulator-resident micro-kernel v3 features (the ones whose pipeline splices the
    SCALARIZE_MARKER and need the two-stage A-scalarization runner). Imported lazily so importing
    pipeline never pulls accum_microkernel / impr_features at module load."""
    from .impr_features import ACCUM_RESIDENT_V3_NAMES
    return frozenset(ACCUM_RESIDENT_V3_NAMES)


def _activation_poly_runner() -> str:
    """The lowering runner with the transcendental->polynomial rewriter spliced in (default-off
    feature). Imported here (not at module top) so importing pipeline never pulls act_poly."""
    from .act_poly import rewrite_source
    return _RUNNER_ACT_POLY_HEAD + rewrite_source() + _RUNNER_ACT_POLY_TAIL

# Parallel/OpenMP path: run the passes in the m2m venv (full LLVM-23 registry), but DUMP the
# LLVM-dialect module instead of translating in-process — the torch-mlir wheel's
# translate_module_to_llvmir segfaults on whole-model OpenMP IR (its OpenMPIRBuilder). The
# standalone llvm-install mlir-translate (same LLVM 23) translates the omp IR cleanly, so the
# translate step runs out-of-process below. The non-parallel paths keep the in-process bridge.
_RUNNER_DUMP = r'''
import sys

from torch_mlir import ir
from torch_mlir.passmanager import PassManager

src_path, out_path, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]
ctx = ir.Context()
with open(src_path) as f:
    module = ir.Module.parse(f.read(), ctx)
PassManager.parse("builtin.module(" + pipeline + ")", ctx).run(module.operation)
with open(out_path, "w") as f:
    f.write(str(module.operation))
print("OK")
'''


class PipelineError(RuntimeError):
    pass


def lower_to_llvm_ir(mlir_text: str, workdir: str | Path | None = None,
                     pipeline: str | None = None, timeout: int = 7200,
                     vectorize: bool = False, transform_schedule: str | None = None,
                     hoist_static_allocs: bool = True, parallel: bool = False,
                     features: "frozenset[str] | None" = None) -> str:
    """Lower upstream-MLIR text to LLVM IR text via the m2m venv. Returns .ll text.

    ``vectorize=True`` selects the native RVV path: writes the transform schedule into
    ``workdir`` and uses :func:`build_rvv_pipeline` so the IR carries fixed-width vector
    ops (real RVV under ``-march=rv64gcv``). ``transform_schedule`` overrides the default
    matmul/batch_matmul schedule (e.g. the elementwise/reduction schedule for the vector
    family) without disturbing it. An explicit ``pipeline`` overrides both defaults.
    """
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="merlin_lower_"))
    work.mkdir(parents=True, exist_ok=True)
    from .impr_features import apply_schedule, normalize
    feats = normalize(features)
    if pipeline is None:
        if vectorize:
            sched = work / "rvv_schedule.mlir"
            sched_text = apply_schedule(transform_schedule or RVV_TRANSFORM_SCHEDULE, feats)
            sched.write_text(sched_text, encoding="utf-8")
            pipeline = build_rvv_pipeline(sched, hoist_static_allocs=hoist_static_allocs,
                                          features=feats)
        elif parallel:
            pipeline = _parallel_pipeline()   # multicore (OpenMP) scalar path — K1 big models
        else:
            pipeline = UPSTREAM_PIPELINE
    src = work / "model.mlir"
    out = work / "model.ll"
    runner = work / "run_lowering.py"
    src.write_text(mlir_text, encoding="utf-8")

    if parallel:
        # Two-step: run passes in the m2m venv (dump LLVM-dialect MLIR), then translate to
        # LLVM IR with the standalone mlir-translate (the in-process bridge crashes on omp).
        from .toolchain import mlir_translate
        llvmdial = work / "model.llvmdialect.mlir"
        runner.write_text(_RUNNER_DUMP, encoding="utf-8")
        proc = subprocess.run(
            [str(m2m_python()), str(runner), str(src), str(llvmdial), pipeline],
            capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 or not llvmdial.is_file():
            raise PipelineError(f"upstream lowering (parallel) failed:\n{proc.stdout}\n{proc.stderr}")
        tproc = subprocess.run(
            [str(mlir_translate()), "--mlir-to-llvmir", str(llvmdial), "-o", str(out)],
            capture_output=True, text=True, timeout=timeout)
        if tproc.returncode != 0 or not out.is_file():
            raise PipelineError(f"mlir-translate (parallel) failed:\n{tproc.stdout}\n{tproc.stderr}")
        return _fix_float_literals(out.read_text(encoding="utf-8"))

    # The vectorized_transcendental_activation feature splices a math.exp/erf/tanh -> arith
    # polynomial rewriter into the runner (run before the pass manager). The accumulator-resident
    # micro-kernel v3 feature splices a two-stage runner that runs the A-operand scalarization rewrite
    # BETWEEN two pass-manager stages (split at the SCALARIZE_MARKER pass name). Default-off; with the
    # feature absent the plain _RUNNER is used and the lowering is byte-identical to the baseline.
    if "vectorized_transcendental_activation" in feats:
        runner_src = _activation_poly_runner()
    elif any(f in feats for f in _accum_microkernel_v3_features()):
        from .accum_microkernel import run_source
        runner_src = run_source()
    else:
        runner_src = _RUNNER
    runner.write_text(runner_src, encoding="utf-8")
    proc = subprocess.run(
        [str(m2m_python()), str(runner), str(src), str(out), pipeline],
        capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not out.is_file():
        raise PipelineError(f"upstream lowering failed:\n{proc.stdout}\n{proc.stderr}")
    return _fix_float_literals(out.read_text(encoding="utf-8"))


import re as _re
import struct as _struct

# The MLIR printer emits float immediates in MLIR hex (`f0x...`). LLVM IR uses bare
# `0x...`: a 16-hex pattern (f64) just drops the `f`; an 8-hex pattern (f32) must be
# widened to the double its value represents.
_F0X_RE = _re.compile(r"\bf0x([0-9A-Fa-f]{8}(?:[0-9A-Fa-f]{8})?)(?![0-9A-Fa-f])")


def _f0x_to_llvm(m: "_re.Match[str]") -> str:
    hexs = m.group(1)
    if len(hexs) == 16:                       # already a double bit pattern
        return f"0x{hexs.upper()}"
    bits = int(hexs, 16)                      # f32 -> widen to double
    f32 = _struct.unpack("<f", _struct.pack("<I", bits))[0]
    dbits = _struct.unpack("<Q", _struct.pack("<d", f32))[0]
    return f"0x{dbits:016X}"


def _fix_float_literals(ll_text: str) -> str:
    """The MLIR LLVM-IR printer emits non-LLVM float literals (bare inf/-inf/nan,
    and `f0x..` f32 hex) that the textual parser rejects — canonicalize them."""
    ll_text = _F0X_RE.sub(_f0x_to_llvm, ll_text)
    # half (fp16) uses the LLVM `0xH<4hex>` literal form, bfloat (bf16) uses `0xR<4hex>` — the MLIR
    # printer emits bare inf/-inf/nan for these too (e.g. an fp16 causal-mask -inf), which the LLVM
    # textual parser rejects. Order matters: replace the widest names first so `float`↛`bfloat` etc.
    return (ll_text
            .replace("bfloat -inf", "bfloat 0xRFF80")
            .replace("bfloat inf", "bfloat 0xR7F80")
            .replace("bfloat nan", "bfloat 0xR7FC0")
            .replace("half -inf", "half 0xHFC00")
            .replace("half inf", "half 0xH7C00")
            .replace("half nan", "half 0xH7E00")
            .replace("float -inf", "float 0xFFF0000000000000")
            .replace("float inf", "float 0x7FF0000000000000")
            .replace("float nan", "float 0x7FF8000000000000")
            .replace("double -inf", "double 0xFFF0000000000000")
            .replace("double inf", "double 0x7FF0000000000000")
            .replace("double nan", "double 0x7FF8000000000000"))
