"""Cross-compiles whole-model binaries for a remote Linux board.

OpenMP selection, compile flags, :func:`build_binary` and :func:`build_session_binary` reuse the
bare-metal and Zephyr preparation and lowering, so a board binary matches what Spike certified.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from merlin.common.paths import repo_root, runtime_dir, tracked_build_dir
from merlin.runtime.backends.linux.harness_source import main_linux_c, main_linux_session_c
from merlin.runtime.backends.linux.remote_session import (
    Profile,
    RemoteLinuxError,
    _run,
    codegen_march,
    profile,
    toolchain_cc,
    toolchain_root,
)

# ---- linux_cross_build -------------------------------------------------------------------------

if TYPE_CHECKING:
    from merlin.runtime.boards.catalog import Board


def _model_compile_flags(
    p: Profile, pkg, features, model_opt: list[str], *, honor_pkg_cflags: bool = True
) -> list[str]:
    """Compose package tuning flags under the board adapter's non-negotiable target identity.

    Packages commonly carry a generic ``-march=rv64gcv`` for portable runners. The board's model
    object needs the adapter's verified VLEN and the board's full extension set, so package march/ABI
    declarations are superseded while every actual optimization flag is preserved. This makes
    flags such as ``-fno-vectorize`` reach the compiler without allowing a package to erase the
    target contract.
    """
    base = [f"-march={codegen_march(p)}", f"-mabi={p.board.mabi}", *model_opt, "-Wno-override-module"]
    if honor_pkg_cflags:
        base = merge_package_cflags(base, getattr(pkg, "cflags", ()))
    from merlin.compiler.optimizations.registry import apply_cflags

    return apply_cflags(base, features or frozenset())


@dataclass(frozen=True)
class _OpenMPSelection:
    parallel_harts: int | None
    provider: str | None
    worker_pool_size: int | None


def _resolve_openmp_selection(*, parallel: bool, parallel_harts: int | None, host_effects=None) -> _OpenMPSelection:
    """Resolve the lowering width and the ABI provider materialized by this adapter."""
    if parallel_harts is not None and int(parallel_harts) < 1:
        raise ValueError("parallel_harts must be positive when provided")
    # A one-core budget is a serial lowering, even when a measurement harness
    # spells it explicitly as ``parallel_harts=1``.  Selecting OpenMP here used
    # to reach a transform schedule that correctly rejects widths below two.
    # Keep the legacy ``parallel=True`` switch strict because it explicitly asks
    # for a parallel lowering.
    if parallel_harts is not None and int(parallel_harts) == 1 and not parallel:
        parallel_harts = None
    if host_effects is None:
        active = bool(parallel or parallel_harts)
        return _OpenMPSelection(
            int(parallel_harts) if parallel_harts is not None else None,
            "llvm" if active else None,
            None,
        )

    from merlin.compiler.lower.host.host_policy import HostLoweringEffects

    if not isinstance(host_effects, HostLoweringEffects):
        raise TypeError("host_effects must be HostLoweringEffects or None")
    policy_harts = host_effects.parallel_harts
    if parallel_harts is not None and policy_harts != int(parallel_harts):
        raise RemoteLinuxError(
            f"explicit parallel_harts={parallel_harts} conflicts with host-policy parallel_harts={policy_harts}"
        )
    if parallel and policy_harts is not None:
        raise RemoteLinuxError("legacy scalar parallel=True conflicts with a host-policy RVV parallel lowering")
    harts = int(parallel_harts) if parallel_harts is not None else policy_harts
    provider = host_effects.openmp_runtime
    if harts is not None and provider is None:
        provider = "llvm"
    return _OpenMPSelection(harts, provider, host_effects.worker_pool_size)


def _require_worker_census(metrics: dict[str, Any], *, requested: int) -> tuple[int, ...]:
    """Fail closed unless the persistent pool observed the exact requested team."""
    observed_threads = int(metrics.get("omp_threads") or 0)
    if observed_threads != int(requested):
        raise RemoteLinuxError(f"persistent OpenMP census observed {observed_threads} threads; requested {requested}")
    raw = metrics.get("omp_worker_cpus")
    if isinstance(raw, int):
        cpus = (raw,)
    elif isinstance(raw, str):
        try:
            cpus = tuple(int(value) for value in raw.split())
        except ValueError as exc:
            raise RemoteLinuxError(f"invalid persistent OpenMP worker CPU census: {raw!r}") from exc
    else:
        cpus = ()
    if len(cpus) != observed_threads:
        raise RemoteLinuxError(
            f"persistent OpenMP worker CPU census has {len(cpus)} entries; expected {observed_threads}"
        )
    if any(cpu < 0 for cpu in cpus):
        raise RemoteLinuxError(f"persistent OpenMP worker CPU census contains invalid CPU: {cpus}")
    if len(set(cpus)) != len(cpus):
        raise RemoteLinuxError(f"persistent OpenMP worker CPU census contains duplicate CPUs: {cpus}")
    return cpus


def _openmp_library(p: Profile) -> Path:
    """The board's cross-built static OpenMP runtime (``libomp.a``), which provides the ``__kmpc_*``
    symbols the multicore (parallel) lowering emits; the thread count is chosen at run time with
    OMP_NUM_THREADS. Built under the repository build root (see the OpenMP path in
    compiler.lower.native.policy) -- a machine asset, so it is not redirected with the output root."""
    if not p.board.openmp_runtime_dir:
        raise RemoteLinuxError(f"board {p.name!r} declares no openmp_runtime_dir; it has no multicore runtime")
    return tracked_build_dir() / p.board.openmp_runtime_dir / "libomp.a"


#: Flag classes the board's model-object build DERIVES for itself and a package may therefore not
#: override. ``-march``/``-mabi`` are TARGET facts (the board's real VLEN, its extensions, its ABI)
#: resolved by :func:`codegen_march` and the board's ``mabi``; a package that also spells an
#: ``-march`` is spelling a weaker one (``rv64gcv`` promises only the RVV MINIMUM VLEN of 128, which
#: doubles every register group on a board with a wider VLEN -- see :func:`codegen_march`). Letting
#: the package win there would silently un-pin the vector length and drop extensions while appearing
#: to be a change of vectorizer flags, so those two are dropped from the package list and the derived
#: values are kept.
_PKG_CFLAGS_DERIVED_HERE = ("-march=", "-mabi=")


def merge_package_cflags(base: list[str], pkg_cflags) -> list[str]:
    """``base`` (the derived model-object flags) extended with the PACKAGE's declared cflags.

    A package's ``knobs.yaml`` may declare cflags (``rvv/hand_v0_int8`` declares
    ``-fno-vectorize -fno-slp-vectorize``). Until this existed the board build never read them, so
    clang's own loop and SLP vectorizers ran on our emitted ``model.ll`` and an unknown share of the
    vector code in the binary was theirs rather than the transform schedule's -- the same
    inert-declaration failure this module's comments warn about, in the other direction: there a flag
    was never handed to the compiler; here a flag was never READ from the thing that declared it.

    FAIL CLOSED. A package cflag that is neither derived here nor understood is an ERROR rather than
    a silent drop: silently dropping is precisely the defect being removed. Flags whose class this
    build derives for itself (:data:`_PKG_CFLAGS_DERIVED_HERE`) are dropped DELIBERATELY and the
    derived value kept, because they are facts about the board rather than package preferences.
    """
    out = list(base)
    for flag in pkg_cflags or ():
        if flag.startswith(_PKG_CFLAGS_DERIVED_HERE):
            continue  # derived from the target; the package's weaker spelling loses
        if not flag.startswith("-"):
            raise RemoteLinuxError(f"package cflag {flag!r} is not a flag; refusing to guess its meaning")
        if flag not in out:
            out.append(flag)
    return out


def _require_toolchain(p: Profile) -> tuple[Path, Path]:
    """The board's cross compiler driver and toolchain prefix, or a fail-closed error naming the variable."""
    cc = toolchain_cc(p)
    root = toolchain_root(p)
    if cc is None or root is None:
        where = (
            p.toolchain_install
            if p.toolchain_install is not None
            else f"${p.env('TOOLCHAIN')} (unset, and no contract declares a default install)"
        )
        raise RemoteLinuxError(f"{p.name} cross-toolchain not found under {where} (set {p.env('TOOLCHAIN')})")
    return cc, root


def _sysroot(p: Profile, root: Path) -> Path | None:
    return root / p.board.toolchain_sysroot if p.board.toolchain_sysroot else None


def build_binary(
    board: Board | Profile,
    model_dir: str | Path,
    work: str | Path,
    pkg,
    inputs_npz: str | Path | None = None,
    force_scalar: bool | None = None,
    parallel: bool = False,
    parallel_harts: int | None = None,
    host_effects=None,
    fallback_policy: str = "allow",
    mmap_weights: bool | None = None,
    kernel_backend: str | None = None,
    dispatch_timing: bool = False,
    op_profile: bool = False,
    dump_cap: int | None = 4096,
    max_session_steps: int | None = None,
    honor_pkg_cflags: bool = False,
) -> Path:
    """Cross-compile a Linux RVV binary for ``board`` from the workload + RVV package.

    Reuses the EXACT spike/Zephyr lowering (``zephyr_preparation.prepare_for_lowering`` ->
    ``lower_model_file(vectorize=True, transform_schedule=pkg.schedule_text)``) so the board binary
    executes the same emitted RVV as the spike correctness run, and the same data-driven C runtime
    artifacts (``c_runtime.generate`` -> model_gen.h / model_call.c / weights.bin / model_io.h).
    Only the harness (``main_linux.c``) and the compile flags differ (glibc Linux, not medany).

    ``mmap_weights``: True -> the weight blob is NOT embedded; the binary ``mmap``s it from a file
    at run time (deployed alongside, ``MERLIN_WEIGHTS`` env). None (default) -> auto by blob size
    (>= the board's ``mmap_weights_threshold_bytes`` -> mmap). When mmap is used a marker file
    ``<work>/USE_MMAP_WEIGHTS`` records the weights.bin path so :func:`run` deploys it.
    Returns the path to the built ELF.

    ``kernel_backend="outlined_int8"`` links the reusable outlined W8A8 RVV GEMM body; ``None``
    (default) lowers every operation through the ordinary pipeline."""
    p = profile(board)
    mabi = p.board.mabi
    if fallback_policy not in ("allow", "forbid"):
        raise ValueError("fallback_policy must be 'allow' or 'forbid'")
    if kernel_backend not in {None, "outlined_int8"}:
        raise RemoteLinuxError(f"unknown kernel backend {kernel_backend!r}")
    openmp = _resolve_openmp_selection(parallel=parallel, parallel_harts=parallel_harts, host_effects=host_effects)
    parallel_harts = openmp.parallel_harts
    # Per-dispatch matmul-bucket timing needs a routed GEMM shim that carries the timer.
    if dispatch_timing:
        raise RemoteLinuxError("dispatch_timing=True requires a routed GEMM shim with a matmul-bucket timer")
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    # PREPACKED WEIGHT LAYOUT (`prepack_weight_layout`, default-off). Apply the model's weight
    # re-layout ONCE, here, instead of on every inference: materialize a rewritten bundle (the
    # argument transposes erased and the arguments retyped, the safetensors bytes physically
    # transposed) and use it as `model_dir` from this point on. It has to reach BOTH consumers --
    # `prepare_for_lowering` below reads `model_dir/model.mlir`, while `c_runtime.generate` (step 3)
    # independently re-parses the SAME bundle for the ABI table and copies its weight payload
    # verbatim into `weights.bin`. A rewrite that reached only one of them would leave the compiled
    # object indexing a transposed weight that the blob stores untransposed. Measured on silicon,
    # interleaved same-session: 1.70x on small_llama int8. The rewrite is cached per (bundle,
    # rewrite) and NEVER mutates the source recapture, which every other session reads.
    from merlin.compiler.lower.options import weight_prepack as _wp

    _wp.ensure_registered()  # unregistered => `_composes` swallows the KeyError and never proposes it
    if _wp.FEATURE in (getattr(pkg, "compiler_features", None) or ()):
        try:
            model_dir, _prepack = _wp.prepacked_bundle(model_dir)
        except _wp.PrepackRefused as exc:
            # Fail CLOSED. Falling back to the stock bundle would build the BASELINE while reporting
            # the lever as applied -- the inert-lever failure, but silent and un-auditable.
            raise RemoteLinuxError(f"{_wp.FEATURE}: {exc}") from exc
        (work / "PREPACKED_BUNDLE").write_text(str(model_dir))
        print(
            f"[prepack] {model_dir.name}: "
            f"{_prepack.get('weights_pre_transposed')} weights pre-transposed, "
            f"{_prepack.get('transposes_removed')} transposes removed, "
            f"{_prepack.get('mib_moved_per_inference_before')} MiB/inference no longer moved "
            f"(cached={_prepack.get('cached')})"
        )
    # AOT CONSTANT-WEIGHT QUANTIZATION (`prequantize_constant_weights`, default-off). This is the
    # same kind of ABI-bearing bundle rewrite as prepacking above: the prepared compiler input and
    # c_runtime's argument table must both see the rewritten signature + payload. Apply it after
    # layout prepacking so two requested bundle transforms form one explicit provenance chain.
    from merlin.compiler.lower.options import weight_prequant as _wq

    _wq.ensure_registered()
    if _wq.FEATURE in (getattr(pkg, "compiler_features", None) or ()):
        if not pkg.is_int8:
            raise RemoteLinuxError(f"{_wq.FEATURE} requires an INT8 package")
        try:
            model_dir, _prequant = _wq.prequantized_bundle(model_dir)
        except _wq.PrequantizeRefused as exc:
            raise RemoteLinuxError(f"{_wq.FEATURE}: {exc}") from exc
        (work / "PREQUANTIZED_BUNDLE").write_text(str(model_dir))
        print(
            f"[prequant] {model_dir.name}: "
            f"{_prequant.get('weights_prequantized')} constant matrices stored as i8+scale "
            f"(cached={_prequant.get('cached')})"
        )
    inputs_npz = inputs_npz or (model_dir / "inputs.npz")
    cc, root = _require_toolchain(p)
    # binutils ld for the binary-blob object (use the toolchain's GNU ld, not lld).
    ld = root / "bin" / f"{p.triple}-ld"
    if not ld.is_file():
        ld = Path(shutil.which("ld") or "ld")

    from merlin.compiler.build import model_preparation as zm
    from merlin.compiler.lower.c import runtime as c_runtime
    from merlin.compiler.lower.model import toolchain
    from merlin.compiler.lower.model.lower import lower_model_file

    # 1. model.mlir -> normalize (same prep passes as the package/spike build) -> model.ll.
    #
    # `prepare_for_lowering`, NOT the lower-level `_prepare_model_mlir` this used to call. That was a
    # real gap, not a style point: three feature families live in the preparation layer, not in the
    # pass pipeline -- per-op register blocking (the `perop_register_block` sentinel is derived from the
    # PREPARED IR, tagged onto it, and swapped for a concrete feature), the non-contraction vec-rank
    # tagging, and matrix-unit routing. Calling the low-level helper meant a package naming any of them
    # either silently lowered as the BASELINE (nothing tags the IR, so no schedule arm matches and every
    # contraction falls to convert-linalg-to-loops) or died in `normalize` on the unresolved sentinel.
    # Since a silicon board is this repo's performance oracle, that made those levers unmeasurable on
    # silicon -- while this function's own docstring claimed it "reuses the EXACT spike/Zephyr
    # lowering". Both `zephyr_build.build_app` and `spike_build.build` go through this same entry
    # point, so now all three agree by construction.
    feats_in = frozenset(getattr(pkg, "compiler_features", []) or []) or None
    prepared, feats = zm.prepare_for_lowering(
        model_dir / "model.mlir",
        work,
        int8_compute=pkg.is_int8,
        features=feats_in,
        harts=(int(parallel_harts) if parallel_harts else 1),
        vlen=p.vlen,
        outline_int8=kernel_backend == "outlined_int8",
    )
    feats = feats or None
    # OUTLINED W8A8 backend (default-off): the preparation seam above replaces each legal rank-2
    # i8xi8->i32 contraction with a call before panel/block expansion. One small MR4/NR32 RVV body
    # is compiled here and linked below; batched attention contractions continue through Merlin.
    outlined_int8_obj = None
    n_outlined_int8_routed = 0
    if kernel_backend == "outlined_int8":
        from merlin.compiler.backend import providers as provided

        oi = provided.component("outlined_int8")

        if not pkg.is_int8:
            raise RemoteLinuxError("kernel_backend='outlined_int8' requires an INT8 package")
        if not oi.is_available():
            raise RemoteLinuxError("kernel_backend='outlined_int8' but its RVV shim is unavailable")
        signatures = oi.load_signatures(work)
        if not signatures:
            raise RemoteLinuxError("outlined INT8 preparation routed no signatures")
        n_outlined_int8_routed = sum(1 for record in json.loads((work / oi.SIDECAR_NAME).read_text()).get("routed", ()))
        outlined_int8_obj = oi.build_object(
            cc,
            [p.target_flag, f"-march={p.march}", f"-mabi={mabi}", "-O3", "-DNDEBUG"],
            signatures,
            work / "outlined_int8",
            parallel=bool(parallel_harts),
        )
    # PER-OP WHOLE-MODEL PROFILE (default-off, additive). Interleave `@merlin_prof_mark(i32)`
    # calls between the top-level ops of @forward so the board can attribute wall time to the ops
    # the model actually executes — the 94-97% of model time the matmul-bucket timer above cannot
    # see. Runs LAST (after any kernel-backend rewrite) so a routed matmul is profiled as the call
    # it became. The id->op table is written next to the build for the driver to join against.
    if op_profile:
        from merlin.compiler.lower.model import profile as opf

        instrumented, prof_table = opf.instrument(prepared.read_text())
        prepared = work / "model.prepared.opprof.mlir"
        prepared.write_text(instrumented)
        opf.write_table(prof_table, work / "opprof_table.json")

    # hoist_static_allocs=False: keep big intermediate buffers on the HEAP (board RAM) instead of
    # promoting them to stack alloca — large models otherwise overflow even a multi-GB stack.
    from merlin.compiler.lower.native.pipeline import PipelineError

    # impr-fork compiler features (PASS/HEURISTIC/PATTERN). Without threading these the board build
    # silently used the BASELINE codegen for any feature-bearing fork — invalidating its
    # benchmark. Pass through so the board binary matches what spike certified.
    # `feats` was resolved above by prepare_for_lowering (which may have swapped a sentinel for a
    # concrete, table-specific feature) — do NOT re-read the package's raw list here, or the schedule
    # would be built from a name the IR was never tagged for.
    if force_scalar is None:
        force_scalar = bool(os.environ.get(p.env("FORCE_SCALAR")))
    if parallel_harts:
        # VECTOR + MULTICORE: the package's RVV schedule with an outer OpenMP-parallel loop
        # layered under it, so the board runs real RVV on several cores at once. This is the
        # same lowering the multicore Zephyr image uses, which makes a silicon board the honest
        # PERFORMANCE oracle for it: spike cannot answer that question (it simulates a spinning
        # hart at full speed), and RTL sim cannot run a whole model. Thread count is chosen at
        # RUN time via OMP_NUM_THREADS, so one binary yields a whole scaling curve.
        # The split `prepare_for_lowering` derived and TAGGED the IR with. Reading it back (rather
        # than re-deriving it here) is what keeps the tags and the schedule arms one decision: a
        # second derivation could disagree with the tags and silently leave every contraction
        # unsplit, which looks exactly like a model that does not parallelize.
        res = lower_model_file(
            prepared,
            work / "lower_vecomp",
            targets=(),
            textual=True,
            vectorize=True,
            transform_schedule=pkg.schedule_text,
            hoist_static_allocs=False,
            features=feats,
            parallel_harts=parallel_harts,
            parallel_chunks=zm.parallel_arms(work),
        )
    elif parallel:
        # Multicore path: scalar int8 datapath + OpenMP parallel loops (no fixed-width
        # vectorize). Used for the big models (rdt/smolvla) that crash the vectorized lowering
        # AND are too slow single-core — the parallel loops fan across the board's cores.
        res = lower_model_file(
            prepared,
            work / "lower_omp",
            targets=(),
            textual=True,
            vectorize=False,
            hoist_static_allocs=False,
            parallel=True,
        )
    else:
        try:
            if force_scalar:
                raise PipelineError(f"forced scalar ({p.env('FORCE_SCALAR')})")
            res = lower_model_file(
                prepared,
                work / "lower",
                targets=(),
                textual=True,
                vectorize=True,
                transform_schedule=pkg.schedule_text,
                hoist_static_allocs=False,
                features=feats,
            )
        except PipelineError:
            # Some models (e.g. xr0's rank-4 two-batch attention) hit a vectorize-path
            # specialization (linalg-specialize-generic-ops) that emits an invalid rank-4
            # linalg.batch_matmul. Fall back to the SCALAR lowering (no specialize pass): the int8
            # datapath is intact, only the contraction stays a scalar loop (correct, unvectorized).
            if fallback_policy == "forbid":
                raise
            res = lower_model_file(
                prepared, work / "lower_scalar", targets=(), textual=True, vectorize=False, hoist_static_allocs=False
            )

    # 2. compile model.ll -> the board's Linux object. The IR is emitted by the repo's LLVM and
    #    carries its attribute syntax (e.g. `captures(none)`), which an older vendor cross-clang
    #    cannot parse, so the .ll -> object step uses THE SAME repo clang re-targeted to the board's
    #    triple; the board toolchain's own compiler builds the C sources + final glibc link below.
    #    NO medany/freestanding — glibc hosted, small code model, fully relocatable.
    clang23 = toolchain.clang()
    model_o = work / "model.o"
    # codegen_march(), not the board's plain march: this is OUR emitted model.ll becoming the model
    # object, so it pays the doubled-LMUL penalty when the march string leaves VLEN at the RVV
    # minimum. The C glue and the baseline shim arms below deliberately keep the plain march so their
    # flags are not re-written mid-campaign. Whether this transfers to whole models is exactly what
    # is being measured.
    # MERLIN_<BOARD>_MODEL_OPT (default -O2 -> byte-identical) lowers the OPT LEVEL of the model
    # object only. It exists for one job: when the same LLVM IR is correct on x86 and wrong on every
    # riscv64 target, the next cut is whether the RISC-V backend's OPTIMIZED codegen is what breaks
    # it, and that needs -O0/-O1 builds of the model while everything else (harness, runtime, flags)
    # is held fixed. Diagnostic only -- never set it for a measurement, since it changes the kernel.
    model_opt = os.environ.get(p.env("MODEL_OPT"), "-O2").split()
    # CFLAGS-CLASS FEATURES apply to the MODEL OBJECT ONLY. Some levers are not expressible in the
    # IR or the transform schedule at all: the vector register-group width comes from clang's own
    # RISC-V vectorizer (`-riscv-v-register-bit-width-lmul`, default 2), so a feature that resolves
    # and is never handed to the compiler emits byte-identical code while reporting as applied --
    # the inert-lever failure this file's own comments warn about. Only the model object gets these;
    # the harness and runtime stay on fixed flags so a measurement changes one thing.
    # ``honor_pkg_cflags=False`` is retained for the explicit A/B diagnostic. Production builds
    # honor package optimization flags while the helper keeps target-owned march/ABI facts pinned.
    _model_flags = _model_compile_flags(p, pkg, feats, model_opt, honor_pkg_cflags=honor_pkg_cflags)
    _run([clang23, p.target_flag, *_model_flags, "-c", res.ll_path, "-o", model_o])
    from merlin.compiler.ir import compile_trace as _trace

    _trace.artifact("object", [model_o], pipeline="codegen")
    # 2b. POST-CODEGEN CENSUS: is the model still IN the object? A backend that deletes reachable
    # code still links, still produces a binary, and then reports a spectacular speedup for
    # computing nothing -- smolvla linked a 512 MB ELF whose `forward` was 3,654 bytes with a
    # complete call set of malloc/memset/roundevenf. Nothing between here and the board could see
    # it. The bound is derived from the prepared IR (one instruction per structured op that reaches
    # an output) and carries no constant; it raises, so an erased model cannot be timed.
    from merlin.compiler.lower.model.codegen_census import require_commensurate as _census_require

    _object_census = _census_require(prepared, model_o, "forward")
    print(f"[census gate object] {_object_census.as_dict()}", flush=True)

    # 3. data-driven runtime artifacts (arg table, ciface, weights.bin, embedded io).
    cgen = work / "cgen"
    # `max_session_steps` caps how much of a session corpus is EMBEDDED AS C LITERALS. It is not a
    # micro-optimization for big corpora: resnet50's 256-step, 154 MB session_inputs.npz becomes a
    # 770 MB model_io.h that costs ~7 GB of RSS to compile, and a correctness or per-step latency
    # run needs a handful of steps. None (the default) embeds the whole corpus, unchanged.
    # PANEL-PACKED WEIGHTS (`prepack_weight_panels`, default-off). `prepare_for_lowering` retyped the
    # packed `@forward` arguments in the module that became `model.ll`; this materializes the bundle
    # whose weights.safetensors and manifest were packed to match and checks the two signatures agree
    # argument by argument before the ABI table is built from it. Returns `model_dir` unchanged when
    # the build packed nothing, so a run naming no feature is byte-identical. The packer belongs to the
    # selected vector support provider; with none selected nothing was packed.
    from merlin.compiler.backend.providers import abi_bundle

    model_dir, _panel = abi_bundle(model_dir, work, prepared)
    if _panel is not None:
        (work / "PANEL_PACKED_BUNDLE").write_text(str(model_dir))
        print(
            f"[weight_panel] {Path(model_dir).name}: {_panel['weights_packed']} weights packed "
            f"into panels of {_panel['panel_widths']} columns, {_panel['args_retyped']} arguments "
            f"retyped, {_panel['abi_args_checked']} ABI arguments checked "
            f"(cached={_panel.get('cached')})"
        )
    cinfo = c_runtime.generate(model_dir, cgen, inputs_npz, max_session_steps=max_session_steps, prepared_dir=work)
    if cinfo.get("has_session_quality"):
        (work / "HAS_SESSION_QUALITY").write_text("1")

    # Decide embed-vs-mmap by the weight-blob size (auto unless caller forced it). A blob at or
    # above the board's threshold would make a binary too large for its scratch directory and force
    # every weight resident; mmap keeps resident RAM to the working set. Small models stay embedded
    # (self-contained, proven path).
    weights_bin = cgen / "weights.bin"
    if mmap_weights is None:
        threshold = int(p.board.mmap_weights_threshold_bytes)
        mmap_weights = weights_bin.stat().st_size >= threshold if weights_bin.is_file() else False

    # 4. weights.bin -> binary blob object (embedded path only; mmap reads the file at run time).
    weights_o = work / "weights_blob.o"
    if not mmap_weights:
        _run([ld, "-r", "-b", "binary", "-o", weights_o, "weights.bin"], cwd=cgen)
    else:
        (work / "USE_MMAP_WEIGHTS").write_text(str(weights_bin))

    # 5. the Linux harness.
    main_c = work / "main_linux.c"
    main_c.write_text(
        main_linux_c(
            p,
            dump_cap=dump_cap,
            mmap_weights=mmap_weights,
            dispatch_timing=dispatch_timing,
            op_profile=op_profile,
            persistent_openmp_pool=openmp.worker_pool_size,
        )
    )

    # 6. link the final binary. Reuse the repo's portable C runtime + generated ciface. Prefer a
    #    static binary (no glibc-version coupling to the board); fall back to dynamic, which is
    #    fine since it runs on the board's own glibc.
    rt = runtime_dir() / "c"
    abi = runtime_dir() / "abi"
    binary = work / p.binary
    opt = os.environ.get(p.env("OPT"), "-O2")

    # The whole-model ciface `model_call.c` is one huge function (200+ weight args). A board whose
    # vendor compiler cannot handle it (a register-scavenger crash on large models, independent of
    # precision and opt level) declares `model_call_compiler: llvm`: just model_call.c is compiled
    # with the repo's clang (it needs no stdlib headers) against the toolchain's sysroot, and the
    # other small C sources + the link stay on the board toolchain, which has the libc/crt and
    # builtin headers. The toolchain links that object fine (ELF/ABI compatible; model.o above is
    # likewise produced by the repo's clang).
    from merlin.runtime.boards.catalog import MODEL_CALL_LLVM

    model_call_c = cgen / "model_call.c"
    sysroot = _sysroot(p, root)
    other_c = [main_c, rt / "merlin_model.c", abi / "mlir_runtime.c"]
    if openmp.provider == "merlin_pthread_pool":
        other_c = other_c + [rt / "libomp_pthread.c"]
    if op_profile:  # per-op tick accumulator the instrumented IR calls
        other_c = other_c + [rt / "merlin_op_prof.c"]
    # MEASUREMENT PROXY for the lean runtime's static arena (env MERLIN_BUMP_MALLOC, default OFF):
    # link a bump allocator that overrides glibc malloc/free, isolating how much of the dispatch
    # overhead is the ~4391 per-op malloc/free the lean arena eliminates. Baseline byte-identical when off.
    if os.environ.get("MERLIN_BUMP_MALLOC"):
        other_c = other_c + [rt / "merlin_bump_linux.c"]

    if (
        p.board.model_call_compiler == MODEL_CALL_LLVM
        and sysroot is not None
        and Path(clang23).exists()
        and sysroot.is_dir()
    ):
        model_call_o = work / "model_call.o"
        _run(
            [
                str(clang23),
                p.target_flag,
                f"-march={p.march}",
                f"-mabi={mabi}",
                opt,
                "-Wno-override-module",
                f"--sysroot={sysroot}",
                f"--gcc-toolchain={root}",
                f"-I{rt}",
                f"-I{cgen}",
                "-c",
                str(model_call_c),
                "-o",
                str(model_call_o),
            ]
        )
        c_inputs = [*other_c, model_call_o]
    else:
        c_inputs = [*other_c, model_call_c]  # the board toolchain compiles model_call.c itself
    base = [
        cc,
        p.target_flag,
        f"-march={p.march}",
        f"-mabi={mabi}",
        opt,
        "-Wno-override-module",
        f"-I{rt}",
        f"-I{cgen}",
        *c_inputs,
        model_o,
    ]
    if outlined_int8_obj is not None:  # reusable W8A8 RVV GEMM body
        base += [str(outlined_int8_obj)]
    if not mmap_weights:
        base += [str(weights_o)]
    if (parallel or parallel_harts) and openmp.provider == "llvm":
        # Resolve the cross-built __kmpc_* symbols against the static libomp; its LLVM runtime
        # needs the C++/dl deps (see the libomp build note). Order matters: libomp before its deps.
        libomp = _openmp_library(p)
        if not libomp.is_file():
            raise RemoteLinuxError(f"parallel build needs {libomp} (cross-build libomp first)")
        base += [str(libomp), "-lstdc++", "-ldl"]
    base += ["-lm", "-lpthread", "-o", binary]
    try:
        _run([*base, "-static"])
    except RemoteLinuxError:
        _run(base)
    if not binary.is_file():
        raise RemoteLinuxError(f"{p.name} cross-compile produced no binary at {binary}")
    _trace.artifact("link", [binary], pipeline="codegen")
    # The object census above is the strong erasure gate because its whole-object count cannot be
    # inflated by libc/runtime code.  It is NOT the delivered-code measurement: unresolved local
    # calls are only attributed to ``forward`` after relocation (122x difference on tiny_llama).
    # Record both roles explicitly and make the linked ELF the only headline census we print.
    _linked_census = _census_require(prepared, binary, "forward")
    _census_report = {
        "gate_object": _object_census.as_dict(),
        "linked_elf": _linked_census.as_dict(),
    }
    (work / "codegen_census.json").write_text(
        json.dumps(_census_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[census linked ELF] {_linked_census.as_dict()}", flush=True)
    if kernel_backend == "outlined_int8":
        (work / "N_OUTLINED_INT8_ROUTED").write_text(str(n_outlined_int8_routed))
    return binary


def build_session_binary(
    board: Board | Profile,
    model_dir: str | Path,
    work: str | Path,
    pkg,
    *,
    force_scalar: bool = False,
    parallel_harts: int | None = None,
    host_effects=None,
    fallback_policy: str = "forbid",
    kernel_backend: str | None = None,
    dump_cap: int | None = 4096,
) -> Path:
    """Cross-compile all programs in a version-2 session and link one board process.

    All stages share one memory and timing domain in a single process.
    """
    p = profile(board)
    mabi = p.board.mabi
    if fallback_policy not in {"allow", "forbid"}:
        raise ValueError("fallback_policy must be 'allow' or 'forbid'")
    openmp = _resolve_openmp_selection(parallel=False, parallel_harts=parallel_harts, host_effects=host_effects)
    parallel_harts = openmp.parallel_harts
    if kernel_backend is not None:
        raise RemoteLinuxError(f"multi-program kernel backend {kernel_backend!r} is not supported")
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cc, root = _require_toolchain(p)
    from merlin.compiler.build import model_preparation as zm
    from merlin.compiler.lower.c import session as session_bundle
    from merlin.compiler.lower.model import toolchain
    from merlin.compiler.lower.model.lower import lower_model_file
    from merlin.compiler.lower.native.pipeline import PipelineError

    generated = work / "session_gen"
    manifest = session_bundle.generate(model_dir, generated)
    clang23 = toolchain.clang()
    sysroot = _sysroot(p, root)
    rt = runtime_dir() / "c"
    abi = runtime_dir() / "abi"
    model_opt = os.environ.get(p.env("MODEL_OPT"), "-O2").split()
    opt = os.environ.get(p.env("OPT"), "-O2")
    features = frozenset(getattr(pkg, "compiler_features", []) or []) or None
    objects: list[Path] = []
    weights: list[dict[str, str]] = []
    stage_censuses: list[tuple[Path, str, dict[str, Any]]] = []

    for record in manifest["programs"]:
        index, name = int(record["index"]), str(record["name"])
        bundle_dir = Path(record["bundle"])
        stage_work = work / f"stage_{index}_{name}"
        stage_work.mkdir(parents=True, exist_ok=True)
        # A session stage is a whole model with a different entrypoint.  Route it through the same
        # preparation seam as the single-program board/Spike/Zephyr builders: per-operation blocks,
        # parallel splits, vector-rank tags, and concrete feature names are derived from THIS
        # stage's prepared IR.  Calling `_prepare_model_mlir` here used to omit those tables and
        # handed the lowering the package's unresolved, session-global request instead.
        prepared, stage_features = zm.prepare_for_lowering(
            bundle_dir / "model.mlir",
            stage_work,
            int8_compute=pkg.is_int8,
            features=features,
            harts=(int(parallel_harts) if parallel_harts else 1),
            vlen=p.vlen,
        )
        stage_features = stage_features or None
        renamed = session_bundle.rename_forward(prepared.read_text(encoding="utf-8"), str(record["entrypoint"]))
        prepared = stage_work / "model.prepared.renamed.mlir"
        prepared.write_text(renamed, encoding="utf-8")
        try:
            if force_scalar:
                raise PipelineError("forced scalar")
            lowered = lower_model_file(
                prepared,
                stage_work / "lower",
                targets=(),
                textual=True,
                vectorize=True,
                transform_schedule=pkg.schedule_text,
                hoist_static_allocs=False,
                features=stage_features,
                parallel_harts=parallel_harts,
                parallel_chunks=zm.parallel_arms(stage_work),
            )
        except PipelineError:
            if fallback_policy == "forbid":
                raise
            lowered = lower_model_file(
                prepared,
                stage_work / "lower_scalar",
                targets=(),
                textual=True,
                vectorize=False,
                hoist_static_allocs=False,
            )
        model_object = stage_work / "model.o"
        # Same cflags-class features as the primary site; a staged sibling that skipped them would
        # measure a different compiler than the one the feature set names.
        _stage_flags = _model_compile_flags(p, pkg, stage_features, model_opt)
        _run([clang23, p.target_flag, *_stage_flags, "-c", lowered.ll_path, "-o", model_object])
        # Same post-codegen census as the primary site (see there): a staged sibling that skipped it
        # would be the one path on which an erased model still reaches a timing run.
        from merlin.compiler.lower.model.codegen_census import require_commensurate as _census_require_stage

        _stage_census = _census_require_stage(prepared, model_object, str(record["entrypoint"]))
        print(f"[census gate object] stage {name!r} {_stage_census.as_dict()}", flush=True)
        stage_censuses.append((prepared, str(record["entrypoint"]), _stage_census.as_dict()))
        model_object = _namespace_stage_object(
            model_object, stage_work / "model.namespaced.o", index=index, entrypoint=str(record["entrypoint"])
        )
        objects.append(model_object)

        cgen = Path(record["generated"])
        call_object = stage_work / "model_call.o"
        call_command = [
            str(clang23),
            p.target_flag,
            f"-march={p.march}",
            f"-mabi={mabi}",
            opt,
            "-Wno-override-module",
        ]
        if sysroot is not None and sysroot.is_dir():
            call_command += [f"--sysroot={sysroot}", f"--gcc-toolchain={root}"]
        _run([*call_command, f"-I{rt}", f"-I{cgen}", "-c", cgen / "model_call.c", "-o", call_object])
        adapter_object = stage_work / "stage_adapter.o"
        _run(
            [
                cc,
                p.target_flag,
                f"-march={p.march}",
                f"-mabi={mabi}",
                opt,
                f"-I{rt}",
                f"-I{cgen}",
                "-c",
                cgen / "stage_adapter.c",
                "-o",
                adapter_object,
            ]
        )
        objects += [call_object, adapter_object]
        weights.append(
            {
                "program": str(index),
                "name": name,
                "path": str(cgen / "weights.bin"),
                "sha256": str(record["weights_sha256"]),
            }
        )

    marker = work / "USE_MMAP_WEIGHTS.json"
    marker.write_text(json.dumps(weights, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    main_c = work / "main_session.c"
    main_c.write_text(
        main_linux_session_c(p, dump_cap, persistent_openmp_pool=openmp.worker_pool_size), encoding="utf-8"
    )
    binary = work / f"{p.binary}_session"
    base = [
        cc,
        p.target_flag,
        f"-march={p.march}",
        f"-mabi={mabi}",
        opt,
        f"-I{rt}",
        f"-I{generated}",
        main_c,
        generated / "merlin_session.c",
        rt / "merlin_model.c",
        abi / "mlir_runtime.c",
        *objects,
    ]
    if openmp.provider == "merlin_pthread_pool":
        base += [rt / "libomp_pthread.c"]
    if parallel_harts and openmp.provider == "llvm":
        libomp = _openmp_library(p)
        if not libomp.is_file():
            raise RemoteLinuxError(f"parallel session build needs {libomp}")
        base += [str(libomp), "-lstdc++", "-ldl"]
    base += ["-lm", "-lpthread", "-o", binary]
    try:
        _run([*base, "-static"])
    except RemoteLinuxError:
        _run(base)
    if not binary.is_file():
        raise RemoteLinuxError(f"{p.name} session cross-compile produced no binary at {binary}")
    # As in the single-program path, retain the object census as the erasure gate and report each
    # preserved stage entry from the linked artifact that actually runs.
    from merlin.compiler.lower.model.codegen_census import require_commensurate as _census_require_linked

    linked_stages = []
    for prepared, entrypoint, object_census in stage_censuses:
        linked = _census_require_linked(prepared, binary, entrypoint)
        linked_stages.append({"entrypoint": entrypoint, "gate_object": object_census, "linked_elf": linked.as_dict()})
        print(f"[census linked ELF] stage {entrypoint!r} {linked.as_dict()}", flush=True)
    (work / "codegen_census.json").write_text(
        json.dumps({"stages": linked_stages}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (work / "HAS_SESSION_QUALITY").write_text("1", encoding="utf-8")
    return binary


def _namespace_stage_object(source: Path, destination: Path, *, index: int, entrypoint: str) -> Path:
    """Namespace compiler-emitted public helpers while preserving the stage ABI entry points.

    MLIR's LLVM lowering emits helpers such as ``dealloc_helper`` with external linkage. Linking
    several independently lowered modules therefore collides even though their public forwards are
    unique. Rename every other defined external symbol in the ELF object; llvm-objcopy also updates
    its internal relocations, while undefined libc/backend references remain untouched.
    """
    llvm_bin = repo_root() / "third_party" / "llvm-install" / "bin"
    nm, objcopy = llvm_bin / "llvm-nm", llvm_bin / "llvm-objcopy"
    if not nm.is_file() or not objcopy.is_file():
        nm_fallback, objcopy_fallback = Path("/usr/lib/llvm-18/bin/llvm-nm"), Path("/usr/lib/llvm-18/bin/llvm-objcopy")
        nm = nm if nm.is_file() else nm_fallback
        objcopy = objcopy if objcopy.is_file() else objcopy_fallback
    if not nm.is_file() or not objcopy.is_file():
        raise RemoteLinuxError("multi-program linking requires llvm-nm and llvm-objcopy")
    listed = _run([nm, "--defined-only", "--extern-only", "--format=posix", source])
    preserve = {entrypoint, f"_mlir_ciface_{entrypoint}"}
    symbols = []
    for line in listed.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] not in preserve:
            symbols.append(parts[0])
    if not symbols:
        shutil.copy2(source, destination)
        return destination
    command: list[object] = [objcopy]
    for symbol in symbols:
        command.append(f"--redefine-sym={symbol}=merlin_stage_{index}__{symbol}")
    command += [source, destination]
    _run(command)
    if not destination.is_file():
        raise RemoteLinuxError(f"stage symbol namespacing produced no object at {destination}")
    return destination
