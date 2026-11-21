"""Build the existing ordered child-process argv from explicit lowering inputs."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from merlin.compiler.ir import compile_trace
from merlin.compiler.lower.model.toolchain import m2m_python
from merlin.compiler.lower.native.contract import PipelineError
from merlin.compiler.lower.native.lifetime import DEALLOC_CHECK_PRELUDE
from merlin.compiler.lower.native.policy import SINK_DEALLOC_PASS
from merlin.compiler.lower.native.recipe import LoweringRecipe
from merlin.compiler.lower.native.stages import DEALLOC_CHECK_TOKEN
from merlin.compiler.lower.options import fusion_guard
from merlin.compiler.lower.options.masked_contraction import FEATURE as _MASKED_FEATURE
from merlin.compiler.lower.options.optional_passes import child_environment
from merlin.compiler.lower.options.source_observation_stage import FEATURE as _SOURCE_OBSERVATION_FEATURE

# ---- arguments ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunnerCommand:
    """The exact native argv and its completion-report gates, selected together."""

    pipeline: str
    argv: list[str]
    fold_broadcast: str
    named_broadcast: str
    alloca_scope: str
    panel_parallel: str
    coarsen: str
    int_softmax: str


def build_command(
    mlir_text: str,
    pipeline: str,
    feats: frozenset[str],
    *,
    src: Path,
    runner: Path,
    stage_out: Path,
    omp: bool,
    parallel_harts: int | None,
    index_bits: int | None,
    data_layout: str | None,
    lowering_selection: dict | None,
    masked_contraction_effects=None,
) -> RunnerCommand:
    from merlin.compiler.lower.options.selfcopy import FEATURE as _SELFCOPY_FEATURE
    from merlin.compiler.lower.options.selfcopy import with_canonicalize as _with_canon
    from merlin.compiler.lower.options.transpose_fuse import FEATURE as _FUSE_TRANSPOSE_FEATURE

    _erase = "1" if _SELFCOPY_FEATURE in feats else "0"
    if _erase == "1":
        pipeline = _with_canon(pipeline)
    if index_bits is not None and lowering_selection is not None:
        # Self-copy selection may append canonicalize/cse. This is the final
        # pass string actually given to the runner, not the pre-feature plan.
        lowering_selection.update(
            {"index_bits": index_bits, "data_layout": data_layout, "effective_pipeline": pipeline}
        )
    # argv[5] gates fuse_transpose_b (default-off). Every runner variant honors it (the act_poly tail
    # used to drop it along with the self-copy erase); with the feature off the lowering stays
    # byte-identical.
    _fuse_tb = "1" if _FUSE_TRANSPOSE_FEATURE in feats else "0"
    # argv[6] gates expand_memref_copy (default-off): every static `memref.copy` becomes a
    # `linalg.copy`, which the pipeline's own convert-linalg-to-loops turns into an emitted scf
    # load/store nest -- so finalize-memref-to-llvm has no copy left to turn into an @memrefCopy or
    # memcpy call. Runs at the same split point as the erase; see llvmlower/copy_expand.py.
    from merlin.compiler.lower.options.copy_expand import FEATURE as _EXPAND_COPY_FEATURE

    _expand_copy = "1" if _EXPAND_COPY_FEATURE in feats else "0"
    # argv[7] gates fold_weight_transpose (default-off): sink a loop-invariant weight transpose into
    # its consumers' indexing_maps. See llvmlower/transpose_maps.py.
    from merlin.compiler.lower.options.transpose_maps import FEATURE as _FOLD_WEIGHT_TRANSPOSE_FEATURE

    _fold_wt = "1" if _FOLD_WEIGHT_TRANSPOSE_FEATURE in feats else "0"
    # argv[8] gates concat_dps (default-off): a tensor.concat's destination-passing producers write
    # straight into the concatenated buffer. See llvmlower/concat_dps.py.
    from merlin.compiler.lower.options.concat_dps import FEATURE as _CONCAT_DPS_FEATURE

    _concat_dps_gate = "1" if _CONCAT_DPS_FEATURE in feats else "0"
    # argv[9] carries the multicore fork/join GRAIN threshold (default-off: "0"). Every
    # `scf.parallel` cheaper than it becomes a serial `scf.for` nest before convert-scf-to-openmp,
    # so no fork is emitted for it. See llvmlower/parallel_grain.py; 0 -> byte-identical lowering.
    from merlin.compiler.lower.options.parallel_grain import threshold_of as _parallel_grain_threshold

    _grain = _parallel_grain_threshold(feats)
    if _grain is not None and not omp:
        import sys as _sys

        # Say so rather than shipping a build whose named lever cannot fire: without the multicore
        # lowering there is no `scf.parallel` at all, so the grain would report 0 and read as inert.
        print(
            "[parallel_grain] WARNING: a grain threshold is named but this lowering is SERIAL "
            "(no parallel_harts/parallel), so there is no scf.parallel to price; the feature "
            "will serialize nothing.",
            file=_sys.stderr,
            flush=True,
        )
    _grain_gate = str(int(_grain)) if _grain is not None else "0"
    # Packed panels carry their own outer worksharing boundary.  It is enabled only on a multicore
    # lowering that actually requested either packer; the runner then proves bufferization removed
    # the tensor carrier before changing the marked loop to ``scf.parallel``.
    # A pack feature can legitimately produce only single-panel carriers. Their loops have one
    # trip and ``marker_ops`` intentionally emits no marker: requiring the parallel rewrite in
    # that case made a correctly packed serial carrier fail with ``regions=0``. Gate on an actual
    # marker CALL in the prepared module (not the declaration kept for uniformity).
    from merlin.compiler.backend import providers as provided
    from merlin.compiler.lower.options.im2col import FEATURE as _IM2COL_PANEL_FEATURE
    from merlin.compiler.lower.options.panel_parallel import has_marker_call_text as _has_panel_marker_call

    # The weight-panel packer belongs to the selected vector support provider; with none selected
    # its feature cannot be in `feats` (normalization would have refused it).
    _weight_panel = provided.optional_component("weight_panel")
    _panel_features = {_IM2COL_PANEL_FEATURE} | ({_weight_panel.FEATURE} if _weight_panel is not None else set())
    _panel_parallel_gate = (
        "1" if (omp and (_panel_features & feats) and _has_panel_marker_call(src.read_text(encoding="utf-8"))) else "0"
    )
    # argv[11:13] carry the ModelBlaster-style per-dispatch team policy.  The cost unit comes from
    # the feature name; the cap is the target-independent worker count already passed into this
    # lowering.  Naming it on a serial build is an error because there is no OpenMP region to edit.
    from merlin.compiler.lower.options.parallel_team import work_of as _parallel_team_work

    _team_work = _parallel_team_work(feats)
    if _team_work is not None and parallel_harts is None:
        raise ValueError("parallel_team_cost requires an explicit parallel_harts worker cap")
    _team_work_gate = str(int(_team_work)) if _team_work is not None else "0"
    _team_cap_gate = str(int(parallel_harts or 0)) if _team_work is not None else "0"
    from merlin.compiler.lower.options.parallel_coarsen import FEATURE as _PARALLEL_COARSEN_FEATURE

    _coarsen_gate = "1" if _PARALLEL_COARSEN_FEATURE in feats else "0"
    if _coarsen_gate == "1" and not omp:
        raise ValueError("coarsen_openmp_regions requires a parallel lowering")
    if _coarsen_gate == "1" and _team_work is not None:
        raise ValueError(
            "coarsen_openmp_regions and parallel_team_cost are incompatible: "
            "team widths must agree across a coarsened region"
        )
    # argv[14] gates the pre-pipeline affine-map fold. Appending the argument preserves every
    # existing runner gate's stable index.
    from merlin.compiler.lower.options.broadcast_fold import FEATURE as _FOLD_BROADCAST_FEATURE

    _fold_broadcast_gate = "1" if _FOLD_BROADCAST_FEATURE in feats else "0"
    # argv[15] selects the after-schedule/pre-generalize named add/mul fold.
    from merlin.compiler.lower.options.named_broadcast_fold import FEATURE as _NAMED_BROADCAST_FOLD_FEATURE

    _named_broadcast_gate = "1" if _NAMED_BROADCAST_FOLD_FEATURE in feats else "0"
    # Requant fusion can create tile-local accumulator scopes inside a parallel region. Lower their
    # structured lifetime before SCF-to-CFG; after that point a scope contains multiple blocks and
    # is verifier-invalid. The pair tags are the durable witness because the user-facing sentinel
    # has already been consumed by per-op schedule derivation.
    _alloca_scope_gate = "1" if omp and "merlin.rqfuse" in mlir_text else "0"
    # argv[20] gates the integer-softmax restructuring, which runs on the module as parsed, before every
    # other pre-pipeline rewrite. Appended after the data layout so no existing slot moves.
    from merlin.compiler.lower.options.nonlinear import ARGV_INDEX as _INT_SOFTMAX_ARGV
    from merlin.compiler.lower.options.nonlinear import FEATURE as _INT_SOFTMAX_FEATURE

    _int_softmax_gate = "1" if _INT_SOFTMAX_FEATURE in feats else "0"
    # OpenMP transport: the runner DUMPS the LLVM-dialect module and the standalone
    # mlir-translate produces the .ll out-of-process (the in-process torch-mlir bridge
    # segfaults on omp IR). Otherwise the runner writes the .ll directly.
    command = [
        str(m2m_python()),
        str(runner),
        str(src),
        str(stage_out),
        pipeline,
        _erase,
        _fuse_tb,
        _expand_copy,
        _fold_wt,
        _concat_dps_gate,
        _grain_gate,
        _panel_parallel_gate,
        _team_work_gate,
        _team_cap_gate,
        _coarsen_gate,
        _fold_broadcast_gate,
        _named_broadcast_gate,
        _alloca_scope_gate,
        data_layout or "",
        "1" if "fold_uniform_fill_copy" in feats else "0",
        "1" if "specialize_contiguous_copy" in feats else "0",
        _int_softmax_gate,
    ]
    # The runner reads the gate at sys.argv[ARGV_INDEX] (command[0] is the interpreter).
    if len(command) - 2 != _INT_SOFTMAX_ARGV:
        raise PipelineError("the lowering runner's argv layout no longer matches int_softmax_table.ARGV_INDEX")
    if masked_contraction_effects is not None:
        from merlin.compiler.lower.options.masked_contraction import ARGV_INDEX as _MASKED_ARGV

        if len(command) - 1 != _MASKED_ARGV:
            raise PipelineError("the lowering runner's argv layout no longer matches masked_contraction.ARGV_INDEX")
        command.append("1")
    return RunnerCommand(
        pipeline,
        command,
        _fold_broadcast_gate,
        _named_broadcast_gate,
        _alloca_scope_gate,
        _panel_parallel_gate,
        _coarsen_gate,
        _int_softmax_gate,
    )


# ---- invoke ------------------------------------------------------------------------------------
# Native diagnostic pass invocation and completed trace harvesting.


def apply_passes(mlir_text: str, pipeline: str, timeout: int = 600) -> str:
    """Run ``pipeline`` over ``mlir_text`` in the m2m venv and return the printed module.

    A diagnostic/test seam, not a build path: it drives the SAME pass registry the lowering drives
    (the torch-mlir wheel's), so a test that pins what a stage does to the IR cannot pass against a
    different LLVM than the one that ships.
    """
    work = Path(tempfile.mkdtemp(prefix="merlin_apply_passes_"))
    src, dst = work / "in.mlir", work / "out.mlir"
    src.write_text(mlir_text, encoding="utf-8")
    script = work / "apply.py"
    script.write_text(
        fusion_guard.inject("import sys\nfrom torch_mlir import ir\nfrom torch_mlir.passmanager import PassManager\n")
        + "ctx = ir.Context()\n"
        "with open(sys.argv[1]) as f:\n"
        "    module = ir.Module.parse(f.read(), ctx)\n"
        "PassManager.parse('builtin.module(' + sys.argv[3] + ')', ctx).run(module.operation)\n"
        "with open(sys.argv[2], 'w') as f:\n"
        "    f.write(str(module.operation))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [str(m2m_python()), str(script), str(src), str(dst), pipeline],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=child_environment(),
    )
    if proc.returncode != 0 or not dst.is_file():
        raise PipelineError(f"apply_passes failed:\n{proc.stdout}\n{proc.stderr}")
    return dst.read_text(encoding="utf-8")


def dealloc_placement_violations(mlir_text: str, timeout: int = 600, *, borrowed_calls=()) -> list[str]:
    """Violations of the placement rule in a post-deallocation memref module.

    Empty list == every allocation whose free was sunk into the alloc's own block is used only
    before that free. Runs the SAME source the build runs, out of process in the m2m venv (the MLIR
    Python bindings live there), so a test and a build cannot disagree about what is checked.
    Closed positive serial replacement loops additionally require a complete
    intrinsic generation/release proof; unknown calls/effects remain barriers.
    """
    import json

    from merlin.compiler.lower.host.borrowed_buffer_calls import bind_native

    prelude = bind_native(DEALLOC_CHECK_PRELUDE, borrowed_calls)

    work = Path(tempfile.mkdtemp(prefix="merlin_dealloc_check_"))
    src = work / "module.mlir"
    src.write_text(mlir_text, encoding="utf-8")
    script = work / "check.py"
    script.write_text(
        "import json, sys\n"
        "from torch_mlir import ir\n" + prelude + "ctx = ir.Context()\n"
        "with open(sys.argv[1]) as f:\n"
        "    module = ir.Module.parse(f.read(), ctx)\n"
        "module.operation.verify()\n"
        "bad, n_alloc, n_sunk = _dealloc_placement_violations(module)\n"
        "print('MERLIN_CHECK ' + json.dumps({'violations': bad, 'allocations': n_alloc, "
        "'sunk': n_sunk}))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [str(m2m_python()), str(script), str(src)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=child_environment(),
    )
    if proc.returncode != 0:
        raise PipelineError(f"dealloc placement check failed:\n{proc.stdout}\n{proc.stderr}")
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("MERLIN_CHECK "))
    return json.loads(line[len("MERLIN_CHECK ") :])["violations"]


def _harvest_native(traced: "tuple[Path, bool] | None", produced: Path, failure: BaseException | None = None) -> None:
    """Index a native pass run's dumps into the open compile trace (``(directory, prune)``; None: no
    trace selects a native pass). A stop selected on a pass that ran before ``failure`` still stops.

    The pass manager ran its whole pipeline in the child before the trace could stop it, so on a stop
    its output (``produced``, the IR after the LAST pass) is removed: nothing past the stop survives."""
    if traced is None:
        return
    try:
        compile_trace.harvest_native(traced[0], pipeline="llvm", prune=traced[1])
    except compile_trace.StopAfterStage as stop:
        produced.unlink(missing_ok=True)
        stop.note = "the pass manager ran on in its child process; the output of the passes after it was discarded"
        if failure is None:
            raise
        stop.note += f"; a later pass failed: {str(failure).strip().splitlines()[0][:200]}"
        raise stop from failure


# ---- reports -----------------------------------------------------------------------------------
# Check completed native reports and bind original artifacts to the recipe.


def check_reports(
    proc: subprocess.CompletedProcess[str],
    *,
    work: Path,
    mlir_text: str,
    feats: frozenset[str],
    arguments: RunnerCommand,
    recipe: LoweringRecipe,
    lowering_selection: dict | None,
    audit,
    retention,
    stage_out: Path,
    omp: bool,
    traced,
    runner: Path,
    borrowed_buffer_calls=None,
    scalar_stage: Path | None = None,
    source_observation_effects=None,
):
    checked = None
    from merlin.compiler.lower.options.automatic_host_features import FEATURE as automatic_host_feature

    if automatic_host_feature in feats:
        from merlin.common.digest import write_pretty_json
        from merlin.compiler.lower.options.automatic_host_features import require_report

        automatic_report = require_report(proc.stdout, features=feats)
        report_path = work / "automatic-host-optimizations.json"
        write_pretty_json(report_path, automatic_report)
        recipe.bind_source("automatic_host_optimizations", report_path)
        if lowering_selection is not None:
            lowering_selection["automatic_host_optimizations"] = automatic_report
    if audit is not None:
        audit.collect_views()
    _harvest_native(traced, stage_out)
    if borrowed_buffer_calls is not None:
        from merlin.compiler.lower.host.borrowed_buffer_calls import verify_native

        checked = verify_native(borrowed_buffer_calls, mlir_text, work, native_stdout=proc.stdout, native_runner=runner)
        from merlin.compiler.lower.host.borrowed_buffer_calls import CHECKPOINT, REPORT

        recipe.bind_source("borrowed_lifetime_checkpoint", work / CHECKPOINT)
        recipe.bind_source("borrowed_lifetime_report", work / REPORT)
        if lowering_selection is not None:
            lowering_selection["borrowed_buffer_lifetime"] = checked
    if scalar_stage is not None:
        import hashlib as _hashlib
        import json as _json

        source_digest = _hashlib.sha256((scalar_stage / "current.mlir").read_bytes()).hexdigest()
        response_digest = _hashlib.sha256((scalar_stage / "response.json").read_bytes()).hexdigest()
        expected = f"OK current_scalar_leaf_stage {source_digest} {response_digest}"
        if [line for line in proc.stdout.splitlines() if line.startswith("OK current_scalar_leaf_stage")] != [expected]:
            raise PipelineError("native scalar rewrite completion is absent or inconsistent")
        for name in ("current.mlir", "request.json", "response.json", "binding.json"):
            recipe.bind_source("scalar_carrier_" + name.replace(".", "_"), scalar_stage / name)
        binding = _json.loads((scalar_stage / "binding.json").read_text())
        binding.update(status="CURRENT_TYPED_EDITS_NATIVE_VERIFIED", native_response_sha256=response_digest)
        if lowering_selection is not None:
            lowering_selection["source_scalar_carrier"] = binding
    if _SOURCE_OBSERVATION_FEATURE in feats:
        from merlin.compiler.lower.options.source_observation_stage import CHECKPOINT
        from merlin.compiler.lower.options.source_observation_stage import (
            require_report as _require_source_observation_report,
        )

        observed_source = _require_source_observation_report(proc.stdout, work, effects=source_observation_effects)
        recipe.bind_source("source_observation", work / CHECKPOINT)
        if lowering_selection is not None:
            lowering_selection["source_observation"] = observed_source
    if audit is not None:
        if retention is not None:
            audit.stage("llvm-dialect", retention.module.read_text(encoding="utf-8"))
        audit.stage(
            "llvm-dialect" if omp else "llvm-translated",
            stage_out.read_text(encoding="utf-8"),
            format="mlir" if omp else "llvm-ir",
        )
    if _MASKED_FEATURE in feats:
        from merlin.compiler.lower.options.masked_contraction import require_report as _require_masked_report

        try:
            _require_masked_report(proc.stdout, work)
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if "specialize_contiguous_copy" in feats:
        from merlin.compiler.lower.options.contiguous_suffix_copy import (
            require_report as _require_contiguous_copy_report,
        )

        try:
            _require_contiguous_copy_report(proc.stdout, work)
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if "fold_uniform_fill_copy" in feats:
        from merlin.compiler.lower.options.uniform_fill_copy import (
            require_report as _require_uniform_fill_report,
        )

        try:
            _require_uniform_fill_report(proc.stdout, work)
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if arguments.fold_broadcast == "1":
        from merlin.compiler.lower.options.broadcast_fold import (
            require_report as _require_broadcast_fold_report,
        )

        try:
            _folded = _require_broadcast_fold_report(proc.stdout, work)
            print(f"[broadcast-fold] folded {_folded} materialized broadcasts")
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if arguments.named_broadcast == "1":
        from merlin.compiler.lower.options.named_broadcast_fold import (
            require_report as _require_named_broadcast_report,
        )

        try:
            _named = _require_named_broadcast_report(proc.stdout, work)
            print(f"[named-broadcast-fold] {_named}")
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if arguments.alloca_scope == "1":
        from merlin.compiler.lower.options.alloca_scope_lower import (
            require_report as _require_alloca_scope_report,
        )

        try:
            _lowered_scopes = _require_alloca_scope_report(proc.stdout, work)
            print(f"[alloca-scope] lowered {_lowered_scopes} structured scope(s) before CFG")
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if arguments.panel_parallel == "1":
        from merlin.compiler.lower.options.panel_parallel import (
            require_complete_report as _require_panel_report,
        )

        try:
            _require_panel_report(proc.stdout, work)
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if arguments.coarsen == "1":
        from merlin.compiler.lower.options.parallel_coarsen import require_report as _require_coarsen_report

        try:
            _require_coarsen_report(proc.stdout)
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if arguments.int_softmax == "1":
        from merlin.compiler.lower.options.nonlinear import require_report as _require_int_softmax_report

        try:
            _ist = _require_int_softmax_report(proc.stdout, work)
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
        print(
            f"[int-softmax-table] {_ist['softmax']} softmax(es), {_ist['int32_sums']} int32 row sum(s), "
            f"{_ist['row_quantizations']} per-row quantization(s), {_ist['scales_moved']} scale(s) moved"
            + (f"; left alone: {'; '.join(_ist['refused'])}" if _ist["refused"] else "")
        )
    if (
        fusion_guard.FUSE_PASS in arguments.pipeline
        and fusion_guard.enabled()
        and fusion_guard.TOKEN not in proc.stdout
    ):
        # The pass ran and the broadcast control function did not: a runner that reached the native
        # PassManager without the guard would silently re-evaluate per-row values per element.
        raise PipelineError(
            f"elementwise fusion ran without its broadcast guard ({fusion_guard.TOKEN!r} absent from "
            f"the runner's output)\n{proc.stdout}"
        )
    if SINK_DEALLOC_PASS in arguments.pipeline and DEALLOC_CHECK_TOKEN not in proc.stdout:
        # The sinking stage ran and the use-after-free check did NOT. That is only reachable from a
        # runner variant that drives the PassManager itself instead of going through `_run_stages`
        # (the accumulator-resident v3 runner does, for its first stage), and it would sink every
        # free with nothing verifying the placement -- silently wrong numbers, not a failed build.
        # Refuse the artifact rather than ship an unchecked one.
        raise PipelineError(
            "the dealloc sinking stage ran but its use-after-free check did not report "
            f"({DEALLOC_CHECK_TOKEN!r} absent from the runner's output). This runner variant "
            "bypasses `_run_stages`; do not combine MERLIN_SINK_DEALLOCS with it until the check "
            f"is wired there too.\n{proc.stdout}"
        )
    return checked
