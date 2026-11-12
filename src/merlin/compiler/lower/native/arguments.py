"""Build the existing ordered child-process argv from explicit lowering inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.compiler.passes.contract import PipelineError
from merlin.llvmlower.toolchain import m2m_python


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
    from merlin.llvmlower.selfcopy import FEATURE as _SELFCOPY_FEATURE
    from merlin.llvmlower.selfcopy import with_canonicalize as _with_canon
    from merlin.llvmlower.transpose_fuse import FEATURE as _FUSE_TRANSPOSE_FEATURE

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
    from merlin.llvmlower.copy_expand import FEATURE as _EXPAND_COPY_FEATURE

    _expand_copy = "1" if _EXPAND_COPY_FEATURE in feats else "0"
    # argv[7] gates fold_weight_transpose (default-off): sink a loop-invariant weight transpose into
    # its consumers' indexing_maps. See llvmlower/transpose_maps.py.
    from merlin.llvmlower.transpose_maps import FEATURE as _FOLD_WEIGHT_TRANSPOSE_FEATURE

    _fold_wt = "1" if _FOLD_WEIGHT_TRANSPOSE_FEATURE in feats else "0"
    # argv[8] gates concat_dps (default-off): a tensor.concat's destination-passing producers write
    # straight into the concatenated buffer. See llvmlower/concat_dps.py.
    from merlin.llvmlower.concat_dps import FEATURE as _CONCAT_DPS_FEATURE

    _concat_dps_gate = "1" if _CONCAT_DPS_FEATURE in feats else "0"
    # argv[9] carries the multicore fork/join GRAIN threshold (default-off: "0"). Every
    # `scf.parallel` cheaper than it becomes a serial `scf.for` nest before convert-scf-to-openmp,
    # so no fork is emitted for it. See llvmlower/parallel_grain.py; 0 -> byte-identical lowering.
    from merlin.llvmlower.parallel_grain import threshold_of as _parallel_grain_threshold

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
    from merlin.compiler.backend import provided
    from merlin.llvmlower.im2col_pack import FEATURE as _IM2COL_PANEL_FEATURE
    from merlin.llvmlower.panel_parallel import has_marker_call_text as _has_panel_marker_call

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
    from merlin.llvmlower.parallel_team import work_of as _parallel_team_work

    _team_work = _parallel_team_work(feats)
    if _team_work is not None and parallel_harts is None:
        raise ValueError("parallel_team_cost requires an explicit parallel_harts worker cap")
    _team_work_gate = str(int(_team_work)) if _team_work is not None else "0"
    _team_cap_gate = str(int(parallel_harts or 0)) if _team_work is not None else "0"
    from merlin.llvmlower.parallel_coarsen import FEATURE as _PARALLEL_COARSEN_FEATURE

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
    from merlin.llvmlower.broadcast_fold import FEATURE as _FOLD_BROADCAST_FEATURE

    _fold_broadcast_gate = "1" if _FOLD_BROADCAST_FEATURE in feats else "0"
    # argv[15] selects the after-schedule/pre-generalize named add/mul fold.
    from merlin.llvmlower.named_broadcast_fold import FEATURE as _NAMED_BROADCAST_FOLD_FEATURE

    _named_broadcast_gate = "1" if _NAMED_BROADCAST_FOLD_FEATURE in feats else "0"
    # Requant fusion can create tile-local accumulator scopes inside a parallel region. Lower their
    # structured lifetime before SCF-to-CFG; after that point a scope contains multiple blocks and
    # is verifier-invalid. The pair tags are the durable witness because the user-facing sentinel
    # has already been consumed by per-op schedule derivation.
    _alloca_scope_gate = "1" if omp and "merlin.rqfuse" in mlir_text else "0"
    # argv[20] gates the integer-softmax restructuring, which runs on the module as parsed, before every
    # other pre-pipeline rewrite. Appended after the data layout so no existing slot moves.
    from merlin.llvmlower.int_softmax_table import ARGV_INDEX as _INT_SOFTMAX_ARGV
    from merlin.llvmlower.int_softmax_table import FEATURE as _INT_SOFTMAX_FEATURE

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
        from merlin.llvmlower.masked_contraction import ARGV_INDEX as _MASKED_ARGV

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
