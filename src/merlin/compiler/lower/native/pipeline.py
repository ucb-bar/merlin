"""Native pass-driver orchestration and explicit source/effect admission.

Reusable final LLVM passes remain below this driver in compiler.lower.llvm."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from merlin.common import compile_trace
from merlin.common.paths import module_source_path
from merlin.compiler.lower.llvm.normalize import normalize_float_literals as _fix_float_literals
from merlin.compiler.lower.llvm.normalize import normalize_hex_literals as _fix_f0x_literals
from merlin.compiler.lower.llvm.passes import INDEX_CONVERSION_PASSES
from merlin.compiler.lower.llvm.passes import bind_index_width as _bind_index_width
from merlin.compiler.lower.native.arguments import build_command
from merlin.compiler.lower.native.invoke import _harvest_native
from merlin.compiler.lower.native.invoke import apply_passes as apply_passes
from merlin.compiler.lower.native.invoke import dealloc_placement_violations as dealloc_placement_violations
from merlin.compiler.lower.native.policy import SINK_DEALLOC_PASS as SINK_DEALLOC_PASS
from merlin.compiler.lower.native.policy import (
    SLICE_UNUSED_ELEMENTWISE_OUTPUT_MARKER as SLICE_UNUSED_ELEMENTWISE_OUTPUT_MARKER,
)
from merlin.compiler.lower.native.policy import SPLIT_FRESH_DEALLOCS_MARKER as SPLIT_FRESH_DEALLOCS_MARKER
from merlin.compiler.lower.native.policy import _parallel_pipeline, _upstream_pipeline
from merlin.compiler.lower.native.reports import check_reports
from merlin.compiler.lower.native.runners import EMIT_DUMP, EMIT_TRANSLATE, _select_runner
from merlin.compiler.lower.native.transport import _residual_vector_dialect_ops as _residual_vector_dialect_ops
from merlin.compiler.lower.native.transport import run_translation, select_retention
from merlin.compiler.lower.native.vector import RVV_TRANSFORM_SCHEDULE as RVV_TRANSFORM_SCHEDULE
from merlin.compiler.lower.native.vector import build_rvv_pipeline as build_rvv_pipeline
from merlin.compiler.lower.native.vector import vec_pre_schedule
from merlin.compiler.passes import native as native_passes
from merlin.compiler.passes import parallel as parallel_schedule
from merlin.compiler.passes.contract import PipelineError as PipelineError
from merlin.llvmlower.optional_passes import child_environment
from merlin.llvmlower.source_scalar_carrier_binding import host_admitted
from merlin.llvmlower.toolchain import m2m_python

_INDEX_CONVERSION_PASSES = INDEX_CONVERSION_PASSES
PARALLEL_ENTRY = parallel_schedule.PARALLEL_ENTRY
_PARALLEL_DIM_NUM_THREADS = parallel_schedule._PARALLEL_DIM_NUM_THREADS
_perop_parallel_schedule = parallel_schedule._perop_parallel_schedule
parallel_transform_schedule = parallel_schedule.parallel_transform_schedule


def implementation_paths() -> tuple[Path, ...]:
    """Source custody for this driver's shared policy and implementation owners.

    Selected transforms and generated runners add their own sources at invocation.
    This is the common lowering closure, not a complete toolchain lock.
    """
    from importlib import import_module

    owners = (
        "merlin.compiler.lower.native.policy",
        "merlin.compiler.lower.native.vector",
        "merlin.compiler.lower.native.stages",
        "merlin.compiler.lower.native.runners",
        "merlin.compiler.lower.native.invoke",
        "merlin.compiler.lower.native.features",
        "merlin.compiler.lower.native.arguments",
        "merlin.compiler.lower.native.reports",
        "merlin.compiler.lower.native.transport",
        "merlin.compiler.lower.native.recipe",
        "merlin.compiler.lower.native.stock",
        "merlin.compiler.lower.scalar.pipeline",
        "merlin.compiler.lower.llvm.passes",
        "merlin.compiler.lower.llvm.normalize",
        "merlin.compiler.passes.contract",
        "merlin.compiler.passes.parallel",
        "merlin.compiler.passes.native",
        "merlin.compiler.passes.policy",
        "merlin.compiler.passes.fragments",
        "merlin.compiler.passes.features",
        "merlin.compiler.selection.features",
        "merlin.compiler.selection.shared",
        "merlin.llvmlower.optional_passes",
        "merlin.llvmlower.automatic_host_features",
        "merlin.compiler.transforms.host",
        "merlin.compiler.transforms.host_contraction",
        "merlin.compiler.transforms.host_reduction",
        "merlin.compiler.transforms.host_broadcast",
        "merlin.compiler.transforms.host_integer_math",
    )
    return (Path(__file__), *(Path(import_module(name).__file__) for name in owners))


@host_admitted
def lower_to_llvm_ir(
    mlir_text: str,
    workdir: str | Path | None = None,
    pipeline: str | None = None,
    timeout: int = 7200,
    vectorize: bool = False,
    transform_schedule: str | None = None,
    hoist_static_allocs: bool = True,
    parallel: bool = False,
    features: "frozenset[str] | None" = None,
    parallel_harts: int | None = None,
    parallel_chunks: "list | None" = None,
    audit=None,
    data_layout: str | None = None,
    index_bits: int | None = None,
    lowering_selection: dict | None = None,
    masked_contraction_effects=None,
    source_observation_effects=None,
    source_scalar_carrier=None,
    retain_llvm_dialect: bool = False,
    borrowed_buffer_calls=None,
    borrowed_arena_lifetimes=None,
) -> str:
    """Lower upstream-MLIR text to LLVM IR text via the m2m venv. Returns .ll text.

    ``data_layout`` sets the selected module layout before translation; None
    retains LLVM's default. Explicit ``index_bits`` binds index conversion.
    ``retain_llvm_dialect=True`` prints the actual serial post-pass module before
    the same in-process translation, retaining custody without correctness proof.
    ``borrowed_arena_lifetimes`` optionally collects the live native-checked
    original effects bound to the exact returned LLVM, for the ordinary allocator.
    ``masked_contraction_effects`` separately requires the closed-mask feature;
    dead-tile omission needs nontrapping/unobserved-flags permission and retains
    observed arithmetic order/types. Default lowering is unchanged.
    ``vectorize`` selects fixed-width vector lowering with a workdir schedule;
    ``transform_schedule`` overrides its schedule and ``pipeline`` overrides
    either default pipeline. ``parallel_harts`` additionally layers OpenMP
    forall chunks under vector scheduling; ``parallel`` selects scalar OpenMP.
    See :func:`merlin.llvmlower.lower.lower_model` for the end-to-end options.
    """
    if type(retain_llvm_dialect) is not bool:
        raise ValueError("retain_llvm_dialect requires an explicit bool")
    if lowering_selection is not None:
        lowering_selection.pop("llvm_dialect_product", None)
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="merlin_lower_"))
    work.mkdir(parents=True, exist_ok=True)
    if retain_llvm_dialect:
        work = work.resolve()
    from merlin.compiler.lower.native.recipe import FILENAME as _RECIPE_FILENAME
    from merlin.compiler.lower.native.recipe import LoweringRecipe

    # A refused invocation must not leave the prior build's successful receipt
    # looking current, including failures before feature/pipeline resolution.
    (work / _RECIPE_FILENAME).unlink(missing_ok=True)
    if borrowed_buffer_calls is not None:
        from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferCalls

        if type(borrowed_buffer_calls) is not BorrowedBufferCalls:
            raise ValueError("explicit original borrowed-buffer selection required")
        borrowed_buffer_calls.revalidate(mlir_text)
    from merlin.llvmlower.source_observation_stage import CHECKPOINT as _SOURCE_CHECKPOINT
    from merlin.llvmlower.source_observation_stage import REPORT as _SOURCE_REPORT

    (work / _SOURCE_CHECKPOINT).unlink(missing_ok=True)
    (work / _SOURCE_REPORT).unlink(missing_ok=True)
    recipe_sources = {
        "pipeline_driver": Path(__file__),
        "native_stage_invocation": Path(native_passes.__file__),
        "loop_ownership_checker": module_source_path("merlin.llvmlower.dealloc_lifetime"),
        "translation_transport": module_source_path("merlin.compiler.lower.native.transport"),
    }
    recipe_sources.update(
        {str(path.relative_to(module_source_path("merlin").parent)): path for path in implementation_paths()}
    )
    from merlin.compiler.lower.native.features import select_features
    from merlin.llvmlower.impr_features import apply_schedule

    feats = select_features(
        features,
        masked_contraction_effects=masked_contraction_effects,
        source_observation_effects=source_observation_effects,
    )
    from merlin.llvmlower.masked_contraction import FEATURE as _MASKED_FEATURE
    from merlin.llvmlower.source_observation_stage import FEATURE as _SOURCE_OBSERVATION_FEATURE
    from merlin.llvmlower.toolchain import _env

    driver = _env("MERLIN_COMPILER_DRIVER", "python")
    if driver not in {"python", "stock-cli"}:
        raise PipelineError(f"unknown upstream compiler driver: {driver!r}")
    if driver == "stock-cli":
        from merlin.compiler.lower.native.stock import lower as lower_stock_cli

        return lower_stock_cli(
            mlir_text,
            work=work,
            pipeline=pipeline,
            features=feats,
            timeout=timeout,
            audit=audit,
            data_layout=data_layout,
            index_bits=index_bits,
            lowering_selection=lowering_selection,
            implementation_sources=recipe_sources,
            unsupported={
                "vectorize": vectorize,
                "transform_schedule": transform_schedule is not None,
                "parallel": parallel,
                "parallel_harts": parallel_harts is not None,
                "parallel_chunks": parallel_chunks is not None,
                "hoist_static_allocs=False": not hoist_static_allocs,
                "source_scalar_carrier": source_scalar_carrier is not None,
                "retain_llvm_dialect": retain_llvm_dialect,
                "borrowed_buffer_calls": borrowed_buffer_calls is not None,
            },
        )
    from merlin.llvmlower.llvm_loop_outline import FEATURE as _OUTLINE_LOOPS
    from merlin.llvmlower.llvm_loop_outline import MERGE_FEATURE as _OUTLINE_MERGE_LOOPS
    from merlin.llvmlower.llvm_loop_outline import outline_loops

    if _OUTLINE_LOOPS in feats and _OUTLINE_MERGE_LOOPS in feats:
        raise PipelineError("Select one explicit LLVM loop outlining policy")
    if {"lower_roundeven_to_intrinsic", "fuse_quantize_round_convert"} <= feats:
        raise PipelineError(
            "lower_roundeven_to_intrinsic and fuse_quantize_round_convert are alternative exact "
            "lowerings of the same math.roundeven operation; select one"
        )
    if parallel_harts is not None and not vectorize:
        raise PipelineError(
            "parallel_harts requires vectorize=True (it layers an outer forall UNDER the "
            "RVV transform schedule); for the scalar OpenMP path use parallel=True"
        )
    # OpenMP IR — whatever produced it — must take the out-of-process translate route below.
    omp = parallel or parallel_harts is not None
    if pipeline is None:
        if vectorize:
            sched = work / "rvv_schedule.mlir"
            sched_text = apply_schedule(transform_schedule or RVV_TRANSFORM_SCHEDULE, feats)
            sched.write_text(sched_text, encoding="utf-8")
            recipe_sources["transform_schedule"] = sched
            par_sched = None
            _tile_aligned_parallel = False
            if parallel_harts is not None:
                par_sched = work / "rvv_parallel_schedule.mlir"
                from merlin.llvmlower.bmm_tail_pad import FEATURE as _BMM_TAIL_PAD_FEATURE

                _tile_aligned_parallel = _BMM_TAIL_PAD_FEATURE in feats
                par_sched.write_text(
                    parallel_transform_schedule(
                        parallel_harts, chunks=parallel_chunks, tile_aligned=_tile_aligned_parallel
                    ),
                    encoding="utf-8",
                )
                recipe_sources["parallel_transform_schedule"] = par_sched
                if parallel_chunks is not None and not parallel_chunks:
                    import sys as _sys

                    # A multicore build whose split set is EMPTY is a single-core build wearing a
                    # multicore build's flags. It happens honestly (no contraction has a parallel dim
                    # that divides into equal, block-legal pieces), and it must never be inferred from
                    # a disappointing wall time.
                    print(
                        "[parallel_harts] WARNING: no contraction could be split without changing "
                        "its register block, so this image carries NO scf.forall and will run on "
                        "one core. See perop_blocks.parallel_chunk_table.",
                        file=_sys.stderr,
                        flush=True,
                    )
            # The non-contraction arms, as their own preloaded library, so they run while the
            # `merlin.vec_r{rank}` tags they match on are still on the ops. None (and no file) when
            # the lever is off -- see `vec_pre_schedule`.
            vec_sched = None
            vec_text = vec_pre_schedule(feats)
            if vec_text is not None:
                vec_sched = work / "rvv_vec_pre_schedule.mlir"
                vec_sched.write_text(vec_text, encoding="utf-8")
                recipe_sources["vector_pre_schedule"] = vec_sched
            pipeline = build_rvv_pipeline(
                sched,
                hoist_static_allocs=hoist_static_allocs,
                features=feats,
                par_sched_path=par_sched,
                vec_sched_path=vec_sched,
                perop_parallel=(parallel_chunks is not None or _tile_aligned_parallel),
            )
        elif parallel:
            pipeline = _parallel_pipeline(feats)  # multicore (OpenMP) scalar path — K1 big models
        else:
            pipeline = _upstream_pipeline(feats)
    if index_bits is not None:
        if not data_layout:
            raise ValueError("explicit index width requires selected compiler data layout")
        pipeline = _bind_index_width(pipeline, index_bits)
    if _SOURCE_OBSERVATION_FEATURE in feats:
        from merlin.llvmlower.source_observation_stage import validate_pipeline as _validate_observation_pipeline

        _validate_observation_pipeline(pipeline)
    scalar_stage = None
    if source_scalar_carrier is not None:
        from merlin.llvmlower.source_stage_transport import insert_marker

        if source_observation_effects is not None and source_observation_effects != source_scalar_carrier.effects:
            raise ValueError("source observation and scalar rewrite require the same explicit effects")
        pipeline = insert_marker(pipeline)
        scalar_stage = work / "source_scalar_carrier"
        scalar_stage.mkdir(exist_ok=False)
        recipe_sources.update(
            scalar_carrier_binding=module_source_path("merlin.llvmlower.source_scalar_carrier_binding"),
            scalar_stage_transport=module_source_path("merlin.llvmlower.source_stage_transport"),
        )
    src = work / "model.mlir"
    out = work / "model.ll"
    runner = work / "run_lowering.py"
    retention = select_retention(work, retain_llvm_dialect, omp=omp, scalar_stage=scalar_stage)
    src.write_text(mlir_text, encoding="utf-8")

    # The vectorized_transcendental_activation feature splices a math.exp/erf/tanh -> arith
    # polynomial rewriter into the runner (run before the pass manager). The accumulator-resident
    # micro-kernel v3 feature splices a two-stage runner that runs the A-operand scalarization rewrite
    # BETWEEN two pass-manager stages (split at the SCALARIZE_MARKER pass name). Default-off; with the
    # feature absent the plain _RUNNER is used and the lowering is byte-identical to the baseline.
    # The SAME selection is used on the OpenMP transport (only `emit` differs), so a multicore build
    # of a v3/act_poly package still gets its rewrites.
    inspection_dir = (
        str(audit.directory)
        if audit is not None and audit.directory is not None and audit.mode in {"compact", "both"}
        else None
    )
    # An open compile trace that selects a native pass gets the same printer: into the audit's directory
    # when an audit already binds it (indexed, never pruned), else into the trace's own.
    native = compile_trace.native_directory() if inspection_dir is None else None
    traced = (
        (native[0], True)
        if native is not None
        else ((Path(inspection_dir), False) if inspection_dir and compile_trace.active() else None)
    )
    runner_src = _select_runner(
        pipeline,
        feats,
        emit=retention.emitter(EMIT_TRANSLATE) if retention is not None else (EMIT_DUMP if omp else EMIT_TRANSLATE),
        inspection_dir=inspection_dir or (str(native[0]) if native is not None else None),
        keep_exact=audit is not None and audit.mode == "both",
        printing=(native[1], native[2]) if native is not None else (True, True),
    )
    if scalar_stage is not None:
        from merlin.llvmlower.source_stage_transport import bind_runner as _bind_scalar_runner

        runner_src = _bind_scalar_runner(
            runner_src,
            directory=scalar_stage,
            max_source_bytes=source_scalar_carrier.max_source_bytes,
            max_response_bytes=source_scalar_carrier.max_response_bytes,
            timeout=timeout,
        )
    if borrowed_buffer_calls is not None:
        from merlin.llvmlower.borrowed_buffer_calls import prepare_native

        if "optimize-allocation-liveness" not in pipeline:
            raise ValueError("borrowed lifetime selection requires the checked post-deallocation boundary")
        runner_src, roster = prepare_native(borrowed_buffer_calls, mlir_text, runner_src, work)
        recipe_sources.update(
            borrowed_buffer_calls=roster,
            borrowed_call_contracts=module_source_path("merlin.llvmlower.borrowed_buffer_calls"),
        )
    runner.write_text(runner_src, encoding="utf-8")
    # argv[4] gates the self-copy erase, so the frozen hand_v0 control keeps its byte-identical
    # lowering unless the feature is explicitly enabled.
    stage_out = retention.translation if retention is not None else ((work / "model.llvmdialect.mlir") if omp else out)
    arguments = build_command(
        mlir_text,
        pipeline,
        feats,
        src=src,
        runner=runner,
        stage_out=stage_out,
        omp=omp,
        parallel_harts=parallel_harts,
        index_bits=index_bits,
        data_layout=data_layout,
        lowering_selection=lowering_selection,
        masked_contraction_effects=masked_contraction_effects,
    )
    pipeline, command = arguments.pipeline, arguments.argv
    if audit is not None:
        audit.stage("upstream-scheduled", mlir_text)  # an open compile trace observes it unaudited too
    recipe_sources.update(prepared_mlir=src, runner=runner)
    recipe = LoweringRecipe(work, features=feats, sources=recipe_sources)
    recipe.command(command)
    if audit is not None and audit.directory is not None:
        from merlin.targetgen.provenance import toolchain_provenance

        audit.command(command, sources=(__file__, runner), provenance=toolchain_provenance())
    proc = run_translation(
        command,
        timeout=timeout,
        audit=audit,
        retention=retention,
        source=src,
        runner=runner,
        dependencies=recipe_sources.values(),
        scalar_stage=scalar_stage,
        scalar_carrier=source_scalar_carrier,
    )
    if proc.returncode != 0 or not stage_out.is_file():
        error = PipelineError(
            f"upstream lowering failed (returncode={proc.returncode}, "
            f"output_exists={stage_out.is_file()}):\n{proc.stdout}\n{proc.stderr}"
        )
        if audit is not None:
            try:
                audit.collect_views()
            except (OSError, ValueError) as audit_error:
                error.add_note(f"IR inspection prefix could not be recorded: {type(audit_error).__name__}")
        _harvest_native(traced, stage_out, failure=error)
        raise error
    checked = check_reports(
        proc,
        work=work,
        mlir_text=mlir_text,
        feats=feats,
        arguments=arguments,
        recipe=recipe,
        lowering_selection=lowering_selection,
        audit=audit,
        retention=retention,
        stage_out=stage_out,
        omp=omp,
        traced=traced,
        runner=runner,
        borrowed_buffer_calls=borrowed_buffer_calls,
        scalar_stage=scalar_stage,
        source_observation_effects=source_observation_effects,
    )
    if omp:
        residual_vector_ops = _residual_vector_dialect_ops(stage_out)
        if residual_vector_ops:
            raise PipelineError(
                "vector dialect survived the lowering pipeline before mlir-translate: "
                + ", ".join(residual_vector_ops)
                + f"\nLLVM-dialect artifact: {stage_out}"
            )
        from merlin.llvmlower.toolchain import mlir_translate

        translate_command = [str(mlir_translate()), "--mlir-to-llvmir", str(stage_out), "-o", str(out)]
        recipe.command(translate_command)
        if audit is not None:
            audit.command(translate_command, sources=(__file__,))
        tproc = subprocess.run(
            translate_command, capture_output=True, text=True, timeout=timeout, env=child_environment()
        )
        if tproc.returncode != 0 or not out.is_file():
            raise PipelineError(f"mlir-translate (parallel) failed:\n{tproc.stdout}\n{tproc.stderr}")
    result = _fix_float_literals((retention.translation if retention is not None else out).read_text(encoding="utf-8"))
    if audit is not None:
        audit.stage("llvm-normalized", result, format="llvm-ir")
    if _OUTLINE_LOOPS in feats or _OUTLINE_MERGE_LOOPS in feats:
        result = outline_loops(
            result,
            work / "llvm-loop-outline",
            timeout=timeout,
            audit=audit,
            merge_identical=_OUTLINE_MERGE_LOOPS in feats,
        )
    if retention is not None:
        retention.bind(recipe, lowering_selection, source=src, runner=runner, llvm_ir=result)
    recipe.returned(result)
    if borrowed_buffer_calls is not None and borrowed_arena_lifetimes is not None:
        from merlin.llvmlower.borrowed_arena import _from_verified_native

        borrowed_arena_lifetimes.append(
            _from_verified_native(borrowed_buffer_calls, mlir_text, work, result, checked, runner)
        )
    return result
