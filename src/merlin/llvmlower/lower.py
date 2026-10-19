"""End-to-end whole-model lowering driver.

linalg-on-tensors text
  → Merlin xDSL passes (quant_ext → linalg, emit_c_interface)
  → upstream MLIR pipeline + translation (model2MLIR venv, torch-mlir wheel)
  → clang-23 codegen (x86 host .so / rv64gcv object).
"""

from __future__ import annotations

import os as _os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph_optimizations import GraphOptimizationContract
    from .host_graph_policy import HostGraphPolicy

from merlin.common import compile_trace
from merlin.common.digest import sha256_file, sha256_text
from merlin.common.ir_audit import IrAudit, audit_mode
from merlin.common.jsonio import write_pretty_json

from .broadcast_fold import ensure_registered as _register_fold_broadcast_into_generic
from .codegen import build_host_shared, compile_ll
from .concat_dps import ensure_registered as _register_concat_dps

# Registering here, and not in `impr_features` itself, keeps the epilogue-fusion stage's definition
# next to the pass list it edits while still making the NAME resolvable: this module is what every
# whole-model backend imports to lower, so by the time a feature set is normalized for a lowering the
# entry exists. Idempotent, so a second import is a no-op.
from .epilogue_fusion import ensure_registered as _register_epilogue_fusion
from .im2col_pack import ensure_registered as _register_im2col_panel_pack
from .masked_contraction import MaskEffectContract
from .named_broadcast_fold import ensure_registered as _register_named_broadcast_fold
from .passes_xdsl import PREPROCESS_STAGES, preprocess_text
from .pipeline import lower_to_llvm_ir
from .prov_cse import ensure_registered as _register_cse_through_provenance
from .source_expression_interval import IntervalEffectContract
from .source_scalar_carrier_binding import host_admitted
from .transpose_maps import ensure_registered as _register_fold_weight_transpose

_register_epilogue_fusion()
_register_fold_weight_transpose()
_register_cse_through_provenance()
_register_concat_dps()
_register_im2col_panel_pack()
_register_fold_broadcast_into_generic()
_register_named_broadcast_fold()


#: The stages :func:`lower_model` records, in order. The native passes run between ``upstream-scheduled``
#: and the LLVM stages; they are ``mlir:<pass>`` stages read off the pass pipeline the build uses, so they
#: are not declared here. A serial build records ``llvm-translated``, an OpenMP one ``llvm-dialect``.
STAGES = compile_trace.declare(
    "llvm",
    (
        "input",
        "host-graph-optimized",
        "host-row-scheduled",
        "host-destination-reused",
        "host-rounding-selected",
        "host-permutation-forwarded",
        "operation-profiled",
        *PREPROCESS_STAGES,
        "upstream",
        "upstream-scheduled",
        "llvm-translated",
        "llvm-dialect",
        "llvm-normalized",
        "llvm-final",
    ),
    entry="merlin.llvmlower.lower_model",
    summary="linalg text -> Merlin xDSL rewrites -> upstream MLIR passes (mlir:<pass>) -> LLVM IR",
)


@dataclass
class LowerResult:
    workdir: Path
    ll_path: Path
    stats: dict[str, Any] = field(default_factory=dict)
    host_so: Path | None = None
    riscv_obj: Path | None = None
    audit_index: Path | None = None


@host_admitted
def lower_model(
    mlir_text: str,
    workdir: str | Path,
    targets: tuple[str, ...] = ("host",),
    textual: bool = False,
    vectorize: bool = False,
    transform_schedule: str | None = None,
    hoist_static_allocs: bool = True,
    parallel: bool = False,
    features: frozenset[str] | None = None,
    parallel_harts: int | None = None,
    parallel_chunks: list | None = None,
    static_arena: bool | None = None,
    ir_audit: bool | str = False,
    audit_sidecars: tuple[str | Path, ...] = (),
    data_layout: str | None = None,
    index_bits: int | None = None,
    masked_contraction_effects: MaskEffectContract | None = None,
    source_observation_effects: IntervalEffectContract | None = None,
    source_scalar_carrier=None,
    retain_llvm_dialect: bool = False,
    graph_optimizations: GraphOptimizationContract | HostGraphPolicy | None = None,
    op_profile: bool = False,
) -> LowerResult:
    """Lower MLIR text end to end; emit per-target artifacts in ``workdir``.

    ``op_profile=True`` inserts checked top-level ``@forward`` markers after
    source graph/row/destination optimization and before upstream preprocessing.
    The table and source identities are returned in ``stats['operation_profile']``.
    Markers can still perturb downstream allocation and machine optimization.

    ``data_layout`` is the target's LLVM data layout, set on the module before translation (see
    :func:`merlin.llvmlower.pipeline.lower_to_llvm_ir`). ``index_bits`` explicitly binds every LLVM
    index conversion pass and is returned with the effective pipeline in ``stats``. A selected build
    supplies it from its compiler observation; this generic API does not infer a host/runtime ABI.

    ``retain_llvm_dialect`` forwards optional serial native product retention.
    Its custody receipt in ``stats`` ends at returned upstream LLVM, before
    later arena/ciface transforms, object production and linking.

    ``graph_optimizations`` selects a source-pinned graph contract or original
    software-bound HostGraphPolicy before preprocessing. The latter binds this
    actual prepared source and retains default host passes and checked fallback.
    Structural
    semantic checks and caller cost constraints are reported; this supplies no
    target placement, hardware qualification or cycle prediction.

    ``masked_contraction_effects`` forwards the caller's explicit nontrapping,
    unobserved-floating-flags contract for closed-mask scheduling. Selection
    still requires its scalar schedule and feature; no effects are inferred.

    ``ir_audit=True`` retains exact named-stage IR and a completion/failure index
    in a fresh workdir child. ``audit_sidecars`` binds existing weights/manifests
    by observed bytes; it neither rewrites the ABI nor externalizes new constants.
    ``ir_audit="compact"`` requests native inspection-only views and exact hashes;
    ``"both"`` also retains exact snapshots. Text-only stages have no compact printer.

    ``textual=True`` uses compatibility pure-text preprocessing without an xDSL
    round-trip. The default route emits generic MLIR at the upstream seam to
    preserve attributes and avoid custom-printer grammar skew.

    ``vectorize=True`` bakes native RVV (fixed-width vector ops) into the IR instead of
    relying on clang auto-vectorization — for the rv64gcv vector-tile target.

    ``parallel_harts=N`` layers an outer OpenMP-parallel loop under that RVV schedule, so
    the object is BOTH vectorized and multicore (the multi-hart vector tile / Zephyr SMP path).
    ``parallel_chunks`` (from ``perop_blocks.distinct_parallel_arms``, produced by the same prepare
    step that tagged the IR) makes that split PER OP and block-preserving, so the emitted kernel is
    the 1-hart kernel with a parallel wrapper around it rather than a differently-blocked one; None
    keeps the legacy class-wide ``num_threads`` split.

    ``static_arena=True`` (or ``MERLIN_STATIC_ARENA=1``) binds the emitted per-intermediate
    ``malloc``/``free`` pairs to one statically planned arena -- see
    :mod:`merlin.llvmlower.arena_bind`. Default OFF: unflagged, the emitted ``.ll`` is byte-identical
    to the baseline, and the report of what was bound lands in ``stats["static_arena"]``.
    """
    if type(op_profile) is not bool:
        raise ValueError("operation profiling requires an explicit boolean selection")
    with IrAudit(
        workdir, enabled=ir_audit, producer="merlin.llvmlower.lower_model", source=__file__, sidecars=audit_sidecars
    ) as audit:
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
        for name in ("operation-profile.json", "model.profiled.mlir", "op_profile_table.json",
                     "host-graph-optimization.json", "host-graph-input.mlir", "host-graph-optimized.mlir",
                     "host-graph-software.yaml", "host-graph-policy.json"):
            (work / name).unlink(missing_ok=True)

        audit.stage("input", mlir_text)
        graph_report = None
        if graph_optimizations is not None:
            from .graph_optimizations import optimize_host_graph
            from .host_graph_policy import HostGraphPolicy

            (work / "host-graph-input.mlir").write_text(mlir_text, encoding="utf-8")
            if type(graph_optimizations) is HostGraphPolicy:
                graph_optimizations.retain(work)
            mlir_text, graph_report = optimize_host_graph(mlir_text, graph_optimizations)
            (work / "host-graph-optimized.mlir").write_text(mlir_text, encoding="utf-8")
            write_pretty_json(work / "host-graph-optimization.json", graph_report)
            audit.stage("host-graph-optimized", mlir_text)
        from .automatic_rows import prepare_automatic_rows

        # Select after the actual lowering family is known. Vector/parallel
        # schedules and explicit source-bound observers retain their route
        # until their physical output forwarding has independent qualification.
        serial_rows = not (
            vectorize
            or parallel
            or transform_schedule is not None
            or parallel_harts is not None
            or parallel_chunks is not None
            or (graph_optimizations is not None and type(graph_optimizations) is not HostGraphPolicy)
            or masked_contraction_effects is not None
            or source_observation_effects is not None
            or source_scalar_carrier is not None
            or index_bits not in (None, 64)
            or (data_layout is not None and index_bits is None)
        )
        mlir_text, row_report = prepare_automatic_rows(
            mlir_text, serial=serial_rows, index_bits=64 if index_bits is None else index_bits, data_layout=data_layout
        )
        if row_report["rewrites"]:
            audit.stage("host-row-scheduled", mlir_text)
        from .automatic_destinations import prepare_automatic_destinations
        from .toolchain import _env

        mlir_text, destination_report = prepare_automatic_destinations(
            mlir_text,
            serial=serial_rows and _env("MERLIN_COMPILER_DRIVER", "python") == "python",
            index_bits=64 if index_bits is None else index_bits,
            data_layout=data_layout,
        )
        if destination_report["rewrites"]:
            from .impr_features import normalize

            features = normalize(set(features or ()) | {"reuse_tensor_destination"})
            audit.stage("host-destination-reused", mlir_text)
        write_pretty_json(work / "automatic-host-destinations.json", destination_report)
        from .automatic_rounding import prepare_automatic_rounding

        mlir_text, rounding_report = prepare_automatic_rounding(
            mlir_text, serial=serial_rows, index_bits=64 if index_bits is None else index_bits, data_layout=data_layout
        )
        if rounding_report["rewrites"]:
            audit.stage("host-rounding-selected", mlir_text)
        write_pretty_json(work / "automatic-host-rounding.json", rounding_report)
        mlir_text, permutation_report = prepare_automatic_destinations(
            mlir_text,
            serial=serial_rows and _env("MERLIN_COMPILER_DRIVER", "python") == "python",
            index_bits=64 if index_bits is None else index_bits,
            data_layout=data_layout,
            permuted_outputs=True,
        )
        if permutation_report["rewrites"]:
            audit.stage("host-permutation-forwarded", mlir_text)
        write_pretty_json(work / "automatic-host-permutations.json", permutation_report)
        profile_report = None
        if op_profile:
            from . import op_profile as profiler

            profile_source_sha = sha256_text(mlir_text)
            mlir_text, profile_table = profiler.instrument(mlir_text, structural=True)
            profile_path = work / "model.profiled.mlir"
            profile_path.write_text(mlir_text, encoding="utf-8")
            table_path = profiler.write_table(profile_table, work / "op_profile_table.json")
            profile_report = {
                "schema": "merlin.lowering_operation_profile.v1",
                "stage": "after_source_optimization_before_preprocessing",
                "source_file_sha256": profile_source_sha,
                "instrumented_source": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
                "table": {"path": str(table_path), "sha256": sha256_file(table_path)},
                "operations": len(profile_table),
                "machine_conformance": "not_established",
            }
            write_pretty_json(work / "operation-profile.json", profile_report)
            audit.stage("operation-profiled", mlir_text)
        if textual:
            from .passes_xdsl import preprocess_text_textual

            upstream_text, stats = preprocess_text_textual(mlir_text)
        else:
            observed = audit.directory is not None or compile_trace.active() is not None
            upstream_text, stats = preprocess_text(mlir_text, audit=audit) if observed else preprocess_text(mlir_text)
        if graph_report is not None:
            stats["graph_optimizations"] = graph_report
        stats["automatic_host_rows"] = row_report
        stats["automatic_host_destinations"] = destination_report
        stats["automatic_host_permutations"] = permutation_report
        stats["automatic_host_rounding"] = rounding_report
        if profile_report is not None:
            stats["operation_profile"] = profile_report
        (work / "model.upstream.mlir").write_text(upstream_text, encoding="utf-8")
        row_report["upstream_file_sha256"] = sha256_text(upstream_text)
        write_pretty_json(work / "automatic-host-rows.json", row_report)

        audit.stage("upstream", upstream_text)
        lowering_selection: dict[str, Any] = {}
        try:
            ll_text = lower_to_llvm_ir(
                upstream_text,
                workdir=work,
                audit=audit,
                vectorize=vectorize,
                transform_schedule=transform_schedule,
                hoist_static_allocs=hoist_static_allocs,
                parallel=parallel,
                features=features,
                parallel_harts=parallel_harts,
                parallel_chunks=parallel_chunks,
                data_layout=data_layout,
                index_bits=index_bits,
                lowering_selection=lowering_selection,
                masked_contraction_effects=masked_contraction_effects,
                source_observation_effects=source_observation_effects,
                source_scalar_carrier=source_scalar_carrier,
                retain_llvm_dialect=retain_llvm_dialect,
            )
        except Exception as exc:
            # A module MLIR refuses to PARSE fails before any pass, and the reader's dump names a line
            # number in a machine-written module — not the captured layer that produced it. The commonest
            # cause is a windowed op (pooling, strided convolution) whose window dims are bound by no
            # indexing map; `window_maps.explain` names the op, the dims and the layer. Diagnosis only,
            # run once the lowering has already failed, and it re-raises the original error either way.
            from .window_maps import explain as _explain_window_maps

            note = _explain_window_maps(upstream_text)
            if note is None:
                raise
            try:  # keep the caller's except-clause working
                enriched = type(exc)(f"{exc}\n\n{note}")
            except Exception:  # noqa: BLE001 -- an exotic constructor: keep the original
                raise exc
            raise enriched from exc
        if "source_observation" in lowering_selection:
            stats["source_observation"] = lowering_selection["source_observation"]
        if "source_scalar_carrier" in lowering_selection:
            stats["source_scalar_carrier"] = lowering_selection["source_scalar_carrier"]
        if "llvm_dialect_product" in lowering_selection:
            stats["llvm_dialect_product"] = lowering_selection["llvm_dialect_product"]
        if index_bits is not None:
            if lowering_selection.get("index_bits") != index_bits or not lowering_selection.get("effective_pipeline"):
                raise ValueError("selected index lowering lost its effective compiler pipeline")
            stats["index_lowering"] = lowering_selection
        if static_arena is None:
            from .optional_passes import switched

            static_arena = switched("static-arena", bool(_os.environ.get("MERLIN_STATIC_ARENA")))
        if static_arena:
            # Bind the emitted heap allocations to one statically planned arena. Kept behind a flag, and
            # applied HERE rather than inside the pass pipeline, because this is the last point the
            # measured whole-model path still passes through as text: `mining.k1.build_k1_binary` calls
            # `lower_model_file` and then compiles the `.ll` it gets back. With the flag off nothing runs
            # and the `.ll` is byte-identical to the frozen baseline.
            from .arena_bind import bind_arena

            ll_text, arena_report = bind_arena(ll_text)
            stats["static_arena"] = arena_report.to_dict()
        # MLIR's C wrapper expands each memref pointer into the implementation's flattened descriptor
        # ABI.  An external call can make LLVM decline to inline a hundreds-of-arguments boundary,
        # leaving target backends to materialize an enormous outgoing frame.  Repair that artificial
        # boundary by ABI width before any target compiles the module.  This is model- and target-agnostic
        # and preserves both public symbols; see llvmlower.ciface_inline.
        from .ciface_inline import inline_wide_ciface_implementations

        ll_text, ciface_inline_report = inline_wide_ciface_implementations(ll_text)
        stats["ciface_boundary_inline"] = ciface_inline_report
        audit.stage("llvm-final", ll_text, format="llvm-ir")
        ll_path = work / "model.ll"
        ll_path.write_text(ll_text, encoding="utf-8")

        result = LowerResult(
            workdir=work,
            ll_path=ll_path,
            stats=stats,
            audit_index=audit.directory / "index.json" if audit.directory is not None else None,
        )
        if "host" in targets:
            result.host_so = build_host_shared(ll_path, work / "model_host.so")
        if "riscv" in targets:
            result.riscv_obj = compile_ll(ll_path, work / "model_rv64gcv.o", "riscv")
        return result


@host_admitted
def lower_model_file(
    mlir_path: str | Path,
    workdir: str | Path,
    targets: tuple[str, ...] = ("host",),
    textual: bool = False,
    vectorize: bool = False,
    transform_schedule: str | None = None,
    hoist_static_allocs: bool = True,
    parallel: bool = False,
    features: frozenset[str] | None = None,
    parallel_harts: int | None = None,
    parallel_chunks: list | None = None,
    static_arena: bool | None = None,
    ir_audit: bool | str = False,
    audit_sidecars: tuple[str | Path, ...] = (),
    data_layout: str | None = None,
    index_bits: int | None = None,
    masked_contraction_effects: MaskEffectContract | None = None,
    source_observation_effects: IntervalEffectContract | None = None,
    source_scalar_carrier=None,
    retain_llvm_dialect: bool = False,
    graph_optimizations: GraphOptimizationContract | HostGraphPolicy | None = None,
    op_profile: bool = False,
) -> LowerResult:
    audit_mode(ir_audit)
    return lower_model(
        Path(mlir_path).read_text(encoding="utf-8"),
        workdir,
        targets,
        textual=textual,
        vectorize=vectorize,
        transform_schedule=transform_schedule,
        hoist_static_allocs=hoist_static_allocs,
        parallel=parallel,
        features=features,
        parallel_harts=parallel_harts,
        parallel_chunks=parallel_chunks,
        static_arena=static_arena,
        ir_audit=ir_audit,
        audit_sidecars=audit_sidecars,
        data_layout=data_layout,
        index_bits=index_bits,
        masked_contraction_effects=masked_contraction_effects,
        source_observation_effects=source_observation_effects,
        source_scalar_carrier=source_scalar_carrier,
        retain_llvm_dialect=retain_llvm_dialect,
        graph_optimizations=graph_optimizations,
        op_profile=op_profile,
    )
