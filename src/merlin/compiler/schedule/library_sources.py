"""Explicit shared compiler source owners; no discovery or import grants.

The export writer and ordinary pass receipts consume this same roster. Adding
an owner requires review of its import/resource closure and authoring boundary.
This roster does not attest loaded code or external dependency implementations.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

# The single reviewed shared-compiler SDK release. Earlier releases are retired:
# their frozen runs keep their archived environment, and no current entry point
# reinterprets current sources as one of them.
CURRENT_SDK_VERSION = 27
CURRENT_LIBRARY_MODULES = (
    "merlin",
    "merlin.common",
    "merlin.common.dense_float_ir",
    "merlin.compiler",
    "merlin.compiler.schedule",
    "merlin.compiler.schedule.blocks",
    "merlin.compiler.schedule.zero_initializer_groups",
    "merlin.compiler.schedule.shared_input_verification",
    "merlin.compiler.schedule.shared_input_schedule",
    "merlin.compiler.schedule.shared_input_groups",
    "merlin.compiler.schedule.panel_transfers",
    "merlin.compiler.schedule.copy_pipeline",
    "merlin.compiler.schedule.periodic_effects",
    "merlin.compiler.schedule.configuration_loops",
    "merlin.compiler.schedule.command_compaction",
    "merlin.compiler.schedule.compact_dense",
    "merlin.compiler.schedule.compact_loaded_sum",
    "merlin.compiler.schedule.spatial_gather",
    "merlin.compiler.schedule.compact_verification",
    "merlin.compiler.schedule.configuration",
    "merlin.compiler.schedule.configuration_ir",
    "merlin.compiler.schedule.copy_elimination",
    "merlin.compiler.schedule.dense_contracts",
    "merlin.compiler.schedule.dense_source",
    "merlin.compiler.schedule.device_spills",
    "merlin.compiler.schedule.fusion",
    "merlin.compiler.schedule.fusion_contracts",
    "merlin.compiler.schedule.fusion_semantics",
    "merlin.compiler.schedule.library_sources",
    "merlin.compiler.schedule.llvm_program",
    "merlin.compiler.schedule.pointwise_source",
    "merlin.compiler.schedule.program_contracts",
    "merlin.compiler.schedule.program_limits",
    "merlin.compiler.schedule.row_contracts",
    "merlin.compiler.schedule.row_softmax",
    "merlin.compiler.schedule.row_softmax_verification",
    "merlin.compiler.schedule.schedule_pipeline",
    "merlin.compiler.schedule.selected_contracts",
    "merlin.compiler.schedule.spill_conformance",
    "merlin.compiler.schedule.spill_contracts",
    "merlin.compiler.schedule.storage_contracts",
    "merlin.compiler.schedule.staged_dense",
    "merlin.compiler.schedule.staged_dense_verification",
    "merlin.compiler.schedule.temporary_storage",
    "merlin.compiler.schedule.window_transfer",
    "merlin.llvmlower",
    "merlin.llvmlower.cpu_softmax",
    "merlin.compiler.kernel_ir",
    "merlin.compiler.kernel_ir.codegen",
    "merlin.compiler.kernel_ir.ir",
    "merlin.compiler.kernel_ir.parse",
    "merlin.semantic_compiler",
    "merlin.semantic_compiler.allocate",
    "merlin.semantic_compiler.allocation_types",
    "merlin.semantic_compiler.egg_bridge",
    "merlin.semantic_compiler.extract",
    "merlin.semantic_compiler.model",
    "merlin.semantic_compiler.rules",
    "merlin.targetgen",
    "merlin.targetgen.isa_dialect_gen",
    "merlin.targetgen.isa_spec",
    "merlin.transforms",
    "merlin.transforms.certificates",
    "merlin.transforms.dense_ir",
    "merlin.transforms.dense_planning",
    "merlin.transforms.normalization",
    "merlin.transforms.pipeline_state",
    "merlin.transforms.pure_scalar",
    "merlin.transforms.scalar_witness",
    "merlin.transforms.unrolling",
    "merlin.verification",
    "merlin.verification.epilogue_applicability",
    "merlin.compiler.schedule.periodic_dense",
    "merlin.compiler.schedule.spatial_producer",
    "merlin.compiler.schedule.spatial_readout",
    "merlin.llvmlower.integer_readout",
    "merlin.llvmlower.integer_producer_range",
    "merlin.llvmlower.requantization",
    "merlin.llvmlower.spatial_scratch_abi",
    "merlin.llvmlower.fresh_tensor_writer",
    "merlin.llvmlower.private_workspace_pool",
    "merlin.llvmlower.borrowed_effects",
    "merlin.xdsl_dialects",
    "merlin.xdsl_dialects._common",
    "merlin.xdsl_dialects.fp8",
    "merlin.compiler.schedule.schedule_ir",
    "merlin.xdsl_dialects.schedule",
    "merlin.xdsl_dialects.contract",
    "merlin.compiler.schedule.loop_schedule",
    "merlin.compiler.kernel_ir.primitives",
    "merlin.compiler.schedule.dense_panel_pipeline",
    "merlin.compiler.schedule.dense_panel_contracts",
    "merlin.compiler.schedule.dense_panel_ir",
    "merlin.compiler.schedule.dense_retained",
    "merlin.compiler.schedule.scf_commands",
    "merlin.xdsl_dialects.runtime",
    "merlin.compiler.schedule.configuration_contexts",
    "merlin.compiler.schedule.configuration_loop_cleanup",
    "merlin.compiler.schedule.dense_panel_readout",
    "merlin.compiler.schedule.tile_load_coalescing",
    "merlin.compiler.schedule.panel_tile_loads",
    "merlin.compiler.schedule.dense_segmented_pipeline",
    "merlin.compiler.schedule.dense_array_retention",
    "merlin.compiler.schedule.api_catalog",
    "merlin.compiler.schedule.rolling_window",
    "merlin.compiler.schedule.rolling_spatial",
    "merlin.xdsl_dialects.rolling_window",
    "merlin.compiler.schedule.spatial_readout_cache",
    "merlin.compiler.schedule.dense_readout_overlap",
    "merlin.llvmlower.host_builder",
    "merlin.llvmlower.llvm_loop_metadata",
    "merlin.compiler.schedule.projection_views",
    "merlin.transforms.tensor_views",
    "merlin.llvmlower.source_region",
    "merlin.llvmlower.source_region_host",
    "merlin.compiler.passes.composition",
    "merlin.compiler.schedule.dense_ready_readout",
    "merlin.compiler.schedule.scf_carried_events",
    "merlin.compiler.kernel_ir.loops",
    "merlin.common.access",
    "merlin.common.compile_trace",
    "merlin.common.ir_audit",
    "merlin.common.ir_lock",
    "merlin.common.paths",
    "merlin.common.provenance",
    "merlin.common.provenance_lost",
    "merlin.common.tree_hash",
    "merlin.frontends",
    "merlin.frontends.linalg_mlir",
    "merlin.integrations",
    "merlin.integrations.model2mlir",
    "merlin.llvmlower.generic_form",
    "merlin.llvmlower.normalization_reassociation",
    "merlin.llvmlower.optional_passes",
    "merlin.compiler.frontend.preprocess",
    "merlin.llvmlower.source_fma_batch",
    "merlin.llvmlower.toolchain",
    "merlin.runtime",
    "merlin.runtime.host_arithmetic",
    "merlin.xdsl_dialects.ir_inspection",
    "merlin.xdsl_dialects.lowering",
    "merlin.xdsl_dialects.lowering.outline",
    "merlin.common.stat_memo",
    "merlin.compiler.schedule.completion_streams",
    "merlin.compiler.schedule.configuration_motion",
    "merlin.compiler.schedule.spatial_max_stripes",
    "merlin.compiler.ir.logical_abi",
    "merlin.compiler.schedule.convolution_exploration",
    "merlin.compiler.schedule.convolution_choices",
    "merlin.compiler.schedule.dense_rhs_residency",
    "merlin.transforms.pipeline_policy",
    "merlin.transforms.pass_exploration",
    "merlin.transforms.evaluated_exploration",
    "merlin.transforms.shape_policy",
    "merlin.compiler.schedule.dense_panel_epochs",
    "merlin.compiler.schedule.pointwise_convolution",
    "merlin.compiler.schedule.affine_gather_runs",
    "merlin.compiler.schedule.batched_dense",
    "merlin.compiler.schedule.convolution_output_groups",
    "merlin.compiler.schedule.loaded_sum_epochs",
    "merlin.compiler.selection.knobs",
    "merlin.transforms.tuning_selection",
    "merlin.transforms.dense_reduction_layout",
    "merlin.llvmlower.layout_propagation",
    "merlin.compiler.schedule.convolution_input_reuse",
    "merlin.perf.factored_storage",
    "merlin.perf",
    "merlin.llvmlower.constant_panel_layout",
    "merlin.compiler.schedule.dense_loaded_sum_composition",
    "merlin.compiler.schedule.dense_loaded_sum_groups",
    "merlin.llvmlower.quantized_sum_source",
    "merlin.llvmlower.generated_private_workspace",
    "merlin.targetgen.oot_starterkit",
    "merlin.targetgen.oot_starterkit.llvm_context",
    "merlin.llvmlower.late_quant_rne",
    "merlin.llvmlower.source_expression_interval",
    "merlin.llvmlower.source_expression_interval_llvm",
    "merlin.llvmlower.bounded_rne_word_cells",
    "merlin.llvmlower.scaled_integer_finite",
    "merlin.llvmlower.scaled_integer_finite_llvm",
    "merlin.llvmlower.bounded_rne_maps",
    "merlin.llvmlower.closed_tensor_insert",
    "merlin.compiler.schedule.projection_dense",
    "merlin.compiler.schedule.projection_provenance",
    "merlin.compiler.schedule.rolling_lookahead",
    "merlin.compiler.schedule.convolution_input_first_use",
    "merlin.compiler.schedule.operand_slots",
    "merlin.compiler.schedule.compact_dense_group",
    "merlin.compiler.schedule.compact_dense_group_check",
    "merlin.compiler.schedule.dense_shared_rhs",
    "merlin.compiler.schedule.dense_shared_rhs_check",
    "merlin.compiler.schedule.shared_rhs_schedule_ir",
    "merlin.compiler.schedule.horizontal_convolution",
    "merlin.common.digest",
    "merlin.common.mlir_query",
    "merlin.common.yaml",
    "merlin.capture",
    "merlin.capture.safetensors",
    "merlin.llvmlower.repeated_pointwise",
    "merlin.llvmlower.contraction_row_slice",
    "merlin.llvmlower.quant_round_check",
    "merlin.llvmlower.segmented_matrix_view",
    "merlin.llvmlower.layout_conformance",
    "merlin.llvmlower.weight_prepack_runtime",
    "merlin.llvmlower.weights_pack",
    "merlin.llvmlower.prepared_operand_effects",
    "merlin.llvmlower.tensor_preparation_identity",
    "merlin.llvmlower.ordered_fma_groups",
    "merlin.transforms.head_batching",
    "merlin.transforms.dense_row_batching",
    "merlin.transforms.dense_row_batching_check",
    "merlin.transforms.shared_rhs_views",
    "merlin.analysis",
    "merlin.compiler.ir.contraction_shapes",
    "merlin.compiler.ir",
    "merlin.compiler.ir.boundary",
    "merlin.compiler.passes",
    "merlin.compiler.passes.xdsl",
    "merlin.llvmlower.device_rewrite",
    "merlin.llvmlower.ordered_fma_matmul",
    "merlin.llvmlower.ordered_fma_rewrite",
    "merlin.llvmlower.quant_round_chain",
    "merlin.semantic_compiler.allocation_schedule",
    "merlin.semantic_compiler.instruction",
    "merlin.semantic_compiler.view_descriptors",
    "merlin.xdsl_dialects.lowering.contraction_coverage",
    "merlin.compiler.schedule.model",
    "merlin.compiler.schedule.residency",
    "merlin.compiler.schedule.block",
    "merlin.compiler.schedule.convolution",
    "merlin.compiler.schedule.execute",
    "merlin.compiler.schedule.readout",
    "merlin.compiler.schedule.interface",
    "merlin.compiler.selection",
    "merlin.compiler.schedule.catalog_dense",
    "merlin.compiler.schedule.catalog_spatial",
    "merlin.compiler.schedule.catalog_source",
    "merlin.compiler.schedule.catalog_effects",
    "merlin.compiler.frontend",
    "merlin.compiler.frontend.repairs",
    "merlin.compiler.ir.provenance",
    "merlin.compiler.quantization",
    "merlin.compiler.quantization.linalg",
    "merlin.compiler.transforms",
    "merlin.compiler.transforms.matmul",
    "merlin.compiler.transforms.dead",
    "merlin.compiler.lower",
    "merlin.compiler.lower.scalar",
    "merlin.compiler.lower.scalar.casts",
    "merlin.compiler.schedule.window_geometry",
    "merlin.compiler.schedule.convolution_input_rows",
    "merlin.compiler.schedule.convolution_transfer_laws",
    "merlin.compiler.schedule.convolution_tiles",
    "merlin.compiler.schedule.convolution_rhs",
    "merlin.compiler.schedule.convolution_admission",
    "merlin.compiler.schedule.convolution_realization",
    "merlin.compiler.schedule.convolution_candidates",
    "merlin.compiler.schedule.convolution_storage",
    "merlin.compiler.schedule.dense_panel_build",
    "merlin.compiler.schedule.dense_panel_projection",
    "merlin.compiler.schedule.dense_panel_verify",
)


def require_current_sdk(version, *, what="shared compiler inventory"):
    """Refuse any SDK selection other than the current release, naming the retired one."""
    if type(version) is not int or version != CURRENT_SDK_VERSION:
        raise ValueError(f"{what} version {version!r} is retired; only SDK{CURRENT_SDK_VERSION} is available")


def library_modules(version=CURRENT_SDK_VERSION):
    """Select the closed reviewed inventory; this grants no import access."""
    require_current_sdk(version)
    return CURRENT_LIBRARY_MODULES


def shared_library_sources(*, version=CURRENT_SDK_VERSION):
    """Current source identities, equal for every treatment of one library."""
    return {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        for name in library_modules(version)
    }


# ----------------------------------------------------------- author-visible admission records
#
# Checked consumers either return a checked result or refuse (``ValueError`` naming the missing
# premise, or ``None`` when a planner's capacity does not fit). The caller then emits its own
# original route, and nothing used to say so. ``admission_point`` records each decision without
# changing it: the wrapped call's arguments, result and exception are passed through untouched,
# and recording failures are swallowed. Records stay in this process (``admission_records``) and,
# unless ``MERLIN_SCHEDULING_ADMISSION_REPORT=0``, one bounded line per distinct decision goes to
# stderr, which the host retains per invocation. They are advisory author feedback produced by the
# copy of this module the compiler imports: never a qualification, grant or performance claim.
ADMISSION_SCHEMA = "merlin.scheduling_admission.v1"
ADMISSION_PREFIX = "merlin-scheduling-admission: "
ADMISSION_ENV = "MERLIN_SCHEDULING_ADMISSION_REPORT"
ADMISSION_LIMIT = 64
ADMISSION_TEXT = 300
_ADMISSIONS: dict = {}
_ADMISSION_STATE = {"dropped": 0, "exit_hook": False}


def _admission_shape(args, kwargs):
    for value in (*args, *kwargs.values()):
        dims = tuple(getattr(value, axis, None) for axis in ("m", "k", "n"))
        if all(type(v) is int for v in dims):
            return dict(zip(("m", "k", "n"), dims))
    return None


def _admission_route(fallback_route, args):
    """The caller's fallback, plus the input checked route kind when the first argument has one."""
    try:
        realization = getattr(args[0], "realization", None) if args else None
        kind = ((realization or {}).get("schedule_ir") or {}).get("pipeline_kind")
    except Exception:  # noqa: BLE001 -- advisory description only
        kind = None
    return fallback_route + (f" (input route: {kind})" if type(kind) is str and kind else "")


def _admission_partial(result):
    refusals = getattr(result, "refusals", None)
    if isinstance(refusals, (list, tuple)) and refusals:
        return [str(value)[:ADMISSION_TEXT] for value in refusals[:4]]
    return None


def _admission_line(document):
    import json
    import os
    import sys

    if os.environ.get(ADMISSION_ENV, "1").strip() == "0" or sys.stderr is None:
        return
    sys.stderr.write(ADMISSION_PREFIX + json.dumps(document, sort_keys=True) + "\n")
    sys.stderr.flush()


def _admission_summary():
    if _ADMISSIONS:
        counts = {str(record["id"]): record["count"] for record in _ADMISSIONS.values()}
        _admission_line({"schema": ADMISSION_SCHEMA, "summary": counts, "dropped": _ADMISSION_STATE["dropped"]})


def _record_admission(api, admitted, premise, route, args, kwargs, result=None):
    import atexit

    shape = _admission_shape(args, kwargs)
    partial = _admission_partial(result) if admitted else None
    key = (api, admitted, premise, route, tuple(sorted((shape or {}).items())), tuple(partial or ()))
    record = _ADMISSIONS.get(key)
    if record is not None:
        record["count"] += 1
        return
    if len(_ADMISSIONS) >= ADMISSION_LIMIT:
        _ADMISSION_STATE["dropped"] += 1
        return
    record = {
        "schema": ADMISSION_SCHEMA,
        "id": len(_ADMISSIONS),
        "api": api,
        "admitted": admitted,
        "failed_premise": premise,
        "fallback_route": route,
        "shape": shape,
        "partial_refusals": partial,
        "count": 1,
    }
    _ADMISSIONS[key] = record
    if not _ADMISSION_STATE["exit_hook"]:
        _ADMISSION_STATE["exit_hook"] = True
        atexit.register(_admission_summary)
    _admission_line({k: v for k, v in record.items() if k != "count"})


def admission_point(api, *, fallback_route, none_refuses=None):
    """Record admit/refuse for one checked consumer; never change its result or exception.

    ``fallback_route`` says what the caller emits when this consumer refuses. ``none_refuses``
    is the premise text for planners whose ``None`` result means "does not fit".
    """
    import functools

    def decorate(function):
        @functools.wraps(function)
        def checked(*args, **kwargs):
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                try:
                    premise = str(error) if isinstance(error, ValueError) else f"{type(error).__name__}: {error}"
                    _record_admission(
                        api, False, premise[:ADMISSION_TEXT], _admission_route(fallback_route, args), args, kwargs
                    )
                except Exception:  # noqa: BLE001 -- recording never changes the refusal
                    pass
                raise
            try:
                if result is None and none_refuses is not None:
                    _record_admission(api, False, none_refuses, _admission_route(fallback_route, args), args, kwargs)
                else:
                    _record_admission(api, True, None, None, args, kwargs, result)
            except Exception:  # noqa: BLE001 -- recording never changes the result
                pass
            return result

        checked.admission_api = api
        return checked

    return decorate


def admission_records():
    """Copies of this process's distinct admission records, with occurrence counts."""
    import copy

    return [copy.deepcopy(record) for record in sorted(_ADMISSIONS.values(), key=lambda r: r["id"])]


def parse_admission_stream(text, *, limit=ADMISSION_LIMIT):
    """Distinct records from one process's stderr; malformed or foreign lines are ignored."""
    import json

    records, counts, dropped = {}, {}, 0
    for line in str(text).splitlines():
        if not line.startswith(ADMISSION_PREFIX):
            continue
        try:
            document = json.loads(line[len(ADMISSION_PREFIX) :])
        except ValueError:
            continue
        if type(document) is not dict or document.get("schema") != ADMISSION_SCHEMA:
            continue
        if type(document.get("summary")) is dict:
            counts = {str(k): v for k, v in document["summary"].items() if type(v) is int and v > 0}
            dropped = document.get("dropped") if type(document.get("dropped")) is int else 0
            continue
        if (
            type(document.get("id")) is not int
            or type(document.get("api")) is not str
            or type(document.get("admitted")) is not bool
            or len(records) >= limit
        ):
            continue
        premise, route = document.get("failed_premise"), document.get("fallback_route")
        shape = document.get("shape")
        partial = document.get("partial_refusals")
        records[document["id"]] = {
            "api": document["api"][:120],
            "admitted": document["admitted"],
            "failed_premise": premise[:ADMISSION_TEXT] if type(premise) is str else None,
            "fallback_route": route[:ADMISSION_TEXT] if type(route) is str else None,
            "shape": (
                {k: shape[k] for k in ("m", "k", "n") if type(shape.get(k)) is int} if type(shape) is dict else None
            ),
            "partial_refusals": (
                [str(v)[:ADMISSION_TEXT] for v in partial[:4]] if type(partial) is list and partial else None
            ),
        }
    rows = []
    for identifier, record in sorted(records.items()):
        rows.append(dict(record, count=counts.get(str(identifier), 1)))
    return rows, dropped
