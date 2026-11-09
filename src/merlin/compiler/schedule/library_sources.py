"""Explicit shared compiler source owners; no discovery or import grants.

The export writer and ordinary pass receipts consume this same roster. Adding
an owner requires review of its import/resource closure and authoring boundary.
This roster does not attest loaded code or external dependency implementations.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

LIBRARY_MODULES_V6 = (
    "merlin",
    "merlin.common",
    "merlin.common.dense_float_ir",
    "merlin.compile",
    "merlin.compile.scheduling",
    "merlin.compile.scheduling.block_schedule",
    "merlin.compile.scheduling.compact_dense",
    "merlin.compile.scheduling.compact_verification",
    "merlin.compile.scheduling.configuration",
    "merlin.compile.scheduling.configuration_ir",
    "merlin.compile.scheduling.copy_elimination",
    "merlin.compile.scheduling.dense_contracts",
    "merlin.compile.scheduling.dense_source",
    "merlin.compile.scheduling.device_spills",
    "merlin.compile.scheduling.fusion",
    "merlin.compile.scheduling.fusion_contracts",
    "merlin.compile.scheduling.fusion_semantics",
    "merlin.compile.scheduling.library_sources",
    "merlin.compile.scheduling.pointwise_source",
    "merlin.compile.scheduling.program_contracts",
    "merlin.compile.scheduling.program_limits",
    "merlin.compile.scheduling.row_contracts",
    "merlin.compile.scheduling.row_softmax",
    "merlin.compile.scheduling.row_softmax_verification",
    "merlin.compile.scheduling.schedule_pipeline",
    "merlin.compile.scheduling.selected_contracts",
    "merlin.compile.scheduling.spill_conformance",
    "merlin.compile.scheduling.spill_contracts",
    "merlin.compile.scheduling.storage_contracts",
    "merlin.compile.scheduling.temporary_storage",
    "merlin.llvmlower",
    "merlin.llvmlower.cpu_softmax",
    "merlin.sched",
    "merlin.sched.codegen",
    "merlin.sched.ir",
    "merlin.sched.ir.expr",
    "merlin.sched.ir.integer_domain",
    "merlin.sched.ir.kernel",
    "merlin.sched.ir.parse",
    "merlin.sched.isa",
    "merlin.sched.isa.types",
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
    "merlin.verify",
    "merlin.verify.epilogue_applicability",
)

# Separate reviewed inventories: additions to V7 must never enlarge a V6
# source release or change the legacy receipt owner's import closure.
LIBRARY_MODULES_V7 = (
    *LIBRARY_MODULES_V6,
    "merlin.compile.scheduling.staged_dense",
    "merlin.compile.scheduling.staged_dense_verification",
)
# Explicit prospective inventory; historical V6/V7 tuples remain unchanged.
LIBRARY_MODULES_V8 = (
    "merlin",
    "merlin.common",
    "merlin.common.dense_float_ir",
    "merlin.compile",
    "merlin.compile.scheduling",
    "merlin.compile.scheduling.block_schedule",
    "merlin.compile.scheduling.zero_initializer_groups",
    "merlin.compile.scheduling.shared_input_verification",
    "merlin.compile.scheduling.shared_input_schedule",
    "merlin.compile.scheduling.shared_input_groups",
    "merlin.compile.scheduling.panel_transfers",
    "merlin.compile.scheduling.copy_pipeline",
    "merlin.compile.scheduling.periodic_effects",
    "merlin.compile.scheduling.configuration_loops",
    "merlin.compile.scheduling.command_compaction",
    "merlin.compile.scheduling.compact_dense",
    "merlin.compile.scheduling.compact_loaded_sum",
    "merlin.compile.scheduling.spatial_gather",
    "merlin.compile.scheduling.compact_verification",
    "merlin.compile.scheduling.configuration",
    "merlin.compile.scheduling.configuration_ir",
    "merlin.compile.scheduling.copy_elimination",
    "merlin.compile.scheduling.dense_contracts",
    "merlin.compile.scheduling.dense_source",
    "merlin.compile.scheduling.device_spills",
    "merlin.compile.scheduling.fusion",
    "merlin.compile.scheduling.fusion_contracts",
    "merlin.compile.scheduling.fusion_semantics",
    "merlin.compile.scheduling.library_sources",
    "merlin.compile.scheduling.llvm_program",
    "merlin.compile.scheduling.pointwise_source",
    "merlin.compile.scheduling.program_contracts",
    "merlin.compile.scheduling.program_limits",
    "merlin.compile.scheduling.row_contracts",
    "merlin.compile.scheduling.row_softmax",
    "merlin.compile.scheduling.row_softmax_verification",
    "merlin.compile.scheduling.schedule_pipeline",
    "merlin.compile.scheduling.selected_contracts",
    "merlin.compile.scheduling.spill_conformance",
    "merlin.compile.scheduling.spill_contracts",
    "merlin.compile.scheduling.storage_contracts",
    "merlin.compile.scheduling.staged_dense",
    "merlin.compile.scheduling.staged_dense_verification",
    "merlin.compile.scheduling.temporary_storage",
    "merlin.compile.scheduling.window_transfer",
    "merlin.llvmlower",
    "merlin.llvmlower.cpu_softmax",
    "merlin.sched",
    "merlin.sched.codegen",
    "merlin.sched.ir",
    "merlin.sched.ir.expr",
    "merlin.sched.ir.integer_domain",
    "merlin.sched.ir.kernel",
    "merlin.sched.ir.parse",
    "merlin.sched.isa",
    "merlin.sched.isa.types",
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
    "merlin.verify",
    "merlin.verify.epilogue_applicability",
)
# Prospective selection only; existing reviewed inventories stay unchanged.
LIBRARY_MODULES_V9 = (*LIBRARY_MODULES_V8, "merlin.compile.scheduling.periodic_dense")
LIBRARY_MODULES_V10 = (
    *LIBRARY_MODULES_V9,
    "merlin.compile.scheduling.spatial_producer",
    "merlin.compile.scheduling.spatial_readout",
    "merlin.llvmlower.integer_readout",
    "merlin.llvmlower.integer_producer_range",
    "merlin.llvmlower.requantization",
)
LIBRARY_MODULES_V11 = (
    *LIBRARY_MODULES_V10,
    "merlin.llvmlower.spatial_scratch_abi",
    "merlin.llvmlower.fresh_tensor_writer",
    "merlin.llvmlower.private_workspace_pool",
    "merlin.llvmlower.borrowed_effects",
    "merlin.xdsl_dialects",
    "merlin.xdsl_dialects._common",
    "merlin.xdsl_dialects.fp8",
)
# Executable schedule dialect consumer; older inventories remain unchanged.
LIBRARY_MODULES_V12 = (
    *LIBRARY_MODULES_V11,
    "merlin.compile.scheduling.schedule_ir",
    "merlin.xdsl_dialects.schedule",
    "merlin.xdsl_dialects.contract",
)
LIBRARY_MODULES_V13 = (
    *LIBRARY_MODULES_V12,
    "merlin.compile.scheduling.loop_schedule",
    "merlin.sched.primitives",
)
LIBRARY_MODULES_V14 = (
    *LIBRARY_MODULES_V13,
    "merlin.compile.scheduling.dense_panel_pipeline",
    "merlin.compile.scheduling.dense_panel_contracts",
    "merlin.compile.scheduling.dense_panel_ir",
    "merlin.compile.scheduling.dense_retained",
    "merlin.compile.scheduling.scf_commands",
    "merlin.xdsl_dialects.runtime",
)
LIBRARY_MODULES_V15 = (
    *LIBRARY_MODULES_V14,
    "merlin.compile.scheduling.configuration_contexts",
    "merlin.compile.scheduling.configuration_loop_cleanup",
    "merlin.compile.scheduling.dense_panel_readout",
)
LIBRARY_MODULES_V16 = (
    *LIBRARY_MODULES_V15,
    "merlin.compile.scheduling.tile_load_coalescing",
    "merlin.compile.scheduling.panel_tile_loads",
    "merlin.compile.scheduling.dense_segmented_pipeline",
    "merlin.compile.scheduling.dense_array_retention",
    "merlin.compile.scheduling.api_catalog",
)
LIBRARY_MODULES_V17 = (
    *LIBRARY_MODULES_V16,
    "merlin.compile.scheduling.rolling_window",
    "merlin.compile.scheduling.rolling_spatial",
    "merlin.xdsl_dialects.rolling_window",
    "merlin.compile.scheduling.spatial_readout_cache",
)
LIBRARY_MODULES_V18 = (
    *LIBRARY_MODULES_V17,
    "merlin.compile.scheduling.dense_readout_overlap",
    "merlin.llvmlower.host_builder",
    "merlin.llvmlower.llvm_loop_metadata",
    "merlin.compile.scheduling.projection_views",
    "merlin.transforms.tensor_views",
    "merlin.llvmlower.source_region",
    "merlin.llvmlower.source_region_host",
    "merlin.compile.composition_catalog",
)
LIBRARY_MODULES_V19 = (*LIBRARY_MODULES_V18, "merlin.compile.scheduling.dense_ready_readout")
LIBRARY_MODULES_V20 = (
    *LIBRARY_MODULES_V19,
    "merlin.sched.ir.events",
    "merlin.compile.scheduling.scf_carried_events",
    "merlin.sched.primitives.event_loops",
    # Source-region preparation already uses these generic compiler owners.
    # Close their current imports explicitly instead of borrowing an installed
    # full compiler or widening a namespace at import time.
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
    "merlin.llvmlower.passes_xdsl",
    "merlin.llvmlower.source_fma_batch",
    "merlin.llvmlower.toolchain",
    "merlin.runtime",
    "merlin.runtime.host_arithmetic",
    "merlin.xdsl_dialects.ir_inspection",
    "merlin.xdsl_dialects.lowering",
    "merlin.xdsl_dialects.lowering.outline",
)
# Prospective combined release; preceding inventories retain their membership.
LIBRARY_MODULES_V21 = (
    *LIBRARY_MODULES_V20,
    "merlin.common.stat_memo",
    "merlin.sched.primitives.loop_handles",
    "merlin.compile.scheduling.completion_streams",
    "merlin.compile.scheduling.configuration_motion",
    "merlin.compile.scheduling.spatial_max_stripes",
)
LIBRARY_MODULES_V22 = (
    *LIBRARY_MODULES_V21,
    "merlin.compile.logical_abi",
    "merlin.compile.scheduling.convolution_exploration",
    "merlin.compile.scheduling.convolution_choices",
    "merlin.compile.scheduling.dense_rhs_residency",
    "merlin.transforms.pipeline_policy",
    "merlin.transforms.pass_exploration",
    "merlin.transforms.evaluated_exploration",
    "merlin.transforms.shape_policy",
    "merlin.compile.scheduling.dense_panel_epochs",
)
LIBRARY_MODULES_V23 = (
    *LIBRARY_MODULES_V22,
    "merlin.compile.scheduling.pointwise_convolution",
    "merlin.compile.scheduling.affine_gather_runs",
    "merlin.compile.scheduling.batched_dense",
    "merlin.compile.scheduling.convolution_output_groups",
    "merlin.compile.scheduling.loaded_sum_epochs",
    "merlin.compile.tuning_knobs",
    "merlin.transforms.tuning_selection",
)
LIBRARY_MODULES_V24 = (
    *LIBRARY_MODULES_V23,
    "merlin.transforms.dense_reduction_layout",
    "merlin.llvmlower.layout_propagation",
    "merlin.compile.scheduling.convolution_input_reuse",
    "merlin.compile.scheduling.tensor_storage",
    "merlin.perf",
    "merlin.perf.factored_storage",
    "merlin.llvmlower.constant_panel_layout",
    "merlin.compile.scheduling.dense_loaded_sum_composition",
    "merlin.compile.scheduling.dense_loaded_sum_groups",
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
    "merlin.compile.scheduling.projection_dense",
    "merlin.compile.scheduling.projection_provenance",
    "merlin.compile.scheduling.rolling_lookahead",
    "merlin.compile.scheduling.convolution_input_first_use",
    "merlin.compile.scheduling.operand_slots",
    "merlin.compile.scheduling.compact_dense_group",
    "merlin.compile.scheduling.compact_dense_group_check",
    "merlin.compile.scheduling.dense_shared_rhs",
    "merlin.compile.scheduling.dense_shared_rhs_check",
    "merlin.compile.scheduling.shared_rhs_schedule_ir",
    "merlin.compile.scheduling.horizontal_convolution",
    # Early source preparation must retain its actual tensor/view and caller
    # checks in the closed SDK, rather than importing a live full checkout.
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
)
LIBRARY_MODULES = LIBRARY_MODULES_V6

# Active refactor release: preserve every earlier ordered roster. These are
# implementation dependencies of the existing V24 public APIs, not new grants.
LIBRARY_MODULES_V25 = (
    *LIBRARY_MODULES_V24,
    "merlin.compile.analysis",
    "merlin.compile.analysis.contraction_shapes",
    "merlin.compiler",
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
)


def canonical_module(name):
    """Map the reviewed legacy ownership names to the explicit fresh SDK owners."""
    renamed = {
        "merlin.verify": "merlin.verification",
        "merlin.verify.epilogue_applicability": "merlin.verification.epilogue_applicability",
        "merlin.compile": "merlin.compiler",
        "merlin.compile.analysis": "merlin.analysis",
        "merlin.compile.analysis.contraction_shapes": "merlin.compiler.ir.contraction_shapes",
        "merlin.compile.logical_abi": "merlin.compiler.ir.logical_abi",
        "merlin.compile.composition_catalog": "merlin.compiler.passes.composition",
        "merlin.compile.tuning_knobs": "merlin.compiler.selection.knobs",
        "merlin.compile.capacity": "merlin.compiler.selection.capacity",
        "merlin.compile.baremetal_model": "merlin.model",
        "merlin.compile.model_execution_inputs": "merlin.runtime.modelinputs",
        "merlin.compile.materialized_model_inputs": "merlin.compiler.frontend.materialized",
        "merlin.compile.route_before_build": "merlin.compiler.planning.model",
        "merlin.compile.complete_group_routing": "merlin.compiler.planning.complete",
        "merlin.compile.model_coverage": "merlin.compiler.planning.coverage",
        "merlin.compile.model_preflight": "merlin.compiler.frontend.preflight",
        "merlin.compile.model_interface": "merlin.compiler.partition.model",
        "merlin.compile.owned_host_model": "merlin.compiler.build.owned",
        "merlin.compile.owned_shared_source": "merlin.compiler.build.siblings",
        "merlin.compile.owned_spatial_source": "merlin.compiler.build.spatial",
        "merlin.compile.completed_model_link": "merlin.verification.completed_model",
        "merlin.compile.scalar_host_qualification": "merlin.verification.scalar_host",
        "merlin.compile.scheduling.block_schedule": "merlin.compiler.schedule.blocks",
        "merlin.compile.connected_regions": "merlin.compiler.lower.connected",
        "merlin.compile.connected_runtime": "merlin.runtime.connected",
        "merlin.compile.connected_host": "merlin.compiler.lower.connectedhost",
        "merlin.compile.fusion_discovery": "merlin.compiler.planning.frontiers",
        "merlin.compile.fusion_workflow": "merlin.compiler.planning.explore",
        "merlin.compile.compiler_observations": "merlin.compiler.observations",
        "merlin.llvmlower.passes_xdsl": "merlin.compiler.frontend.preprocess",
    }
    if name in renamed:
        return renamed[name]
    if name.startswith("merlin.compile.analysis."):
        return name.replace("merlin.compile.analysis", "merlin.analysis", 1)
    return name.replace("merlin.compile.scheduling", "merlin.compiler.schedule", 1)


# A new source release, not a reinterpretation of any earlier SDK selection.
# The additional files are the implementation closure of the same public APIs.
LIBRARY_MODULES_V26 = tuple(dict.fromkeys((
    *(canonical_module(name) for name in LIBRARY_MODULES_V25),
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
)))


def library_modules(version=26):
    """Select a closed reviewed inventory; this grants no import access."""
    if type(version) is not int or version not in (
        6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
    ):
        raise ValueError("shared compiler inventory requires version 6 through 26")
    return {
        6: LIBRARY_MODULES_V6,
        7: LIBRARY_MODULES_V7,
        8: LIBRARY_MODULES_V8,
        9: LIBRARY_MODULES_V9,
        10: LIBRARY_MODULES_V10,
        11: LIBRARY_MODULES_V11,
        12: LIBRARY_MODULES_V12,
        13: LIBRARY_MODULES_V13,
        14: LIBRARY_MODULES_V14,
        15: LIBRARY_MODULES_V15,
        16: LIBRARY_MODULES_V16,
        17: LIBRARY_MODULES_V17,
        18: LIBRARY_MODULES_V18,
        19: LIBRARY_MODULES_V19,
        20: LIBRARY_MODULES_V20,
        21: LIBRARY_MODULES_V21,
        22: LIBRARY_MODULES_V22,
        23: LIBRARY_MODULES_V23,
        24: LIBRARY_MODULES_V24,
        25: LIBRARY_MODULES_V25,
        26: LIBRARY_MODULES_V26,
    }[version]


def shared_library_sources(*, version=26):
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
                    _record_admission(api, False, premise[:ADMISSION_TEXT], _admission_route(fallback_route, args),
                                      args, kwargs)
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
            document = json.loads(line[len(ADMISSION_PREFIX):])
        except ValueError:
            continue
        if type(document) is not dict or document.get("schema") != ADMISSION_SCHEMA:
            continue
        if type(document.get("summary")) is dict:
            counts = {str(k): v for k, v in document["summary"].items() if type(v) is int and v > 0}
            dropped = document.get("dropped") if type(document.get("dropped")) is int else 0
            continue
        if (type(document.get("id")) is not int or type(document.get("api")) is not str
                or type(document.get("admitted")) is not bool or len(records) >= limit):
            continue
        premise, route = document.get("failed_premise"), document.get("fallback_route")
        shape = document.get("shape")
        partial = document.get("partial_refusals")
        records[document["id"]] = {
            "api": document["api"][:120],
            "admitted": document["admitted"],
            "failed_premise": premise[:ADMISSION_TEXT] if type(premise) is str else None,
            "fallback_route": route[:ADMISSION_TEXT] if type(route) is str else None,
            "shape": ({k: shape[k] for k in ("m", "k", "n") if type(shape.get(k)) is int}
                      if type(shape) is dict else None),
            "partial_refusals": ([str(v)[:ADMISSION_TEXT] for v in partial[:4]]
                                 if type(partial) is list and partial else None),
        }
    rows = []
    for identifier, record in sorted(records.items()):
        rows.append(dict(record, count=counts.get(str(identifier), 1)))
    return rows, dropped
