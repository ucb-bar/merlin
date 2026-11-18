"""Explicit shared compiler source owners; no discovery or import grants.

The export writer and ordinary pass receipts consume this same roster. Adding
an owner requires review of its import/resource closure and authoring boundary.
This roster does not attest loaded code or external dependency implementations.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

# ---- library_sources ---------------------------------------------------------------------------

# The single reviewed shared-compiler SDK release. Earlier releases are retired:
# their frozen runs keep their archived environment, and no current entry point
# reinterprets current sources as one of them.
CURRENT_SDK_VERSION = 30
CURRENT_LIBRARY_MODULES = (
    "merlin.compiler.backend",
    "merlin.compiler.records",
    "merlin.compiler.backend.protocol",
    "merlin.compiler.backend.interface",
    "merlin.compiler.ir.interface_contract",
    "merlin.compiler.ir.readout",
    "merlin.compiler.lower.quant.readout_match",
    "merlin.compiler.lower.interface_materialization",
    "merlin.compiler.dialects.interface",
    "merlin.compiler.dialects.visibility",
    "merlin.compiler.schedule",
    "merlin.compiler.transforms",
    "merlin.compiler.dialects",
    "merlin.runtime",
    "merlin.compiler.dialects.lowering",
    "merlin.development.targetgen.oot_starterkit",
    "merlin.analysis.perf",
    "merlin",
    "merlin.compiler.frontend",
    "merlin.compiler.quantization",
    "merlin.compiler.lower",
    "merlin.compiler",
    "merlin.compiler.ir",
    "merlin.compiler.schedule.shared_operands",
    "merlin.compiler.schedule.panels",
    "merlin.compiler.schedule.storage",
    "merlin.compiler.schedule.loops",
    "merlin.compiler.schedule.dense",
    "merlin.compiler.schedule.spatial",
    "merlin.compiler.schedule.elementwise",
    "merlin.compiler.schedule.pipeline",
    "merlin.compiler.schedule.convolution",
    "merlin.compiler.lower.options",
    "merlin.compiler.kernel_ir",
    "merlin.compiler.semantic",
    "merlin.compiler.semantic.search",
    "merlin.development",
    "merlin.target",
    "merlin.target.isa",
    "merlin.compiler.transforms.dense",
    "merlin.verification",
    "merlin.verification.references",
    "merlin.compiler.lower.quant",
    "merlin.compiler.lower.host",
    "merlin.compiler.lower.source",
    "merlin.common",
    "merlin.compiler.lower.model",
    "merlin.compiler.partition",
    "merlin.compiler.transforms.exploration",
    "merlin.compiler.selection",
    "merlin.analysis",
    "merlin.analysis.programs",
    "merlin.compiler.lower.device",
    "merlin.development.targetgen",
    "merlin.capture",
    "merlin.compiler.lower.c",
    "merlin.compiler.schedule.nests",
    "merlin.compiler.ir.xdsl_io",
    "merlin.compiler.schedule.blocks",
    "merlin.compiler.schedule.shared_operands.shared_inputs",
    "merlin.compiler.schedule.panels.transfers",
    "merlin.compiler.schedule.storage.buffer_pipelines",
    "merlin.compiler.schedule.storage.periodic_effects",
    "merlin.compiler.schedule.loops.configuration_loops",
    "merlin.compiler.schedule.loops.command_compaction",
    "merlin.compiler.schedule.dense.compact",
    "merlin.compiler.schedule.dense.loaded_sums",
    "merlin.compiler.schedule.spatial.max_stripes",
    "merlin.compiler.schedule.dense.verification",
    "merlin.compiler.schedule.loops.configuration",
    "merlin.compiler.schedule.loops.configuration_ir",
    "merlin.compiler.schedule.storage.temporaries",
    "merlin.compiler.schedule.dense.contracts",
    "merlin.compiler.schedule.dense.source",
    "merlin.compiler.schedule.storage.spills",
    "merlin.compiler.schedule.elementwise.fusion",
    "merlin.compiler.schedule.pipeline.fusion_contracts",
    "merlin.compiler.schedule.library_sources",
    "merlin.compiler.schedule.loops.counted_loops",
    "merlin.compiler.schedule.elementwise.pointwise_source",
    "merlin.compiler.schedule.pipeline.program_contracts",
    "merlin.compiler.schedule.pipeline.program_limits",
    "merlin.compiler.schedule.pipeline.row_contracts",
    "merlin.compiler.schedule.elementwise.row_softmax_checks",
    "merlin.compiler.schedule.elementwise.row_softmax",
    "merlin.compiler.schedule.pipeline.stages",
    "merlin.compiler.schedule.pipeline.selected_contracts",
    "merlin.compiler.schedule.pipeline.spill_contracts",
    "merlin.compiler.schedule.pipeline.storage_contracts",
    "merlin.compiler.schedule.dense.staged",
    "merlin.compiler.schedule.convolution.windows",
    "merlin.compiler.lower.options.nonlinear",
    "merlin.compiler.kernel_ir.codegen",
    "merlin.compiler.kernel_ir.ir",
    "merlin.compiler.kernel_ir.parse",
    "merlin.compiler.semantic.search.allocation",
    "merlin.compiler.semantic.allocation_types",
    "merlin.compiler.semantic.search.extraction",
    "merlin.compiler.semantic.model",
    "merlin.compiler.semantic.rules",
    "merlin.target.isa.dialect_generation",
    "merlin.target.isa.spec",
    "merlin.compiler.transforms.certificates",
    "merlin.compiler.transforms.dense.ir",
    "merlin.compiler.transforms.dense.planning",
    "merlin.compiler.transforms.structural_rewrites",
    "merlin.compiler.transforms.pipeline_state",
    "merlin.verification.references.epilogue_applicability",
    "merlin.compiler.schedule.spatial.producer",
    "merlin.compiler.schedule.spatial.readouts",
    "merlin.compiler.lower.quant.readout",
    "merlin.compiler.lower.quant.requantization",
    "merlin.compiler.lower.host.storage",
    "merlin.compiler.lower.host.writers",
    "merlin.compiler.lower.host.borrowed_effects",
    "merlin.compiler.dialects.xdsl_context",
    "merlin.compiler.schedule.dense.plans",
    "merlin.compiler.dialects.schedule",
    "merlin.compiler.dialects.contract",
    "merlin.compiler.kernel_ir.primitives",
    "merlin.compiler.schedule.panels.pipeline",
    "merlin.compiler.schedule.panels.construction",
    "merlin.compiler.schedule.panels.retained",
    "merlin.compiler.schedule.loops.scf_commands",
    "merlin.compiler.dialects.runtime",
    "merlin.compiler.schedule.loops.configuration_contexts",
    "merlin.compiler.schedule.panels.readouts",
    "merlin.compiler.schedule.panels.load_coalescing",
    "merlin.compiler.schedule.panels.tile_loads",
    "merlin.compiler.schedule.panels.segmented_pipeline",
    "merlin.compiler.schedule.panels.array_retention",
    "merlin.compiler.schedule.api_catalog",
    "merlin.compiler.schedule.spatial.rolling_window",
    "merlin.compiler.lower.host.builder",
    "merlin.compiler.schedule.shared_operands.projections",
    "merlin.compiler.transforms.canonicalization",
    "merlin.compiler.lower.source.region",
    "merlin.compiler.schedule.panels.ready_readout",
    "merlin.compiler.schedule.panels.ready_schedule",
    "merlin.compiler.schedule.loops.completion_events",
    "merlin.compiler.kernel_ir.loops",
    "merlin.common.access",
    "merlin.common.source_aliases",
    "merlin.compiler.ir.compile_trace",
    "merlin.compiler.ir.lowering_audit",
    "merlin.common.paths",
    "merlin.common.provenance",
    "merlin.common.provenance_claims",
    "merlin.common.digest",
    "merlin.compiler.frontend.linalg_mlir",
    "merlin.common.external_tools",
    "merlin.compiler.lower.model.generic_form",
    "merlin.compiler.lower.options.optional_passes",
    "merlin.compiler.frontend.preprocess",
    "merlin.compiler.lower.source.dense",
    "merlin.compiler.lower.model.toolchain",
    "merlin.compiler.transforms.host_arithmetic",
    "merlin.compiler.partition.outline",
    "merlin.compiler.ir.logical_abi",
    "merlin.compiler.schedule.convolution.exploration",
    "merlin.compiler.schedule.convolution.choices",
    "merlin.compiler.schedule.shared_operands.shared_rhs",
    "merlin.compiler.transforms.exploration.branch_policy",
    "merlin.compiler.transforms.exploration.pass_orders",
    "merlin.compiler.transforms.exploration.cycle_feedback",
    "merlin.compiler.transforms.exploration.shape_assessment",
    "merlin.compiler.transforms.exploration.shape_proposals",
    "merlin.compiler.schedule.panels.epochs",
    "merlin.compiler.schedule.convolution.dense_views",
    "merlin.compiler.schedule.convolution.input_rows",
    "merlin.compiler.schedule.dense.batched",
    "merlin.compiler.schedule.convolution.output_groups",
    "merlin.compiler.selection.knobs",
    "merlin.compiler.transforms.dense.row_batching",
    "merlin.compiler.lower.host.propagation",
    "merlin.compiler.schedule.convolution.input_residency",
    "merlin.analysis.programs.storage_encoding",
    "merlin.compiler.lower.host.layout",
    "merlin.compiler.schedule.panels.loaded_sums",
    "merlin.compiler.lower.device.outline",
    "merlin.compiler.ir.llvm_context",
    "merlin.compiler.lower.quant.late_quant_rne",
    "merlin.compiler.lower.source.interval",
    "merlin.compiler.lower.source.interval_llvm",
    "merlin.compiler.lower.source.finite",
    "merlin.compiler.schedule.spatial.rolling_lookahead",
    "merlin.compiler.schedule.shared_operands.shared_lhs",
    "merlin.compiler.schedule.shared_operands.shared_lhs_checks",
    "merlin.compiler.schedule.shared_operands.shared_rhs_checks",
    "merlin.compiler.ir.mlir_query",
    "merlin.common.record_formats",
    "merlin.capture.safetensors",
    "merlin.compiler.lower.source.repeated_pointwise",
    "merlin.compiler.lower.source.row_slice",
    "merlin.compiler.lower.source.dense_group",
    "merlin.compiler.lower.options.weight_prepack_runtime",
    "merlin.compiler.lower.c.weights",
    "merlin.compiler.lower.source.prepared",
    "merlin.compiler.lower.source.ordered_fma",
    "merlin.compiler.transforms.dense.head_batching",
    "merlin.compiler.transforms.dense.row_batching_check",
    "merlin.compiler.transforms.dense.shared_rhs_views",
    "merlin.compiler.ir.contraction_shapes",
    "merlin.compiler.ir.boundary",
    "merlin.compiler.lower.device.offload",
    "merlin.compiler.semantic.search.allocation_schedule",
    "merlin.compiler.semantic.instruction",
    "merlin.compiler.semantic.view_descriptors",
    "merlin.compiler.schedule.nests.model",
    "merlin.compiler.schedule.nests.construction",
    "merlin.compiler.schedule.nests.execution",
    "merlin.compiler.schedule.dense.capabilities",
    "merlin.compiler.schedule.spatial.capabilities",
    "merlin.compiler.schedule.catalog_source",
    "merlin.compiler.schedule.catalog_effects",
    "merlin.compiler.frontend.repairs",
    "merlin.compiler.ir.provenance",
    "merlin.compiler.quantization.linalg",
    "merlin.compiler.schedule.convolution.tile_transfers",
    "merlin.compiler.schedule.convolution.storage",
    "merlin.compiler.schedule.convolution.realization",
    "merlin.compiler.schedule.panels.verification",
    "merlin.compiler.schedule.pipeline.contracts",
    "merlin.compiler.schedule.pipeline.actions",
    "merlin.compiler.build",
    "merlin.compiler.ir.static_cfg",
    "merlin.compiler.transforms.radix_product",
    "merlin.compiler.lower.quant.guarded_mean",
    "merlin.compiler.lower.quant.enclosed_readout",
    "merlin.compiler.lower.quant.affine_joint",
    "merlin.compiler.lower.quant.affine_rectifier",
    "merlin.compiler.lower.quant.affine_domain",
    "merlin.compiler.lower.source.integer_domains",
    "merlin.compiler.lower.source.shared_permutation",
    "merlin.compiler.lower.source.segmented_inputs",
    "merlin.compiler.frontend.quantized",
    "merlin.compiler.partition.callback_regions",
    "merlin.compiler.build.host_regions",
    "merlin.compiler.build.llvm_bundle",
    "merlin.compiler.build._host_regions_worker",
    "merlin.compiler.lower.quant.affine_pair",
    "merlin.common.target_data",
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
