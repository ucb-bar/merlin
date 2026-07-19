"""The CCA ⇄ CompilerAction bijection contract — the single, machine-checkable link between what the
Common Compute Abstraction *captures* and what the compiler *exposes*.

The framework rests on one invariant, scoped per backend:

1. **Capture-completeness** — every CCA facet field is CLASSIFIED (``IDENTITY`` / ``LEVER`` /
   ``BACKEND_STUB``). Adding a field to ``cca.py`` without a row here is an error: you must say what
   kind of thing it is. (So the CCA can never silently grow a field nobody reasoned about.)
2. **Exposure-completeness** — for a backend, the set of ``LEVER`` axes equals the set of axes
   ``action_catalog`` actually routes to a compiler seam. A lever field with no route (the abstraction
   promises something the compiler can't change) or a routed axis with no field (the compiler exposes
   something the abstraction doesn't capture) is a bijection break.

The classification lives in ``FIELD_REGISTRY`` (hand-authored, reviewed source of truth). ``schema_axes``
reflects the real ``cca.py`` dataclasses and ``routed_axes`` reflects the real ``action_catalog._ROUTES``
— nothing is hardcoded, so the two can't drift apart unnoticed. ``check_bijection`` diffs them.

Multiple routes per axis are legal and ARE the escalation ladder (e.g. ``accumulator_resident``:
PASS → CODEGEN); the contract only requires each such ladder to use DISTINCT action classes.

Known-open gaps (a lever field awaiting its route, or a route awaiting its backing field) are tracked
explicitly in ``KNOWN_OPEN`` with the reason and the workstream that closes them, so the enforcing test
stays GREEN while the roadmap shrinks that set to empty — at which point the invariant is fully closed.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import action_catalog
from .cca import (ComputeFacet, DataflowFacet, MemoryFacet, EnvelopeFacet, SpatialFacet,
                  VectorFacet)

# facet name (the axis prefix) -> the dataclass whose fields it exposes.
FACET_CLASSES = {
    "compute": ComputeFacet,
    "vector": VectorFacet,
    "memory": MemoryFacet,
    "envelope": EnvelopeFacet,
    "spatial": SpatialFacet,
    "dataflow": DataflowFacet,
}

IDENTITY = "IDENTITY"          # names the region, not a thing we change (the op key)
LEVER = "LEVER"                # a compute property that MUST map to an exposed compiler modification
BACKEND_STUB = "BACKEND_STUB"  # a facet for a backend whose lifters/routes are not yet instantiated
METRIC = "METRIC"              # a measured performance-diagnostic outcome (feeds gap_analysis), NOT a
#                                standalone compiler lever — excluded from the LEVER<->route bijection.


@dataclass(frozen=True)
class FieldSpec:
    axis: str                       # "compute.contraction_form"
    classification: str             # IDENTITY | LEVER | BACKEND_STUB | METRIC
    backends: tuple[str, ...]       # the backends for which this axis is meaningful
    note: str = ""


# One row per facet.field. Reviewed by hand — this is where "we decided this property matters and how".
# RVV is the first fully-instantiated backend; spatial (gemmini) / dataflow (npu) fields are stubs until
# their lifters + routes land, so they are BACKEND_STUB and excluded from the RVV bijection.
FIELD_REGISTRY: dict[str, FieldSpec] = {
    # --- compute (target-agnostic) ---
    "compute.op": FieldSpec("compute.op", IDENTITY, ("rvv", "gemmini", "npu"),
                            "the operation being computed — the join key, not a lever"),
    "compute.contraction_form": FieldSpec("compute.contraction_form", LEVER, ("rvv",),
                                          "fused_fma vs mul_add -> PASS fused_vfmacc_contraction"),
    "compute.accumulator_dtype": FieldSpec("compute.accumulator_dtype", LEVER, ("rvv",),
                                           "accumulate width (i32/f32) -> KNOB dtype_strategy"),
    "compute.widening": FieldSpec("compute.widening", LEVER, ("rvv",),
                                  "i8xi8->i32 widening MAC -> KNOB dtype_strategy=int8_w8a8"),
    "compute.reduction_form": FieldSpec("compute.reduction_form", LEVER, ("rvv",),
                                        "tree/vredsum reduction -> HEURISTIC lower_multi_reduction"),
    "compute.register_block": FieldSpec("compute.register_block", LEVER, ("rvv",),
                                        "MRxNR register block -> KNOB register-block MR"),
    "compute.epilogue": FieldSpec("compute.epilogue", LEVER, ("rvv",),
                                  "requant/narrow fused into store -> PASS fuse-requant-narrowing-store"),
    "compute.accumulator_resident": FieldSpec("compute.accumulator_resident", LEVER, ("rvv",),
                                              "accumulator stays in vregs across K -> PASS/CODEGEN ladder"),
    "compute.nr_is_vsetvlmax": FieldSpec("compute.nr_is_vsetvlmax", LEVER, ("rvv",),
                                         "VL-adaptive output width -> HEURISTIC NR=vsetvlmax"),
    "compute.activation_vectorization": FieldSpec("compute.activation_vectorization", LEVER, ("rvv",),
                                                  "vectorized poly vs scalar libm -> PASS "
                                                  "vectorized_transcendental_activation"),
    # --- vector (RVV/SIMD) ---
    "vector.sew": FieldSpec("vector.sew", LEVER, ("rvv",),
                            "element width -> KNOB dtype_strategy (sew follows element dtype)"),
    "vector.lmul": FieldSpec("vector.lmul", LEVER, ("rvv",),
                             "register grouping -> KNOB vector_sizes (widen N to raise LMUL)"),
    "vector.vl_strategy": FieldSpec("vector.vl_strategy", LEVER, ("rvv",),
                                    "vsetvl loop vs fixed vsetivli -> PASS vl-polymorphic-tail"),
    "vector.tail": FieldSpec("vector.tail", LEVER, ("rvv",),
                             "ta/tu tail policy -> KNOB tail_policy / PASS vl-polymorphic-tail (tu)"),
    # --- memory (data-movement / packing — the #1 expert GEMM lever, lifted from decode.memory) ---
    "memory.access_pattern": FieldSpec("memory.access_pattern", LEVER, ("rvv",),
                                       "packed unit-stride panel vs strided model-layout gather -> "
                                       "PASS operand packing (vfmacc_packed / a layout-assignment pass)"),
    "memory.panel_reuse": FieldSpec("memory.panel_reuse", METRIC, ("rvv",),
                                    "loads/FMA amortization — a diagnostic outcome of register-block + "
                                    "packing, not a standalone lever"),
    "memory.a_broadcast_vf": FieldSpec("memory.a_broadcast_vf", METRIC, ("rvv",),
                                       "A streamed via vfmacc.vf (no rebuild ladder) — a diagnostic "
                                       "outcome of instruction selection"),
    # --- region (the code AROUND the loop — where the measured expert gap actually lived) ---
    "envelope.calls_in_loop": FieldSpec("envelope.calls_in_loop", LEVER, ("rvv",),
                                      "a call inside a loop body is per-iteration overhead whatever "
                                      "it calls -> PASS eliminate-epilogue-copy"),
    "envelope.runtime_calls": FieldSpec("envelope.runtime_calls", LEVER, ("rvv",),
                                      "runtime escape (memrefCopy et al) instead of emitted code -> "
                                      "PASS/CODEGEN in-place accumulator writeback"),
    "envelope.work_ins_per_mac": FieldSpec("envelope.work_ins_per_mac", METRIC, ("rvv",),
                                         "N^3 coefficient — hot-loop efficiency, a diagnostic outcome"),
    "envelope.overhead_ins_per_output": FieldSpec("envelope.overhead_ins_per_output", METRIC, ("rvv",),
                                                "N^2 coefficient — per-tile overhead, a diagnostic "
                                                "outcome that SIZES the region gap"),
    # --- spatial (gemmini) — stubs until the spatial lifter + gemmini routes land ---
    "spatial.pe_rows": FieldSpec("spatial.pe_rows", BACKEND_STUB, ("gemmini",)),
    "spatial.pe_cols": FieldSpec("spatial.pe_cols", BACKEND_STUB, ("gemmini",)),
    "spatial.dataflow": FieldSpec("spatial.dataflow", BACKEND_STUB, ("gemmini",)),
    "spatial.accumulator_resident": FieldSpec("spatial.accumulator_resident", BACKEND_STUB, ("gemmini",)),
    # --- dataflow (npu) — stubs until the npu lifter + npu routes land ---
    "dataflow.engine_ops": FieldSpec("dataflow.engine_ops", BACKEND_STUB, ("npu",)),
    "dataflow.dma_pattern": FieldSpec("dataflow.dma_pattern", BACKEND_STUB, ("npu",)),
    "dataflow.onchip_resident": FieldSpec("dataflow.onchip_resident", BACKEND_STUB, ("npu",)),
}


# Gaps the roadmap has NOT closed yet, tracked explicitly so the enforcing test is a ratchet (GREEN now,
# fails on NEW drift) rather than a committed RED test. Each entry names the axis and WHY it is still
# open. WS-C Phase 2 removes these as the work lands; when both sets are empty the bijection is fully
# enforced. The three that remain are the genuinely-hard ones (no cheap/overfit close):
#   - compute.reduction_form: the baseline vectorizes only matmul/batch_matmul, so a softmax-sum/norm
#       reduction isn't vectorized at all -> closing it needs a NEW schedule feature that vectorizes the
#       reduction op AND lowers multi_reduction -> vredsum, spike-certified (real codegen, not a knob).
#   - vector.tail: CAPTURE done (lift_asm now records ta/tu); the route needs a NEW tail_policy schedule
#       seam (ta) + the tu masked-tail path shared with vl_strategy — a new seam, not a registration.
#   - compute.mr_adapts_to_m: whether the compiler clamps MR=min(MR,M) so a small-M matmul vectorizes is
#       a property of the TILING HEURISTIC given the shape M — NOT soundly liftable from a single kernel's
#       asm (lift_asm has no M/shape context; a matmul kernel's asm looks the same whether or not it would
#       adapt on a hypothetical small-M input). Captured from the schedule/mining side; a backing-field
#       asm lift is deferred rather than faked with an overfit heuristic.
#   - layout.transpose_materialized: the route (PASS impr_features:fuse_transpose_b) is real and
#       forkable NOW, and its whole-model win is MEASURED (openvla -6.5% on K1, cos 0.9999999). But the
#       divergence is a WHOLE-MODEL, cross-op GRAPH property — "a standalone linalg.transpose materializes
#       the matmul's B operand" — visible in the linalg IR, NOT in a single contraction kernel's asm that
#       lift_asm sees. The backing facet field needs a graph-level (IR) lifter, not the per-kernel asm
#       lifter; that lift is the deferred work-item (a new LayoutFacet), so the route is recorded here as
#       an orphan_route rather than faking a bare FieldSpec no facet populates.
KNOWN_OPEN: dict[str, dict[str, tuple[str, ...]]] = {
    "rvv": {
        # LEVER fields still awaiting a route (need a NEW schedule seam — see the per-axis notes above).
        # CLOSED so far: compute.accumulator_dtype + vector.sew (both -> KNOB dtype_strategy);
        # memory.access_pattern (BB1b -> PASS impr_features:vfmacc_packed operand pre-packing).
        "orphan_fields": (
            "compute.reduction_form",
            "vector.tail",
        ),
        # routed axes still awaiting a backing ComputeFacet field.
        # CLOSED so far: compute.activation_vectorization (field + lift_asm inferer added).
        "orphan_routes": (
            "compute.mr_adapts_to_m",
            "layout.transpose_materialized",   # fuse_transpose_b: route + measured win real; graph-level
                                               # (IR) backing-field lifter deferred (see note above).
        ),
    },
}


@dataclass
class BijectionReport:
    backend: str
    orphan_fields: list[str] = field(default_factory=list)     # LEVER axis with no route
    orphan_routes: list[str] = field(default_factory=list)     # routed axis that is not a LEVER field
    unclassified: list[str] = field(default_factory=list)      # schema field absent from FIELD_REGISTRY
    ladder_errors: list[str] = field(default_factory=list)     # multi-route axis with a non-distinct ladder

    @property
    def clean(self) -> bool:
        return not (self.orphan_fields or self.orphan_routes or self.unclassified or self.ladder_errors)

    def unexpected(self) -> "BijectionReport":
        """The report minus the KNOWN_OPEN allowlist — what the ratchet test must find empty. ``unclassified``
        and ``ladder_errors`` are NEVER allowlisted (they are always hard errors)."""
        known = KNOWN_OPEN.get(self.backend, {})
        of = sorted(set(self.orphan_fields) - set(known.get("orphan_fields", ())))
        orr = sorted(set(self.orphan_routes) - set(known.get("orphan_routes", ())))
        return BijectionReport(self.backend, of, orr, list(self.unclassified), list(self.ladder_errors))


def schema_axes() -> set[str]:
    """Every ``facet.field`` axis reflected from the real ``cca.py`` dataclasses (never hardcoded)."""
    return {f"{fname}.{fld.name}"
            for fname, cls in FACET_CLASSES.items()
            for fld in dataclasses.fields(cls)}


def leverable_axes(backend: str) -> set[str]:
    """Axes classified LEVER for this backend — the properties that MUST map to a compiler seam."""
    return {s.axis for s in FIELD_REGISTRY.values()
            if s.classification == LEVER and backend in s.backends}


def _routes(backend: str) -> list:
    return action_catalog._ROUTES.get(backend, [])


def routed_axes(backend: str) -> set[str]:
    """Axes ``action_catalog`` actually routes to a compiler seam for this backend (reflected)."""
    return {r.axis for r in _routes(backend)}


def _ladder_errors(backend: str) -> list[str]:
    """Axes with multiple routes whose action classes are not distinct (a malformed escalation ladder)."""
    by_axis: dict[str, list[str]] = {}
    for r in _routes(backend):
        by_axis.setdefault(r.axis, []).append(r.action_class)
    errs = []
    for axis, classes in by_axis.items():
        if len(classes) > 1 and len(set(classes)) != len(classes):
            errs.append(f"{axis}: non-distinct ladder classes {classes}")
    return sorted(errs)


def check_bijection(backend: str) -> BijectionReport:
    """Diff what the CCA captures (LEVER axes) against what the compiler exposes (routed axes)."""
    schema = schema_axes()
    classified = set(FIELD_REGISTRY)
    unclassified = sorted(schema - classified)
    lever = leverable_axes(backend)
    routed = routed_axes(backend)
    return BijectionReport(
        backend=backend,
        orphan_fields=sorted(lever - routed),
        orphan_routes=sorted(routed - lever),
        unclassified=unclassified,
        ladder_errors=_ladder_errors(backend),
    )


def dump_contract(backend: str, path: str | Path, *, toolchain_version: dict[str, Any] | None = None) -> Path:
    """Write the versioned bijection contract as YAML (a regenerable artifact under out/). One row per
    LEVER axis: its route class(es), seam, and forkable-now status; plus the current known-open gaps."""
    import yaml

    routes_by_axis: dict[str, list] = {}
    for r in _routes(backend):
        routes_by_axis.setdefault(r.axis, []).append(r)

    from . import regions as _regions  # lazy: regions imports this module (avoid an import cycle)

    rows = []
    for axis in sorted(leverable_axes(backend) | routed_axes(backend)):
        spec = FIELD_REGISTRY.get(axis)
        rs = routes_by_axis.get(axis, [])
        region = _regions.region_for_axis(axis)
        rows.append({
            "axis": axis,
            "classification": spec.classification if spec else "UNBACKED_ROUTE",
            "region": region.key if region else None,   # the compiler region governing this axis (C3 taxonomy)
            "routes": [{"action_class": r.action_class, "target_seam": r.target_seam,
                        "forkable_now": r.forkable_now} for r in rs],
            "note": spec.note if spec else "",
        })

    report = check_bijection(backend)
    doc = {
        "backend": backend,
        "toolchain": toolchain_version or {},
        "bijection_clean": report.clean,
        "known_open": KNOWN_OPEN.get(backend, {}),
        "report": {"orphan_fields": report.orphan_fields, "orphan_routes": report.orphan_routes,
                   "unclassified": report.unclassified, "ladder_errors": report.ladder_errors},
        "axes": rows,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p
