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
from .cca import (ComputeFacet, CoverageFacet, DispatchFacet, LayoutFacet, MemoryFacet,
                  EnvelopeFacet, SimtFacet, SpatialFacet, VectorFacet)

# facet name (the axis prefix) -> the dataclass whose fields it exposes.
def _facet_classes() -> dict:
    """The facets a CCA actually has, DERIVED from the CCA rather than listed here.

    This was a hand-written dict, which made the capture-completeness check above unable to detect
    the thing it exists to detect: a facet missing from the list is not merely unclassified, it is
    invisible, so adding ``CommunicationFacet`` with seven unclassified fields left the whole suite
    green. A completeness check whose universe is hand-maintained can only ever confirm what someone
    already remembered.

    ``cca_compare._facet_names`` learned this first ("this list used to be a literal, which meant a
    newly added facet was silently never compared"); the same fix belongs here.
    """
    import dataclasses as _dc

    from merlin.kernels import cca as _cca

    by_type = {n: o for n, o in vars(_cca).items() if _dc.is_dataclass(o) and n.endswith("Facet")}
    out: dict = {}
    for fld in _dc.fields(_cca.CCA):
        for tname, cls in by_type.items():
            if tname in str(fld.type):
                out[fld.name] = cls
                break
    return out


#: Facet name -> facet class, derived. Kept as a module-level mapping because callers index it.
FACET_CLASSES = _facet_classes()

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
# RVV is the first fully-instantiated backend; gemmini (spatial) is being instantiated (Workstream A):
# its geometry is IDENTITY and its dataflow/accumulator-residency are LEVER (routes landing next). The
# dataflow (npu) fields remain BACKEND_STUB until their lifter + routes land.
#
# `backends` MIXES TWO VOCABULARIES, on purpose. An entry is either a concrete backend name ("rvv") —
# a DIRECT tag, unconditionally leverable for that backend — or a facet FAMILY ("compute", "spatial") —
# a family-indirect tag, leverable only for a backend that actually ROUTES the axis. See
# `leverable_axes`. The family tag is what keeps a target-agnostic property from having to name every
# target that has it: an axis like `compute.accumulator_resident` is meaningful for any compute
# endpoint, and enumerating "rvv", "gemmini", "opu", "radiance", "atlas", ... here would both rot on
# the next target and put target-name literals in library code that `check_no_target_name` forbids.
# The routed gate is what makes the family tag safe: a backend never inherits a family axis its
# hardware does not expose, because it can only inherit one it registered a route for.
FIELD_REGISTRY: dict[str, FieldSpec] = {
    # --- compute (target-agnostic) ---
    "compute.op": FieldSpec("compute.op", IDENTITY, ("rvv", "gemmini", "npu"),
                            "the operation being computed — the join key, not a lever"),
    "compute.contraction_form": FieldSpec("compute.contraction_form", LEVER, ("rvv", "compute"),
                                          "fused_fma vs mul_add -> PASS fused_vfmacc_contraction"),
    "compute.accumulator_dtype": FieldSpec("compute.accumulator_dtype", LEVER, ("rvv", "compute"),
                                           "accumulate width (i32/f32) -> KNOB dtype_strategy"),
    "compute.widening": FieldSpec("compute.widening", LEVER, ("rvv", "compute"),
                                  "i8xi8->i32 widening MAC -> KNOB dtype_strategy=int8_w8a8"),
    "compute.reduction_form": FieldSpec("compute.reduction_form", LEVER, ("rvv", "compute"),
                                        "tree/vredsum reduction -> HEURISTIC lower_multi_reduction"),
    "compute.register_block": FieldSpec("compute.register_block", LEVER, ("rvv", "compute"),
                                        "MRxNR register block -> KNOB register-block MR"),
    "compute.epilogue": FieldSpec("compute.epilogue", LEVER, ("rvv", "compute"),
                                  "requant/narrow fused into store -> PASS fuse-requant-narrowing-store"),
    "compute.accumulator_resident": FieldSpec("compute.accumulator_resident", LEVER, ("rvv", "compute"),
                                              "accumulator stays in vregs across K -> PASS/CODEGEN ladder"),
    # NOT family-tagged, unlike its neighbours: `vsetvlmax` is an RVV instruction, so "is NR the
    # architectural maximum vector length" is a question only a vector-length-agnostic ISA can ask. A
    # fixed-tile matrix unit has no VL to be adaptive to. Leave it direct-tagged.
    "compute.nr_is_vsetvlmax": FieldSpec("compute.nr_is_vsetvlmax", LEVER, ("rvv",),
                                         "VL-adaptive output width -> HEURISTIC NR=vsetvlmax"),
    "compute.activation_vectorization": FieldSpec("compute.activation_vectorization", LEVER, ("rvv", "compute"),
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
    # --- spatial — the SYSTOLIC family (any target with a mesh, not one named target). Scoped by the
    # family tag "spatial": a concrete target counts these as its levers iff it registered spatial routes
    # (see `_target_families`). pe_rows/pe_cols are FIXED geometry (mesh DIM the compiler must target but
    # cannot change) -> IDENTITY, excluded from the bijection. dataflow (WS/OS) and accumulator-residency
    # ARE compiler choices -> LEVER; their routes are DERIVED per target by targetgen/rtl_backend from
    # the discovered mesh/accumulator.
    "spatial.pe_rows": FieldSpec("spatial.pe_rows", IDENTITY, ("spatial",),
                                 "fixed systolic-array rows (mesh DIM) — a target constant, not a lever"),
    "spatial.pe_cols": FieldSpec("spatial.pe_cols", IDENTITY, ("spatial",),
                                 "fixed systolic-array cols (mesh DIM) — a target constant, not a lever"),
    "spatial.dataflow": FieldSpec("spatial.dataflow", LEVER, ("spatial",),
                                  "WS vs OS dataflow selection (a discovered mesh implies this lever)"),
    "spatial.accumulator_resident": FieldSpec("spatial.accumulator_resident", LEVER, ("spatial",),
                                              "output stays PE/accumulator-resident across the reduction "
                                              "(a discovered accumulator memory implies this lever)"),
    # --- simt (a threads-of-control engine: muon/radiance) ---
    # Family-tagged "simt", like spatial.*, so any target with that engine picks these up when it
    # routes them -- and no target without one ever does. BACKEND_STUB until the lifter that fills
    # them and the routes that expose them land: a field nothing can populate and nothing can change
    # is not a lever, and calling it one would promise the loop an action that does not exist.
    "simt.warps": FieldSpec("simt.warps", BACKEND_STUB, ("simt",),
                            "warps cooperating on the region -> partition lever (routes pending)"),
    "simt.threads_per_warp": FieldSpec("simt.threads_per_warp", IDENTITY, ("simt",),
                                       "fixed warp width -- a target constant the compiler targets, "
                                       "not a choice (cf. spatial.pe_rows)"),
    "simt.smem_resident": FieldSpec("simt.smem_resident", BACKEND_STUB, ("simt",),
                                    "operand tile staged in shared memory across the reduction "
                                    "-> residency lever (routes pending)"),
    # PROMOTED out of BACKEND_STUB on the condition stated just above ("until the lifter that fills them
    # and the routes that expose them land"): the role lifter populates SimtFacet.barriers_in_loop from
    # the sync role it counts in the emitted stream, and targetgen/rtl_backend DERIVES the route + its
    # CODEGEN ladder for any endpoint whose own instruction table binds a sync role on a simt engine.
    # Both halves exist, so leaving it a stub made the derived route an ORPHAN and the bijection dirty
    # for every SIMT target -- the check reporting a gap in its own bookkeeping, not in the compiler.
    # Family-indirect like spatial.*: it counts as a lever only for a target that actually routes it.
    "simt.barriers_in_loop": FieldSpec("simt.barriers_in_loop", LEVER, ("simt",),
                                       "barriers inside the reduction loop -> hw-sync placement "
                                       "(an endpoint that binds a sync role implies this lever)"),
    "simt.divergence": FieldSpec("simt.divergence", BACKEND_STUB, ("simt",),
                                 "uniform vs divergent control flow -> predication/partition lever"),
    # --- dispatch: HOW the endpoint is driven (engine-AGNOSTIC: everything is driven by something) ---
    # The largest structural gap the CCA had. Gemmini's biggest expert win is dispatch SHAPE, not
    # arithmetic: a loop descriptor hands the whole nest to the endpoint's own sequencer, matmul_ws
    # elides re-preloads it does not need, and the scratchpad ids double-buffer across tiles. None of
    # that is expressible in contraction form, vector config or access pattern, so no divergence could
    # ever be raised for it -- and features/dispatch.py was already extracting the counts with nowhere
    # to put them.
    "dispatch.n_dispatches": FieldSpec(
        "dispatch.n_dispatches", METRIC, ("dispatch",),
        "METRIC: commands issued to the endpoint -- an outcome, not a knob"),
    "dispatch.config_fraction": FieldSpec(
        "dispatch.config_fraction", METRIC, ("dispatch",),
        "METRIC: share of dispatches that only set state -- diagnoses re-configuration overhead"),
    "dispatch.descriptor_reuse": FieldSpec(
        "dispatch.descriptor_reuse", LEVER, ("dispatch",),
        "endpoint state set once and inherited vs re-set per tile -> the preload-elision route the RTL "
        "backend derives for any endpoint whose table binds a config role"),
    "dispatch.loop_offloaded": FieldSpec(
        "dispatch.loop_offloaded", LEVER, ("dispatch",),
        "a loop nest handed to the endpoint's own sequencer -> derived for any endpoint whose table "
        "binds a loop_descriptor role; governed by the runtime-layout-encoding region"),
    "dispatch.double_buffered_banks": FieldSpec(
        "dispatch.double_buffered_banks", BACKEND_STUB, ("dispatch",),
        "on-chip banks alternated across tiles -> arena-plan lever (routes pending)"),
    "dispatch.dma_overlap": FieldSpec(
        "dispatch.dma_overlap", LEVER, ("dispatch",),
        "bulk movement issued to overlap with compute -> derived for any endpoint binding a dma role; "
        "governed by the hw-sync region"),
    "dispatch.dma_issue_to_wait": FieldSpec(
        "dispatch.dma_issue_to_wait", METRIC, ("dispatch",),
        "instructions between a DMA issue and the sync that waits for it (0 = strictly serial) -> the "
        "MAGNITUDE behind dispatch.dma_overlap's boolean, so a stream that overlaps by one instruction "
        "is distinguishable from one that overlaps a whole tile"),
    # --- communication: what crosses a boundary (dispatch/program scope) ---
    # ALL BACKEND_STUB. There is no lifter for this facet yet and therefore no route: a field that
    # cannot be routed is a stub, never a quiet LEVER. They are declared now, rather than when the
    # lifter lands, because the contract's job is to say what the vocabulary IS -- and because the
    # check that should have caught seven undeclared fields could not see the facet at all until
    # FACET_CLASSES stopped being a hand-written list.
    "communication.host_device_bytes": FieldSpec(
        "communication.host_device_bytes", BACKEND_STUB, ("communication",),
        "bytes crossing host<->device per invocation; None is NOT zero -- an unmeasured transfer is "
        "the one that surprises you"),
    "communication.engine_engine_bytes": FieldSpec(
        "communication.engine_engine_bytes", BACKEND_STUB, ("communication",),
        "bytes moved between engines of one target -- invisible to any single-engine facet"),
    "communication.mechanism": FieldSpec(
        "communication.mechanism", BACKEND_STUB, ("communication",),
        "how movement is performed (dma | simt | scalar_copy | mixed) -- a lever once a route exists"),
    "communication.intermediate_materialized": FieldSpec(
        "communication.intermediate_materialized", BACKEND_STUB, ("communication",),
        "an intermediate written to memory only to be read straight back by the next stage"),
    "communication.resident_across_calls": FieldSpec(
        "communication.resident_across_calls", BACKEND_STUB, ("communication",),
        "an operand kept resident across invocations instead of re-fetched"),
    "communication.copy_compute_overlap": FieldSpec(
        "communication.copy_compute_overlap", BACKEND_STUB, ("communication",),
        "movement issued so it overlaps the compute it feeds rather than serializing with it"),
    "communication.fences": FieldSpec(
        "communication.fences", BACKEND_STUB, ("communication",),
        "explicit ordering primitives on the boundary, counted"),
    # --- layout: how operands are laid out BEFORE the region runs ---
    # transpose_materialized is ALREADY routed in the action catalog with no facet behind it, so the
    # divergence that route exists to answer could never be raised: a route nothing can trigger.
    "layout.transpose_materialized": FieldSpec(
        "layout.transpose_materialized", LEVER, ("layout",),
        "a transpose written to memory vs folded into the access -> the route already registered in "
        "the action catalog, which until now had no field able to trigger it"),
    "layout.operand_major": FieldSpec(
        "layout.operand_major", LEVER, ("layout",),
        "k-major vs m/n-major operand packing -> derived for any endpoint binding a load role, routed to "
        "the target's own packing seam"),
    "layout.prepack_required": FieldSpec(
        "layout.prepack_required", IDENTITY, ("layout",),
        "the endpoint REQUIRES an offline-packed panel -- a property of the silicon, not a choice"),
    # --- memory: on-chip residency (engine-agnostic) ---
    # A compile-time FEASIBILITY predicate, not a performance hint: failing it produced 0/16384 correct
    # elements silently, and nothing in the CCA could say why.
    "memory.capacity_fit": FieldSpec(
        "memory.capacity_fit", LEVER, ("memory",),
        "does the working set fit the discovered on-chip capacity -> derived wherever an accumulator "
        "memory is discovered; overrunning it is not slow, it is silently wrong"),
    "memory.onchip_bytes_required": FieldSpec(
        "memory.onchip_bytes_required", METRIC, ("memory",),
        "METRIC: bytes the region's working set needs on chip"),
    "memory.banks_used": FieldSpec(
        "memory.banks_used", METRIC, ("memory",),
        "METRIC: distinct on-chip banks the region occupies -- a kernel using one bank of four leaves "
        "three idle, and no cycle count says so"),
    "memory.spill_reason": FieldSpec(
        "memory.spill_reason", METRIC, ("memory",),
        "METRIC: which operand class overran the capacity (operand/accumulator/both/none)"),
    "memory.dma_pattern": FieldSpec(
        "memory.dma_pattern", BACKEND_STUB, ("memory",),
        "burst/strided/scatter-gather bulk movement -> DMA-shape lever (routes pending); folded here "
        "from the retired dataflow facet, since movement is not an engine"),
    "memory.onchip_resident": FieldSpec(
        "memory.onchip_resident", BACKEND_STUB, ("memory",),
        "which operand stays on chip across the reduction -> residency lever (routes pending)"),
    # --- compute: the field behind a route that already existed ---
    "compute.mr_adapts_to_m": FieldSpec(
        "compute.mr_adapts_to_m", LEVER, ("compute",),
        "does the register block's M extent shrink for a small-M region -> the small-M route already "
        "registered in the action catalog, which had no field to trigger it"),
    # --- coverage (whole-model, target-agnostic) ---
    # These are GRAPH-level, which is the point: every other facet is lifted from ONE kernel's asm, so a
    # loss that lives in the graph -- an entire contraction class left unclaimed, or the ~88% of linalg
    # ops that are not contractions at all -- was structurally invisible to the CCA and therefore could
    # never be routed to an action. Measured: whisper_tiny claims only 65.9% of its MACs, and every
    # model (tiny_llama included) leaves ~86-89% of its linalg ops off the vectorized path.
    "coverage.claimed_mac_fraction": FieldSpec(
        "coverage.claimed_mac_fraction", LEVER, ("rvv",),
        "share of the model's MACs in contraction classes the schedule claims -> KNOB per-op-class "
        "block so a degenerate extent stops declaring a whole class scalar"),
    "coverage.unclaimed_op_classes": FieldSpec(
        "coverage.unclaimed_op_classes", LEVER, ("rvv",),
        "contraction classes left to convert-linalg-to-loops -> PASS per-op register block"),
    "coverage.non_contraction_op_fraction": FieldSpec(
        "coverage.non_contraction_op_fraction", LEVER, ("rvv",),
        "share of ops the contraction-only schedule never matches -> PASS vectorize the "
        "non-contraction generics (elementwise/layout/gather tail)"),
}


# Gaps the roadmap has NOT closed yet, tracked explicitly so the enforcing test is a ratchet (GREEN now,
# fails on NEW drift) rather than a committed RED test. Each entry names the axis and WHY it is still
# open. WS-C Phase 2 removes these as the work lands; when both sets are empty the bijection is fully
# enforced. The remaining ones are the genuinely-hard ones (no cheap/overfit close):
#   - compute.reduction_form: CLOSED. The vectorize_reduction PASS (impr_features) vectorizes the
#       standalone reduction (softmax/norm row-reduce, linalg.reduce) and lowers multi_reduction ->
#       vector.reduction -> a hardware horizontal reduce; PROVEN on emitted code (gen_reduce_f32/
#       gen_softmax_f32 decoded under -fno-vectorize: vfredusum.vs present, baseline emits none). The
#       route (action_catalog: compute.reduction_form -> impr_features:vectorize_reduction) now backs
#       the lever, so it is no longer an orphan_field.
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
            "vector.tail",
        ),
        # routed axes still awaiting a backing ComputeFacet field.
        # CLOSED so far: compute.activation_vectorization (field + lift_asm inferer added).
        # CLOSED: both of these were routes with no backing FIELD, so the divergence each route exists
        # to answer could never be raised -- an action wired to a question nothing could ask.
        # `layout.transpose_materialized` now has LayoutFacet behind it and `compute.mr_adapts_to_m` a
        # field on ComputeFacet, both populated by the role lifter.
        "orphan_routes": (),
    },
    # Per-target (example: gemmini): the two spatial LEVER axes (dataflow, accumulator-residency) are
    # backed by routes the generic, derivation-driven backend (targetgen/rtl_backend.py) DERIVES from the
    # discovered mesh/accumulator and registers into the agnostic core — no per-target content in the
    # core. With the backend registered the bijection is CLEAN (no allowlisted gaps); the remaining work
    # is to make those routes forkable_now (the target's OOT codegen threads the derived opts), a
    # forkable-status gap surfaced by seam_location/escalation_ladder, NOT a bijection break.
    "gemmini": {
        "orphan_fields": (),
        "orphan_routes": (),
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


def _target_families(backend: str) -> set[str]:
    """The facet FAMILIES a concrete target participates in — derived from the axes it registers routes
    for (a target that routes ``spatial.*`` axes is in the ``spatial`` family). This lets FIELD_REGISTRY
    tag a facet by family (``"spatial"``, any systolic target) rather than by a target name, so the
    classification stays target-agnostic while the bijection is still checked per concrete target."""
    return {r.axis.split(".", 1)[0] for r in _routes(backend)}


def leverable_axes(backend: str) -> set[str]:
    """Axes classified LEVER for this backend. A DIRECT classification (``backend in s.backends``, e.g.
    rvv's own registry list) is leverable unconditionally. A FAMILY-INDIRECT match (only via the facet
    family the target participates in, e.g. a ``spatial`` axis for a systolic target) is leverable only
    when the target's RTL actually ADMITS it — i.e. it registered a route for it — so a target does not
    inherit family axes its hardware lacks (e.g. atlas has a mesh but no accumulator memory, so
    ``spatial.accumulator_resident`` is NOT leverable for atlas and is not a phantom orphan)."""
    fams = _target_families(backend)
    routed = routed_axes(backend)
    out: set[str] = set()
    for s in FIELD_REGISTRY.values():
        if s.classification != LEVER:
            continue
        if backend in s.backends:                          # direct: always leverable
            out.add(s.axis)
        elif (fams & set(s.backends)) and s.axis in routed:  # family-indirect: only if the RTL admits it
            out.add(s.axis)
    return out


def _routes(backend: str) -> list:
    action_catalog.ensure_backend(backend)                 # derive+register this backend's routes on first use
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
