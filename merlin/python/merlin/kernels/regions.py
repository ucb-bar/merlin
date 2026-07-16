"""The compiler-region registry — the 8 regions of the Merlin compiler as EXPLICIT, registrable
edit-points, so an agent (or engineer) never has to search the compiler to know WHERE to change a
thing and HOW to add a new knob/heuristic/pass there.

Each ``Region`` names the compiler modules that implement it, the ``EditPoint``s (a concrete file +
the mechanism + a one-line "how to add another"), and the CCA facet axes it governs. The axis↔region
map is also the region taxonomy the CCA⇄lever bijection (``cca_contract``) broadens onto: every RVV
LEVER axis belongs to exactly one region (checked by ``check_regions``).

This is metadata ABOUT the compiler, deliberately honest: where a region has no clean seam yet (e.g.
quantization is a hardcoded pass sequence, dispatch scheduling has no policy knob), the EditPoint says
so — those honest gaps are the concrete C3 work-items, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EditPoint:
    """One registrable place to change the compiler within a region."""
    kind: str                 # FLAG | KNOB | HEURISTIC | PASS | CODEGEN | REGISTRY | DATA
    seam: str                 # the routable target_seam string (see action_catalog.SEAM_FILES)
    file: str                 # repo-relative file to edit
    how_to_add: str           # one line: how to add ANOTHER edit-point of this kind here
    registry: str | None = None   # the register() fn to call, if this seam is a registry (else None)
    forkable_now: bool = True     # False = an honest gap (a work-item, no clean seam yet)


@dataclass(frozen=True)
class Region:
    key: str
    title: str
    summary: str
    modules: tuple[str, ...]       # repo-relative compiler modules that implement the region
    edit_points: tuple[EditPoint, ...]
    cca_axes: tuple[str, ...] = ()  # the CCA facet axes this region governs


# The 8 regions (grounded in the real modules; honest about where a seam is missing).
REGIONS: dict[str, Region] = {
    "quantization": Region(
        key="quantization", title="Quantization / QDQ",
        summary="QDQ-format handling; real quantized compute (i8xi8->i32 vwmacc) vs fake-quant "
                "(dequant back to bf16/fp32).",
        modules=("merlin/python/merlin/llvmlower/passes_quant_int.py",
                 "merlin/python/merlin/frontends/quant_ext.py",
                 "merlin/python/merlin/frontends/mixed_precision.py"),
        edit_points=(
            EditPoint("KNOB", "schedule:dtype_strategy",
                      "merlin/python/merlin/rvvgen/from_strategy.py",
                      "select the datapath dtype strategy (int8_w8a8 / bf16-f32acc)"),
            EditPoint("PASS", "quant:passes_quant_int",
                      "merlin/python/merlin/llvmlower/passes_quant_int.py",
                      "GAP: the six lower_*_int passes are a hardcoded sequence gated by a bool; "
                      "add a quant-pass registry (mirror impr_features) to toggle/order them",
                      forkable_now=False),
        ),
        cca_axes=("compute.widening", "compute.accumulator_dtype", "vector.sew")),
    "global-passes": Region(
        key="global-passes", title="Global optimization passes (backend-agnostic)",
        summary="The frozen baseline pass list + Merlin-authored xDSL passes; changed only via "
                "default-off impr_features pipeline hooks.",
        modules=("merlin/python/merlin/llvmlower/pipeline.py",
                 "merlin/python/merlin/llvmlower/passes_xdsl.py"),
        edit_points=(
            EditPoint("PASS", "impr_features:<name>",
                      "merlin/python/merlin/llvmlower/impr_features.py",
                      "register(ImprFeature(name, edit_pipeline=...)) — a default-off pipeline edit",
                      registry="merlin.llvmlower.impr_features.register"),
        )),
    "dispatch-gen": Region(
        key="dispatch-gen", title="Dispatch generation / clustering",
        summary="Outlining func @forward into per-dispatch kernels + the driver; dedup/clustering.",
        modules=("merlin/python/merlin/xdsl_dialects/lowering/outline.py",
                 "merlin/python/merlin/runtime/dispatch_runtime.py"),
        edit_points=(
            EditPoint("HEURISTIC", "outline:clustering",
                      "merlin/python/merlin/xdsl_dialects/lowering/outline.py",
                      "GAP: fusion-root grouping / clustering granularity is internal to the pass; "
                      "no knob yet — expose a clustering-policy parameter",
                      forkable_now=False),
        )),
    "tiling-instsel-fusion": Region(
        key="tiling-instsel-fusion",
        title="Tiling / instruction selection / scheduling / inner-loop / fusion / accumulation",
        summary="The transform-dialect schedule + the default-off feature registry — the well-exposed "
                "region (register blocking, vfmacc, accumulator residency, packing, tail clamps).",
        modules=("merlin/python/merlin/llvmlower/pipeline.py",
                 "merlin/python/merlin/llvmlower/impr_features.py",
                 "merlin/python/merlin/rvvgen/from_strategy.py"),
        edit_points=(
            EditPoint("KNOB", "schedule:<knob>",
                      "merlin/python/merlin/rvvgen/from_strategy.py",
                      "add an op_match/lowering_patterns/contraction_strategy knob to render_schedule"),
            EditPoint("PASS", "impr_features:<name>",
                      "merlin/python/merlin/llvmlower/impr_features.py",
                      "register(ImprFeature(name, edit_schedule=...)) — a default-off schedule edit",
                      registry="merlin.llvmlower.impr_features.register"),
        ),
        cca_axes=("compute.contraction_form", "compute.register_block", "compute.nr_is_vsetvlmax",
                  "compute.reduction_form", "compute.accumulator_resident", "compute.epilogue",
                  "compute.mr_adapts_to_m", "compute.activation_vectorization",
                  "vector.lmul", "vector.vl_strategy", "vector.tail")),
    "heuristics": Region(
        key="heuristics", title="Optimization heuristics (when to apply what)",
        summary="The CCA->lever router + escalation ladder; DSE candidate selection.",
        modules=("merlin/python/merlin/kernels/action_catalog.py",
                 "merlin/python/merlin/dse_guidance/candidates.py"),
        edit_points=(
            EditPoint("HEURISTIC", "route:_RVV_ROUTES",
                      "merlin/python/merlin/kernels/action_catalog.py",
                      "add a _Route(axis, when, action_class, target_seam, intended_facet) row"),
        )),
    "dispatch-scheduling": Region(
        key="dispatch-scheduling", title="Dispatch scheduling (cross-dispatch)",
        summary="The runtime program (memory plan, parallel modes) + arena planner + multicore partition.",
        modules=("merlin/python/merlin/runtime/program.py",
                 "merlin/python/merlin/xdsl_dialects/lowering/arena_plan.py",
                 "merlin/python/merlin/xdsl_dialects/lowering/schedule_dispatch.py"),
        edit_points=(
            EditPoint("DATA", "program:MerlinProgram",
                      "merlin/python/merlin/runtime/program.py",
                      "GAP: the arena planner / partitioner have no registrable policy hook yet",
                      forkable_now=False),
        )),
    "asm-emission": Region(
        key="asm-emission", title="ASM / instruction emission (target dialect -> codegen)",
        summary="Backend selection registry + the custom-ISA inline-asm hatch + per-backend codegen.",
        modules=("merlin/python/merlin/runtime/backends/base.py",
                 "merlin/python/merlin/llvmlower/custom_isa.py",
                 "merlin/python/merlin/runtime/backends/rvv_codegen.py"),
        edit_points=(
            EditPoint("REGISTRY", "backend:_REGISTRY",
                      "merlin/python/merlin/runtime/backends/base.py",
                      "register a target backend (name -> module -> TargetClass) — one entry",
                      registry="merlin.runtime.backends.base"),
            EditPoint("CODEGEN", "custom_isa:inline_asm",
                      "merlin/python/merlin/llvmlower/custom_isa.py",
                      "declare a custom instruction via merlin.inline_asm -> .insn (no LLVM fork)"),
        )),
    "runtime-hooks": Region(
        key="runtime-hooks", title="Runtime hooks (layout / encoding / AOT / HW sync)",
        summary="Extensible program/command-buffer formats (opcodes, capability flags, im2col recipes, "
                "arena offsets); HW sync emitted per codegen backend.",
        modules=("merlin/python/merlin/runtime/program.py",
                 "merlin/python/merlin/runtime/commandbuffer.py"),
        edit_points=(
            EditPoint("DATA", "program:opcodes",
                      "merlin/python/merlin/runtime/program.py",
                      "add a namespaced opcode / capability flag to the program format"),
            EditPoint("CODEGEN", "codegen:hw_sync",
                      "merlin/python/merlin/runtime/backends/rvv_codegen.py",
                      "GAP: barrier/sync is hand-written per codegen backend; no layout/sync registry",
                      forkable_now=False),
        )),
}


def region_for_axis(axis: str) -> Region | None:
    """The region governing a CCA facet axis (e.g. 'compute.register_block' -> tiling-instsel-fusion)."""
    for r in REGIONS.values():
        if axis in r.cca_axes:
            return r
    return None


def all_edit_points() -> list[tuple[str, EditPoint]]:
    """(region key, EditPoint) for every registrable edit-point across the compiler."""
    return [(r.key, ep) for r in REGIONS.values() for ep in r.edit_points]


def check_regions() -> list[str]:
    """Sanity/coverage invariant, returns problems (empty = OK):
    - every region names ≥1 real module file + has ≥1 edit-point;
    - every RVV CCA LEVER axis (from cca_contract) is governed by EXACTLY one region — this is the
      region taxonomy the bijection contract broadens onto."""
    from ..common.paths import repo_root
    from . import cca_contract

    problems: list[str] = []
    root = repo_root()
    for key, r in REGIONS.items():
        if not r.edit_points:
            problems.append(f"region {key}: no edit points")
        for m in r.modules:
            if not (root / m).is_file():
                problems.append(f"region {key}: module missing on disk: {m}")
    # axis coverage: each lever axis maps to exactly one region
    lever = cca_contract.leverable_axes("rvv")
    covered: dict[str, list[str]] = {}
    for r in REGIONS.values():
        for ax in r.cca_axes:
            covered.setdefault(ax, []).append(r.key)
    for ax in sorted(lever):
        regs = covered.get(ax, [])
        if len(regs) == 0:
            problems.append(f"lever axis {ax}: not governed by any region")
        elif len(regs) > 1:
            problems.append(f"lever axis {ax}: governed by multiple regions {regs}")
    return problems
