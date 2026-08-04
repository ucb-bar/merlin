"""The compiler-region registry — every distinct region of the Merlin compiler as an EXPLICIT,
registrable edit-point, organized by compilation PHASE, so an agent (or engineer) never has to search
the compiler to know WHERE to change a thing and HOW to add a new knob/heuristic/pass/dialect there.

Two levels, by design:
- **phases** (compilation stages): frontend -> global -> dispatch -> kernel-codegen -> memory ->
  emission -> runtime, plus cross-cutting and target-gen. The pipeline stages.
- **regions** (the ~one-per-concern fine grain — the point): each names the real modules that
  implement it, its ``EditPoint``s (concrete file + mechanism + one-line how-to-add + the register()
  hook where one exists), and the CCA facet axes it governs. Every RVV CCA LEVER axis is governed by
  EXACTLY one region (``check_regions``) — this is the taxonomy the CCA<->lever bijection broadens onto.

TARGET-AGNOSTIC: these are compiler-wide concerns (RVV is the first instantiation; gemmini/others plug
in per-target edit-points). The ``target-gen`` phase is where the eventual agentic target-DIALECT
generation hooks in — the same registry drives it.

Honest by construction: where a region has no clean seam yet, its EditPoint says so (``forkable_now=
False``) — those gaps are the concrete work-items, never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

# The compilation phases (stages), in pipeline order + the two non-linear buckets.
PHASES = ("frontend", "global", "dispatch", "kernel-codegen", "memory", "emission", "runtime",
          "cross-cutting", "target-gen")


@dataclass(frozen=True)
class EditPoint:
    """One registrable place to change the compiler within a region."""
    kind: str                 # FLAG | KNOB | HEURISTIC | PASS | CODEGEN | REGISTRY | DATA | DIALECT
    seam: str                 # a routable target_seam label
    file: str                 # repo-relative file to edit
    how_to_add: str           # one line: how to add ANOTHER edit-point of this kind here
    registry: str | None = None   # the register() fn to call, if this seam is a registry (else None)
    forkable_now: bool = True     # False = an honest gap (a work-item, no clean seam yet)


@dataclass(frozen=True)
class Region:
    key: str
    phase: str                # one of PHASES
    title: str
    summary: str
    modules: tuple[str, ...]       # repo-relative compiler modules that implement the region
    edit_points: tuple[EditPoint, ...]
    cca_axes: tuple[str, ...] = ()  # the CCA facet axes this region governs (1:1 across regions)


def _r(key, phase, title, summary, modules, edit_points, cca_axes=()):
    return Region(key, phase, title, summary, tuple(modules), tuple(edit_points), tuple(cca_axes))


def _ep(kind, seam, file, how, registry=None, forkable=True):
    return EditPoint(kind, seam, file, how, registry, forkable)


_IMPR = "merlin.llvmlower.impr_features.register"
_PQI = "merlin/python/merlin/llvmlower/passes_quant_int.py"
_PIPE = "merlin/python/merlin/llvmlower/pipeline.py"
_IF = "merlin/python/merlin/llvmlower/impr_features.py"
_FS = "merlin/python/merlin/rvvgen/from_strategy.py"

REGIONS: dict[str, Region] = {r.key: r for r in [
    # ---- frontend / ingest ------------------------------------------------------------
    _r("graph-ingest", "frontend", "Graph ingestion & op coverage",
       "Bring the flattened exported graph into our linalg IR; op coverage; the graph the CCA "
       "exploration starts from.",
       ("merlin/python/merlin/frontends/linalg_mlir.py", "merlin/python/merlin/frontends/facts.py",
        "merlin/python/merlin/frontends/registry.py"),
       [_ep("REGISTRY", "frontend:adapter", "merlin/python/merlin/frontends/registry.py",
            "register a frontend adapter (model2MLIR/gguf/...) for a new graph source")]),
    _r("numerics-precision", "frontend", "Decomposition & numerics / mixed-precision policy",
       "Per-module format rules + op decomposition/normalization before compute lowering.",
       ("merlin/python/merlin/frontends/mixed_precision.py",
        "merlin/python/merlin/frontends/quant_ext.py"),
       [_ep("DATA", "frontend:mixed_precision_policy",
            "merlin/python/merlin/frontends/mixed_precision.py",
            "add a MixedPrecisionPolicy rule (per-module dtype/format)")]),
    # ---- global (backend-agnostic) ----------------------------------------------------
    _r("quantization", "global", "Quantization / QDQ <-> real-quant",
       "Real quantized compute (i8xi8->i32 vwmacc) vs fake-quant (dequant back to bf16/fp32).",
       (_PQI, "merlin/python/merlin/frontends/quant_ext.py"),
       [_ep("KNOB", "schedule:dtype_strategy", _FS,
            "select the datapath dtype strategy (int8_w8a8 / bf16-f32acc)"),
        _ep("PASS", "quant:apply_quant", "merlin/python/merlin/llvmlower/quant_passes.py",
            "register a QuantPass in quant_passes (toggle/reorder the int8 datapath) — the six "
            "lower_*_int passes are now a registry, not a hardcoded sequence",
            registry="merlin.llvmlower.quant_passes")],
       cca_axes=("compute.widening", "compute.accumulator_dtype", "vector.sew")),
    _r("global-passes", "global", "Global optimization passes (backend-agnostic)",
       "The frozen baseline pass list + Merlin-authored xDSL passes; changed via default-off "
       "impr_features pipeline hooks.",
       (_PIPE, "merlin/python/merlin/llvmlower/passes_xdsl.py"),
       [_ep("PASS", "impr_features:<name>", _IF,
            "register(ImprFeature(name, edit_pipeline=...)) — a default-off pipeline edit", _IMPR)]),
    # ---- dispatch ---------------------------------------------------------------------
    _r("dispatch-gen", "dispatch", "Dispatch generation / clustering",
       "Outline func @forward into per-dispatch kernels + driver; fusion-root grouping.",
       ("merlin/python/merlin/xdsl_dialects/lowering/outline.py",
        "merlin/python/merlin/runtime/dispatch_runtime.py"),
       [_ep("HEURISTIC", "outline:clustering",
            "merlin/python/merlin/xdsl_dialects/lowering/outline.py",
            "GAP: clustering granularity is internal to the pass; expose a clustering-policy param",
            forkable=False)]),
    _r("dispatch-scheduling", "dispatch", "Dispatch scheduling (cross-dispatch)",
       "Cross-dispatch order + multicore partition + the runtime program's memory/parallel plan.",
       ("merlin/python/merlin/runtime/program.py",
        "merlin/python/merlin/xdsl_dialects/lowering/schedule_dispatch.py"),
       [_ep("DATA", "program:schedule",
            "merlin/python/merlin/xdsl_dialects/lowering/schedule_dispatch.py",
            "GAP: partitioner has no registrable policy hook yet", forkable=False)]),
    # ---- kernel codegen (the fine concerns) -------------------------------------------
    _r("data-tiling", "kernel-codegen", "Data tiling",
       "The transform-schedule tile decision (register block / output-tile width), including WHICH "
       "contraction classes it claims at all — a block no extent admits declines the class to scalar.",
       (_PIPE, _FS, _IF),
       [_ep("KNOB", "schedule:op_match", _FS, "add/adjust an op_match tile in render_schedule")],
       cca_axes=("compute.register_block", "compute.nr_is_vsetvlmax",
                 "coverage.claimed_mac_fraction", "coverage.unclaimed_op_classes")),
    _r("vectorization", "kernel-codegen", "Vectorization",
       "Scoped vectorize + vector width/LMUL/VL strategy/tail policy, including the SCOPE: which "
       "ops are vectorized at all versus left to convert-linalg-to-loops.",
       (_PIPE, _IF),
       [_ep("KNOB", "schedule:vector_sizes", _FS, "adjust vector_sizes (LMUL) in render_schedule"),
        _ep("PASS", "impr_features:<name>", _IF, "register a vectorization feature (edit_schedule)",
            _IMPR)],
       cca_axes=("vector.lmul", "vector.vl_strategy", "vector.tail",
                 "coverage.non_contraction_op_fraction")),
    _r("instruction-selection", "kernel-codegen", "Instruction selection",
       "Which instruction form is emitted (fused vfmacc vs mul_add; custom ISA via .insn).",
       (_IF, "merlin/python/merlin/llvmlower/custom_isa.py"),
       [_ep("PASS", "impr_features:fused_vfmacc_contraction", _IF,
            "register a lowering-pattern feature that selects an instruction form", _IMPR),
        _ep("CODEGEN", "custom_isa:inline_asm", "merlin/python/merlin/llvmlower/custom_isa.py",
            "declare a custom instruction via merlin.inline_asm -> .insn (no LLVM fork)")],
       cca_axes=("compute.contraction_form",)),
    _r("instruction-scheduling", "kernel-codegen", "Instruction scheduling",
       "Instruction ordering / latency hiding — largely the LLVM backend via clang -O.",
       ("merlin/python/merlin/llvmlower/codegen.py",),
       [_ep("FLAG", "cflag:sched", "merlin/python/merlin/llvmlower/codegen.py",
            "GAP: scheduling is LLVM's; only reachable via clang flags — no direct merlin seam",
            forkable=False)]),
    _r("inner-loop", "kernel-codegen", "Inner-loop computation",
       "The micro-kernel body (e.g. vectorized transcendental activation polynomial).",
       ("merlin/python/merlin/llvmlower/accum_microkernel.py", _IF),
       [_ep("PASS", "impr_features:vectorized_transcendental_activation", _IF,
            "register an inner-loop-body feature (edit_schedule / a microkernel emitter)", _IMPR)],
       cca_axes=("compute.activation_vectorization",)),
    _r("fusion", "kernel-codegen", "Fusion",
       "Op fusion, incl. fusing the requant/narrow epilogue into the store.",
       (_IF, _PIPE),
       [_ep("PASS", "pass:fuse-requant-narrowing-store", _IF,
            "GAP: register a fusion feature (edit_pipeline) — e.g. fuse requant+vnclip into the store",
            forkable=False)],
       cca_axes=("compute.epilogue",)),
    _r("accumulation", "kernel-codegen", "Accumulation / accumulator residency",
       "Keep the accumulator register-resident across the reduction; reduction form (tree/vredsum).",
       (_IF, "merlin/python/merlin/llvmlower/accum_microkernel.py"),
       [_ep("PASS", "impr_features:accumulator_resident_microkernel", _IF,
            "register an accumulation feature; the spill-free closer needs a CODEGEN microkernel emitter",
            _IMPR)],
       cca_axes=("compute.accumulator_resident", "compute.reduction_form")),
    # ---- memory / layout --------------------------------------------------------------
    _r("bufferization-memplan", "memory", "Bufferization & memory planning",
       "Tensor->memref bufferization, out-param buffers, the AOT static arena plan.",
       (_PIPE, "merlin/python/merlin/xdsl_dialects/lowering/arena_plan.py"),
       [_ep("PASS", "impr_features:<name>", _IF,
            "register a bufferization-pipeline edit (e.g. eliminate-empty-tensors placement)", _IMPR)],
       cca_axes=("envelope.runtime_calls", "envelope.calls_in_loop")),
    _r("layout-packing", "memory", "Layout assignment / packing",
       "Compile-time operand layout + panel packing (contiguous register-tile panels) — the packed "
       "unit-stride B panel that is the #1 expert GEMM lever.",
       (_IF, _FS),
       [_ep("PASS", "impr_features:vfmacc_packed", _IF,
            "register a packing feature (transform.structured.pack layout)", _IMPR)],
       cca_axes=("memory.access_pattern",)),
    # ---- emission ---------------------------------------------------------------------
    _r("target-lowering", "emission", "Target lowering / legalization",
       "Dialect -> LLVM mechanical descent (convert-*-to-llvm, reconcile-unrealized-casts).",
       (_PIPE,),
       [_ep("PASS", "impr_features:<name>", _IF,
            "GAP: the convert-*-to-llvm set is fixed; a legalization edit needs an edit_pipeline hook",
            _IMPR, forkable=False)]),
    _r("asm-emission", "emission", "ASM / instruction emission (codegen)",
       "Backend selection registry + per-backend codegen + cflags.",
       ("merlin/python/merlin/runtime/backends/base.py",
        "merlin/python/merlin/runtime/backends/rvv_codegen.py",
        "merlin/python/merlin/llvmlower/codegen.py"),
       [_ep("REGISTRY", "backend:_REGISTRY", "merlin/python/merlin/runtime/backends/base.py",
            "register a target backend (name -> module -> TargetClass) — one entry",
            "merlin.runtime.backends.base"),
        _ep("FLAG", "cflag:march", "merlin/python/merlin/runtime/backends/zephyr_model.py",
            "add/adjust an RVV cflag / march feature")]),
    # ---- runtime ----------------------------------------------------------------------
    _r("runtime-layout-encoding", "runtime", "Runtime layout & encodings",
       "Extensible program/command-buffer formats (opcodes, capability flags, im2col recipes).",
       ("merlin/python/merlin/runtime/program.py",
        "merlin/python/merlin/runtime/commandbuffer.py"),
       [_ep("DATA", "program:opcodes", "merlin/python/merlin/runtime/program.py",
            "add a namespaced opcode / capability flag / layout-encoding to the program format")]),
    _r("aot-opt", "runtime", "Ahead-of-time (once) optimizations",
       "AOT-once products: the static arena memory plan, prepacked weights.",
       ("merlin/python/merlin/xdsl_dialects/lowering/arena_plan.py",),
       [_ep("DATA", "arena:plan", "merlin/python/merlin/xdsl_dialects/lowering/arena_plan.py",
            "extend the AOT arena/prepack plan emitted once at compile time")]),
    _r("hw-sync", "runtime", "HW synchronization",
       "Barrier / hart-partition / event synchronization emitted for multicore execution.",
       ("merlin/python/merlin/runtime/backends/rvv_codegen.py",),
       [_ep("CODEGEN", "codegen:hw_sync", "merlin/python/merlin/runtime/backends/rvv_codegen.py",
            "GAP: barrier/sync is hand-written per codegen backend; no sync-strategy registry yet",
            forkable=False)]),
    # ---- cross-cutting ----------------------------------------------------------------
    _r("heuristics", "cross-cutting", "Optimization heuristics (when to apply what)",
       "The CCA->lever router + escalation ladder; DSE candidate selection.",
       ("merlin/python/merlin/kernels/action_catalog.py",
        "merlin/python/merlin/dse_guidance/candidates.py"),
       [_ep("HEURISTIC", "route:_RVV_ROUTES", "merlin/python/merlin/kernels/action_catalog.py",
            "add a _Route(axis, when, action_class, target_seam, intended_facet) row")]),
    _r("cost-model-capabilities", "cross-cutting", "Cost model & target capabilities",
       "The cost model + the datatype->compute-unit target-capability manifests (also the INPUT to "
       "agentic target-dialect generation).",
       ("merlin/python/merlin/dse/cost_model.py",
        "merlin/python/merlin/targetgen/capability_manifests.py"),
       [_ep("DATA", "capability:manifest", "merlin/python/merlin/targetgen/capability_manifests.py",
            "add/extend a target-capability manifest (datatype -> compute-unit)")]),
    # ---- target-dialect generation (the agentic targetgen path) -----------------------
    _r("target-dialect-gen", "target-gen", "Target dialect / contract / interface generation",
       "Generate a new target's dialect + contract + interface + lowering scaffolding — where the "
       "eventual agentic target-dialect generation (gemmini etc.) plugs into this same registry.",
       ("merlin/python/merlin/targetgen/cli.py",
        "merlin/python/merlin/xdsl_dialects/contract.py",
        "merlin/python/merlin/xdsl_dialects/interface.py"),
       [_ep("DIALECT", "targetgen:emit", "merlin/python/merlin/targetgen/cli.py",
            "generate a target dialect/contract/interface via merlin-targetgen build --emit ...")]),
]}


def phases() -> tuple[str, ...]:
    return PHASES


def regions_by_phase(phase: str) -> list[Region]:
    return [r for r in REGIONS.values() if r.phase == phase]


def region_for_axis(axis: str) -> Region | None:
    """The region governing a CCA facet axis (e.g. 'compute.register_block' -> data-tiling)."""
    for r in REGIONS.values():
        if axis in r.cca_axes:
            return r
    return None


def all_edit_points() -> list[tuple[str, EditPoint]]:
    """(region key, EditPoint) for every registrable edit-point across the compiler."""
    return [(r.key, ep) for r in REGIONS.values() for ep in r.edit_points]


def check_regions() -> list[str]:
    """Sanity/coverage invariant, returns problems (empty = OK):
    - every region has a valid phase, ≥1 real module file, and ≥1 edit-point;
    - every RVV CCA LEVER axis is governed by EXACTLY one region (the bijection's region taxonomy)."""
    from ..common.paths import repo_root
    from . import cca_contract

    problems: list[str] = []
    root = repo_root()
    for key, r in REGIONS.items():
        if r.phase not in PHASES:
            problems.append(f"region {key}: unknown phase {r.phase!r}")
        if not r.edit_points:
            problems.append(f"region {key}: no edit points")
        for m in r.modules:
            if not (root / m).is_file():
                problems.append(f"region {key}: module missing on disk: {m}")
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
