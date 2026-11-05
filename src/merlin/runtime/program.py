"""``MerlinProgram`` — the target-independent, lean whole-model PROGRAM the HW-agnostic runtime
replays.

This is the superset of :class:`~merlin.xdsl_dialects.lowering.dispatch_program.DispatchProgram`
(the DAG-verified command buffer of dispatch/view nodes over stable buffer ids) plus the two things
a LEAN runtime needs that the bare dispatch program lacks:

* a compile-time **static memory plan** (:mod:`~merlin.xdsl_dialects.lowering.arena_plan`) — one
  arena + per-intermediate-buffer offsets, so the runtime binds ``arena_base()+offset`` with ZERO
  per-op allocation (vs the ~4391 per-op mallocs today); and
* **target/dispatch metadata** — a kernel table (id → compiled symbol + fused roots + required
  capability), per-node parallel annotation (``none|forall|launch``), an extensible/namespaced
  opcode set, and per-program capability flags — the hooks that let custom-ISA / SIMT / research
  HW plug in via an adapter without changing the program format.

The program is serializable data (``to_dict``/``to_json``). The explicit
:mod:`merlin.compile.dispatch_c` exporter generates a serial C driver from a
bounded memory plan and compiled per-node wrappers. It does not interpret tensor
operations or automatically compile missing wrappers. The monolithic model path
remains separate; see ``docs/design/static_arena_wiring.md``.

The arena the measured whole-model build actually gets does NOT come through here. It is planned
directly over the emitted LLVM IR by :mod:`merlin.llvmlower.arena_bind`, which shares this module's
placement core (``arena_plan.pack_disjoint``) but derives liveness from the ``malloc``/``free`` pairs
the build emits rather than from a dispatch DAG.
Default-off and additive — the existing monolithic/outlined paths are untouched.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..xdsl_dialects.lowering.arena_plan import (
    ArenaRequirements,
    MemoryPlan,
    check_bounded_arena,
    plan_arena,
    plan_bounded_arena,
)
from ..xdsl_dialects.lowering.dispatch_program import DispatchProgram
from ..xdsl_dialects.lowering.global_plan import GlobalPlan, verify_global_plan
from ..xdsl_dialects.lowering.global_plan_emission import (
    GlobalPlanEmission,
    GlobalPlanEmitter,
    emit_global_plan,
)

PROGRAM_ABI_VERSION = "0.1"

if TYPE_CHECKING:
    from merlin.perf.global_planner import GlobalPlanPolicy, GlobalPlanResult, TargetPlanningAdapter


@dataclass
class KernelEntry:
    id: str  # stable kernel id (== dispatch symbol)
    symbol: str  # compiled C symbol the replay invokes
    roots: list[str] = field(default_factory=list)  # fused root op names (provenance)
    capability: str = "scalar"  # required target capability (scalar|rvv|systolic|simt|...)
    # Model-layer provenance carried from the dispatch node's ``prov.*`` (see the outliner). The
    # SAME key the cross-compiler compare (ExecuTorch↔Merlin) and the section slicer join on. ``role``
    # is intentionally NOT stored here — it is derived from ``fqn`` downstream (``dse_guidance``) to
    # keep ``runtime`` free of an analysis-layer import (``dse_guidance`` already imports ``runtime``).
    region_id: str = ""  # prov.region_id (e.g. "matmul_3")
    fqn: str = ""  # prov.fqn (deepest nn.Module path, e.g. "blocks.0.attn.q")


@dataclass
class MerlinProgram:
    abi_version: str
    entry: str
    dispatch: DispatchProgram  # the DAG (buffers, nodes, args, results)
    memory_plan: MemoryPlan  # static arena + offsets
    kernels: dict[str, KernelEntry]  # symbol -> KernelEntry
    capabilities: list[str]  # capabilities this program requires of a target
    parallel: dict[int, dict] = field(default_factory=dict)  # node index -> {mode, grid, ...}
    opcodes: list[str] = field(default_factory=list)  # namespaced opcode set used
    # Optional because the legacy program ABI remains byte-identical unless the production compiler
    # explicitly elects the global-plan path.  ``shadow`` mode can therefore analyse a model without
    # changing the command buffer that executes it; ``emit`` attaches the exact plan lowering consumed.
    global_plan: GlobalPlan | None = None
    global_plan_emission: GlobalPlanEmission | None = None
    memory_requirements: ArenaRequirements | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "abi_version": self.abi_version,
            "entry": self.entry,
            "dispatch": self.dispatch.to_dict(),
            "memory_plan": self.memory_plan.to_dict(),
            "kernels": {k: vars(v) for k, v in self.kernels.items()},
            "capabilities": list(self.capabilities),
            "parallel": {str(k): v for k, v in self.parallel.items()},
            "opcodes": list(self.opcodes),
        }
        if self.global_plan is not None:
            result["global_plan"] = self.global_plan.to_dict()
        if self.global_plan_emission is not None:
            result["global_plan_emission"] = self.global_plan_emission.receipt()
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def stats(self) -> dict[str, Any]:
        d = self.dispatch
        return {
            "n_nodes": len(d.nodes),
            "n_dispatches": d.n_dispatches,
            "n_views": sum(1 for n in d.nodes if n.kind == "view"),
            "n_buffers": len(d.buffers),
            "n_kernels": len(self.kernels),
            "arena_bytes": self.memory_plan.arena_bytes,
            "arena_reuse_factor": self.memory_plan.stats.get("reuse_factor"),
            "mallocs_eliminated": self.memory_plan.stats.get("n_intermediate_buffers"),
            "capabilities": list(self.capabilities),
            "global_plan": self.global_plan.digest if self.global_plan is not None else None,
            "global_plan_emitted": self.global_plan_emission is not None,
        }


def build_program(
    dispatch: DispatchProgram,
    *,
    capability: str = "rvv",
    abi_version: str = PROGRAM_ABI_VERSION,
    global_plan: GlobalPlan | None = None,
    global_plan_emitter: GlobalPlanEmitter | None = None,
    memory_requirements: ArenaRequirements | Callable[[DispatchProgram], ArenaRequirements] | None = None,
    memory_plan: MemoryPlan | None = None,
) -> MerlinProgram:
    """Assemble a :class:`MerlinProgram` from a dispatch program: plan the arena, build the kernel
    table from the dispatch nodes, and tag the required capability. The opcode set is the distinct
    node ops (namespaced as needed by a target adapter); v1 parallel annotation is empty (the replay
    runs nodes serially; multi-core is layered in via ``parallel`` later, node-level).
    A supplied memory plan is checked against the rewritten graph and explicit
    requirements without invoking allocation again.
    """
    if global_plan is not None:
        problems = verify_global_plan(dispatch, global_plan)
        if problems:
            raise ValueError("cannot attach invalid global plan: " + "; ".join(problems))
    if global_plan_emitter is not None and global_plan is None:
        raise ValueError("a global-plan emitter requires a selected global plan")
    emission = (
        emit_global_plan(dispatch, global_plan, global_plan_emitter)
        if global_plan is not None and global_plan_emitter is not None
        else None
    )
    # An emitter changes the program that is actually replayed. Without one, attaching a plan is
    # explicitly shadow analysis and preserves the legacy dispatch byte-for-byte.
    if emission is not None:
        dispatch = emission.dispatch
    # A selected emitter can fuse nodes or insert conversions. Derive lifetimes
    # only after that rewrite, against the graph that will actually execute.
    if callable(memory_requirements):
        memory_requirements = memory_requirements(dispatch)
    if memory_plan is not None:
        if memory_requirements is None:
            raise ValueError("reusing a memory plan requires explicit memory requirements")
        check_bounded_arena(dispatch, memory_requirements, memory_plan)
        mem = memory_plan
    else:
        mem = (
            plan_bounded_arena(dispatch, memory_requirements)
            if memory_requirements is not None
            else plan_arena(dispatch)
        )
    kernels: dict[str, KernelEntry] = {}
    opcodes: set[str] = set()
    for node in dispatch.nodes:
        opcodes.add(node.op if node.kind == "view" else f"dispatch:{node.op}")
        if node.kind == "dispatch" and node.op not in kernels:
            # The dispatch node's prov keys are the model2MLIR ``prov.*`` spellings (``prov.op`` /
            # ``prov.fqn`` / ``prov.region_id``), NOT bare ``root``/``op``/``name`` — the old filter
            # matched none of them, so ``roots`` was silently always ``[]`` and the layer identity was
            # dropped when building the kernel table. Read the real keys and carry region_id/fqn through.
            prov = node.prov or {}
            roots = [prov[k] for k in ("prov.op", "prov.name") if k in prov]
            kernels[node.op] = KernelEntry(
                id=node.op,
                symbol=node.op,
                roots=roots,
                capability=capability,
                region_id=prov.get("prov.region_id", ""),
                fqn=prov.get("prov.fqn", ""),
            )
    return MerlinProgram(
        abi_version=abi_version,
        entry=dispatch.entry,
        dispatch=dispatch,
        memory_plan=mem,
        kernels=kernels,
        capabilities=[capability],
        opcodes=sorted(opcodes),
        global_plan=global_plan,
        global_plan_emission=emission,
        memory_requirements=memory_requirements,
    )


def plan_program(
    dispatch: DispatchProgram,
    adapter: TargetPlanningAdapter,
    emitter: GlobalPlanEmitter,
    *,
    memory_requirements: Callable[[DispatchProgram], ArenaRequirements],
    policy: GlobalPlanPolicy | None = None,
) -> tuple[MerlinProgram, GlobalPlanResult]:
    """Choose placement/fusion, emit it, then allocate the resulting graph.

    Target providers own admitted implementations and explicit conversion
    routes. No complete plan means a build failure, not a host fallback.
    The returned search report retains unresolved performance bounds.
    """
    from merlin.perf.global_planner import optimize_program

    result = optimize_program(dispatch, adapter, policy=policy)
    if result.plan is None:
        raise ValueError("no executable placement plan: " + "; ".join(result.refusals))
    program = build_program(
        dispatch,
        capability="compiled",
        global_plan=result.plan,
        global_plan_emitter=emitter,
        memory_requirements=memory_requirements,
    )
    # One default capability for all kernels would conceal mixed placement.
    # Each component's checked receipt gives the actual selected placement.
    assert program.global_plan_emission is not None
    placements = {item.id: item.placement for item in result.plan.selected}
    # A transition's destination describes its output storage, not the processor
    # executing a copy/conversion. Its implementation binding supplies that.
    placements.update({item.id: "transition" for item in result.plan.transitions})
    assigned = {}
    for component in (*program.global_plan_emission.regions, *program.global_plan_emission.transitions):
        for index in component.node_indices:
            node = program.dispatch.nodes[index]
            if node.kind == "dispatch":
                entry = program.kernels[node.op]
                placement = placements[component.plan_id]
                if node.op in assigned and assigned[node.op] != placement:
                    raise ValueError("shared kernel symbol has conflicting placements")
                assigned[node.op] = placement
                entry.capability = placement
    program.capabilities = sorted({entry.capability for entry in program.kernels.values()})
    return program, result
