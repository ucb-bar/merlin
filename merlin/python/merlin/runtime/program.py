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

The program is pure data (``to_dict``/``to_json``): the Python reference runtime
(:mod:`merlin.runtime.dispatch_runtime` / :mod:`merlin.runtime.simulator`) and the C replay engine
(``merlin/runtime/c/merlin_program.c``) both consume it; a target adapter lowers it to its encoding.
Default-off and additive — the existing monolithic/outlined paths are untouched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..xdsl_dialects.lowering.arena_plan import MemoryPlan, plan_arena
from ..xdsl_dialects.lowering.dispatch_program import DispatchProgram

PROGRAM_ABI_VERSION = "0.1"


@dataclass
class KernelEntry:
    id: str                              # stable kernel id (== dispatch symbol)
    symbol: str                          # compiled C symbol the replay invokes
    roots: list[str] = field(default_factory=list)   # fused root op names (provenance)
    capability: str = "scalar"           # required target capability (scalar|rvv|gemmini|simt|...)


@dataclass
class MerlinProgram:
    abi_version: str
    entry: str
    dispatch: DispatchProgram            # the DAG (buffers, nodes, args, results)
    memory_plan: MemoryPlan              # static arena + offsets
    kernels: dict[str, KernelEntry]      # symbol -> KernelEntry
    capabilities: list[str]              # capabilities this program requires of a target
    parallel: dict[int, dict] = field(default_factory=dict)   # node index -> {mode, grid, ...}
    opcodes: list[str] = field(default_factory=list)          # namespaced opcode set used

    def to_dict(self) -> dict[str, Any]:
        return {
            "abi_version": self.abi_version,
            "entry": self.entry,
            "dispatch": self.dispatch.to_dict(),
            "memory_plan": self.memory_plan.to_dict(),
            "kernels": {k: vars(v) for k, v in self.kernels.items()},
            "capabilities": list(self.capabilities),
            "parallel": {str(k): v for k, v in self.parallel.items()},
            "opcodes": list(self.opcodes),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def stats(self) -> dict[str, Any]:
        d = self.dispatch
        return {
            "n_nodes": len(d.nodes), "n_dispatches": d.n_dispatches,
            "n_views": sum(1 for n in d.nodes if n.kind == "view"),
            "n_buffers": len(d.buffers), "n_kernels": len(self.kernels),
            "arena_bytes": self.memory_plan.arena_bytes,
            "arena_reuse_factor": self.memory_plan.stats.get("reuse_factor"),
            "mallocs_eliminated": self.memory_plan.stats.get("n_intermediate_buffers"),
            "capabilities": list(self.capabilities),
        }


def build_program(dispatch: DispatchProgram, *, capability: str = "rvv",
                  abi_version: str = PROGRAM_ABI_VERSION) -> MerlinProgram:
    """Assemble a :class:`MerlinProgram` from a dispatch program: plan the arena, build the kernel
    table from the dispatch nodes, and tag the required capability. The opcode set is the distinct
    node ops (namespaced as needed by a target adapter); v1 parallel annotation is empty (the replay
    runs nodes serially; multi-core is layered in via ``parallel`` later, node-level).
    """
    mem = plan_arena(dispatch)
    kernels: dict[str, KernelEntry] = {}
    opcodes: set[str] = set()
    for node in dispatch.nodes:
        opcodes.add(node.op if node.kind == "view" else f"dispatch:{node.op}")
        if node.kind == "dispatch" and node.op not in kernels:
            roots = [v for k, v in node.prov.items() if k in ("root", "op", "name")]
            kernels[node.op] = KernelEntry(id=node.op, symbol=node.op, roots=roots,
                                           capability=capability)
    return MerlinProgram(
        abi_version=abi_version, entry=dispatch.entry, dispatch=dispatch, memory_plan=mem,
        kernels=kernels, capabilities=[capability], opcodes=sorted(opcodes))
