"""``merlin-partition-dispatches`` — schedule the dispatch DAG across harts.

The dispatch program (``dispatch_program.py``) is a pure-dataflow DAG: every buffer is
written by exactly one node, so any dependency-respecting execution order yields identical
results. That makes multicore scheduling a graph problem, not a correctness risk: place
each node on a hart and in an order such that no node runs before a node that produces one
of its inputs, and independent nodes run concurrently.

This produces a **level-synchronous schedule** — nodes grouped into dependency levels
(ASAP longest-path), a barrier between levels, and within each level the nodes balanced
across ``n_harts`` by longest-processing-time-first. Because a data dependency u→v forces
``level(v) > level(u)``, no two nodes in the same level depend on each other, so running a
whole level in parallel is always safe (`validate` proves it). The C / spike runtime walks
this schedule: each hart runs its slice of a level, then all harts sync at the barrier.

Cost model (for balancing + the makespan estimate): a matmul costs M·N·K, a batched
matmul B·M·N·K, everything else its output element count. The reported speedup is the
serial cost over the level-synchronous makespan — the parallelism actually available in
the model (intra-layer: independent q/k/v projections, attention heads, elementwise).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any

from .._common import HAS_XDSL


@dataclass
class Schedule:
    n_harts: int
    levels: list[list[int]]              # node indices per dependency level (barriers between)
    hart_of: dict[int, int]              # node index -> hart
    cost_of: dict[int, int]              # node index -> cost
    serial_cost: int                     # sum of all node costs (1 hart, no parallelism)
    critical_path_cost: int              # longest dependency chain by cost (ideal lower bound)
    makespan: int                        # level-synchronous cost with n_harts

    @property
    def speedup(self) -> float:
        return self.serial_cost / max(1, self.makespan)

    @property
    def depth(self) -> int:
        return len(self.levels)

    @property
    def max_width(self) -> int:
        return max((len(lv) for lv in self.levels), default=0)

    def hart_loads(self) -> list[int]:
        loads = [0] * self.n_harts
        for n, h in self.hart_of.items():
            loads[h] += self.cost_of[n]
        return loads


def _producers(prog) -> dict[str, int]:
    """buffer id -> index of the node that produces it (single-writer dataflow)."""
    out: dict[str, int] = {}
    for i, node in enumerate(prog.nodes):
        for b in node.outputs:
            out[b] = i
    return out


def dependencies(prog) -> list[set[int]]:
    """For each node, the set of predecessor node indices (producers of its inputs)."""
    prod_of = _producers(prog)
    deps: list[set[int]] = []
    for node in prog.nodes:
        deps.append({prod_of[b] for b in node.inputs if b in prod_of})
    return deps


def node_cost(prog, idx: int) -> int:
    node = prog.nodes[idx]
    out_elems = 1
    if node.outputs:
        out_elems = max(prod(prog.buffers[b].shape) or 1 for b in node.outputs)
    op = node.prov.get("prov.op", "") if node.kind == "dispatch" else ""
    if op in ("matmul", "batch_matmul"):
        # contraction dim K from the first input's trailing dim
        k = 1
        if node.inputs:
            shp = prog.buffers[node.inputs[0]].shape
            k = shp[-1] if shp else 1
        return max(1, out_elems * k)
    return max(1, out_elems)


def asap_levels(prog, deps: list[set[int]]) -> list[int]:
    """Longest-path level of each node (nodes are already in a topological order)."""
    level = [0] * len(prog.nodes)
    for i in range(len(prog.nodes)):
        if deps[i]:
            level[i] = 1 + max(level[p] for p in deps[i])
    return level


def schedule(prog, n_harts: int = 4) -> Schedule:
    """Level-synchronous, load-balanced schedule of the dispatch DAG over ``n_harts``."""
    deps = dependencies(prog)
    level = asap_levels(prog, deps)
    cost = {i: node_cost(prog, i) for i in range(len(prog.nodes))}

    levels: list[list[int]] = [[] for _ in range(max(level, default=-1) + 1)]
    for i, lv in enumerate(level):
        levels[lv].append(i)

    # within each level: longest-processing-time-first onto the least-loaded hart
    hart_of: dict[int, int] = {}
    makespan = 0
    for lv in levels:
        loads = [0] * n_harts
        for n in sorted(lv, key=lambda x: cost[x], reverse=True):
            h = min(range(n_harts), key=lambda x: loads[x])
            hart_of[n] = h
            loads[h] += cost[n]
        makespan += max(loads) if loads else 0

    serial = sum(cost.values())
    # critical path by cost (ideal lower bound regardless of hart count)
    cp = [0] * len(prog.nodes)
    for i in range(len(prog.nodes)):
        cp[i] = cost[i] + (max((cp[p] for p in deps[i]), default=0))
    critical = max(cp, default=0)

    return Schedule(n_harts=n_harts, levels=levels, hart_of=hart_of, cost_of=cost,
                    serial_cost=serial, critical_path_cost=critical, makespan=makespan)


def validate(prog, sched: Schedule) -> list[str]:
    """Prove the schedule is dependency-safe: every edge crosses a barrier upward.

    For each dependency u→v, ``level(u) < level(v)`` (so u runs, then a barrier, then v).
    Equivalently no two nodes in one level depend on each other — making whole-level
    parallel execution correct on a single-writer dataflow program.
    """
    problems: list[str] = []
    level_of: dict[int, int] = {}
    for lv, nodes in enumerate(sched.levels):
        for n in nodes:
            level_of[n] = lv
    deps = dependencies(prog)
    for v, preds in enumerate(deps):
        for u in preds:
            if level_of.get(u, -1) >= level_of.get(v, -1):
                problems.append(f"node {v} (level {level_of.get(v)}) depends on node {u} "
                                f"(level {level_of.get(u)}) — not strictly earlier")
    assigned = set(sched.hart_of)
    if assigned != set(range(len(prog.nodes))):
        problems.append("not every node is assigned to a hart")
    return problems


@dataclass
class PartitionResult:
    schedule: Schedule
    stats: dict[str, Any] = field(default_factory=dict)


def partition_dispatches(prog, n_harts: int = 4) -> PartitionResult:
    """Schedule + validate a dispatch program for multicore; raise on an invalid schedule."""
    if not HAS_XDSL:
        raise RuntimeError("xDSL is required")
    sched = schedule(prog, n_harts=n_harts)
    problems = validate(prog, sched)
    if problems:
        raise RuntimeError("invalid schedule: " + "; ".join(problems[:5]))
    stats = {
        "n_harts": n_harts,
        "nodes": len(prog.nodes),
        "depth": sched.depth,
        "max_width": sched.max_width,
        "serial_cost": sched.serial_cost,
        "critical_path_cost": sched.critical_path_cost,
        "makespan": sched.makespan,
        "speedup": round(sched.speedup, 2),
        "hart_loads": sched.hart_loads(),
    }
    return PartitionResult(schedule=sched, stats=stats)


def emit_schedule_c(prog, sched: Schedule, name: str = "MERLIN_SCHEDULE") -> str:
    """Emit the partitioned schedule as a C header the multicore runtime consumes.

    One ``merlin_dispatch_t`` row per dispatch node, in (level, hart) order: the kernel
    symbol, its dependency level, the hart it runs on, and the buffer ids it reads/writes.
    A level-synchronous executor (e.g. the Zephyr ``merlin_mt_rvv_dispatch`` thread pool)
    runs all rows of one level across harts, barriers, then advances — exactly the
    structure validated by :func:`validate`.
    """
    buf_index = {bid: i for i, bid in enumerate(sorted(prog.buffers))}
    rows = []
    order = sorted(range(len(prog.nodes)),
                   key=lambda n: (next(l for l, ns in enumerate(sched.levels) if n in ns),
                                  sched.hart_of[n]))
    for n in order:
        node = prog.nodes[n]
        level = next(l for l, ns in enumerate(sched.levels) if n in ns)
        ins = node.inputs + [""] * (4 - len(node.inputs))
        in_idx = ", ".join(str(buf_index.get(b, -1)) for b in ins[:4])
        out_idx = buf_index.get(node.outputs[0], -1) if node.outputs else -1
        rows.append(f'  {{ "{node.op}", {level}, {sched.hart_of[n]}, '
                    f'{{{in_idx}}}, {len(node.inputs)}, {out_idx} }},')
    lines = [
        "/* Generated by merlin.xdsl_dialects.lowering.schedule_dispatch. */",
        "#ifndef MERLIN_SCHEDULE_H", "#define MERLIN_SCHEDULE_H",
        "typedef struct { const char *kernel; int level; int hart;",
        "                 int in_buf[4]; int n_in; int out_buf; } merlin_dispatch_t;",
        f"#define {name}_N {len(rows)}",
        f"#define {name}_LEVELS {sched.depth}",
        f"#define {name}_HARTS {sched.n_harts}",
        f"static const merlin_dispatch_t {name}[{name}_N] = {{",
        *rows,
        "};", "#endif",
    ]
    return "\n".join(lines) + "\n"
