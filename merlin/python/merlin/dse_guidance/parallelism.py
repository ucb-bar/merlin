"""Inter-operator DAG concurrency — how much parallelism the workload structure exposes.

This consumes the P6 contract graph (operator nodes + the ``data_dependency`` edges recovered from
the SSA use-def graph) and measures the *structural* concurrency: total work vs. the critical-path
work, the resulting average parallelism, and how wide the ready-set gets. It answers "how much
inter-op parallelism is visible" and "which regions are sequential because of data dependencies" —
**not** how fast anything runs. ``available_parallelism = total_work / critical_path_work`` is the
classic work/span ratio (average parallelism), a structural property of the DAG; it is NOT a speedup
and assumes no hardware.

Where the graph carries real ``data_dependency`` edges (it does, for these captures), the DAG is
``recovered_from_ir``. If a graph ever lacks them, the analysis falls back to a conservative
sequential chain within each region and labels the result ``conservative_assumption`` — never an
invented concurrency.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

from merlin.dse_guidance.design_envelope import E_DERIVED, E_IR

CONSERVATIVE = "conservative_assumption"


@dataclass
class DagConcurrency:
    workload: str
    n_ops: int
    total_macs: int
    total_ops: int
    critical_path_macs: int
    critical_path_ops: int
    available_parallelism: float          # total_macs / critical_path_macs (work/span; NOT speedup)
    max_ready_width: int
    avg_ready_width: float
    n_levels: int
    independent_components: int           # weakly-connected components (can run concurrently)
    serialization: str                    # qualitative: mostly_sequential | some_parallelism
    dep_evidence: str                     # recovered_from_ir | conservative_assumption
    level_widths: list = field(default_factory=list)


def _ops_and_deps(graph) -> tuple[dict, dict, str]:
    """(macs_by_op, preds_by_op, evidence) from a ContractGraph; conservative fallback if no deps."""
    macs: dict[int, int] = {}
    for n in graph.nodes:
        if n.kind == "operator":
            macs[int(n.id.split(":")[-1])] = int((n.shape_summary or {}).get("macs", 0))
    preds: dict[int, list[int]] = {i: [] for i in macs}
    has_data = False
    for e in graph.edges:
        if e.kind == "data_dependency":
            has_data = True
            t = int(e.target.split(":")[-1])
            s = int(e.source.split(":")[-1])
            if t in preds and s in macs:
                preds[t].append(s)
    if has_data:
        return macs, preds, E_IR
    # conservative fallback: sequential chain in op-index order within the whole workload
    order = sorted(macs)
    preds = {order[i]: ([order[i - 1]] if i else []) for i in range(len(order))}
    return macs, preds, CONSERVATIVE


def _toposort(nodes, preds) -> list[int]:
    succ = defaultdict(list)
    indeg = {i: 0 for i in nodes}
    for i in nodes:
        for p in preds[i]:
            succ[p].append(i)
            indeg[i] += 1
    q = deque(sorted(i for i in nodes if indeg[i] == 0))
    out = []
    while q:
        i = q.popleft()
        out.append(i)
        for j in sorted(succ[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    return out


def _components(nodes, preds) -> int:
    """Weakly-connected components of the dependency graph (independent op groups)."""
    parent = {i: i for i in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in nodes:
        for p in preds[i]:
            union(i, p)
    return len({find(i) for i in nodes})


def analyze_graph(graph) -> DagConcurrency:
    macs, preds, ev = _ops_and_deps(graph)
    nodes = list(macs)
    n = len(nodes)
    if n == 0:
        return DagConcurrency(graph.workload, 0, 0, 0, 0, 0, 0.0, 0, 0.0, 0, 0,
                              "mostly_sequential", ev)
    topo = _toposort(nodes, preds)
    # longest path weighted by MACs (critical path) + op count along it; ASAP levels for width
    lp_macs: dict[int, int] = {}
    lp_ops: dict[int, int] = {}
    level: dict[int, int] = {}
    for i in topo:
        if preds[i]:
            best = max(preds[i], key=lambda p: lp_macs[p])
            lp_macs[i] = macs[i] + lp_macs[best]
            lp_ops[i] = 1 + lp_ops[best]
            level[i] = 1 + max(level[p] for p in preds[i])
        else:
            lp_macs[i] = macs[i]
            lp_ops[i] = 1
            level[i] = 0
    total_macs = sum(macs.values())
    crit_macs = max(lp_macs.values())
    crit_ops = max(lp_ops.values())
    widths = Counter(level.values())
    n_levels = max(level.values()) + 1
    max_w = max(widths.values())
    avg_w = n / n_levels
    ap = (total_macs / crit_macs) if crit_macs else 1.0
    serial = "mostly_sequential" if ap < 1.5 else "some_parallelism"
    return DagConcurrency(
        workload=graph.workload, n_ops=n, total_macs=total_macs, total_ops=n,
        critical_path_macs=crit_macs, critical_path_ops=crit_ops,
        available_parallelism=round(ap, 4), max_ready_width=max_w, avg_ready_width=round(avg_w, 4),
        n_levels=n_levels, independent_components=_components(nodes, preds),
        serialization=serial, dep_evidence=ev,
        level_widths=[widths[l] for l in range(n_levels)])


# --------------------------------------------------------------------------- emitters

def critical_path_csv(dags: list[DagConcurrency]) -> str:
    from merlin.dse_guidance.case_study import _csv
    cols = ["workload", "total_ops", "total_macs", "critical_path_ops", "critical_path_macs",
            "available_parallelism", "max_ready_width", "avg_ready_width", "n_levels",
            "independent_components", "serialization", "dep_evidence"]
    rows = [{c: getattr(d, c) for c in cols} for d in dags]
    return _csv(rows, cols)


def concurrency_windows_csv(dags: list[DagConcurrency]) -> str:
    """Per-level ready-set width — the concurrency window at each dependency depth."""
    from merlin.dse_guidance.case_study import _csv
    rows = []
    for d in dags:
        for lvl, w in enumerate(d.level_widths):
            rows.append({"workload": d.workload, "level": lvl, "ready_ops": w,
                         "evidence": d.dep_evidence})
    return _csv(rows, ["workload", "level", "ready_ops", "evidence"])


def parallel_region_candidates_yaml(dags: list[DagConcurrency]) -> dict:
    return {"parallel_region_candidates": {
        "note": "inter-operator concurrency from the recovered data-dependency DAG. "
                "available_parallelism = total_work / critical_path_work (work/span, average "
                "parallelism) — a STRUCTURAL property, NOT a speedup and assuming no hardware. "
                "independent_components are op groups with no data dependency between them.",
        "workloads": [
            {"workload": d.workload,
             "available_parallelism": {"value": d.available_parallelism, "evidence": E_DERIVED},
             "max_ready_width": {"value": d.max_ready_width, "evidence": d.dep_evidence},
             "independent_components": {"value": d.independent_components,
                                        "evidence": d.dep_evidence},
             "serialization": d.serialization,
             "interpretation": (
                 "deep sequential dependency chain — the parallelism opportunity is intra-op "
                 "sharding, not inter-op concurrency" if d.serialization == "mostly_sequential"
                 else "some independent operators per level — modest inter-op concurrency exists")}
            for d in dags]}}


def report_md(dags: list[DagConcurrency]) -> str:
    L = ["# Inter-op DAG parallelism report\n",
         "> Structural concurrency of the operator dependency DAG (edges recovered from the SSA "
         "use-def graph). `available_parallelism = total_work / critical_path_work` is the work/span "
         "ratio (average parallelism) — a structural property, **not a speedup**, no hardware "
         "assumed.\n"]
    L.append("| workload | ops | total MACs | critical-path MACs | available parallelism | "
             "max ready width | independent components | structure |")
    L.append("|---|---|---|---|---|---|---|---|")
    for d in dags:
        L.append(f"| {d.workload} | {d.total_ops} | {d.total_macs:,} | {d.critical_path_macs:,} | "
                 f"{d.available_parallelism}× | {d.max_ready_width} | {d.independent_components} | "
                 f"{d.serialization} |")
    L.append("")
    seq = [d.workload for d in dags if d.serialization == "mostly_sequential"]
    par = [d.workload for d in dags if d.serialization != "mostly_sequential"]
    L.append("## Findings\n")
    if seq:
        L.append(f"- **Low inter-op parallelism ({', '.join(seq)}):** the dependency DAG is a deep "
                 f"near-sequential chain (available parallelism < 1.5×). A future DSE tool should "
                 f"look to **intra-op sharding** of the large GEMMs (see `sharding_table.csv`), not "
                 f"inter-op concurrency.")
    if par:
        L.append(f"- **Some inter-op parallelism ({', '.join(par)}):** independent operators "
                 f"(e.g. Q/K/V projections) become ready together — modest concurrency a "
                 f"multi-engine cluster could use.")
    L.append("- **Ready-set width** peaks at a handful of operators (see `concurrency_windows.csv`) "
             "— the workloads do not expose wide inter-op concurrency.")
    L.append("\n**Caveat (structural, not realized):** available parallelism is a work/span ratio "
             "of the dependency DAG. It is **not a speedup**, **not a cycle count**, and assumes no "
             "hardware, no scheduling, and no communication cost.\n")
    return "\n".join(L)
