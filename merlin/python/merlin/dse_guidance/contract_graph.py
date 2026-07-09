"""Multi-rate workload contract graph — the central IR later phases consume.

A flat dataflow graph says ``op1 -> op2 -> op3``. A DSE engine needs the *multi-rate* structure a
flat capture erases: the backbone runs once per replan, the action head runs K times, a decoder
runs token-by-token, the control loop consumes H actions at a fixed rate, some state persists
across the K-loop, some state is carried, some operators are shardable. This module assembles that
structure into one typed graph (nodes + edges) by joining everything the earlier phases recovered:

  * :mod:`.topology` / :mod:`.temporal` — phases, roles, K/H/control rate, loop-invariant/carried state
  * :mod:`.attribution`                 — per-region real IR facts (MACs, bytes, dispatch proxy)
  * :mod:`.operator_geometry` (P5)       — per-operator shapes / classes
  * :mod:`.numerical_contract`           — per-region dtype + accumulator
  * :mod:`.state_lifetime`               — state objects + their lifetime scopes
  * :mod:`.phase_rate`                   — cadence classification + rate constants

**Dependencies are recovered, not guessed.** The flat ``model.mlir`` is a real SSA dataflow IR, so
the true operator data dependencies are *in* the capture: ``attribution.matmul_dependencies`` walks
the use-def chains to recover, for each matmul, which earlier matmul results feed it — emitted as
``data_dependency`` edges (``recovered_from_ir``, ``can_pipeline=False``). The once-per-replan→
repeated-head ordering is a recovered ``control_dependency``; loop-invariant / loop-carried /
boundary-crossing state edges come from the recovered state lifetimes; the cross-replan
backbone/head overlap is a ``pipeline_candidate`` derived from the absence of shared loop-carried
state. No speedup, cycle, or area claim anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.dse_guidance import phase_rate as PR
from merlin.dse_guidance.design_envelope import E_CONFIG, E_DERIVED, E_FQN, E_IR, E_NA
from merlin.dse_guidance.state_lifetime import SCOPE_CARRIED, SCOPE_CROSSES, SCOPE_INVARIANT

# Node kinds.
KIND_OPERATOR = "operator"
KIND_REGION = "region"
KIND_PHASE = "phase"
KIND_LOOP_BODY = "loop_body"
KIND_PIPELINE_STAGE = "pipeline_stage"
KIND_RUNTIME_COMMAND = "runtime_command"
KIND_STATE_OBJECT = "state_object"
NODE_KINDS = (KIND_OPERATOR, KIND_REGION, KIND_PHASE, KIND_LOOP_BODY, KIND_PIPELINE_STAGE,
              KIND_RUNTIME_COMMAND, KIND_STATE_OBJECT)

# Edge kinds.
EDGE_DATA = "data_dependency"
EDGE_CONTROL = "control_dependency"
EDGE_STATE_LIFETIME = "state_lifetime"
EDGE_LOOP_CARRIED = "loop_carried"
EDGE_LOOP_INVARIANT = "loop_invariant"
EDGE_PIPELINE_CANDIDATE = "pipeline_candidate"
EDGE_COMMAND = "command_dependency"
EDGE_UNKNOWN = "unknown_dependency"
EDGE_KINDS = (EDGE_DATA, EDGE_CONTROL, EDGE_STATE_LIFETIME, EDGE_LOOP_CARRIED, EDGE_LOOP_INVARIANT,
              EDGE_PIPELINE_CANDIDATE, EDGE_COMMAND, EDGE_UNKNOWN)


@dataclass
class Node:
    id: str
    workload: str
    kind: str
    region_role: str | None = None
    parent_region: str | None = None
    operator_refs: list[str] = field(default_factory=list)
    rate: dict | None = None
    shape_summary: dict | None = None
    work_summary: dict | None = None
    state_summary: dict | None = None
    numerical_contract_summary: dict | None = None
    parallelism_summary: dict | None = None
    evidence: str = E_NA
    note: str = ""


@dataclass
class Edge:
    source: str
    target: str
    kind: str
    tensor: str | None = None
    bytes: int | None = None
    lifetime: str | None = None
    can_pipeline: object = "unknown"      # True | False | "unknown"
    evidence: str = E_NA
    note: str = ""


@dataclass
class ContractGraph:
    workload: str
    workload_class: str
    rate_model: dict
    nodes: list[Node]
    edges: list[Edge]


def _shardable(op_kind: str | None) -> dict:
    """Conservative structural parallelism summary for a matmul-like op (no perf claim)."""
    return {"shardable_dim": "N", "shardable_evidence": E_DERIVED,
            "region_overlap": "unknown", "overlap_evidence": E_NA}


def build_graph(case, shapes, nc, state_records, dependencies=None) -> ContractGraph:
    wl = case.workload
    topo = case.topo
    K = int(case.K)
    nodes: list[Node] = []
    edges: list[Edge] = []
    rm = PR.rate_model(topo)
    deps = dependencies or ()      # per-matmul real data deps (recovered_from_ir), indexed by op

    # ---- phase nodes (from the recovered topology regions) ----
    phase_by_role: dict[str, str] = {}
    for r in topo.temporal.regions:
        cad = PR.classify_cadence(r.role, topo.workload_class, r.invocation_count, K)
        period, pev = PR.phase_period_s(cad, topo)
        invs = r.invocation_count or (K if r.role == "repeated_head" else 1)
        pid = f"{wl}:phase:{r.name}"
        nodes.append(Node(
            id=pid, workload=wl, kind=KIND_PHASE, region_role=r.role,
            rate={"cadence": cad, "cadence_evidence": E_FQN, "invocations": invs,
                  "invocations_evidence": (E_CONFIG if r.role == "repeated_head" else E_IR),
                  "period_s": period, "period_evidence": pev},
            state_summary={"loop_invariant_state": list(r.loop_invariant_state),
                           "loop_carried_state": list(r.loop_carried_state),
                           "produces": list(r.produces), "consumes": list(r.consumes)},
            evidence=E_FQN))
        if r.role:
            phase_by_role[r.role] = pid

    # ---- region nodes (from attribution) ----
    region_dtype = {d["region"]: d for d in (nc.per_region_dtype if nc else [])}
    for reg in case.attribution.regions:
        role = reg.role
        rid = f"{wl}:region:{role}"
        f = reg.facts
        attributed = reg.attribution_status == "attributed"
        cad = PR.classify_cadence(role, topo.workload_class, f.get("invocations"), K)
        ncs = region_dtype.get(role)
        nodes.append(Node(
            id=rid, workload=wl, kind=KIND_REGION, region_role=role,
            parent_region=phase_by_role.get(role),
            operator_refs=[f"{wl}:op:{i}" for i in reg.matmul_indices],
            rate={"cadence": cad, "invocations": f.get("invocations"),
                  "invocations_evidence": (E_CONFIG if role == "repeated_head" else E_IR)},
            work_summary={"matmul_count": f.get("matmul_count"),
                          "macs_per_invocation": f.get("macs_per_invocation"),
                          "macs_per_replan": f.get("macs_total"),
                          "weight_bytes": f.get("weight_bytes"),
                          "activation_bytes_per_invocation": f.get("activation_bytes_per_invocation"),
                          "activation_bytes_per_replan": f.get("activation_bytes_total"),
                          "dispatch_count_proxy": f.get("matmul_count"),
                          "evidence": (E_IR if attributed else E_NA)},
            numerical_contract_summary=(
                {"dtype": ncs["dtype"], "n_matmuls": ncs["n"],
                 "accumulator_dtype": (nc.accumulator_dtype if nc else None), "evidence": E_IR}
                if ncs else {"dtype": "unavailable", "evidence": E_NA}),
            parallelism_summary=_shardable("matmul"),
            evidence=(E_FQN if attributed else E_NA),
            note=("" if attributed else "role not recoverable from the flat capture")))


    # ---- operator nodes (from P5 geometry) ----
    for s in shapes:
        nodes.append(Node(
            id=f"{wl}:op:{s.op_index}", workload=wl, kind=KIND_OPERATOR,
            region_role=s.region_role, parent_region=f"{wl}:region:{s.region_role}",
            shape_summary={"op_kind": s.op_kind, "M": s.M, "N": s.N, "K": s.K, "macs": s.macs,
                           "shape_class": s.shape_class, "semantic_class": s.semantic_class,
                           "rhs_weight_bytes": s.rhs_weight_bytes, "output_bytes": s.output_bytes,
                           "dtype": s.dtype, "attention_or_conv": "unavailable",
                           "primitive_coverage_ref": "tile_waste_table.csv", "evidence": E_IR},
            work_summary={"macs": s.macs, "evidence": E_IR},
            numerical_contract_summary={"dtype": s.dtype, "evidence": s.evidence_dtype},
            parallelism_summary=_shardable(s.op_kind),
            evidence=E_IR))
        # real data-dependency edges recovered from the SSA use-def graph (producer -> this op)
        for j in (deps[s.op_index] if s.op_index < len(deps) else ()):
            edges.append(Edge(
                source=f"{wl}:op:{j}", target=f"{wl}:op:{s.op_index}", kind=EDGE_DATA,
                can_pipeline=False, evidence=E_IR,
                note="result of op feeds this op (recovered from SSA use-def)"))

    # ---- loop-body node for the repeated head ----
    head = case.attribution.role("repeated_head")
    if head and head.attribution_status == "attributed" and K > 1:
        nodes.append(Node(
            id=f"{wl}:loop:repeated_head", workload=wl, kind=KIND_LOOP_BODY,
            region_role="repeated_head", parent_region=phase_by_role.get("repeated_head"),
            operator_refs=[f"{wl}:op:{i}" for i in head.matmul_indices],
            rate={"cadence": PR.classify_cadence("repeated_head", topo.workload_class,
                                                 head.facts.get("invocations"), K),
                  "trip_count": K, "trip_count_evidence": E_CONFIG},
            work_summary={"macs_per_invocation": head.facts.get("macs_per_invocation"),
                          "macs_per_replan": head.facts.get("macs_total"), "evidence": E_IR},
            evidence=E_FQN))

    # ---- state-object nodes + their lifetime edges ----
    head_rid = f"{wl}:region:repeated_head"
    for st in state_records:
        sid = f"{wl}:state:{st.state}"
        nodes.append(Node(
            id=sid, workload=wl, kind=KIND_STATE_OBJECT,
            state_summary={"lifetime_scope": st.lifetime_scope, "bytes": st.bytes,
                           "bytes_evidence": st.bytes_evidence, "reused_times": st.reused_times,
                           "implied_abstraction": st.implied_abstraction,
                           "produced_by": st.produced_by, "consumed_by": st.consumed_by},
            evidence=st.scope_evidence))
        if st.lifetime_scope == SCOPE_INVARIANT:
            edges.append(Edge(sid, head_rid, EDGE_LOOP_INVARIANT, tensor=st.state, bytes=st.bytes,
                              lifetime="loop_invariant", can_pipeline=True,
                              evidence=(st.bytes_evidence if st.bytes else st.scope_evidence),
                              note="read-only across the K-loop; safe to keep resident"))
        elif st.lifetime_scope == SCOPE_CARRIED:
            edges.append(Edge(sid, head_rid, EDGE_LOOP_CARRIED, tensor=st.state, bytes=st.bytes,
                              lifetime="loop_carried", can_pipeline=False, evidence=st.scope_evidence,
                              note="updated each step; serializes the loop w.r.t. this state"))
        else:  # crosses boundary
            src = phase_by_role.get("backbone_once", sid)
            edges.append(Edge(src, head_rid, EDGE_STATE_LIFETIME, tensor=st.state, bytes=st.bytes,
                              lifetime="crosses_boundary", can_pipeline="unknown",
                              evidence=st.scope_evidence,
                              note="produced once per replan, consumed across the K-loop"))

    # ---- phase-level control + pipeline-candidate edges ----
    bb = phase_by_role.get("backbone_once")
    hd = phase_by_role.get("repeated_head")
    if bb and hd:
        if not topo.state_crossing_boundaries():
            edges.append(Edge(
                bb, hd, EDGE_CONTROL, can_pipeline="unknown", evidence=E_FQN,
                note="once-per-replan backbone precedes the repeated head (role split recovered "
                     "from prov.fqn); the specific tensors crossing the boundary are unavailable "
                     "in the flat capture"))
        edges.append(Edge(
            hd, bb, EDGE_PIPELINE_CANDIDATE, can_pipeline=True, evidence=E_DERIVED,
            note="the next replan's backbone shares no loop-carried state with this replan's head, "
                 "so the two rates are structurally overlappable (async pipelining candidate)"))

    return ContractGraph(workload=wl, workload_class=topo.workload_class, rate_model=rm,
                         nodes=nodes, edges=edges)


# --------------------------------------------------------------------------- emitters

def _node_obj(n: Node) -> dict:
    d = {"id": n.id, "kind": n.kind, "region_role": n.region_role,
         "parent_region": n.parent_region, "evidence": n.evidence}
    for k in ("operator_refs", "rate", "shape_summary", "work_summary", "state_summary",
              "numerical_contract_summary", "parallelism_summary"):
        v = getattr(n, k)
        if v:
            d[k] = v
    if n.note:
        d["note"] = n.note
    return d


def _edge_obj(e: Edge) -> dict:
    d = {"source": e.source, "target": e.target, "kind": e.kind,
         "can_pipeline": e.can_pipeline, "evidence": e.evidence}
    if e.tensor is not None:
        d["tensor"] = e.tensor
    if e.bytes is not None:
        d["bytes"] = e.bytes
    if e.lifetime is not None:
        d["lifetime"] = e.lifetime
    if e.note:
        d["note"] = e.note
    return d


def to_yaml_obj(graphs: list[ContractGraph]) -> dict:
    return {"workload_contract_graph": {
        "note": "multi-rate workload contract graph. Phases/roles/state recovered from prov.fqn "
                "topology; operator/region facts and operator-to-operator data dependencies "
                "recovered_from_ir (the SSA use-def graph); rate constants (K/H/control rate) "
                "recovered_from_model_config; the replan deadline is derived. No speedup claimed.",
        "node_kinds": list(NODE_KINDS),
        "edge_kinds": list(EDGE_KINDS),
        "graphs": [
            {"workload": g.workload, "workload_class": g.workload_class,
             "rate_model": g.rate_model,
             "nodes": [_node_obj(n) for n in g.nodes],
             "edges": [_edge_obj(e) for e in g.edges]}
            for g in graphs],
    }}


def _counts(g: ContractGraph) -> dict:
    from collections import Counter
    nk = Counter(n.kind for n in g.nodes)
    ek = Counter(e.kind for e in g.edges)
    known = sum(1 for e in g.edges if e.evidence != E_NA)
    return {"nodes": len(g.nodes), "edges": len(g.edges),
            "node_kinds": dict(nk), "edge_kinds": dict(ek),
            "edges_known": known, "edges_unknown": len(g.edges) - known}


def phase_rate_csv(graphs: list[ContractGraph]) -> str:
    from merlin.dse_guidance.corpus import _csv
    rows = []
    for g in graphs:
        for n in g.nodes:
            if n.kind != KIND_PHASE:
                continue
            r = n.rate or {}
            rows.append({"workload": g.workload, "phase": n.id.split(":")[-1],
                         "role": n.region_role, "cadence": r.get("cadence"),
                         "invocations": r.get("invocations"),
                         "period_s": r.get("period_s") if r.get("period_s") is not None
                         else "unavailable",
                         "period_evidence": r.get("period_evidence"),
                         "cadence_evidence": r.get("cadence_evidence")})
    return _csv(rows, ["workload", "phase", "role", "cadence", "invocations", "period_s",
                       "period_evidence", "cadence_evidence"])


def multi_rate_contract_yaml(graphs: list[ContractGraph]) -> dict:
    out = []
    for g in graphs:
        phases = [{"phase": n.id.split(":")[-1], "role": n.region_role,
                   "cadence": (n.rate or {}).get("cadence"),
                   "invocations": (n.rate or {}).get("invocations")}
                  for n in g.nodes if n.kind == KIND_PHASE]
        crossings = [{"state": e.tensor, "lifetime": e.lifetime, "bytes": e.bytes,
                      "evidence": e.evidence}
                     for e in g.edges
                     if e.kind in (EDGE_STATE_LIFETIME, EDGE_LOOP_INVARIANT, EDGE_LOOP_CARRIED)]
        out.append({"workload": g.workload, "workload_class": g.workload_class,
                    "rate_model": g.rate_model, "phases": phases,
                    "states_across_rates": crossings})
    return {"multi_rate_contract": {
        "note": "rate constants (K/H/control_rate) are recovered_from_model_config (the model's "
                "published architecture); the replan deadline is derived from H / control_rate. "
                "No speedup/cycle claim.",
        "workloads": out}}


def summary_md(graphs: list[ContractGraph]) -> str:
    L = ["# Workload contract graph — summary\n",
         "> The multi-rate workload contract graph: phases (cadence), regions (real IR facts), "
         "operators (P5 geometry), and state objects (lifetimes), with typed edges. Operator data "
         "dependencies are recovered from the SSA use-def graph (`data_dependency`, "
         "`recovered_from_ir`). **No speedup / cycle / area claim.**\n"]
    L.append("## Graph size per workload\n")
    L.append("| workload | class | nodes | edges | phase | region | operator | loop_body | "
             "state | data-dep edges | edges w/ recovered evidence |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in graphs:
        c = _counts(g)
        nk = c["node_kinds"]
        L.append(f"| {g.workload} | {g.workload_class} | {c['nodes']} | {c['edges']} | "
                 f"{nk.get(KIND_PHASE,0)} | {nk.get(KIND_REGION,0)} | {nk.get(KIND_OPERATOR,0)} | "
                 f"{nk.get(KIND_LOOP_BODY,0)} | {nk.get(KIND_STATE_OBJECT,0)} | "
                 f"{c['edge_kinds'].get(EDGE_DATA,0)} | {c['edges_known']} |")
    L.append("")
    L.append("## Repeated structure (which workloads have a K-loop / token-loop)\n")
    L.append("| workload | repeated head | cadence | trip count (K) | loop-invariant state |")
    L.append("|---|---|---|---|---|")
    for g in graphs:
        loop = next((n for n in g.nodes if n.kind == KIND_LOOP_BODY), None)
        inv = [e.tensor for e in g.edges if e.kind == EDGE_LOOP_INVARIANT]
        cad = (loop.rate or {}).get("cadence") if loop else "—"
        K = (loop.rate or {}).get("trip_count") if loop else "—"
        L.append(f"| {g.workload} | {'yes' if loop else 'no'} | {cad} | {K} | "
                 f"{', '.join(inv) if inv else '—'} |")
    L.append("")
    L.append("## What downstream DSE phases can now consume\n")
    L.append("- **Phase/rate scheduling:** per-phase cadence + the rate model (K/H/control rate, "
             "derived replan deadline) — enough to reason about the once-vs-K-vs-control rate split.")
    L.append("- **Residency:** loop-invariant state edges (weights) carry byte size + reuse count.")
    L.append("- **Partition:** region nodes carry per-region MACs/bytes and the recovered "
             "backbone/head split.")
    L.append("- **Primitive sizing:** operator nodes link to the P5 shape classes / coverage table.")
    L.append("- **Scheduling/overlap:** real `data_dependency` edges (from the SSA use-def graph) "
             "give the true intra-phase ordering, and the cross-replan `pipeline_candidate` edge "
             "marks the backbone/head overlap the rate split permits.")
    L.append("\nEvery node and edge carries a provenance label: structure and data dependencies are "
             "`recovered_from_ir`, roles/cadence `recovered_from_prov_fqn`, the rate constants "
             "`recovered_from_model_config`, the replan deadline `derived_requirement`.\n")
    return "\n".join(L)


def rate_mismatch_report_md(graphs: list[ContractGraph]) -> str:
    L = ["# Rate-mismatch report\n",
         "> The multi-rate contract makes the rate mismatches explicit: a backbone that runs once "
         "per replan, an action head that runs K times, and a control loop that consumes H actions "
         "at a fixed frequency are three different rates the flat capture collapsed into one. "
         "**Structural only — no speedup, no cycle budget claimed.**\n"]
    L.append("## Per-workload rate structure\n")
    L.append("| workload | backbone | repeated head | K | H | control_rate_hz | replan deadline (s) |")
    L.append("|---|---|---|---|---|---|---|")
    for g in graphs:
        rm = g.rate_model
        has_bb = any(n.region_role == "backbone_once" and n.kind == KIND_PHASE for n in g.nodes)
        loop = next((n for n in g.nodes if n.kind == KIND_LOOP_BODY), None)
        head_cad = (loop.rate or {}).get("cadence") if loop else "—"
        dl = rm["replan_deadline_s"]["value"]
        L.append(f"| {g.workload} | {'once_per_replan' if has_bb else '—'} | "
                 f"{head_cad} | {rm['K']['value']} | {rm['H']['value']} | "
                 f"{rm['control_rate_hz']['value']} | {dl if dl is not None else 'unavailable'} |")
    L.append("")
    L.append("## Provenance of every field\n")
    L.append("- **`recovered_from_ir`:** region roles' MACs/bytes, operator shapes, loop-invariant "
             "weight bytes, and the **operator data-dependency edges** (from the SSA use-def graph).")
    L.append("- **`recovered_from_prov_fqn`:** the region roles and the once-vs-repeated cadence "
             "split (backbone once, head repeated).")
    L.append("- **`recovered_from_model_config`:** K, H, control_rate_hz (the model's published "
             "architecture constants, from the model registry).")
    L.append("- **`derived_requirement`:** the replan deadline (= H / control_rate) and the "
             "cross-replan pipeline-candidate overlap.")
    L.append("")
    L.append("## Dependency knowledge (fully recovered)\n")
    for g in graphs:
        ek = _counts(g)["edge_kinds"]
        L.append(f"- **{g.workload}:** {ek.get(EDGE_DATA,0)} `data_dependency` edges recovered from "
                 f"the SSA use-def graph, plus the backbone→head `control_dependency`, the "
                 f"loop-invariant weight edge, and the cross-replan `pipeline_candidate`. Every "
                 f"edge carries a recovered/derived evidence label.")
    L.append("\nThe graph is a complete **structural** multi-rate contract: every node and edge is "
             "recovered from the capture, the model config, or derived from them — what runs at "
             "which rate, which state persists, and which operator feeds which. Per-phase wall-clock "
             "timing is a runtime *measurement* (orthogonal to this static contract), not a missing "
             "structural fact.\n")
    return "\n".join(L)
