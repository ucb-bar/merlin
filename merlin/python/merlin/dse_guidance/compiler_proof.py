"""Compiler-proof / transformability matrix — what must be proven to exploit each abstraction.

A future DSE engine should not explore an abstraction the compiler cannot actually target. Each
DSE candidate already carries the *compiler proof* it would require (the immutability / bounded-K /
layout-persistence fact a compiler must establish — see :data:`candidates.CANDIDATE_CATALOG` and
:class:`candidates.DseCandidate`). This module does not invent new proofs; it **organizes the
existing ones into a cross-workload matrix** and attaches an honest *status*:

  proven_for_workload  — the capture/topology already establishes it (recovered_from_prov_fqn)
  assumed              — it rests on an assumed reference (e.g. K is a reference value)
  unknown              — the capture erased what the proof needs (e.g. packed layout dequantized)

Status is taken as the *weakest* observed across workloads, so the matrix never over-claims. No
speedup, cycle, or capacity number appears.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.dse_guidance.candidates import CANDIDATE_CATALOG
from merlin.dse_guidance.design_envelope import E_ASSUMED, E_FQN, E_NA

# Axes whose proof needs numerical structure the flat (dequantized) capture erased -> "unknown".
_LAYOUT_LOWBIT_AXES = frozenset({
    "packed_layout_preservation", "native_lowbit_compute", "resident_packed_lowbit_weights",
    "fused_dequant_matmul", "fused_requant_epilogue", "quantized_KV_cache",
})

# Status strength (weakest first) — aggregation takes the minimum across workloads.
_STATUS_RANK = {"unknown": 0, "assumed": 1, "proven_for_workload": 2}
_STATUS_EVIDENCE = {"unknown": E_NA, "assumed": E_ASSUMED, "proven_for_workload": E_FQN}


def proof_status(axis: str, case) -> str:
    """Honest per-(axis, workload) status derived only from existing recovered signals."""
    if axis in _LAYOUT_LOWBIT_AXES:
        return "unknown"                       # captures are dequantized -> layout/low-bit erased
    topo, attr = case.topo, case.attribution
    head = attr.role("repeated_head") if attr else None
    head_ok = head is not None and head.attribution_status == "attributed"
    if axis == "resident_action_head_weights" and head_ok \
            and any("weight" in s.lower() for s in topo.loop_invariant_state()):
        return "proven_for_workload"           # weights immutable across K recovered from prov.fqn
    if axis == "resident_prefix_kv" and topo.state_crossing_boundaries():
        return "proven_for_workload"           # prefix/KV crossing recovered from the topology
    return "assumed"                           # rests on assumed K / reference deadline


@dataclass
class ProofRow:
    axis: str
    system_abstraction: str
    compiler_proof_needed: str
    status: str
    status_evidence: str
    workloads_suggesting: list[str] = field(default_factory=list)


def proof_matrix(packages) -> list[ProofRow]:
    """Aggregate the per-axis compiler proofs across workloads (weakest status wins)."""
    by_axis: dict[str, ProofRow] = {}
    for p in packages:
        case = p["case"]
        for c in p.get("cands", []):
            st = proof_status(c.axis, case)
            row = by_axis.get(c.axis)
            if row is None:
                by_axis[c.axis] = ProofRow(
                    axis=c.axis, system_abstraction=c.system_abstraction,
                    compiler_proof_needed=c.compiler_proof, status=st,
                    status_evidence=_STATUS_EVIDENCE[st], workloads_suggesting=[case.workload])
            else:
                if case.workload not in row.workloads_suggesting:
                    row.workloads_suggesting.append(case.workload)
                if _STATUS_RANK[st] < _STATUS_RANK[row.status]:    # weakest across workloads
                    row.status = st
                    row.status_evidence = _STATUS_EVIDENCE[st]
    return sorted(by_axis.values(), key=lambda r: (-len(r.workloads_suggesting), r.axis))


def compiler_proof_csv(packages) -> str:
    from merlin.dse_guidance.case_study import _csv
    rows = [{
        "axis": r.axis, "system_abstraction": r.system_abstraction,
        "compiler_proof_needed": r.compiler_proof_needed, "status": r.status,
        "status_evidence": r.status_evidence,
        "workloads_suggesting": "; ".join(r.workloads_suggesting),
    } for r in proof_matrix(packages)]
    return _csv(rows, ["axis", "system_abstraction", "compiler_proof_needed", "status",
                       "status_evidence", "workloads_suggesting"])


def catalog_proof(axis: str) -> str | None:
    """The canonical proof string for a structural axis (used by tests to assert nothing invented)."""
    cat = CANDIDATE_CATALOG.get(axis)
    return cat["compiler_proof"] if cat else None
