"""Small capsule witness sets for finite source obligations, without correctness claims.

An exact minimum is claimed only when singleton-witness obligations force a set
of capsules that already covers every witnessable obligation. Otherwise a
deterministic greedy set cover is reported only as an upper bound.
"""

from __future__ import annotations

SCHEMA = "merlin.phase0.witness_basis.v1"


def build_witness_basis(applications: dict, capsules: list[dict], *, coverage_status: str | None = None) -> dict:
    """Select inventoried capsule identities for observed source signatures/edges.

    The universe is deliberately limited to the operation and conditional
    typed-edge rows already computed by the Phase 0 coverage commitment.
    Other conformance facets and all compiler/target execution require their
    own evidence and are outside this set-cover calculation.
    """
    cohort = {row["name"]: row for row in capsules if row.get("status") == "inventoried"}
    universe = []
    for application, report in sorted(applications.items()):
        for kind, key in (("source_operation", "operations"), ("typed_edge_candidate", "transfers")):
            for index, row in enumerate(report.get(key) or []):
                universe.append(
                    {
                        "id": f"{application}/{key}/{index}",
                        "kind": kind,
                        "application": application,
                        "source_id": row.get("id"),
                        "role": row.get("role") if kind == "source_operation" else None,
                        "coverage_status": row.get("status"),
                        "witnesses": sorted(set(row.get("witnesses") or []) & cohort.keys()),
                    }
                )

    by_capsule = {name: set() for name in cohort}
    forced_by = {}
    witnessable = set()
    for index, row in enumerate(universe):
        if not row["witnesses"]:
            continue
        witnessable.add(index)
        for name in row["witnesses"]:
            by_capsule[name].add(index)
        if len(row["witnesses"]) == 1:
            forced_by.setdefault(row["witnesses"][0], row["id"])

    selected = set(forced_by)
    covered = set().union(*(by_capsule[name] for name in selected)) if selected else set()
    added = []
    while covered != witnessable:
        remaining = witnessable - covered
        options = [(len(members & remaining), name) for name, members in by_capsule.items() if name not in selected]
        gain, name = min(options, key=lambda item: (-item[0], item[1]))
        if gain == 0:
            raise ValueError("witnessable obligation has no admitted capsule witness")
        selected.add(name)
        added.append(name)
        covered.update(by_capsule[name])
    # Remove redundant greedy choices; forced members are a hard lower bound.
    for name in reversed(added):
        if witnessable <= set().union(*(by_capsule[other] for other in selected - {name})):
            selected.remove(name)

    exact = len(selected) == len(forced_by)
    selected_rows = [
        {
            "name": name,
            "sha256": cohort[name].get("sha256"),
            "program_sha256": cohort[name].get("program_sha256"),
            "n_witnessed_obligations": len(by_capsule[name]),
        }
        for name in sorted(selected)
    ]
    return {
        "schema": SCHEMA,
        "parent_coverage_status": coverage_status,
        "scope": (
            "source operation signature presence and conditional typed-edge presence in the selected cohort; "
            "other conformance facets are separate"
        ),
        "universe": {
            "obligations": universe,
            "n_total": len(universe),
            "n_witnessable": len(witnessable),
            "n_unwitnessed": len(universe) - len(witnessable),
        },
        "uncovered_obligations": [row for row in universe if not row["witnesses"]],
        "selection": {
            "selected_capsules": selected_rows,
            "n_selected": len(selected_rows),
            "claim": "exact_minimum" if exact else "upper_bound",
            "claim_scope": "witnessable universe only",
            "lower_bound": len(forced_by) if forced_by else int(bool(witnessable)),
            "forced_by_singleton_obligation": dict(sorted(forced_by.items())),
            "proof": (
                "Each forced capsule is the sole witness of a distinct obligation; "
                "the forced set covers every witnessable obligation."
                if exact
                else "Greedy cover of the witnessable obligations; minimum cardinality is not proven."
            ),
        },
        "formal_proof": {
            "eligibility": "not_assessed",
            "verdict": "not_proven",
            "scope": (
                "Whole-module SMT eligibility and correctness are independent of finite capsule witness coverage; "
                "this Phase 0 report does not run a formal proof."
            ),
        },
        "qualification": (
            "A capsule witnesses an observed signature or typed edge only. Selection does not establish "
            "reviewed placement, compiler lowering, source closure, target execution, or numerical correctness. "
            "No capsule is removed from the selected cohort."
        ),
    }
