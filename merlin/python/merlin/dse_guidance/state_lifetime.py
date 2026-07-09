"""State-lifetime / residency analysis — which tensors persist, and the abstraction they imply.

A flat capture shows every tensor as a one-shot value. The recovered topology
(:mod:`.topology`) already knows which state is *loop-invariant* (read-only across the K-loop),
*loop-carried* (updated each step), or *crosses the backbone -> head boundary* (produced once,
consumed K times). This module turns that recovered structure into a structured per-state table a
DSE engine can read directly: for each state, its lifetime scope, the abstraction it implies
(resident weight object, loop-carried state handle, prefix/KV object, action-chunk buffer), how
many times it is reused, and — only where the capture actually carries it — its byte size.

It invents nothing: the only byte fact a flat capture exposes is the repeated-head weight bytes
(`recovered_from_ir`); every other state's size is emitted as ``unavailable`` (``E_NA``), never a
guess. It claims no speedup, capacity, or cycle number.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.dse_guidance.design_envelope import E_FQN, E_IR, E_NA

# Lifetime scopes, strongest (most binding for residency) first.
SCOPE_CROSSES = "crosses_boundary"
SCOPE_CARRIED = "loop_carried"
SCOPE_INVARIANT = "loop_invariant"
_SCOPE_RANK = {SCOPE_CROSSES: 3, SCOPE_CARRIED: 2, SCOPE_INVARIANT: 1}

# state-name keyword -> the HW/SW abstraction object its lifetime implies. No match -> unavailable
# (we do not guess an abstraction for a state we cannot name).
_IMPLIED: tuple[tuple[tuple[str, ...], str], ...] = (
    (("weight",), "resident_weight_object"),
    (("latent", "noise", "x_t"), "loop_carried_state_handle"),
    (("kv", "prefix"), "prefix_kv_object"),
    (("feature", "embed"), "prefix_kv_object"),
    (("action", "chunk"), "action_chunk_buffer"),
)


def _implied_abstraction(state: str) -> str:
    low = state.lower()
    for keywords, abstraction in _IMPLIED:
        if any(k in low for k in keywords):
            return abstraction
    return "unavailable"


@dataclass
class StateRecord:
    state: str
    lifetime_scope: str             # loop_invariant | loop_carried | crosses_boundary
    bytes: int | None               # only known for weight states; else None -> "unavailable"
    bytes_evidence: str             # E_IR when from weight_bytes, else E_NA
    produced_by: str | None
    consumed_by: str | None
    reused_times: int | None
    implied_abstraction: str
    scope_evidence: str = E_FQN     # scopes are recovered from the prov.fqn topology


def _head_weight_bytes(attribution) -> int | None:
    head = attribution.role("repeated_head") if attribution else None
    if head and head.attribution_status == "attributed":
        return head.facts.get("weight_bytes")
    return None


def state_records(topo, attribution) -> list[StateRecord]:
    """Per-state lifetime records from the recovered topology + attribution (no invented bytes)."""
    weight_bytes = _head_weight_bytes(attribution)
    by_state: dict[str, StateRecord] = {}

    def _upsert(state: str, scope: str, produced_by=None, consumed_by=None, reused=None) -> None:
        existing = by_state.get(state)
        # keep the strongest (most binding) scope when a state shows up in several views
        if existing and _SCOPE_RANK[existing.lifetime_scope] >= _SCOPE_RANK[scope]:
            # still backfill producer/consumer/reuse if the stronger record lacks them
            if produced_by and not existing.produced_by:
                existing.produced_by = produced_by
            if consumed_by and not existing.consumed_by:
                existing.consumed_by = consumed_by
            if reused is not None and existing.reused_times is None:
                existing.reused_times = reused
            return
        # bytes only attach to weight states (the single byte fact a flat capture carries)
        is_weight = "weight" in state.lower()
        b = weight_bytes if (is_weight and weight_bytes) else None
        by_state[state] = StateRecord(
            state=state, lifetime_scope=scope, bytes=b,
            bytes_evidence=(E_IR if b is not None else E_NA),
            produced_by=produced_by, consumed_by=consumed_by, reused_times=reused,
            implied_abstraction=_implied_abstraction(state))

    # 1) boundary crossings: produced once by the backbone, consumed K times by the head.
    for c in topo.state_crossing_boundaries():
        _upsert(c["state"], SCOPE_CROSSES, produced_by=c.get("produced_by"),
                consumed_by=c.get("consumed_by"), reused=c.get("reused_times"))
    # 2) loop-invariant state read across the K-loop (reused K times).
    for s in sorted(topo.loop_invariant_state()):
        _upsert(s, SCOPE_INVARIANT, reused=topo.K)
    # 3) loop-carried state updated each step (carried, not read-only — no reuse count).
    for s in sorted(topo.loop_carried_state()):
        _upsert(s, SCOPE_CARRIED)

    # stable order: scope strength desc, then name
    return sorted(by_state.values(),
                  key=lambda r: (-_SCOPE_RANK[r.lifetime_scope], r.state))


def to_yaml_obj(records: list[StateRecord], workload: str) -> dict:
    return {"state_lifetime": {
        "workload": workload,
        "note": "lifetime scopes recovered from prov.fqn topology; only weight bytes are known "
                "from IR — other sizes are 'unavailable', never invented. No speedup/capacity claimed.",
        "states": [
            {"state": r.state, "lifetime_scope": r.lifetime_scope,
             "bytes": r.bytes, "bytes_evidence": r.bytes_evidence,
             "produced_by": r.produced_by, "consumed_by": r.consumed_by,
             "reused_times": r.reused_times, "implied_abstraction": r.implied_abstraction,
             "scope_evidence": r.scope_evidence}
            for r in records],
    }}


def resident_state_csv(packages) -> str:
    """Cross-workload resident-state table from the accumulated case packages."""
    from merlin.dse_guidance.corpus import _csv
    rows = []
    for p in packages:
        for r in p.get("state", []):
            rows.append({
                "workload": p["case"].workload, "state": r.state,
                "lifetime_scope": r.lifetime_scope,
                "bytes": r.bytes if r.bytes is not None else "unavailable",
                "bytes_evidence": r.bytes_evidence,
                "produced_by": r.produced_by or "unavailable",
                "consumed_by": r.consumed_by or "unavailable",
                "reused_times": r.reused_times if r.reused_times is not None else "unavailable",
                "implied_abstraction": r.implied_abstraction,
                "scope_evidence": r.scope_evidence,
            })
    return _csv(rows, ["workload", "state", "lifetime_scope", "bytes", "bytes_evidence",
                       "produced_by", "consumed_by", "reused_times", "implied_abstraction",
                       "scope_evidence"])
