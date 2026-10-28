"""Retire a pending certificate record whose own job finished without an attributable result.

``promote()`` marks a capsule's cert tier ``pending`` when it enqueues the job, and the scheduler treats a
pending record as known -- an in-flight job must not be bought twice. That is only right while the job is
in flight. When the job completes and ``record_cert`` cannot attribute its result to the record (the
boundary join refused it, the result names no execution identity the record can be checked against, or it
states no verdict), nothing else will ever resolve that record, so the capsule read "pending" for the rest
of the run and the scheduler never asked for its cert tier again. Measured on the fresh6 campaign: 101
capsules stranded at L3 pending, and from about +90 min every promotion said "0 capsule(s) want L3".

Retiring is narrow on purpose. It needs the completed job's own request id, so only a completion that
provably belongs to THIS record retires it; a replay of an older job (``completions.recover``) cannot
retire a newer record for the same bytes. The retired record keeps every field it had and is not a
verdict: :data:`~merlin.targetgen.oracle_schedule.UNATTRIBUTED_RESULT` reads as "no verdict" to the
scheduler, so the next loop verdict for those bytes re-enqueues them under a new request id.
"""
from __future__ import annotations

import time


def retire(st, name, tier, key, *, request, reason, completed=None, log=None) -> bool:
    """Mark ``st[name][tier]`` record ``key`` unattributed when it is the pending record ``request`` was
    enqueued for. Returns whether it did. ``st`` is modified in place; the caller saves it."""
    from merlin.targetgen.oracle_schedule import UNATTRIBUTED_RESULT

    from . import promotion as P

    if not request or key is None:
        return False
    slots = P._slots_ro(st, name, tier)
    entry = slots.get(str(key))
    # Legacy records without a request cannot establish completion ownership.
    if (not isinstance(entry, dict) or entry.get("status") != "pending"
            or entry.get("request") != request):
        return False
    retired = dict(entry, status=UNATTRIBUTED_RESULT, unattributed={
        "request": request, "reason": reason, "completed_execution_digest": completed, "at": time.time()})
    P._slots(st, name, tier)[str(key)] = retired
    mirror = (st.get(name) or {}).get(tier)
    if isinstance(mirror, dict) and P.record_identity(mirror) == str(key):
        st[name][tier] = retired
    if log is not None:
        print(f"[promote] {name} {tier} pending record retired for re-queue: job {request} finished "
              f"but its result could not be attributed ({reason})", file=log, flush=True)
    return True


def completed_request(function):
    """Retire unresolved records after the original recorder, inside its ledger transaction.

    The broker supplies the completed request id; no result field can select it. Recovery of an older
    completion cannot retire a retry because each attempt has a distinct id recorded on its slot.
    """
    from functools import wraps

    @wraps(function)
    def record(ws, verdict, cert_tier, log=None, identity=None, source_identity_verified=False, *, request=None):
        from . import promotion as P
        from .promotion_verdicts import hydrated

        verdict = hydrated(ws, verdict)  # the full row carries the L3 boundary the join needs

        if request and identity is not None:
            state = P._tier_state(ws)
            rows = []
            for row in (verdict or {}).get("per_capsule") or ():
                entry = P._slots_ro(state, row.get("capsule"), cert_tier).get(str(identity))
                if (isinstance(entry, dict) and entry.get("status") == "pending"
                        and entry.get("request") not in (None, request)):
                    continue
                rows.append(row)
            verdict = {**(verdict or {}), "per_capsule": rows}
        resolved = function(ws, verdict, cert_tier, log, identity, source_identity_verified)
        if not request or identity is None:
            return resolved
        state = P._tier_state(ws)
        changed = False
        for name in state:
            changed |= retire(state, name, cert_tier, identity, request=request,
                              reason="completed job supplied no attributable certificate", log=log)
        if changed:
            P._save_tier_state(ws, state)
        return resolved

    return record


def requeue_attempt(slots, key) -> int | None:
    """``None`` when ``key`` already holds a record that must not be re-bought (pending, pass, fail,
    joined), else the attempt number a new request for it carries: 0 for a first request, n+1 after a
    retired one, so its request id differs from every earlier request for the same bytes."""
    from merlin.targetgen.oracle_schedule import UNATTRIBUTED_RESULT

    entry = slots.get(key)
    if entry is None:
        return 0
    if isinstance(entry, dict) and entry.get("status") == UNATTRIBUTED_RESULT:
        return int(entry.get("attempt") or 0) + 1
    return None
