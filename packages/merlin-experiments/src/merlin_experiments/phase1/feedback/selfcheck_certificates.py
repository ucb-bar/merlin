"""Keep the certificate tier an author self-check already executed on its exact snapshot.

When a self-check's own selection ran the cert tier, the broker skips promotion -- re-running those
bytes buys the same simulation twice -- but nothing recorded the result either, so a passing gSIM L3
run of a required case (whole models among them) never reached the physical-correctness ledger and
the grade kept it ``unverified``. This records it, under the promotion ledger's own rules:

* the certificate belongs to the source snapshot the broker verified the check graded
  (``source_ws``), by its whole-submission digest and, when the row carries one, its execution
  identity -- never to the live, possibly edited, workspace;
* only rows that state a passed, certified cert-tier barrier, run on the engine a promotion would
  use and under the instrument a promotion would record, are imported;
* bytes that already hold a cert-tier record (outstanding, passed or failed) are left alone;
* the result is written by ``record_cert`` against a record created for exactly those bytes, so
  attribution is the recorder's, not this module's.

Joint-campaign required cases only. Like promotion, it is an optimisation and never gates a run.
"""

from __future__ import annotations

from pathlib import Path


def _rows(verdict, cert_tier, roster):
    for row in verdict.get("per_capsule") or ():
        if (isinstance(row, dict) and row.get("capsule") in roster and row.get("pass") is True
                and row.get("certified_complete") is True and row.get("screened_only") is not True
                and row.get("barrier_tier") == cert_tier and row.get("barrier_status") == "pass"):
            yield row


def import_executed(ws, verdict, *, loop_tier, cert_tier, source_ws, context, log=None) -> list[str]:
    """Record the cert-tier passes ``verdict`` already executed; returns ``record_cert``'s lines."""
    from .. import joint_control
    from . import joint_physical_correctness as physical
    from . import promotion as P

    roster = physical.roster() if joint_control.selected() is not None else set()
    if (not roster or (loop_tier, cert_tier) != ("L2", "L3") or not isinstance(verdict, dict)
            or verdict.get("error") or not P.cert_already_executed(verdict, cert_tier)):
        return []
    rows = list(_rows(verdict, cert_tier, roster))
    if not rows:
        return []
    engine = (verdict.get("tier_engines") or {}).get(cert_tier)
    sim = P.cert_sim(cert_tier, context=context)
    instrument = P.cert_instrument(cert_tier, context=context)
    if sim is None or engine != sim or instrument is None:
        if log is not None:
            print(f"[promote] self-check {cert_tier} not imported: ran on {engine!r}, a promotion would use "
                  f"{sim!r}" + ("" if instrument else "; certificate instrument unavailable"), file=log, flush=True)
        return []
    return P._ledger_transaction(_record)(ws, rows, cert_tier=cert_tier, instrument=instrument,
                                          source_ws=Path(source_ws), log=log)


def _record(ws, rows, *, cert_tier, instrument, source_ws, log):
    from . import promotion as P

    digest, comps, _ = P.submission_digests(source_ws)
    state = P._tier_state(ws)
    staged = []
    for row in rows:
        name = str(row["capsule"])
        execution = row.get("execution_digest") if P._valid_execution(row.get("execution_digest")) else None
        identity = P.current_identity(execution, digest)
        if identity in P._slots_ro(state, name, cert_tier):
            continue  # these bytes already hold a record; a running promotion resolves its own
        entry = {"status": "pending", "digest": digest, "components": dict(comps), "instrument": instrument,
                 "origin": "author_selfcheck"}
        if execution is not None:
            entry["execution_digest"] = execution
        P._record(state, name, cert_tier, entry)
        staged.append((name, identity, execution))
    if not staged:
        return []
    P._save_tier_state(ws, state)
    recorded = []
    for name, identity, execution in staged:
        row = {"capsule": name, "pass": True, "barrier_tier": cert_tier, "barrier_status": "pass",
               **({"execution_digest": execution} if execution is not None else {})}
        resolved = P.record_cert.__wrapped__(ws, {"per_capsule": [row]}, cert_tier, log, identity=identity,
                                             source_identity_verified=True)
        if not resolved:
            # Never leave a record outstanding for a job nobody will run.
            state = P._tier_state(ws)
            slots = P._slots(state, name, cert_tier)
            if (slots.get(identity) or {}).get("status") == "pending":
                del slots[identity]
                if P.record_identity(state[name].get(cert_tier)) == identity:
                    state[name][cert_tier] = next(reversed(slots.values()), None) if slots else None
                    if state[name][cert_tier] is None:
                        del state[name][cert_tier]
                P._save_tier_state(ws, state)
        recorded += resolved
    if log is not None and recorded:
        print(f"[promote] self-check {cert_tier} certificates kept for source {digest}: {recorded}",
              file=log, flush=True)
    return recorded
