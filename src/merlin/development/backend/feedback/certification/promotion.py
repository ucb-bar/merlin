"""Record loop-tier verdicts and enqueue certificate work for BOTH grading brokers.

Every producer of feedback considers promotion. Wiring only the asynchronous broker
silently missed synchronous verdicts; the shared implementation prevents that drift.
Scheduling policy stays in ``merlin.development.targetgen.oracles.schedule``. This module owns
verdict recording, content identities and the pending/certified ledger.

Identity is narrowest-first:

* Execution identity binds each capsule's executable bytes and recorded target/hardware
  revisions. Byte-identical programs can retain certificates across compiler edits;
  Merlin's own source revision is not a program identity.
* Component identity asks which submission files the declared commands can read. Its
  vocabulary comes from the manifest, not this module. Use validated ``components``
  first, otherwise derive command entrypoints, transitive local imports and declared
  literal paths. Every capsule invokes every command, so this can spare inert notes
  and documentation, not invent a per-capsule source dependency declaration.
* Whole-submission identity is the conservative fallback when neither is established.

Missing narrower evidence is explicit in ``fallback_reason`` and the log: correct
conservatism must not hide broken identity extraction. Source decomposition alone
cannot retain one capsule's certificate when another emitted program changes;
execution identity supplies that distinction.

A certificate always belongs to the bytes that earned it. ``record_cert`` resolves
the enqueue-time pending identity without rehashing the edited live submission.
Unattributable results remain unrecorded. Private snapshot creation and recovery are
shared in ``snapshots``; narrowing never credits a verdict to different bytes.
"""

from __future__ import annotations

import fcntl
import os
import time
from functools import wraps
from pathlib import Path

from merlin.development.backend.context import InvocationContext
from merlin.development.backend.feedback.certification import (
    promotion_admission as pending_retirement,
)
from merlin.development.backend.feedback import verdicts as recorded_verdicts
from merlin.development.backend.feedback.certification.promotion_admission import affordable_cert_items, cert_sim
from merlin.development.backend.feedback.certification.promotion_ledger import (
    _cert_cover,
    _log_invalidations,
    _mandatory_cert_members,
    _no_narrower_cause,
    _record,
    _recorded_tiers,
    _save_tier_state,
    _slots,
    _slots_ro,
    _submission_digest,
    _submission_digests,
    _submission_file_bytes,
    _tier_state,
    _valid_execution,
    _verdict_for,
    current_identity,
    record_identity,
    submission_digests,
    submission_tree_digest,
)
from merlin.development.backend.feedback.certification.promotion_sources import (
    _DYNAMIC_IMPORT,
    _ast,
    _closure,
    _command_roots,
    _files_at,
    _manifest_strings,
    _py_imports,
    _py_module_file,
    _scan,
    _src_suffixes,
    _under,
)
from merlin.development.backend.feedback.verdicts import cert_already_executed, row_status
from merlin.development.backend.feedback.execution import snapshots
from merlin.development.targetgen.oracles.schedule import CERT_LEDGER as _LEDGER
from merlin.development.targetgen.oracles.schedule import FAIL as _FAIL
from merlin.development.targetgen.oracles.schedule import PASS as _PASS
from merlin.evaluation.capsule.evidence import certificates as capsule_adapters
from merlin.evaluation.capsule.evidence import certificates as capsule_availability

# ---- promotion ---------------------------------------------------------------------------------

_NEUTRAL_SIM = "contract"  # "grade on whatever tier this target's contract resolves to"


def resolve_tiers(ws, *, context: InvocationContext, capsules_root: Path | None = None):
    """`(loop_tier, cert_tier, cover)` for this target, DERIVED from its own adapter map.

    loop = the fastest tier the corpus declares; cert = the deepest reachable above it. A target that
    exposes one tier gets `(None, None, None)` -- promotion disabled rather than a second tier invented.
    """
    from merlin.development.backend.feedback.campaign.correctness import roster

    if roster():
        return "L2", "L3", roster()
    try:
        from merlin.development.targetgen.corpus.materialization import declared_oracle_tiers
        from merlin.target.registry.descriptor_loading import load_target_experiment

        te = load_target_experiment(context.descriptor)
        decl = declared_oracle_tiers(*((capsules_root,) if capsules_root is not None else te.graded_roots()))
        loop = capsule_availability.qa_loop_adapters(te.target, te.sim_via, declared_tiers=decl)
        full = capsule_adapters.oracle_adapters(te.target, te.sim_via)
        deeper = sorted(set(full) - set(loop))
        if not (loop and deeper):
            return None, None, None
        return sorted(loop)[0], deeper[-1], _cert_cover(ws, context=context, capsules_root=capsules_root)
    except Exception:  # noqa: BLE001 -- unresolvable ladder: no promotion, and the caller says so
        return None, None, None


_LEDGER_KEEP = 16


def _ledger_transaction(function):
    """Serialize complete ledger read/modify/write operations across grading brokers."""

    @wraps(function)
    def locked(ws, *args, **kwargs):
        root = Path(ws, "qa")
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".tier_state.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return function(ws, *args, **kwargs)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    return locked


@_ledger_transaction
@pending_retirement.completed_request
def record_cert(ws, verdict, cert_tier, log=None, identity=None, source_identity_verified: bool = False) -> list[str]:
    """Write a COMPLETED promotion's result into the tier state, against the bytes that earned it.

    `promote()` marks a capsule `pending` when it enqueues the cert job, and the broker's reap skips
    `_promote` for a job that WAS a promotion -- correctly, so a cert verdict cannot re-enqueue itself.
    But nothing then wrote the outcome back, so a capsule stayed `pending` forever: the certificate was
    earned on real RTL and discarded, and the next loop verdict re-certified the same bytes.

    Measured on merlincirct_arm4_func_20260901_v4 and _p2: promotions fired and COMPLETED (21 and 3
    `simdone_promo*` respectively, one verified `barrier_tier=L3 barrier_status=pass`), while both tier
    states showed only `L3: pending` and never once `L3: pass`.

    The digest is NOT recomputed here. A cert belongs to the exact bytes that were pending when the job
    was enqueued; re-hashing now would attribute it to whatever the agent has edited since. So this only
    resolves an existing pending entry, and leaves anything else alone -- a result with no pending entry
    is a result we cannot attribute, and that is recorded by doing nothing rather than by guessing.

    That property is enforced against the per-identity LEDGER rather than against a single slot. The
    single slot was overwritten by the next re-enqueue, so a completed job routinely arrived to find a
    record for OTHER bytes: with an artifact identity that was refused (the cert was dropped), and
    WITHOUT one it was silently accepted and the certificate was re-attributed to bytes that never earned
    it. Looking the completed job's own identity up in the ledger resolves exactly the record it belongs
    to, and refuses when no such record exists.

    ``identity`` is the ledger key the ENQUEUER wrote onto the request, forwarded by the reap. It is what
    lets a result whose reader produced no artifact identity still be attributed exactly, instead of
    falling back to "the one outstanding record" (which cannot be used at all once two are outstanding).
    A result that DOES carry an artifact identity normally decides for itself: the job may have launched
    after an edit, and then only the identity it actually ran may be credited. The sole exception is a
    broad ``submission:...`` pending record whose broker copied and verified the enqueue-time source
    before launch (``source_identity_verified``). That result may resolve the broad record because the
    verified snapshot proves which source bytes produced the reported executable; an exact-identity
    mismatch is never relaxed.
    """
    rows = (verdict or {}).get("per_capsule") or []
    if not rows:
        return []
    st = _tier_state(ws)
    resolved = []
    for row in rows:
        name = str((row or {}).get("capsule") or "")
        if not name:
            continue
        slots = _slots_ro(st, name, cert_tier)
        completed = row.get("execution_digest")
        if _valid_execution(completed):
            # The identity the job ACTUALLY ran. It selects its own record; a job whose bytes hold no
            # record is unattributable, however many other records this capsule has.
            key = str(completed)
            entry = slots.get(key)
            if not isinstance(entry, dict) or entry.get("status") != "pending":
                requested = slots.get(str(identity)) if identity is not None else None
                from merlin.development.backend.feedback.certification.tier_evidence import (
                    resolve as resolve_boundary,
                )

                joined = (
                    resolve_boundary(st, name, str(identity), row, source_identity_verified=source_identity_verified)
                    if cert_tier == "L3"
                    else None
                )
                if joined is not None:
                    entry = joined
                    mirror = (st.get(name) or {}).get(cert_tier)
                    if isinstance(mirror, dict) and record_identity(mirror) == str(identity):
                        st[name][cert_tier] = dict(joined)
                elif (
                    source_identity_verified
                    and str(identity or "").startswith("submission:")
                    and isinstance(requested, dict)
                    and requested.get("status") == "pending"
                ):
                    # The broker made an isolated copy and verified its source digest against this broad
                    # request identity before executing it. Preserve the actual program identity as
                    # evidence without changing ``record_identity(entry)``: this ledger slot remains an
                    # assertion about the verified source bytes, while the exact program keeps its own
                    # independently keyed certificate.
                    key, entry = str(identity), requested
                    entry = dict(entry)
                    entry["certified_execution_digest"] = str(completed)
                else:
                    if log is not None:
                        print(
                            f"[promote] {name} {cert_tier} result not recorded: no outstanding record "
                            f"for the execution identity this job ran",
                            file=log,
                            flush=True,
                        )
                    continue
        elif identity is not None:
            # The enqueuer stated which record this job was launched for. That is a fact about the
            # request, not a guess about the result, so it attributes exactly -- including when several
            # records are outstanding at once, which is the case the heuristic below cannot serve.
            key = str(identity)
            entry = slots.get(key)
            if not isinstance(entry, dict) or entry.get("status") != "pending":
                if log is not None:
                    print(
                        f"[promote] {name} {cert_tier} result not recorded: the record this job was "
                        f"enqueued for is no longer outstanding",
                        file=log,
                        flush=True,
                    )
                continue
            if _valid_execution(entry.get("execution_digest")):
                # The record names an exact artifact and the result does not carry one -- so whether the
                # job ran THAT artifact cannot be established. The request only says what was asked for;
                # the agent may have edited between enqueue and launch. Fail closed: an unreadable
                # artifact identity is never evidence that the right artifact ran.
                if log is not None:
                    print(
                        f"[promote] {name} {cert_tier} result not recorded: the record names an "
                        f"execution identity and the result carries none",
                        file=log,
                        flush=True,
                    )
                continue
        else:
            # No artifact identity on the result and none on the request: the only attributable case is a
            # single outstanding record that ALSO carries no artifact identity. Two outstanding records
            # and a result that cannot say which it belongs to is precisely the misattribution this
            # guards -- refuse.
            outstanding = {i: e for i, e in slots.items() if isinstance(e, dict) and e.get("status") == "pending"}
            if len(outstanding) != 1:
                if log is not None and outstanding:
                    print(
                        f"[promote] {name} {cert_tier} result not recorded: {len(outstanding)} "
                        f"outstanding records and the result carries no execution identity, so which "
                        f"bytes earned it cannot be determined",
                        file=log,
                        flush=True,
                    )
                continue
            key, entry = next(iter(outstanding.items()))
            if _valid_execution(entry.get("execution_digest")):
                if log is not None:
                    print(
                        f"[promote] {name} {cert_tier} result not recorded: the outstanding record "
                        f"names an execution identity this result does not carry",
                        file=log,
                        flush=True,
                    )
                continue
        status = row_status(row)
        if status is None:
            # The job ran and came back saying nothing this recorder can read. Resolving the pending
            # record to `fail` would spend a real RTL certificate on a fabricated verdict; leaving it
            # pending re-runs it, which is the fail-closed direction.
            if log is not None:
                print(
                    f"[promote] {name} {cert_tier} result not recorded: the result states no verdict "
                    f"this recorder can read (neither a boolean 'pass' nor a 'status' of pass/fail)",
                    file=log,
                    flush=True,
                )
            continue
        entry = dict(entry)
        entry["status"] = status
        slots_w = _slots(st, name, cert_tier)
        slots_w.pop(key, None)  # re-insert: a just-resolved record is the freshest, not the oldest
        slots_w[key] = entry
        # The mirror moves only when it is the SAME record. Moving it to a different identity is exactly
        # the re-attribution this function exists to prevent.
        mirror = (st.get(name) or {}).get(cert_tier)
        if isinstance(mirror, dict) and record_identity(mirror) == key:
            st[name][cert_tier] = entry
        resolved.append(f"{name}={status}")
    if resolved:
        _save_tier_state(ws, st)
        from merlin.development.backend.feedback.campaign.correctness import project

        project(ws, st)
        if log is not None:
            print(f"[promote] {cert_tier} recorded: {resolved}", file=log, flush=True)
    return resolved


def cert_instrument(cert_tier: str, *, context: InvocationContext) -> str | None:
    """Identity of the JUDGE this promotion's certificate will be earned under, or ``None``.

    A certificate describes a verdict produced by a particular grading path on a particular engine, and
    a run whose grading path has moved is not the run that earned it. Recorded ON the ledger entry so a
    later grade can decide whether the certificate still describes ITS instrument -- without it, the
    entry states which bytes were certified and says nothing about who certified them, and a reader that
    trusted it would carry a verdict produced by a different judge.

    ``None`` when the instrument cannot be established, and then nothing is recorded -- an entry with no
    instrument can never be carried (``tier_cache._from_ledger``), which is the fail-closed direction.
    """
    try:
        from merlin.development.targetgen.oracles.tier_cache import instrument_digest
        from merlin.evaluation.capsule.evidence.tiers import rtl_tiers_of
        from merlin.target.registry.descriptor_loading import load_target_experiment

        te = load_target_experiment(context.descriptor)
        return instrument_digest(te.target, cert_tier, rtl_tier=cert_tier in rtl_tiers_of(te.target))
    except Exception:  # noqa: BLE001 -- unresolvable instrument: record none, so nothing is carried
        return None


@_ledger_transaction
def promote(
    ws,
    ch,
    verdict,
    loop_tier,
    cert_tier,
    cover,
    log,
    *,
    source_ws=None,
    context: InvocationContext,
    capsules_root: Path | None = None,
    closed_loop_rows: dict | None = None,
):
    """Record what the loop tier just learned, and enqueue cert jobs for what it unlocked.

    Returns the capsule names promoted. Enqueues by writing a `simreq_` the broker's own queue picks up --
    the same path an agent request takes, so it inherits the constrained-runner validation rather than
    routing around it.
    """
    from merlin.development.backend.feedback.certification.promotion_identity_promotion import decomposition

    verdict = recorded_verdicts.hydrated(ws, verdict)  # compact rows hold their boundary in the case store
    from merlin.development.backend.campaign import control as joint_control
    from merlin.development.backend.feedback.campaign import correctness as physical

    joint_required = physical.roster() if joint_control.selected() is not None else None
    if joint_required is not None:
        if not joint_required or (loop_tier, cert_tier) != ("L2", "L3") or not closed_loop_rows:
            return []
        source_ws = Path(source_ws) if source_ws is not None else Path(ws)
        ready = physical.eligible(closed_loop_rows, source_root=source_ws / "submission")
        verdict = dict(
            verdict, per_capsule=[row for row in verdict.get("per_capsule", ()) if row.get("capsule") in ready]
        )
        cover = joint_required
    import json as _j

    from merlin.development.targetgen.oracles.schedule import (
        CapsuleState,
        Verdict,
        schedule,
        valid_execution_digest,
    )

    # A driver grade runs a frozen ``cand_*/submission`` rather than the moving workspace. Attribute
    # its verdict to that source tree; the broker will run only if it can reproduce this digest in an
    # isolated copy. Broker-produced verdicts omit ``source_ws`` because they already grade ``ws``.
    source_ws = Path(source_ws) if source_ws is not None else Path(ws)
    digest, comps, rejected = submission_digests(source_ws)
    d = decomposition(source_ws)
    st = _tier_state(ws)
    _instrument = cert_instrument(cert_tier, context=context)
    _loop_instrument = cert_instrument(loop_tier, context=context)
    # Same program bytes under a different judge are historical evidence, not
    # a reason to suppress the newly required physical execution. Preserve the
    # complete old ledger before retiring terminal records; pending jobs keep
    # their original identity until their own completion is resolved.
    from merlin.development.backend.feedback.certification.tier_evidence import retire as retire_instrument

    retirement = retire_instrument(ws, st, tier=cert_tier, instrument=_instrument)
    if retirement is not None:
        print(
            f"[promote] retained {retirement['retired_records']} historical {cert_tier} "
            f"record(s) for another instrument: {retirement['path']}",
            file=log,
            flush=True,
        )
    if rejected:
        # Loud, not silent: a rejected name owns no bytes, so the component it was meant to be never
        # moves and every certificate keeps riding on the files it failed to claim.
        print(
            f"[promote] manifest components outside the declared command vocabulary, ignored: {sorted(rejected)}",
            file=log,
            flush=True,
        )
    print(
        f"[promote] components: {d['source'] or 'none (whole-submission digest)'}"
        + (
            f" -- {len(comps)} component(s), {len(d['inert'])} file(s) outside every command's reach"
            if d["source"]
            else ""
        ),
        file=log,
        flush=True,
    )

    # What EVERY capsule rides on: all of it. "Everything" is expressed as the set of components rather
    # than as the whole digest, and the two differ by exactly the bytes the decomposition PROVED no
    # command can read. With no decomposition the fallback is `None`, which is the whole digest, i.e. the
    # behaviour before the decomposition existed. No capsule may narrow this -- see WHY NO CAPSULE
    # DECLARES ITS OWN DEPENDENCY SET above for why a per-capsule claim cannot be truthful.
    #
    # The per-capsule saving comes from the EXECUTION identity instead, one layer down: the source-level
    # decomposition can only spare bytes no command reads (the same set for every capsule), while the
    # executable identity distinguishes capsule from capsule, because each capsule emits its own program.
    # That is the layer at which "capsule 1 moved, capsule 2 did not" is a truthful statement.
    deps = tuple(sorted(comps)) or None

    # Record what the loop tier just learned, keyed by the bytes that earned it -- the exact per-capsule
    # executable identity when the reader produced one, and BOTH the whole digest and the per-component
    # decomposition regardless, because a verdict that carries only the whole digest cannot be
    # re-examined per component later (it comes back UNDETERMINABLE, which re-runs).
    execution_by_name = {}
    boundary_by_name = {}
    refusal_names = set()
    n_pass = n_fail = 0
    unreadable = []
    for row in verdict.get("per_capsule") or []:
        name = row.get("capsule")
        if not name:
            continue
        name = str(name)
        refusal = row.get("expected_refusal")
        if isinstance(refusal, dict) and refusal.get("status") == "met":
            # A correct stated refusal satisfies its obligation but executes no loop tier.
            # It cannot authorize a native certificate, including from an older ledger slot.
            refusal_names.add(name)
            continue
        execution_digest = row.get("execution_digest")
        if not valid_execution_digest(execution_digest):
            execution_digest = None
        execution_by_name[name] = execution_digest
        boundary_by_name[name] = row.get("tier_boundary")
        # WHAT the row said, over BOTH producers' spellings, or nothing. A row that states no verdict is
        # not a failed row: recording it as one is what silently retired 96 capsules a round.
        status = row_status(row)
        if status is None:
            unreadable.append(name)
            continue
        n_pass += status == _PASS
        n_fail += status == _FAIL
        entry = {"status": status, "digest": digest, "components": dict(comps)}
        if _loop_instrument is not None:
            entry["instrument"] = _loop_instrument
        if execution_digest is not None:
            entry["execution_digest"] = execution_digest
        if boundary_by_name[name] is not None:
            entry["tier_boundary"] = boundary_by_name[name]
        # RECORD WHY, on disk, when this verdict can only be compared against bytes every capsule shares.
        # A conservative fallback that says nothing is why the defect read as correct behaviour for a
        # whole round.
        _why_broad = _no_narrower_cause(execution_digest, comps)
        if _why_broad:
            entry["fallback_reason"] = _why_broad
        _record(st, name, loop_tier, entry)

    # WHAT to run next is `oracle_schedule`'s decision, not this file's. The rules (a cert tier is gated
    # on the loop tier passing; the cert tier runs a representative cover; a verdict already earned by
    # these bytes is never re-run) were implemented here once and in the scheduler once, which is one
    # implementation too many -- two expressions of the same policy drift, and the one that drifts is
    # whichever has no tests. The scheduler has them; this is now only plumbing.
    # Each capsule is judged against the record earned by EXACTLY its current bytes when the ledger holds
    # one, so an edit elsewhere does not throw its certificate away; anything else keeps today's
    # comparison and is still put through `invalidated_by`.
    identity_by_name = {n: current_identity(execution_by_name.get(n), digest) for n in st}
    states = []
    for n, e in st.items():
        if n in refusal_names:
            continue
        _id = identity_by_name[n]
        verdicts = {}
        for t in _recorded_tiers(e):
            v = _verdict_for(st, n, t, _id)
            if isinstance(v, dict):
                verdicts[t] = Verdict(
                    v.get("status"), v.get("digest"), dict(v.get("components") or {}), v.get("execution_digest")
                )
        states.append(
            CapsuleState(
                name=n,
                digest=digest,
                verdicts=verdicts,
                components=dict(comps),
                depends_on=deps,
                execution_digest=execution_by_name.get(n),
            )
        )
    # A representative subset is an optional-cert cost policy, not authority to omit
    # a tier the admitted manifest requires. Keep the same scheduler, PASS gate and
    # identity ledger; only its eligibility set includes these mandatory members.
    if cover is not None and joint_required is None:
        cover = set(cover) | _mandatory_cert_members(context=context, capsules_root=capsules_root, cert_tier=cert_tier)
    want = [
        w
        for w in schedule(states, tier_order=[loop_tier, cert_tier], cert_tiers=(cert_tier,), cert_cover=cover)
        if w.tier == cert_tier
    ]
    # A DECLARED per-capsule budget may decline a capsule here. Inert unless one is declared.
    if joint_required is None:
        want, _budget_declined = affordable_cert_items(
            want, cert_tier=cert_tier, log=log, context=context, capsules_root=capsules_root
        )
    else:
        # Own correctness budget, never a calibration price; only a closed receipt unlocks a case, and
        # only under continuous promotion (off by default: L3 is then a best-effort final step).
        want = [w for w in want if w.capsule in ready and joint_control.continuous_l3_promotion()]

    # WHAT THIS VERDICT ACTUALLY SAID, every time, in one line. An idle promotion and a broken one are
    # indistinguishable by design -- the call is wrapped in a try/except so it can never gate a run -- so
    # the only thing that can tell them apart is an accounting line that is printed unconditionally. The
    # defect this closes ran for 8 hours behind exactly two log lines, neither of which was a count.
    # `unreadable` is named, not summarised away: it is the shape a THIRD verdict producer would take.
    print(
        f"[promote] {loop_tier}: {n_pass} pass, {n_fail} fail"
        + (
            f", {len(unreadable)} stating no verdict ({', '.join(sorted(unreadable)[:3])}"
            + (", ..." if len(unreadable) > 3 else "")
            + ")"
            if unreadable
            else ""
        )
        + f"; {cert_tier} cover={len(cover) if cover is not None else 'all'}"
        + f"; {len(want)} capsule(s) want {cert_tier}",
        file=log,
        flush=True,
    )

    # WHICH component requeued each capsule, so a reader of the log can see why a certificate was dropped
    # rather than only that the count went up. A run that requeues everything and one that requeues one
    # capsule are indistinguishable from the promotion count alone. A whole-submission cause additionally
    # NAMES the inputs that were missing -- the diagnosis of this defect stalled on 21 identical
    # `<whole-submission> (changed)` lines that named none of them.
    _log_invalidations(
        states,
        execution_by_name,
        comps,
        loop_tier,
        cert_tier,
        log,
        observed_names=set(execution_by_name) if joint_required is not None else None,
    )

    promoted = []
    _sim = cert_sim(cert_tier, context=context)
    from merlin.development.backend.feedback.certification import promotion_admission as admission

    try:
        current_source = admission.observe_source(Path(ws)) if want else None
    except (OSError, ValueError):
        current_source = None
    if joint_required is not None and _instrument is None:
        print("[promote] selected certificate instrument unavailable; physical work deferred", file=log, flush=True)
        want = []
    if want and _sim is None:
        # Say it once, loudly. Marking a capsule pending for a job that cannot be enqueued is how a
        # capsule strands at `pending` and never resolves.
        print(
            f"[promote] no --sim this broker accepts serves {cert_tier}; {len(want)} capsule(s) NOT "
            f"enqueued and NOT marked pending",
            file=log,
            flush=True,
        )
    for w in want:
        if _sim is None:
            continue
        # A frozen loop can keep completing after the author changes source.
        # Retain its loop evidence, but do not replenish obsolete native work.
        refusal = None
        if current_source is None:
            refusal = "current source identity unavailable"
        elif current_source.digest != digest:
            refusal = admission.reason(
                w.capsule,
                current=current_source,
                original=admission.Source(digest, comps),
                current_instrument=_instrument,
                original_instrument=_instrument,
                original_execution=execution_by_name.get(w.capsule),
                observed_execution=admission.current_execution(
                    st, w.capsule, loop_tier, current_source.digest, instrument=_loop_instrument
                ),
            )
        if refusal and joint_required is None:
            print(f"[promote] {w.capsule} {cert_tier} auto promotion deferred: {refusal}", file=log, flush=True)
            continue
        execution_digest = execution_by_name.get(w.capsule)
        token = execution_digest[:16] if execution_digest is not None else digest
        if _instrument is not None:
            import hashlib

            token = hashlib.sha256(f"{token}:{_instrument}".encode()).hexdigest()[:16]
        key = identity_by_name.get(w.capsule) or current_identity(execution_digest, digest)
        attempt = pending_retirement.requeue_attempt(_slots_ro(st, w.capsule, cert_tier), key)
        if attempt is None:
            continue
        jid = f"promo{len(promoted)}_{token}_r{attempt}_{w.capsule}"[:80]
        req = ch / f"simreq_{jid}.json"
        if req.exists():
            # The ledger says these bytes hold no record, yet a request for them is on the queue. That is
            # a state the ledger cannot explain (a hand-cleared state file, a request written by a broker
            # whose state write was lost). Say so instead of skipping silently: the silent version is how
            # a capsule stranded with a request nobody would ever answer.
            print(
                f"[promote] {w.capsule} {cert_tier}: {req.name} is already queued while the tier "
                f"state holds no record for these bytes; NOT re-enqueued and NOT marked outstanding",
                file=log,
                flush=True,
            )
            continue
        # Freeze before enqueue: L3 may wait behind minutes of work while the agent keeps editing.
        snapshot, snapshot_error = snapshots.create_promotion_snapshot(
            ws, source_ws, digest, submission_digest=_submission_digest
        )
        if snapshot_error:
            print(f"[promote] {w.capsule} {cert_tier}: NOT enqueued -- {snapshot_error}", file=log, flush=True)
            continue
        try:
            req.write_text(
                _j.dumps(
                    # `identity` travels WITH the request so the reap can hand a completed result back to the
                    # exact record it was launched for. Without it, a result whose reader produced no artifact
                    # identity is unattributable as soon as a second record is outstanding -- and the certificate
                    # the RTL just paid for is dropped.
                    {
                        "sim": _sim,
                        "capsules": w.capsule,
                        "workers": 1,
                        "tiers": cert_tier,
                        "promoted": True,
                        **({"joint_physical_correctness": True} if joint_required is not None else {}),
                        "identity": key,
                        "submission_digest": digest,
                        "submission_snapshot": snapshot,
                        "submitted_at": time.time(),
                    }
                )
            )
        except Exception:
            # No queue record owns this snapshot, so nothing else can clean it up.
            import shutil as _shutil

            _snapshot_path = snapshots.promotion_snapshot_path(ws, snapshot)
            if _snapshot_path is not None:
                _shutil.rmtree(_snapshot_path, ignore_errors=True)
            raise
        # Mark pending only once the request is actually on the queue.
        pending = {"status": "pending", "digest": digest, "components": dict(comps), "request": jid, "attempt": attempt}
        if boundary_by_name.get(w.capsule) is not None:
            pending["tier_boundary"] = boundary_by_name[w.capsule]
        if execution_digest is not None:
            pending["execution_digest"] = execution_digest
        # WHICH JUDGE this certificate will have been earned under. `record_cert` copies this record
        # forward when it resolves, so the resolved certificate carries it too, and a later grade can
        # decide whether the verdict still describes its own instrument instead of assuming it does.
        if _instrument is not None:
            pending["instrument"] = _instrument
        if _loop_instrument is not None:
            pending["loop_instrument"] = _loop_instrument
        _why_broad = _no_narrower_cause(execution_digest, comps)
        if _why_broad:
            pending["fallback_reason"] = _why_broad
        _record(st, w.capsule, cert_tier, pending)
        promoted.append(w.capsule)
    _save_tier_state(ws, st)
    if joint_required is not None:
        physical.project(ws, st, source=digest, instrument=_instrument, closed_rows=closed_loop_rows)
    if promoted:
        print(f"[promote] {loop_tier} pass -> {cert_tier}: {promoted}", file=log, flush=True)
    return promoted
