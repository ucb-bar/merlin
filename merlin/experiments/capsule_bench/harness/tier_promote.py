"""Loop-tier -> cert-tier promotion, shared by BOTH grading brokers.

This lives in its own module because hooking only one broker is exactly the bug it was written to avoid.
Promotion was first wired into the async oracle alone, and a live run then showed the agent using the
SYNCHRONOUS self-check seven times to the async path's two -- so eight verdicts completed and promotion
fired zero times. The rule is: wherever a verdict is produced, promotion is considered. One module, two
call sites, no third copy.

The POLICY (which capsule earns the cert tier, and when a verdict may be reused) is not here either --
it is `merlin.targetgen.oracle_schedule`, which has the tests. This module is the plumbing that records
what a verdict said and enqueues what the policy asks for.
"""
from __future__ import annotations

import time
from pathlib import Path

_NEUTRAL_SIM = "contract"   # "grade on whatever tier this target's contract resolves to"


def resolve_tiers(ws):
    """`(loop_tier, cert_tier, cover)` for this target, DERIVED from its own adapter map.

    loop = the fastest tier the corpus declares; cert = the deepest reachable above it. A target that
    exposes one tier gets `(None, None, None)` -- promotion disabled rather than a second tier invented.
    """
    try:
        from merlin.targetgen import capsule_runner as _CR
        from merlin.targetgen.contract.materialize import declared_oracle_tiers
        from merlin.targetgen.target_experiment import load_target_experiment
        import _common as _C
        te = load_target_experiment(_C.EXP / "target_experiment.yaml")
        decl = declared_oracle_tiers(*te.graded_roots())
        loop = _CR.qa_loop_adapters(te.target, te.sim_via, declared_tiers=decl)
        full = _CR.oracle_adapters(te.target, te.sim_via)
        deeper = sorted(set(full) - set(loop))
        if not (loop and deeper):
            return None, None, None
        return sorted(loop)[0], deeper[-1], _cert_cover(ws)
    except Exception:  # noqa: BLE001 -- unresolvable ladder: no promotion, and the caller says so
        return None, None, None


def _submission_digest(ws) -> str:
    """A content address for the submission the verdict was earned against.

    Verdicts are keyed by BYTES, not by round: unchanged bytes never need re-grading, and changed bytes
    invalidate exactly the capsules they touch. Without this the loop re-certifies thirty-odd capsules to
    learn about the one that moved.
    """
    import hashlib
    h = hashlib.sha256()
    for f in sorted(Path(ws, "submission").rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            h.update(f.relative_to(ws).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _cert_cover(ws) -> set | None:
    """Which capsules are worth certifying at all. The hardware cannot tell two capsules in the same
    (family, dtype) cell apart, so certifying both spends minutes to learn nothing. `None` on any failure
    -> certify anything eligible, because a cover that silently comes back empty is indistinguishable
    from everything already being done."""
    try:
        from merlin.targetgen.contract.materialize import cert_capsule_cover
        from merlin.targetgen.target_experiment import load_target_experiment
        import _common as _C
        te = load_target_experiment(_C.EXP / "target_experiment.yaml")
        # Pass the tile edge so the cover certifies PARTIAL tiles as their own cell. A cover built on
        # family and dtype alone can pick, per cell, the capsule whose extents happen to divide evenly and
        # then certify no ragged extent anywhere -- and a partial tile is exactly what a functional model
        # is least able to stand in for (a taped-out unit here got `n % 64 != 0` wrong while every
        # functional check passed).
        _td = None
        try:
            from merlin.targetgen.corpus_spec import _tile_dim
            from merlin.targetgen.target_experiment import load_capability_manifest
            _td = int(_tile_dim(te.target, load_capability_manifest(te.target).contract)) or None
        except Exception:  # noqa: BLE001 -- no derivable tile edge: cover without the alignment axis
            _td = None
        # Exclusions travel WITH the roots. A capsule the descriptor withholds from the paid loop cannot
        # stand for its cell, because promotion only enqueues capsules in the cover — picking one that
        # never runs retires the cell for a certificate nobody will produce.
        return set(cert_capsule_cover(te.graded_roots(), tile_dim=_td,
                                      exclude=set(getattr(te, "graded_exclude", ()) or ()))["capsules"])
    except Exception:  # noqa: BLE001 -- no resolvable corpus: stay permissive, never silently empty
        return None


def _tier_state(ws) -> dict:
    import json as _j
    f = Path(ws, "qa", "tier_state.json")
    try:
        return _j.loads(f.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_tier_state(ws, st) -> None:
    import json as _j
    f = Path(ws, "qa", "tier_state.json")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(_j.dumps(st, indent=2))


#: A cert job that never reached the simulator is NOT a verdict about the hardware. It is recorded under
#: its own status so it neither certifies nor fabricates a failure -- and, because the scheduler treats
#: anything that is not ``unknown`` as settled for those bytes, it also does not spin re-running a job
#: that crashes deterministically. It is surfaced in the log instead, which is how it gets fixed.
CERT_ERROR = "error"


def _cert_recorded(ws) -> set:
    """The cert response ids already folded into the tier state."""
    import json as _j
    f = Path(ws, "qa", "cert_recorded.json")
    try:
        return set(_j.loads(f.read_text()))
    except Exception:  # noqa: BLE001 -- absent or unreadable: nothing recorded yet
        return set()


def _save_cert_recorded(ws, seen: set) -> None:
    import json as _j
    f = Path(ws, "qa", "cert_recorded.json")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(_j.dumps(sorted(seen)))


def _cert_row_status(row: dict) -> tuple[str, str]:
    """``(status, note)`` for one graded capsule row of a cert response.

    A row that carries no tier verdict did not certify anything -- the package crashed, or the runner
    timed out -- and calling that ``fail`` would attribute a hardware verdict to a tooling failure. The
    two are told apart by whether the run produced a tier result at all, not by guessing from the plane.
    """
    if row.get("pass"):
        return "pass", ""
    fail = row.get("failure") or {}
    plane, cat = fail.get("plane") or "", fail.get("category") or ""
    if row.get("tiers"):
        return "fail", ""
    return CERT_ERROR, f"{plane}/{cat}".strip("/") or "no tier result"


def drain_cert_responses(ws, ch, cert_tier, log) -> dict:
    """Fold every completed cert response into the tier state. Returns ``{status: count}`` for this call.

    The cert tier was written as ``pending`` when the job was enqueued and NOTHING ever wrote it back, so
    a capsule stayed ``pending`` for the life of the run however its certification actually went. Across
    nineteen radiance runs that produced 3523 completed certs and not one recorded verdict -- and because
    the state never moved, the fact that every one of them crashed at the parse plane was invisible too.
    A verdict nobody records is indistinguishable from a verdict nobody produced.

    The digest comes from the RESPONSE ID, not from the tree as it stands now: the id is
    ``promo<n>_<digest>_<capsule>``, and that digest is the submission the cert actually graded. Keying it
    to the current tree would attribute an old verdict to new bytes, which is the one thing the
    content-addressing exists to prevent.
    """
    import json as _j
    seen = _cert_recorded(ws)
    st = _tier_state(ws)
    counts: dict[str, int] = {}
    fresh = []
    for f in sorted(Path(ch).glob("simresp_promo*.json")):
        jid = f.name[len("simresp_"):-len(".json")]
        if jid in seen:
            continue
        try:
            doc = _j.loads(f.read_text())
        except Exception:  # noqa: BLE001 -- a half-written response is read again next call
            continue
        # Prefer the digest the grade REPORTS over the one the job was named for. They differ whenever
        # the agent edited between enqueue and execution, and the verdict belongs to the bytes that were
        # actually graded -- attributing it to the requested bytes would mark a submission certified on
        # evidence from a different one. Older responses predate the reported field and fall back to the
        # job name, which is the best available answer for them.
        parts = jid.split("_", 2)
        digest = (doc.get("submission_digest") or (parts[1] if len(parts) >= 3 else "")) or ""
        for row in (doc.get("per_capsule") or []):
            name = row.get("capsule")
            if not name or not digest:
                continue
            status, note = _cert_row_status(row)
            st.setdefault(name, {})[cert_tier] = {"status": status, "digest": digest}
            counts[status] = counts.get(status, 0) + 1
            if status == CERT_ERROR:
                fresh.append(f"{name}:{note}")
        seen.add(jid)
    if counts:
        _save_tier_state(ws, st)
        _save_cert_recorded(ws, seen)
        print(f"[promote] recorded {cert_tier} verdicts: {counts}", file=log, flush=True)
        if fresh:
            # Loud on purpose: a cert that cannot run is a TOOLING failure and reads exactly like an
            # agent failure in every downstream score.
            print(f"[promote] {cert_tier} could not certify (tooling, not the submission): "
                  f"{sorted(set(fresh))[:6]}", file=log, flush=True)
    return counts


def promote(ws, ch, verdict, loop_tier, cert_tier, cover, log):
    """Record what the loop tier just learned, and enqueue cert jobs for what it unlocked.

    Returns the capsule names promoted. Enqueues by writing a `simreq_` the broker's own queue picks up --
    the same path an agent request takes, so it inherits the constrained-runner validation rather than
    routing around it.
    """
    import json as _j
    from merlin.targetgen.oracle_schedule import CapsuleState, Verdict, schedule

    # Fold in everything the cert tier finished since last time FIRST, so the scheduler sees those
    # verdicts on this pass rather than re-enqueueing work that is already done.
    drain_cert_responses(ws, ch, cert_tier, log)

    digest = _submission_digest(ws)
    st = _tier_state(ws)

    # Record what the loop tier just learned, keyed by the bytes that earned it.
    for row in (verdict.get("per_capsule") or []):
        name = row.get("capsule")
        if not name:
            continue
        st.setdefault(name, {})[loop_tier] = {
            "status": "pass" if row.get("pass") else "fail", "digest": digest}

    # WHAT to run next is `oracle_schedule`'s decision, not this file's. The rules (a cert tier is gated
    # on the loop tier passing; the cert tier runs a representative cover; a verdict already earned by
    # these bytes is never re-run) were implemented here once and in the scheduler once, which is one
    # implementation too many -- two expressions of the same policy drift, and the one that drifts is
    # whichever has no tests. The scheduler has them; this is now only plumbing.
    states = [CapsuleState(name=n, digest=digest,
                           verdicts={t: Verdict(v.get("status"), v.get("digest"))
                                     for t, v in (e or {}).items() if isinstance(v, dict)})
              for n, e in st.items()]
    want = [w for w in schedule(states, tier_order=[loop_tier, cert_tier], cert_tiers=(cert_tier,),
                                cert_cover=cover)
            if w.tier == cert_tier]

    promoted = []
    for w in want:
        st.setdefault(w.capsule, {})[cert_tier] = {"status": "pending", "digest": digest}
        jid = f"promo{len(promoted)}_{digest}_{w.capsule}"[:80]
        if not (ch / f"simreq_{jid}.json").exists():
            (ch / f"simreq_{jid}.json").write_text(_j.dumps(
                {"sim": _NEUTRAL_SIM, "capsules": w.capsule, "workers": 1, "tiers": cert_tier,
                 "promoted": True, "submitted_at": time.time()}))
            promoted.append(w.capsule)
    _save_tier_state(ws, st)
    if promoted:
        print(f"[promote] {loop_tier} pass -> {cert_tier}: {promoted}", file=log, flush=True)
    return promoted
