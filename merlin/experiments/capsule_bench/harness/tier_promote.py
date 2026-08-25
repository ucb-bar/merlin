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
        return set(cert_capsule_cover(te.graded_roots(), tile_dim=_td)["capsules"])
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


def promote(ws, ch, verdict, loop_tier, cert_tier, cover, log):
    """Record what the loop tier just learned, and enqueue cert jobs for what it unlocked.

    Returns the capsule names promoted. Enqueues by writing a `simreq_` the broker's own queue picks up --
    the same path an agent request takes, so it inherits the constrained-runner validation rather than
    routing around it.
    """
    import json as _j
    from merlin.targetgen.oracle_schedule import CapsuleState, Verdict, schedule

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
