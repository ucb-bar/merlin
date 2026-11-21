"""Submission identities and certificate-ledger records for promotion."""

from __future__ import annotations

import os
from pathlib import Path

from merlin.development.backend.context import InvocationContext
from merlin.development.targetgen.oracles.schedule import CERT_LEDGER as _LEDGER


def _submission_digest(ws) -> str:
    """A content address for the submission the verdict was earned against.

    Verdicts are keyed by BYTES, not by round: unchanged bytes never need re-grading, and changed bytes
    invalidate exactly the capsules they touch. Without this the loop re-certifies thirty-odd capsules to
    learn about the one that moved.
    """
    return submission_digests(ws)[0]


def _submission_file_bytes(sub):
    """The ledger's canonical source stream, independent of tree location and of ``build/`` state
    (``run_inputs.outside_snapshot``), which no grade or frozen copy carries. Bytes are reused while a file
    keeps its stat identity (:mod:`merlin.common.digest`); each completion used to re-read every file."""
    from merlin.common import digest as stat_memo
    from merlin.development.backend.session.inputs.run_record import outside_snapshot

    sub = Path(sub)
    for path in sorted(sub.rglob("*")):
        if path.is_file() and not outside_snapshot(path.relative_to(sub).parts):
            body = stat_memo.cached(
                "file_bytes", str(path.absolute()), lambda path=path: stat_memo.file_signature(path), path.read_bytes
            )
            yield path, ("submission/" + path.relative_to(sub).as_posix()).encode(), body


def submission_tree_digest(sub) -> str:
    """Recompute the existing short ledger digest from an already reopened snapshot."""
    import hashlib

    digest = hashlib.sha256()
    for _, key, body in _submission_file_bytes(sub):
        digest.update(key)
        digest.update(body)
    return digest.hexdigest()[:16]


def submission_digests(ws) -> tuple:
    """``(whole_digest, {component: digest}, rejected_names)``, computed once per unchanged submission
    tree (see :mod:`.tree_value_cache`)."""
    from merlin.development.backend.feedback.execution import snapshots as tree_value_cache

    return tree_value_cache.cached(Path(ws, "submission"), lambda: _submission_digests(ws))


def _submission_digests(ws) -> tuple:
    """``(whole_digest, {component: digest}, rejected_names)`` for the submission on disk.

    The whole digest is byte-for-byte what it always was -- one pass, sorted paths, path bytes then file
    bytes -- so turning components on never invalidates a certificate by itself. Each file additionally
    feeds the hasher of EVERY component that owns it (a shared helper genuinely belongs to each command
    that imports it, and a change to it must move all of their digests). A file the decomposition cannot
    place feeds ``UNATTRIBUTED``, which every capsule depends on; a file it places OUTSIDE every command's
    reach feeds nothing, which is where the saving comes from. The component map is ``{}`` when there is
    no decomposition at all, and the scheduler reads that as "compare the whole submission"
    (see ``CapsuleState._dep_components``).
    """
    import hashlib

    from merlin.development.backend.feedback.certification.promotion_identity_promotion import decomposition

    d = decomposition(ws)
    owners, names = d["owners"], d["names"]
    h = hashlib.sha256()
    parts = {}
    sub = Path(ws, "submission")
    for f, key, body in _submission_file_bytes(sub):
        h.update(key)
        h.update(body)
        for owner in owners.get(f.relative_to(sub).as_posix(), ()):
            ph = parts.get(owner)
            if ph is None:
                ph = parts[owner] = hashlib.sha256()
            ph.update(key)
            ph.update(body)
    # A declared component with no files on disk still gets an entry -- an EMPTY digest, not a missing
    # one. Missing would read as undeterminable and re-run forever; empty is a real, comparable state
    # that changes the moment the component gains a file.
    comps = {
        name: (parts[name].hexdigest()[:16] if name in parts else hashlib.sha256().hexdigest()[:16]) for name in names
    }
    return h.hexdigest()[:16], comps, d["rejected"]


def _cert_cover(ws, *, context: InvocationContext, capsules_root: Path | None = None) -> set | None:
    """Which capsules are worth certifying at all. The hardware cannot tell two capsules in the same
    (family, dtype) cell apart, so certifying both spends minutes to learn nothing. `None` on any failure
    -> certify anything eligible, because a cover that silently comes back empty is indistinguishable
    from everything already being done."""
    try:
        from merlin.development.targetgen.corpus.materialization import cert_capsule_cover
        from merlin.target.registry.descriptor_loading import load_target_experiment

        te = load_target_experiment(context.descriptor)
        # Pass the tile edge so the cover certifies PARTIAL tiles as their own cell. A cover built on
        # family and dtype alone can pick, per cell, the capsule whose extents happen to divide evenly and
        # then certify no ragged extent anywhere -- and a partial tile is exactly what a functional model
        # is least able to stand in for (a taped-out unit here got `n % 64 != 0` wrong while every
        # functional check passed).
        _td = None
        try:
            from merlin.development.targetgen.corpus.binding import _tile_dim
            from merlin.target.registry.experiment_descriptor import load_capability_manifest

            _td = int(_tile_dim(te.target, load_capability_manifest(te.target).contract)) or None
        except Exception:  # noqa: BLE001 -- no derivable tile edge: cover without the alignment axis
            _td = None
        # Exclusions travel WITH the roots. A capsule the descriptor withholds from the paid loop cannot
        # stand for its cell, because promotion only enqueues capsules in the cover — picking one that
        # never runs retires the cell for a certificate nobody will produce.
        from merlin.development.targetgen.corpus.capsule_io import discover_capsules

        _source = discover_capsules(
            ((capsules_root,) if capsules_root is not None else te.graded_roots()),
            labels={"public", "dev"},
            contract=context.repo / "src/merlin/resources/contracts",
        )
        _excluded = set(te.effective_exclusions(cap.get("name") for cap in _source))
        return set(
            cert_capsule_cover(
                tuple(Path(cap["__dir__"]) for cap in _source),
                labels={"public", "dev"},
                tile_dim=_td,
                exclude=_excluded,
            )["capsules"]
        )
    except Exception:  # noqa: BLE001 -- no resolvable corpus: stay permissive, never silently empty
        return None


def _mandatory_cert_members(*, context: InvocationContext, capsules_root: Path | None, cert_tier: str) -> set[str]:
    """Declared mandatory tiers cannot be suppressed by representative coverage.

    Inspect only the selected graded public/dev cohort, never performance or hidden roots.
    Descriptor exclusions still apply. Optional certification keeps its ordinary cover.
    """
    from merlin.development.targetgen.corpus.capsule_io import discover_capsules
    from merlin.target.registry.descriptor_loading import load_target_experiment

    te = load_target_experiment(context.descriptor)
    caps = discover_capsules(
        ((capsules_root,) if capsules_root is not None else te.graded_roots()),
        labels={"public", "dev"},
        contract=context.repo / "src/merlin/resources/contracts",
    )
    excluded = set(te.effective_exclusions(cap.get("name") for cap in caps))
    return {
        str(cap["name"])
        for cap in caps
        if cap.get("name") and cap["name"] not in excluded and cert_tier in (cap.get("required_oracle_tiers") or ())
    }


def _tier_state(ws) -> dict:
    import json as _j

    f = Path(ws, "qa", "tier_state.json")
    try:
        return _j.loads(f.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_tier_state(ws, st) -> None:
    """Replace the state file ATOMICALLY.

    Two brokers write this file (the synchronous self-check broker and the async simjob broker). A plain
    ``write_text`` truncates first, so the other broker can read a half-written one; `_tier_state` answers
    an unparseable read with ``{}``, and the next save then PERSISTS that empty dict -- every capsule's
    recorded verdict gone, and re-bought at minutes each. A temp file in the same directory plus
    ``os.replace`` makes the swap atomic, so a concurrent reader sees either the old state or the new one
    and never an empty one.
    """
    import json as _j

    f = Path(ws, "qa", "tier_state.json")
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    tmp.write_text(_j.dumps(st, indent=2))
    os.replace(tmp, f)


def _valid_execution(value) -> bool:
    """Whether *value* is the artifact identity the result readers emit.

    Imported, never restated: two copies of this predicate is how the recorder and the scheduler come to
    disagree about what counts as an identity, and the disagreement always resolves as a silent fallback.
    """
    from merlin.development.targetgen.oracles.schedule import valid_execution_digest

    return valid_execution_digest(value)


def record_identity(entry) -> str:
    """Which bytes one recorded verdict belongs to, as a ledger key.

    The exact executable identity when the run that earned it produced one, else the whole-submission
    digest. The two key spaces cannot collide: an execution identity is 64 hex characters, and the
    fallback carries a prefix no digest can spell.
    """
    ed = (entry or {}).get("execution_digest")
    if _valid_execution(ed):
        return str(ed)
    return "submission:" + str((entry or {}).get("digest") or "")


def current_identity(execution_digest, digest) -> str:
    """The ledger key for the bytes on disk RIGHT NOW, in the same space as :func:`record_identity`."""
    if _valid_execution(execution_digest):
        return str(execution_digest)
    return "submission:" + str(digest)


def _slots_ro(st, name, tier) -> dict:
    """``{identity: entry}`` for one (capsule, tier). READ-ONLY.

    It must not create the capsule: a result for a capsule nobody promoted has to leave no trace at all
    (that is what makes an unattributable result recordable as "nothing happened" rather than as a guess).

    A state file written before the ledger existed carries only the single mirror entry. That is seeded
    into the returned map rather than reported as absent -- absent would read as "nothing was ever
    recorded here" and re-buy the cert tier for bytes that already hold a verdict.
    """
    led = (st.get(name) or {}).get(_LEDGER)
    slots = led.get(tier) if isinstance(led, dict) else None
    if isinstance(slots, dict) and slots:
        return slots
    mirror = (st.get(name) or {}).get(tier)
    return {record_identity(mirror): mirror} if isinstance(mirror, dict) else {}


def _slots(st, name, tier) -> dict:
    """:func:`_slots_ro`, materialized in ``st`` so it can be written to."""
    per = st.setdefault(name, {})
    led = per.get(_LEDGER)
    if not isinstance(led, dict):
        led = per[_LEDGER] = {}
    slots = led.get(tier)
    if not isinstance(slots, dict):
        slots = led[tier] = dict(_slots_ro(st, name, tier))
    return slots


def _record(st, name, tier, entry) -> None:
    """Store one verdict for (capsule, tier): in the ledger under the bytes it belongs to, AND as the
    ``st[capsule][tier]`` mirror every existing reader of this file already uses. The mirror is the most
    recently written record; the ledger is every record, bounded by :data:`_LEDGER_KEEP`."""
    from merlin.development.backend.feedback.certification.promotion import _LEDGER_KEEP

    key = record_identity(entry)
    slots = _slots(st, name, tier)
    slots.pop(key, None)  # re-insert so the most recently touched record is the newest
    slots[key] = entry
    st.setdefault(name, {})[tier] = entry
    # Evict oldest-first (dict order is insertion order, and the state file round-trips it), skipping the
    # record just written and every outstanding one.
    for old in list(slots):
        if len(slots) <= _LEDGER_KEEP:
            break
        if old == key or (slots[old] or {}).get("status") == "pending":
            continue
        del slots[old]


def _log_invalidations(states, executions, components, loop_tier, cert_tier, log, *, observed_names=None):
    """Report only current observed cases in per-completion joint callbacks.

    Other ledger entries have no fresh executable identity in this callback;
    comparing them to this source cannot establish that their code changed.
    Scheduling, evidence retention and ordinary non-joint diagnostics are untouched.
    """
    from merlin.development.targetgen.oracles.schedule import UNATTRIBUTED_RESULT, WHOLE_SUBMISSION

    unobserved = 0
    for state in states:
        if observed_names is not None and state.name not in observed_names:
            unobserved += 1
            continue
        broad = _no_narrower_cause(executions.get(state.name), components)
        for tier in (loop_tier, cert_tier):
            for why in state.invalidated_by(tier) if tier in state.verdicts else ():
                extra = f"; no narrower cause: {broad}" if why.component == WHOLE_SUBMISSION and broad else ""
                retried = getattr(why, "reason", None) == UNATTRIBUTED_RESULT  # not a changed program
                said = "requeued: earlier job unattributed" if retried else f"invalidated by {why}"
                print(f"[promote] {state.name} {tier} {said}{extra}", file=log, flush=True)
    if unobserved:
        print(
            f"[promote] {unobserved} historical capsule(s) not re-observed by this callback; "
            "no new invalidation diagnosis for them",
            file=log,
            flush=True,
        )


def _recorded_tiers(per) -> list:
    """Every tier one capsule holds a verdict for, from the mirror and the ledger both. The reserved
    ledger key is never mistaken for a tier -- a phantom tier would enter the scheduler as a verdict."""
    out = {t for t, v in (per or {}).items() if t != _LEDGER and isinstance(v, dict)}
    led = (per or {}).get(_LEDGER)
    if isinstance(led, dict):
        out |= {t for t, v in led.items() if isinstance(v, dict)}
    return sorted(out)


def _verdict_for(st, name, tier, identity):
    """The record to judge (capsule, tier) by: the one earned by EXACTLY the current bytes when the
    ledger holds it, otherwise the mirror.

    Preferring the exact-identity record is what stops a certificate being discarded by an edit it does
    not depend on. It cannot loosen anything: the record that comes back is still put through
    ``CapsuleState.invalidated_by``, so a record for other bytes is invalidated rather than trusted, and
    a capsule with no matching record falls back to exactly today's comparison.
    """
    slots = _slots_ro(st, name, tier)
    hit = slots.get(identity)
    if isinstance(hit, dict):
        return hit
    mirror = (st.get(name) or {}).get(tier)
    return mirror if isinstance(mirror, dict) else None


def _no_narrower_cause(execution_digest, comps) -> str | None:
    """Why this capsule's staleness cannot be decided by anything narrower than bytes every capsule
    shares, or ``None`` when the narrowest identity was available.

    The conservative fallback stays the fallback, but it stops being SILENT. Every line of the log that
    diagnosed this defect read ``invalidated by <whole-submission> (changed)`` and named none of the
    inputs that were missing, so a correct conservative answer and a broken decomposition looked
    identical. Each clause below names one input that was absent -- so the reason is recorded, on disk
    and in the log, rather than inferred.

    NOTE the two clauses are different KINDS of loss and both are worth naming. Without an execution
    identity, every capsule rides on the same submission bytes and one capsule's edit requeues the whole
    corpus. Without a decomposition on top of that, even bytes no declared command can read count.
    """
    if execution_digest is not None:
        return None  # the narrowest identity is available; no fallback happened
    why = ["no execution identity for this capsule in this verdict"]
    if not comps:
        why.append("the submission has no component decomposition, declared or derived")
    return "; ".join(why)
