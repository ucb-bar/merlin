"""Loop-tier -> cert-tier promotion, shared by BOTH grading brokers.

This lives in its own module because hooking only one broker is exactly the bug it was written to avoid.
Promotion was first wired into the async oracle alone, and a live run then showed the agent using the
SYNCHRONOUS self-check seven times to the async path's two -- so eight verdicts completed and promotion
fired zero times. The rule is: wherever a verdict is produced, promotion is considered. One module, two
call sites, no third copy.

The POLICY (which capsule earns the cert tier, and when a verdict may be reused) is not here either --
it is `merlin.targetgen.oracle_schedule`, which has the tests. This module is the plumbing that records
what a verdict said and enqueues what the policy asks for.

What this file DOES own is the content addressing: the whole-submission digest and its decomposition into
per-component digests (`submission_digests`). The decomposition exists because the whole-submission digest
invalidates every certificate on every edit -- fine while a round is proving functional completeness once,
ruinous for an optimization phase that edits the compiler continuously and would re-buy the minutes-per-
capsule cert tier for the whole corpus each time. The component vocabulary is DERIVED from the
submission's own manifest, never listed here; see `component_vocabulary`.
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
    return submission_digests(ws)[0]


def _manifest(ws) -> dict | None:
    """The submission's own package manifest, or ``None`` if it cannot be read.

    ``None`` is the undeterminable state and is handled as such by every caller: no vocabulary, no
    decomposition, whole-submission digest. It is NOT "the submission declares no components".
    """
    import yaml
    try:
        doc = yaml.safe_load(Path(ws, "submission", "manifest.yaml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- absent/unparseable manifest: undeterminable, never assumed empty
        return None
    return doc if isinstance(doc, dict) else None


def component_vocabulary(ws) -> set | None:
    """The legal component names for this submission, DERIVED from its manifest's ``commands`` keys.

    Why the command keys and not a list written down here. The experiment ABI already requires every
    package to declare exactly these entrypoints (``merlin/contract/schemas/manifest.schema.json``:
    ``commands.required = [parse, lower_interface_to_target, emit_command_buffer,
    lower_target_to_llvm]``, open for a target to add more), and they are the SAME axis the grader drives
    the submission through -- ``oot_runner._resolve_argv`` looks each capsule's compile step up by these
    keys. So "which command's files moved" is exactly "which step of grading this capsule could now answer
    differently", with no per-target code and no vocabulary invented here.

    The two alternatives were rejected on the facts: the Merlin pass registry
    (``xdsl_dialects.lowering.passes.CATALOG``) names Merlin's OWN passes, whose ``entry`` fields are
    ``merlin.*`` dotted paths -- an out-of-tree submission does not contain those modules, so its files
    cannot be attributed to them; and a bundle's ``allowed`` paths are read-only REPO grants
    (``merlin/contract/``, ``third_party/llvm-install/``), not submission files, so they cannot decompose
    a submission at all.

    Both spellings of the renamed 4th entrypoint resolve, reusing ``oot_runner``'s own alias map rather
    than restating it -- two copies of an alias table drift, and the one that drifts is the untested one.
    ``None`` when the manifest is unreadable (undeterminable), never an empty set.
    """
    m = _manifest(ws)
    if m is None or not isinstance(m.get("commands"), dict):
        return None
    names = {str(k) for k in m["commands"]}
    try:
        from merlin.targetgen.oot_runner import _ENTRYPOINT_ALIASES as _AL
    except Exception:  # noqa: BLE001 -- merlin not importable from the harness: names as declared
        _AL = {}
    return names | {_AL[n] for n in names if n in _AL}


def component_paths(ws) -> tuple[dict, list] | None:
    """``({component: (path-prefix, ...)}, [rejected, ...])`` from the submission's own manifest.

    The manifest's top-level ``components:`` block maps a component name to the submission-relative paths
    that implement it. It needs no schema change: ``manifest.schema.json`` is ``additionalProperties:
    true`` at the top level (its per-command objects are NOT, which is why the attribution lives beside
    ``commands`` rather than inside each one).

    A key outside :func:`component_vocabulary` is REJECTED and returned in the second slot rather than
    honoured, so a typo cannot quietly mint a component that no capsule's ``depends_on`` will ever match
    and that therefore holds no bytes -- a component with no files never changes, so every certificate
    depending on it would live forever.

    ``None`` when there is no vocabulary or no ``components`` block: undeterminable/undeclared, and the
    caller falls back to the whole-submission digest.
    """
    m = _manifest(ws)
    vocab = component_vocabulary(ws)
    if m is None or vocab is None or not isinstance(m.get("components"), dict):
        return None
    out, rejected = {}, []
    for name, paths in m["components"].items():
        name = str(name)
        if name not in vocab:
            rejected.append(name)
            continue
        if isinstance(paths, str):
            paths = [paths]
        out[name] = tuple(_rel_prefix(p) for p in (paths or []) if str(p).strip())
    # Returned even when `out` is empty: an all-rejected block must still be REPORTED. Swallowing it
    # would make "every component name was a typo" look exactly like "no components declared".
    return out, rejected


def _rel_prefix(p) -> str:
    """One declared path, normalized to submission-relative posix. Structural, no pattern matching."""
    s = str(p).strip().replace("\\", "/").strip("/")
    while s.startswith("./"):
        s = s[2:]
    if s == "submission":
        return ""
    if s.startswith("submission/"):
        s = s[len("submission/"):]
    return s


def _owner(rel: str, prefixes: dict) -> str | None:
    """Which component owns submission-relative ``rel``; the LONGEST declared prefix wins so a nested
    grant (``mlir_oot/lowering/tile.py``) is not swallowed by its parent (``mlir_oot/``).

    Two DIFFERENT components claiming it at the same depth is a tie, and a tie returns ``None`` -- the
    file falls to ``UNATTRIBUTED``, which every capsule depends on. Breaking the tie by dict order would
    make ownership depend on the order keys happen to appear in the manifest, so the same file could be
    attributed differently after a cosmetic reordering and a certificate would survive an edit it should
    not have. Fail closed: an ambiguously-owned file belongs to everyone.
    """
    best, best_len, tied = None, -1, False
    for name, pres in prefixes.items():
        for pre in pres:
            if pre and not (rel == pre or rel.startswith(pre + "/")):
                continue
            if len(pre) > best_len:
                best, best_len, tied = name, len(pre), False
            elif len(pre) == best_len and name != best:
                tied = True
    return None if tied else best


def submission_digests(ws) -> tuple:
    """``(whole_digest, {component: digest}, rejected_names)`` for the submission on disk.

    The whole digest is byte-for-byte what it always was -- one pass, sorted paths, path bytes then file
    bytes -- so turning components on never invalidates a certificate by itself. Each file additionally
    feeds its owning component's hasher; a file no component claims feeds ``UNATTRIBUTED``, which every
    capsule depends on. The component map is ``{}`` when nothing is declared, and the scheduler reads that
    as "compare the whole submission" (see ``CapsuleState._dep_components``).
    """
    import hashlib
    from merlin.targetgen.oracle_schedule import UNATTRIBUTED
    decl = component_paths(ws)
    prefixes, rejected = decl if decl else ({}, [])
    h = hashlib.sha256()
    parts = {}
    sub = Path(ws, "submission")
    for f in sorted(sub.rglob("*")):
        if not (f.is_file() and "__pycache__" not in f.parts):
            continue
        key, body = f.relative_to(ws).as_posix().encode(), f.read_bytes()
        h.update(key)
        h.update(body)
        if not prefixes:
            continue
        owner = _owner(f.relative_to(sub).as_posix(), prefixes) or UNATTRIBUTED
        ph = parts.get(owner)
        if ph is None:
            ph = parts[owner] = hashlib.sha256()
        ph.update(key)
        ph.update(body)
    comps = {}
    if prefixes:
        # A declared component with no files on disk still gets an entry -- an EMPTY digest, not a missing
        # one. Missing would read as undeterminable and re-run forever; empty is a real, comparable state
        # that changes the moment the component gains a file.
        for name in prefixes:
            comps[name] = parts[name].hexdigest()[:16] if name in parts else hashlib.sha256().hexdigest()[:16]
        comps[UNATTRIBUTED] = (parts[UNATTRIBUTED].hexdigest()[:16] if UNATTRIBUTED in parts
                               else hashlib.sha256().hexdigest()[:16])
    return h.hexdigest()[:16], comps, rejected


def capsule_dependencies(roots) -> dict:
    """``{capsule name: (component, ...)}`` from each capsule's own ``depends_on``.

    Read straight off ``capsule.yaml`` rather than through the validating loader: a corpus bug must not
    turn promotion off, and a capsule this cannot read simply has no declared dependency set -- which
    means "depends on everything", the fail-closed answer.
    """
    import yaml
    out = {}
    for root in roots or ():
        for cy in sorted(Path(root).glob("*/capsule.yaml")):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except Exception:  # noqa: BLE001 -- unreadable capsule: no declaration, so whole-submission
                continue
            dep = cap.get("depends_on")
            name = cap.get("name") or cy.parent.name
            if isinstance(dep, str):
                dep = [dep]
            if isinstance(dep, list) and dep:
                out[str(name)] = tuple(str(d) for d in dep)
    return out


def _graded_roots():
    """The corpus roots this experiment grades, or ``()`` when the descriptor is unreachable."""
    try:
        from merlin.targetgen.target_experiment import load_target_experiment
        import _common as _C
        return tuple(load_target_experiment(_C.EXP / "target_experiment.yaml").graded_roots())
    except Exception:  # noqa: BLE001 -- no descriptor here: no declarations, so whole-submission digests
        return ()


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


def promote(ws, ch, verdict, loop_tier, cert_tier, cover, log):
    """Record what the loop tier just learned, and enqueue cert jobs for what it unlocked.

    Returns the capsule names promoted. Enqueues by writing a `simreq_` the broker's own queue picks up --
    the same path an agent request takes, so it inherits the constrained-runner validation rather than
    routing around it.
    """
    import json as _j
    from merlin.targetgen.oracle_schedule import CapsuleState, Verdict, schedule

    digest, comps, rejected = submission_digests(ws)
    deps = capsule_dependencies(_graded_roots())
    st = _tier_state(ws)
    if rejected:
        # Loud, not silent: a rejected name means a capsule's declared dependency can never match, so it
        # falls back to the whole submission and quietly loses the saving it was written to get.
        print(f"[promote] manifest components outside the declared command vocabulary, ignored: "
              f"{sorted(rejected)}", file=log, flush=True)

    # Record what the loop tier just learned, keyed by the bytes that earned it -- BOTH the whole digest
    # and the per-component decomposition, because a verdict that carries only the whole digest cannot be
    # re-examined per component later (it comes back UNDETERMINABLE, which re-runs).
    for row in (verdict.get("per_capsule") or []):
        name = row.get("capsule")
        if not name:
            continue
        st.setdefault(name, {})[loop_tier] = {
            "status": "pass" if row.get("pass") else "fail", "digest": digest, "components": dict(comps)}

    # WHAT to run next is `oracle_schedule`'s decision, not this file's. The rules (a cert tier is gated
    # on the loop tier passing; the cert tier runs a representative cover; a verdict already earned by
    # these bytes is never re-run) were implemented here once and in the scheduler once, which is one
    # implementation too many -- two expressions of the same policy drift, and the one that drifts is
    # whichever has no tests. The scheduler has them; this is now only plumbing.
    states = [CapsuleState(name=n, digest=digest,
                           verdicts={t: Verdict(v.get("status"), v.get("digest"),
                                                dict(v.get("components") or {}))
                                     for t, v in (e or {}).items() if isinstance(v, dict)},
                           components=dict(comps), depends_on=deps.get(n))
              for n, e in st.items()]
    want = [w for w in schedule(states, tier_order=[loop_tier, cert_tier], cert_tiers=(cert_tier,),
                                cert_cover=cover)
            if w.tier == cert_tier]

    # WHICH component requeued each capsule, so a reader of the log can see why a certificate was dropped
    # rather than only that the count went up. A run that requeues everything and one that requeues one
    # capsule are indistinguishable from the promotion count alone.
    for s in states:
        for tier in (loop_tier, cert_tier):
            # Only tiers that HAD a verdict: a tier nobody ever ran was not invalidated by anything, and
            # logging that would bury the real signal under one line per capsule per round.
            for why in (s.invalidated_by(tier) if tier in s.verdicts else ()):
                print(f"[promote] {s.name} {tier} invalidated by {why}", file=log, flush=True)

    promoted = []
    for w in want:
        st.setdefault(w.capsule, {})[cert_tier] = {
            "status": "pending", "digest": digest, "components": dict(comps)}
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
