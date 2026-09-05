"""Loop-tier -> cert-tier promotion, shared by BOTH grading brokers.

This lives in its own module because hooking only one broker is exactly the bug it was written to avoid.
Promotion was first wired into the async oracle alone, and a live run then showed the agent using the
SYNCHRONOUS self-check seven times to the async path's two -- so eight verdicts completed and promotion
fired zero times. The rule is: wherever a verdict is produced, promotion is considered. One module, two
call sites, no third copy.

The POLICY (which capsule earns the cert tier, and when a verdict may be reused) is not here either --
it is `merlin.targetgen.oracle_schedule`, which has the tests. This module is the plumbing that records
what a verdict said and enqueues what the policy asks for.

What this file DOES own is content addressing. A per-capsule execution digest identifies the exact ELF and
hardware revision that a tier ran; whole-submission and per-component digests are the conservative fallback
when no such artifact can be identified. This prevents a source-only edit that emits identical code from
re-buying every minutes-per-capsule cert while still failing closed for legacy or incomplete results. The
component vocabulary is DERIVED from the submission's own manifest, never listed here; see
`component_vocabulary`.

Verdicts are RETAINED per identity, not overwritten (see `_LEDGER`). A cert job takes minutes and the
agent keeps editing while it runs, so the next loop verdict used to re-enqueue the capsule and overwrite
the record the in-flight job belonged to -- which then arrived unattributable and was dropped. Keeping one
record per identity means a completed job resolves the record enqueued FOR IT, and a certificate survives
an edit that is later reverted; nothing is ever consulted unless its identity matches the current bytes
exactly, so retention cannot make a stale certificate look valid.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

_NEUTRAL_SIM = "contract"   # "grade on whatever tier this target's contract resolves to"


from merlin.targetgen.rtl_engine_policy import ELABORATED_RTL as _ELABORATED_RTL


def cert_sim(cert_tier: str) -> str | None:
    """The ``--sim`` token the BROKER will accept for ``cert_tier``, or None when none does.

    ``promote()`` used to write :data:`_NEUTRAL_SIM` unconditionally. That is correct only for a target
    whose ladder comes from its own contract, where ``--sim`` does not apply and the sentinel is the ONLY
    accepted token. A target that declares a bespoke sim ladder accepts that ladder's names and REJECTS
    the sentinel -- so every promotion request such a target wrote was refused, while the capsule had
    already been marked ``pending`` a few lines earlier and therefore stayed pending forever.

    Measured on the live gemmini round merlincirct_arm4_func_20260901_v4: 6 promotion requests, every one
    answered "rejected: --sim 'contract' is not accepted for this target. Use 'spike' or 'verilator' or
    'vcs'", and 2 capsules stranded at L3 pending. Promotion had never fired on a bespoke-sim target.

    Derived from the same allowlist the broker validates against, so the two cannot drift apart again.
    Returns None rather than a guess when nothing serves the tier: the caller must then NOT enqueue, and
    must not mark the capsule pending for a job that can never run.
    """
    try:
        from simjob_broker import _allowed_sims
        allowed = tuple(_allowed_sims())
    except Exception:  # noqa: BLE001 -- broker not importable: keep the historical sentinel
        return _NEUTRAL_SIM
    if allowed == (_NEUTRAL_SIM,):
        return _NEUTRAL_SIM
    try:
        import _common as _C
        from merlin.targetgen.runner_config import runner_config_from_manifest
        from merlin.targetgen.target_experiment import (load_capability_manifest,
                                                        load_target_experiment)
        te = load_target_experiment(_C.EXP / "target_experiment.yaml")
        cfg = runner_config_from_manifest(load_capability_manifest(te.target))
        required = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
        if required and cert_tier in set(cfg.rtl_tiers or ()):
            # A tier is a fidelity, not a historical binary binding.  Under an experiment-wide pin the
            # required engine serves every elaborated-RTL tier it implements; consulting tier_sim first
            # would recover the manifest's old Verilator label, find it excluded by the broker, and
            # silently disable promotion in a GSIM-only run.
            return required if required in allowed else None
        sim = (cfg.tier_sim or {}).get(cert_tier)
        # THE CONTRACT NAMES A FIDELITY, NOT A BINARY. `tier_sim` used to read `{L3: verilator}` and the
        # allowlist happened to contain that word, so this returned an engine by accident. Once the
        # contract said what it means -- `{L3: elaborated_rtl}` -- the sentinel matched no `--sim` token,
        # this returned None, and promotion silently switched off for every unpinned run: the caller logs
        # "no --sim serves L3" once and then never enqueues, which is indistinguishable from a round with
        # nothing to promote. Resolve the sentinel through the SAME availability policy the contract
        # comment names, so the fidelity is declared in one place and the engine chosen in one place.
        if sim == _ELABORATED_RTL:
            from merlin.targetgen.capsule_runner import chipyard_l3_selection
            sim = str((chipyard_l3_selection(te.target) or {}).get("engine") or "").strip() or None
    except Exception:  # noqa: BLE001 -- unresolvable map: no promotion, and the caller says so
        return None
    return sim if sim in allowed else None


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


def execution_digest(capsule_result: str | Path) -> str | None:
    """Content identity for exactly what one capsule's hardware tier executes.

    The identity covers the ELF bytes and the target/hardware revisions recorded beside the result. It
    deliberately excludes Merlin's source commit: a source edit that emits byte-identical code has not
    changed the program RTL certifies. Missing ELF, target, or a concrete hardware revision returns
    ``None`` so scheduling falls back to the conservative submission/component digest.
    """
    import hashlib
    import json
    import yaml

    cr = Path(capsule_result)
    try:
        result = json.loads(cr.read_text(encoding="utf-8"))
        manifest = yaml.safe_load((cr.parent / "run_manifest.yaml").read_text(encoding="utf-8"))
        elf = cr.parent / "generated" / "package_kernel.elf"
        target = manifest.get("target") if isinstance(manifest, dict) else None
        shas = result.get("toolchain_shas") if isinstance(result, dict) else None
        if not isinstance(target, str) or not target.strip() or not isinstance(shas, dict):
            return None
        hardware = {}
        for key, value in shas.items():
            if str(key).lower() == "merlin":
                continue
            # Hardware pins are full git/content hashes. UNKNOWN, abbreviated, or otherwise malformed
            # provenance cannot safely identify the design that executed the program.
            if (not isinstance(key, str) or not key or not isinstance(value, str)
                    or len(value) not in (40, 64)
                    or not all(c in "0123456789abcdef" for c in value)):
                return None
            hardware[key] = value
        if not hardware or not elf.is_file():
            return None
        payload = {
            "version": 1,
            "target": target,
            "hardware": hardware,
            "executable_sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
    except Exception:  # noqa: BLE001 -- missing/unreadable provenance is the conservative fallback
        return None


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
    """Replace the state file ATOMICALLY.

    Two brokers write this file (the sync self-check broker and the async simjob broker). A plain
    ``write_text`` truncates first, so the other broker can read a half-written file; `_tier_state`
    answers an unparseable read with ``{}``, and the next save then persists that empty dict -- every
    capsule's recorded verdict gone, and re-bought at minutes each. A temp file in the same directory
    plus ``os.replace`` makes the swap atomic, so a concurrent reader sees either the old state or the
    new one and never an empty one.
    """
    import json as _j
    import os
    f = Path(ws, "qa", "tier_state.json")
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    tmp.write_text(_j.dumps(st, indent=2))
    os.replace(tmp, f)


# ------------------------------------------------------------------------------------------------
# The certificate LEDGER: one recorded verdict per (capsule, tier, the bytes it belongs to).
#
# The single-slot record this replaces was the reason a paid-for certificate was thrown away. A cert job
# takes minutes; the agent keeps editing while it runs; the next loop verdict re-enqueued the capsule and
# OVERWROTE the pending record. When the in-flight job then finished, `record_cert` found a record for
# different bytes, could not attribute the result (correctly -- see its docstring), and dropped it. So the
# RTL time bought nothing and the capsule read `certified: None` forever. Measured on
# merlincirct_arm4_func_20260901_codex1 round 1: 17 cert jobs, all passing 1/1, zero certificates
# recorded.
#
# Keyed per identity, a verdict is RETAINED instead of discarded: a completed job resolves the record that
# was enqueued FOR IT, and an edit that changes a capsule's program leaves the old record in place rather
# than destroying it -- so if those exact bytes come back, so does their certificate. Nothing is ever
# consulted unless its identity matches the current bytes EXACTLY, which is why retaining cannot turn a
# stale certificate into a valid one.
# ------------------------------------------------------------------------------------------------
_LEDGER = "<certs>"   # reserved key inside a capsule's tier map; delimiters no tier label can produce

# How many records one (capsule, tier) retains. The state file is re-read and re-written on EVERY verdict,
# and a continuous round produces dozens per hour, so an unbounded ledger would grow the hot file without
# limit. The bound only ever costs a re-run (the conservative direction): an evicted record means the
# scheduler no longer knows those bytes passed, never that it believes they did. An OUTSTANDING record is
# never evicted -- that is an in-flight job whose result would otherwise become unattributable.
_LEDGER_KEEP = 16


def _valid_execution(value) -> bool:
    """Whether *value* is the artifact identity the result readers emit.

    Imported, never restated: two copies of this predicate is how the recorder and the scheduler come to
    disagree about what counts as an identity, and the disagreement always resolves as a silent fallback.
    """
    from merlin.targetgen.oracle_schedule import valid_execution_digest
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
    key = record_identity(entry)
    slots = _slots(st, name, tier)
    slots.pop(key, None)                 # re-insert so the most recently touched record is the newest
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


def _no_narrower_cause(execution_digest, comps, depends_on) -> str | None:
    """Why nothing narrower than the whole submission can decide this capsule's staleness, or ``None``
    when something narrower can.

    The conservative fallback stays the fallback, but it stops being SILENT. Every line of the log that
    diagnosed this defect read ``invalidated by <whole-submission> (changed)`` and named none of the
    inputs that were missing, so a correct conservative answer and a broken decomposition looked
    identical. Each clause below names one input that was absent -- so the reason is recorded, on disk
    and in the log, rather than inferred.
    """
    if execution_digest is not None:
        return None                      # the narrowest identity is available; no fallback happened
    why = ["no execution identity for this capsule in this verdict"]
    if not depends_on:
        why.append("the capsule declares no depends_on")
    if not comps:
        why.append("the submission manifest declares no components")
    return "; ".join(why)


def record_cert(ws, verdict, cert_tier, log=None, identity=None) -> list[str]:
    """Write a COMPLETED promotion's result into the tier state, against the bytes that earned it.

    `promote()` marks a capsule `pending` when it enqueues the cert job, and the broker's reap skips
    `_promote` for a job that WAS a promotion -- correctly, so a cert verdict cannot re-enqueue itself.
    But nothing then wrote the outcome back, so a capsule stayed `pending` forever: the certificate was
    earned on real RTL and discarded, and the next loop verdict re-certified the same bytes.

    Measured on merlincirct_arm4_func_20260901_v4 and _p2: promotions fired and COMPLETED (21 and 3
    `simdone_promo*` respectively, one verified `barrier_tier=L3 barrier_status=pass`), while both
    tier states showed only `L3: pending` and never once `L3: pass`.

    The digest is NOT recomputed here. A cert belongs to the exact bytes that were pending when the job
    was enqueued; re-hashing now would attribute it to whatever the agent has edited since. So this only
    resolves an existing pending entry, and leaves anything else alone -- a result with no pending entry
    is a result we cannot attribute, and that is recorded by doing nothing rather than by guessing.

    That property is now enforced against the per-identity LEDGER rather than against a single slot. The
    single slot was overwritten by the next re-enqueue, so a completed job routinely arrived to find a
    record for OTHER bytes: with an artifact identity that was refused (the cert was dropped), and
    WITHOUT one it was silently accepted and the certificate was re-attributed to bytes that never earned
    it. Looking the completed job's own identity up in the ledger resolves exactly the record it belongs
    to, and refuses when no such record exists.

    ``identity`` is the ledger key the ENQUEUER wrote onto the request, forwarded by the reap. It is what
    lets a result whose reader produced no artifact identity still be attributed exactly, instead of
    falling back to "the one outstanding record" (which cannot be used at all once two are outstanding).
    A result that DOES carry an artifact identity always decides for itself: the job may have launched
    after an edit, and then only the identity it actually ran may be credited.
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
                if log is not None:
                    print(f"[promote] {name} {cert_tier} result not recorded: no outstanding record "
                          f"for the execution identity this job ran", file=log, flush=True)
                continue
        elif identity is not None:
            # The enqueuer stated which record this job was launched for. That is a fact about the
            # request, not a guess about the result, so it attributes exactly -- including when several
            # records are outstanding at once, which is the case the heuristic below cannot serve.
            key = str(identity)
            entry = slots.get(key)
            if not isinstance(entry, dict) or entry.get("status") != "pending":
                if log is not None:
                    print(f"[promote] {name} {cert_tier} result not recorded: the record this job was "
                          f"enqueued for is no longer outstanding", file=log, flush=True)
                continue
            if _valid_execution(entry.get("execution_digest")):
                # The record names an exact artifact and the result does not carry one -- so whether the
                # job ran THAT artifact cannot be established. The request only says what was asked for;
                # the agent may have edited between enqueue and launch. Fail closed: an unreadable
                # artifact identity is never evidence that the right artifact ran.
                if log is not None:
                    print(f"[promote] {name} {cert_tier} result not recorded: the record names an "
                          f"execution identity and the result carries none", file=log, flush=True)
                continue
        else:
            # No artifact identity on the result and none on the request: the only attributable case is a
            # single outstanding record that ALSO carries no artifact identity. Two outstanding records
            # and a result that cannot say which it belongs to is precisely the misattribution this
            # guards -- refuse.
            outstanding = {i: e for i, e in slots.items()
                           if isinstance(e, dict) and e.get("status") == "pending"}
            if len(outstanding) != 1:
                if log is not None and outstanding:
                    print(f"[promote] {name} {cert_tier} result not recorded: {len(outstanding)} "
                          f"outstanding records and the result carries no execution identity, so which "
                          f"bytes earned it cannot be determined", file=log, flush=True)
                continue
            key, entry = next(iter(outstanding.items()))
            if _valid_execution(entry.get("execution_digest")):
                if log is not None:
                    print(f"[promote] {name} {cert_tier} result not recorded: the outstanding record "
                          f"names an execution identity this result does not carry", file=log, flush=True)
                continue
        passed = bool(row.get("pass"))
        entry = dict(entry)
        entry["status"] = "pass" if passed else "fail"
        slots_w = _slots(st, name, cert_tier)
        slots_w.pop(key, None)           # re-insert: a just-resolved record is the freshest, not the oldest
        slots_w[key] = entry
        # The mirror moves only when it is the SAME record. Moving it to a different identity is exactly
        # the re-attribution this function exists to prevent.
        mirror = (st.get(name) or {}).get(cert_tier)
        if isinstance(mirror, dict) and record_identity(mirror) == key:
            st[name][cert_tier] = entry
        resolved.append(f"{name}={'pass' if passed else 'fail'}")
    if resolved:
        _save_tier_state(ws, st)
        if log is not None:
            print(f"[promote] {cert_tier} recorded: {resolved}", file=log, flush=True)
    return resolved


def promote(ws, ch, verdict, loop_tier, cert_tier, cover, log):
    """Record what the loop tier just learned, and enqueue cert jobs for what it unlocked.

    Returns the capsule names promoted. Enqueues by writing a `simreq_` the broker's own queue picks up --
    the same path an agent request takes, so it inherits the constrained-runner validation rather than
    routing around it.
    """
    import json as _j
    from merlin.targetgen.oracle_schedule import (WHOLE_SUBMISSION, CapsuleState, Verdict, schedule,
                                                   valid_execution_digest)

    digest, comps, rejected = submission_digests(ws)
    deps = capsule_dependencies(_graded_roots())
    st = _tier_state(ws)
    if rejected:
        # Loud, not silent: a rejected name means a capsule's declared dependency can never match, so it
        # falls back to the whole submission and quietly loses the saving it was written to get.
        print(f"[promote] manifest components outside the declared command vocabulary, ignored: "
              f"{sorted(rejected)}", file=log, flush=True)

    # Record what the loop tier just learned. The exact per-capsule executable identity is preferred when
    # present; the whole/component submission digests remain the conservative fallback for legacy or
    # undeterminable rows.
    execution_by_name = {}
    for row in (verdict.get("per_capsule") or []):
        name = row.get("capsule")
        if not name:
            continue
        name = str(name)
        execution_digest = row.get("execution_digest")
        if not valid_execution_digest(execution_digest):
            execution_digest = None
        execution_by_name[name] = execution_digest
        entry = {
            "status": "pass" if (row.get("pass") if "pass" in row
                                   else row.get("status") == "pass") else "fail",
            "digest": digest, "components": dict(comps)}
        if execution_digest is not None:
            entry["execution_digest"] = execution_digest
        # RECORD WHY, on disk, when this verdict can only be compared against the whole submission. A
        # conservative fallback that says nothing is why the defect read as correct behaviour for a round.
        _why_broad = _no_narrower_cause(execution_digest, comps, deps.get(name))
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
        _id = identity_by_name[n]
        verdicts = {}
        for t in _recorded_tiers(e):
            v = _verdict_for(st, n, t, _id)
            if isinstance(v, dict):
                verdicts[t] = Verdict(v.get("status"), v.get("digest"),
                                      dict(v.get("components") or {}), v.get("execution_digest"))
        states.append(CapsuleState(name=n, digest=digest, verdicts=verdicts,
                                   components=dict(comps), depends_on=deps.get(n),
                                   execution_digest=execution_by_name.get(n)))
    want = [w for w in schedule(states, tier_order=[loop_tier, cert_tier], cert_tiers=(cert_tier,),
                                cert_cover=cover)
            if w.tier == cert_tier]

    # WHICH component requeued each capsule, so a reader of the log can see why a certificate was dropped
    # rather than only that the count went up. A run that requeues everything and one that requeues one
    # capsule are indistinguishable from the promotion count alone. A whole-submission cause additionally
    # NAMES the inputs that were missing -- the diagnosis of this defect stalled on 21 identical
    # `<whole-submission> (changed)` lines that named none of them.
    for s in states:
        _broad = _no_narrower_cause(execution_by_name.get(s.name), comps, deps.get(s.name))
        for tier in (loop_tier, cert_tier):
            # Only tiers that HAD a verdict: a tier nobody ever ran was not invalidated by anything, and
            # logging that would bury the real signal under one line per capsule per round.
            for why in (s.invalidated_by(tier) if tier in s.verdicts else ()):
                _extra = f"; no narrower cause: {_broad}" if (why.component == WHOLE_SUBMISSION
                                                              and _broad) else ""
                print(f"[promote] {s.name} {tier} invalidated by {why}{_extra}", file=log, flush=True)

    promoted = []
    _sim = cert_sim(cert_tier)
    if want and _sim is None:
        # Say it once, loudly. Marking a capsule pending for a job that cannot be enqueued is how a
        # capsule strands at `pending` and never resolves.
        print(f"[promote] no --sim this broker accepts serves {cert_tier}; {len(want)} capsule(s) NOT "
              f"enqueued and NOT marked pending", file=log, flush=True)
    for w in want:
        if _sim is None:
            continue
        execution_digest = execution_by_name.get(w.capsule)
        token = execution_digest[:16] if execution_digest is not None else digest
        key = identity_by_name.get(w.capsule) or current_identity(execution_digest, digest)
        if key in _slots_ro(st, w.capsule, cert_tier):
            # These exact bytes already hold a cert-tier record -- outstanding, passed, or failed. Buying
            # a second copy of a verdict we already own is the 17-jobs-for-two-capsules waste itself.
            continue
        jid = f"promo{len(promoted)}_{token}_{w.capsule}"[:80]
        req = ch / f"simreq_{jid}.json"
        if req.exists():
            # The ledger says these bytes hold no record, yet a request for them is on the queue. That is
            # a state the ledger cannot explain (a hand-cleared state file, a request written by a broker
            # whose state write was lost). Say so instead of skipping silently: the silent version is how
            # a capsule stranded with a request nobody would ever answer.
            print(f"[promote] {w.capsule} {cert_tier}: {req.name} is already queued while the tier "
                  f"state holds no record for these bytes; NOT re-enqueued and NOT marked outstanding",
                  file=log, flush=True)
            continue
        req.write_text(_j.dumps(
            # `identity` travels WITH the request so the reap can hand a completed result back to the
            # exact record it was launched for. Without it, a result whose reader produced no artifact
            # identity is unattributable as soon as a second record is outstanding -- and the certificate
            # the RTL just paid for is dropped.
            {"sim": _sim, "capsules": w.capsule, "workers": 1, "tiers": cert_tier,
             "promoted": True, "identity": key, "submitted_at": time.time()}))
        # Mark pending only once the request is actually on the queue.
        pending = {"status": "pending", "digest": digest, "components": dict(comps)}
        if execution_digest is not None:
            pending["execution_digest"] = execution_digest
        _why_broad = _no_narrower_cause(execution_digest, comps, deps.get(w.capsule))
        if _why_broad:
            pending["fallback_reason"] = _why_broad
        _record(st, w.capsule, cert_tier, pending)
        promoted.append(w.capsule)
    _save_tier_state(ws, st)
    if promoted:
        print(f"[promote] {loop_tier} pass -> {cert_tier}: {promoted}", file=log, flush=True)
    return promoted
