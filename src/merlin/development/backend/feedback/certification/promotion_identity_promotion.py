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
from merlin.development.backend.feedback.certification.promotion_admission import affordable_cert_items, cert_sim
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


def execution_digest(capsule_result: str | Path) -> str | None:
    """Content identity for exactly what one capsule's hardware tier executes.

    The identity covers the bytes of every executable the run produced and the target/hardware
    revisions recorded beside the result. It
    deliberately excludes Merlin's source commit: a source edit that emits byte-identical code has not
    changed the program RTL certifies. Missing ELF, target, or a concrete hardware revision returns
    ``None`` so scheduling falls back to the conservative submission/component digest.

    This is the identity BOTH verdict readers (``qa_check._execution_digest_from_result`` and
    ``agent_selfcheck``) put on every row. It must live here, next to the other content addressing:
    when it went missing the readers still called it, their ``except`` swallowed the ImportError, and
    every row reported ``execution_digest: null`` -- a digest that could not be computed presenting
    exactly like a submission with no artifact identity.
    """
    import json

    import yaml

    from merlin.development.targetgen.oracles.tier_cache import execution_identity
    from merlin.evaluation.capsule.evidence.certificates import run_executables

    cr = Path(capsule_result)
    try:
        result = json.loads(cr.read_text(encoding="utf-8"))
        manifest = yaml.safe_load((cr.parent / "run_manifest.yaml").read_text(encoding="utf-8"))
        # THE HASH ITSELF LIVES IN THE LIBRARY, not here. This function gathers the three inputs off
        # disk; `tier_cache.execution_identity` turns them into the identity. It was a second copy for
        # exactly as long as nothing else needed it -- and the moment the RUNNER had to ask "do these
        # bytes already hold a certificate?", a second copy would have been two answers to one question,
        # differing only in whichever detail drifted first (a pin the recorder skipped and the reader
        # did not, a payload key spelled differently). The consequence of such a drift is not a crash:
        # it is a cache that never hits, or -- far worse -- one that hits for bytes the recorder meant
        # something else by. One implementation, two callers.
        # WHICH executables is also the library's question, not this reader's. Asking for one by name
        # excluded every target whose grade links a differently-named one, or several -- and the
        # exclusion presented as a null digest rather than as an error.
        return execution_identity(
            target=manifest.get("target") if isinstance(manifest, dict) else None,
            executables=run_executables(cr.parent / "generated"),
            toolchain_shas=result.get("toolchain_shas") if isinstance(result, dict) else None,
        )
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
    package to declare exactly these entrypoints (``src/merlin/resources/contracts/schemas/manifest.schema.json``:
    ``commands.required = [parse, lower_interface_to_target, emit_command_buffer,
    lower_target_to_llvm]``, open for a target to add more), and they are the SAME axis the grader drives
    the submission through -- ``oot_runner._resolve_argv`` looks each capsule's compile step up by these
    keys. So "which command's files moved" is exactly "which step of grading this capsule could now answer
    differently", with no per-target code and no vocabulary invented here.

    The two alternatives were rejected on the facts: the Merlin pass registry
    (``xdsl_dialects.lowering.passes.CATALOG``) names Merlin's OWN passes, whose ``entry`` fields are
    ``merlin.*`` dotted paths -- an out-of-tree submission does not contain those modules, so its files
    cannot be attributed to them; and a bundle's ``allowed`` paths are read-only REPO grants
    (``src/merlin/resources/contracts/``, ``third_party/llvm-install/``), not submission files, so they cannot decompose
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
        from merlin.development.targetgen.packaging.loading import _ENTRYPOINT_ALIASES as _AL
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
    honoured, so a typo cannot quietly mint a component that owns no files -- a component with no files
    never changes, and since every capsule depends on every component, every certificate would then live
    forever across edits to the files the typo failed to claim.

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
        s = s[len("submission/") :]
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


INERT = "<inert>"

_DERIVED_CACHE: dict = {}


def derived_components(ws):
    """``({component: frozenset(rel)}, frozenset(unattributed), frozenset(inert))`` or None to refuse.

    Paths are submission-relative. See the block comment above for what each bucket means and for the one
    judgement this makes. ``None`` is "no derivation" and the caller must then use the whole-submission
    digest -- never a partial answer.
    """
    m = _manifest(ws)
    sub = Path(ws, "submission")
    if m is None or not sub.is_dir():
        return None
    if str(m.get("language") or "").strip().lower() != "python":
        # Only Python's import graph is traceable from here. Refusing is the honest answer for a package
        # whose surface is a compiled binary: we cannot see what it opens.
        return None
    roots = _command_roots(sub, m)
    if roots is None:
        return None
    key = (
        str(sub),
        tuple(
            sorted(
                (f.relative_to(sub).as_posix(), f.stat().st_mtime_ns, f.stat().st_size)
                for f in sub.rglob("*")
                if f.is_file() and "__pycache__" not in f.parts
            )
        ),
    )
    hit = _DERIVED_CACHE.get(key)
    if hit is not None:
        return hit
    every = [f for f in sub.rglob("*") if f.is_file() and "__pycache__" not in f.parts]

    # Files the MANIFEST names OUTSIDE the per-command sections are charged to every component -- a build
    # input, a second tool, anything the package points at globally: a change to one of those, or to the
    # manifest itself, can change what every command does. `commands` and `entrypoints` are deliberately
    # excluded, because those ARE the per-command attribution (`_command_roots` resolves them); folding
    # them in here would charge each command's own script to all the others and collapse the whole
    # decomposition back to one bucket.
    _global = {k: v for k, v in m.items() if k not in ("commands", "entrypoints")}
    shared = {sub / "manifest.yaml"} if (sub / "manifest.yaml").is_file() else set()
    for s in _manifest_strings(_global):
        p = _under(sub, s)
        if p is not None:
            shared.update(_files_at(p, s))

    comps, kind_sufs, memo = {}, set(_src_suffixes()), {}
    resolved = {}  # Components share most literals; resolve each once per derivation.
    for name, rs in roots.items():
        got = _closure(sub, rs, memo)
        if got is None:
            return None  # a file in the surface we cannot read: refuse, do not guess
        files, lits, dyn_sufs = got
        for s in lits:  # a data file opened at a fixed, resolvable path
            if s not in resolved:
                p = _under(sub, s)
                resolved[s] = None if p is None else _files_at(p, s)
            if resolved[s] is not None:
                files.update(resolved[s])
        files |= shared
        comps[name] = frozenset(f.relative_to(sub).as_posix() for f in files)
        kind_sufs |= {f.suffix for f in files} | dyn_sufs

    claimed = set().union(*comps.values()) if comps else set()
    unattributed, inert = set(), set()
    from merlin.development.backend.session.inputs.run_record import outside_snapshot

    for f in every:
        rel = f.relative_to(sub).as_posix()
        if rel in claimed:
            continue
        (unattributed if f.suffix in kind_sufs and not outside_snapshot(Path(rel).parts) else inert).add(rel)
    out = (comps, frozenset(unattributed), frozenset(inert))
    # Bounded: a broker lives for the whole run and the agent edits continuously, so an unbounded map
    # keyed by submission state grows one entry per edit for hours. Only the current state is ever a hit.
    if len(_DERIVED_CACHE) > 4:
        _DERIVED_CACHE.clear()
    _DERIVED_CACHE[key] = out
    return out


def decomposition(ws) -> dict:
    """How this submission's bytes decompose, and WHERE that decomposition came from.

    ``{"owners": {rel: {component, ...}}, "names": {component, ...}, "rejected": [...],
      "source": "declared"|"derived"|None, "inert": frozenset(rel)}``

    "Declare, else derive": an agent-declared ``components:`` block wins, and is used even when some of
    its names were rejected -- but a block whose names were ALL rejected leaves nothing to decompose with,
    so it falls through to the derivation rather than to the whole submission. Either way the rejected
    names are carried out for the caller to report; a typo that quietly costs the saving is the failure
    mode this exists to make visible.
    """
    from merlin.development.targetgen.oracles.schedule import UNATTRIBUTED

    sub = Path(ws, "submission")
    rels = (
        [f.relative_to(sub).as_posix() for f in sub.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
        if sub.is_dir()
        else []
    )
    decl = component_paths(ws)
    prefixes, rejected = decl if decl else ({}, [])
    if prefixes:
        # INERT IS NOT UNATTRIBUTED. A declared block says who owns what; it says nothing about which
        # bytes any command can READ. This branch returned an empty inert set, so every file no prefix
        # claimed -- a REPORT.md, an agent note, a scratch listing -- became UNATTRIBUTED. UNATTRIBUTED is
        # a dependency of EVERY capsule (oracle_schedule.py), so one edit to one such file invalidated the
        # whole corpus at once. Measured on a live run: 253 "L2 invalidated by <unattributed>" against 2
        # promotion events, still climbing, with the run's own log reading "0 file(s) outside every
        # command's reach" -- which was len(frozenset()), not a measurement.
        #
        # The reachability closure already answers this, so ask it for that ONE fact. Declared ownership
        # still wins outright: the closure is never allowed to say who owns a file, only whether anything
        # could read it. When it refuses (a non-Python surface, an unreadable file) there is no inert set
        # and the old behaviour stands, which errs toward invalidating rather than toward a stale cert.
        der = derived_components(ws)
        inert = der[2] if der is not None else frozenset()
        owners, fell_through = {}, False
        for rel in rels:
            own = _owner(rel, prefixes)
            if own is not None:
                owners[rel] = {own}
            elif rel not in inert:
                owners[rel] = {UNATTRIBUTED}
                fell_through = True
        return {
            "owners": owners,
            "names": set(prefixes) | ({UNATTRIBUTED} if fell_through else set()),
            "rejected": rejected,
            "source": "declared",
            "inert": inert,
        }
    der = derived_components(ws)
    if der is None:
        return {"owners": {}, "names": set(), "rejected": rejected, "source": None, "inert": frozenset()}
    comps, unattributed, inert = der
    owners = {}
    for name, files in comps.items():
        for rel in files:
            owners.setdefault(rel, set()).add(name)
    for rel in unattributed:
        owners.setdefault(rel, set()).add(UNATTRIBUTED)
    names = set(comps) | ({UNATTRIBUTED} if unattributed else set())
    return {"owners": owners, "names": names, "rejected": rejected, "source": "derived", "inert": inert}
