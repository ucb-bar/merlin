"""Do not re-buy a certificate for bytes that have not changed.

The cert tier is the whole cost of a grade. Measured on one live gemmini round
(``out/runs/gemmini/capsule-bench/merlin_assisted/merlincirct_g4p1_20260905``, ``_qa_work/runs_05``):
the screen tier cost 2.53 min of adapter wall over 82 capsules and the cert tier 44.4 min over 76,
so 94.7% of the grade was the cert tier. A converged submission re-pays that on EVERY post-turn grade,
for capsules whose emitted program is byte-identical to the one certified minutes earlier -- often in
the same turn, by the promotion path, which already records the result against the digest that earned
it. The ledger was being written and nothing read it.

This module is the reader. It is a CACHE, and a wrong hit is worse than the waste it saves, so every
rule below is written in the direction of re-running:

**The key is the bytes plus the instrument.** Two questions have to be the same before a verdict may
be carried -- "is this the same PROGRAM on the same DEVICE?" and "is this the same JUDGE?" The first is
:func:`execution_identity`, which is not a new identity: it is the one
``tier_promote.execution_digest`` already binds a certificate to (the ELF bytes, the target, and the
hardware revisions), and that function now delegates here so the two cannot drift. The second is
:func:`instrument_digest` -- the bytes of the code that decides a verdict, plus, for a tier answered by
elaborated RTL, WHICH ENGINE would answer it today.

What is in the instrument key and what is not:

* IN -- the grading path (:func:`grading_path`): the ladder and finalizer, the grade driver, the tier
  policy, the reuse scheduler, the oracle adapters, the engine-selection policy, the backend registry,
  and the TARGET'S OWN backend package (derived from the backend registry, never named here). These are
  the files whose bytes decide what a tier verdict means; a change to any of them makes a stored verdict
  a description of a different judge. This is the shape ``BASELINE.json``'s ``grading_path_digest``
  records, computed the same way (``merlin.common.provenance.source_digest``, which hashes the bytes
  actually READ, so a dirty tree is a different instrument).
* IN -- the elaborated-RTL engine token, for an RTL tier. A tier is a FIDELITY, not a binary: the same
  L3 is answered by Verilator or GSIM depending on availability, and the two are not the same
  instrument. Cheap to establish (the same selection ``tier_promote.cert_sim`` already resolves), and
  it turns "certified on the engine that is no longer selected" into a miss rather than a hit.
* OUT -- the ELF build path (``contract/compile.py``) and the compiler that produced the lowered MLIR.
  Their effect is entirely inside the executable, which the execution identity already hashes byte for
  byte. Including them would only cost re-runs for edits the identity has already accounted for.
* OUT -- the RTL revision. Also already in the execution identity, which carries every declared
  hardware pin (``toolchain_shas`` minus merlin's own commit).
* OUT -- merlin's git commit. An edit that leaves the graded ELF and the whole grading path
  byte-identical has not changed the program, the device, or the judge. Keying on the commit would
  invalidate every certificate on every unrelated edit, which is the waste this exists to remove.

**Fail closed, everywhere.** No ELF, no target, an ``UNKNOWN`` or malformed hardware pin, an unreadable
grading-path file, an RTL tier whose engine cannot be established, a store that will not parse, a stored
record whose own copy of the key does not match what was asked for -- every one of them returns ``None``
and the tier is re-executed. A cache that cannot miss is the same defect as a check that cannot fail.

**Only a PASS is carried.** A stored ``fail`` is re-run. A failure is what an agent acts on -- its plane,
its category, its detail, its first mismatch -- and a record that carries only the word "fail" would
replace actionable feedback with an assertion. Re-running a failing capsule is also the cheap direction:
the ladder fail-fasts, so a refuted capsule never reaches the expensive tier anyway.

**The first tier the ladder executes is always paid for.** The executable identity is only available
once something in THIS run has built the ELF, and it is the first executed tier's adapter that builds
it -- so that tier is never carried, on any target, without a single rule saying so. Which tier that is
belongs to ``tier_policy.tier_order`` (cheapest-measured-first, with never-yet-measured tiers ahead of
priced ones so the ladder always learns a price); on a calibrated target it is the screen, which is
exactly where this wants it: the cheap tier is re-executed against the bytes on disk every time, and the
expensive one is what gets carried.

Tier semantics are untouched by any of this. A carried L3 is an L3 verdict earned by these exact bytes
on this exact instrument -- never a cheaper tier speaking for a tier it did not run. A screen may
eliminate; it still may never certify, because a carry copies a tier's verdict onto the SAME tier and
nothing else.

**A carried verdict says so.** :func:`carried_block` is attached to the tier record, the tier record
keeps ``measured_now: false``, and the capsule result grows a ``tier_reuse`` block naming which tiers
were executed and which were carried. Wall-clock ``timing`` and ``concurrency`` are DROPPED from a
carried record -- no time was spent now, and a copied duration would be a fabricated measurement.
``cycles`` are kept: a cycle count is a property of the program and the device, both of which the
execution identity pins, and it is concurrency-invariant (verified) where a wall time is not.

Two stores are read, narrowest first:

1. this module's own content-addressed store (default ``out/artifacts/cache/tier-certs/``, one file per
   key, so concurrent capsule workers never lose each other's writes), written by the runner for every
   tier it executes -- including the tiers an async promotion job executes, which is how a promoted
   certificate becomes reusable without any extra bookkeeping;
2. a run's ``qa/tier_state.json`` promotion ledger, when one is named. Its entries carry a status and an
   execution identity but no tier record, so a carry from there reports ``cycles: null`` and names the
   ledger as its source. An entry that does not also record the instrument it was earned under can never
   match -- which is every entry written before this module existed.

Disable with ``MERLIN_TIER_CERT_CACHE=0``; point it elsewhere by setting that variable to a directory.
Name a promotion ledger with ``MERLIN_TIER_CERT_LEDGER`` (os.pathsep-separated paths).
"""
from __future__ import annotations

import hashlib
import json
import os
import datetime as _dt
from pathlib import Path

__all__ = ["EXECUTION_IDENTITY_VERSION", "RECORD_VERSION", "ARTIFACTS_OF_THE_EARNING_RUN",
           "cache_root", "ledger_paths",
           "execution_identity", "grading_path", "instrument_digest", "lookup", "record",
           "carried_block", "reuse_block", "disabled"]

#: Version of the payload :func:`execution_identity` hashes. Bumping it invalidates every stored record,
#: which is the correct effect of changing what "the same program on the same device" means.
EXECUTION_IDENTITY_VERSION = 1

#: Version of the on-disk record shape. A record written by another version is not read.
RECORD_VERSION = 1

#: Environment knob: a directory to use as the store, or an off switch.
_CACHE_ENV = "MERLIN_TIER_CERT_CACHE"
#: Environment knob: promotion ledgers (``qa/tier_state.json``) to consult, os.pathsep-separated.
_LEDGER_ENV = "MERLIN_TIER_CERT_LEDGER"

#: Spellings of "off". A value that is neither one of these nor a usable directory is a PATH.
_OFF = frozenset({"0", "off", "no", "false", "none", "disabled"})

#: The one status that may be carried. See the module docstring: a stored failure is re-run.
_CARRYABLE = "pass"

#: Keys dropped from a carried tier record because they describe an act of measurement that did not
#: happen now. Everything else is a property of the program and the device, which the key pins.
_NOT_MEASURED_NOW = ("timing", "concurrency")

#: Keys on a stored tier record that name FILES IN THE RUN THAT EARNED IT. They are kept in the store
#: (they are how a reader finds the console a certificate was read off) but must not be presented as
#: artifacts of the run that carries it: a path that resolves to nothing in this run directory reads as
#: a missing artifact rather than as an artifact that lives somewhere else. The runner moves them into
#: the ``carried`` block instead.
ARTIFACTS_OF_THE_EARNING_RUN = ("evidence", "console_log", "console_bytes")


def disabled() -> bool:
    """Whether the operator has switched the cache off."""
    return (os.environ.get(_CACHE_ENV, "") or "").strip().lower() in _OFF


def cache_root() -> "Path | None":
    """The store directory, or ``None`` when the cache is off or its root cannot be created."""
    if disabled():
        return None
    raw = (os.environ.get(_CACHE_ENV, "") or "").strip()
    try:
        if raw:
            d = Path(raw)
            d.mkdir(parents=True, exist_ok=True)
            return d
        from merlin.common.artifacts import cache_dir
        return cache_dir("tier-certs")
    except OSError:                      # an unwritable store is a cache that cannot be used, not a fault
        return None


def ledger_paths() -> tuple:
    """Promotion ledgers (``qa/tier_state.json``) named by the environment, as existing files."""
    raw = (os.environ.get(_LEDGER_ENV, "") or "").strip()
    if not raw or disabled():
        return ()
    out = []
    for token in raw.split(os.pathsep):
        token = token.strip()
        if not token:
            continue
        p = Path(token)
        if p.is_file():
            out.append(p)
    return tuple(out)


# ---------------------------------------------------------------------------------------------
# WHICH BYTES ON WHICH DEVICE
# ---------------------------------------------------------------------------------------------

def _valid_pin(name, value) -> bool:
    """Whether one ``toolchain_shas`` entry identifies a revision precisely enough to key on.

    Hardware pins are full git/content hashes. ``UNKNOWN``, an abbreviated sha, or anything else is a
    revision nobody could establish -- and a certificate must never be attributed to a device whose
    identity was a guess.
    """
    return (isinstance(name, str) and bool(name)
            and isinstance(value, str) and len(value) in (40, 64)
            and all(c in "0123456789abcdef" for c in value))


def execution_identity(*, target, executable, toolchain_shas) -> "str | None":
    """Content identity for exactly what one capsule's hardware tier executes, or ``None``.

    ``{ELF bytes} x {target} x {every declared hardware pin}``. Merlin's own commit is deliberately
    excluded: a source edit that emits byte-identical code has not changed the program the RTL
    certified. This is the SINGLE implementation of that identity --
    ``tier_promote.execution_digest`` gathers the same three inputs off disk and calls it -- so the
    recorder that binds a certificate to bytes and the reader that spends one cannot disagree about
    what "the same bytes" means.
    """
    try:
        elf = Path(executable)
        if not isinstance(target, str) or not target.strip() or not isinstance(toolchain_shas, dict):
            return None
        hardware = {}
        for key, value in toolchain_shas.items():
            if isinstance(key, str) and key.lower() == "merlin":
                continue
            if not _valid_pin(key, value):
                return None
            hardware[key] = value
        if not hardware or not elf.is_file():
            return None
        payload = {
            "version": EXECUTION_IDENTITY_VERSION,
            "target": target,
            "hardware": hardware,
            "executable_sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
        }
        return _digest_of(payload)
    except OSError:                      # unreadable artifact: no identity, so the tier re-runs
        return None


def _digest_of(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# ---------------------------------------------------------------------------------------------
# WHICH JUDGE
# ---------------------------------------------------------------------------------------------

#: Repo-relative source files whose bytes decide what a tier verdict MEANS. Not a list of facts about
#: any target -- these are merlin's own grading modules, and the target's contribution is derived from
#: the backend registry in :func:`grading_path`. See the module docstring for what is deliberately
#: absent (the ELF build path and the RTL revision, both already inside the execution identity).
_GRADING_MODULES = (
    "merlin/python/merlin/targetgen/capsule_runner.py",
    "merlin/python/merlin/targetgen/capsule_grade.py",
    "merlin/python/merlin/targetgen/tier_policy.py",
    "merlin/python/merlin/targetgen/oracle_schedule.py",
    "merlin/python/merlin/targetgen/heavy_oracles.py",
    "merlin/python/merlin/targetgen/program_oracle.py",
    "merlin/python/merlin/targetgen/rtl_engine_policy.py",
    "merlin/python/merlin/runtime/backends/base.py",
    "merlin/python/merlin/targetgen/tier_cache.py",
)


def grading_path(target: "str | None" = None) -> "tuple[Path, ...] | None":
    """Every file whose bytes decide a verdict, or ``None`` when one of them cannot be located.

    ``None`` is not "no instrument" -- it is "the instrument cannot be established", and the caller must
    then re-run. A grading path silently missing a member would key certificates on a partial judge.

    The target's own backend contributes its whole package directory, DERIVED from the backend registry
    (``merlin.runtime.backends.base.get_backend``) rather than named here: the backend owns ``run_elf``
    and ``parse_output``, so its bytes decide what a console means, and a repo that adds a target must
    not have to edit this module.
    """
    from merlin.common.paths import repo_root
    root = Path(repo_root())
    files = []
    for rel in _GRADING_MODULES:
        p = root / rel
        if not p.is_file():
            return None
        files.append(p)
    if target:
        try:
            from merlin.runtime.backends import base as _backends
            mod = _backends.get_backend(str(target))
            home = Path(getattr(mod, "__file__", "") or "").parent
        except Exception:                # noqa: BLE001 -- no resolvable backend: no instrument identity
            return None
        if not home.is_dir():
            return None
        files.extend(sorted(p for p in home.rglob("*.py") if "__pycache__" not in p.parts))
    return tuple(sorted(set(files)))


#: Memoized per (target, tier). The engine probe asks a backend whether a simulator is runnable, which
#: is cheap but not free, and a grade asks this question once per capsule per tier.
_INSTRUMENT_MEMO: dict = {}


def _engine_token(target: "str | None", tier: str, rtl_tier: bool) -> "str | None":
    """The elaborated-RTL engine that would answer ``tier`` today, ``""`` for a non-RTL tier, or ``None``.

    A tier index names a FIDELITY, and two engines answering at that fidelity are two instruments. When
    the tier is answered by elaborated RTL and the engine cannot be established, there is no instrument
    identity and nothing may be carried.
    """
    if not rtl_tier:
        return ""
    if not target:
        return None
    try:
        from .capsule_runner import describe_l3_engine
        sel = describe_l3_engine(str(target))
    except Exception:                    # noqa: BLE001 -- unresolvable selection: fail closed
        return None
    if not isinstance(sel, dict) or not sel.get("available"):
        return None
    engine = str(sel.get("engine") or "").strip()
    return engine or None


def instrument_digest(target: "str | None", tier: str, *, rtl_tier: bool) -> "str | None":
    """Identity of the JUDGE for one (target, tier): the grading path's bytes plus the engine.

    ``None`` whenever any input is missing, which makes the tier re-run. Uses
    ``merlin.common.provenance.source_digest``, so the digest covers the bytes actually READ -- a dirty
    working tree is a different instrument, and a certificate earned under it does not silently carry
    into a clean one.
    """
    key = (str(target or ""), str(tier), bool(rtl_tier))
    if key in _INSTRUMENT_MEMO:
        return _INSTRUMENT_MEMO[key]
    value = None
    files = grading_path(target)
    engine = _engine_token(target, tier, rtl_tier)
    if files and engine is not None:
        from merlin.common.provenance import source_digest
        value = _digest_of({
            "version": RECORD_VERSION,
            "grading_path": source_digest([str(p) for p in files]),
            "engine": engine,
            "tier": str(tier),
        })
    _INSTRUMENT_MEMO[key] = value
    return value


# ---------------------------------------------------------------------------------------------
# THE STORE
# ---------------------------------------------------------------------------------------------

def _key_fields(capsule: str, tier: str, identity: str, instrument: str) -> dict:
    return {"version": RECORD_VERSION, "capsule": str(capsule), "tier": str(tier),
            "execution_identity": str(identity), "instrument": str(instrument)}


def _record_path(root: Path, capsule: str, tier: str, identity: str, instrument: str) -> Path:
    """One file per key. A directory of independent files, not one shared document: capsule workers run
    in parallel and two brokers write concurrently, and a read-modify-write of a shared file loses
    records -- which costs a re-run at best and, on a truncated read, wipes every recorded verdict."""
    return root / (_digest_of(_key_fields(capsule, tier, identity, instrument)) + ".json")


def lookup(capsule: str, tier: str, identity, instrument, *,
           root: "Path | None" = None, ledgers=()) -> "dict | None":
    """The record that certifies EXACTLY these bytes on EXACTLY this instrument, or ``None``.

    Everything is re-verified against the stored record's own copy of the key, so a record reached by a
    path that is somehow wrong -- a hand-edited store, a hash collision, a half-written file -- is a MISS
    rather than a hit for the wrong capsule. Only ``pass`` is returned; see the module docstring.
    """
    from .oracle_schedule import valid_execution_digest
    if not valid_execution_digest(identity) or not valid_execution_digest(instrument):
        return None
    want = _key_fields(capsule, tier, identity, instrument)
    root = root if root is not None else cache_root()
    if root is not None:
        try:
            raw = json.loads((_record_path(root, capsule, tier, identity, instrument)
                              ).read_text(encoding="utf-8"))
        except (OSError, ValueError):    # absent, unreadable or corrupt: re-run
            raw = None
        if isinstance(raw, dict) and all(raw.get(k) == v for k, v in want.items()):
            if raw.get("status") == _CARRYABLE and isinstance(raw.get("tier_result"), dict):
                return dict(raw)
    for led in (tuple(ledgers) or ledger_paths()):
        hit = _from_ledger(led, capsule, tier, identity, instrument)
        if hit is not None:
            return hit
    return None


def _from_ledger(path, capsule: str, tier: str, identity: str, instrument: str) -> "dict | None":
    """A promotion ledger entry (``qa/tier_state.json``) that certifies these bytes on this instrument.

    The ledger is ``tier_promote``'s: ``{capsule: {"<certs>": {tier: {identity: entry}}}}``, where an
    entry states a status, the digests it belongs to, and -- for an entry written since this module
    existed -- the instrument it was earned under. An entry with no recorded instrument can never match,
    because the judge it was earned under is unknown and an unknown judge is not this one.

    A ledger entry carries no tier record, so what comes back states the verdict and nothing else. The
    caller reports ``cycles: null`` rather than inventing one.
    """
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):        # absent or corrupt ledger: re-run
        return None
    if not isinstance(state, dict):
        return None
    per = state.get(str(capsule))
    if not isinstance(per, dict):
        return None
    from .oracle_schedule import CERT_LEDGER
    led = per.get(CERT_LEDGER)
    slots = led.get(str(tier)) if isinstance(led, dict) else None
    entry = slots.get(str(identity)) if isinstance(slots, dict) else None
    if not isinstance(entry, dict):
        return None
    if entry.get("status") != _CARRYABLE or entry.get("execution_digest") != identity:
        return None
    if entry.get("instrument") != instrument:
        return None
    out = _key_fields(capsule, tier, identity, instrument)
    out.update({"status": _CARRYABLE, "tier_result": {}, "source": "promotion ledger",
                "source_path": str(path), "recorded_at": entry.get("recorded_at")})
    return out


def record(capsule: str, tier: str, identity, instrument, *, status: str, tier_result: dict,
           root: "Path | None" = None, run_id: "str | None" = None) -> "Path | None":
    """Store one EXECUTED tier verdict against the bytes and the instrument that produced it.

    Returns the file written, or ``None`` when nothing was stored (cache off, no identity, no
    instrument, a status that may not be carried, or an unwritable store). Never raises: a cache that
    can fail a grade is worse than no cache.
    """
    from .oracle_schedule import valid_execution_digest
    if status != _CARRYABLE or not isinstance(tier_result, dict):
        return None
    if not valid_execution_digest(identity) or not valid_execution_digest(instrument):
        return None
    root = root if root is not None else cache_root()
    if root is None:
        return None
    payload = _key_fields(capsule, tier, identity, instrument)
    payload.update({
        "status": status,
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_id": run_id,
        "tier_result": {k: v for k, v in tier_result.items() if k not in _NOT_MEASURED_NOW},
    })
    dest = _record_path(root, capsule, tier, identity, instrument)
    try:
        root.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, dest)             # atomic: a concurrent reader sees old or new, never partial
    except OSError:
        return None
    return dest


# ---------------------------------------------------------------------------------------------
# SAYING SO
# ---------------------------------------------------------------------------------------------

def carried_block(hit: dict) -> dict:
    """The provenance a carried tier record must carry, so nobody reads it as freshly measured."""
    return {
        "carried": True,
        "execution_identity": hit.get("execution_identity"),
        "instrument": hit.get("instrument"),
        "earned_at": hit.get("recorded_at"),
        "earned_by_run": hit.get("run_id"),
        "source": hit.get("source") or "tier certificate cache",
        "note": ("this tier was NOT executed in this run: the same program was certified at this tier "
                 "on this instrument, and that verdict is carried. cycles are a property of the "
                 "program and the device and are carried with it; wall-clock timing is not carried, "
                 "because no time was spent measuring now"),
    }


def reuse_block(tiers) -> dict:
    """``{"executed": [...], "carried": [...]}`` for one capsule's tier records.

    Emitted on every capsule result, not only when something was carried: a run in which nothing was
    reused and a run in which the accounting was never computed have to look different, or a cached
    grade and a fresh one read the same.
    """
    executed, carried = [], []
    for tier in sorted(tiers or {}):
        rec = tiers[tier]
        block = rec.get("carried") if isinstance(rec, dict) else getattr(rec, "carried", None)
        (carried if isinstance(block, dict) and block.get("carried") else executed).append(tier)
    return {"executed": executed, "carried": carried,
            "note": ("carried tiers were not executed in this run; their verdict was earned earlier by "
                     "the same executable on the same instrument (merlin.targetgen.tier_cache)")}
