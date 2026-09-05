"""Materialize a sandbox view of the capsule corpus from the single source of truth.

The capsule-bench sandbox needs the *public* capsules (``label: public``) with their oracle-tier
requirement capped to what the sandbox can actually reach — the bwrap sandbox has no VCS/FireSim, so
tiers above L2 are unreachable. Everything else is byte-identical to ``merlin/contract/capsules/``.

Deriving this view (instead of hand-maintaining a committed copy) keeps ONE source of truth: the
committed sandbox copy at ``…/scripts/full_public_capsules/`` drifted from the contract (it had
``required_oracle_tiers`` reindented and L3 dropped by hand). ``materialize_public_capsules`` is that
transform, made explicit and testable.

Usage (as a library):
    from merlin.targetgen.contract.materialize import materialize_public_capsules
    materialize_public_capsules(dest, tier_ceiling="L2")

CLI:
    python -m merlin.targetgen.contract.materialize <dest_dir> [--tier-ceiling L2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import yaml

from .schemas import contract_dir

# The complete capsule bundle (see generate_corpus.py / AGENT.md).  The PyTorch/linalg sources are
# agent-visible grounding. A whole-model capsule's externalized weights are an operator-only compile/
# runtime input named by the interface: materialization must preserve them exactly, while the sandbox's
# answer-surface policy masks them from the agent. They are optional because op capsules do not carry them;
# when present, omitting them creates a valid-looking grading corpus that cannot reproduce the model.
_CAPSULE_FILES = (
    "capsule.yaml", "capsule.interface.mlir", "capsule.pytorch.py", "capsule.linalg.mlir",
    "capsule.weights.safetensors", "golden.yaml", "expected_instruction_coverage.yaml", "README.md",
)
_TIER_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5"]
_DEFAULT_CEILING = "L2"  # bwrap sandbox: numerics + spike, no VCS/FireSim (L3+).


def _name_set_sha256(names) -> str:
    return hashlib.sha256(
        json.dumps(sorted(str(x) for x in names), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _loaded_descriptor_sha256(te) -> str:
    """Return the descriptor identity parsed into ``te`` and reject post-load byte drift."""
    path = Path(te.path)
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = getattr(te, "descriptor_sha256", None)
    if loaded is not None and loaded != current:
        raise ValueError(
            f"target experiment descriptor changed after it was loaded: {path}; refusing to bind "
            "a cohort derived from stale parsed fields to different descriptor bytes")
    return str(loaded or current)


def validate_materialized_cohort(root: str | Path, te) -> dict:
    """Validate an immutable public cohort against its exact, currently loaded descriptor.

    The cohort record is not a self-attestation: this recomputes every value available from the
    descriptor and materialized directory.  In particular it compares the record's descriptor digest
    to the bytes parsed into ``te`` and to the descriptor bytes still on disk, so changing the descriptor
    after materialization fails closed instead of leaving a plausible 64-character but stale digest.
    """
    cohort_root = Path(root).resolve(strict=True)
    record_path = cohort_root / ".cohort_admission.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or malformed cohort admission record: {record_path}") from exc
    if not isinstance(record, dict) or record.get("version") != 1:
        raise ValueError(f"unsupported cohort admission record: {record_path}")

    descriptor_sha = _loaded_descriptor_sha256(te)
    if record.get("descriptor_sha256") != descriptor_sha:
        raise ValueError("materialized cohort descriptor digest does not match the current descriptor")

    admitted = sorted(p.name for p in cohort_root.iterdir() if p.is_dir())
    capability = sorted(getattr(te, "graded_capability_exclude", ()) or ())
    resource = sorted(getattr(te, "graded_resource_exclude", ()) or ())
    excluded = sorted(getattr(te, "graded_exclude", ()) or ())
    explicit = bool(capability or resource)
    expected_policy = ("descriptor_capability_and_resource_v1" if explicit else
                       "descriptor_exclusions_legacy" if excluded else "all_discovered")
    expected_capability_n = len(capability)
    expected_resource_n = len(resource) if explicit else len(excluded)
    expected = {
        "policy": expected_policy,
        "n_admitted_capsules": len(admitted),
        "n_capability_excluded": expected_capability_n,
        "n_resource_excluded": expected_resource_n,
        "excluded_name_set_sha256": _name_set_sha256(excluded),
        "admitted_name_set_sha256": _name_set_sha256(admitted),
        "resource_policy": getattr(te, "graded_resource_policy", None),
        "required_admitted_models": sorted(getattr(te, "graded_required_models", ()) or ()),
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                f"materialized cohort {field} does not match its descriptor/content: "
                f"record={record.get(field)!r}, expected={value!r}")
    if record.get("n_source_capsules") != len(admitted) + len(excluded):
        raise ValueError("materialized cohort source count does not match admitted plus excluded")

    expected_source = getattr(te, "graded_expected_source_capsules", None)
    expected_admitted = getattr(te, "graded_expected_admitted_capsules", None)
    if expected_source is not None and record.get("n_source_capsules") != expected_source:
        raise ValueError("materialized cohort source count drifted from the descriptor expectation")
    if expected_admitted is not None and len(admitted) != expected_admitted:
        raise ValueError("materialized cohort admitted count drifted from the descriptor expectation")
    if not set(expected["required_admitted_models"]).issubset(admitted):
        raise ValueError("materialized cohort is missing a descriptor-required model capstone")
    return record


def _public_capsule_dirs(contract: str | Path | None = None) -> list[Path]:
    """Every capsule dir whose capsule.yaml is label: public, across isa/layers/model_slices.

    This materializes THIS (gemmini) contract's public capsules — its capsules live at
    ``capsules/<category>/<cap>/capsule.yaml`` (rel-depth 3). Another target's corpus that nests under the
    same root (e.g. ``capsules/atlas/<category>/<cap>/capsule.yaml``, rel-depth 4) must NOT be materialized
    into the gemmini sandbox, so restrict to the direct category/capsule depth."""
    root = contract_dir(contract) / "capsules"
    out = []
    for cap_yaml in sorted(root.rglob("capsule.yaml")):
        if len(cap_yaml.relative_to(root).parts) != 3:   # skip nested per-target corpora (atlas/, …)
            continue
        try:
            cap = yaml.safe_load(cap_yaml.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if cap.get("label") == "public":
            out.append(cap_yaml.parent)
    return out


def _public_capsule_dirs_in(corpus_roots: list[Path]) -> list[Path]:
    """Every ``label: public`` capsule dir found DIRECTLY under the given corpus roots (``<root>/<cap>/
    capsule.yaml``). Target-AGNOSTIC: the caller passes the descriptor's own ``capsule_corpus`` + sibling
    corpora, so any target's public set is materialized without a per-target root/depth assumption."""
    out: list[Path] = []
    for root in corpus_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for cap_yaml in sorted(root.glob("*/capsule.yaml")):
            try:
                cap = yaml.safe_load(cap_yaml.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if cap.get("label") == "public":
                out.append(cap_yaml.parent)
    return out


def _cap_tiers(tier_ceiling: str) -> list[str]:
    if tier_ceiling not in _TIER_ORDER:
        raise ValueError(f"unknown tier ceiling {tier_ceiling!r} (expected one of {_TIER_ORDER})")
    keep = _TIER_ORDER[: _TIER_ORDER.index(tier_ceiling) + 1]
    return keep


# The RTL-derived / numeric oracle tiers. L0/L1 are the integer reference/simulate floor (marked
# not_applicable on a float datapath); a real numeric grade always rides one of these.
_RTL_TIERS = frozenset({"L2", "L3", "L4", "L5"})


def _cap_required(tiers: list[str], keep: set[str]) -> tuple[list[str], list[str]]:
    """Cap a capsule's ``required_oracle_tiers`` to what THIS phase can reach. Returns
    ``(kept, unreachable)`` — a PURE intersection plus the declared tiers this phase cannot reach.

    NEVER substitutes. An earlier revision, when capping removed every RTL/numeric tier a capsule
    required, appended the phase ceiling instead — so a capsule that declared the cycle-accurate cert
    tier was silently graded against a DIFFERENT, cheaper tier it had never asked for, and the result
    was reported as if it were the declared one. Measured cost: an entire agent run where the declared
    tier ran fine and returned real verdicts while the substituted tier's endpoint hung, and every
    capsule failed on a gate the capsule never named.

    The declared tier is the capsule's contract. When a phase cannot reach it the honest outcome is to
    say so (the caller fails closed on the empty/floor-only result and names the unreachable tier), not
    to quietly grade something else. Choosing a phase ceiling that the corpus actually declares is the
    caller's job — see :func:`declared_oracle_tiers` and ``qa_loop_adapters(declared_tiers=...)``."""
    kept = [t for t in tiers if t in keep]
    unreachable = [t for t in tiers if t not in keep]
    return kept, unreachable


def declared_oracle_tiers(*roots: str | Path) -> set[str]:
    """The union of every ``required_oracle_tiers`` entry declared by the capsules under ``roots``.

    The CORPUS is the authority on which oracle tiers a grade must ride. Deriving a phase's tier from
    this (rather than from "whichever tier the endpoint reaches fastest") is what keeps grading on the
    tier the capsule actually declared. Target-agnostic by construction: it reads capsule yaml, never a
    target name."""
    out: set[str] = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for cap_yaml in sorted(root.rglob("capsule.yaml")):
            try:
                cap = yaml.safe_load(cap_yaml.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            tiers = cap.get("required_oracle_tiers")
            if isinstance(tiers, list):
                out |= {str(t) for t in tiers}
    return out


def materialize_public_capsules(dest: str | Path, *, tier_ceiling: str = _DEFAULT_CEILING,
                                contract: str | Path | None = None,
                                corpus_roots: list[Path] | None = None,
                                exclude: tuple[str, ...] | set[str] | None = None) -> list[str]:
    """Derive the sandbox public-capsule view into ``dest``. Returns the capsule names written.

    Copies the capsule bundle verbatim (including optional PyTorch/linalg sources and whole-model
    weights), then rewrites ``capsule.yaml``'s ``required_oracle_tiers`` to the subset reachable
    at/below ``tier_ceiling`` (preserving every other field exactly).

    ``corpus_roots`` (target-AGNOSTIC): materialize the public capsules found directly under these roots
    (the descriptor's ``capsule_corpus`` + sibling corpora). When omitted, falls back to the legacy
    gemmini-contract discovery (``contract``) for backward compatibility.

    ``exclude`` is the descriptor's ``grading.exclude_capsules`` — capsule DIRECTORY NAMES this
    experiment withholds from the public graded set (see
    :attr:`~merlin.targetgen.target_experiment.TargetExperiment.graded_exclude` for why one exists). The
    names are matched against what the corpus actually holds and an exclusion matching NOTHING raises:
    a mistyped or stale name would otherwise silently re-admit an expensive capsule, and the failure
    mode of a *quietly wider* set is a run that blows its wall-clock budget with no signal saying why.
    """
    keep = set(_cap_tiers(tier_ceiling))
    dest = Path(dest)
    written: list[str] = []
    sources = (_public_capsule_dirs_in(corpus_roots) if corpus_roots is not None
               else _public_capsule_dirs(contract))
    source_names = [source.name for source in sources]
    if len(source_names) != len(set(source_names)):
        duplicates = sorted(name for name in set(source_names) if source_names.count(name) > 1)
        raise ValueError(f"public corpus roots contain duplicate capsule names: {duplicates}")
    drop = set(exclude or ())
    if drop:
        present = {s.name for s in sources}
        unknown = sorted(drop - present)
        if unknown:
            raise ValueError(
                f"grading.exclude_capsules names {unknown} but the public corpus holds no such "
                f"capsule(s); it has {sorted(present)}. Refusing to materialize, because an exclusion "
                f"that matches nothing silently GROWS the graded set back to full size.")
        sources = [s for s in sources if s.name not in drop]
    for src in sources:
        name = src.name
        d = dest / name
        d.mkdir(parents=True, exist_ok=True)
        for f in _CAPSULE_FILES:
            sp = src / f
            if not sp.is_file():
                continue
            if f == "capsule.yaml":
                cap = yaml.safe_load(sp.read_text(encoding="utf-8")) or {}
                tiers = cap.get("required_oracle_tiers")
                if isinstance(tiers, list):
                    kept, unreachable = _cap_required(tiers, keep)
                    cap["required_oracle_tiers"] = kept
                    if unreachable and not any(t in _RTL_TIERS for t in kept):
                        # Capping left NO numeric/RTL tier: this phase cannot certify the capsule at all.
                        # Carry the dropped DECLARED tiers forward so the grader can fail closed and NAME
                        # them, instead of reporting a bare "no oracle" — or, as it once did, silently
                        # substituting the ceiling tier and grading against something never declared.
                        # Only annotated in that case, so a capsule whose numeric floor survives capping
                        # (the ordinary case) stays byte-identical to the source.
                        cap["unreachable_required_oracle_tiers"] = unreachable
                        cap["oracle_tier_ceiling"] = tier_ceiling
                (d / f).write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
            else:
                shutil.copyfile(sp, d / f)
        written.append(name)
    return sorted(written)


#: A build dir is only collectable once it is at least this old AND provably unheld. Age is a
#: heuristic about *staleness*; it is NOT a liveness proof, and treating it as one is what produced a
#: 1/32 official verdict for a submission that passes 33/34 (see the GC in ``public_capsules_for``).
_GC_MIN_AGE_S = 900

_BUILD_INFIX = ".build."
_LEASE_PREFIX = ".lease."


def _build_owner_pid(name: str) -> int | None:
    """The PID that built a ``.<target>.build.<pid>.<hex>`` staging dir, or None if ``name`` is not one.

    Parsed STRUCTURALLY (``str.partition``), never by regex: a too-narrow pattern here fails open --
    it would report "not a staging dir" for a real one and let the GC delete it.
    """
    _, sep, tail = name.partition(_BUILD_INFIX)
    if not sep:
        return None
    pid_text, _, rest = tail.partition(".")
    if not rest:                      # a build dir always carries the uuid suffix too
        return None
    try:
        return int(pid_text)
    except ValueError:
        return None


def _lease_owner_pid(name: str) -> int | None:
    """The PID holding a ``.lease.<pid>.<hex>`` reader lease, or None if unparseable."""
    if not name.startswith(_LEASE_PREFIX):
        return None
    tail = name[len(_LEASE_PREFIX):]
    pid_text, _, rest = tail.partition(".")
    if not rest:
        return None
    try:
        return int(pid_text)
    except ValueError:
        return None


def _pid_alive(pid: int | None) -> bool:
    """Whether ``pid`` still exists. FAIL CLOSED: anything we cannot determine counts as ALIVE, so an
    unreadable/odd entry is never the reason a live reader's inputs get deleted."""
    if pid is None or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                   # exists, owned by another user
    except OSError:
        return True                   # cannot tell -> treat as held
    return True


def _staging_dir_of(path: str | Path) -> Path | None:
    """The ``.<target>.build.<pid>.<hex>`` staging dir a resolved path lies inside, or None.

    Checked on the path itself and each ancestor, because the cert-tier grade is handed a LIST of
    per-capsule directories under the cohort (``<build>/<CAPSULE>``) rather than the cohort root, and a
    self-only check silently pinned nothing for exactly the grade that holds its paths longest.
    """
    real = Path(path).resolve()
    for cand in (real, *real.parents):
        if _build_owner_pid(cand.name) is not None:
            return cand
    return None


def pin_cohort_builds(*roots: str | Path) -> list[Path]:
    """Pin the materialized cohort build(s) behind ``roots`` for this process, returning the leases taken
    (empty when none of them is a staged build -- e.g. a plain committed corpus dir, which nothing GCs).

    WHY a reader needs this. ``public_capsules_for`` returns the per-target SYMLINK, but a grader resolves
    it to the concrete build dir ONCE and then reads capsules out of that absolute path for as long as the
    grade runs (``capsule_grade.grade`` does exactly that, so the root cannot move under a sibling
    thread's chdir). The functional grade of ~32 capsules already exceeds the GC's 15-minute cutoff; the
    cert tier is ~30-40 min PER capsule. Meanwhile any concurrent self-check or debug-broker request
    re-materializes the same target, republishes the symlink and collects old builds. Without a lease the
    collector cannot tell "old" from "old and still being read", and it guessed wrong.

    A lease is a SYMLINK named for the holding PID and pointing at the build dir it protects: atomic to
    create, self-describing, and a dead holder is detectable with no bookkeeping. Taking new leases DROPS
    this process's previous ones, so a long-lived loop process holds only what it is currently reading and
    cannot pin every generation it ever graded. Leases are advisory -- losing one costs a stale directory
    its early collection, never a live reader its inputs.
    """
    release_cohort_builds()
    taken: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            staging = _staging_dir_of(root)
        except OSError:
            continue
        if staging is None or staging in seen:
            continue
        seen.add(staging)
        lease = staging.parent / f"{_LEASE_PREFIX}{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            os.symlink(staging.name, lease)
        except OSError:
            continue          # unwritable cache -> no lease; the GC still honours builder liveness
        taken.append(lease)
    return taken


def release_cohort_builds() -> None:
    """Drop every lease this process holds. Safe to call repeatedly."""
    from merlin.common.artifacts import cache_dir

    try:
        base = cache_dir("capsule_bench_public")
    except OSError:
        return
    for lease in base.glob(f"{_LEASE_PREFIX}{os.getpid()}.*"):
        try:
            lease.unlink()
        except OSError:
            pass


def _held_build_names(base: Path) -> set[str]:
    """Build-dir names a LIVE reader still holds a lease on, clearing leases whose holder is gone."""
    held: set[str] = set()
    for lease in base.glob(f"{_LEASE_PREFIX}*"):
        try:
            target_name = os.readlink(lease)
        except OSError:
            continue
        if _pid_alive(_lease_owner_pid(lease.name)):
            held.add(Path(target_name).name)
            continue
        try:
            lease.unlink()
        except OSError:
            pass
    return held


def public_capsules_for(te, *, tier_ceiling: str | None = None) -> Path:
    """The public-capsule set to grade / self-check against for a target — DERIVED from its descriptor's
    ``capsule_corpus`` (+ sibling corpora), materialized into a per-target cache and returned. This is the
    target-agnostic replacement for the legacy committed ``scripts/full_public_capsules`` smoke fixture:
    gemmini gets its descriptor-declared cohort, atlas gets its own fp8/bf16 set, any target its own —
    with no per-target hardcode and no gemmini leak.

    ``tier_ceiling`` caps ``required_oracle_tiers`` to what the caller can reach. The default is the
    per-round loop tier the corpus itself DECLARES — ``qa_loop_adapters`` is asked for the fastest
    endpoint tier that is also a declared required tier, so the loop always grades against a tier the
    capsules asked for. When the endpoint reaches NONE of the declared tiers this FAILS CLOSED with the
    declared-vs-reachable sets named, rather than capping onto a substitute tier."""
    from merlin.common.artifacts import cache_dir
    from merlin.common.paths import repo_root
    from merlin.targetgen import capsule_runner as _CR
    root = repo_root()
    roots = ([te.capsule_corpus] if te.capsule_corpus else [])
    roots += [root / rel.rstrip("/") for rel in te.corpus_siblings()]
    if tier_ceiling is None:
        declared = declared_oracle_tiers(*roots)
        loop = _CR.qa_loop_adapters(te.target, te.sim_via, declared_tiers=declared)
        if not loop:
            reach = sorted(_CR.oracle_adapters(te.target, te.sim_via))
            if reach:
                # The endpoint DOES expose tiers, just none the corpus declared. Substituting one of them
                # is precisely the defect; refuse and name both sets.
                raise ValueError(
                    f"target {te.target!r}: its capsule corpus declares required oracle tiers "
                    f"{sorted(declared)} but the endpoint reaches {reach} — no declared tier is "
                    f"reachable, so this phase cannot grade these capsules. Refusing to substitute a "
                    f"tier the capsules never declared; make the declared tier reachable or fix the "
                    f"corpus.")
            # The endpoint reaches NOTHING (no mlc/sim/model venv in this environment). That is an
            # honestly ABSENT oracle, not a substitution risk: materialize at the legacy floor and let
            # the runner report each capsule's missing tier as unavailable, as it always has.
            tier_ceiling = _DEFAULT_CEILING
        else:
            # Cap to the highest DECLARED tier this endpoint can REACH -- not to the cheap per-round
            # screen tier. Capping at the screen silently rewrote the capsule's contract: gemmini's
            # capsules declare L3 (verilator), the loop tier is L2 (spike), so every materialized capsule
            # came out requiring only L2. Two things followed, both invisible. A capsule that passed
            # spike was recorded `pass` with the RTL tier never run, and
            # `_cycle_accurate_checkpoint_enabled` -- which asks whether any cert tier is MANDATORY in
            # this corpus -- found none and skipped the verilator barrier outright. The whole run graded
            # on the functional model with the elaborated RTL never executed once.
            #
            # The ceiling exists to avoid REQUIRING a tier the endpoint cannot reach. That is what it
            # now does, and no more; how much of the reachable ladder a given PHASE buys is the phase's
            # decision (fail-fast + covering set + certify budget), not something to bake into the
            # corpus by weakening what the capsules ask for.
            reach = set(_CR.oracle_adapters(te.target, te.sim_via))
            usable = (declared & reach) or set(loop)
            tier_ceiling = max(usable, key=lambda t: _TIER_ORDER.index(t) if t in _TIER_ORDER else -1)
    # Concurrency-safe publish. Several A/B arms materialize the SAME target's public set at once; the old
    # ``rmtree(dest) + rebuild`` in place let one arm delete another's half-built cache mid-read (corrupt or
    # missing capsules -> wrong grades). Instead build into a UNIQUE versioned dir, then ATOMICALLY repoint a
    # per-target symlink at it (os.replace is atomic even over an existing symlink). A reader always follows
    # the symlink to a COMPLETE, immutable build; the last writer wins the symlink; nobody rmtrees a dir
    # another arm is reading. The cache namespace is purgeable, so stale builds are best-effort GC'd once
    # they are old AND provably unheld -- see the GC at the end of this function for why "old" alone was
    # not enough to make that last clause true.
    base = cache_dir("capsule_bench_public")
    base.mkdir(parents=True, exist_ok=True)
    link = base / te.target
    ver = base / f".{te.target}.build.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    shutil.rmtree(ver, ignore_errors=True)
    written = materialize_public_capsules(
        ver, tier_ceiling=tier_ceiling, corpus_roots=roots,
        exclude=getattr(te, "graded_exclude", ()))
    # Seal the descriptor's source->formal-cohort transform beside the immutable materialized root.
    # The official grader imports and validates this record instead of pretending the already-filtered
    # cache was the source pool (which reported 34/34 and erased 14 declared exclusions).
    source_names = sorted(p.name for p in _public_capsule_dirs_in(roots))
    admitted_names = sorted(written)
    capability_excluded = sorted(getattr(te, "graded_capability_exclude", ()) or ())
    resource_excluded = sorted(getattr(te, "graded_resource_exclude", ()) or ())
    declared_excluded = sorted(getattr(te, "graded_exclude", ()) or ())
    if capability_excluded or resource_excluded:
        policy = "descriptor_capability_and_resource_v1"
        if sorted(set(capability_excluded) | set(resource_excluded)) != declared_excluded:
            raise ValueError("explicit capability/resource exclusions do not equal graded_exclude")
    elif declared_excluded:
        # A claim-bearing formal run must migrate this ambiguous legacy form; keeping a named policy in
        # the record makes the phase gate reject it rather than silently guessing why rows disappeared.
        policy = "descriptor_exclusions_legacy"
        resource_excluded = declared_excluded
    else:
        policy = "all_discovered"
    if set(source_names) - set(admitted_names) != set(declared_excluded):
        raise ValueError("materialized cohort does not match the descriptor's declared exclusions")
    expected_source = getattr(te, "graded_expected_source_capsules", None)
    expected_admitted = getattr(te, "graded_expected_admitted_capsules", None)
    if expected_source is not None and len(source_names) != expected_source:
        raise ValueError(
            f"source cohort drift: descriptor expects {expected_source}, discovered {len(source_names)}")
    if expected_admitted is not None and len(admitted_names) != expected_admitted:
        raise ValueError(
            f"admitted cohort drift: descriptor expects {expected_admitted}, wrote {len(admitted_names)}")
    descriptor_sha = _loaded_descriptor_sha256(te)
    cohort_record = {
        "version": 1,
        "policy": policy,
        "n_source_capsules": len(source_names),
        "n_admitted_capsules": len(admitted_names),
        "n_capability_excluded": len(capability_excluded),
        "n_resource_excluded": len(resource_excluded),
        "excluded_name_set_sha256": _name_set_sha256(declared_excluded),
        "admitted_name_set_sha256": _name_set_sha256(admitted_names),
        "resource_policy": getattr(te, "graded_resource_policy", None),
        "required_admitted_models": sorted(getattr(te, "graded_required_models", ()) or ()),
        "descriptor_sha256": descriptor_sha,
    }
    (ver / ".cohort_admission.json").write_text(
        json.dumps(cohort_record, indent=2) + "\n", encoding="utf-8")
    validate_materialized_cohort(ver, te)
    tmp_link = base / f".{te.target}.lnk.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    os.symlink(ver.name, tmp_link)            # relative target -> resolves beside the versioned dir
    if link.exists() and not link.is_symlink():
        shutil.rmtree(link, ignore_errors=True)   # one-time migration off the old in-place real dir
    os.replace(tmp_link, link)                # atomic publish (replaces the prior symlink in one step)
    # Best-effort GC of this target's OTHER finished builds. AGE IS NOT A LIVENESS PROOF, and assuming it
    # was is the bug this block used to be: the old rule collected any build older than 15 minutes "that no
    # live reader can still hold a path into". Readers hold them far longer than that. ``capsule_grade.grade``
    # resolves the per-target symlink to a CONCRETE build dir once, up front, and then reads capsules out of
    # that absolute path for the whole grade -- tens of minutes to hours once a cert tier runs. Measured: an
    # arm materialized this target at 03:14:22, the age-only GC rmtree'd the build dir a LIVE sibling grade
    # had already resolved, and every capsule that grade touched afterwards surfaced as
    # ``schema / structural_invariant_violation`` -- a 1/32 official verdict blaming the agent's package for
    # an input the harness had deleted, while the same submission scored 33/34 minutes earlier.
    #
    # So a build is collected only when it is old AND provably unheld, on both counts we can actually
    # check: its BUILDER is gone (the pid is in the dir name) and no LIVE READER holds a lease on it. Every
    # unknown answers "held" -- an uncollected stale dir costs disk in a purgeable namespace, while a
    # wrongly collected one costs a run its verdict.
    cutoff = time.time() - _GC_MIN_AGE_S
    held = _held_build_names(base)
    for old in base.glob(f".{te.target}{_BUILD_INFIX}*"):
        if old.name == ver.name or old.name in held:
            continue
        if _pid_alive(_build_owner_pid(old.name)):
            continue
        try:
            if old.stat().st_mtime < cutoff:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    return link


def cert_capsule_cover(corpus_roots, *, labels: set[str] | None = None,
                       tile_dim: int | None = None, exclude: set[str] | None = None) -> dict:
    """The REPRESENTATIVE subset a cycle-accurate cert tier should run, when the functional tier runs
    everything. Returns ``{"capsules": [...], "cells": [...], "uncovered": [...], "basis": {...}}``.

    Why a subset at all. The functional tier answers "does it compute the right value" and is cheap; the
    cert tier answers "does the hardware actually do it" -- encoding, protocol, resource limits -- and is
    minutes per capsule. Running cert on everything spends most of a run re-proving the same hardware
    facts, and on this repo's SIMT target that was 80% of an agent round for a verdict the score never
    read. Running it on nothing leaves the RTL claim unevidenced. A cover is the middle: full coverage at
    the functional tier, representative coverage at the cert tier.

    Why THESE axes. Representativeness must track where the HARDWARE differs, not where the numerics do:

      * ``semantic_family`` -- a contraction drives different RTL than a normalization or a movement.
      * operand ``dtype``   -- the proxy for WHICH compute unit runs it (block-scaled microscaling formats
                              go to the MX PE; ordinary floats to the SIMT lanes) and for datapath width.

      * tile ALIGNMENT (only when ``tile_dim`` is given) -- whether the capsule's extents divide evenly
        by the target's tile edge, or leave a partial tile. This is the axis a functional model is least
        able to stand in for: a partial tile changes addressing and the working-set boundary, and this
        repo has already been bitten by exactly that -- a taped-out unit computed partial N tiles
        (``n % 64 != 0``) wrongly while every functional check passed. A cover built on family and dtype
        alone can pick, for each cell, the one capsule whose extents happen to divide evenly, and then
        certify no partial tile anywhere. Passing ``tile_dim`` closes that blind spot; omitting it leaves
        it open, which is why ``basis`` reports which axes were actually used.

    All are declared per capsule and read as data, so a new target's cover falls out of its own corpus
    with no edit here. ``expected_instruction_coverage.instruction_classes`` would be the most faithful
    axis of all and is deliberately NOT used: every capsule in this repo declares it empty, so selecting on
    it would silently return a cover of one. That is recorded in ``basis`` so the caller can see which
    axes actually carried the choice rather than assuming all of them did.

    Greedy set cover, which is within a log factor of optimal and, more usefully, is explainable: each
    chosen capsule is the one adding the most uncovered cells. ``uncovered`` is returned rather than
    swallowed -- a cell no capsule can cover is a corpus gap the caller should surface, not hide.
    """
    labels = labels or {"public"}
    # A capsule the descriptor EXCLUDES FROM GRADING cannot represent its cell. The cover names, per
    # cell, the one capsule a cert tier should spend minutes on -- and promotion only ever enqueues a
    # capsule that is in it. Choosing one that never runs retires the cell for a certificate nobody will
    # produce, which is strictly worse than leaving the cell uncovered: uncovered is REPORTED.
    #
    # Measured on radiance: `contraction/i64/partial` is the whole-model cell, and the greedy pick landed
    # on M1_lstmnetvit_fp32 -- one of the three models `grading.exclude_capsules` withholds from the paid
    # loop. So M0_small_llama_fp32, the model that actually runs, was not in the cover and could never be
    # promoted to the cert tier. The whole-model capstone could pass its functional tier forever and never
    # reach RTL, for a reason nothing reported.
    exclude = set(exclude or ())
    rows = []
    for root in ([corpus_roots] if isinstance(corpus_roots, (str, Path)) else corpus_roots):
        for cy in sorted(Path(root).glob("*/capsule.yaml")):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if cap.get("label") not in labels:
                continue
            if (cap.get("name") or cy.parent.name) in exclude:
                continue
            sem = cap.get("semantic") or {}
            dts = sorted({str(t.get("dtype")) for t in (cap.get("inputs") or []) if t.get("dtype")})
            if not dts:
                continue
            align = None
            if tile_dim and tile_dim > 0:
                extents = [int(x) for t in (cap.get("inputs") or [])
                           for x in (t.get("shape") or []) if str(x).lstrip("-").isdigit()]
                # THE REQUIREMENT'S OWN CLASSIFIER, imported rather than restated. This computed
                # `partial if any remainder else aligned`, which cannot name the third class
                # `Cell.alignment` demands -- so every `*_sub_tile` cell was required and uncoverable,
                # the exact debt the note below describes, and a capsule barely occupying its tile was
                # counted as covering the nearly-full case it does not exercise.
                from merlin.targetgen.conformance import classify_alignment
                align = classify_alignment(extents, tile_dim)
            # A capsule covers its own family AND any family it FUSES. A target may declare a family
            # reachable only in composition (`composed_with: [contraction]`), which makes a standalone
            # capsule for it the wrong capsule -- the eligibility oracle refuses one as a false fallback.
            # Crediting only `semantic_family` therefore left such a cell permanently uncoverable while
            # the requirement kept demanding it: a gap no capsule could close, reported forever as debt.
            fams = [sem.get("semantic_family")] + list(sem.get("composed_families") or ())
            rows.append({"name": cap.get("name") or cy.parent.name,
                         "family": sem.get("semantic_family"),
                         "families": tuple(f for f in fams if f),
                         "dtypes": dts, "align": align})

    def _cells(r):
        return {(f, dt, r["align"]) for f in r["families"] for dt in r["dtypes"]}

    cells = {c for r in rows for c in _cells(r)}
    uncovered, chosen = set(cells), []
    while uncovered:
        best, gain = None, 0
        for r in rows:
            if r in chosen:
                continue
            g = len(_cells(r) & uncovered)
            if g > gain:
                best, gain = r, g
        if best is None:                      # nothing left can cover what remains -> report it
            break
        chosen.append(best)
        uncovered -= _cells(best)

    n_class = sum(1 for r in rows if r.get("instruction_classes"))
    return {
        "capsules": sorted(r["name"] for r in chosen),
        "cells": sorted("/".join(x for x in (f, dt, al) if x) for f, dt, al in cells),
        "uncovered": sorted("/".join(x for x in (f, dt, al) if x) for f, dt, al in uncovered),
        "basis": {"axes": ["semantic_family", "dtype"] + (["tile_alignment"] if tile_dim else []),
                  "tile_dim": tile_dim, "n_candidates": len(rows),
                  "n_cells": len(cells), "n_chosen": len(chosen),
                  "instruction_classes_available": n_class,
                  "note": ("instruction_classes is declared empty by every capsule in this corpus, so it "
                           "could not be used as an axis" if not n_class else "")},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Materialize the sandbox public-capsule view.")
    ap.add_argument("dest", help="destination directory (gitignored sandbox / runs path)")
    ap.add_argument("--tier-ceiling", default=_DEFAULT_CEILING,
                    help=f"highest reachable oracle tier (default {_DEFAULT_CEILING})")
    ap.add_argument("--contract", default=None, help="contract dir override")
    a = ap.parse_args(argv)
    names = materialize_public_capsules(a.dest, tier_ceiling=a.tier_ceiling, contract=a.contract)
    print(f"materialized {len(names)} public capsules -> {a.dest} (tiers <= {a.tier_ceiling})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
