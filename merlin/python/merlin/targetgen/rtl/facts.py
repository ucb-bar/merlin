"""Target-agnostic resolution of RTL-derived facts and the introspect run/cache location.

RTL facts are a **generated artifact**: they are what our CIRCT/firtool tooling EXTRACTS from the
target's RTL (``circt_introspect`` -> the HW-dialect decoder/op graph), never a hand-written file. The
working copy lives in the **purgeable** cache (``out/artifacts/cache/rtl_introspect/<target>/facts.json``,
gitignored) and is REGENERATED on demand by :func:`ensure_facts` when the cache is cold.

A target MAY also ship a REVIEWED pin of that extraction in its backend package
(``<package>/contracts/rtl_facts/facts.json``). That pin is what makes the artifact reachable where the
RTL is not — above all inside the agent sandbox, which grants the pin and masks both the external RTL
checkout and the purgeable cache. :func:`_committed_facts_path` resolves it; see that function for why
the package is not always named after the target.

This module is the single place that maps a target name -> its facts artifact and its purgeable
scratch dir, so no consumer hardcodes the gemmini path (they used to, with three different
``parents[]`` depths). Mirrors :func:`merlin.targetgen.contract.schemas.contract_dir`.

A TARGET NAME IS NOT ALWAYS A DESIGN. A config variant of another target's generator, and a family name
the registry resolves to one elaborated configuration, both have no elaboration of their own — asking
for facts under such a name extracts against a design that does not exist and writes an empty artifact
that reads as "this hardware has no structure". :func:`facts_alias` resolves the name to the design its
OWN declaration names (the residual's ``facts_target``, else the target registry), and
:func:`load_facts` stamps the redirect onto the doc it returns so a fact is never silently attributed
to the wrong device.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

from merlin.common.paths import artifacts_dir, targets_dir

# Re-entrancy guard: ``ensure_facts`` regenerates by importing ``circt_introspect`` (which imports
# this module) — the guard makes a regeneration that transitively re-asks for the same target fail
# loud instead of recursing forever.
_REGENERATING: set[str] = set()


def target_base(target: str) -> Path:
    """Per-target home: the curated ``merlin/targets/<t>`` if it exists, else the generated
    ``artifacts/targets/<t>`` (covers targets like muon that have no hand-curated reference dir)."""
    ref = targets_dir() / target
    if ref.is_dir():
        return ref
    return artifacts_dir() / "targets" / target


def rtl_facts_path(target: str, *, explicit: str | Path | None = None) -> Path:
    """Resolve the RTL facts artifact PATH (pure — no I/O, no regeneration): explicit >
    ``$MERLIN_RTL_FACTS`` > the purgeable cache ``out/artifacts/cache/rtl_introspect/<t>/facts.json``.

    This resolves to the GENERATED artifact's location; it never points at ``merlin/targets/<t>``.
    Use :func:`ensure_facts` / :func:`load_facts` when you need the file to actually exist (they
    regenerate the cache when it is cold)."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("MERLIN_RTL_FACTS")
    if env:
        return Path(env)
    return rtl_cache_dir(target) / "facts.json"

def _facts_declares(path: Path) -> str | None:
    """The target a committed facts artifact says it is ABOUT (``facts.target``), or None when it says
    nothing / cannot be read. Pure content — this is how a pin is matched to a target without trusting
    the directory it happens to sit in."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    body = doc.get("facts") if isinstance(doc, dict) else None
    name = body.get("target") if isinstance(body, dict) else None
    return str(name) if name else None


def _committed_facts_candidates(target: str):
    """Every place ``target``'s reviewed pin could live, best first. All DERIVED, no per-target literal.

    1. Where the target's OWN DESCRIPTOR says its contracts live (``backend_package_dir`` ->
       :attr:`~merlin.targetgen.target_experiment.TargetExperiment.rtl_facts_pin`). This is the same
       string the BUNDLE GENERATOR grants, so the accessor looks exactly where the sandbox mounts.
    2. The naming convention ``merlin/targets/<target>/contracts/rtl_facts/`` — true whenever the
       experiment target and the package that serves it share a name.
    3. The package the TARGET REGISTRY resolves (an out-of-tree package's ``contracts/rtl_facts/``),
       the same "the manifest says WHAT, the registry says WHERE" split
       :func:`merlin.targetgen.sandbox.bwrap._resolve_target_package_grant` already uses for the mount.
    4. Any committed package pin that DECLARES this target. A TARGET'S PACKAGE DOES NOT ALWAYS SHARE ITS
       NAME (a SoC served by its core's package), and inside the agent sandbox neither the descriptor nor
       the target contract is mounted — the granted pin itself is the only thing that can say who it is
       for, so it is asked.
    """
    from merlin.common.paths import repo_root, targets_dir

    seen: set[Path] = set()

    def _emit(cand):
        if cand is None:
            return None
        cand = Path(cand)
        if cand in seen:
            return None
        seen.add(cand)
        return cand

    try:
        from merlin.targetgen.target_experiment import descriptor_for, load_target_experiment
        desc = descriptor_for(target)
        if desc is not None:
            got = _emit(repo_root() / load_target_experiment(desc).rtl_facts_pin / "facts.json")
            if got is not None:
                yield got
    except Exception:  # noqa: BLE001 - no/unreadable descriptor: fall through to the conventions below
        pass

    try:
        root = targets_dir()
    except Exception:  # noqa: BLE001 - no targets root here means no committed artifact
        root = None
    if root is not None:
        got = _emit(root / target / "contracts" / "rtl_facts" / "facts.json")
        if got is not None:
            yield got

    try:
        from merlin.targetgen import target_registry
        resolved = Path(target_registry.resolve(target).facts_path)
    except Exception:  # noqa: BLE001 - unresolvable target: nothing to add
        resolved = None
    # The registry hands a REFERENCE target the purgeable cache path, which `ensure_facts` has already
    # tried and which is not a committed artifact in any case; only a package-owned pin is a candidate.
    if resolved is not None and resolved != rtl_facts_path(target):
        got = _emit(resolved)
        if got is not None:
            yield got

    if root is not None and root.is_dir():
        for cand in sorted(root.glob("*/contracts/rtl_facts/facts.json")):
            got = _emit(cand)
            if got is not None and _facts_declares(got) == target:
                yield got


def _committed_facts_path(target: str):
    """The reviewed, in-tree RTL-facts artifact for ``target``, or None when it ships none.

    Candidates come from :func:`_committed_facts_candidates` (derived — no per-target literal) and are
    accepted only when the artifact does not claim to be about a DIFFERENT target. That content check is
    what lets one package hold the pin of the SoC it serves without that pin being handed back for the
    package's own name: a pin is matched to a target by what it SAYS, never by where it sits.

    What this closes: an experiment target served by a differently-named core package looked for its pin
    under its own name, where nothing can ever exist. The cache is not granted in the agent sandbox, so
    both lookups missed, regeneration produced ``facts: {}`` (the external RTL is deliberately not
    exposed), and every RTL-derived authoring tool the arm is granted raised ``FactsEmpty`` — measured on
    the SIMT target as an all-``None`` ``rtl_backend.target_profile`` and a launch NO-GO, while the bundle
    it was handed mounted a perfectly good artifact.
    """
    for cand in _committed_facts_candidates(target):
        if not cand.is_file():
            continue
        declared = _facts_declares(cand)
        if declared is None or declared == target:
            return cand
    return None



#: Per-process memo for :func:`ensure_facts`: ``(target, path) -> (stat stamp, resolved path)``. Never
#: persisted. Cleared implicitly by any write to the artifact, since the stamp is part of the key's value.
_RESOLVED_CACHE: dict[tuple[str, str], tuple[tuple, Path]] = {}


def _stat_stamp(p: Path) -> tuple:
    """A cheap identity for the file at ``p`` — ``(exists, mtime_ns, size)``, or ``(False,)`` when absent.
    Any regeneration changes it, so a memo keyed on it cannot serve a superseded artifact."""
    try:
        st = p.stat()
    except OSError:
        return (False,)
    return (True, st.st_mtime_ns, st.st_size)


def clear_resolution_cache() -> None:
    """Drop the ensure_facts memo (use after regenerating an artifact through a path this process did
    not write itself)."""
    _RESOLVED_CACHE.clear()


def _declared_extractor(target: str) -> str | None:
    """The fact extractor ``target``'s OWN compute-unit family declares, or None when it cannot be
    resolved (no manifest, no known kind — the caller then accepts whatever is cached)."""
    try:
        from ..families import family_profile, known_kinds
        from .mlc_bridge import _resolve_kind
        kind = _resolve_kind(target)
        return family_profile(kind).fact_extractor if kind in known_kinds() else "circt_static"
    except Exception:  # noqa: BLE001 — unresolvable family ⇒ no opinion about the cache
        return None


def _written_by_another_family(doc, target: str) -> bool:
    """True when a cached artifact was written by a DIFFERENT family's extractor than the one this
    target's family declares.

    The failure this catches: the production dispatch defaulted every unrecognised extractor name to the
    systolic CIRCT one, so a spatial tile's artifact was produced by an extractor that cannot see a
    command-buffer tile — a near-empty body that then WON the cache lookup forever, because it is not
    empty enough to look wrong. The artifact records which module wrote it, so the disagreement is
    readable; an artifact that records no generator (an older family adapter) is accepted rather than
    invalidated on a guess."""
    name = ((doc or {}).get("generator") or {}).get("name") if isinstance(doc, dict) else None
    if not isinstance(name, str) or not name:
        return False
    extractor = _declared_extractor(target)
    if extractor is None:
        return False
    try:
        _mode, produce = _producer_for(extractor)
    except Exception:  # noqa: BLE001 — unregistered extractor is reported when we regenerate, not here
        return False
    expected = getattr(produce, "__module__", None)
    return bool(expected) and name != expected


#: Cache for :func:`facts_alias` — the residual is a plain YAML side-input, but it is read on every
#: facts resolution and the answer cannot change within a process.
_FACTS_ALIAS_CACHE: dict[str, tuple[str, str | None]] = {}


def facts_alias(target: str) -> str:
    """The target whose RTL-facts ARTIFACT is ``target``'s, per ``target``'s OWN declaration.

    A config variant that shares another target's generator, decoder and mesh declares that in its
    capability residual as ``facts_source: rtl`` + ``facts_target: <other>`` — the same field
    :func:`merlin.targetgen.capability_manifests.derive_manifest` already honours when it grounds the
    variant's structural body. This resolver honours it too, so the two agree.

    What that closes: the manifest deriver read the variant's structural facts from the declared source
    and got a full body, while every OTHER consumer went through :func:`load_facts` under the variant's
    own name — where mlc registers no elaboration, because the variant is a config of a generator mlc
    elaborates under the base name. The extractor therefore ran against a design that does not exist,
    wrote ``facts: {}``, and the variant read as hardware with no structure at all. One declaration, two
    readers, opposite answers.

    Only ``facts_source: rtl`` aliases: that value is precisely the claim "my facts.json IS that
    target's". A ``simt``/``spatial`` source names a key for a DIFFERENT extractor, and redirecting the
    shared artifact on it would hand a target another machine's geometry. Fail-closed to the identity
    whenever there is no residual, no alias, or an unreadable one — never a guess.

    Reads the residual as a plain YAML side-input (the seam :func:`~.mlc_bridge._arc_target` already
    uses), so this never triggers manifest derivation and cannot recurse back through facts loading."""
    return _resolve_alias(target)[0]


def facts_alias_reason(target: str) -> str | None:
    """WHY ``target``'s facts artifact is another target's, in the words of whatever declared it — or
    None when there is no redirect. The redirect is provenance, so it must be quotable."""
    alias, why = _resolve_alias(target)
    return why if alias != target else None


def _resolve_alias(target: str) -> tuple[str, str | None]:
    """``(alias, why)`` — the uncached-once resolution behind :func:`facts_alias`."""
    if target in _FACTS_ALIAS_CACHE:
        return _FACTS_ALIAS_CACHE[target]
    alias, why = target, None
    try:
        from ..capability_manifests import _load_residual
        residual = _load_residual(target) or {}
        declared = residual.get("facts_target")
        if residual.get("facts_source") == "rtl" and isinstance(declared, str) and declared:
            alias = declared
            why = (f"{target!r}'s capability residual declares `facts_source: rtl` + "
                   f"`facts_target: {declared}` — its structural facts ARE that target's")
    except Exception:  # noqa: BLE001 — no residual / unreadable ⇒ fall through to the registry
        alias, why = target, None
    if alias == target:
        # No residual declaration: the TARGET REGISTRY's own resolution is the next-best statement of
        # declared identity. A NAME IS NOT ALWAYS THE DESIGN — a family name that resolves to one
        # elaborated configuration is the registry's answer everywhere else (contract, dialect plan,
        # backend), and the facts artifact is the one place it was not asked. Asking it here is what
        # stops a family name from extracting against an elaboration that does not exist and publishing
        # the emptiness as hardware.
        try:
            from ..target_registry import resolve as _resolve_target
            declared_name = _resolve_target(target).name
            if isinstance(declared_name, str) and declared_name and declared_name != target:
                alias = declared_name
                why = (f"the target registry resolves {target!r} to the elaborated design "
                       f"{declared_name!r} — the same resolution it already serves for the contract, "
                       f"the dialect plan and the backend")
        except Exception:  # noqa: BLE001 — unresolvable ⇒ identity, never a guess
            alias, why = target, None
    if alias != target and _FACTS_ALIAS_CACHE.get(alias, (alias, None))[0] != alias:
        # fail closed rather than follow an alias chain we cannot prove terminates
        alias, why = target, None
    _FACTS_ALIAS_CACHE[target] = (alias, why)
    return alias, why


def _has_facts(doc) -> bool:
    """True when a facts artifact carries a NON-EMPTY facts body. An empty body is not an extraction
    result — see :class:`FactsEmpty`."""
    return bool(isinstance(doc, dict) and doc.get("facts"))


def ensure_facts(target: str, *, explicit: str | Path | None = None) -> Path:
    """Resolve the facts artifact and GUARANTEE it exists, REGENERATING it from the RTL into the
    purgeable cache when the cache is cold.

    Resolution, in order: explicit / ``$MERLIN_RTL_FACTS`` win and are used as-is (an override that does
    not exist is a hard, loud ``FileNotFoundError`` — we never silently regenerate over a caller's pin);
    then a cached artifact that actually CARRIES FACTS and was written by this target's own family;
    then the target's declared facts source (:func:`facts_alias`); then the committed pin; then a
    regeneration through the extractor the target's family declares (:func:`_dump_facts_for_kind`). The
    first regen is slow (CIRCT ~seconds), every subsequent read is an instant cache hit.

    Two things deliberately do NOT count as a cache hit, because both are the fossil of a failed run
    rather than a fact about the hardware: an artifact whose ``facts`` body is EMPTY, and one whose
    recorded generator is a DIFFERENT family's extractor than this target's family declares. Either one
    served from cache masks every source that could answer, permanently — which is how a target read as
    "hardware with no structure" while its own extractor derived the lot.

    Honest degradation: extraction needs the CIRCT/mlc toolchain (extract-from-RTL is by design). When
    that toolchain is absent, ``build_facts`` falls back to the Scala-header parse — a KNOWN-weaker
    legal set — so we emit a loud warning first rather than silently serving the degraded facts."""
    p = rtl_facts_path(target, explicit=explicit)
    # Per-process memo of "this artifact, as it stands on disk, is the answer for this target".
    # The checks below (parse the body, ask the target's family who should have written it) are cheap
    # ONCE and measurably not cheap per call -- ensure_facts used to be a single `is_file()` and is
    # called thousands of times across a suite. Keyed on the file's identity AND its mtime/size, so a
    # regeneration by this or any other process invalidates the memo rather than pinning a stale answer.
    stamp = _stat_stamp(p)
    memo = _RESOLVED_CACHE.get((target, str(p)))
    if memo is not None and memo[0] == stamp:
        return memo[1]
    cached = _read_facts_doc(p) if p.is_file() else None
    if _has_facts(cached) and not _written_by_another_family(cached, target):
        _RESOLVED_CACHE[(target, str(p))] = (stamp, p)
        return p
    if explicit is not None or os.environ.get("MERLIN_RTL_FACTS"):
        if p.is_file():
            return p           # a caller's own pin is used as-is, empty or not — never regenerated over
        raise FileNotFoundError(
            f"RTL facts override does not exist: {p} (explicit=/$MERLIN_RTL_FACTS is used as-is and "
            "is never regenerated over)")
    # An EMPTY cached artifact is NOT a cache hit. It is the fossil of an extraction that read nothing,
    # and returning it let one stale failure mask every source that could have served the target — the
    # committed pin, the target's own declared facts source, and a re-run that would now succeed. It is
    # kept only as a last resort below, once those have all been asked.
    #
    # A target may DECLARE that its facts artifact is another target's (a config variant of the same
    # generator). Asked before regeneration, because a variant mlc elaborates only under the base name
    # has nothing of its own to extract and would write another empty artifact.
    alias = facts_alias(target)
    if alias != target:
        served = ensure_facts(alias)
        _RESOLVED_CACHE[(target, str(p))] = (stamp, served)
        return served
    # Cache cold: prefer the COMMITTED, reviewed artifact before regenerating from RTL.
    #
    # `rtl_facts_path` points at a PURGEABLE cache under out/artifacts/cache/. The agent sandbox grants
    # the committed pin (merlin/targets/<t>/contracts/rtl_facts/) but not that cache, and /scratch is
    # tmpfs-masked -- so inside the box the cache always misses and every arm-4 RTL tool fell through to
    # a live CIRCT extraction that needs the external chipyard checkout, which the sandbox deliberately
    # does not expose. Net effect: `gen_isa_module` and friends were granted to arm-4 and could not run,
    # for every model, on every target. The committed artifact is the provenance-carrying one anyway
    # (hardware_pins reviews it); the cache is a regeneration convenience, so falling back to the commit
    # is both correct and what makes the grant mean something.
    committed = _committed_facts_path(target)
    if committed is not None and committed.is_file():
        _RESOLVED_CACHE[(target, str(p))] = (stamp, committed)
        return committed
    if cached is not None and UNKNOWN_KEY in cached:
        # Already a RECORDED fail-closed extraction ("nothing was grounded, and here is why"). Re-running
        # a hopeless extraction on every read would be slow and no more honest, so the record stands.
        return p
    if target in _REGENERATING:
        raise RuntimeError(f"re-entrant RTL-facts regeneration for target {target!r}")
    _warn_if_degraded(target)
    _REGENERATING.add(target)
    try:
        _dump_facts_for_kind(p, target)
    finally:
        _REGENERATING.discard(target)
    if not p.is_file():
        raise RuntimeError(f"RTL-facts regeneration produced no artifact at {p}")
    return p


#: How each fact-extraction FAMILY produces an artifact. The key is the extractor name a compute-unit
#: family DECLARES (:attr:`merlin.targetgen.families.FamilyProfile.fact_extractor`); the value says
#: whether the callable RETURNS a facts doc (this module writes it through the guarded writer) or WRITES
#: the artifact itself (the seam snapshots and guards it).
#:
#: A table rather than an ``if`` chain because the chain silently DEFAULTED: only ``simt_config`` was
#: branched on and every other name — including ``opu``, which the spatial family has declared since it
#: was added — fell through to the systolic CIRCT extractor. That extractor finds no RoCC decoder on a
#: command-buffer tile, so it wrote a near-empty artifact that read as "this hardware has no structure"
#: while the tile's OWN extractor derives its geometry, capacities, datapaths and latencies from the
#: state manifest. :func:`_producer_for` FAILS CLOSED on a name that is not registered here, so the next
#: family added cannot inherit that silence.
_ARTIFACT_PRODUCERS: dict[str, str] = {
    "circt_static": "writes",
    "simt_config": "returns",
    "opu": "returns",
}


def _producer_for(extractor: str):
    """``(mode, callable)`` for a declared extractor name, or a hard failure naming it.

    Function-local imports throughout: every extractor module reads this one, so a module-level import
    would be circular."""
    if extractor == "simt_config":
        from .mlc_bridge import simt_facts
        return _ARTIFACT_PRODUCERS[extractor], simt_facts
    if extractor == "opu":
        from .spatial_introspect import spatial_facts
        return _ARTIFACT_PRODUCERS[extractor], spatial_facts
    if extractor == "circt_static":
        from .circt_introspect import dump_facts
        return _ARTIFACT_PRODUCERS[extractor], dump_facts
    raise RuntimeError(
        f"no artifact producer is registered for fact extractor {extractor!r} (registered: "
        f"{sorted(_ARTIFACT_PRODUCERS)}). A family that declares an extractor nothing produces must "
        f"fail here rather than fall through to another family's extractor and publish its silence as "
        f"a fact about the hardware.")


def _dump_facts_for_kind(p, target: str) -> None:
    """Extract ``target``'s facts with the extractor ITS OWN family declares, and write the artifact.

    Regeneration used to call the systolic CIRCT extractor for every target, whatever its kind. On a core
    with no RoCC decoder that extractor finds nothing and writes a well-formed artifact with an EMPTY
    facts body -- which then reads as "the RTL was never extracted" and fails the pre-spend readiness
    gate, while the target's OWN extractor derives its geometry, capacities and instruction encoding
    perfectly well. The arms that are supposed to be GROUNDED in RTL facts were being handed nothing.

    Kind-routed like :func:`~.mlc_bridge.fact_bundle_for` and :func:`~.mlc_bridge.render_fact_bundle_for`
    already are, so this is the same seam applied to PRODUCTION rather than to reading and rendering.
    Unchanged for every ``circt_static`` family (systolic/vector/scalar): they resolve to the same
    ``dump_facts`` call as before.
    """
    from ..families import family_profile, known_kinds
    # Same function-local mlc_bridge import _warn_if_degraded already uses (mlc_bridge reads this module,
    # so a module-level import would be circular). Kind comes from the target's DECLARED identity.
    from .mlc_bridge import _resolve_kind
    kind = _resolve_kind(target)
    extractor = family_profile(kind).fact_extractor if kind in known_kinds() else "circt_static"
    mode, produce = _producer_for(extractor)
    if mode == "returns":
        doc = produce(target)
        if not (doc or {}).get("facts"):
            # An extractor that grounded nothing must SAY SO in the artifact. Writing a bare `facts: {}`
            # is what made this class of failure invisible: an empty body reads downstream as "this
            # hardware has no such structure" when the truth is "the extractor did not read anything".
            write_facts_guarded(p, _unknown_artifact(target, extractor, doc))
            return
        write_facts_guarded(p, doc)
        return
    # Extractors that write the artifact THEMSELVES (circt_static for systolic/vector/scalar) cannot be
    # handed the guarded writer, so the seam guards them: snapshot, let the extractor write, and refuse
    # the result if it hollowed a fact out. Doing it HERE rather than in each extractor is what makes the
    # ratchet a property of every compute-unit family — including one added later, whose author will not
    # know to opt in.
    before = _read_facts_doc(p)
    produce(p, target=target)
    _refuse_hollowed(p, before)
    _record_empty_extraction(p, target, extractor)


#: Top-level artifact key holding ``{fact name -> why it could not be derived}``. Deliberately NOT inside
#: ``facts``: putting a reason in the body would make an ungrounded artifact test as populated, and the
#: whole point is that :class:`FactsEmpty` still fires — now carrying the reason with it.
UNKNOWN_KEY = "unknown"


def _unknown_artifact(target: str, extractor: str, doc: dict | None) -> dict:
    """A well-formed artifact that records an extraction which grounded NOTHING, with the reason."""
    doc = dict(doc or {})
    unknown = dict(doc.get(UNKNOWN_KEY) or {})
    unknown.setdefault("facts", (
        f"the {extractor!r} fact extractor produced no facts for {target!r}. This is a MISSING INPUT "
        f"(the elaboration / state manifest / introspect the family needs was not reachable), not a "
        f"statement that the hardware has no structure."))
    doc.setdefault("schema_version", "2.0")
    doc.setdefault("inputs", {"target": target})
    doc["facts"] = doc.get("facts") or {}
    doc[UNKNOWN_KEY] = unknown
    return doc


def _record_empty_extraction(p, target: str, extractor: str) -> None:
    """Annotate a just-written artifact whose body came out EMPTY with WHY, in place.

    The self-writing extractors record their inputs faithfully (``hw_sha: "missing"`` is what located this
    whole class of bug) but publish the emptiness itself without comment. This turns that silence into a
    written record, so the next reader sees a reason instead of an absence."""
    import json as _json
    from pathlib import Path as _Path
    doc = _read_facts_doc(p)
    if doc is None or doc.get("facts") or UNKNOWN_KEY in doc:
        return
    inputs = doc.get("inputs") or {}
    doc[UNKNOWN_KEY] = {"facts": (
        f"the {extractor!r} fact extractor ran for {target!r} and grounded NOTHING "
        f"(inputs: {', '.join(f'{k}={v!r}' for k, v in sorted(inputs.items()))}). A MISSING INPUT, not "
        f"a hardware fact: point the family's toolchain at a reachable elaboration and re-run.")}
    _Path(p).write_text(_json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _warn_if_degraded(target: str) -> None:
    """Warn LOUDLY when facts are about to be extracted without the CIRCT/mlc toolchain (the fallback
    Scala-header parse yields a known-weaker legal set) — honest degradation, never a silent wrong."""
    try:
        from .mlc_bridge import mlc_available
        ok, why = mlc_available()
    except Exception as e:  # noqa: BLE001 — mlc not importable is itself the degraded case
        ok, why = False, f"mlc_bridge import failed: {e}"
    if not ok:
        warnings.warn(
            f"RTL facts for {target!r}: CIRCT/mlc extraction unavailable ({why}); falling back to the "
            "Scala-header parse (KNOWN-weaker legal funct set). Facts derived-from-RTL require the "
            "toolchain by design — install/point MERLIN_MLC_DIR for faithful extraction.",
            RuntimeWarning, stacklevel=3)


def target_contract_path(target: str, *, explicit: str | Path | None = None) -> Path:
    """Resolve the target contract yaml: explicit > ``$MERLIN_TARGET_CONTRACT`` > ``<base>/contracts/target_contract.yaml``."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("MERLIN_TARGET_CONTRACT")
    if env:
        return Path(env)
    return target_base(target) / "contracts" / "target_contract.yaml"


def dialect_plan_path(target: str, *, explicit: str | Path | None = None) -> Path:
    """Resolve the target's dialect plan: explicit > ``<base>/contracts/dialect_plan.yaml``."""
    if explicit:
        return Path(explicit)
    return target_base(target) / "contracts" / "dialect_plan.yaml"


#: Top-level key stamped on a doc that was SERVED FOR one target out of ANOTHER target's artifact.
SERVED_FOR_KEY = "served_for"


def load_facts(target: str, *, explicit: str | Path | None = None) -> dict[str, Any]:
    """Load and parse the facts artifact, regenerating the cache from the RTL if it is cold
    (see :func:`ensure_facts`). This is the accessor consumers should use to READ facts.

    When the artifact belongs to a DIFFERENT target than the one asked for (a config variant served out
    of the design its own residual declares — see :func:`facts_alias`), the returned doc is stamped with
    ``served_for``, saying who asked, whose artifact answered, and what that redirect does NOT cover.
    Recording which elaboration a fact came from is the repo's hardware-provenance rule; a redirect that
    left no trace would make the variant's facts indistinguishable from its base's, which is exactly the
    confusion "a result attributed to the wrong device" describes. The stamp lives OUTSIDE ``facts``, so
    the body every consumer reads is byte-identical to the base's."""
    doc = json.loads(ensure_facts(target, explicit=explicit).read_text(encoding="utf-8"))
    alias = facts_alias(target)
    if alias != target and isinstance(doc, dict) and SERVED_FOR_KEY not in doc:
        doc[SERVED_FOR_KEY] = {
            "target": target,
            "artifact_of": alias,
            "why": facts_alias_reason(target),
            "not_covered": ("STRUCTURAL facts only (geometry, capacities, command decode). Anything the "
                            "variant changes about the DATAPATH — element dtypes, scaling, requant — is "
                            "NOT in this artifact and must come from the variant's own contract; reading "
                            "it here would report the base design's numerics as the variant's."),
        }
    return doc


class FactsDowngrade(RuntimeError):
    """A regeneration would replace an existing facts artifact with a STRICTLY WEAKER one.

    :class:`FactsEmpty` covers the artifact that came out wholly empty. This covers the subtler and more
    dangerous case: an artifact that regenerates *populated* but with individual facts hollowed out,
    because an optional extractor was missing from the environment. Measured on the muon artifact, a
    regeneration without ``MERLIN_MLC_DIR`` keeps every key and still turns

        instruction_classes: [AUIPC, BRANCH, CUSTOM0, ...] -> []
        address_spaces:      {"global": 0, "shared": 1}    -> None
        max_src_operands:    3 (RTL-derived)               -> 4 (ISA-doc fallback)

    Nothing failed and nothing was logged on that path -- ``_warn_if_degraded`` guards ``ensure_facts``,
    not a direct extractor run -- so the only thing standing between a gutted artifact and every
    downstream consumer was noticing by hand. Downstream, an empty ``instruction_classes`` reads as "this
    endpoint has no ISA", which is the opposite of the truth and precisely the confusion
    :class:`FactsEmpty` exists to prevent.

    The rule is the repo's existing ratchet, applied to facts: evidence may only get RICHER. A genuine
    hardware change that legitimately removes a fact passes ``allow_downgrade=True`` and says so.
    """


def hollowed_facts(old: dict, new: dict) -> list[str]:
    """Facts present-and-populated in ``old`` that ``new`` empties or drops. Pure; no I/O.

    "Weaker" is deliberately narrow -- a value going to ``None``/``[]``/``{}``/absent -- so that ordinary
    churn (a count changing, an evidence string getting longer) is never mistaken for a downgrade. That
    keeps the guard quiet enough to leave on.
    """
    def _hollow(v) -> bool:
        return v is None or v == [] or v == {} or v == ""

    out: list[str] = []
    for key, ofacts in (old.get("facts") or {}).items():
        nfacts = (new.get("facts") or {}).get(key)
        if nfacts is None:
            out.append(key)
            continue
        if isinstance(ofacts, dict) and isinstance(nfacts, dict):
            for k, ov in ofacts.items():
                if _hollow(ov):
                    continue
                if _hollow(nfacts.get(k)):
                    out.append(f"{key}.{k}")
        elif not _hollow(ofacts) and _hollow(nfacts):
            out.append(key)
    return sorted(out)


def _read_facts_doc(path):
    """The facts doc at ``path``, or ``None`` when absent/unreadable. Never raises: a snapshot that
    cannot be taken means there is nothing to protect, not that the write should fail."""
    import json as _json
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.is_file():
        return None
    try:
        return _json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _refuse_hollowed(path, before) -> None:
    """After an extractor wrote ``path`` itself, restore ``before`` and raise if facts were hollowed.

    The extractor already overwrote the file, so the artifact is put BACK before raising -- a guard that
    reports the downgrade but leaves the gutted file in place would have destroyed the thing it exists
    to protect.
    """
    import json as _json
    from pathlib import Path as _Path
    if before is None:
        return
    after = _read_facts_doc(path)
    if after is None:
        return
    lost = hollowed_facts(before, after)
    if not lost:
        return
    _Path(path).write_text(_json.dumps(before, indent=2) + "\n", encoding="utf-8")
    raise FactsDowngrade(
        f"refusing the regeneration of {path}: it would hollow out {lost} (the previous artifact has "
        f"been restored). A hollowed fact is the signature of a MISSING EXTRACTOR, not of hardware that "
        f"lost a feature — check the toolchain this target's family needs (e.g. MERLIN_MLC_DIR).")


def write_facts_guarded(path, doc: dict, *, allow_downgrade: bool = False) -> None:
    """Write a facts artifact, REFUSING to hollow out an existing one.

    Every extractor should write through here rather than calling ``write_text`` itself: the hole this
    closes is not in any one extractor but in the fact that a partial extraction looks exactly like a
    successful one on the way out.
    """
    import json as _json
    from pathlib import Path as _Path

    p = _Path(path)
    if p.is_file() and not allow_downgrade:
        try:
            old = _json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        lost = hollowed_facts(old, doc) if old else []
        if lost:
            raise FactsDowngrade(
                f"refusing to overwrite {p}: regeneration would hollow out {lost}. This is what a "
                f"MISSING EXTRACTOR looks like (e.g. MERLIN_MLC_DIR unset gives an ISA-doc fallback that "
                f"empties instruction_classes) — point the toolchain and re-run. If the hardware really "
                f"lost these facts, pass allow_downgrade=True.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps(doc, indent=2) + "\n", encoding="utf-8")


class FactsEmpty(RuntimeError):
    """The facts artifact exists but carries NO extracted facts.

    Distinct from "this target has no instruction decode" (:class:`NotImplementedError` from
    :func:`decode_body`) because the two demand opposite responses: an ISA-less endpoint genuinely has no
    decode table and the right answer is "not applicable", whereas an empty artifact means the extractor
    never found the RTL and every fact downstream of it is absent — a hard blocker that must not be
    reported as "N/A for this endpoint". Both radiance's and mx_gemmini's cached artifacts are in this
    state (``hw_sha: "missing"``, ``facts: {}``), and because an empty dict satisfied the old
    is-it-a-dict check they read as a valid decode body and crashed the generators one layer down with
    ``KeyError: 'interfaces'`` — a broken-tool symptom for a missing-input cause.
    """


def unknown_reasons(facts: dict[str, Any]) -> dict[str, str]:
    """The artifact's RECORDED reasons for facts it could not derive (``{fact name -> why}``).

    Empty when the artifact records none — which, for an artifact with an empty body, is itself the older
    and worse failure mode this key exists to end."""
    rec = facts.get(UNKNOWN_KEY) if isinstance(facts, dict) else None
    return {str(k): str(v) for k, v in rec.items()} if isinstance(rec, dict) else {}


def _empty_reason(facts: dict[str, Any]) -> str:
    """The suffix an :class:`FactsEmpty` message carries: the artifact's own recorded reasons, or a note
    that it recorded none."""
    rec = unknown_reasons(facts)
    if rec:
        return " Recorded reason(s): " + "; ".join(f"{k}: {v}" for k, v in sorted(rec.items()))
    return (" The artifact records NO reason for the emptiness, which is itself a defect: an extractor "
            "that grounds nothing must write why (facts.UNKNOWN_KEY).")


def facts_body(facts: dict[str, Any], target: str, *, needs: str) -> dict[str, Any]:
    """The body of a facts artifact (``facts["facts"]``) for a consumer that does NOT read a decode table.

    Same fail-closed contract as :func:`decode_body` for an EMPTY body (a missing input, worth fixing) and
    for a non-dict artifact -- it only drops the decode-shape requirement. A numeric-shape checker is
    derived from ``datapaths``/``memories`` and never looks at an opcode, so demanding a
    ``funct_decode_table`` of it refuses a target whose facts are entirely sufficient for the job.
    """
    body = facts.get("facts") if isinstance(facts, dict) else None
    if isinstance(body, dict) and body:
        return body
    if isinstance(body, dict):          # present but EMPTY -> nothing was extracted; see FactsEmpty
        inputs = (facts.get("inputs") or {}) if isinstance(facts, dict) else {}
        raise FactsEmpty(
            f"{target}: the RTL-facts artifact is EMPTY (facts: {{}}), so {needs} cannot be derived — "
            f"the extractor produced no facts (inputs: hw_mlir={inputs.get('hw_mlir')!r} "
            f"hw_sha={inputs.get('hw_sha')!r}). This is a MISSING INPUT: re-run introspection with the "
            f"RTL reachable (MERLIN_MLC_DIR / the design's hw.mlir). Artifact: {rtl_facts_path(target)}."
            + _empty_reason(facts))
    shape = sorted(facts) if isinstance(facts, dict) else type(facts).__name__
    raise NotImplementedError(
        f"{target}: this RTL-facts artifact carries no facts body, so {needs} cannot be derived. "
        f"The artifact holds {shape}.")


def decode_body(facts: dict[str, Any], target: str, *, needs: str) -> dict[str, Any]:
    """The decode-shaped body of a facts artifact (``facts["facts"]``), or a clear refusal.

    Not every accelerator HAS an instruction decode. A command-buffer spatial tile is driven over one-hot
    op ports and has no opcode, no funct field and no decode table at all, so its fact bundle carries a
    different shape entirely -- and a consumer that reaches straight for ``facts["facts"]`` greets that
    with ``KeyError: 'facts'``, which reads as a broken tool rather than as "this generator does not
    apply to this class of target". The distinction matters when onboarding: one is a bug to fix, the
    other is a capability the target genuinely does not have, and only one of them should be worked on.

    ``needs`` names what the caller was going to read, so the message says which fact was missing.
    """
    body = facts.get("facts") if isinstance(facts, dict) else None
    if isinstance(body, dict) and body:
        # Non-empty is not the same as DECODE-shaped, which is what this function's name promises and what
        # every caller goes on to read (each looks up the ``funct_decode_table`` interface). A self-hosted
        # ISA core carries a populated body with no decode table at all; returning it sent each generator
        # off to fail in its own way -- and the readiness gate, whose N/A verdict keys on the refusal
        # RAISED HERE ("the single place that distinction is made"), reported a target with perfectly good
        # facts as three broken generators. Refuse structurally, by the interfaces the body declares.
        if not any(i.get("name") == "funct_decode_table"
                   for i in (body.get("interfaces") or []) if isinstance(i, dict)):
            declared = sorted(i.get("name") for i in (body.get("interfaces") or [])
                              if isinstance(i, dict) and i.get("name"))
            raise NotImplementedError(
                f"{target}: these RTL facts carry no instruction-decode body, so {needs} cannot be "
                f"derived. The endpoint declares {declared or 'no interfaces'} and no RoCC "
                f"funct_decode_table — a capability this class of target does not have, not a missing "
                f"input. Artifact: {rtl_facts_path(target)}")
        return body
    if isinstance(body, dict):          # present but EMPTY -> nothing was extracted; see FactsEmpty
        inputs = (facts.get("inputs") or {}) if isinstance(facts, dict) else {}
        raise FactsEmpty(
            f"{target}: the RTL-facts artifact is EMPTY (facts: {{}}), so {needs} cannot be derived — "
            f"the extractor produced no facts (inputs: hw_mlir={inputs.get('hw_mlir')!r} "
            f"hw_sha={inputs.get('hw_sha')!r}). This is a MISSING INPUT, not an ISA-less endpoint: "
            f"re-run introspection with the RTL reachable (MERLIN_MLC_DIR / the design's hw.mlir) "
            f"rather than treating the generators as inapplicable. Artifact: {rtl_facts_path(target)}."
            + _empty_reason(facts))
    shape = sorted(facts) if isinstance(facts, dict) else type(facts).__name__
    raise NotImplementedError(
        f"{target}: these RTL facts carry no instruction-decode body, so {needs} cannot be derived. "
        f"The artifact holds {shape}. A command-buffer or otherwise ISA-less target has no decode table "
        f"by construction -- it needs a generator for ITS endpoint, not this one.")


def rtl_cache_dir(target: str, *, ensure: bool = False) -> Path:
    """Purgeable introspect scratch (hw.mlir input, ``*.ll``/``*.o``, arcilator bins, per-run
    facts.json) under ``artifacts/cache/rtl_introspect/<target>/`` — never inside ``merlin/``.

    Mirrors :func:`merlin.common.artifacts.cache_dir` (``artifacts/cache/<ns>/``, PURGEABLE) without
    forcing directory creation at import time; pass ``ensure=True`` when about to write."""
    d = artifacts_dir() / "cache" / "rtl_introspect" / target
    if ensure:
        d.mkdir(parents=True, exist_ok=True)
    return d
