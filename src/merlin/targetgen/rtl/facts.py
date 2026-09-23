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

import contextlib
import json
import os
import warnings
from pathlib import Path
from typing import Any

from merlin.common.digest import is_sha256, sha256_file
from merlin.common.paths import artifacts_dir

# Re-entrancy guard: ``ensure_facts`` regenerates by importing ``circt_introspect`` (which imports
# this module) — the guard makes a regeneration that transitively re-asks for the same target fail
# loud instead of recursing forever.
_REGENERATING: set[str] = set()


def target_base(target: str) -> Path:
    """The selected support provider's home, using the shared read-only registry."""
    from merlin.targetgen.target_registry import resolve

    return resolve(target).base


def _selected_external_facts(target: str) -> tuple[bool, Path | None]:
    """Selected OOT support is authoritative even when it ships no facts.

    Name-keyed caches and descriptor aliases cannot establish identity with this
    provider. Read its pin directly; no provider digest or hardware qualification
    is inferred from its location. Invalid selected resources fail closed.
    """
    from merlin.targetgen.providers import contained_resource
    from merlin.targetgen.target_registry import resolve

    selected = resolve(target)
    if selected.kind != "external":
        return False, None
    path = selected.facts_path
    if not path.exists() and not path.is_symlink():
        return True, None
    path = contained_resource(selected.base, str(path.relative_to(selected.base)))
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("facts"), dict):
        raise ValueError(f"{path}: expected an RTL facts mapping")
    declared = doc["facts"].get("target")
    if declared is not None and declared != selected.name:
        raise ValueError(f"{path}: facts target {declared!r} differs from selected target {selected.name!r}")
    return True, path


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

    selected, pin = _selected_external_facts(target)
    if selected:
        if pin is not None:
            yield pin
        return

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


#: Version tokens that are family-specific, for artifacts written before ``family`` was stamped.
#: Not a shape map and not a fallback ladder: these two tokens each name exactly one extractor's output
#: and always did. The third family's artifacts spell a bare number that three different shapes share,
#: so it cannot be read this way and is resolved from the generator instead.
_VERSION_FAMILY = {"simt-facts/v0": "simt_config", "spatial-facts/v0": "opu"}


def family_of(doc: Any, *, target: str | None = None) -> tuple[str | None, str]:
    """``(family, basis)`` for a facts artifact -- which extractor's SHAPE its body has.

    The discriminator every consumer needed and none had. Forty-odd modules recognise a facts body by
    reaching for the key they want, so the three shapes on disk are told apart by a key each, and adding
    a fourth family means discovering each of those sites by breaking it.

    Resolved in order of how directly the artifact says it: the stamped ``family``; then the module that
    generated it, matched against what each known family declares as its extractor; then the two version
    tokens that are family-specific by construction. **Fails closed** -- the body's own shape is never
    consulted, because inferring a family from the keys present is exactly the sniffing this replaces,
    and it answers confidently for an artifact that grounded nothing.
    """
    if not isinstance(doc, dict):
        return None, "not a facts artifact"
    declared = doc.get("family")
    if isinstance(declared, str) and declared:
        # "Stamped on it", not "stamped by its extractor": one committed artifact is hand-promoted, its
        # generator is prose rather than a module, and a human declared the family. Both are stamps.
        return declared, "stamped on the artifact"
    generator = ((doc.get("generator") or {}) if isinstance(doc.get("generator"), dict) else {}).get("name")
    if isinstance(generator, str) and generator:
        for kind, extractor in _known_extractors().items():
            try:
                _mode, produce = _producer_for(extractor)
            except Exception:  # noqa: BLE001 — an unregistered extractor cannot claim this artifact
                continue
            if getattr(produce, "__module__", None) == generator:
                return extractor, f"generator {generator!r} is the extractor the {kind!r} family declares"
    version = doc.get("schema_version")
    if isinstance(version, str) and version in _VERSION_FAMILY:
        return _VERSION_FAMILY[version], f"schema_version {version!r} names one extractor's output"
    if target is not None:
        extractor = _declared_extractor(target)
        if extractor:
            return None, (
                f"the artifact stamps no family, names no generator this repo registers, and carries "
                f"schema_version {version!r}; {target!r} DECLARES the {extractor!r} family, but taking "
                "that as the artifact's own would assert the very agreement worth checking"
            )
    return None, f"no family stamped, no registered generator named, and schema_version {version!r} names none"


def _known_extractors() -> dict[str, str]:
    """``{compute-unit kind: the fact extractor that family declares}``, from the family profiles."""
    from ..families import family_profile, known_kinds

    out: dict[str, str] = {}
    for kind in known_kinds():
        try:
            out[kind] = family_profile(kind).fact_extractor
        except Exception:  # noqa: BLE001 — a family with no declared extractor claims no artifact
            continue
    return out


def validate_facts(doc: Any, *, target: str | None = None) -> list[str]:
    """Problems with a facts artifact, as strings; ``[]`` when it is well-formed for its family.

    Reports rather than raises, because the first caller is a gate that must say what is wrong with
    every artifact rather than stop at the first.
    """
    from merlin.targetgen.contract.schemas import ContractViolation, validate

    family, basis = family_of(doc, target=target)
    if family is None:
        return [f"family is undecidable: {basis}"]
    candidate = dict(doc)
    candidate.setdefault("family", family)
    try:
        validate(candidate, "rtl_facts")
    except ContractViolation as exc:
        return [f"family {family!r} ({basis}): {exc}"]
    except FileNotFoundError as exc:
        return [f"the rtl_facts schema is not readable: {exc}"]
    return []


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
    from merlin.targetgen.target_registry import resolve

    selected = resolve(target)
    if selected.kind == "external":
        # A same-name legacy residual/cache cannot redirect selected OOT facts.
        # A registry-declared directory alias still names the actual design.
        why = f"the target registry resolves {target!r} to selected support design {selected.name!r}"
        return selected.name, why if selected.name != target else None
    if target in _FACTS_ALIAS_CACHE:
        return _FACTS_ALIAS_CACHE[target]
    alias, why = target, None
    try:
        from ..capability_manifests import _load_residual

        residual = _load_residual(target) or {}
        declared = residual.get("facts_target")
        if residual.get("facts_source") == "rtl" and isinstance(declared, str) and declared:
            alias = declared
            why = (
                f"{target!r}'s capability residual declares `facts_source: rtl` + "
                f"`facts_target: {declared}` — its structural facts ARE that target's"
            )
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
                why = (
                    f"the target registry resolves {target!r} to the elaborated design "
                    f"{declared_name!r} — the same resolution it already serves for the contract, "
                    f"the dialect plan and the backend"
                )
        except Exception:  # noqa: BLE001 — unresolvable ⇒ identity, never a guess
            alias, why = target, None
    if alias != target and _FACTS_ALIAS_CACHE.get(alias, (alias, None))[0] != alias:
        # fail closed rather than follow an alias chain we cannot prove terminates
        alias, why = target, None
    _FACTS_ALIAS_CACHE[target] = (alias, why)
    return alias, why


#: Suffix marking a body key that RECORDS A REFUSAL rather than states a fact about the silicon.
#: ``datapaths_undeterminable`` is the one in the tree today: the cell-geometry reader appends its own
#: reason there when it cannot locate a compute element, which is exactly the right thing to do -- and
#: it lands INSIDE ``facts``, so a body that grounded nothing at all came out as
#: ``{"datapaths_undeterminable": [...]}``, tested as POPULATED, and won the cache lookup forever.
#:
#: Measured in this tree: k1_cpu, saturn, toy_npu, rvv, toy_vec, voyager_accel, gemmini_universal and
#: some twenty test targets were each cached in exactly that state -- a note about what could not be
#: read, mistaken for hardware structure by every reader, with no ``unknown`` block ever written
#: because the empty-extraction recorder saw a non-empty body and returned early. Zero consumers read the
#: key; every consumer was affected by it.
#:
#: A SUFFIX rather than a fixed name so the next reader that records a refusal in the body is covered
#: without editing this module -- the convention is the contract. Matched with ``str.endswith``; no
#: pattern matching.
UNDETERMINABLE_SUFFIX = "_undeterminable"


def grounded_facts(doc) -> dict[str, Any]:
    """The part of a facts body that STATES something rather than recording a refusal.

    Drops only the recorded-refusal keys (see :data:`UNDETERMINABLE_SUFFIX`), so
    "the extractor ran and read nothing" is distinguishable from "the extractor read structure".
    An EMPTY section is deliberately kept: `_has_facts` is a non-emptiness test and not a completeness
    test, because completeness is per-extractor and per-class -- an extractor that reported
    ``interfaces: []`` did produce a result, and whether that result meets its class's obligation is
    :func:`unmet_obligations`' question, asked with the kind set in hand. Pure; accepts either a whole
    artifact or ``None``.
    """
    body = doc.get("facts") if isinstance(doc, dict) else None
    if not isinstance(body, dict):
        return {}
    return {k: v for k, v in body.items() if not k.endswith(UNDETERMINABLE_SUFFIX)}


def _has_facts(doc) -> bool:
    """True when a facts artifact carries a body that GROUNDED something. An empty body -- or one
    holding only recorded refusals -- is not an extraction result; see :class:`FactsEmpty`."""
    return bool(grounded_facts(doc))


def _declared_cache_pins_match(doc: dict) -> bool:
    """Check existing extractor/FIR/feature commitments before reusing a cache.

    This is not hardware qualification. Artifacts without these optional pins
    remain legacy, unverified evidence. A declared pin whose source cannot be
    resolved or no longer matches is not a reusable cache entry.
    """
    from merlin.common.paths import module_source_path

    inputs = doc.get("inputs") or {}
    if not isinstance(inputs, dict):
        return False

    checked = {}

    def matches(path, digest):
        if not is_sha256(digest) or not isinstance(path, (str, Path)) or not path:
            return False
        try:
            if path not in checked:
                checked[path] = sha256_file(path)
            return checked[path] == digest
        except OSError:
            return False

    extractor = inputs.get("extractor_sha256")
    if extractor is not None:
        generator = doc.get("generator") or {}
        # Only the core extractor currently declares this commitment. Do not
        # import arbitrary generator names from an evidence document.
        if not isinstance(generator, dict) or generator.get("name") != "merlin.targetgen.rtl.circt_introspect":
            return False
        if not matches(module_source_path("merlin.targetgen.rtl.circt_introspect"), extractor):
            return False
    reader = inputs.get("extraction_reader_sha256")
    if reader is not None and not matches(module_source_path("merlin.targetgen.rtl.extraction_contract"), reader):
        return False
    extraction_contract = inputs.get("extraction_contract_sha256")
    if extraction_contract is not None:
        target = inputs.get("target")
        if not isinstance(target, str) or not target or not matches(target_contract_path(target), extraction_contract):
            return False
    body = doc.get("facts") or {}
    if not isinstance(body, dict):
        return False
    interfaces = body.get("interfaces") or []
    fir_sources = []
    for item in interfaces:
        if not isinstance(item, dict):
            continue
        digest = item.get("source_sha256")
        if digest in (None, "unresolved", "missing", "n/a"):
            continue
        if not matches(item.get("source"), digest):
            return False
        if item.get("name") == "elaborated_rtl_features":
            fir_sources.append(item.get("source"))
    fir_digest = inputs.get("fir_sha256")
    if fir_digest not in (None, "unresolved", "missing", "n/a"):
        sources = [inputs["fir_path"]] if inputs.get("fir_path") else fir_sources
        if not sources or not all(matches(path, fir_digest) for path in sources):
            return False
    return True


def ensure_facts(target: str, *, explicit: str | Path | None = None) -> Path:
    """Resolve the facts artifact and GUARANTEE it exists, REGENERATING it from the RTL into the
    purgeable cache when the cache is cold.

    Resolution, in order: explicit / ``$MERLIN_RTL_FACTS`` win and are used as-is (an override that does
    not exist is a hard, loud ``FileNotFoundError`` — we never silently regenerate over a caller's pin);
    then an OOT support provider's own reviewed pin, when that provider is selected (missing means
    unavailable, never another provider's cache); otherwise a cached artifact that actually CARRIES
    FACTS, matches its declared source commitments and was written by this target's own family;
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
    result = _resolve_facts(target, explicit=explicit, regenerate=True)
    assert result is not None
    return result


def find_facts(target: str, *, explicit: str | Path | None = None) -> Path | None:
    """Same evidence selection as ensure_facts, but never extract or create files.

    Missing selected support facts are unavailable, not permission to mix another
    provider's evidence. Malformed selected resources and missing explicit
    overrides still raise. This is the discovery interface, not qualification.
    """
    return _resolve_facts(target, explicit=explicit, regenerate=False)


def _resolve_facts(target: str, *, explicit: str | Path | None, regenerate: bool) -> Path | None:
    if explicit is None and not os.environ.get("MERLIN_RTL_FACTS"):
        selected, pin = _selected_external_facts(target)
        if selected:
            if pin is None:
                if not regenerate:
                    return None
                raise FileNotFoundError(f"{target}: selected support provider has no contracts/rtl_facts/facts.json")
            return pin
    p = rtl_facts_path(target, explicit=explicit)
    if explicit is not None or os.environ.get("MERLIN_RTL_FACTS"):
        if p.is_file():
            return p  # Explicit evidence is not a regenerable default cache.
        raise FileNotFoundError(f"RTL facts override does not exist: {p}; never regenerated over")
    # Per-process memo of "this artifact, as it stands on disk, is the answer for this target".
    # Memoize family admission, not source commitments: the latter must be rechecked
    # even when the artifact itself is unchanged. An edited FIR otherwise keeps an
    # unchanged facts-file stamp and would reuse stale feature claims indefinitely.
    stamp = _stat_stamp(p)
    memo = _RESOLVED_CACHE.get((target, str(p)))
    cached = _read_facts_doc(p) if p.is_file() else None
    pins_match = isinstance(cached, dict) and _declared_cache_pins_match(cached)
    if memo is not None and memo[0] == stamp and memo[1] == p and pins_match:
        return p
    if _has_facts(cached) and pins_match and not _written_by_another_family(cached, target):
        _RESOLVED_CACHE[(target, str(p))] = (stamp, p)
        return p
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
        served = _resolve_facts(alias, explicit=None, regenerate=regenerate)
        if served is not None:
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
    if not regenerate:
        return None
    if cached is not None and pins_match and UNKNOWN_KEY in cached:
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
        f"a fact about the hardware."
    )


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

    THE KIND IS A SET, AND SO IS THE EXTRACTION. This used to resolve ONE kind while the READING path
    (:func:`~.mlc_bridge._extractors_for` + :func:`~.mlc_bridge._merge_fact_bundles`) already resolved the
    whole set and merged what each extractor found. The two disagreed for every hybrid in the tree:
    ``saturn`` declares ``kinds=('vector', 'spatial')`` and READS as ``('circt_static', 'opu')`` while
    PRODUCING ``circt_static`` alone, so its outer-product tile geometry was never written under the name
    ``saturn``; ``muon`` declares ``('simt', 'systolic')`` and produced ``simt_config`` alone. Production
    now walks the same set and folds the results with the same collision policy, so what a target reads
    and what a target's artifact contains are the same question asked twice.

    AND THE CLASS HAS AN OBLIGATION. Routing an extractor was never a statement about what the extractor
    owed, so an extractor that read nothing produced a well-formed artifact with an empty body -- which
    downstream is indistinguishable from "this silicon has no such structure". Each kind now declares
    ``FamilyProfile.required_facts``, and an artifact that does not meet them records WHICH obligation
    went unmet (:func:`unmet_obligations`) rather than publishing the gap as a fact.
    """
    # Same function-local mlc_bridge import _warn_if_degraded already uses (mlc_bridge reads this module,
    # so a module-level import would be circular). Kinds come from the target's DECLARED identity.
    from .mlc_bridge import KindUnresolved, _extractors_for, _resolve_kinds

    try:
        kinds = _resolve_kinds(target)
        extractors = _extractors_for(target)
    except KindUnresolved as exc:
        # REFUSED, not defaulted. Routing an unresolvable kind to the static reader is how a vector CPU
        # came to be extracted by the systolic extractor, silently, for every consumer. The refusal is
        # RECORDED in the artifact so the next reader is told why rather than handed a shape.
        write_facts_guarded(p, _unresolved_kind_artifact(target, exc))
        return
    if len(extractors) == 1:
        mode, _produce = _producer_for(extractors[0])
        doc = _produce_one(p, target, extractors[0])
        # A self-writing family has ALREADY been through the guarded writer, inside its own extractor,
        # against the artifact that was there before. The write below re-serialises those same bytes
        # plus the stamps, so re-running the ratchet would compare the artifact to itself and refuse on
        # a stamp. Every other write is guarded exactly as before.
        rewrite_of_own_bytes = mode == "writes"
    else:
        # A hybrid: every extractor produces into scratch and the merge decides the artifact, so no
        # single datapath's result ever lands at the target's own path unmerged.
        docs = [_produce_one(None, target, e) for e in extractors]
        doc = merge_fact_artifacts(docs, tuple(extractors))
        rewrite_of_own_bytes = False
    doc = _stamp_family(doc, extractors[0], families=tuple(extractors))
    doc = _stamp_toolchain(doc)
    doc = _record_unmet_obligations(doc, target, kinds, tuple(extractors))
    write_facts_guarded(p, doc, allow_downgrade=rewrite_of_own_bytes)


def _produce_one(p, target: str, extractor: str) -> dict:
    """Run ONE family's extractor and return its artifact doc.

    ``p`` is the artifact path for the self-writing families (``circt_static``), which write the file
    themselves and cannot be handed the guarded writer -- so the seam guards them: snapshot, let the
    extractor write, and refuse the result if it hollowed a fact out. Doing it HERE rather than in each
    extractor is what makes the ratchet a property of every compute-unit family, including one added
    later whose author will not know to opt in. Pass ``p=None`` (a hybrid's non-primary extractor) to
    have such a family write to a scratch path instead, so one datapath's artifact never lands at the
    target's own path unmerged.
    """
    mode, produce = _producer_for(extractor)
    if mode == "returns":
        return dict(produce(target) or {})
    if p is None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="merlin-facts-") as tmp:
            scratch = Path(tmp) / "facts.json"
            produced = produce(scratch, target=target)
            # The FILE is the extractor's answer -- the return value is a convenience some writers do
            # not bother with, and reading the path is the same thing the in-place branch below does.
            return _read_facts_doc(scratch) or dict(produced or {})
    before = _read_facts_doc(p)
    produce(p, target=target)
    _refuse_hollowed(p, before)
    return _read_facts_doc(p) or {}


def _stamp_family(doc: dict, extractor: str, *, families: tuple[str, ...] = ()) -> dict:
    """Record WHICH EXTRACTOR produced this artifact, and therefore which shape its body has.

    Stamped at write time, by the only code that knows the answer without inferring it. Every reader
    that wants the family otherwise has to work it out from the keys present, which answers confidently
    for an artifact that grounded nothing and has to be taught about each new family by breaking it.
    An existing stamp is never overwritten: a hand-promoted artifact declares its own.

    ``families`` is stamped only for a HYBRID -- a target whose declared kinds route to more than one
    extractor, whose artifact therefore carries more than one family's shape at once. A single-family
    artifact is byte-identical to what this wrote before, so nothing downstream sees a new key until a
    target genuinely has two datapaths.
    """
    out = dict(doc or {})
    out.setdefault("family", extractor)
    if len(families) > 1:
        out.setdefault(FAMILIES_KEY, list(families))
    return out


#: Top-level artifact key holding ``{fact name -> why it could not be derived}``. Deliberately NOT inside
#: ``facts``: putting a reason in the body would make an ungrounded artifact test as populated, and the
#: whole point is that :class:`FactsEmpty` still fires — now carrying the reason with it.
UNKNOWN_KEY = "unknown"


#: Top-level artifact key naming EVERY extractor that contributed, for a hybrid whose declared kinds
#: route to more than one. Absent on a single-family artifact, so nothing existing grows a key.
FAMILIES_KEY = "families"

#: Top-level artifact key holding readings two extractors both derived and DISAGREED on. Kept rather
#: than resolved: a hybrid whose two halves report different geometries has a real problem, and a merge
#: that hides it turns that problem into a wrong number nobody can trace.
CONFLICTS_KEY = "conflicts"


def merge_fact_artifacts(docs: list[dict], extractors: tuple[str, ...]) -> dict:
    """Union several extractors' ARTIFACTS into the one artifact a target's name resolves to.

    THE COLLISION POLICY IS NOT RE-DECIDED HERE. The reading path already arbitrates two extractors
    that disagree (:func:`~.mlc_bridge._merge_fact_bundles`: evidence wins, and two derived readings
    that differ are BOTH kept under ``conflicts`` rather than letting the first-listed datapath win),
    and production must not evolve a second opinion about the same question. The bodies are therefore
    flattened to that function's ``fields`` shape -- one entry per named block, keyed
    ``(section, block name)`` -- arbitrated by it, and folded back. The flatten/fold is lossless: a
    block goes in and comes out as itself.

    Block ORDER follows first appearance, and a block that only one extractor saw is kept verbatim, so
    a hybrid's artifact is the union of what its datapaths ground rather than whichever half ran first.
    """
    from .mlc_bridge import _merge_fact_bundles

    bundles = [
        {"target": (d.get("facts") or {}).get("target"), "method": e, "fields": _flatten_body(d, e)}
        for d, e in zip(docs, extractors, strict=False)
    ]
    merged = _merge_fact_bundles(bundles, extractors)
    head = docs[0] if docs else {}
    out: dict[str, Any] = {}
    for key in ("schema_version", "generator", "provenance", "cross_check", "rtl_hierarchy_counts"):
        if key in head:
            out[key] = head[key]
    inputs: dict[str, Any] = {}
    unknown: dict[str, Any] = {}
    for doc, extractor in zip(docs, extractors, strict=False):
        # Per-extractor, so a hybrid's two input closures do not silently overwrite each other: the
        # primary's inputs stay where every reader already looks, and the rest are named by the family
        # that read them.
        primary = extractor == extractors[0]
        for k, v in (doc.get("inputs") or {}).items():
            inputs.setdefault(k if primary else f"{extractor}.{k}", v)
        for k, v in (doc.get(UNKNOWN_KEY) or {}).items():
            # Two families can both fail to ground a section of the same name; qualifying the second
            # keeps both reasons, because "the tile geometry was unreadable" and "the lane widths were
            # unreadable" are different gaps that happen to share a word.
            unknown.setdefault(k if k not in unknown else f"{extractor}.{k}", v)
    if inputs:
        out["inputs"] = inputs
    out["facts"] = _unflatten_body(merged.get("fields") or {})
    if unknown:
        out[UNKNOWN_KEY] = unknown
    conflicts = merged.get("conflicts") or []
    if conflicts:
        out[CONFLICTS_KEY] = [
            {
                "fact": f"{c['field'][0]}.{c['field'][1]}" if c["field"][1] else str(c["field"][0]),
                "kept": c["kept"].get("value"),
                "kept_by": c["kept"].get("source"),
                "also": c["also"].get("value"),
                "also_by": c["also"].get("source"),
            }
            for c in conflicts
        ]
    return out


def _flatten_body(doc: dict, extractor: str) -> dict:
    """A facts body as the bundle ``fields`` map the reading path's merge arbitrates.

    Named-block LISTS (the one shape every family agrees on -- ``interfaces``/``memories``/``arrays``/
    ``datapaths``/``timing``) flatten one entry per block, so two extractors that each ground a
    DIFFERENT block of the same section end up with both rather than with one section winning whole.
    A block with no name is keyed by its position, which is stable within one extractor's output.
    """
    fields: dict = {}
    for section, value in (doc.get("facts") or {}).items():
        if isinstance(value, list):
            for i, block in enumerate(value):
                name = (block.get("name") or block.get("module")) if isinstance(block, dict) else None
                fields[(section, name if name else f"#{i}")] = {
                    "value": block,
                    "derived": True,
                    "source": extractor,
                }
        else:
            fields[(section, None)] = {"value": value, "derived": True, "source": extractor}
    return fields


def _unflatten_body(fields: dict) -> dict:
    """The inverse of :func:`_flatten_body`: the merged ``fields`` map back as a facts body."""
    body: dict[str, Any] = {}
    for (section, block), rec in fields.items():
        if block is None:
            body[section] = rec.get("value")
        else:
            body.setdefault(section, []).append(rec.get("value"))
    return body


def required_facts_for(kinds) -> tuple[str, ...]:
    """The UNION of ``FamilyProfile.required_facts`` over every kind ``target`` declares.

    A union, not an intersection: each declared compute unit is a separate claim about the silicon, so
    a hybrid that declares a mesh AND a tile owes the facts of both. An unknown kind contributes
    nothing rather than a guessed obligation.
    """
    from ..families import family_profile, known_kinds

    out: list[str] = []
    for kind in kinds or ():
        if kind not in known_kinds():
            continue
        for name in family_profile(kind).required_facts:
            if name not in out:
                out.append(name)
    return tuple(out)


def unmet_obligations(doc, kinds) -> tuple[str, ...]:
    """Which of ``kinds``' :attr:`~merlin.targetgen.families.FamilyProfile.required_facts` this artifact
    did NOT ground, in declaration order.

    The taxonomy's missing half. It said which extractor runs for a class and never said what the
    extractor owed, so "read the mesh" and "read nothing" produced artifacts that differ only in size
    and every consumer invented its own threshold. Checked against :func:`grounded_facts`, so a body
    holding only recorded refusals meets nothing -- which is the truth about it.
    """
    body = grounded_facts(doc)
    # PRESENT AND POPULATED. An empty section is the extractor saying it looked and wrote nothing down,
    # which downstream is indistinguishable from "this device has none" -- the very confusion the
    # obligation exists to surface. Key presence alone would let `arrays: []` discharge a mesh's
    # obligation to state its geometry.
    return tuple(name for name in required_facts_for(kinds) if not body.get(name))


def _record_unmet_obligations(doc: dict, target: str, kinds, extractors: tuple[str, ...]) -> dict:
    """Annotate ``doc`` with the class obligations its body does not meet, one named entry each.

    The gap is recorded, the partial body is KEPT. Discarding what an extractor did ground would trade
    one silent wrong for another -- the reason the artifact is worth writing at all is the evidence in
    it, and the reason the ``unknown`` block exists is that a gap must not read as a fact.
    """
    unmet = unmet_obligations(doc, kinds)
    grounded = grounded_facts(doc)
    if not unmet and grounded and kinds:
        return doc
    out = dict(doc)
    unknown = dict(out.get(UNKNOWN_KEY) or {})
    if not kinds:
        # NOTHING declared this target's compute-unit class, so there was no obligation to check and the
        # generic static reader ran by default. Recorded rather than left implicit: an artifact produced
        # with no class behind it is not attributable to one, and the silent version of this state is
        # how "we never decided what this device is" came to look exactly like "we extracted it".
        unknown.setdefault(
            "compute_unit_kind",
            (
                f"no compute-unit kind is declared for {target!r} (no capability contract, and the "
                f"arc-model registry names none), so no class obligation could be checked and the "
                f"{list(extractors)} default extractor(s) ran. The body is what that reader found; it is "
                f"not attributable to a datapath class."
            ),
        )
    if not grounded:
        unknown.setdefault(
            "facts",
            (
                f"the {list(extractors)} fact extractor(s) ran for {target!r} and grounded NOTHING. This is "
                f"a MISSING INPUT (the elaboration / state manifest / introspect the class needs was not "
                f"reachable), not a statement that the hardware has no structure."
            ),
        )
    for name in unmet:
        unknown.setdefault(
            name,
            (
                f"{target!r} declares compute-unit kind(s) {list(kinds)}, whose class REQUIRES {name!r} "
                f"in the facts body, and the {list(extractors)} extractor(s) did not ground it. Recorded "
                f"as UNKNOWN rather than left absent: an absent section reads as 'this device has no "
                f"{name}', which is a claim about silicon made out of a missing input."
            ),
        )
    out[UNKNOWN_KEY] = unknown
    out.setdefault("schema_version", "2.0")
    out.setdefault("inputs", {"target": target})
    out["facts"] = out.get("facts") or {}
    return out


def _unresolved_kind_artifact(target: str, exc: Exception) -> dict:
    """The artifact written when the target's own COMPUTE-UNIT KIND could not be resolved.

    Nothing was extracted, because nothing could decide which extractor to run. That is a different
    failure from "the extractor read nothing" and it is recorded as its own reason, so the fix (the
    target's declaration) is named rather than guessed at from an empty body.
    """
    return {
        "schema_version": "2.0",
        "inputs": {"target": target},
        "facts": {},
        UNKNOWN_KEY: {
            "compute_unit_kind": (
                f"{target!r}'s compute-unit kind could not be resolved, so NO fact extractor was run: {exc}"
            )
        },
    }


def _stamp_toolchain(doc: dict) -> dict:
    """Record WHICH BUILD of the extraction toolchain produced this artifact, under ``inputs``.

    A facts artifact already digests the bytes it READ -- the input IR, merlin's own extractor -- and
    recorded nothing about the CIRCT/firtool build that parsed them, so two artifacts extracted from
    identical RTL by different CIRCT builds were indistinguishable and no fact could be bound to the
    toolchain revision behind it. The repo's hardware-provenance rule is that a result records which
    revision it came from; the toolchain that read the hardware is part of that, not outside it.

    ABSENCE AND UNKNOWN ARE DIFFERENT STATES and both are preserved. A tool whose identity cannot be
    determined is recorded as ``UNKNOWN`` WITH the reason. An artifact that carries no ``toolchain`` at
    all -- every committed pin, which predates the field -- says nothing about its toolchain, and must
    never be read as a mismatch. An existing stamp is never overwritten.
    """
    out = dict(doc or {})
    inputs = dict(out.get("inputs") or {})
    if TOOLCHAIN_KEY in inputs:
        return out
    try:
        from .mlc_bridge import toolchain_identity

        inputs[TOOLCHAIN_KEY] = toolchain_identity()
    except Exception as exc:  # noqa: BLE001 — a probe that cannot run is itself recorded, never omitted
        inputs[TOOLCHAIN_KEY] = {"unknown_reason": f"the toolchain could not be probed: {type(exc).__name__}: {exc}"}
    out["inputs"] = inputs
    return out


#: Key under ``inputs`` holding the extraction TOOLCHAIN's identity (see :func:`_stamp_toolchain`).
#: Its absence means "not recorded" -- an artifact older than the field -- and never "they matched".
TOOLCHAIN_KEY = "toolchain"


def toolchain_of(doc) -> dict | None:
    """The toolchain identity an artifact RECORDS, or ``None`` when it records none.

    ``None`` and ``{"circt_opt": {"version": "UNKNOWN", ...}}`` are different answers and callers must
    keep them apart: the first is an artifact that predates the field, the second is an extraction that
    looked and could not tell. Collapsing them would make every committed pin look like a failed probe.
    """
    inputs = (doc or {}).get("inputs") if isinstance(doc, dict) else None
    rec = inputs.get(TOOLCHAIN_KEY) if isinstance(inputs, dict) else None
    return rec if isinstance(rec, dict) else None


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
            RuntimeWarning,
            stacklevel=3,
        )


def target_contract_path(target: str, *, explicit: str | Path | None = None) -> Path:
    """Resolve contract: explicit > ``$MERLIN_TARGET_CONTRACT`` > selected support."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("MERLIN_TARGET_CONTRACT")
    if env:
        return Path(env)
    from merlin.targetgen.target_registry import resolve

    return resolve(target).contract_path


def dialect_plan_path(target: str, *, explicit: str | Path | None = None) -> Path:
    """Resolve the target's dialect plan: explicit > ``<base>/contracts/dialect_plan.yaml``."""
    if explicit:
        return Path(explicit)
    from merlin.targetgen.target_registry import resolve

    return resolve(target).dialect_plan_path


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
    # Explicit evidence and selected support pins must not inherit an unrelated
    # descriptor/residual alias merely because it shares the requested name.
    override = explicit is not None or bool(os.environ.get("MERLIN_RTL_FACTS"))
    alias = target if override else facts_alias(target)
    if alias != target and isinstance(doc, dict) and SERVED_FOR_KEY not in doc:
        doc[SERVED_FOR_KEY] = {
            "target": target,
            "artifact_of": alias,
            "why": facts_alias_reason(target),
            "not_covered": (
                "STRUCTURAL facts only (geometry, capacities, command decode). Anything the "
                "variant changes about the DATAPATH — element dtypes, scaling, requant — is "
                "NOT in this artifact and must come from the variant's own contract; reading "
                "it here would report the base design's numerics as the variant's."
            ),
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


def body_if_present(target: str) -> dict[str, Any]:
    """``target``'s facts body, or ``{}`` when there is none -- the LAX reader, named so it can be found.

    Twenty-odd consumers spelled this inline as ``(load_facts(t) or {}).get("facts") or {}``. A consumer
    that must not proceed on missing facts should call :func:`facts_body` instead, which refuses with the
    reason; this one is for code whose own logic already treats an empty body as "nothing derived"."""
    return (load_facts(target) or {}).get("facts") or {}


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
        if key.endswith(UNDETERMINABLE_SUFFIX):
            # A recorded refusal is not evidence, so LOSING one is not a downgrade -- it is the
            # extraction finally succeeding. Counting it refused exactly the regeneration this ratchet
            # exists to make room for: an artifact whose body held only
            # `datapaths_undeterminable` could not be replaced by one that actually read the datapath.
            continue
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
        f"lost a feature — check the toolchain this target's family needs (e.g. MERLIN_MLC_DIR)."
    )


def _configuration_of(doc) -> str | None:
    """The elaborated configuration an artifact says it describes, when it says.

    Recorded by every extractor as ``facts.source.config`` and, until now, read by nothing. It is the
    one field that distinguishes two artifacts of the SAME design that describe different devices.
    """
    body = (doc or {}).get("facts", doc) or {}
    src = body.get("source")
    if isinstance(src, dict) and src.get("config"):
        return str(src["config"])
    return None


class FactsConfigMismatch(FactsDowngrade):
    """A write would replace one configuration's facts with another configuration's.

    A design has many configurations -- different mesh edge, bank count, accumulator depth, dtype
    support -- and this cache holds ONE artifact per target NAME. So a write carrying a different
    configuration does not add one, it silently replaces a different device's facts, and every capacity
    and width derived downstream is then attributed to an elaboration that did not produce it.

    A subclass of :class:`FactsDowngrade` deliberately: a caller that already handles a refusal to
    overwrite keeps handling this one, and the distinct type is there for a caller that wants to tell
    "weaker facts about this device" from "facts about another device".
    """


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
        was, now = _configuration_of(old), _configuration_of(doc)
        if was and now and was != now:
            raise FactsConfigMismatch(
                f"refusing to overwrite {p}: it records configuration {was!r} and this extraction is "
                f"{now!r}. This cache holds one artifact per target NAME, so writing here does not add "
                f"a configuration -- it replaces a different device's facts with these, and every "
                f"capacity, width and geometry derived downstream would be attributed to an elaboration "
                f"that did not produce it. Give the configuration its own target name -- which is "
                f"what the registry already does for a second configuration of one generator -- or "
                f"pass allow_downgrade=True if this slot really should describe {now!r} from now on."
            )
        lost = hollowed_facts(old, doc) if old else []
        if lost:
            raise FactsDowngrade(
                f"refusing to overwrite {p}: regeneration would hollow out {lost}. This is what a "
                f"MISSING EXTRACTOR looks like (e.g. MERLIN_MLC_DIR unset gives an ISA-doc fallback that "
                f"empties instruction_classes) — point the toolchain and re-run. If the hardware really "
                f"lost these facts, pass allow_downgrade=True."
            )
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(p, _json.dumps(doc, indent=2) + "\n")


def _write_atomic(p, text: str) -> None:
    """Write ``text`` to ``p`` so that a reader sees either the old bytes or the new ones, never half.

    THE CASE THIS CLOSES IS NOT THE ONE ABOVE. :func:`write_facts_guarded` already refuses a
    regeneration that would HOLLOW an artifact -- a semantic downgrade, where the extractor ran and
    read less than it should have. It did not cover the mechanical one: a plain ``write_text`` that
    is interrupted leaves a TRUNCATED file, and a truncated artifact is not empty, so
    :func:`_has_facts` -- which asks only whether the ``facts`` body is non-empty -- accepts it as a
    cache hit. Measured 2026-09-19: a partial artifact was read as authoritative, a derived
    ``target_contract.yaml`` then declared an ``endpoint_kind`` its own provenance string
    contradicted, and the result cost eleven tests that looked exactly like a regression.

    Making the partial file IMPOSSIBLE is preferable to teaching the reader to detect one, because
    "complete" is a property of each extractor's own output and would have to be redefined every time
    a fact is added -- whereas these artifacts are regenerable by construction, so the only thing that
    must never happen is a half-written one being mistaken for a whole one.
    """
    import os as _os
    import tempfile as _tempfile

    fd, tmp = _tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".partial")
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            # The rename below is atomic, but it only orders against data the filesystem has. Without
            # this, a crash after the rename can publish the new NAME over the old CONTENT.
            _os.fsync(handle.fileno())
        _os.replace(tmp, str(p))
    except BaseException:
        # Leaving a `.partial` behind would reintroduce exactly the artifact this exists to prevent,
        # and it is named so that anything which does survive is obviously not a facts document.
        with contextlib.suppress(OSError):
            _os.unlink(tmp)
        raise


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
    return (
        " The artifact records NO reason for the emptiness, which is itself a defect: an extractor "
        "that grounds nothing must write why (facts.UNKNOWN_KEY)."
    )


def facts_body(facts: dict[str, Any], target: str, *, needs: str) -> dict[str, Any]:
    """The body of a facts artifact (``facts["facts"]``) for a consumer that does NOT read a decode table.

    Same fail-closed contract as :func:`decode_body` for an EMPTY body (a missing input, worth fixing) and
    for a non-dict artifact -- it only drops the decode-shape requirement. A numeric-shape checker is
    derived from ``datapaths``/``memories`` and never looks at an opcode, so demanding a
    ``funct_decode_table`` of it refuses a target whose facts are entirely sufficient for the job.
    """
    body = facts.get("facts") if isinstance(facts, dict) else None
    if isinstance(body, dict) and grounded_facts(facts):
        return body
    if isinstance(body, dict):  # present but GROUNDED NOTHING -> see FactsEmpty
        inputs = (facts.get("inputs") or {}) if isinstance(facts, dict) else {}
        raise FactsEmpty(
            f"{target}: the RTL-facts artifact GROUNDED NOTHING (body: {sorted(body)}), so {needs} "
            f"cannot be derived — the extractor produced no facts (inputs: "
            f"hw_mlir={inputs.get('hw_mlir')!r} hw_sha={inputs.get('hw_sha')!r}). This is a MISSING "
            f"INPUT: re-run introspection with the RTL reachable (MERLIN_MLC_DIR / the design's "
            f"hw.mlir). Artifact: {rtl_facts_path(target)}." + _empty_reason(facts)
        )
    shape = sorted(facts) if isinstance(facts, dict) else type(facts).__name__
    raise NotImplementedError(
        f"{target}: this RTL-facts artifact carries no facts body, so {needs} cannot be derived. "
        f"The artifact holds {shape}."
    )


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
    if isinstance(body, dict) and grounded_facts(facts):
        # Non-empty is not the same as DECODE-shaped, which is what this function's name promises and what
        # every caller goes on to read (each looks up the ``funct_decode_table`` interface). A self-hosted
        # ISA core carries a populated body with no decode table at all; returning it sent each generator
        # off to fail in its own way -- and the readiness gate, whose N/A verdict keys on the refusal
        # RAISED HERE ("the single place that distinction is made"), reported a target with perfectly good
        # facts as three broken generators. Refuse structurally, by the interfaces the body declares.
        if not any(
            i.get("name") == "funct_decode_table" for i in (body.get("interfaces") or []) if isinstance(i, dict)
        ):
            declared = sorted(
                i.get("name") for i in (body.get("interfaces") or []) if isinstance(i, dict) and i.get("name")
            )
            raise NotImplementedError(
                f"{target}: these RTL facts carry no instruction-decode body, so {needs} cannot be "
                f"derived. The endpoint declares {declared or 'no interfaces'} and no RoCC "
                f"funct_decode_table — a capability this class of target does not have, not a missing "
                f"input. Artifact: {rtl_facts_path(target)}"
            )
        return body
    if isinstance(body, dict):  # present but GROUNDED NOTHING -> see FactsEmpty
        inputs = (facts.get("inputs") or {}) if isinstance(facts, dict) else {}
        raise FactsEmpty(
            f"{target}: the RTL-facts artifact GROUNDED NOTHING (body: {sorted(body)}), so {needs} "
            f"cannot be derived — the extractor produced no facts (inputs: "
            f"hw_mlir={inputs.get('hw_mlir')!r} hw_sha={inputs.get('hw_sha')!r}). This is a MISSING "
            f"INPUT, not an ISA-less endpoint: re-run introspection with the RTL reachable "
            f"(MERLIN_MLC_DIR / the design's hw.mlir) rather than treating the generators as "
            f"inapplicable. Artifact: {rtl_facts_path(target)}." + _empty_reason(facts)
        )
    shape = sorted(facts) if isinstance(facts, dict) else type(facts).__name__
    raise NotImplementedError(
        f"{target}: these RTL facts carry no instruction-decode body, so {needs} cannot be derived. "
        f"The artifact holds {shape}. A command-buffer or otherwise ISA-less target has no decode table "
        f"by construction -- it needs a generator for ITS endpoint, not this one."
    )


def rtl_cache_dir(target: str, *, ensure: bool = False) -> Path:
    """Purgeable introspect scratch (hw.mlir input, ``*.ll``/``*.o``, arcilator bins, per-run
    facts.json) under ``artifacts/cache/rtl_introspect/<target>/`` — never inside ``merlin/``.

    Mirrors :func:`merlin.common.artifacts.cache_dir` (``artifacts/cache/<ns>/``, PURGEABLE) without
    forcing directory creation at import time; pass ``ensure=True`` when about to write."""
    d = artifacts_dir() / "cache" / "rtl_introspect" / target
    if ensure:
        d.mkdir(parents=True, exist_ok=True)
    return d
