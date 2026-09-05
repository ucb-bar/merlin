"""Target-agnostic resolution of RTL-derived facts and the introspect run/cache location.

RTL facts are a **generated artifact**: they are what our CIRCT/firtool tooling EXTRACTS from the
target's RTL (``circt_introspect`` -> the HW-dialect decoder/op graph), never a hand-committed file.
There is no committed ``facts.json`` pin any more — the artifact lives in the **purgeable** cache
(``out/artifacts/cache/rtl_introspect/<target>/facts.json``, gitignored) and is REGENERATED on demand
by :func:`ensure_facts` when the cache is cold. A target's only tracked definition is its reviewed
yaml (``target_contract.yaml`` + the human-owned ``dialect_plan.yaml``).

This module is the single place that maps a target name -> its facts artifact and its purgeable
scratch dir, so no consumer hardcodes the gemmini path (they used to, with three different
``parents[]`` depths). Mirrors :func:`merlin.targetgen.contract.schemas.contract_dir`.
"""
from __future__ import annotations

import json
import hashlib
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

def _committed_facts_path(target: str):
    """The reviewed, in-tree RTL-facts artifact for ``target``, or None when it ships none.

    Derived from the targets root -- no per-target literal -- so a new target is covered by dropping its
    facts file in the same place."""
    from merlin.common.paths import targets_dir
    try:
        p = targets_dir() / target / "contracts" / "rtl_facts" / "facts.json"
    except Exception:  # noqa: BLE001 - no targets root here means no committed artifact
        return None
    return p if p.is_file() else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _circt_facts_current(path: Path) -> bool:
    """Whether a default CIRCT facts artifact is bound to this extractor and its exact FIRRTL.

    A committed artifact may be read in a sandbox where the external RTL checkout is intentionally
    hidden. In that case the two independently recorded full FIR digests must still agree. When the
    source is reachable, its current bytes must agree too. Any old schema, truncated digest, mismatched
    source, or unparseable artifact is stale and must not establish a build-feature capability.

    Other fact-extractor families retain their own cache policy; this check is deliberately scoped by
    the artifact's generator identity.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    generator = document.get("generator") or {}
    if generator.get("name") != "merlin.targetgen.rtl.circt_introspect":
        return True

    from . import circt_introspect

    inputs = document.get("inputs") or {}
    if generator.get("version") != circt_introspect.GENERATOR_VERSION:
        return False
    try:
        extractor_digest = _file_sha256(Path(circt_introspect.__file__).resolve())
    except OSError:
        return False
    if inputs.get("extractor_sha256") != extractor_digest:
        return False

    features = [
        interface for interface in ((document.get("facts") or {}).get("interfaces") or [])
        if interface.get("name") == "elaborated_rtl_features"
    ]
    if len(features) != 1:
        return False
    feature = features[0]
    status = feature.get("status")
    value = (feature.get("features") or {}).get("max_pool")
    fir_digest = inputs.get("fir_sha256")
    feature_digest = feature.get("source_sha256")
    source = feature.get("source")
    has_fir_digest = isinstance(fir_digest, str) and len(fir_digest) == 64
    has_feature_digest = isinstance(feature_digest, str) and len(feature_digest) == 64
    if has_fir_digest:
        if not has_feature_digest or fir_digest != feature_digest:
            return False
        if not isinstance(source, str) or not source:
            return False
        source_path = Path(source)
    else:
        # A non-Chisel target may genuinely have no FIR input. Any other spelling is neither a full
        # digest nor an explicit absence state and therefore cannot key a cache entry.
        if fir_digest not in ("n/a", "missing", "unresolved"):
            return False
        source_path = None
    if source_path is not None and source_path.is_file():
        try:
            if _file_sha256(source_path) != fir_digest:
                return False
        except OSError:
            return False
    if status == "unknown":
        return value is None
    return status == "derived" and isinstance(value, bool) and has_fir_digest



def ensure_facts(target: str, *, explicit: str | Path | None = None) -> Path:
    """Resolve the facts artifact and GUARANTEE it exists, REGENERATING it from the RTL into the
    purgeable cache when the cache is cold.

    Resolution: explicit / ``$MERLIN_RTL_FACTS`` win and are used as-is (an override that does not
    exist is a hard, loud ``FileNotFoundError`` — we never silently regenerate over a caller's pin).
    Otherwise the cache path is used; if it is missing we invoke ``circt_introspect.dump_facts`` to
    extract facts from the RTL and write the cache, then return the path. The first regen is slow
    (CIRCT ~seconds), every subsequent read is an instant cache hit.

    Honest degradation: extraction needs the CIRCT/mlc toolchain (extract-from-RTL is by design). When
    that toolchain is absent, ``build_facts`` falls back to the Scala-header parse — a KNOWN-weaker
    legal set — so we emit a loud warning first rather than silently serving the degraded facts."""
    p = rtl_facts_path(target, explicit=explicit)
    override = explicit is not None or bool(os.environ.get("MERLIN_RTL_FACTS"))
    if override:
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"RTL facts override does not exist: {p} (explicit=/$MERLIN_RTL_FACTS is used as-is and "
            "is never regenerated over)")
    if p.is_file() and _circt_facts_current(p):
        return p
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
    if committed is not None and committed.is_file() and _circt_facts_current(committed):
        return committed
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
    if not _circt_facts_current(p):
        raise RuntimeError(
            f"RTL-facts regeneration produced a stale or unbound CIRCT artifact at {p}")
    return p


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
    import json as _json
    from ..families import family_profile, known_kinds
    # Same function-local mlc_bridge import _warn_if_degraded already uses (mlc_bridge reads this module,
    # so a module-level import would be circular). Kind comes from the target's DECLARED identity.
    from .mlc_bridge import _resolve_kind
    kind = _resolve_kind(target)
    extractor = family_profile(kind).fact_extractor if kind in known_kinds() else "circt_static"
    if extractor == "simt_config":
        from .mlc_bridge import simt_facts
        body = simt_facts(target)
        if not body:
            raise RuntimeError(
                f"no SIMT introspect served {target!r}, so its facts artifact would be empty; register "
                f"an introspect for it rather than writing a body that reads as un-extracted RTL")
        write_facts_guarded(p, body)
        return
    # Extractors that write the artifact THEMSELVES (circt_static for systolic/vector/scalar; opu for
    # spatial) cannot be handed the guarded writer, so the seam guards them: snapshot, let the extractor
    # write, and refuse the result if it hollowed a fact out. Doing it HERE rather than in each extractor
    # is what makes the ratchet a property of every compute-unit family — including one added later,
    # whose author will not know to opt in.
    from .circt_introspect import dump_facts      # function-local: circt_introspect imports this module
    before = _read_facts_doc(p)
    dump_facts(p, target=target)
    _refuse_hollowed(p, before)


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


def load_facts(target: str, *, explicit: str | Path | None = None) -> dict[str, Any]:
    """Load and parse the facts artifact, regenerating the cache from the RTL if it is cold
    (see :func:`ensure_facts`). This is the accessor consumers should use to READ facts."""
    return json.loads(ensure_facts(target, explicit=explicit).read_text(encoding="utf-8"))


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
            f"RTL reachable (MERLIN_MLC_DIR / the design's hw.mlir). Artifact: {rtl_facts_path(target)}")
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
            f"rather than treating the generators as inapplicable. Artifact: {rtl_facts_path(target)}")
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
