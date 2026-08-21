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
    if p.is_file():
        return p
    if explicit is not None or os.environ.get("MERLIN_RTL_FACTS"):
        raise FileNotFoundError(
            f"RTL facts override does not exist: {p} (explicit=/$MERLIN_RTL_FACTS is used as-is and "
            "is never regenerated over)")
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
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(body, indent=2), encoding="utf-8")
        return
    from .circt_introspect import dump_facts      # function-local: circt_introspect imports this module
    dump_facts(p, target=target)


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
