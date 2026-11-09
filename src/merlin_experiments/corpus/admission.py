"""Host corpus workflows whose tier policy depends on constructed evaluator adapters.

Core owns conformance derivation, copying, admission validation, coverage and
leases. This module owns asking real evaluator factories what they constructed;
advertised tier metadata is never a substitute for that observation. Construction
does not itself prove that an oracle can execute or certify a submitted program.
"""

from __future__ import annotations

from pathlib import Path

from merlin.targetgen import conformance
from merlin.targetgen.contract import materialize
from merlin.targetgen.contract.materialize import validate_materialized_cohort

__all__ = [
    "conformance_spec",
    "constructed_oracle_tiers",
    "public_capsules_for",
    "validate_materialized_cohort",
]


def constructed_oracle_tiers(target: str) -> list:
    """Actual constructed tier keys, sorted; unresolved or empty remains unknown.

    Use the standard descriptor for this target, not the single-descriptor override.
    The original best-effort conformance query intentionally turns construction
    failures into an empty observation, never an invented cheaper tier.
    """
    try:
        from merlin.targetgen import capsule_runner as CR
        from merlin.targetgen import corpora
        from merlin.targetgen.target_experiment import load_target_experiment

        desc = corpora.standard_descriptor_path(str(target))
        if not desc.is_file():
            return []
        te = load_target_experiment(desc)
        return sorted(CR.oracle_adapters(target, te.sim_via) or {})
    except Exception:  # noqa: BLE001 — unresolvable: report nothing
        return []


def conformance_spec(target: str, captures: dict[str, str | Path], **kwargs) -> dict:
    """Derive the existing requirement with a fresh constructed-tier observation."""
    return conformance.derive_spec(target, captures, oracle_tiers=lambda: constructed_oracle_tiers(target), **kwargs)


def public_capsules_for(
    te, *, tier_ceiling: str | None = None, corpus_roots: list[Path] | None = None, destination: Path | None = None
) -> Path:
    """Publish the public cohort without substituting undeclared oracle tiers.

    An explicit ceiling bypasses all evaluator imports and construction. Otherwise
    retain the native QA-loop query followed by a second full construction: overrides,
    constructor failures and the empty-declaration legacy behavior remain authoritative.
    """
    roots = materialize.descriptor_corpus_roots(te) if corpus_roots is None else corpus_roots
    if tier_ceiling is None:
        from merlin.targetgen import capsule_runner as CR

        declared = materialize.declared_oracle_tiers(*roots)
        loop = CR.qa_loop_adapters(te.target, te.sim_via, declared_tiers=declared)
        if not loop:
            reach = sorted(CR.oracle_adapters(te.target, te.sim_via))
            if reach:
                raise ValueError(
                    f"target {te.target!r}: its capsule corpus declares required oracle tiers "
                    f"{sorted(declared)} but the endpoint reaches {reach} — no declared tier is "
                    f"reachable, so this phase cannot grade these capsules. Refusing to substitute a "
                    f"tier the capsules never declared; make the declared tier reachable or fix the "
                    f"corpus."
                )
            # No adapter was constructed: retain the legacy floor. The unchanged
            # materializer still records unreachable declared tiers for the grader.
            tier_ceiling = materialize._DEFAULT_CEILING
        else:
            # Preserve the highest declared reachable tier, not just the cheap
            # loop tier; otherwise materialization silently removes cert barriers.
            reach = set(CR.oracle_adapters(te.target, te.sim_via))
            usable = (declared & reach) or set(loop)
            tier_ceiling = max(
                usable, key=lambda tier: materialize._TIER_ORDER.index(tier) if tier in materialize._TIER_ORDER else -1
            )
    options = {"destination": destination} if destination is not None else {}
    return materialize.materialize_public_cohort(te, tier_ceiling=tier_ceiling, corpus_roots=roots, **options)
