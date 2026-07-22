"""Build a shared :class:`BenchTargetSpec` from a declarative per-target experiment descriptor.

This puts ANY target on the shared bench spine (``run_perf`` / ``redacted_grade``) without per-target
wiring: the corpus + target name come from the ``target_experiment.yaml`` descriptor (the declarative
setup), the capsule runner is injected, and the perf headline is a small extractor. So a target's perf
+ self-check reuse the one shared loop + report — no hardcoded gemmini path.
"""
from __future__ import annotations

from typing import Any, Callable

from .spec import BenchTargetSpec


def spec_from_experiment(te, runner: Any, *, perf_tier: str = "L2",
                         perf_fields: Callable[[dict], dict] | None = None,
                         labels: tuple[str, ...] | None = ("public", "dev"),
                         contract: str | None = None, name: str | None = None) -> BenchTargetSpec:
    """A :class:`BenchTargetSpec` for the target described by ``te`` (a
    :class:`~merlin.targetgen.target_experiment.TargetExperiment`), driven by ``runner`` (any module
    exposing ``discover_capsules`` + ``run_capsule``). ``perf_fields`` extracts the perf headline from a
    tier result (default: cycles)."""
    return BenchTargetSpec(
        name=name or te.target.capitalize(),
        runner=runner,
        corpus_root=te.capsule_corpus,
        labels=set(labels) if labels else None,
        contract=contract,
        perf_tier=perf_tier,
        perf_fields=perf_fields or (lambda tier: {"cycles": tier.get("cycles")}),
        peak_note=f"{te.target} (corpus {te.capsule_corpus.name if te.capsule_corpus else '?'})",
    )
