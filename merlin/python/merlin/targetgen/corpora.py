"""The single place that resolves capsule-corpus locations the LIBRARY reads (reference-by-location).

No library module should hardcode a corpus path; they call :func:`capsule_corpus_roots` /
:func:`find_capsule` instead. This is the ONE sanctioned indirection to a corpus that still lives
under ``experiments/`` (the boundary lint allowlists this module) — see the note below.

Corpora:
- ``merlin/contract/capsules`` — the frozen graded ABI suite (canonical).
- ``merlin/experiments/gemmini_perf_bench/kernels`` — the perf-bench corpus. It is *library-consumed*
  (the RTL checks screen it), so by the consumption-direction rule it is a benchmark input and its
  proper home is ``merlin/benchmarks/``. Relocating it is DEFERRED: three untracked, concurrently-edited
  perf-bench scripts still read it in place, and moving it would break that in-flight work. When they
  land, change the one line below (and repoint the perf harness) — every reader already goes through here.
"""
from __future__ import annotations

from pathlib import Path

from merlin.common.paths import merlin_dir


def capsule_corpus_roots() -> list[Path]:
    """Existing capsule-corpus roots the library may screen (canonical first)."""
    roots = [
        merlin_dir() / "contract" / "capsules",
        merlin_dir() / "experiments" / "gemmini_perf_bench" / "kernels",  # perf benchmark (to relocate)
    ]
    return [r for r in roots if r.is_dir()]


def descriptor_path(target: str) -> Path:
    """Where ``target``'s experiment descriptor lives — the per-target convention path, or whatever
    ``MERLIN_TARGET_EXPERIMENT`` overrides it to. Lives here because this module is the one allowed to
    know the ``experiments/`` layout; callers elsewhere took the convention path directly and silently
    ignored the override, so an out-of-tree descriptor resolved for some readers and not others."""
    import os

    from merlin.common.paths import repo_root
    override = os.environ.get("MERLIN_TARGET_EXPERIMENT", "").strip()
    if override:
        return Path(override)
    return (repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets" / target
            / "target_experiment.yaml")


def experiment_for(target: str):
    """``target``'s parsed descriptor, or None when it ships none / names a different target."""
    desc = descriptor_path(target)
    if not desc.is_file():
        return None
    try:
        from merlin.targetgen.target_experiment import load_target_experiment
        te = load_target_experiment(desc)
    except Exception:                                     # noqa: BLE001 — unreadable descriptor
        return None
    return te if str(getattr(te, "target", "")) == target else None


def graded_capsule_roots(target: str, *, hidden: bool = False) -> list[Path]:
    """The roots that make up ``target``'s GRADED suite, or the canonical corpus if it has no descriptor.

    ``hidden=True`` returns the holdout roots instead. They are a SEPARATE tree, deliberately excluded
    from the public roots (``corpus_siblings`` skips ``hidden/``), so a hidden grade that reused the
    public roots would match nothing and report a 0/0 "pass" that never ran. Empty is a real answer for a
    target that ships no holdouts, and is returned as empty rather than papered over with a fallback.

    A target's suite is not one directory: the capsules are split by kind into sibling categories
    (``isa`` / ``layers`` / ``model`` / ``model_slices``), and different targets keep those siblings in
    different places -- gemmini's sit at the corpus root, atlas's under ``atlas/``. Passing their common
    parent is NOT the fix, because that parent holds every target's corpus at once; see the warning in
    :func:`merlin.targetgen.capsule_common.discover_capsules`.

    Measured consequence of getting this wrong: grading the gemmini package against
    ``merlin/contract/capsules`` pulled in 173 capsules from seven targets, marked 89 of them "outside
    this target's declared capability", and reported ``1/84`` -- a number that reads like a catastrophic
    regression and means nothing. The target's own suite is 36.

    ``TargetExperiment.graded_roots()`` is the resolution the A/B launchers and ``readiness_check.py``
    already use; this exposes it to library callers that have only a target NAME. Honours
    ``MERLIN_TARGET_EXPERIMENT`` (the same override ``capsule_bench/harness/_common.py`` reads) so a
    target whose descriptor lives out of tree still resolves.
    """
    desc = descriptor_path(target)
    if not desc.is_file():
        return [] if hidden else capsule_corpus_roots()[:1]   # no descriptor: canonical corpus, unsplit
    try:
        from merlin.targetgen.target_experiment import load_target_experiment
        te = load_target_experiment(desc)
        if str(getattr(te, "target", "")) != target:      # an override naming a DIFFERENT target
            return [] if hidden else capsule_corpus_roots()[:1]
        roots = [r for r in (te.hidden_roots() if hidden else te.graded_roots()) if r.is_dir()]
    except Exception:                                     # noqa: BLE001 — unreadable descriptor
        return [] if hidden else capsule_corpus_roots()[:1]
    if hidden:
        return roots                                      # empty == this target ships no holdouts
    return roots or capsule_corpus_roots()[:1]


def find_capsule(name: str) -> Path | None:
    """Locate a capsule directory by name across the corpus roots (first match wins)."""
    for root in capsule_corpus_roots():
        for cy in root.rglob("capsule.yaml"):
            if cy.parent.name == name:
                return cy.parent
    return None
