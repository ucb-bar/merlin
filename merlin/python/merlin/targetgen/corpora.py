"""The single place that resolves corpus locations the LIBRARY reads (reference-by-location).

No library module should hardcode a corpus path; they call :func:`capsule_corpus_roots` /
:func:`find_capsule` / :func:`kernel_corpus_root` instead. The locations themselves are DATA — the corpus
registry ``merlin/contract/corpora.yaml`` — and this module is its one reader, so adding or moving a corpus
is a registry edit. It is also the ONE sanctioned indirection to a corpus that still lives under
``experiments/`` (the boundary lint allowlists this module).

Corpora:
- capsule corpora (registry ``capsule_corpora``) — the frozen graded ABI suite (canonical, first) and the
  perf-bench corpus. The latter is *library-consumed* (the RTL checks screen it), so by the
  consumption-direction rule it is a benchmark input and its proper home is ``merlin/benchmarks/``.
  Relocating it is DEFERRED: untracked, concurrently-edited perf-bench scripts still read it in place.
  When they land, change its registry line (and repoint the perf harness) — every reader goes through here.
- expert kernel corpora (registry ``kernel_corpora``) — the framework checkouts the kernel-mining layer
  builds from, keyed by framework, each with the ``kernel.source`` spellings that name it, its layout and
  where its checkout is found.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from merlin.common.paths import merlin_dir

_REGISTRY = ("contract", "corpora.yaml")        # under merlin/


@lru_cache(maxsize=None)
def _load_registry(path: Path) -> dict:
    from merlin.common.yaml import load_yaml
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"corpus registry {path} is not a mapping")
    return data


def _registry() -> dict:
    """The corpus registry. A missing or malformed one RAISES: read as empty, it would make every corpus
    reader report "nothing to screen", which looks exactly like a clean result."""
    return _load_registry(merlin_dir().joinpath(*_REGISTRY))


def capsule_corpus_roots() -> list[Path]:
    """Existing capsule-corpus roots the library may screen (canonical first)."""
    roots = [merlin_dir() / str(rel) for rel in _registry().get("capsule_corpora") or []]
    return [r for r in roots if r.is_dir()]


def kernel_corpora() -> dict[str, dict]:
    """Every expert kernel corpus the registry declares, ``{framework: spec}``, in declaration order."""
    return {str(k): dict(v or {}) for k, v in (_registry().get("kernel_corpora") or {}).items()}


def kernel_corpus_for_source(source: str | None) -> str | None:
    """The corpus (framework name) a ``kernel.source`` spelling names, matched case-insensitively, or None."""
    want = (source or "").lower()
    if not want:
        return None
    for name, spec in kernel_corpora().items():
        if want == name.lower() or want in {str(s).lower() for s in spec.get("sources") or []}:
            return name
    return None


def kernel_corpus_env(name: str) -> str:
    """The environment variable pointing at corpus ``name``'s checkout: ``MERLIN_<NAME>_REPO`` -- the
    convention the kernel-index CLI already reads, derived from the name rather than listed."""
    return f"MERLIN_{name.upper()}_REPO"


def kernel_corpus_root(name: str) -> Path | None:
    """Root of expert corpus ``name``'s checkout, or None when the registry declares no such corpus.

    Precedence: :func:`kernel_corpus_env` when set; then a local clone at ``<repo>/tmp/kernels/<checkout>``
    if one is actually there; then the corpus's declared ``ext`` fallback (a directory inside an external
    checkout this repo already configures, via ``ext_path``) when that is on disk; else the local-clone
    path, so a fresh checkout's error names the location a clone is expected at rather than someone else's
    machine.
    """
    spec = kernel_corpora().get(name)
    if spec is None:
        return None
    env = os.environ.get(kernel_corpus_env(name))
    if env:
        return Path(env)
    from merlin.common.paths import repo_root
    local = repo_root() / "tmp" / "kernels" / str(spec.get("checkout") or name)
    if local.is_dir():
        return local
    ext = spec.get("ext") or {}
    if ext.get("name"):
        try:
            from merlin.common.paths import ext_path
            cand = ext_path(str(ext["name"])) / str(ext.get("subpath") or "")
            if cand.is_dir():
                return cand
        except (KeyError, ImportError):
            pass
    return local


def capsule_store_targets() -> list[str]:
    """Every target that owns a capsule store, discovered from the tree.

    A caller that has no target name yet needs this before it can call anything keyed on one, and the
    alternative — each caller listing the directory itself — is how a target path gets duplicated in
    five modules and how a target NAME gets typed into library code. It is deliberately NOT
    :func:`~merlin.targetgen.capability_manifests.discovered_targets`: that set is targets shipping a
    capability residual, which is a different (and differently-sized) population. Ordered, so a report
    built from it is stable across runs.
    """
    root = merlin_dir() / "experiments" / "capsule_bench" / "targets"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


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


def source_experiment_env(target: str) -> list[str]:
    """Load ``targets/<target>/experiment.env`` into ``os.environ``, setting ONLY keys not already
    present, and return the keys it set.

    The harness does this for every arm before a run; a caller that grades a package WITHOUT going
    through the harness did not, and the difference is not cosmetic. One target's certifying tier is a
    program-driven Verilator sim registered through ``MERLIN_EXT_<TARGET>_VSIM``; with the variable
    absent there is no adapter, and its own profile spells out the consequence -- every capsule reports
    ``incomplete``, never a pass. Fail-closed is right, but reporting a whole suite incomplete because a
    path was not sourced is a tooling artifact wearing a verdict's clothes.

    The process environment always WINS, so an exported var is never overridden. Structured KEY=VALUE
    parse, ``#`` comments; target-agnostic (keyed off the descriptor's own directory). This module is the
    one allowed to know the ``experiments/`` layout.
    """
    import os

    desc = descriptor_path(target)
    f = desc.parent / "experiment.env"
    set_keys: list[str] = []
    if not f.is_file():
        return set_keys
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            set_keys.append(k)
    return set_keys


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


def perf_capsule_roots(target: str) -> list[Path]:
    """The roots holding ``target``'s PERFORMANCE capsules, or ``[]`` when it ships none.

    The library-facing form of :meth:`TargetExperiment.perf_roots`, resolved from a target NAME like
    its graded and hidden siblings here. Performance capsules are excluded from the functional suite by
    the underscore convention, so asking the graded roots for them reports that they do not exist --
    which reads as "this target has no performance families" and is how an optimization run comes to
    refuse on capsules that are sitting right there.
    """
    desc = descriptor_path(target)
    if not desc.is_file():
        return []
    try:
        from merlin.targetgen.target_experiment import load_target_experiment
        te = load_target_experiment(desc)
        if str(getattr(te, "target", "")) != target:      # an override naming a DIFFERENT target
            return []
        return [r for r in te.perf_roots() if r.is_dir()]
    except Exception:                                     # noqa: BLE001 — unreadable descriptor
        return []


def find_capsule(name: str) -> Path | None:
    """Locate a capsule directory by name across the corpus roots (first match wins)."""
    for root in capsule_corpus_roots():
        for cy in root.rglob("capsule.yaml"):
            if cy.parent.name == name:
                return cy.parent
    return None
