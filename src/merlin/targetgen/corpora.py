"""The single place that resolves corpus locations the LIBRARY reads (reference-by-location).

No library module should hardcode a corpus path; they call :func:`capsule_corpus_roots` /
:func:`find_capsule` / :func:`kernel_corpus_root` instead. The locations themselves are DATA — the corpus
registry ``src/merlin/resources/contracts/corpora.yaml`` — and this module is its one reader, so adding or moving a corpus
is a registry edit. It is also the ONE sanctioned indirection to a corpus that still lives under
``experiments/`` (the boundary lint allowlists this module).

Capsule inputs are historical read-only diagnostics, kept under ``experiments/``
and excluded from installed distributions. Expert kernel corpora are external
framework checkouts keyed by source spelling, layout, and selected owner.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from merlin.common.paths import checkout_root, data_path, repo_root, resolve_grant

_REGISTRY = ("contract", "corpora.yaml")  # checkout resource or bundled core data


@cache
def _load_registry(path: Path) -> dict:
    from merlin.common.yaml import load_yaml

    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"corpus registry {path} is not a mapping")
    return data


def _registry() -> dict:
    """The corpus registry. A missing or malformed one RAISES: read as empty, it would make every corpus
    reader report "nothing to screen", which looks exactly like a clean result."""
    return _load_registry(data_path(*_REGISTRY))


def capsule_corpus_roots(*, owner_root: Path | None = None) -> list[Path]:
    """Existing historical capsule-corpus roots for read-only diagnostics.

    New grading uses a descriptor-selected, reviewed release; callers must not
    treat this broad registry as an implicit corpus for a target.

    ``owner_root`` selects an explicit source checkout for historical-corpus
    protection when a process is otherwise rooted in an external experiment
    workspace. The registry remains mandatory and is read from that owner.
    """
    if owner_root is None:
        registry = _registry()
        selected = os.environ.get("MERLIN_REPO_ROOT") or checkout_root()
        if selected is None:
            return []
        root = Path(selected)
    else:
        root = Path(owner_root)
        registry = _load_registry(resolve_grant("merlin/contract/corpora.yaml", root=root))
    roots = [resolve_grant(str(rel), root=root) for rel in registry.get("capsule_corpora") or []]
    return [r for r in roots if r.is_dir()]


def kernel_corpora() -> dict[str, dict]:
    """Every expert kernel corpus the registry declares, ``{framework: spec}``, in declaration order."""
    return {str(k): dict(v or {}) for k, v in (_registry().get("kernel_corpora") or {}).items()}


def sched_corpora() -> dict[str, dict]:
    """Every corpus the scheduling-IR expressiveness measurement may read, ``{name: spec}``.

    Separate from :func:`kernel_corpora` because the consumer is: the mining layer BUILDS a corpus,
    this one only reads a corpus already on disk and asks whether it can stand as expert evidence.
    Each spec names a ``pin`` in the hardware-pin registry rather than a path, so an unresolvable
    corpus can report the ENVIRONMENT VARIABLE it needs instead of a zero denominator.
    """
    return {str(k): dict(v or {}) for k, v in (_registry().get("sched_corpora") or {}).items()}


def sched_corpus_record(spec: dict) -> Path | None:
    """The project-relative eligibility record a ``sched_corpora`` spec declares, or None."""
    rel = str(spec.get("provenance_record") or "")
    return resolve_grant(rel) if rel else None


def kernel_corpus_for_source(source: str | None) -> str | None:
    """The corpus (framework name) a ``kernel.source`` spelling names, matched case-insensitively, or None."""
    want = (source or "").lower()
    if not want:
        return None
    for name, spec in kernel_corpora().items():
        if want == name.lower() or want in {str(s).lower() for s in spec.get("sources") or []}:
            return name
    return None


def kernel_corpus_target(name: str | None) -> str | None:
    """The target that owns corpus ``name``'s kernel-ceiling bench config (its registry ``target:``), or
    None when the corpus is unregistered or declares no owner."""
    spec = kernel_corpora().get(name or "")
    owner = str((spec or {}).get("target") or "").strip()
    return owner or None


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
    root = repo_root() / "experiments" / "studies" / "capsule_bench" / "targets"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def descriptor_paths(target: str = "*") -> list[Path]:
    """Retained experiment descriptors, in the same deterministic owner order."""
    root = repo_root()
    patterns = (
        "examples/*/{target}/target/descriptor.yaml",
        "experiments/studies/*/targets/{target}/target_experiment.yaml",
        "experiments/inputs/target-inputs/{target}/target_experiment.yaml",
    )
    return [path for pattern in patterns for path in sorted(root.glob(pattern.format(target=target))) if path.is_file()]


def descriptor_path(target: str) -> Path:
    """Where ``target``'s experiment descriptor lives — the per-target convention path, or whatever
    ``MERLIN_TARGET_EXPERIMENT`` overrides it to. Lives here because this module is the one allowed to
    know the ``experiments/`` layout; callers elsewhere took the convention path directly and silently
    ignored the override, so an out-of-tree descriptor resolved for some readers and not others."""
    import os

    override = os.environ.get("MERLIN_TARGET_EXPERIMENT", "").strip()
    if override:
        return Path(override)
    return standard_descriptor_path(target)


def standard_descriptor_path(target: str) -> Path:
    """The CONVENTION path for ``target``'s descriptor, ignoring ``MERLIN_TARGET_EXPERIMENT``.

    Separate from :func:`descriptor_path` because the override names ONE descriptor, not a mapping: a
    caller asking for target *X* while the variable points at *Y* must not be handed *Y*. A caller that
    needs the override validated against the target it asked for reads the variable itself and falls
    back to this. Both spellings live here so the ``experiments/`` layout has exactly one owner.
    """
    from merlin.common.paths import repo_root
    from merlin.targetgen.target_experiment import _declared_target

    for candidate in descriptor_paths():
        if _declared_target(candidate) == target:
            return candidate
    return repo_root() / "experiments" / "studies" / "capsule_bench" / "targets" / target / "target_experiment.yaml"


def source_experiment_env(target: str | None = None, *, descriptor: str | Path | None = None) -> list[str]:
    """Load ``targets/<target>/experiment.env`` into ``os.environ``, setting ONLY keys not already
    present, and return the keys it set.

    The harness does this for every arm before a run; a caller that grades a package WITHOUT going
    through the harness did not, and the difference is not cosmetic. One target's certifying tier is a
    program-driven Verilator sim registered through ``MERLIN_EXT_<TARGET>_VSIM``; with the variable
    absent there is no adapter, and its own profile spells out the consequence -- every capsule reports
    ``incomplete``, never a pass. Fail-closed is right, but reporting a whole suite incomplete because a
    path was not sourced is a tooling artifact wearing a verdict's clothes.

    The process environment always WINS, so an exported var is never overridden. Structured KEY=VALUE
    parse, ``#`` comments; target-agnostic (keyed off declared resource ownership, defaulting to
    descriptor siblings). An explicit descriptor lets invocation setup
    source tooling before parsing the target identity, without guessing an in-tree location.
    """
    import os

    if descriptor is None:
        if target is None:
            raise ValueError("source_experiment_env requires a target or explicit descriptor")
        desc = descriptor_path(target)
    else:
        desc = Path(descriptor)
    from .target_experiment import descriptor_resources_root

    f = descriptor_resources_root(desc) / "experiment.env"
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
    except Exception:  # noqa: BLE001 — unreadable descriptor
        return None
    return te if str(getattr(te, "target", "")) == target else None


def graded_capsule_roots(target: str, *, hidden: bool = False) -> list[Path]:
    """The roots that make up ``target``'s descriptor-selected GRADED suite.

    Missing, mismatched, unreadable or empty descriptors return no roots: a
    cross-target historical corpus is never an implicit grading default.

    ``hidden=True`` returns the holdout roots instead. They are a SEPARATE tree, deliberately excluded
    from the public roots (``corpus_siblings`` skips ``hidden/``), so a hidden grade that reused the
    public roots would match nothing and report a 0/0 "pass" that never ran. Empty is a real answer for a
    target that ships no holdouts, and is returned as empty rather than papered over with a fallback.

    A target's suite is not one directory: the capsules are split by kind into sibling categories
    (``isa`` / ``layers`` / ``model`` / ``model_slices``), and different targets keep those siblings in
    different places -- one target's sit at the corpus root, another's under ``<target>/``. Passing their common
    parent is NOT the fix, because that parent holds every target's corpus at once; see the warning in
    :func:`merlin.targetgen.capsule_common.discover_capsules`.

    Measured consequence of getting this wrong: grading the gemmini package against
    ``experiments/corpora`` pulled in 173 capsules from seven targets, marked 89 of them "outside
    this target's declared capability", and reported ``1/84`` -- a number that reads like a catastrophic
    regression and means nothing. The target's own suite is 36.

    ``TargetExperiment.graded_roots()`` is the resolution the A/B launchers and ``readiness_check.py``
    already use; this exposes it to library callers that have only a target NAME. Honours
    ``MERLIN_TARGET_EXPERIMENT`` (the same override ``capsule_bench/harness/_common.py`` reads) so a
    target whose descriptor lives out of tree still resolves.
    """
    desc = descriptor_path(target)
    if not desc.is_file():
        return []
    try:
        from merlin.targetgen.target_experiment import load_target_experiment

        te = load_target_experiment(desc)
        if str(getattr(te, "target", "")) != target:  # an override naming a DIFFERENT target
            return []
        roots = [r for r in (te.hidden_roots() if hidden else te.graded_roots()) if r.is_dir()]
    except Exception:  # noqa: BLE001 — unreadable descriptor
        return []
    if hidden:
        return roots  # empty == this target ships no holdouts
    return roots


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
        if str(getattr(te, "target", "")) != target:  # an override naming a DIFFERENT target
            return []
        return [r for r in te.perf_roots() if r.is_dir()]
    except Exception:  # noqa: BLE001 — unreadable descriptor
        return []


def find_capsule(name: str) -> Path | None:
    """Locate a capsule directory by name across the corpus roots (first match wins)."""
    for root in capsule_corpus_roots():
        for cy in root.rglob("capsule.yaml"):
            if cy.parent.name == name:
                return cy.parent
    return None
