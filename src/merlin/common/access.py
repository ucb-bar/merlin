"""Declared agent-access identities, independent of source layout and optional imports.

This is policy data, not plugin discovery. A module keeps its withheld identity when its
implementation moves or is not installed. Filesystem masks, transcript audits and candidate import
checks consume the same declarations. Register a renamed module here *before* moving its source;
neither a missing file nor an unavailable research package grants access.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.machinery import all_suffixes
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class PythonSourceRoot:
    """One explicitly supported checkout location for an import namespace."""

    namespace: str
    path: str


PYTHON_SOURCE_ROOTS = (
    PythonSourceRoot("merlin", "merlin/python/merlin"),
    PythonSourceRoot("merlin", "src/merlin"),
    PythonSourceRoot("merlin", "packages/merlin-experiments/src/merlin"),
    PythonSourceRoot("merlin_experiments", "packages/merlin-experiments/src/merlin_experiments"),
    PythonSourceRoot("merlin", "packages/merlin-dse/src/merlin"),
    PythonSourceRoot("merlin_dse", "packages/merlin-dse/src/merlin_dse"),
    PythonSourceRoot("merlin", "packages/merlin-mining/src/merlin"),
    PythonSourceRoot("merlin_mining", "packages/merlin-mining/src/merlin_mining"),
    PythonSourceRoot("merlin", "packages/merlin-analysis/src/merlin"),
    PythonSourceRoot("merlin_analysis", "packages/merlin-analysis/src/merlin_analysis"),
)

# This is the existing public input grammar exception, NOT an exemption for the core or an extension.
PUBLIC_INPUT_MODULE = "merlin.xdsl_dialects.interface"


# The transcript audit and every consumer of its recorded evidence share one vocabulary.
# Advisory hits mean containment worked, or the content was owned/granted. Anything not
# explicitly advisory is a violation, including malformed records and future hit kinds.
AUDIT_ADVISORY_KINDS: frozenset[str] = frozenset(
    {"blocked_probe", "recon_probe", "owned_read", "granted_read", "pattern_mention"}
)
AUDIT_VIOLATION_KINDS: frozenset[str] = frozenset({"path_read", "oracle_use"})


def audit_hit_is_violation(hit: object) -> bool:
    """True iff this audit hit means withheld content actually reached the agent.

    FAIL CLOSED: anything that is not a well-formed hit carrying a kind from
    :data:`AUDIT_ADVISORY_KINDS` counts as a violation. A new hit kind is therefore disqualifying
    until it is deliberately declared advisory here -- never silently waved through.
    """
    if not isinstance(hit, dict):
        return True
    return hit.get("kind") not in AUDIT_ADVISORY_KINDS


def audit_token_in(text: str, tokens: tuple[str, ...]) -> str | None:
    """Match withheld paths by shape, not generic module stems inside prose.

    Qualified tokens retain substring matching; a bare stem must be a path
    component, optionally followed by an extension. Callers parse command/source
    roles before supplying individual path or module spellings.
    """
    lowered = text.lower()
    components = [part for part in lowered.replace("\\", "/").split("/") if part]
    for token in tokens:
        if "/" in token or "." in token:
            if token in lowered:
                return token
        elif any(part == token or part.startswith(token + ".") for part in components):
            return token
    return None


def module_matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def is_harness_module(module: str) -> bool:
    return any(module_matches(module, source.namespace) for source in PYTHON_SOURCE_ROOTS)


def is_public_input_module(module: str) -> bool:
    return module_matches(module, PUBLIC_INPUT_MODULE)


def module_name_for(rel_path: str) -> str | None:
    """Translate a declared checkout path to its real import name, without importing it.

    Reject absolute/escaping paths and unrelated distributions. Package ``__init__.py`` names its
    package; source directories name module prefixes. Existence is deliberately irrelevant.
    """
    path = PurePosixPath(str(rel_path).strip())
    if path.is_absolute() or ".." in path.parts:
        return None
    for source in PYTHON_SOURCE_ROOTS:
        try:
            tail = path.relative_to(source.path)
        except ValueError:
            continue
        parts = list(tail.parts)
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts and parts[-1] == "__init__":
            parts.pop()
        if not all(part.isidentifier() for part in parts):
            return None
        return ".".join((source.namespace, *parts))
    return None


@dataclass(frozen=True)
class ModuleAccess:
    """One withheld logical module and all reviewed import spellings of its implementation."""

    identity: str
    origin: str
    modules: tuple[str, ...]
    directory: bool = False


def _module(module: str, origin: str, *, directory: bool = False, aliases: tuple[str, ...] = ()) -> ModuleAccess:
    # The staged evaluation split retains the targetgen tail; aliases are DENIES, never new grants.
    modules = (module, *aliases)
    modules += tuple(
        "merlin_experiments.evaluation." + name.removeprefix("merlin.targetgen.")
        for name in modules
        if name.startswith("merlin.targetgen.")
    )
    return ModuleAccess(module, origin, modules, directory)


MODULE_ACCESS = (
    _module("merlin.perf.analysis_worker", "grader"),
    _module("merlin.perf.isolated_probe_provider", "grader"),
    _module("merlin.perf.controlled_context_provider", "grader"),
    _module("merlin.perf.paired_context_provider", "grader"),
    _module("merlin.perf.host_region_qualifier", "grader"),
    _module("merlin.perf.host_physical_transition_qualifier", "grader"),
    _module("merlin.perf.lane_migration_qualifier", "grader"),
    _module("merlin.perf.source_contraction_preparation", "grader"),
    _module("merlin.perf.source_convolution_preparation", "grader"),
    _module("merlin.perf.source_program_pair", "grader"),
    _module("merlin.perf.source_initializer_elision", "grader"),
    _module("merlin.perf.source_program_pair_provider", "grader"),
    _module("merlin_experiments.execution", "grader", directory=True),
    _module("merlin_experiments.measured_launch", "grader"),
    _module("merlin_experiments.frozen_python", "grader", aliases=("perf_frozen_python",)),
    _module("merlin_experiments.source_snapshot", "grader", aliases=("perf_snapshot",)),
    _module("merlin_experiments.corpus.admission", "grader"),
    _module("merlin_experiments.corpus.numeric_policy", "grader"),
    _module("merlin_experiments.phase0", "grader", directory=True),
    _module("merlin_experiments.phase2", "grader", directory=True),
    _module("merlin_experiments.phase1.run_inputs", "grader"),
    _module("merlin_experiments.phase1.treatments", "grader"),
    _module("merlin_experiments.phase1.corpus_inputs", "grader"),
    _module("merlin_experiments.phase1.source_inputs", "grader"),
    _module("merlin_experiments.phase1.providers", "grader", directory=True),
    _module("merlin_experiments.phase1.brokers", "grader", directory=True),
    _module("merlin_experiments.phase1.feedback", "grader", directory=True),
    _module("merlin_experiments.phase1.telemetry", "grader", directory=True),
    _module("merlin_experiments.phase1.session", "grader"),
    _module("merlin_experiments.phase1.authoring", "grader"),
    _module("merlin_experiments.phase1.audit", "grader"),
    _module("merlin_experiments.phase1.runtime_environment", "grader"),
    _module("merlin_experiments.phase1.controller", "grader"),
    _module("merlin_experiments.phase1.__main__", "grader"),
    _module("merlin_experiments.phase1.task_staging", "grader"),
    _module("merlin_experiments.phase1.workspace_transport", "grader"),
    _module("merlin_experiments.phase1.recovery", "grader"),
    _module("merlin_experiments.phase1.conformance", "grader"),
    _module("merlin_experiments.corpus.release", "grader"),
    _module("merlin_experiments.corpus.preparation", "grader"),
    _module("merlin.runtime.reference", "oracle"),
    _module("merlin.runtime.simulator", "oracle"),
    _module("merlin.runtime.backends", "oracle", directory=True),
    _module(
        "merlin.targetgen.program_oracle",
        "oracle",
        aliases=("merlin.targetgen.program_values", "merlin.targetgen.program_engine_policy"),
    ),
    _module("merlin.targetgen.muon_oracles", "oracle"),  # historical identity; now an OOT adapter
    # Historical names remain denied after their Gemmini implementation moves OOT.
    _module("merlin.targetgen.eval.gemmini_conformance", "grader"),
    _module("merlin.targetgen.eval.gemmini_suite", "grader"),
    _module("merlin.targetgen.eval.gemmini_dispatcher", "grader"),
    _module("merlin.targetgen.agent.gemmini_kernel_slot", "grader"),
    _module("gemmini_conformance", "grader", directory=True),
    _module("merlin.targetgen.oracle_helpers.npu_emit", "oracle"),  # historical support identity
    _module("atlas_program_emit", "oracle"),
    _module("merlin.targetgen.heavy_oracles", "oracle"),
    _module("merlin.targetgen.rtl.mlc_bridge", "oracle"),
    _module("merlin.targetgen.rocc.decode", "grader"),
    _module("merlin.targetgen.trace_check", "grader"),
    _module("merlin.targetgen.capsule_grade", "grader"),
    _module(
        "merlin.targetgen.capsule_golden",
        "grader",
        aliases=("merlin.targetgen.golden_provenance", "merlin.targetgen.capsule_inputs"),
    ),
    _module("merlin.targetgen.numeric_falsifiability", "grader"),
    _module("merlin.targetgen.rtl.gen_rocc_replay", "grader"),
    _module("merlin.verify.replay", "grader"),
    _module("merlin.verify.replay_layers", "grader"),
    _module("merlin.targetgen.model_slice_export", "grader"),
    _module("merlin.targetgen.group_capsules", "grader"),
    _module("merlin.targetgen.store_probe", "grader"),
    _module("merlin.targetgen.evaluation_cohort", "grader"),
    _module(
        "merlin.targetgen.capsule_runner",
        "grader",
        aliases=("merlin.targetgen.oracle_policy", "merlin.targetgen._capsule_bundle_worker"),
    ),
    _module("merlin.targetgen.capsule_dram", "grader"),
    _module(
        "merlin.targetgen.oot_runner",
        "grader",
        aliases=("merlin.targetgen.package_runtime", "merlin.targetgen.package_certification"),
    ),
    _module("merlin.targetgen.coverage_report", "grader"),
    _module("merlin.targetgen.eval", "grader", directory=True),
)


def declared_modules(origin: str) -> tuple[str, ...]:
    """Declared identity does not disappear when the corresponding optional source is absent."""
    return tuple(dict.fromkeys(module for item in MODULE_ACCESS if item.origin == origin for module in item.modules))


def module_paths(item: ModuleAccess) -> tuple[str, ...]:
    """All supported source locations, including both sides of a staged compatibility cutover."""
    paths: list[str] = []
    for module in item.modules:
        for source in PYTHON_SOURCE_ROOTS:
            if not module_matches(module, source.namespace):
                continue
            tail = module[len(source.namespace) :].lstrip(".").replace(".", "/")
            base = str(PurePosixPath(source.path) / tail)
            # A module may become a package without changing its import identity.
            paths.extend((base,) if item.directory else (base + ".py", base))
    return tuple(dict.fromkeys(paths))


def runtime_package_roots(root: Path) -> tuple[tuple[str, Path], ...]:
    """Installed/active harness package roots without importing any oracle or grader.

    A broad Python-toolchain bind exposes installed files even when a checkout shadows their
    imports. Inspect every active path and the checkout's bound venv, not just the winning import.
    Already-loaded namespace paths include editable contributions outside the checkout. Preserve
    lexical AND resolved paths because a borrowed venv is rebound at both destinations.
    """
    namespaces = tuple(dict.fromkeys(source.namespace for source in PYTHON_SOURCE_ROOTS))
    search = [Path(entry or ".").absolute() for entry in sys.path]
    for environment in (root / ".venv", Path(sys.prefix)):
        for library in ("lib", "lib64"):
            search.extend((environment / library).glob("python*/site-packages"))
        search.append(environment / "Lib/site-packages")
    candidates = [(namespace, path / namespace) for path in search for namespace in namespaces]
    # Setuptools editable namespaces include a synthetic __path__ entry. Read the ALREADY LOADED
    # finder's declarative mapping instead of executing a path hook or importing a private leaf.
    editable = {
        data["PATH_PLACEHOLDER"]: data
        for loaded in tuple(sys.modules.values())
        if loaded is not None
        and isinstance((data := vars(loaded)).get("PATH_PLACEHOLDER"), str)
        and isinstance(data.get("MAPPING"), dict)
        and isinstance(data.get("NAMESPACES"), dict)
    }
    for entry in sys.path:
        if entry in editable:
            data = editable[entry]
            for name, location in data["MAPPING"].items():
                if any(module_matches(name, namespace) for namespace in namespaces):
                    candidates.append((name, Path(location)))
            for name, locations in data["NAMESPACES"].items():
                if any(module_matches(name, namespace) for namespace in namespaces):
                    candidates.extend((name, Path(location)) for location in locations)
    for namespace in namespaces:
        loaded = sys.modules.get(namespace)
        for entry in vars(loaded).get("__path__", ()) if loaded is not None else ():
            if entry in editable:
                data = editable[entry]
                for name, location in data["MAPPING"].items():
                    if module_matches(name, namespace):
                        candidates.append((name, Path(location)))
                for name, locations in data["NAMESPACES"].items():
                    if module_matches(name, namespace):
                        candidates.extend((name, Path(location)) for location in locations)
                continue
            path = Path(entry).absolute()
            if not path.is_dir():
                raise RuntimeError(f"cannot mask non-filesystem harness namespace {namespace}: {path}")
            candidates.append((namespace, path))
    locations: dict[tuple[str, Path], None] = {}
    for namespace, path in candidates:
        if path.is_dir() or any(Path(str(path) + suffix).is_file() for suffix in all_suffixes()):
            locations[namespace, path.absolute()] = None
            locations[namespace, path.resolve()] = None
    return tuple(locations)


def _implementation_paths(base: Path, *, directory: bool) -> tuple[Path, ...]:
    paths = [base]
    if not directory:
        paths.extend(Path(str(base) + suffix) for suffix in all_suffixes())
        # Mask compiled copies as well as source: a .pyc preserves the answer-bearing program.
        paths.extend((base.parent / "__pycache__").glob(base.name + ".*.pyc"))
    return tuple(path for path in paths if path.exists())


def module_locations(root: Path, item: ModuleAccess) -> tuple[Path, ...]:
    """All checkout, installed and compiled copies of a withheld identity, not just first import."""
    roots = [(source.namespace, root / source.path) for source in PYTHON_SOURCE_ROOTS]
    roots.extend(runtime_package_roots(root))
    paths: dict[Path, None] = {}
    for module in item.modules:
        for namespace, package in roots:
            if module_matches(module, namespace):
                tail = module[len(namespace) :].lstrip(".").split(".")
                for path in _implementation_paths(package.joinpath(*tail), directory=item.directory):
                    paths[path] = None
    return tuple(paths)


def unresolved_modules(root: Path) -> tuple[ModuleAccess, ...]:
    """Explicit migration diagnostics, not a claim that missing declarations are safe to read.

    Core-only installs and historically evicted OOT adapters legitimately lack some implementations.
    Callers validating a migration compare these identities before/after, rather than accepting an
    empty filesystem coverage result as evidence that everything was classified.
    """
    return tuple(item for item in MODULE_ACCESS if not module_locations(root, item))


def legacy_module_paths(origin: str) -> tuple[str, ...]:
    """Compatibility view for callers still persisting the legacy repo-relative grant format."""
    return tuple(
        "merlin/python/" + item.identity.replace(".", "/") + ("" if item.directory else ".py")
        for item in MODULE_ACCESS
        if item.origin == origin
    )


# Resource ownership and physical layout are distinct. These aliases are consulted together during
# staged moves so a stale duplicate answer file cannot survive merely because the new copy was found.
CONTRACT_RESOURCE_ROOTS = (
    "merlin/contract",
    "src/merlin/_data/contract",
    "packages/merlin-experiments/src/merlin_experiments/_data/contract",
)


def contract_resource_roots(root: Path, *parts: str) -> tuple[Path, ...]:
    paths = [root.joinpath(rel, *parts) for rel in CONTRACT_RESOURCE_ROOTS]
    paths.extend(
        package.joinpath("_data/contract", *parts)
        for namespace, package in runtime_package_roots(root)
        if namespace in {"merlin", "merlin_experiments"}
    )
    return tuple(dict.fromkeys(paths))


# Persistent answer-key identity is core metadata: importing an optional scorer to discover it would
# expose existing keys whenever that scorer is uninstalled. The producer re-exports these constants.
KEY_FILENAME = "recovery_key.yaml"
KEY_ENV = "MERLIN_RECOVERY_KEY"
KEY_TOPIC = "perf-recovery"
KEY_TOPIC_FOLDED = ("perf-studies", "recovery")
