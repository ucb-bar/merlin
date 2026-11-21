"""The ``plugin`` block of a target contract: what its keys mean, and whether they point anywhere.

A target package extends Merlin by DECLARING things in its contract's ``plugin`` block rather than by
being named anywhere in core. One such key is already load-bearing (``backend``, consumed by
:mod:`merlin.runtime.backends.registry`); others are pointers to a package's own prototype artifacts.

This module exists because the block had no schema and no validation, and that combination is worse than
it sounds. Two failure modes it allows, both observed:

* **A reader cannot tell a live seam from a pointer.** Two shipped contracts declare ``dialect_module``
  and ``lowering_entrypoint``. Nothing consumes them — which is *correct*, because they name
  feasibility prototypes that say so in their own docstrings and are deliberately off the grading path.
  But nothing said so at the declaration site, so "unconsumed" reads as "broken seam", and the obvious
  repair — wiring them into the dialect loader — would load a module with no ``SPEC_OPS`` and fail.
* **A typo is silent.** ``backend`` misspelled is not a broken backend, it is *no* backend, discovered
  as a missing feature much later.

So: keys are declared here with their meaning and their status, unknown keys are rejected, and every
reference is checked to point at a file that exists. Validation does not require a consumer — a pointer
that has rotted is worth catching whether or not anything loads it.

Resolution never mutates ``sys.path``. The package root is registered as a synthetic namespace package
and submodules are imported through it, so two packages that both ship a ``lowering.py`` cannot shadow
each other, and a package's own relative imports still resolve. That is the same discipline
:mod:`merlin.runtime.backends.registry` uses for ``plugin.backend``, generalised so every future key
(oracles, suites, cost models) gets it for free instead of re-implementing it.
"""

from __future__ import annotations

import ast
import importlib
import importlib.machinery
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ---- plugins -----------------------------------------------------------------------------------

#: Where synthetic per-package namespaces live. Never a real package on disk.
_NAMESPACE_ROOT = "merlin._oot_targets"

#: Separator between a module reference and an attribute within it (``pkg.mod:attr``).
ATTR_SEP = ":"


@dataclass(frozen=True)
class PluginKey:
    """One recognised key of the ``plugin`` block."""

    name: str
    summary: str
    #: True when core actually loads it. False means the package declares a pointer for humans and
    #: tooling — recorded so "nothing consumes this" is never mistaken for "this is broken".
    consumed: bool
    #: What the reference must resolve to: a module, an attribute inside one, or a file/directory.
    expects: str


PLUGIN_KEYS: dict[str, PluginKey] = {
    "extension": PluginKey(
        "extension",
        "Versioned operation extension declaring offers and prepared-program lowering; "
        "loaded explicitly from its selected support provider.",
        consumed=True,
        expects="path",
    ),
    "rvv_codegen": PluginKey(
        "rvv_codegen",
        "Provider-owned vector compiler interface; loaded explicitly for vector lowering, "
        "without registering a runtime backend or changing scalar compilation.",
        consumed=True,
        expects="path",
    ),
    "matrix_lowering": PluginKey(
        "matrix_lowering",
        "Explicit support-provider module for matrix geometry, routing policy, prepared-IR "
        "rewriting, signature loading and object building. Unit and configuration are supplied "
        "separately from the support target; no native implementation is selected implicitly.",
        consumed=True,
        expects="path",
    ),
    "backend": PluginKey(
        "backend",
        "A runtime backend module (or package directory) that self-registers via "
        "runtime.backends.base.register(). The only fully live seam in the repo.",
        consumed=True,
        expects="path",
    ),
    "spatial_introspect": PluginKey(
        "spatial_introspect",
        "A spatial tensor-tile RTL reader exposing ARC_TARGETS, build_fact_bundle(), render_fact_bundle() "
        "and spatial_facts(). rtl.mlc_bridge registers every declared one and serves the arc models it "
        "names, so core names no spatial unit.",
        consumed=True,
        expects="path",
    ),
    "simt_introspect": PluginKey(
        "simt_introspect",
        "The SIMT RTL introspect this package serves, as `<plugin.backend>:<attribute>` -- an attribute of "
        "the package's own registered backend exposing TARGET + build_facts(). rtl.mlc_bridge registers it "
        "when no introspect is registered, so core names no SIMT target and nothing is imported twice.",
        consumed=True,
        expects="attr",
    ),
    "reference_programs": PluginKey(
        "reference_programs",
        "The package's own reference-program corroboration tool: it builds the target's upstream "
        "reference programs with its backend's toolchain, runs them on its simulators and compares the "
        "output to the Tensor goldens. Loaded by load_declared (capsule-bench preflight, experiment "
        "drivers). A target that declares none has no such table; it never borrows another target's.",
        consumed=True,
        expects="path",
    ),
    "dialect": PluginKey(
        "dialect",
        "The package's own TARGET-DIALECT module (or package directory): a module that calls "
        "xdsl_dialects.lowering.target_lowering.register_dialect_spec() at import, contributing its "
        "TargetSpec and its target-op -> opcode map. Loaded by target_lowering._ensure_dialects_discovered "
        "through the same plugin discovery the backends use. Distinct from `dialect_module` below, which "
        "is a pointer to a feasibility prototype nothing loads.",
        consumed=True,
        expects="path",
    ),
    "sim_oracle": PluginKey(
        "sim_oracle",
        "The bespoke-simulator ORACLE this package contributes, as a module that calls "
        "capsule_runner.register_sim_oracle() at import. Loaded by "
        "capsule_runner._ensure_sim_oracles_discovered; a target that declares none is graded by the "
        "generic arc/program oracle and never borrows another target's simulator.",
        consumed=True,
        expects="path",
    ),
    "sim_oracle_metadata": PluginKey(
        "sim_oracle_metadata",
        "Core-only oracle metadata registration module. Imports oracle_policy and registers tier_plan= "
        "without importing evaluation or creating adapters. May name the same module as sim_oracle "
        "when that module is import-light. Loaded only from trusted support providers.",
        consumed=True,
        expects="path",
    ),
    "rocc_semantics": PluginKey(
        "rocc_semantics",
        "The host-side RoCC ABI module (isa_constants / decode_instruction / instruction_funct, each "
        "taking the target) that a GENERIC backend serves as its `rocc_semantics` attribute. Usually a "
        "core module reference; read by the generic chipyard RoCC backend, never imported by name.",
        consumed=True,
        expects="path",
    ),
    "rtl_checks": PluginKey(
        "rtl_checks",
        "The structural RTL-check provider (load_default_facts / project_facts / screen / "
        "compile_trace_checks / render_trace) a generic backend exposes as `rocc_semantics.rtl_checks`.",
        consumed=True,
        expects="path",
    ),
    "isa_headers": PluginKey(
        "isa_headers",
        "A DATA resource of the provider: the ISA-headers spec (schema merlin.isa_headers.v1) naming "
        "the upstream header checkout by pin and content digest, its include roots, CRT, link script and "
        "flags. The bytes are referenced, never copied into the provider.",
        consumed=True,
        expects="file",
    ),
    "path": PluginKey(
        "path",
        "Injected by target_registry for external packages — the package root. Not authored by hand.",
        consumed=True,
        expects="path",
    ),
    # The two pointer keys below are declared by shipped contracts and loaded by nothing, ON PURPOSE.
    # They name feasibility prototypes that say so in their own docstrings, sit off the grading path,
    # and are written as FLAT modules meant to be run directly (`python .../lowering.py`) rather than
    # imported: they have no __init__.py and import their siblings absolutely. `load_module` therefore
    # cannot import them without putting their directory on sys.path, which is the shadowing hazard this
    # module exists to avoid — so it refuses, and that refusal is correct. Stated here so the next reader
    # does not "repair" the loader, or wire these into the dialect loader (they expose no SPEC_OPS and
    # would fail), or convert them into packages (which would break their documented direct execution).
    "dialect_module": PluginKey(
        "dialect_module",
        "Pointer to the package's own MLIR/SIMT dialect prototype. NOT the staged pipeline's target "
        "dialect — that is the package's dialect.py.",
        consumed=False,
        expects="module",
    ),
    "lowering_entrypoint": PluginKey(
        "lowering_entrypoint",
        "Pointer to the package's own lowering demonstration, as `module:callable`. NOT the lowering "
        "tables the staged pipeline reads from lowering.yaml.",
        consumed=False,
        expects="attr",
    ),
}


class PluginError(ValueError):
    """A plugin block that cannot be trusted: an unknown key, or a reference that points nowhere."""


def validate(plugin: dict[str, Any] | None, *, root: Path | None = None, where: str = "plugin") -> list[str]:
    """Problems with a ``plugin`` block. Empty list means it is coherent.

    Checks two things and deliberately not a third. It checks that every key is recognised (an
    unrecognised key is a typo or an invention, and either way nothing will honour it), and that every
    reference resolves to a file that exists when ``root`` is given. It does NOT check that a key has a
    consumer, because some keys are pointers by design — that fact belongs in :data:`PLUGIN_KEYS`, where
    it is stated once, rather than being re-derived by whoever next greps for callers.
    """
    problems: list[str] = []
    for key, value in (plugin or {}).items():
        spec = PLUGIN_KEYS.get(key)
        if spec is None:
            problems.append(
                f"{where}.{key}: unrecognised plugin key (known: {sorted(PLUGIN_KEYS)}). Nothing will "
                "honour it, and an unknown key is silently ignored — which is how a misspelled "
                "'backend' becomes no backend at all."
            )
            continue
        if root is None or not isinstance(value, str) or not value:
            continue
        if key == "path":
            try:
                provider_root(root, value)
            except PluginError as exc:
                problems.append(f"{where}.path: {exc}")
            continue
        target = resolve_reference(Path(root), value, spec.expects)
        if target is None:
            problems.append(
                f"{where}.{key}: {value!r} does not resolve to a file under {root}"
                + ("" if spec.expects == "file" else " nor to a module of Merlin's installed core")
            )
    return problems


#: The installed core package. A plugin reference that names a module UNDER it (``merlin.<...>``) and
#: that the provider does not ship itself is a reference to GENERIC core code: a data-only provider
#: declares which generic implementation serves it instead of carrying a copy of one.
CORE_PACKAGE = "merlin"
#: Synthetic namespaces this module and the backend registry mint; never a core module.
_SYNTHETIC_PREFIXES = ("merlin._oot",)


def core_module_path(reference: str, expects: str = "module") -> Path | None:
    """The source file of a dotted reference to a module of Merlin's installed core, or None.

    Only a fully dotted ``merlin.<pkg>...<module>`` spelling qualifies (never a path, never a
    synthetic plugin namespace), and the file found must be a real ``.py`` under the active core's
    import root. Parent packages are imported by the lookup, which is safe: they are core code.
    """
    if expects == "file":
        return None
    module_ref = reference.split(ATTR_SEP, 1)[0] if expects == "attr" else reference
    parts = module_ref.split(".")
    if (
        len(parts) < 2
        or parts[0] != CORE_PACKAGE
        or not all(part.isidentifier() for part in parts)
        or any(module_ref.startswith(prefix) for prefix in _SYNTHETIC_PREFIXES)
    ):
        return None
    from merlin.common.paths import python_source_dir

    try:
        spec = importlib.util.find_spec(module_ref)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.has_location:
        return None
    try:
        origin = Path(spec.origin).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return origin if is_core_module_source(origin) else None


def _core_package_dirs() -> tuple[Path, ...]:
    """The installed distributions' ``merlin`` package directories (core first).

    The namespace also spans the checkout's ``merlin/`` data tree (tests, contracts); that is not a
    distribution, so only the active core and sibling ``src``/``site-packages`` roots count.
    """
    from merlin.common.paths import python_source_dir

    core = (python_source_dir() / CORE_PACKAGE).resolve()
    found = [core]
    package = sys.modules.get(CORE_PACKAGE)
    for entry in getattr(package, "__path__", ()) or ():
        path = Path(entry).resolve()
        if path not in found and path.name == CORE_PACKAGE and path.parent.name in {"src", "site-packages"}:
            found.append(path)
    return tuple(found)


def is_core_module_source(path: str | Path) -> bool:
    """Whether ``path`` is the ``.py`` source of a module of an installed Merlin distribution."""
    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    if resolved.suffix != ".py" or not resolved.is_file():
        return False
    return any(resolved.is_relative_to(root) for root in _core_package_dirs())


def resolve_reference(root: Path, reference: str, expects: str) -> Path | None:
    """A provider-owned reference first; otherwise a dotted reference to a core module.

    Provider precedence keeps every existing package's resolution unchanged. The core fallback lets a
    data-only provider name the generic implementation that serves it (``plugin.backend:
    merlin.runtime.backends.<generic>``) instead of shipping target code.
    """
    owned = _reference_path(Path(root), reference, expects)
    if owned is not None:
        return owned
    return core_module_path(reference, expects)


def provider_root(root: str | Path, declared_root: str | Path | None = None) -> Path:
    """Canonical selected provider authority; plugin metadata cannot redirect it."""
    try:
        canonical = Path(root).resolve(strict=True)
        if not canonical.is_dir():
            raise ValueError("provider root is not a directory")
        if declared_root and Path(declared_root).resolve(strict=True) != canonical:
            raise ValueError("plugin.path differs from the selected provider root")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PluginError(f"invalid selected provider root: {exc}") from exc
    return canonical


def _reference_path(root: Path, reference: str, expects: str) -> Path | None:
    """An owned reference, or None for missing/escaping paths and package initializers.

    Accepts the three spellings a contract may legitimately use: a path relative to the package root
    (``backend.py``, or a directory), a dotted module (``pkg.mod``), and a dotted module with an
    attribute (``pkg.mod:fn``).
    """
    module_ref = reference.split(ATTR_SEP, 1)[0] if expects == "attr" else reference
    relative = Path(module_ref)
    if not module_ref or relative.is_absolute() or ".." in relative.parts:
        return None
    try:
        canonical = provider_root(root)
        root = canonical
        candidates = [root / relative]
        if "/" not in module_ref and not module_ref.endswith(".py"):
            parts = module_ref.split(".")
            if all(part.isidentifier() for part in parts):
                candidates += [root.joinpath(*parts).with_suffix(".py"), root.joinpath(*parts) / "__init__.py"]
        for candidate in candidates:
            if not candidate.exists():
                continue
            if not candidate.resolve(strict=True).is_relative_to(canonical):
                return None
            # Python executes package initializers before the leaf. A contained
            # leaf cannot license an escaping parent __init__.py or directory.
            directory = candidate if candidate.is_dir() else candidate.parent
            while True:
                if not directory.resolve(strict=True).is_relative_to(canonical):
                    return None
                if directory != root or candidate == root:
                    init = directory / "__init__.py"
                    if init.exists() and not init.resolve(strict=True).is_relative_to(canonical):
                        return None
                if directory == root:
                    break
                directory = directory.parent
            return candidate
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def _namespace_for(root: Path, name: str) -> str:
    """Register ``root`` as a synthetic package so its submodules import without touching sys.path."""
    if _NAMESPACE_ROOT not in sys.modules:
        parent = types.ModuleType(_NAMESPACE_ROOT)
        parent.__path__ = []  # a namespace with no on-disk location of its own
        sys.modules[_NAMESPACE_ROOT] = parent
    full = f"{_NAMESPACE_ROOT}.{name}"
    existing = sys.modules.get(full)
    if existing is None:
        package = types.ModuleType(full)
        package.__path__ = [str(root)]
        sys.modules[full] = package
    elif getattr(existing, "__path__", None) != [str(root)]:
        raise PluginError(
            f"two packages claim the namespace {full!r} ({getattr(existing, '__path__', None)} vs "
            f"[{str(root)!r}]) — resolve the name collision rather than letting one shadow the other"
        )
    return full


def load_module(root: str | Path, reference: str, *, package_name: str):
    """Import the module a plugin reference names, by file path, from ``root``.

    ``package_name`` scopes the synthetic namespace, so two packages shipping the same module name stay
    distinct. Raises :class:`PluginError` rather than returning None: a caller asking to load a plugin
    has already decided it needs one.
    """
    root_path = provider_root(root)
    reference = reference.split(ATTR_SEP, 1)[0]
    module_ref = reference
    if module_ref.endswith(".py"):
        module_ref = module_ref[: -len(".py")]
    module_ref = ".".join(Path(module_ref).parts) if "/" in module_ref else module_ref
    if _reference_path(root_path, reference, "module") is None:
        if core_module_path(reference, "module") is not None:
            # Generic core code the provider SELECTS but does not ship: import it under its own name.
            try:
                return importlib.import_module(reference)
            except Exception as exc:  # noqa: BLE001 — report which plugin failed
                raise PluginError(f"importing core module {reference!r} failed: {type(exc).__name__}: {exc}") from exc
        raise PluginError(f"{reference!r} does not resolve to a module under {root_path}")
    # Inspect Python's actual file-finder precedence without importing parents:
    # a namespace directory can be shadowed by a sibling module/extension.
    search = [str(root_path)]
    parts = module_ref.split(".")
    for index, part in enumerate(parts):
        if not part:
            raise PluginError(f"{reference!r} is not a module under {root_path}")
        spec = importlib.machinery.PathFinder.find_spec(part, search)
        if spec is None:
            raise PluginError(f"{reference!r} does not resolve to a module under {root_path}")
        try:
            if spec.origin is not None and not Path(spec.origin).resolve(strict=True).is_relative_to(root_path):
                raise ValueError("module origin escapes provider root")
            locations = list(spec.submodule_search_locations or [])
            if any(not Path(path).resolve(strict=True).is_relative_to(root_path) for path in locations):
                raise ValueError("package search path escapes provider root")
            if index < len(parts) - 1 and not locations:
                raise ValueError("parent reference is not a package")
        except (OSError, RuntimeError, ValueError) as exc:
            raise PluginError(f"{reference!r} cannot import under {root_path}: {exc}") from exc
        search = locations
    namespace = _namespace_for(root_path, package_name)
    try:
        return importlib.import_module(f"{namespace}.{module_ref}")
    except Exception as exc:  # noqa: BLE001 — report which plugin failed
        raise PluginError(f"importing {reference!r} from {root_path} failed: {type(exc).__name__}: {exc}") from exc


def load_object(root: str | Path, reference: str, *, package_name: str) -> Any:
    """Resolve a ``module:attribute`` plugin reference to the attribute itself."""
    if ATTR_SEP not in reference:
        raise PluginError(f"{reference!r} names no attribute (expected 'module{ATTR_SEP}attribute')")
    module_ref, _, attr = reference.partition(ATTR_SEP)
    module = load_module(root, module_ref, package_name=package_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise PluginError(f"{module_ref!r} has no attribute {attr!r}") from exc


def resolve_support(target: str):
    """Resolve executable support from explicit selection, never metadata fallback."""
    from merlin.target.registry import identity as target_registry

    root = target_registry.explicit_targets().get(target)
    if root is None:
        vendored = target_registry.default_support_root(target)
        if vendored is not None:
            # Only reachable with the variable SET: unset, the vendored provider is the selection.
            raise PluginError(
                f"{target}: executable support requires explicit MERLIN_TARGET_PATH selection, and the "
                f"value set here does not list this target's vendored provider {vendored}; an explicit "
                "value replaces the default, so unset the variable or add that path"
            )
        raise PluginError(
            f"{target}: executable support requires explicit MERLIN_TARGET_PATH selection "
            f"(or, with it unset, an in-repo examples/*/{target_registry.IN_REPO_SUPPORT_DIR} provider "
            "declaring this target)"
        )
    selected = target_registry.resolve(target)
    if selected.base.resolve() != root.resolve():
        raise PluginError(f"{target}: resolved provider differs from explicit support selection")
    return selected


def load_declared(target: str, key: str):
    """Import the module that ``target``'s own contract declares under ``plugin.<key>``.

    Shared code uses this to reach a target-owned tool without naming the target. The caller passes the
    target it was given and the key it needs, and the package's contract names the file. Fails closed,
    with the reason, when the key is not a loadable one, the target cannot be resolved, or its contract
    declares nothing under the key. A target without the tool gets no tool, never a borrowed one.
    """
    spec = PLUGIN_KEYS.get(key)
    if spec is None or not spec.consumed or key == "path":
        raise PluginError(f"plugin.{key}: not a loadable plugin key (known: {sorted(PLUGIN_KEYS)})")
    try:
        info = resolve_support(target)
        block = info.plugin()
    except Exception as exc:  # noqa: BLE001 — carry which target and why
        raise PluginError(f"{target!r}: cannot read its contract's plugin block ({type(exc).__name__}: {exc})") from exc
    reference = block.get(key)
    if not isinstance(reference, str) or not reference:
        raise PluginError(f"{target!r} declares no plugin.{key} in {info.contract_path}")
    root = provider_root(info.base, block.get("path"))
    return load_module(root, reference, package_name=target)


# ---- package -----------------------------------------------------------------------------------
# What a target package can DO — one classifier, asked by every loader instead of each guessing.
#
# THE PROBLEM THIS IS FOR. Merlin has two loaders that each take a directory and hope:
#
# * :func:`merlin.runtime.backends.registry._load_oot_backend` wants ``contracts/target_contract.yaml`` with
#   a ``plugin.backend`` naming a module that self-registers;
# * :func:`merlin.development.targetgen.packaging.loading.load_package` wants ``manifest.yaml`` (experiment ABI v0.1), an
#   executable tool, and four subprocess ``commands``.
#
# Each loader discovers a mismatch by failing somewhere inside itself — a ``KeyError``,
# a ``ContractViolation``, a missing attribute — long after the point where the honest answer was simply
# "this package does not do that".
#
# WHAT CONVERGES, AND WHAT DELIBERATELY DOES NOT. Not the file layout. ``oot_runner``'s ``manifest.yaml``
# is a FROZEN experiment ABI (``src/merlin/resources/contracts/mlir_oot_backend_contract.yaml``, ``version: "0.1"``),
# schema-validated, mirrored, and depended on by certified packages and in-flight agent sessions; changing
# its required keys would retro-invalidate artifacts that already carry verdicts. What converges is the
# VOCABULARY: one closed set of capability names, one answer shape, inferred from what packages already
# have on disk. Nothing migrates and nothing is invalidated.
#
# THE ANSWER SHAPE IS THE POINT. A capability is either provided WITH EVIDENCE (which file, which key) or
# absent WITH THE EXACT MISSING FILE OR KEY as its reason. There is no third state and no boolean: a bare
# ``False`` is how "this package has no compiler" and "we could not tell" became the same value, and that
# is the shape every one of the failures above already has.
#
# EXPLICIT BEATS INFERRED. A package may DECLARE ``package_capabilities:`` and that wins. The key is
# deliberately NOT ``capabilities:`` — that name is already taken in ``target_contract.yaml``, where it
# holds the hardware's op/layout/capacity surface, and reusing it would make one key mean two unrelated
# things depending on which reader got there first.
#
# NO CALLERS, ON PURPOSE (at first). This lands as a pure classifier — it reads files, it answers
# questions, it changes nobody's behaviour — so that the classification can be reviewed and tested on its
# own before three live loaders start refusing things because of it. That is the same discipline
# :mod:`merlin.evaluation.route_admission` documents for itself.

#: The closed capability vocabulary. Closed because an open one cannot be reasoned about: a caller
#: asking for a capability nobody produces would get a plausible-looking absence instead of an error.
CAPABILITIES: tuple[str, ...] = ("backend", "compiler", "dialect")

#: What each capability MEANS, stated once so a reader never has to infer it from the inference rules.
CAPABILITY_MEANING: dict[str, str] = {
    "backend": (
        "the package ships a runtime backend module that self-registers with "
        "merlin.runtime.backends.registry.register(), so get_backend(<target>) returns it"
    ),
    "compiler": (
        "the package ships an out-of-tree compiler satisfying the experiment ABI v0.1 — a manifest.yaml "
        "with a resolvable tool and the four certification commands, invoked only by subprocess"
    ),
    "dialect": (
        "the package ships a target dialect the staged pipeline can load — dialect.py exposing SPEC_OPS "
        "plus a lowering.yaml, or a plugin.dialect module that self-registers a TargetSpec"
    ),
}

#: Where an explicit declaration may live, in precedence order. Both files are read because the two
#: package shapes keep their metadata in different places and neither is going to move.
_DECLARATION_FILES: tuple[str, ...] = ("contracts/target_contract.yaml", "manifest.yaml")

#: The key an explicit declaration uses. NOT ``capabilities`` — see the module docstring.
DECLARATION_KEY = "package_capabilities"

#: The contract file the backend capability is declared in.
_CONTRACT = "contracts/target_contract.yaml"
#: The experiment-ABI manifest the compiler capability is declared in.
_MANIFEST = "manifest.yaml"
#: The four certification commands the ABI requires. Read from the frozen contract rather than
#: restated, so this module cannot drift from the schema the runner actually validates against.
_REQUIRED_COMMANDS: tuple[str, ...] = (
    "parse",
    "lower_interface_to_target",
    "emit_command_buffer",
    "lower_target_to_llvm",
)


class PackageCapabilityMissing(LookupError):
    """A caller asked a package for a capability it does not provide, and here is exactly what is missing."""

    def __init__(self, root: Path, capability: Capability):
        self.root = root
        self.capability = capability
        super().__init__(capability.explain(root))


@dataclass(frozen=True)
class Capability:
    """One capability of one package: provided with evidence, or absent with a named cause.

    ``missing`` names FILES AND KEYS, not prose — ``manifest.yaml:commands.emit_command_buffer``,
    ``contracts/target_contract.yaml:plugin.backend`` — because the caller's next action is to go look
    at that exact thing, and a sentence it has to parse first is a sentence it will print instead.
    """

    name: str
    provided: bool
    #: ``declared`` when the package stated it, ``inferred`` when it was read off the files present.
    source: str
    #: Why it IS provided (which file, which key, which resolved path). Empty when absent.
    evidence: str = ""
    #: The exact files/keys whose absence is the reason. Empty when provided.
    missing: tuple[str, ...] = ()

    def explain(self, root: Path | None = None) -> str:
        where = f" at {root}" if root is not None else ""
        if self.provided:
            return f"package{where} provides {self.name!r} ({self.source}): {self.evidence}"
        return (
            f"package{where} does not provide {self.name!r} ({self.source}): "
            f"{CAPABILITY_MEANING[self.name]}. Missing: {', '.join(self.missing) or 'unknown'}"
        )


def _provided(name: str, evidence: str, *, source: str = "inferred") -> Capability:
    return Capability(name=name, provided=True, source=source, evidence=evidence)


def _absent(name: str, *missing: str, source: str = "inferred") -> Capability:
    return Capability(name=name, provided=False, source=source, missing=tuple(missing))


def _load_yaml(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """``(mapping, problem)`` — a document that is absent, unparseable, or not a mapping yields None."""
    if not path.is_file():
        return None, "absent"
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, f"unreadable ({type(exc).__name__}: {exc})"
    if not isinstance(doc, dict):
        return None, f"not a mapping (got {type(doc).__name__})"
    return doc, None


# --------------------------------------------------------------------------- explicit declaration
def _declared(root: Path) -> dict[str, Capability]:
    """Capabilities the package DECLARES, from the first declaration file that carries the key.

    A declaration is either a bool or a mapping with ``provided`` plus ``evidence`` / ``missing``. An
    unknown capability name RAISES: the vocabulary is closed, and a package inventing a name would
    otherwise be silently ignored — the same failure mode ``plugins.validate`` exists to stop.
    """
    for rel in _DECLARATION_FILES:
        doc, _ = _load_yaml(root / rel)
        if doc is None:
            continue
        block = doc.get(DECLARATION_KEY)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(f"{root / rel}: {DECLARATION_KEY} must be a mapping, got {type(block).__name__}")
        unknown = sorted(set(block) - set(CAPABILITIES))
        if unknown:
            raise ValueError(
                f"{root / rel}: {DECLARATION_KEY} declares unknown capabilities {unknown}; "
                f"the vocabulary is {list(CAPABILITIES)}"
            )
        out: dict[str, Capability] = {}
        for name, value in block.items():
            if isinstance(value, bool):
                out[name] = (
                    _provided(name, f"declared in {rel}", source="declared")
                    if value
                    else _absent(name, f"{rel}:{DECLARATION_KEY}.{name} = false", source="declared")
                )
                continue
            if not isinstance(value, dict) or "provided" not in value:
                raise ValueError(f"{root / rel}: {DECLARATION_KEY}.{name} must be a bool or a mapping with 'provided'")
            if value["provided"]:
                out[name] = _provided(name, str(value.get("evidence") or f"declared in {rel}"), source="declared")
            else:
                miss = value.get("missing") or [f"{rel}:{DECLARATION_KEY}.{name}.provided = false"]
                out[name] = _absent(name, *(str(m) for m in miss), source="declared")
        return out
    return {}


# --------------------------------------------------------------------------- inference: backend
def _infer_backend(root: Path) -> Capability:
    doc, problem = _load_yaml(root / _CONTRACT)
    if doc is None:
        return _absent("backend", f"{_CONTRACT} ({problem})")
    plugin = doc.get("plugin")
    if not isinstance(plugin, dict) or not plugin.get("backend"):
        return _absent("backend", f"{_CONTRACT}:plugin.backend")
    rel = str(plugin["backend"])

    resolved = _reference_path(root, rel, "path")
    if resolved is None:
        return _absent("backend", f"{_CONTRACT}:plugin.backend -> {rel} (resolves to nothing under {root})")
    return _provided("backend", f"{_CONTRACT}:plugin.backend -> {resolved.relative_to(root)}")


# --------------------------------------------------------------------------- inference: compiler
def _infer_compiler(root: Path) -> Capability:
    doc, problem = _load_yaml(root / _MANIFEST)
    if doc is None:
        return _absent("compiler", f"{_MANIFEST} ({problem})")
    # A registry-format manifest and an ABI manifest share this filename with disjoint schemas, so the
    # discriminator is read as DATA: only the ABI manifest carries the artifact_type the contract names.
    from merlin.target.contract.interfaces import abi as schemas

    try:
        schemas.validate_manifest(doc)
    except schemas.ContractViolation as exc:
        return _absent("compiler", f"{_MANIFEST}: {exc}")
    except Exception as exc:  # noqa: BLE001 — a schema machinery fault is UNKNOWN, and says so
        return _absent("compiler", f"{_MANIFEST}: schema check unavailable ({type(exc).__name__}: {exc})")
    build = doc.get("build")
    tool_key = "build.tool_output" if build else "entrypoints.tool"
    tool_rel = build.get("tool_output") if build else (doc.get("entrypoints") or {}).get("tool")
    if not tool_rel:
        return _absent("compiler", f"{_MANIFEST}:{tool_key}")
    tool = root / str(tool_rel)
    # A `build` block means the tool is PRODUCED by the build, so its absence before the build is not
    # a missing capability. Without a build block the tool must already be there.
    if not build and not tool.exists():
        return _absent("compiler", f"{_MANIFEST}:{tool_key} -> {tool_rel} (no such file under {root})")
    commands = doc.get("commands") or {}
    absent_cmds = [f"{_MANIFEST}:commands.{c}" for c in _REQUIRED_COMMANDS if c not in commands]
    if absent_cmds:
        return _absent("compiler", *absent_cmds)
    return _provided("compiler", f"{_MANIFEST}:{tool_key} -> {tool_rel}, all {len(_REQUIRED_COMMANDS)} commands")


# --------------------------------------------------------------------------- inference: dialect
def _defines_module_level(path: Path, symbol: str) -> bool:
    """Does ``path`` bind ``symbol`` at module level? Structural (AST), never a text match.

    Importing would be the other way to ask, and it is the wrong one here: importing a generated
    dialect executes it, and a classifier that has side effects cannot be asked speculatively.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == symbol:
                    return True
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == symbol:
            return True
    return False


def _infer_dialect(root: Path) -> Capability:
    missing: list[str] = []
    dialect_py, lowering = root / "dialect.py", root / "lowering.yaml"
    if dialect_py.is_file() and lowering.is_file() and _defines_module_level(dialect_py, "SPEC_OPS"):
        return _provided("dialect", "dialect.py defines SPEC_OPS, beside lowering.yaml")
    if not dialect_py.is_file():
        missing.append("dialect.py")
    elif not _defines_module_level(dialect_py, "SPEC_OPS"):
        missing.append("dialect.py:SPEC_OPS")
    if not lowering.is_file():
        missing.append("lowering.yaml")

    doc, problem = _load_yaml(root / _CONTRACT)
    plugin = doc.get("plugin") if isinstance(doc, dict) else None
    if not isinstance(plugin, dict) or not plugin.get("dialect"):
        missing.append(f"{_CONTRACT}:plugin.dialect" if doc is not None else f"{_CONTRACT} ({problem})")
        return _absent("dialect", *missing)
    rel = str(plugin["dialect"])

    resolved = _reference_path(root, rel, "path")
    if resolved is None:
        missing.append(f"{_CONTRACT}:plugin.dialect -> {rel} (resolves to nothing under {root})")
        return _absent("dialect", *missing)
    return _provided("dialect", f"{_CONTRACT}:plugin.dialect -> {resolved.relative_to(root)}")


_INFER = {"backend": _infer_backend, "compiler": _infer_compiler, "dialect": _infer_dialect}


# --------------------------------------------------------------------------- the classifier
def capabilities_of(root: str | Path) -> dict[str, Capability]:
    """Every capability in :data:`CAPABILITIES`, settled for the package at ``root``.

    Every name is settled — a capability is never simply absent from the result, because "not in the
    dict" is the shape that let a missing answer read as a negative one. An explicit
    ``package_capabilities`` declaration wins over inference, and says so in ``source``.
    """
    path = Path(root)
    declared = _declared(path)
    return {name: declared.get(name) or _INFER[name](path) for name in CAPABILITIES}


def capability(root: str | Path, name: str) -> Capability:
    """One capability. Raises ``KeyError`` for a name outside the closed vocabulary."""
    if name not in CAPABILITIES:
        raise KeyError(f"unknown capability {name!r}; the vocabulary is {list(CAPABILITIES)}")
    path = Path(root)
    return _declared(path).get(name) or _INFER[name](path)


def require(root: str | Path, name: str) -> Capability:
    """The capability, or raise :class:`PackageCapabilityMissing` naming the exact missing file/key."""
    cap = capability(root, name)
    if not cap.provided:
        raise PackageCapabilityMissing(Path(root), cap)
    return cap


def describe(root: str | Path) -> str:
    """One line per capability — what a loader would print when it refuses."""
    path = Path(root)
    return "\n".join(cap.explain(path) for _, cap in sorted(capabilities_of(path).items()))


# ---- rtl_backend -------------------------------------------------------------------------------
# Generic, DERIVATION-DRIVEN target backend — the target-agnostic replacement for per-target plugins.
#
# The hard rule (see the derive-dont-overfit memory / repo convention): everything target-specific is
# DERIVED from mlc's RTL discovery, never hand-written per target. Given a ``target``, this module reads
# mlc's discovery — the legal opcode set (the ISA, from the decoder's ``comb.icmp`` fan-out), the memory
# map (operand/accumulator banks), and the mesh DIM — and derives the compiler-modification surface: the
# structural levers the discovered hardware IMPLIES, routed to the target's own (OOT) codegen seams.
#
# A new accelerator plugs in by being registered with mlc (its RTL → firtool/arcilator → discovery); no
# new Python here. Anything genuinely not derivable is the rare EXCEPTION and belongs in a declarative
# per-target artifact (YAML/MLIR) or a tool parameter, never in this module or the agnostic core.


@dataclass(frozen=True)
class TargetProfile:
    """What the generic backend needs, entirely DERIVED from mlc RTL discovery (no hand facts)."""

    target: str
    legal_opcodes: tuple[int, ...] | None  # the ISA the decoder actually matches
    memory_map: dict | None  # operand/accumulator bank handles + row bytes
    dim: int | None  # systolic mesh DIM (None if the target has no mesh)

    @property
    def has_mesh(self) -> bool:
        return self.dim is not None

    @property
    def has_accumulator(self) -> bool:
        return bool(self.memory_map and self.memory_map.get("accum_mem"))

    @property
    def discovered_nothing(self) -> bool:
        """True when NO field was grounded, i.e. discovery did not run rather than found bare hardware.

        Every field is None exactly when mlc or the artifact was unavailable, and that is
        indistinguishable, downstream, from a target whose RTL genuinely has no mesh and no accumulator.
        :func:`lever_derivation_gaps` exists to keep the two apart.
        """
        return self.legal_opcodes is None and self.memory_map is None and self.dim is None


def _load_facts_callable():
    """``rtl.facts.load_facts``, however this module happens to be imported.

    STAGED FLAT, AND THAT BROKE IT. In the experiment sandbox this module is not imported as
    ``merlin.target.registry.plugins`` but staged at the workspace root and imported as plain
    ``rtl_backend`` — at which point ``from .rtl.facts import ...`` raises "attempted relative import
    with no known parent package". The caller caught that, returned None, fell through to the live
    bridge (also absent in the box), and handed back a profile with every field None. MEASURED: the
    mandated authoring command ``derived_levers(target_profile('gemmini'))`` returned ``[]`` for a
    target with a 16x16 mesh and a 64 KiB accumulator, and the run was marked NOT CONFORMANT on
    ``rtl_derived_levers_used`` for a reason that had nothing to do with the work.

    The ``rtl`` package IS staged beside this module, so the absolute form resolves there; try the
    package-relative form first so an in-tree import is unchanged.
    """
    try:
        from merlin.target.rtl.facts import load_facts
    except ImportError:
        from rtl.facts import load_facts  # flat-staged: `rtl` is a top-level package beside us
    return load_facts


def _mlc_bridge_module():
    """``rtl.mlc_bridge``, by the same two routes and for the same reason as :func:`_load_facts_callable`.

    Raises ``ImportError`` when neither resolves, which is the documented masked-oracle case the caller
    degrades on — so a masked bridge and a mis-staged one still look the same HERE, but the facts path
    above now succeeds first and the caller no longer reaches this at all in the sandbox.
    """
    try:
        from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge
    except ImportError:
        from rtl import mlc_bridge  # flat-staged
    return mlc_bridge


def _profile_from_reviewed_facts(target: str) -> TargetProfile | None:
    """Build the structural profile from the target's reviewed RTL-facts artifact.

    ``rtl.mlc_bridge`` also contains the bit-exact cosim/readback oracle, so the experiment sandbox
    correctly masks that module.  The reviewed ``facts.json`` pin is the answer-free output of the same
    discovery and is the right authoring input: arrays, memories, and the decoder's legal funct set,
    without a callable route to expected outputs.  ``None`` means no populated artifact was available;
    callers may then try live discovery outside the sandbox.
    """
    try:
        load_facts = _load_facts_callable()
        body = (load_facts(target) or {}).get("facts") or {}
    except Exception:  # noqa: BLE001 — absent/unreadable reviewed artifact: try live discovery below
        return None
    if not body:
        return None

    arrays = body.get("arrays") or ()
    mesh = next((a for a in arrays if a.get("name") == "mesh" and a.get("rows") is not None), None)
    if mesh is None:
        mesh = next((a for a in arrays if a.get("rows") is not None and a.get("cols") is not None), None)
    dim = int(mesh["rows"]) if mesh is not None else None

    memories = {str(m.get("name")): m for m in (body.get("memories") or ()) if m.get("name")}
    memory_map: dict = {"memories": memories}
    if "scratchpad" in memories:
        memory_map["operand_mem"] = memories["scratchpad"]
    if "accumulator" in memories:
        memory_map["accum_mem"] = memories["accumulator"]

    decoder = next((i for i in (body.get("interfaces") or ()) if i.get("legal_funct") is not None), None)
    legal = tuple(sorted({int(v) for v in (decoder.get("legal_funct") or ())})) if decoder else ()
    return TargetProfile(target=target, legal_opcodes=legal or None, memory_map=memory_map, dim=dim)


def target_profile(target: str) -> TargetProfile:
    """Derive the target profile from reviewed RTL facts, then live discovery when needed.

    Fields are ``None`` when neither source is available — the caller degrades honestly and never
    fabricates a hardware fact.  Preferring the reviewed pin also makes this promised authoring command
    runnable in the deny-by-default sandbox, where the oracle-bearing live bridge is intentionally masked.
    """
    reviewed = _profile_from_reviewed_facts(target)
    if reviewed is not None:
        return reviewed
    try:
        mlc_bridge = _mlc_bridge_module()
    except ImportError:
        # The live bridge is an ORACLE module (mlc arc cosim + DRAM readback) and the sandbox masks it
        # BY DESIGN. Preferring the reviewed pin above is what normally keeps this promised authoring
        # command runnable there -- but a target with no reviewed facts falls through to here, and a
        # hard import then raises `cannot import name 'mlc_bridge'` INSIDE the box. That contradicts
        # this function's stated contract two lines up ("fields are None when neither source is
        # available -- the caller degrades honestly and never fabricates a hardware fact") and reported
        # radiance as a launch NO-GO for correctly enforced masking. Degrade, as documented.
        return TargetProfile(target=target, legal_opcodes=None, memory_map=None, dim=None)
    ops = mlc_bridge.discover_legal_opcodes(target) if mlc_bridge.mlc_available()[0] else {}
    return TargetProfile(
        target=target,
        legal_opcodes=tuple(ops.get("legal_opcodes") or ()) or None,
        memory_map=mlc_bridge.discovered_memory_map(target),
        dim=mlc_bridge.discovered_dim(target),
    )


def derived_levers(profile: TargetProfile) -> list[str]:
    """The structural compiler levers the DISCOVERED hardware implies — derived, never hand-listed.

    A systolic mesh implies a dataflow choice (WS/OS); an accumulator memory implies an
    accumulator-residency choice. Targets without that structure simply don't expose those levers. This
    is how the CCA/route surface stays target-agnostic: the levers come from what the RTL has."""
    levers: list[str] = []
    if profile.has_mesh:
        levers.append("spatial.dataflow")
    if profile.has_accumulator:
        levers.append("spatial.accumulator_resident")
        # An accumulator memory is also what makes on-chip residency a compile-time FEASIBILITY
        # question rather than a performance hint: a working set that overruns it produces wrong
        # answers silently, so the fit is a lever the hardware implies.
        levers.append("memory.capacity_fit")
    levers.extend(_endpoint_levers(profile.target))
    return levers


def _endpoint_levers(target: str) -> list[str]:
    """Levers implied by what the target's compute ENDPOINT can be told to do.

    Derived from the roles its own encoding table binds (``kernels.endpoints``), which is the same
    derivation the lifter reads — so a target cannot be offered a lever whose instruction it does not
    have, and a target that GAINS one gets the lever without an edit here. This is the dispatch surface
    the RTL profile alone cannot see: a mesh and an accumulator say nothing about whether the endpoint
    accepts a loop descriptor, and on gemmini that descriptor is the single biggest expert win.
    """
    try:
        from merlin.analysis.observation import compute_endpoints as _ep

        eps = _ep.endpoints_for(target)
    except Exception:  # noqa: BLE001 — no endpoint declared / unreadable: derive nothing, claim nothing
        return []
    roles: set[str] = set()
    for endpoint in eps:
        roles |= set(endpoint.roles)
    out: list[str] = []
    if "loop_descriptor" in roles:
        out.append("dispatch.loop_offloaded")
    if "config" in roles:
        out.append("dispatch.descriptor_reuse")
    if "dma" in roles:
        out.append("dispatch.dma_overlap")
    if "sync" in roles:
        out.append("simt.barriers_in_loop" if any(e.engine == "simt" for e in eps) else "dispatch.dma_overlap")
    if "weight_load" in roles or "operand_load" in roles:
        out.append("layout.operand_major")
    return sorted(dict.fromkeys(out))


def lever_derivation_gaps(profile: TargetProfile) -> tuple[str, ...]:
    """Why :func:`derived_levers` came back short, when the reason is "nothing was discovered".

    An empty lever list has two very different causes and the list itself cannot tell them apart: the RTL
    was read and has no mesh or accumulator memory, or nothing was read at all. MEASURED on the spatial
    tensor tile: every field comes back None and ``derived_levers`` returns ``[]``, which reads as "this
    accelerator exposes no structural levers" when what happened is that the discovery path produced
    nothing. A caller acting on that would report a target with no levers rather than a missing
    capability, so the silence is surfaced here instead of being inferred from an empty list.

    (This is also why the matrix-unit CCA lifter derives residency from the emitted instruction stream
    rather than from this profile: the stream is present whether or not RTL discovery is.)
    """
    if profile.discovered_nothing:
        return (
            f"no RTL fact was grounded for {profile.target!r} (legal_opcodes, memory_map and dim are "
            "all absent), so an empty lever list means UNKNOWN, not 'this hardware has no levers'",
        )
    gaps: list[str] = []
    if profile.dim is None:
        gaps.append("mesh dimension not discovered, so no dataflow lever is claimed")
    if not profile.has_accumulator:
        gaps.append("no accumulator memory discovered, so no residency lever is claimed")
    return tuple(gaps)


# Each derived lever's action class + the SpatialFacet target value it drives towards. This is
# agnostic-per-FACET metadata (a property of the shared SpatialFacet schema), NOT per-target: any
# systolic accelerator's dataflow/residency levers mean the same thing.
_LEVER_META = {
    "spatial.dataflow": (
        "HEURISTIC",
        "os",
        "select the systolic dataflow (WS/OS) — a discovered mesh implies this choice",
    ),
    "spatial.accumulator_resident": (
        "PASS",
        True,
        "keep the output accumulator-resident across the reduction — a "
        "discovered accumulator memory implies this choice",
    ),
    "memory.capacity_fit": (
        "HEURISTIC",
        True,
        "tile so the working set fits the discovered on-chip capacity — overrunning "
        "it is not slow, it is silently wrong",
    ),
    # --- dispatch: implied by what the endpoint's own instruction table accepts ---
    "dispatch.loop_offloaded": (
        "PASS",
        True,
        "hand the loop nest to the endpoint's own sequencer instead of issuing "
        "the nest command by command — the endpoint binds a loop_descriptor role",
    ),
    "dispatch.descriptor_reuse": (
        "KNOB",
        True,
        "set endpoint state once and let the rest of the stream inherit it, instead of re-configuring per tile",
    ),
    "dispatch.dma_overlap": ("HEURISTIC", True, "issue bulk movement so it overlaps the compute it feeds"),
    "simt.barriers_in_loop": (
        "CODEGEN",
        0,
        "hoist barriers out of the reduction loop — a barrier inside it says the "
        "engine cannot hold its state across the reduction",
    ),
    "layout.operand_major": ("KNOB", "k_major", "pack the operand in the major order the endpoint's load role streams"),
}


def apply_codegen_opts(opts: dict, features, profile: TargetProfile) -> dict:
    """Enable each derived lever named in ``features`` (each must be a lever the discovered HW implies).
    Empty ``features`` returns ``opts`` unchanged — the baseline. Target-agnostic (keyed by lever name)."""
    if not features:
        return dict(opts)
    levers = set(derived_levers(profile))
    out = dict(opts)
    for f in sorted(features):
        if f not in levers:
            raise KeyError(f"{f!r} is not a derived lever for {profile.target!r} (have {sorted(levers)})")
        out[f] = _LEVER_META[f][1]
    return out


def _resolver(spec, profile: TargetProfile) -> dict:
    """Realize the target-agnostic MicrokernelSpec as codegen knobs clamped to the DISCOVERED DIM."""
    dim = profile.dim or 1
    feats = (
        frozenset({"spatial.accumulator_resident"})
        if (getattr(spec, "k_block", False) and profile.has_accumulator)
        else frozenset()
    )
    return {
        "tile_rows": min(int(getattr(spec, "MR", dim) or dim), dim),
        "tile_cols": min(int(getattr(spec, "NR", dim) or dim), dim),
        "k_tile": getattr(spec, "KC", dim),
        "opts": apply_codegen_opts({}, feats, profile),
    }


def register(target: str, oot_package: str | None = None) -> None:
    """DERIVE + register the target's compiler-modification surface into the agnostic core — no per-target
    Python. Routes come from ``derived_levers(profile)``; the codegen seam is OOT-package-relative (the
    agent edits ITS generated middle-end); the micro-kernel resolver clamps to the discovered DIM.
    forkable_now is False until the target's OOT codegen threads the opts. Idempotent."""
    from merlin.analysis.kernels.cca import action_catalog as ac
    from merlin.analysis.mining import microkernel

    prof = target_profile(target)
    ac.register_seam(
        "rtl_codegen",
        "<oot_package>/lowering/  (the generated OOT backend's command/tile-program emitter — thread the "
        "derived CodegenOpts through it)",
        "the target's OOT codegen emitter (a discovery-derived lever is applied here)",
        True,
        backend=target,
    )
    for lever in derived_levers(prof):
        action_class, target_value, change = _LEVER_META[lever]
        ac.register_route(
            target,
            ac._Route(
                axis=lever,
                when=lambda d, _l=lever: bool(d.expert) and d.expert != d.ours,
                action_class=action_class,
                target_seam=f"rtl_codegen:{lever}",
                change=change,
                forkable_now=False,
                expected_effect="expected (not yet measured): the discovered lever routed to the OOT codegen",
                intended_facet={lever: target_value},
            ),
        )
    microkernel.register_resolver(target, lambda spec, _p=prof: _resolver(spec, _p))


def lift_cca_from_trace(trace: dict, profile: TargetProfile, *, op: str = "matmul", source: str = "trace"):
    """Lift a DECODED accelerator command trace into a CCA via the agnostic ``cca.lift_spatial`` — using
    only generic accelerator-command concepts (a compute op present; an accumulate flag; an accumulator
    readout width) plus the discovered DIM. The raw-stream DECODING is the target's concern (mlc's
    behavioral opcode→effect map, or a target decoder); this consumes the decoded histogram."""
    from merlin.analysis.kernels.cca import abstraction as cca

    insns = trace.get("instructions", [])
    hist = trace.get("summary", {}).get("class_histogram", {})
    has_compute = any(v for k, v in hist.items() if "COMPUTE" in k.upper())
    acc_resident = any("ACCUM" in k.upper() and v for k, v in hist.items()) or any(i.get("accumulate") for i in insns)
    i32 = any(i.get("readout") == "i32" for i in insns)
    counts = {
        "acc_resident": acc_resident if has_compute else None,
        "widening": has_compute,
        "acc_dtype": ("i32" if i32 else "i8") if has_compute else None,
    }
    return cca.lift_spatial(
        counts, op=op, source=source, pe_rows=profile.dim, pe_cols=profile.dim, backend=profile.target
    )


# Self-register as the derivation-driven route deriver: a seam-menu call for any non-RVV backend
# (cca_contract / action_catalog) then lazily DERIVES + registers that target's RTL levers via
# action_catalog.ensure_backend -> register(target). Import-time + guarded so a core-only environment
# (no targetgen import) is unaffected. register() is idempotent and no-ops without RTL access.
try:
    from merlin.analysis.kernels.cca import action_catalog as _ac

    _ac.register_deriver(register)
except Exception:  # noqa: BLE001 — kernels package layout differs / partial env
    pass
