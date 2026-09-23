"""The ``plugin`` block of a target contract: what its keys mean, and whether they point anywhere.

A target package extends Merlin by DECLARING things in its contract's ``plugin`` block rather than by
being named anywhere in core. One such key is already load-bearing (``backend``, consumed by
:mod:`merlin.runtime.backends.base`); others are pointers to a package's own prototype artifacts.

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
:mod:`merlin.runtime.backends.base` uses for ``plugin.backend``, generalised so every future key
(oracles, suites, cost models) gets it for free instead of re-implementing it.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        "dialect — that is the package's dialect.py, loaded by targetgen.registry.load_target.",
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
        target = _reference_path(Path(root), value, spec.expects)
        if target is None:
            problems.append(f"{where}.{key}: {value!r} does not resolve to a file under {root}")
    return problems


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
    from merlin.targetgen import target_registry

    root = target_registry.explicit_targets().get(target)
    if root is None:
        raise PluginError(f"{target}: executable support requires explicit MERLIN_TARGET_PATH selection")
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
