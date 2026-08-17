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
    "backend": PluginKey(
        "backend",
        "A runtime backend module (or package directory) that self-registers via "
        "runtime.backends.base.register(). The only fully live seam in the repo.",
        consumed=True, expects="path"),
    "path": PluginKey(
        "path",
        "Injected by target_registry for external packages — the package root. Not authored by hand.",
        consumed=True, expects="path"),
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
        consumed=False, expects="module"),
    "lowering_entrypoint": PluginKey(
        "lowering_entrypoint",
        "Pointer to the package's own lowering demonstration, as `module:callable`. NOT the lowering "
        "tables the staged pipeline reads from lowering.yaml.",
        consumed=False, expects="attr"),
}


class PluginError(ValueError):
    """A plugin block that cannot be trusted: an unknown key, or a reference that points nowhere."""


def validate(plugin: dict[str, Any] | None, *, root: Path | None = None,
             where: str = "plugin") -> list[str]:
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
                "'backend' becomes no backend at all.")
            continue
        if root is None or not isinstance(value, str) or not value:
            continue
        target = _reference_path(Path(root), value, spec.expects)
        if target is None:
            problems.append(f"{where}.{key}: {value!r} does not resolve to a file under {root}")
    return problems


def _reference_path(root: Path, reference: str, expects: str) -> Path | None:
    """The file a plugin reference names, or None when nothing is there.

    Accepts the three spellings a contract may legitimately use: a path relative to the package root
    (``backend.py``, or a directory), a dotted module (``pkg.mod``), and a dotted module with an
    attribute (``pkg.mod:fn``).
    """
    module_ref = reference.split(ATTR_SEP, 1)[0] if expects == "attr" else reference
    direct = root / module_ref
    if direct.exists():
        return direct
    parts = module_ref.split(".")
    as_module = root.joinpath(*parts).with_suffix(".py")
    if as_module.is_file():
        return as_module
    as_package = root.joinpath(*parts) / "__init__.py"
    return as_package if as_package.is_file() else None


def _namespace_for(root: Path, name: str) -> str:
    """Register ``root`` as a synthetic package so its submodules import without touching sys.path."""
    if _NAMESPACE_ROOT not in sys.modules:
        parent = types.ModuleType(_NAMESPACE_ROOT)
        parent.__path__ = []            # a namespace with no on-disk location of its own
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
            f"[{str(root)!r}]) — resolve the name collision rather than letting one shadow the other")
    return full


def load_module(root: str | Path, reference: str, *, package_name: str):
    """Import the module a plugin reference names, by file path, from ``root``.

    ``package_name`` scopes the synthetic namespace, so two packages shipping the same module name stay
    distinct. Raises :class:`PluginError` rather than returning None: a caller asking to load a plugin
    has already decided it needs one.
    """
    root_path = Path(root)
    module_ref = reference.split(ATTR_SEP, 1)[0]
    if module_ref.endswith(".py"):
        module_ref = module_ref[: -len(".py")].replace("/", ".")
    if _reference_path(root_path, reference, "module") is None:
        raise PluginError(f"{reference!r} does not resolve to a module under {root_path}")
    namespace = _namespace_for(root_path, package_name)
    try:
        return importlib.import_module(f"{namespace}.{module_ref}")
    except Exception as exc:                                  # noqa: BLE001 — report which plugin failed
        raise PluginError(f"importing {reference!r} from {root_path} failed: "
                          f"{type(exc).__name__}: {exc}") from exc


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
