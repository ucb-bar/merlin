"""Closed source loading for the isolated recipe interpreter and legacy facade.

Every implementation is edited in its responsibility owner. The direct-file
entry derives one namespace from those exact definitions, preserving its old
global patch semantics without importing installed Merlin or xDSL.
"""

from __future__ import annotations
import __future__

import ast
import contextlib
import hashlib
import importlib.util
import json
import math
import sys
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

OWNERS = ("effects", "graphs", "calibration", "eager", "annotation", "pt2e", "application")
DEPENDENCIES = (
    "_activation_contractions",
    "_capture_integer_mm",
    "_dynamic_contractions",
    "_static_attention",
    "_static_planes",
    "_recipe_observers",
    "_gptq",
    "_integer_epilogue",
    "_integer_operand_sum",
    "_integer_residual",
    "_integer_pool",
    "_projection_context",
    "_projection_input_reuse",
    "_recipe_extensions",
    "_recipe_agreement",
    "quant_layer_plan",
    "_recipe_specs",
)
SPEC_EXPORTS = (
    "RECIPE_SCHEMA",
    "DEFAULT_OBSERVER_EPSILON",
    "ACTIVATION_CONTRACTION",
    "MIN_ACTIVATION_REDUCTION",
    "_TORCH_DTYPE",
    "RecipeError",
    "_no_batch_norm_fold_needed",
    "_require",
    "_torch_dtype",
    "_observer_name",
    "_spec",
    "_recipe_operators",
)
CONSTANTS = {
    "_PT2E_CLEANUP_LOCK",
    "AGREEMENT_SAMPLE_LIMIT",
    "_QUANTIZE_TRANSFORMS",
    "FRAMEWORK_CANNOT_EXPRESS",
    "_SDPA_FIELDS",
    "_INHERITED_META",
}
_ROOT = Path(__file__).resolve().parent
_TARGETGEN = _ROOT.parents[2] / "targetgen"
_LOCK = threading.RLock()


def _bytes(path):
    if path.is_symlink() or not path.is_file():
        raise ImportError(f"recipe source is absent or indirect: {path}")
    return path.read_bytes()


def source_identity():
    """Observe the exact closed implementation roster, not framework dependencies."""
    files = (
        Path(__file__).resolve(),
        _ROOT / "__init__.py",
        _TARGETGEN / "_recipe_quantizer.py",
        *(_ROOT / (name + ".py") for name in OWNERS),
        *(_TARGETGEN / (name + ".py") for name in DEPENDENCIES),
    )
    return tuple((str(path), hashlib.sha256(_bytes(path)).hexdigest()) for path in files)


def _key(identity):
    return "_merlin_recipe_" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def helper(name):
    """Load only an admitted sibling file; preloaded foreign leaves refuse."""
    if name not in DEPENDENCIES:
        raise ImportError(f"recipe dependency is not admitted: {name}")
    path = _TARGETGEN / (name + ".py")
    with _LOCK:
        selected = None
        for spelling in ("merlin.targetgen." + name, name):
            existing = sys.modules.get(spelling)
            if existing is None:
                continue
            if Path(getattr(existing, "__file__", "")).resolve() != path:
                raise ImportError(f"preloaded recipe dependency differs from selected source: {spelling}")
            if selected is not None and selected is not existing:
                raise ImportError(f"competing recipe dependency identities: {name}")
            selected = existing
        if selected is None:
            before = _bytes(path)
            canonical = "merlin.targetgen." + name
            spec = importlib.util.spec_from_file_location(canonical, path)
            selected = importlib.util.module_from_spec(spec)
            sys.modules[name] = selected
            sys.modules[canonical] = selected
            try:
                spec.loader.exec_module(selected)
                if _bytes(path) != before:
                    raise ImportError(f"recipe dependency changed while loading: {path}")
            except BaseException:
                sys.modules.pop(name, None)
                sys.modules.pop(canonical, None)
                raise
        observed = hashlib.sha256(_bytes(path)).hexdigest()
        if getattr(selected, "__recipe_source_sha256__", observed) != observed:
            raise ImportError(f"selected recipe dependency changed after loading: {path}")
        selected.__recipe_source_sha256__ = observed
        sys.modules[name] = selected
        sys.modules["merlin.targetgen." + name] = selected
        return selected


def prepare_helpers():
    # All admitted helpers have framework-free module initialization. Their
    # existing lazy cross-imports now resolve these exact objects, not sys.path.
    for name in DEPENDENCIES:
        helper(name)


class _Facade(ModuleType):
    """Retain observed legacy global patch behavior across the new real owners."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if not name.startswith("__"):
            for namespace in self.__dict__.get("__recipe_owner_namespaces__", ()):
                if name in namespace:
                    namespace[name] = value


def adopt(module):
    """Join package and registered bare-file imports of the same source selection."""
    identity = source_identity()
    if module.__dict__.get("__recipe_sources__", identity) != identity:
        raise ImportError("recipe facade source changed after loading")
    key = _key(identity)
    with _LOCK:
        existing = sys.modules.get(key)
        if existing is not None:
            if existing.__dict__.get("__recipe_sources__") != identity:
                raise ImportError("recipe facade source identity changed")
            return existing
        modules = tuple(sys.modules[__package__ + "." + name] for name in OWNERS)
        for owner in modules:
            for name, value in tuple(vars(owner).items()):
                if not name.startswith("__"):
                    module.__dict__.setdefault(name, value)
        for name in SPEC_EXPORTS:
            module.__dict__.setdefault(name, getattr(helper("_recipe_specs"), name))
        module.__dict__["__recipe_owner_namespaces__"] = tuple(vars(owner) for owner in modules if owner is not module)
        module.__dict__["__recipe_sources__"] = identity
        module.__class__ = _Facade
        sys.modules[key] = module
        return module


def _definitions(tree, owner):
    """Admit the same definitions and explicit dependencies in both workflows."""
    admitted = []
    administrative = {
        "_specs = helper('_recipe_specs')",
        "_owner = adopt(sys.modules[__name__])",
        "sys.modules[__name__] = _owner",
    }
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            admitted.append(node)
        elif isinstance(node, ast.Assign):
            if all(isinstance(target, ast.Name) and target.id in CONSTANTS for target in node.targets):
                admitted.append(node)
            elif ast.unparse(node) in administrative:
                continue
            elif (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in SPEC_EXPORTS
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "_specs"
                and node.value.attr == node.targets[0].id
            ):
                continue
            else:
                raise ImportError(f"recipe owner has an undeclared portable binding: {owner}:{node.lineno}")
        elif isinstance(node, ast.Import):
            if any(
                alias.name not in {"contextlib", "json", "math", "sys", "threading"} or alias.asname is not None
                for alias in node.names
            ):
                raise ImportError(f"recipe owner has an undeclared portable import: {owner}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            allowed = (
                {"annotations"}
                if node.module == "__future__" and not node.level
                else {"Iterable", "Mapping"}
                if node.module == "collections.abc" and not node.level
                else {"Any"}
                if node.module == "typing" and not node.level
                else None
                if node.level == 1 and node.module in {*OWNERS, "portable"}
                else set()
            )
            if (
                any(alias.asname is not None or alias.name == "*" for alias in node.names)
                or allowed is not None
                and any(alias.name not in allowed for alias in node.names)
            ):
                raise ImportError(f"recipe owner has an undeclared portable import: {owner}:{node.lineno}")
        elif not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            raise ImportError(f"recipe owner has an undeclared portable statement: {owner}:{node.lineno}")
    return admitted


def populate(namespace):
    """Derive direct-file definitions from the selected owners, without a monolith.

    Unregistered importlib execution keeps its supplied globals, as before, so
    patching that returned module still affects its functions. Registered bare
    imports and package imports share the same facade object. All helper class
    and record identities are shared in either workflow.
    """
    identity = source_identity()
    key = _key(identity)
    registered = sys.modules.get(namespace["__name__"])
    if namespace.get("__recipe_sources__", identity) != identity:
        raise ImportError("recipe facade source changed after loading")
    with _LOCK:
        existing = sys.modules.get(key)
        if registered is not None and vars(registered) is namespace and existing is not None:
            if existing.__dict__.get("__recipe_sources__") != identity:
                raise ImportError("recipe facade source identity changed")
            return existing
        prepare_helpers()
        namespace.update(
            contextlib=contextlib,
            json=json,
            math=math,
            threading=threading,
            Iterable=Iterable,
            Mapping=Mapping,
            Any=Any,
            helper=helper,
        )
        namespace.update((name, getattr(helper("_recipe_specs"), name)) for name in SPEC_EXPORTS)
        for owner in OWNERS:
            path = _ROOT / (owner + ".py")
            tree = ast.parse(_bytes(path), filename=str(path))
            admitted = _definitions(tree, owner)
            unit = ast.Module(body=admitted, type_ignores=[])
            exec(compile(unit, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
        if source_identity() != identity:
            raise ImportError("selected recipe source changed during portable assembly")
        namespace["__recipe_sources__"] = identity
        if registered is not None and vars(registered) is namespace:
            sys.modules[key] = registered
        return registered
