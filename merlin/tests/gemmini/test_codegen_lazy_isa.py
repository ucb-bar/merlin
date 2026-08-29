"""Deriving a target's ISA facts must not be an IMPORT-time side effect.

The emitter binds its encoding constants from a capability manifest plus an RTL fact bundle. It used
to do that at module scope, which quietly made a target's data a prerequisite for merely *naming* the
module. That matters because the generic contract-compile path imports one helper (``_harness_c``)
from here: on a checkout without those facts it failed at import, and the traceback pointed at an
import line rather than at the code that actually needed the facts.

The fix is laziness, not removal — the constants are still derived, still from the same single source.
So the property under test is narrow and exact: importing resolves nothing, first USE resolves, and
what gets resolved is unchanged.
"""
from __future__ import annotations

import importlib
import sys

import pytest


#: The emitter was EVICTED from core to the target's own package (0f645a12), and this test kept
#: naming the core path -- so all five cases here have been failing with ModuleNotFoundError ever
#: since, rather than testing anything. The module is loaded from the target's package by file path,
#: the way `base._load_oot_backend` loads any out-of-tree backend, under a synthetic module name so
#: `importlib.import_module`/`sys.modules` behave exactly as before.
#: The emitter was EVICTED from core to the target's own package (0f645a12) and this test kept naming
#: the core path, so all five cases here died with ModuleNotFoundError and tested nothing. It is
#: loaded from the target's package the way `base._load_oot_backend` loads a package DIRECTORY --
#: under a synthetic parent whose __path__ is the backend dir -- because the module has a sibling
#: relative import (`from .gemmini_codegen import ...`) and so cannot be imported as a top-level file.
PKG = "gemmini_backend_undertest"
MODULE = f"{PKG}.gemmini_codegen_mlir"


def _backend_dir():
    """The target's backend package, resolved through the registry rather than hardcoded."""
    from merlin.targetgen.rtl.facts import target_base

    return target_base("gemmini") / "backend"


@pytest.fixture(autouse=True)
def _install_backend_package():
    """Put the evicted backend package on the import graph for the duration of each test."""
    import types

    d = _backend_dir()
    if not (d / "gemmini_codegen_mlir.py").is_file():
        pytest.skip(f"gemmini backend package not present: {d}")
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(d)]                    # makes `PKG.<sibling>` and `from .x import` resolve
        sys.modules[PKG] = pkg
    yield
    for name in [n for n in sys.modules if n == PKG or n.startswith(PKG + ".")]:
        sys.modules.pop(name, None)


def _reimport_with_poisoned_loaders(monkeypatch):
    """Import the emitter with both fact loaders rigged to explode, so any import-time
    resolution is a hard failure rather than something that merely happens to work here."""
    import merlin.targetgen.rtl.facts as rf
    import merlin.targetgen.target_experiment as te

    def boom(*_a, **_k):
        raise AssertionError("target data resolved at import time")

    monkeypatch.setattr(te, "load_capability_manifest", boom)
    monkeypatch.setattr(rf, "load_facts", boom)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    return importlib.import_module(MODULE)


def test_importing_the_emitter_resolves_no_target_data(monkeypatch):
    module = _reimport_with_poisoned_loaders(monkeypatch)
    assert module is not None, "import must succeed with no manifest and no RTL facts present"


def test_the_generic_contract_path_imports_without_target_data(monkeypatch):
    """The reason the laziness matters: this module is generic and imports a helper from the emitter."""
    _reimport_with_poisoned_loaders(monkeypatch)
    monkeypatch.delitem(sys.modules, "merlin.targetgen.contract.compile", raising=False)
    assert importlib.import_module("merlin.targetgen.contract.compile") is not None


def test_first_use_still_resolves_and_fails_loudly_when_the_data_is_absent(monkeypatch):
    """Laziness must defer the read, never skip it. A constant that silently defaulted would be far
    worse than an import error: it would emit a plausible, wrong instruction encoding."""
    module = _reimport_with_poisoned_loaders(monkeypatch)
    with pytest.raises(AssertionError, match="import time"):
        _ = module.DIM


def test_every_derived_name_is_reachable_as_a_module_attribute():
    """`__getattr__` (PEP 562) keeps the laziness an implementation change, not an API change —
    callers and tests still read `gemmini_codegen_mlir.DIM`."""
    module = importlib.import_module(MODULE)
    for name in module._DERIVED:
        assert isinstance(getattr(module, name), int), name


def test_an_unknown_attribute_still_raises_attribute_error():
    """A catch-all `__getattr__` that returned something for every name would turn a typo into a
    mystery value rather than an error."""
    module = importlib.import_module(MODULE)
    with pytest.raises(AttributeError, match="no attribute"):
        _ = module.NOT_A_REAL_CONSTANT
