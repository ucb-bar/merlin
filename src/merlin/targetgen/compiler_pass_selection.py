"""Original-host compiler pass selection before retained source is frozen.

The registry supplies live independently checked owners. Preparation creates a
separate compiler, never replaces an executing/certified package. A sealed
original-parent child may replay a source-pinned factory which reruns admission;
serialized successful owner records cannot instantiate capabilities.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import importlib.util
import inspect
import json
import shutil
import stat
import threading
from dataclasses import asdict
from pathlib import Path
from types import FunctionType

from merlin.common.digest import sha256_file
from merlin.llvmlower.reviewed_pass_capabilities import PassCapabilityBinding, ReviewedPassCapabilities

SCHEMA = "merlin.compiler_pass_selection.v1"
_ACTIVE = contextvars.ContextVar("original_compiler_pass_selection", default=None)
_LOCK = threading.RLock()
_SCOPES = 0


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not path.is_absolute() or path.resolve() != path:
        raise ValueError("compiler pass source must be an exact regular absolute leaf")
    return {"path": str(path), "sha256": sha256_file(path)}


def _tree(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError("compiler pass package must be an ordinary absolute directory")
    rows, size = [], 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("compiler pass package contains an unsupported file")
        if path.is_file():
            size += path.stat().st_size
            if len(rows) >= 8192 or size > 32 << 20:
                raise ValueError("compiler pass package exceeds its original source bound")
            rows.append((path.relative_to(root).as_posix(), sha256_file(path)))
    return _digest(rows)


def copy_compiler_source(source, destination):
    """Create an owned writable copy, preserving every original source byte.

    Read-only Phase 1 snapshots remain immutable. Target preparation may change
    only this fresh disjoint destination before the ordinary edit/source freeze.
    """
    source, destination = Path(source), Path(destination)
    origin = _tree(source)
    if (
        destination.exists() or destination.is_symlink() or not destination.is_absolute()
        or destination.resolve() != destination or destination.is_relative_to(source)
        or source.is_relative_to(destination) or any(path.is_symlink() for path in destination.parents)
    ):
        raise ValueError("compiler source copy requires a fresh disjoint destination")
    shutil.copytree(source, destination, symlinks=False)
    for path in (destination, *destination.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    if _tree(source) != origin or _tree(destination) != origin:
        raise ValueError("compiler source bytes changed during owned preparation copy")
    return destination


class OriginalPassFactory:
    """Explicit host callback selection; its body must run original admission.

    This is a source custody interface, not a qualification issuer. Only an
    original host can select the callback and inputs. Child restoration is
    permitted only by its independently checked sealed-parent transport.
    """

    def __init__(self, callback, inputs, *, sources=()):
        if type(callback) is not FunctionType or callback.__name__ != callback.__qualname__:
            raise ValueError("compiler pass restoration requires an original top-level callback")
        source = inspect.getsourcefile(callback)
        if source is None:
            raise ValueError("compiler pass restoration callback has no source")
        self.callback, self.inputs = callback, _plain(inputs)
        self._code = callback.__code__
        self._record = {
            "schema": "merlin.original_pass_factory.v1",
            "callback": {**_pin(Path(source).resolve()), "symbol": callback.__name__},
            "sources": [_pin(Path(path)) for path in sources],
            "inputs": self.inputs,
        }
        if len(json.dumps(self._record)) > 2 << 20 or len(self._record["sources"]) > 128:
            raise ValueError("compiler pass factory source/input closure exceeds its bound")
        self.identity = _digest(self._record)
        self.revalidate()

    def revalidate(self):
        row = self._record
        if (
            self.callback.__code__ is not self._code
            or self.callback.__name__ != row["callback"]["symbol"]
            or _pin(Path(inspect.getsourcefile(self.callback)).resolve())
            != {key: row["callback"][key] for key in ("path", "sha256")}
            or any(_pin(Path(pin["path"])) != pin for pin in row["sources"])
            or self.inputs != row["inputs"]
            or _digest(row) != self.identity
        ):
            raise ValueError("original compiler pass factory or inputs changed")

    def record(self):
        self.revalidate()
        return copy.deepcopy(self._record)

    def create(self, *, work_root):
        self.revalidate()
        registry = self.callback(copy.deepcopy(self.inputs), work_root=Path(work_root))
        self.revalidate()
        if type(registry) is not ReviewedPassCapabilities:
            raise ValueError("original compiler pass factory did not return live reviewed owners")
        registry.revalidate()
        return registry


class CompilerPassSelection:
    def __init__(self, registry, binding, *, mechanisms, factory=None):
        if type(registry) is not ReviewedPassCapabilities or type(binding) is not PassCapabilityBinding:
            raise ValueError("compiler passes require original live registry and exact binding")
        if type(mechanisms) is not tuple or not 1 <= len(mechanisms) <= 32 or len(set(mechanisms)) != len(mechanisms):
            raise ValueError("compiler pass mechanisms require a bounded unique explicit tuple")
        if factory is not None and type(factory) is not OriginalPassFactory:
            raise ValueError("compiler pass reconstruction requires an original host factory")
        self.registry, self.binding, self.mechanisms, self.factory = registry, binding, mechanisms, factory
        self._record = {
            "schema": SCHEMA,
            "registry": registry.record(),
            "binding": asdict(binding),
            "mechanisms": list(mechanisms),
            "resolutions": [
                None if (capability := registry.resolve(mechanism, binding)) is None else capability.identity
                for mechanism in mechanisms
            ],
            "factory": None if factory is None else factory.record(),
        }
        self.identity = _digest(self._record)
        self.revalidate()

    def revalidate(self):
        if (
            type(self.registry) is not ReviewedPassCapabilities
            or self.registry.record() != self._record["registry"]
            or asdict(self.binding) != self._record["binding"]
            or list(self.mechanisms) != self._record["mechanisms"]
            or [
                None if (capability := self.registry.resolve(mechanism, self.binding)) is None else capability.identity
                for mechanism in self.mechanisms
            ]
            != self._record["resolutions"]
            or (None if self.factory is None else self.factory.record()) != self._record["factory"]
            or _digest(self._record) != self.identity
        ):
            raise ValueError("original compiler pass selection changed")

    def record(self):
        self.revalidate()
        return copy.deepcopy(self._record)

    def resolve(self, mechanism):
        self.revalidate()
        if mechanism not in self.mechanisms:
            raise ValueError("compiler pass mechanism was not originally selected")
        return self.registry.resolve(mechanism, self.binding)

    def prepare_compiler(self, mechanism, source, destination, *, policy):
        """Prepare once before source/edit freeze; absent capability refuses."""
        capability = self.resolve(mechanism)
        if capability is None:
            raise ValueError("no exact reviewed compiler pass capability")
        source, destination = Path(source), Path(destination)
        origin = _tree(source)
        if (
            destination.exists()
            or destination.is_symlink()
            or not destination.is_absolute()
            or destination.resolve() != destination
            or destination.is_relative_to(source)
            or source.is_relative_to(destination)
            or any(path.is_symlink() for path in destination.parents)
        ):
            raise ValueError("compiler pass preparation requires a fresh disjoint destination")
        result = capability.owner.prepare_compiler(source, destination, policy=_plain(policy))
        self.revalidate()
        if result != destination or _tree(source) != origin:
            raise ValueError("compiler pass changed its original source or preparation boundary")
        return {
            "schema": "merlin.prepared_compiler_pass.v1",
            "selection": self.record(),
            "mechanism": mechanism,
            "source": str(source),
            "source_sha256": origin,
            "prepared": str(destination),
            "prepared_sha256": _tree(destination),
            "authority": "prepare before original retained-source freeze; final realization checks required",
        }

    def check_functional_guard(self, baseline, candidate, baseline_result, candidate_result, *, grade_root):
        """A selected pass cannot publish without its original realization gate."""
        reports = {}
        for mechanism in self.mechanisms:
            capability = self.resolve(mechanism)
            if capability is None:
                continue
            check = getattr(capability.owner, "check_functional_guard", None)
            if not callable(check):
                raise ValueError("selected compiler pass lacks its original final emitted-program checker")
            reports[mechanism] = check(baseline, candidate, baseline_result, candidate_result, grade_root=grade_root)
            self.revalidate()
        return reports


def restore_original_selection(record, *, work_root):
    """Only an explicit original host or verified parent transport may call this.

    The CLI independently selects and pins operator input bytes; a child needs
    its independently checked sealed-parent transport. This executes the
    original source-pinned admission callback, then compares the
    actual live registry. It does not deserialize successful capability owners.
    """
    if type(record) is not dict or set(record) != {
        "schema",
        "registry",
        "binding",
        "mechanisms",
        "resolutions",
        "factory",
    }:
        raise ValueError("original compiler pass transport is incomplete")
    factory = record["factory"]
    if record["schema"] != SCHEMA or type(factory) is not dict:
        raise ValueError("compiler pass transport has no original independent admission factory")
    if (
        set(factory) != {"schema", "callback", "sources", "inputs"}
        or factory["schema"] != "merlin.original_pass_factory.v1"
    ):
        raise ValueError("unknown original compiler pass factory transport")
    callback = factory["callback"]
    for pin in ({key: callback[key] for key in ("path", "sha256")}, *factory["sources"]):
        if _pin(Path(pin["path"])) != pin:
            raise ValueError("original compiler pass reconstruction source changed")
    name = "merlin_original_pass_factory_" + callback["sha256"]
    spec = importlib.util.spec_from_file_location(name, callback["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owner = OriginalPassFactory(
        getattr(module, callback["symbol"]),
        factory["inputs"],
        sources=tuple(Path(pin["path"]) for pin in factory["sources"]),
    )
    if owner.record() != factory:
        raise ValueError("compiler pass reconstruction selected a different original factory")
    selected = CompilerPassSelection(
        owner.create(work_root=work_root),
        PassCapabilityBinding(**record["binding"]),
        mechanisms=tuple(record["mechanisms"]),
        factory=owner,
    )
    if selected.record() != record:
        raise ValueError("independent compiler pass reconstruction returned a different registry")
    return selected


def active_compiler_passes():
    selected = _ACTIVE.get()
    with _LOCK:
        if selected is None and _SCOPES:
            raise RuntimeError("compiler passes escaped their original scoped owner")
    if selected is not None:
        selected.revalidate()
    return selected


@contextlib.contextmanager
def scoped_compiler_passes(selection):
    global _SCOPES
    if selection is not None and type(selection) is not CompilerPassSelection:
        raise ValueError("compiler pass scope requires original host selection")
    if selection is not None:
        selection.revalidate()
    token = _ACTIVE.set(selection)
    with _LOCK:
        _SCOPES += selection is not None
    try:
        yield
        if selection is not None:
            selection.revalidate()
    finally:
        with _LOCK:
            _SCOPES -= selection is not None
        _ACTIVE.reset(token)
