"""Explicit live-parent qualification delegation for the original sealed child.

This transports a qualification the original parent already owns. Target code
must opt in and recheck all original bytes before reconstructing its owner. A
snapshot, ordinary JSON or this generic transport cannot issue qualification.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import inspect
import json
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType

from merlin.common.digest import sha256_file
from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapabilities
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection

PROTOCOL = "merlin.original_parent_pass_qualification.v1"
SCHEMA = "merlin.sealed_parent_pass_delegation.v1"
FIELD = "parent_pass_delegation"
MAX_BYTES = 3 << 20
_PARENT = contextvars.ContextVar("original_parent_pass_delegation", default=None)
_CHILD = contextvars.ContextVar("restored_parent_pass_qualification", default=None)
_REDEEMING = contextvars.ContextVar("original_pass_qualification_redemption", default=None)
_ISSUED = weakref.WeakKeyDictionary()
_TOKENS = weakref.WeakKeyDictionary()
_ISSUER = object()


def _plain(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(raw.encode()) > MAX_BYTES:
        raise ValueError("original parent qualification exceeds the unchanged bounded transport")
    return json.loads(raw)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _function(function):
    if type(function) is not FunctionType or function.__name__ != function.__qualname__:
        raise ValueError("qualification delegation needs original top-level source-bound functions")
    path = Path(inspect.getsourcefile(function) or "")
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError("qualification delegation function has no exact original source")
    return dict(path=str(path), sha256=sha256_file(path), symbol=function.__name__)


@dataclass(frozen=True, eq=False)
class OriginalParentPassDelegation:
    """Select real live parent owners and original export/redeem implementations."""

    selection: CompilerPassSelection
    exporter: object
    redeemer: object
    _record: object = None
    _callbacks: object = None

    def __post_init__(self):
        if type(self.selection) is not CompilerPassSelection or self.selection.factory is None:
            raise ValueError("qualification delegation needs its original live selected pass factory")
        selected = self.selection.record()
        if selected["factory"]["inputs"].get("parent_delegation_protocol") != PROTOCOL:
            raise ValueError("original pass factory did not explicitly opt in to parent delegation")
        callbacks = (_function(self.exporter), _function(self.redeemer))
        snapshot = _plain(self.exporter(self.selection.registry))
        if self.selection.record() != selected:
            raise ValueError("original parent qualification changed during readonly export")
        record = _plain(dict(schema=SCHEMA, protocol=PROTOCOL, selection=selected,
                             exporter=callbacks[0], redeemer=callbacks[1], snapshot=snapshot,
                             qualification_origin="inherited actual original live parent; not fresh pair evaluation"))
        object.__setattr__(self, "_record", record)
        object.__setattr__(self, "_callbacks", (self.exporter.__code__, self.redeemer.__code__))
        _ISSUED[self] = _digest(record)
        self.revalidate()

    def revalidate(self):
        if (type(self) is not OriginalParentPassDelegation or self not in _ISSUED
                or _ISSUED[self] != _digest(self._record)
                or type(self.selection) is not CompilerPassSelection
                or self.selection.record() != self._record["selection"]
                or (_function(self.exporter), _function(self.redeemer))
                != (self._record["exporter"], self._record["redeemer"])
                or (self.exporter.__code__, self.redeemer.__code__) != self._callbacks
                or _plain(self.exporter(self.selection.registry)) != self._record["snapshot"]):
            raise ValueError("original live parent qualification or delegation source changed")
        self.selection.revalidate()

    def record(self):
        self.revalidate()
        return copy.deepcopy(self._record)


def active_original_parent_pass_delegation():
    selected = _PARENT.get()
    if selected is not None:
        selected.revalidate()
    return selected


@contextlib.contextmanager
def scoped_original_parent_pass_delegation(selection):
    if type(selection) is not OriginalParentPassDelegation:
        raise ValueError("qualification delegation scope needs its actual original parent owner")
    selection.revalidate()
    token = _PARENT.set(selection)
    try:
        yield
        selection.revalidate()
    finally:
        _PARENT.reset(token)


@contextlib.contextmanager
def scoped_selected_parent_pass_delegation(selection):
    """Carry an explicitly declared original factory through normal workers.

    Older factories retain full child admission. New factories must explicitly
    name both original callbacks in their already pinned input record. Those
    callbacks must be top-level functions in the exact factory source leaf.
    This only transports a live qualification; current executions stay fresh.
    """
    current = active_original_parent_pass_delegation()
    if current is not None:
        if current.selection is not selection:
            raise ValueError("normal worker selected another original parent qualification")
        yield current
        current.revalidate()
        return
    if selection is None:
        yield None
        return
    if type(selection) is not CompilerPassSelection:
        raise ValueError("normal worker qualification requires its original live pass selection")
    selection.revalidate()
    factory = selection.factory
    declarations = None if factory is None else factory.inputs.get("parent_delegation_callbacks")
    if declarations is None:
        yield None
        selection.revalidate()
        return
    if (factory.inputs.get("parent_delegation_protocol") != PROTOCOL
            or type(declarations) is not dict or set(declarations) != {"exporter", "redeemer"}
            or any(type(name) is not str or not name.isidentifier() for name in declarations.values())):
        raise ValueError("normal worker qualification needs explicit original protocol and callback symbols")
    callbacks = []
    source = factory.record()["callback"]
    for name in (declarations["exporter"], declarations["redeemer"]):
        callback = factory.callback.__globals__.get(name)
        pin = _function(callback)
        if pin != {"path": source["path"], "sha256": source["sha256"], "symbol": name}:
            raise ValueError("normal worker qualification callback is outside the original factory source")
        callbacks.append(callback)
    delegation = OriginalParentPassDelegation(selection, exporter=callbacks[0], redeemer=callbacks[1])
    with scoped_original_parent_pass_delegation(delegation):
        yield delegation


@dataclass(frozen=True, eq=False)
class _OriginalParentQualification:
    _proof: object
    _record: object
    _issuer: object

    def revalidate(self):
        from .compiler_profile_child import verified_parent_pass_delegation

        if (type(self) is not _OriginalParentQualification or self not in _TOKENS
                or self._issuer is not _ISSUER or _CHILD.get() is not self
                or _TOKENS[self] != _digest(self._record)
                or verified_parent_pass_delegation(self._proof) != self._record):
            raise ValueError("pass qualification needs the actual original verified sealed parent")

    def require(self, snapshot):
        self.revalidate()
        if _REDEEMING.get() is not self or _plain(snapshot) != self._record["snapshot"]:
            raise ValueError("pass qualification snapshot is outside its authenticated original redemption")
        selected = self._record["selection"]
        return copy.deepcopy(dict(registry=selected["registry"], factory_inputs=selected["factory"]["inputs"]))

    def redeem(self, callback, inputs, *, decoder, work_root):
        self.revalidate()
        factory = self._record["selection"]["factory"]
        if (_function(callback) != factory["callback"] or _plain(inputs) != factory["inputs"]
                or _function(decoder) != self._record["redeemer"]):
            raise ValueError("pass qualification redemption selects a foreign original factory or decoder")
        token = _REDEEMING.set(self)
        try:
            registry = decoder(copy.deepcopy(self._record["snapshot"]), token=self, work_root=Path(work_root))
        finally:
            _REDEEMING.reset(token)
        self.revalidate()
        if (type(registry) is not ReviewedPassCapabilities
                or registry.record() != self._record["selection"]["registry"]):
            raise ValueError("inherited qualification did not reconstruct the same original live registry")
        return registry

    def receipt(self):
        self.revalidate()
        return dict(schema=PROTOCOL, qualification_origin=self._record["qualification_origin"],
                    delegation_sha256=_digest(self._record),
                    selection_sha256=_digest(self._record["selection"]),
                    snapshot_sha256=_digest(self._record["snapshot"]), fresh_pair_evaluation=False)


def active_original_parent_pass_qualification():
    token = _CHILD.get()
    if token is not None:
        token.revalidate()
    return token


@contextlib.contextmanager
def restored_original_parent_qualification(proof):
    """Only the original sealed-child verifier can supply this live proof."""
    from .compiler_profile_child import verified_parent_pass_delegation

    record = verified_parent_pass_delegation(proof)
    token = _OriginalParentQualification(proof, record, _ISSUER)
    _TOKENS[token] = _digest(record)
    previous = _CHILD.set(token)
    try:
        token.revalidate()
        yield token
        token.revalidate()
    finally:
        _CHILD.reset(previous)
        _TOKENS.pop(token, None)
