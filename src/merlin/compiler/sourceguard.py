"""Observe explicit stage-owned source files and original Python callables.

This leaf discovers no stages or paths. Its callers supply fixed internal owner
readers; observations grant no source access, semantic or loaded-code proof.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from merlin.compiler.artifact import digest


def validator_identity(callback):
    function = getattr(callback, "__func__", callback)
    code = getattr(function, "__code__", None)
    if code is None:
        raise ValueError("implementation validator must be an original Python function or bound method")
    return callback, function, code, getattr(callback, "__self__", None)


# Preserve existing callable/pickle spelling while the real owner is this leaf.
validator_identity.__module__ = "merlin.compiler.ir.semantics"


def _members(reader):
    return (
        *reader(),
        *((sys.modules[__name__], name) for name in ("observe", "validator_identity", "_members")),
        (SourceObservation, "revalidate"),
        (SourceObservation, "record"),
    )


@dataclass(frozen=True)
class SourceObservation:
    stage: str
    _reader: object
    _reader_binding: tuple
    _members: tuple
    _bindings: tuple
    _files: tuple

    def revalidate(self):
        if any(a is not b for a, b in zip(validator_identity(self._reader), self._reader_binding, strict=True)):
            raise ValueError("shared source-region owner reader changed")
        members = _members(self._reader)
        if len(members) != len(self._members) or any(
            a is not c or b != d for (a, b), (c, d) in zip(members, self._members, strict=True)
        ):
            raise ValueError("shared source-region owner roster changed")
        for (owner, name), binding in zip(members, self._bindings, strict=True):
            current = validator_identity(getattr(owner, name))
            if any(a is not b for a, b in zip(current, binding, strict=True)):
                raise ValueError("shared source-region implementation callable changed")
        for owner, path, sha256 in self._files:
            if Path(owner.__file__).resolve() != path or digest(path) != sha256:
                raise ValueError("shared source-region implementation bytes changed")

    def record(self):
        self.revalidate()
        return {
            "stage": self.stage,
            "files": [
                {"module": module.__name__, "path": str(path), "sha256": value} for module, path, value in self._files
            ],
            "callables": [function.__module__ + "." + function.__qualname__ for _, function, _, _ in self._bindings],
            "scope": "direct source bytes and original callable identities; not loaded-code or semantic proof",
        }


def observe(stage, reader):
    """Capture stage-selected actual owners before their first callback."""
    binding = validator_identity(reader)
    members = _members(reader)
    bindings = tuple(validator_identity(getattr(owner, name)) for owner, name in members)
    modules = tuple(dict.fromkeys(sys.modules[function.__module__] for _, function, _, _ in (*bindings, binding)))
    files = tuple((module, Path(module.__file__).resolve(), digest(module.__file__)) for module in modules)
    result = SourceObservation(stage, reader, binding, members, bindings, files)
    result.revalidate()
    return result
