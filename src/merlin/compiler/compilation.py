"""Compilation request orchestration and backend artifact admission.

The manifest records production and compatibility claims. It is not independent
numerical verification, hardware qualification, or permission to execute code.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from merlin.compiler.records import (
    SCHEMA as SCHEMA,
    Artifact as Artifact,
    Compilation,
    CompileOptions,
    Deployment as Deployment,
    Source,
    digest,
    _positive as _positive,
    _text as _text,
)

# ---- artifact ----------------------------------------------------------------------------------


# ---- request -----------------------------------------------------------------------------------
# Framework-independent input and option records for public compilation.


@dataclass(frozen=True)
class CompileRequest:
    source: Source
    output: Path
    options: CompileOptions
    preparation: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", Path(self.output).resolve())
        if self.preparation is not None:
            from merlin.compiler.frontend.staged import StagedPreparation

            if type(self.preparation) is not StagedPreparation:
                raise ValueError("compilation preparation requires an original checked staged binding")
            self.preparation.revalidate(self.source)


# ---- invocation --------------------------------------------------------------------------------
# Invoke a resolved compiler and admit its products against the original request.


def request_identity(request):
    original = (
        (request.source.text, request.source.kind, request.source.origin, request.source.sha256),
        request.output,
        json.dumps(asdict(request.options), sort_keys=True, allow_nan=False),
    )
    if request.preparation is None:
        return original
    request.preparation.revalidate(request.source)
    return (
        *original,
        id(request.preparation),
        json.dumps(request.preparation.record(), sort_keys=True, allow_nan=False),
    )


def validate_request(request, provider, *, selected=None):
    """Validate through the same Merlin-owned boundary used by compilation.

    System compilation uses this before producing any region, preserving its
    all-regions validation transaction. Returned handoffs bind transformed input
    separately from original-source tickets and staged precision preparation.
    """
    from merlin.compiler.backend.boundary import prepare

    original = request_identity(request)
    if request.preparation is not None:
        admission = getattr(provider, "validate_preparation", None)
        if not callable(admission):
            raise ValueError("selected backend does not support explicit staged preparation admission")
        if admission(request) is not None:
            raise ValueError("staged preparation admission must return None or raise refusal")
        if request_identity(request) != original:
            raise ValueError("preparation admission changed the original compilation request")
    handoff = prepare(request, provider, selected)
    actual = handoff.request if handoff is not None else request
    ticket = handoff.selected if handoff is not None else selected
    actual_identity = request_identity(actual)
    if ticket is not None:
        ticket.revalidate(actual)
        provider.validate_selected(actual, ticket)
    else:
        provider.validate(actual)
    if request_identity(request) != original or request_identity(actual) != actual_identity:
        raise ValueError("backend validation changed the original compilation request")
    if handoff is not None:
        handoff.revalidate()
    return handoff


def invoke(request, name, provider, *, selected=None):
    """Share input preparation, custody and artifact admission for every caller.

    Interface providers receive only Merlin-prepared Interface IR. Legacy
    providers retain their declared input. Neither path proves equivalence.
    """
    if request.output.exists():
        raise FileExistsError(f"compilation requires a fresh output directory: {request.output}")
    original = request_identity(request)
    handoff = validate_request(request, provider, selected=selected)
    actual = handoff.request if handoff is not None else request
    ticket = handoff.selected if handoff is not None else selected
    actual_identity = request_identity(actual)
    if ticket is not None:
        ticket.revalidate(actual)
        result = provider.compile_selected(actual, ticket)
    else:
        result = provider.compile(actual)
    if request_identity(request) != original or request_identity(actual) != actual_identity:
        raise ValueError("backend changed the original compilation request")
    if not isinstance(result, Compilation):
        raise TypeError(f"backend {name!r} did not return a Compilation")
    if result.directory != actual.output or result.backend != name or result.source_sha256 != actual.source.sha256:
        raise ValueError("backend compilation identity does not match the selected request")
    if ticket is not None:
        ticket.admit(actual, result)
    if handoff is not None:
        result = handoff.publish(result)
    selected = request.options
    if any(value is not None for value in (selected.system, selected.platform, selected.harts)):
        if result.deployment is None:
            raise ValueError("backend omitted the explicitly selected deployment contract")
        for key in ("system", "platform", "harts"):
            value = getattr(selected, key)
            if value is not None and getattr(result.deployment, key) != value:
                raise ValueError(f"backend changed explicitly selected {key}")
    result = replace(
        result,
        metadata={
            **result.metadata,
            "request": {
                "source_kind": request.source.kind,
                "source_origin": request.source.origin,
                "source_sha256": request.source.sha256,
                "backend": name,
                "options": asdict(selected),
                **({"preparation": request.preparation.record()} if request.preparation is not None else {}),
            },
        },
    )
    result.write()
    return result


# ---- sourceguard -------------------------------------------------------------------------------
# Observe explicit stage-owned source files and original Python callables.
#
# This leaf discovers no stages or paths. Its callers supply fixed internal owner
# readers; observations grant no source access, semantic or loaded-code proof.


from merlin.compiler.backend.protocol import validator_identity


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
