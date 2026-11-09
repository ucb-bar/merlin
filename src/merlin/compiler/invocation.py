"""Invoke a resolved compiler and admit its products against the original request."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from merlin.compiler.artifact import Compilation


def request_identity(request):
    original = (
        (request.source.text, request.source.kind, request.source.origin, request.source.sha256), request.output,
        json.dumps(asdict(request.options), sort_keys=True, allow_nan=False),
    )
    if request.preparation is None:
        return original
    request.preparation.revalidate(request.source)
    return (*original, id(request.preparation),
            json.dumps(request.preparation.record(), sort_keys=True, allow_nan=False))


def invoke(request, name, provider, *, selected=None):
    """Share validation, request custody and publication across CLI and region calls.

    This is artifact admission, not independent numerical verification. A failed
    provider can leave diagnostic products, but no successful shared manifest.
    """
    if request.output.exists():
        raise FileExistsError(f"compilation requires a fresh output directory: {request.output}")
    original = request_identity(request)
    if request.preparation is not None:
        admission = getattr(provider, "validate_preparation", None)
        if not callable(admission):
            raise ValueError("selected backend does not support explicit staged preparation admission")
        if admission(request) is not None:
            raise ValueError("staged preparation admission must return None or raise refusal")
        if request_identity(request) != original:
            raise ValueError("preparation admission changed the original compilation request")
    if selected is not None:
        selected.revalidate(request)
        provider.validate_selected(request, selected)
    else:
        provider.validate(request)
    if request_identity(request) != original:
        raise ValueError("backend validation changed the original compilation request")
    if selected is not None:
        selected.revalidate(request)
        result = provider.compile_selected(request, selected)
    else:
        result = provider.compile(request)
    if request_identity(request) != original:
        raise ValueError("backend changed the original compilation request")
    if not isinstance(result, Compilation):
        raise TypeError(f"backend {name!r} did not return a Compilation")
    if (result.directory != request.output or result.backend != name or
            result.source_sha256 != request.source.sha256):
        raise ValueError("backend compilation identity does not match the selected request")
    if selected is not None:
        selected.admit(request, result)
    selected = request.options
    if any(value is not None for value in (selected.system, selected.platform, selected.harts)):
        if result.deployment is None:
            raise ValueError("backend omitted the explicitly selected deployment contract")
        for key in ("system", "platform", "harts"):
            value = getattr(selected, key)
            if value is not None and getattr(result.deployment, key) != value:
                raise ValueError(f"backend changed explicitly selected {key}")
    result = replace(result, metadata={
        **result.metadata,
        "request": {"source_kind": request.source.kind, "source_origin": request.source.origin,
                    "source_sha256": request.source.sha256, "backend": name,
                    "options": asdict(selected),
                    **({"preparation": request.preparation.record()} if request.preparation is not None else {})},
    })
    result.write()
    return result
