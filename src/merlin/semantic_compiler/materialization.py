"""Conservative identities for provider-owned, invocation-local materialization.

This module does not copy, relabel, or authenticate compiled artifacts. A target
provider must replay its publication checks and explicitly support rebinding
before it can reuse a materialization for an alpha-equivalent request.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from hashlib import sha256
from time import perf_counter
from typing import Any, Callable, MutableMapping

from .codegen import CodegenOptions
from .model import KernelRequest


def _encode(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_request_record(
    request: KernelRequest, *, source_identity_policy: str = "preserve"
) -> dict[str, Any]:
    """Normalize explicit node references and discard only ``prov.*`` attrs.

    Node order and ordered boundary ports remain significant. Attribute values,
    including strings equal to node IDs, are opaque semantic data. The default
    preserves source_identity: only a provider that establishes that it is pure
    provenance may explicitly select the ``provenance`` policy. This function
    makes no assumption that a digest-shaped source_identity is provenance.
    """
    if source_identity_policy not in {"preserve", "provenance"}:
        raise ValueError("source identity policy must be preserve or provenance")
    names = {node.id: f"node_{index}" for index, node in enumerate(request.nodes)}
    nodes = []
    for node in request.nodes:
        row = node.record()
        row["id"] = names[node.id]
        row["inputs"] = [names[child] for child in node.inputs]
        row["attrs"] = {key: value for key, value in node.attrs if not key.startswith("prov.")}
        nodes.append(row)
    return {
        "schema": "merlin.canonical_materialization_request.v1",
        "nodes": nodes,
        "outputs": [names[name] for name in request.outputs],
        "output_storages": list(request.output_storages),
        "input_storages": [[names[name], storage] for name, storage in request.input_storages],
        "target_identity": request.target_identity,
        "lowering_policy": request.lowering_policy,
        "source_identity": request.source_identity if source_identity_policy == "preserve" else "",
        "constants": [
            {**binding.record(), "node_id": names[binding.node_id]}
            for binding in sorted(request.constants, key=lambda binding: names[binding.node_id])
        ],
        "physical_extents": sorted((names[name], extent) for name, extent in request.physical_extents),
        "boundary_layouts": sorted((names[name], layout) for name, layout in request.boundary_layouts),
        "output_layouts": list(request.output_layouts),
    }


def canonical_request(request: KernelRequest, *, source_identity_policy: str = "preserve") -> KernelRequest:
    """Construct the same checked source boundary used by the canonical key.

    Providers opt into compiling this request and must regenerate publications
    for original sources through their binding hook. Opaque attribute strings
    are never rewritten.
    """
    canonical_request_record(request, source_identity_policy=source_identity_policy)
    names = {node.id: f"node_{index}" for index, node in enumerate(request.nodes)}
    return replace(
        request,
        nodes=tuple(replace(node, id=names[node.id], inputs=tuple(names[name] for name in node.inputs),
                            attrs=tuple((key, value) for key, value in node.attrs if not key.startswith("prov.")))
                    for node in request.nodes),
        outputs=tuple(names[name] for name in request.outputs),
        input_storages=tuple((names[name], storage) for name, storage in request.input_storages),
        source_identity=request.source_identity if source_identity_policy == "preserve" else "",
        constants=tuple(replace(binding, node_id=names[binding.node_id]) for binding in request.constants),
        physical_extents=tuple((names[name], extent) for name, extent in request.physical_extents),
        boundary_layouts=tuple((names[name], layout) for name, layout in request.boundary_layouts),
    )


@dataclass(frozen=True)
class MaterializationKey:
    """Digest for lookup plus the full canonical identity for collision checks."""

    digest: str
    identity: bytes


def materialization_key(
    request: KernelRequest,
    *,
    target_choice: object,
    options: CodegenOptions,
    compilation_identity: object,
    source_identity_policy: str = "preserve",
) -> MaterializationKey:
    """Include every invocation choice supplied by the calling provider.

    target_choice and compilation_identity must be JSON values. The latter must
    account for applicable snapshot, toolchain, ABI, search limits, environment,
    and provider revision identities; this utility cannot infer those facts.
    No persistent cache compatibility or artifact reuse is implied by this key.
    """
    identity = _encode({
        "schema": "merlin.materialization_key.v1",
        "request": canonical_request_record(request, source_identity_policy=source_identity_policy),
        "source_identity_policy": source_identity_policy,
        "target_choice": target_choice,
        "codegen_options": options.identity_record(),
        "compilation_identity": compilation_identity,
    })
    return MaterializationKey(sha256(identity).hexdigest(), identity)


class MaterializationCache:
    """Collision-guarded storage, optionally backed by materialization_cache().

    Values belong to the provider, which remains responsible for validating and
    rebinding them. Buckets never match on the digest alone. Use one namespace
    per provider and materialization kind when sharing an invocation dictionary.
    """

    def __init__(self, namespace: str, storage: MutableMapping | None = None):
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("materialization cache needs an explicit namespace")
        self.namespace = namespace
        self.storage = {} if storage is None else storage

    def get(self, key: MaterializationKey, default: object = None) -> Any:
        bucket = self.storage.get(("merlin.materialization.v1", self.namespace, key.digest), ())
        for identity, value in bucket:
            if identity == key.identity:
                return value
        return default

    def put(self, key: MaterializationKey, value: object) -> None:
        bucket_key = ("merlin.materialization.v1", self.namespace, key.digest)
        bucket = self.storage.setdefault(bucket_key, [])
        for index, (identity, _) in enumerate(bucket):
            if identity == key.identity:
                bucket[index] = (key.identity, value)
                return
        bucket.append((key.identity, value))


_TOOL_OUTPUTS = ContextVar("compiler_tool_outputs", default=None)
_REGION_IDENTITY = ContextVar("compiler_region_identity", default=None)


@contextmanager
def materialization_region(key: MaterializationKey):
    """Partition exact tool reuse by a provider-approved canonical region.

    Full identity bytes, rather than their digest, are embedded in the inner
    tool key. This scope never skips original compilation or publication.
    """
    if not isinstance(key, MaterializationKey):
        raise TypeError("materialization region needs a canonical key")
    token = _REGION_IDENTITY.set(key.identity)
    try:
        yield
    finally:
        _REGION_IDENTITY.reset(token)



@contextmanager
def materialization_session(*, enabled: bool, reuse_current: bool = False):
    """Scope deterministic compiler tool reuse to one application compilation.

    Entries create fresh caches, including nested invocations. Only explicit
    ``reuse_current=True`` joins an active session with the same enabled state;
    source verification always starts an independent disabled session.
    Statistics are caller-owned measurements, never authenticated publication
    bytes. Disabled sessions measure calls but execute every invocation.
    """
    if type(enabled) is not bool:
        raise TypeError("materialization session needs an explicit Boolean")
    if type(reuse_current) is not bool:
        raise TypeError("materialization session reuse needs an explicit Boolean")
    session = _TOOL_OUTPUTS.get()
    if reuse_current and session is not None and (session[0] is not None) == enabled:
        yield session[1]
        return
    stats = {"region_replays": 0, "region_calls": 0, "region_compiles": 0,
             "region_materializations": 0, "region_hits": 0,
             "calls": 0, "hits": 0, "misses": 0, "elapsed_seconds": 0.0}
    token = _TOOL_OUTPUTS.set(({} if enabled else None, stats))
    start = perf_counter()
    try:
        yield stats
    finally:
        stats["elapsed_seconds"] = perf_counter() - start
        _TOOL_OUTPUTS.reset(token)


def cached_tool_output(
    *, namespace: str, input_bytes: bytes, compilation_identity: object, invoke: Callable[[], bytes],
) -> bytes:
    """Reuse exact deterministic tool output while preserving original exports.

    Providers may opt in only for side-effect-free invocations. The identity
    must include executable content, complete arguments, relevant environment,
    snapshot/provider revisions, options and search limits. ``input_bytes`` is
    the complete tool input, including constants; no names or provenance are
    removed here. A provider must keep its per-original-request compilation,
    publication and replay checks around this narrowly scoped tool call.

    Only successful byte results are retained. Outside an application session,
    or in a disabled session, every call executes. Full identity comparison
    guards digest collisions, and namespaces separate provider tool contracts.
    """
    if type(input_bytes) is not bytes:
        raise TypeError("compiler tool input must be immutable bytes")
    if not isinstance(compilation_identity, dict) or not compilation_identity:
        raise ValueError("compiler tool reuse needs an explicit compilation identity")
    identity = _encode({
        "schema": "merlin.compiler_tool_output.v1",
        "compilation_identity": compilation_identity,
        "region_identity": (_REGION_IDENTITY.get().hex() if _REGION_IDENTITY.get() is not None else None),
    }) + b"\0" + input_bytes
    key = MaterializationKey(sha256(identity).hexdigest(), identity)
    session = _TOOL_OUTPUTS.get()
    storage, stats = session if session is not None else (None, None)
    # Validate the namespace even when reuse is disabled.
    cache = MaterializationCache(namespace, storage)
    if stats is not None:
        stats["calls"] += 1
    missing = object()
    cached = cache.get(key, missing) if storage is not None else missing
    if cached is not missing:
        stats["hits"] += 1
        return cached
    if stats is not None:
        stats["misses"] += 1
    output = invoke()
    if type(output) is not bytes:
        raise TypeError("compiler tool output must be immutable bytes")
    if storage is not None:
        cache.put(key, output)
    return output
