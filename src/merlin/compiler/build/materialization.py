"""Conservative identities for provider-owned, invocation-local materialization.

This module does not copy, relabel, or authenticate compiled artifacts. A target
provider must replay its publication checks and explicitly support rebinding
before it can reuse a materialization for an alpha-equivalent request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Callable, MutableMapping

from merlin.compiler.semantic.codegen import CodegenOptions
from merlin.compiler.semantic.model import KernelRequest


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
    """Collision-guarded storage keyed by a canonical materialization identity.

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


def cached_tool_output(
    *, namespace: str, input_bytes: bytes, compilation_identity: object, invoke: Callable[[], bytes],
) -> bytes:
    """Run one deterministic compiler tool invocation with checked byte input and output.

    No output is retained, so every call executes. The caller still names its
    tool contract and the complete compilation identity (executable content,
    arguments, environment, revisions, options and search limits) that exact
    reuse would require.
    """
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("compiler tool invocation needs an explicit namespace")
    if type(input_bytes) is not bytes:
        raise TypeError("compiler tool input must be immutable bytes")
    if not isinstance(compilation_identity, dict) or not compilation_identity:
        raise ValueError("compiler tool reuse needs an explicit compilation identity")
    output = invoke()
    if type(output) is not bytes:
        raise TypeError("compiler tool output must be immutable bytes")
    return output
