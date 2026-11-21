"""Admit proposed synchronous wrappers using the original platform validator.

A compiler may publish wrapper requirements without claiming its device has
completed or its writes are CPU-visible. Only the selected caller can discharge
those requirements against its platform facts and exact emitted artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from merlin.compiler.backend.c_abi import CRegionABI, CRegionABI2, CWorkspaceABI, read_c_abi
from merlin.compiler.backend.protocol import validator_identity


def _record(compilation):
    return json.dumps(compilation.to_dict(), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class CompletionRequest:
    """Exact products and selected resource requiring completion/visibility review.

    The callback must independently check that returning zero from this wrapper
    means all submitted device work and required CPU-visibility actions finished.
    Its approval is limited to these source/artifact/topology identities.
    """

    compilation: object
    abi: CRegionABI | CRegionABI2 | CWorkspaceABI
    topology: object
    resource: str
    memory_domain: str


def admit_call(compilation, *, topology, resource, memory_domain, validate_completion=None):
    """Read an existing ABI or independently admit a proposed wrapper contract.

    Legacy published C-call contracts keep their current admission route.
    Canonical Interface device calls always need original-caller admission.
    Newly proposed requirements never become callable from a backend assertion,
    a fence spelling, a successful object build, or a truthy callback result.
    """
    existing = compilation.metadata.get("c_call")
    requirements = compilation.metadata.get("c_call_requirements")
    if existing is not None and requirements is not None:
        raise ValueError("compiled region cannot publish both admitted and proposed C-call contracts")
    boundary = compilation.metadata.get("compiler_boundary")
    interface_device = (
        isinstance(boundary, dict)
        and boundary.get("input_ir") == "merlin.interface.v1"
        and topology.resource(resource).kind != "cpu"
    )
    if requirements is None and not interface_device:
        return read_c_abi(compilation), {"status": "existing_call_contract"}
    if validate_completion is None:
        raise ValueError(
            "device wrapper needs original-caller validate_completion admission "
            "for selected-platform completion and CPU visibility"
        )
    binding = validator_identity(validate_completion)
    candidate = replace(
        compilation,
        metadata={**compilation.metadata, "c_call": requirements if requirements is not None else existing},
    )
    abi = read_c_abi(candidate)
    topology.revalidate()
    if abi.resource_kind != topology.resource(resource).kind:
        raise ValueError("proposed wrapper resource kind differs from selected execution resource")
    topology.require_access(resource, memory_domain, read=True, write=True)
    request = CompletionRequest(compilation, abi, topology, resource, memory_domain)
    identity, topology_identity, abi_identity = _record(compilation), topology.sha256, abi.record()
    if validate_completion(request) is not None:
        raise ValueError("completion admission must return None or raise its explicit refusal")
    if any(a is not b for a, b in zip(validator_identity(validate_completion), binding, strict=True)):
        raise ValueError("original platform completion validator changed")
    topology.revalidate()
    compilation.verify()
    if _record(compilation) != identity or topology.sha256 != topology_identity or abi.record() != abi_identity:
        raise ValueError("completion admission changed source, products, topology or proposed ABI")
    return abi, {
        "status": "admitted_by_original_caller",
        "source_sha256": compilation.source_sha256,
        "topology_sha256": topology_identity,
        "resource": resource,
        "memory_domain": memory_domain,
        "validator": f"{binding[1].__module__}.{binding[1].__qualname__}",
        "execution_observed": False,
    }
