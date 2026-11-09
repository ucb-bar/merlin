"""Admit original-source proposals against selected execution contracts.

This checks custody, source meaning identity, boundaries and the bounded shared
memory route. Implementation semantics remain the original caller's independent
validator responsibility. Provider declarations and matching hashes alone do not
prove numerical correctness or observed hardware execution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

from merlin.compiler.ir.semantics import validator_identity as _validator_identity
from merlin.system.topology import SelectedTopology
from merlin.xdsl_dialects.lowering.global_plan import BufferRepresentation, RegionAlternative, ValueRepresentation
from merlin.xdsl_dialects.lowering.global_plan_emission import _component_boundary


def _record(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_members():
    import sys

    from merlin.compiler.ir import boundary, semantics
    from merlin.transforms import certificates

    owner = sys.modules[__name__]
    return (
        *(
            (owner, name)
            for name in ("_source_members", "_check", "_boundary", "source_numerical_identity", "_validator_identity")
        ),
        (AdmittedProposal, "revalidate"),
        (AdmittedProposal, "source_observation"),
        *semantics._source_members(),
        (boundary, "live_values"),
        (certificates, "ir_digest"),
        (certificates, "ir_text"),
    )


def source_numerical_identity(source, nodes: tuple[int, ...]) -> str:
    """Bind original IR arithmetic/order/parameters; this is not equivalence proof."""
    from merlin.transforms.certificates import ir_text

    source.revalidate()
    if type(nodes) is not tuple or not nodes or any(type(n) is not int for n in nodes):
        raise ValueError("proposal needs explicit original operation indices")
    if tuple(sorted(set(nodes))) != nodes or nodes[0] < 0 or nodes[-1] >= len(source.operations):
        raise ValueError("proposal operation indices are outside the original source")
    # Whole-source identity binds captured constants and dynamic parameter producers,
    # while exact ordinals distinguish otherwise identical operation sites.
    record = (
        source.stamp.source_sha256,
        tuple((source.operation_ordinals[n], ir_text(source.operations[n])) for n in nodes),
    )
    return hashlib.sha256(_record(record).encode()).hexdigest()


@dataclass(frozen=True)
class LoweringRecipe:
    """An exact provider implementation claim, not numerical verification."""

    implementation_sha256: str
    pipeline: str
    guards: tuple[tuple[str, str], ...]
    artifacts: tuple[tuple[str, str], ...]
    abi: str

    def __post_init__(self):
        if (
            type(self.implementation_sha256) is not str
            or len(self.implementation_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.implementation_sha256)
        ):
            raise ValueError("lowering recipe requires an exact implementation SHA-256")
        if any(type(v) is not str or not v for v in (self.pipeline, self.abi)):
            raise ValueError("lowering recipe requires explicit pipeline and ABI identities")
        if (
            type(self.guards) is not tuple
            or not self.guards
            or any(
                type(row) is not tuple or len(row) != 2 or any(type(v) is not str or not v for v in row)
                for row in self.guards
            )
            or len(dict(self.guards)) != len(self.guards)
        ):
            raise ValueError("lowering recipe requires immutable named guard identities")
        if (
            type(self.artifacts) is not tuple
            or not self.artifacts
            or any(
                type(row) is not tuple or len(row) != 2 or any(type(v) is not str or not v for v in row)
                for row in self.artifacts
            )
            or len(set(self.artifacts)) != len(self.artifacts)
        ):
            raise ValueError("lowering recipe requires explicit unique artifact kind/role pairs")


@dataclass(frozen=True)
class SourceProposal:
    alternative: RegionAlternative
    provider: str
    resource: str
    memory_domain: str
    abi: str
    source_sha256: str
    operation_ordinals: tuple[int, ...]
    numerical_sha256: str
    required_features: tuple[str, ...] = ()
    lowering: LoweringRecipe | None = None

    def __post_init__(self):
        if type(self.alternative) is not RegionAlternative:
            raise ValueError("source proposal requires an existing typed RegionAlternative")
        self.alternative.__post_init__()
        if self.lowering is not None:
            if type(self.lowering) is not LoweringRecipe:
                raise ValueError("source proposal requires a typed selected lowering recipe")
            self.lowering.__post_init__()
            if self.lowering.abi != self.abi:
                raise ValueError("lowering recipe changes the proposal ABI")
        for field in ("provider", "resource", "memory_domain", "abi", "source_sha256", "numerical_sha256"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"source proposal requires explicit {field}")
        if type(self.operation_ordinals) is not tuple or any(type(n) is not int for n in self.operation_ordinals):
            raise ValueError("source proposal requires original immutable operation ordinals")
        if (
            type(self.required_features) is not tuple
            or any(type(f) is not str or not f for f in self.required_features)
            or len(set(self.required_features)) != len(self.required_features)
        ):
            raise ValueError("source proposal features must be immutable unique names")


def _boundary(source, nodes):
    inputs, outputs = _component_boundary(source.program, nodes)
    order = {name: index for index, name in enumerate(source.program.buffers)}

    def representations(names):
        return tuple(
            BufferRepresentation(name, ValueRepresentation("tensor_ssa", "logical", source.program.buffers[name].dtype))
            for name in sorted(names, key=order.__getitem__)
        )

    return representations(inputs), representations(outputs)


# Public protocol spelling; historical internal consumers retain exact identity.
proposal_boundary = _boundary


def _check(source, proposal, topology, *, original_cpu):
    from merlin.llvmlower.offload_plan import OffloadSource

    if (
        type(source) is not OffloadSource
        or type(proposal) is not SourceProposal
        or type(topology) is not SelectedTopology
    ):
        raise ValueError("admission requires original typed source, proposal and selected topology")
    source.revalidate()
    if source.graph_mode != "ssa":
        raise ValueError("source proposal admission requires the actual SSA graph binding")
    topology.revalidate()
    proposal.__post_init__()
    alternative = proposal.alternative
    numerical = source_numerical_identity(source, alternative.nodes)
    if proposal.source_sha256 != source.stamp.source_sha256 or proposal.operation_ordinals != tuple(
        source.operation_ordinals[n] for n in alternative.nodes
    ):
        raise ValueError("proposal belongs to different original source operations")
    if proposal.numerical_sha256 != numerical:
        raise ValueError("proposal numerical contract differs from original source")
    if (alternative.inputs, alternative.outputs) != _boundary(source, alternative.nodes):
        raise ValueError("proposal changes a live source boundary or its logical representation")
    if alternative.resident_inputs or alternative.resident_outputs:
        raise ValueError("proposal residency has no implemented materialization")
    resource = topology.resource(proposal.resource)
    if not set(proposal.required_features) <= set(resource.features):
        raise ValueError("resource does not declare the proposal's required implementation features")
    topology.require_access(
        proposal.resource,
        proposal.memory_domain,
        read=bool(alternative.inputs),
        write=bool(alternative.outputs) or bool(alternative.scratch_bytes),
    )
    memory = topology.memory(proposal.memory_domain)
    if type(alternative.scratch_bytes) is not int or alternative.scratch_bytes < 0:
        raise ValueError("proposal scratch requires an exact nonnegative byte count")
    if alternative.scratch_bytes > memory.capacity_bytes:
        raise ValueError("proposal scratch exceeds its selected memory domain")
    if original_cpu:
        from merlin.compiler.ir.semantics import region_semantic_attributes

        if region_semantic_attributes(
            source.module, source.entry, tuple(source.operations[n] for n in alternative.nodes)
        ):
            raise ValueError("original scalar fallback cannot discharge annotated source numerical obligations")
        if (proposal.provider, proposal.abi, resource.kind) != ("merlin.scalar", "inlined", "cpu"):
            raise ValueError("original CPU preservation requires the selected CPU resource and inlined ABI")
        if len(alternative.nodes) != 1 or alternative != source.cpu_alternatives[alternative.nodes[0]]:
            raise ValueError("inlined CPU alternative differs from the original program")
        if proposal.required_features:
            raise ValueError("original scalar preservation cannot add implementation feature claims")
    else:
        if proposal.abi != "logical_tensor_v1":
            raise ValueError("provider proposal has no implemented logical tensor boundary ABI")
        from xdsl.dialects.builtin import TensorType

        from merlin.compiler.ir.boundary import live_values

        inputs, outputs = live_values(tuple(source.operations[n] for n in alternative.nodes))
        if any(type(value.type) is not TensorType for value in (*inputs, *outputs)):
            raise ValueError("logical tensor region ABI does not admit raw scalar boundaries")


@dataclass(frozen=True)
class AdmittedProposal:
    source: object
    proposal: SourceProposal
    topology: SelectedTopology
    _validator: Callable | None
    _validator_binding: tuple | None
    _original_cpu: bool
    _record: str
    _owners: tuple
    _sources: object

    @property
    def alternative(self) -> RegionAlternative:
        return self.proposal.alternative

    def revalidate(self):
        owners = (self.source, self.proposal, self.topology, self._validator, self._sources)
        if len(owners) != len(self._owners) or any(a is not b for a, b in zip(owners, self._owners, strict=True)):
            raise ValueError("admitted proposal original owner changed")
        if _record(asdict(self.proposal)) != self._record:
            raise ValueError("admitted proposal changed; replan")
        self._sources.revalidate()
        _check(self.source, self.proposal, self.topology, original_cpu=self._original_cpu)
        self._sources.revalidate()
        if self._validator is not None:
            binding = _validator_identity(self._validator)
            if self._validator_binding is None or any(
                a is not b for a, b in zip(binding, self._validator_binding, strict=True)
            ):
                raise ValueError("original implementation validator changed")
            result = self._validator(self.source, self.proposal)
            self._sources.revalidate()
            if result is not None:
                raise ValueError("implementation validator must raise on refusal and otherwise return None")
            # A validator is not permitted to change either the claim or its premises.
            if _record(asdict(self.proposal)) != self._record or any(
                a is not b for a, b in zip(_validator_identity(self._validator), binding, strict=True)
            ):
                raise ValueError("implementation validation changed its original binding")
            _check(self.source, self.proposal, self.topology, original_cpu=self._original_cpu)
        elif not self._original_cpu:
            raise ValueError("provider implementation lacks original-caller validation")
        self._sources.revalidate()

    def source_observation(self):
        return self._sources.record()


def admit_proposal(source, proposal: SourceProposal, topology: SelectedTopology, *, validate_implementation):
    """Run the independent original-caller implementation validator, not provider approval.

    The caller retains ownership of capability/numerical/ABI evidence and passes
    its validating function separately from provider claims. The function must
    revalidate those evidence sources each invocation and raise on refusal.
    """
    from merlin.compiler.sourceguard import observe

    sources = observe("proposal", _source_members)
    binding = _validator_identity(validate_implementation)
    result = AdmittedProposal(
        source,
        proposal,
        topology,
        validate_implementation,
        binding,
        False,
        _record(asdict(proposal)),
        (source, proposal, topology, validate_implementation, sources),
        sources,
    )
    result.revalidate()
    return result


def admit_original_cpu(source, index: int, topology: SelectedTopology, *, resource: str, memory_domain: str):
    """Preserve original computation as inlinable CPU IR; no fake call or provider."""
    from merlin.compiler.sourceguard import observe

    sources = observe("proposal", _source_members)
    if type(index) is not int or not 0 <= index < len(source.operations):
        raise ValueError("original CPU selection requires an exact source index")
    proposal = SourceProposal(
        source.cpu_alternatives[index],
        "merlin.scalar",
        resource,
        memory_domain,
        "inlined",
        source.stamp.source_sha256,
        (source.operation_ordinals[index],),
        source_numerical_identity(source, (index,)),
    )
    result = AdmittedProposal(
        source,
        proposal,
        topology,
        None,
        None,
        True,
        _record(asdict(proposal)),
        (source, proposal, topology, None, sources),
        sources,
    )
    result.revalidate()
    return result
