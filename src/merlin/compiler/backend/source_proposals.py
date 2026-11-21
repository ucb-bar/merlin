"""Admit original-source proposals against selected execution contracts.

This checks custody, source meaning identity, boundaries and the bounded shared
memory route. Implementation semantics remain the original caller's independent
validator responsibility. Provider declarations and matching hashes alone do not
prove numerical correctness or observed hardware execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from merlin.compiler.compilation import request_identity
from merlin.compiler.backend.protocol import (
    LoweringRecipe,
    ProposalBudget as ProposalBudget,
    ProposalContext as ProposalContext,
    SourcePattern as SourcePattern,
    validator_identity as _validator_identity,
)
from merlin.compiler.partition.global_plan import (
    BufferRepresentation,
    CycleInterval,
    RegionAlternative,
    ValueRepresentation,
)
from merlin.compiler.partition.global_plan_emission import _component_boundary
from merlin.target.system import SelectedTopology

# ---- sourceproposal ----------------------------------------------------------------------------


def _record(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_members():
    import sys

    from merlin.compiler.ir import boundary, semantics
    from merlin.compiler.transforms import certificates

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
    from merlin.compiler.transforms.certificates import ir_text

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
    from merlin.compiler.lower.device.plan import OffloadSource

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
    from merlin.compiler.compilation import observe

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
    from merlin.compiler.compilation import observe

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


# ---- discovery ---------------------------------------------------------------------------------
# Finite source-proposal discovery claims; no legality or execution authority.


@dataclass(frozen=True)
class ProposalPage:
    proposals: tuple
    discovery: dict

    def __post_init__(self):

        if (
            type(self.proposals) is not tuple
            or len(self.proposals) > 1024
            or any(type(p) is not SourceProposal for p in self.proposals)
            or type(self.discovery) is not dict
        ):
            raise ValueError("proposal page requires typed immutable proposals and discovery trace")


# ---- scalaroffers ------------------------------------------------------------------------------
# Source-bound callable CPU implementation claims with explicit storage ceilings.
#
# No native compilation occurs during discovery. Default zero workspace retains
# its native obligation. Opt-in workspace derives exact original tensor values;
# the caller ceiling is not an estimate or extra charge beyond source lifetimes.
# Actual native ABI/effect and arena admission remain required before linking.


def propose(source, *, resource, memory_domain, cuts=(), budget, preparation=None, workspace_limit=0):
    if preparation is not None:
        preparation.revalidate()
    source.revalidate()
    count = len(source.operations)
    if not count:
        raise ValueError("callable CPU discovery requires a nonempty source operation graph")
    if count > budget.max_source_operations:
        raise ValueError("CPU proposal discovery exceeded the source operation budget")
    for value in source.program.buffers.values():
        elements = 1
        for extent in value.shape:
            if type(extent) is not int or extent < 0:
                raise ValueError("callable CPU proposals require static storage extents")
            elements *= extent
        if elements > budget.max_tensor_elements:
            raise ValueError("CPU proposal discovery exceeded the tensor element budget")
    boundaries = sorted({0, count, *cuts})
    if any(type(n) is not int or not 0 <= n <= count for n in boundaries):
        raise ValueError("CPU proposal cuts must bind actual source operation positions")
    intervals = [(a, b, "piece") for a, b in zip(boundaries, boundaries[1:]) if (a, b) != (0, count)]
    intervals.append((0, count, "whole"))
    intervals = [(start, end, kind) for start, end, kind in intervals if end - start <= budget.max_region_operations]
    if len(intervals) > budget.max_candidates:
        raise ValueError("CPU complement proposals exceed the selected candidate budget")
    result = []
    for start, end, kind in intervals:
        nodes = tuple(range(start, end))
        inputs, outputs = proposal_boundary(source, nodes)
        metadata = (
            ("native_abi_admission", "required_before_control_or_link"),
            ("tensor_workspace_limit", "0"),
            ("workspace_status", "conditional_not_observed"),
        )
        if workspace_limit:
            from merlin.compiler.ir.boundary import live_values
            from merlin.compiler.lower.native.caller_buffers import plan

            operations = tuple(source.operations[n] for n in nodes)
            _, live_outputs = live_values(operations)
            workspace, _ = plan(operations, live_outputs)
            metadata = (
                ("native_abi_admission", "required_before_control_or_link"),
                ("tensor_workspace_limit", str(workspace_limit)),
                ("workspace_status", "source_backed_native_admission_pending"),
                (
                    "source_workspace",
                    json.dumps(
                        workspace.record() if workspace is not None else None, sort_keys=True, separators=(",", ":")
                    ),
                ),
            )
            metadata = tuple(sorted(metadata))
        alternative = RegionAlternative(
            id=f"merlin.scalar/{0 if kind == 'piece' else 1}-{kind}/{start:06d}-{end:06d}",
            nodes=nodes,
            implementation="scalar.callable.v1",
            placement=resource,
            cycles=CycleInterval.unknown("no measured full-region CPU cost; storage ceiling is not an estimate"),
            inputs=inputs,
            outputs=outputs,
            scratch_bytes=0,
            metadata=metadata,
        )
        result.append(
            SourceProposal(
                alternative,
                "merlin.scalar",
                resource,
                memory_domain,
                "logical_tensor_v1",
                source.stamp.source_sha256,
                tuple(source.operation_ordinals[n] for n in nodes),
                source_numerical_identity(source, nodes),
            )
        )
    source.revalidate()
    return tuple(result)


@dataclass(frozen=True)
class WorkspaceProposals:
    limit: int

    def propose(self, source, *, resource, memory_domain, cuts=(), budget, preparation=None):
        if type(self.limit) is not int or self.limit <= 0:
            raise ValueError("CPU workspace proposals require an explicit positive caller ceiling")
        limit = self.limit
        result = propose(
            source,
            resource=resource,
            memory_domain=memory_domain,
            cuts=cuts,
            budget=budget,
            preparation=preparation,
            workspace_limit=limit,
        )
        if self.limit != limit:
            raise ValueError("original CPU workspace proposal ceiling changed")
        return result


# ---- proposal ----------------------------------------------------------------------------------
# Operation offers and the existing explicitly selected matrix implementation.
#
# The offers/lower split adapts OPU-VME's extension_offers at revision
# 2f513d484f5d68016811139344ffdfbf8d23cd95. This branch's original selector is
# Boolean and has no cost model: its offered costs remain unknown, never zero.
#
# These are provider claims for one queried operation descriptor. They do not
# establish numerical legality, hardware capability, source-IR identity, resource
# independence, or observed execution. Existing provider and build admission stays
# mandatory. This is not the full heterogeneous region proposal protocol.


@dataclass(frozen=True)
class OperationOffer:
    """One provider's declared implementation choice, with an optional cycle estimate."""

    owner: str
    unit: str
    estimated_cycles: float | None = None
    detail: Any = None

    def __post_init__(self) -> None:
        for name, value in (("owner", self.owner), ("unit", self.unit)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"an offer requires a nonempty {name}")
        cost = self.estimated_cycles
        if cost is not None and (not math.isfinite(cost) or cost < 0):
            raise ValueError("an offer needs finite nonnegative estimated cycles or None")


@dataclass(frozen=True)
class MatrixRouting:
    """Which matrix extension a build routes contractions to, and in which configuration.

    Carried as one object so the tile edge has exactly ONE source. The alternative -- deriving the edge
    here from the caller's ``vlen`` and again at object-build time from the config's Scala -- is two
    statements of the same fact that can disagree, and the disagreement would be silent: the selector
    would choose contractions for a geometry the compiled kernel does not have.

    ``select`` overrides the default tile-filling decision, which is the seam the cost model and the
    e-graph plug into. Nothing here decides profitability on its own.

    ``support_target`` selects executable OOT support on ``MERLIN_TARGET_PATH``;
    it is not inferred from ``unit`` or ``config``. The provider declares
    ``plugin.matrix_lowering`` with geometry, selector, rewrite_prepared_file,
    load_signatures and build_object callables. Rewrite reports preserve count,
    signatures, skipped and sub_tile(); builds expose object_path, tile_edge,
    scalar_tile, scratch_bytes, parallel_tiles and to_dict().
    """

    support_target: str  # explicitly selected OOT support identity
    unit: str  # a unit declared by that provider
    config: str  # the elaborated hardware configuration
    select: Callable[[Any], bool] | None = None
    kernel: str | None = None  # explicit registry name or source path; None retains the provider default

    def __post_init__(self) -> None:
        for name, value in self.identity().items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"matrix routing requires an explicit nonempty {name}")
        if self.kernel is not None and (not isinstance(self.kernel, str) or not self.kernel.strip()):
            raise ValueError("matrix kernel selection must be a nonempty name or source path")

    def provider(self):
        """Resolve executable support explicitly, never from the unit's name."""
        from merlin.target.registry.plugins import PluginError, load_declared

        provider = load_declared(self.support_target, "matrix_lowering")
        required = ("geometry", "selector", "rewrite_prepared_file", "load_signatures", "build_object")
        missing = [name for name in required if not callable(getattr(provider, name, None))]
        if missing:
            raise PluginError(f"{self.support_target!r}: matrix_lowering lacks callable {missing}")
        return provider

    def identity(self) -> dict[str, str]:
        return {"support_target": self.support_target, "unit": self.unit, "config": self.config}

    def tile_edge(self) -> int:
        """The edge, read from this configuration's own declaration.

        Through the contract's own accessor rather than re-resolving the paths here: this used to rebuild
        them, which meant two places had to agree about where a unit's configs live -- and when the contract
        grew a second declaration site they stopped agreeing.
        """
        return self.provider().geometry(unit=self.unit, config=self.config)[0]

    def route_feature(self) -> str | None:
        """The feature name that requests this routing, as the selected extension declares it."""
        from merlin.target.registry.plugins import load_declared

        bound = load_declared(self.support_target, "extension").bind(self)
        name = getattr(bound, "route_feature_name", None)
        return name() if callable(name) else None

    def selector(self) -> Callable[[Any], bool]:
        return self.select if self.select is not None else self.provider().selector(self.tile_edge())


class MatrixExtension:
    """Bind the current matrix provider and expose its actual selection as offers.

    Provider resolution, the original selector and configuration geometry are
    evaluated in the same order as the preparation caller previously used.
    Every query invokes the original selector exactly once; no answers are
    cached by shapes or used to infer support for another operation.
    """

    def __init__(self, routing: MatrixRouting) -> None:
        self.routing = routing
        self.provider = routing.provider()
        self._selector = routing.selector()
        self.tile_edge = routing.tile_edge()

    def offers(self, operation: Any) -> tuple[OperationOffer, ...]:
        if self._selector(operation):
            return (OperationOffer(self.routing.support_target, self.routing.unit, detail=operation),)
        return ()

    def __call__(self, operation: Any) -> bool:
        return bool(self.offers(operation))

    def __getattr__(self, name: str) -> Any:
        # Some explicit selectors expose additional provider-observed metadata.
        # Forward it to the original object, as the OPU adapter does.
        return getattr(self._selector, name)

    def lower(self, prepared: Any, work: Any) -> Any:
        return self.provider.rewrite_prepared_file(prepared, work, select=self, tile_edge=self.tile_edge)


# ---- selected ----------------------------------------------------------------------------------
# Bind an admitted offer through outlining and admit its actual product receipt.
#
# Receipts establish compiler custody and explicit realization claims. Independent
# semantic verification and execution admission remain separate authorities.


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class LoweringTicket:
    partition: object
    admitted: object
    region: object
    entry: str
    _request: tuple
    _record: str
    _owners: tuple

    @classmethod
    def bind(cls, partition, region, entry, request):
        partition.revalidate()
        item = next((item for item in partition.selection.selected if item.alternative.id == region.alternative), None)
        if item is None or item.proposal.lowering is None:
            raise ValueError("selected lowering requires an original recipe-bearing offer")
        ticket = cls(partition, item, region, entry, request_identity(request), "", (partition, item, region))
        object.__setattr__(ticket, "_record", _json(ticket._describe()))
        ticket.revalidate(request)
        return ticket

    def _describe(self):
        selection, proposal, region = self.partition.selection, self.admitted.proposal, self.region
        return {
            "schema": "merlin.selected_lowering.v1",
            "proposal": asdict(proposal),
            "topology_sha256": selection.topology.sha256,
            "storage_bound_bytes": selection.storage_bound_bytes,
            "storage_scope": "source liveness plus maximum serial scratch; local device resources checked by backend",
            "partition": {
                "alternative": region.alternative,
                "provider": region.provider,
                "symbol": region.symbol,
                "source_ordinals": region.source_ordinals,
                "inputs": region.inputs,
                "outputs": region.outputs,
                "entry": self.entry,
            },
            "source_observations": {
                "proposal": self.admitted.source_observation(),
                "partition": self.partition.source_observation(),
            },
            "request": (self._request[0], str(self._request[1]), self._request[2]),
        }

    @property
    def sha256(self):
        return hashlib.sha256(self._record.encode()).hexdigest()

    def record(self):
        return json.loads(self._record)

    def revalidate(self, request):
        if any(a is not b for a, b in zip((self.partition, self.admitted, self.region), self._owners, strict=True)):
            raise ValueError("selected lowering ticket owner changed")
        self.partition.revalidate()
        self.admitted.revalidate()
        if not any(item is self.admitted for item in self.partition.selection.selected) or not any(
            region is self.region for region in self.partition.regions
        ):
            raise ValueError("selected lowering ticket lost original partition ownership")
        proposal, region = self.admitted.proposal, self.region
        if (
            proposal.lowering is None
            or region.alternative != proposal.alternative.id
            or region.provider != proposal.provider
            or region.source_ordinals != proposal.operation_ordinals
            or region.inputs != tuple(rep.buffer for rep in proposal.alternative.inputs)
            or region.outputs != tuple(rep.buffer for rep in proposal.alternative.outputs)
        ):
            raise ValueError("selected lowering ticket differs from selected source/ports/recipe")
        if request_identity(request) != self._request or _json(self._describe()) != self._record:
            raise ValueError("selected lowering ticket or request changed")
        if region.request_source(self.entry).text != request.source.text:
            raise ValueError("selected lowering body differs from original outlined source")

    def acknowledgment(self, request, result):
        """Provider's explicit receipt over actual artifacts, never a correctness certificate."""
        self.revalidate(request)
        result.verify()
        return {
            "ticket_sha256": self.sha256,
            "recipe": asdict(self.admitted.proposal.lowering),
            "artifacts": [
                {
                    "path": a.path.relative_to(result.directory).as_posix(),
                    "kind": a.kind,
                    "role": a.role,
                    "sha256": a.sha256,
                }
                for a in result.artifacts
            ],
        }

    def admit(self, request, result):
        actual = self.acknowledgment(request, result)
        # Normalize immutable tuples to their portable JSON representation.
        if _json(result.metadata.get("selected_lowering")) != _json(actual):
            raise ValueError("backend omitted or changed the exact selected lowering acknowledgment/products")
        required = set(self.admitted.proposal.lowering.artifacts)
        if not required <= {(a.kind, a.role) for a in result.artifacts}:
            raise ValueError("backend omitted selected lowering artifact kinds/roles")
