"""Explicit source-discovery components for the ordinary system compiler.

Compiler claims pass through original-caller source/target admission. A selected
component supplies no application accuracy policy or independent hardware truth.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from merlin.compiler.backend.sourceproposal import SourceProposal, _validator_identity, admit_proposal
from merlin.compiler.pipeline import RegionBackend
from merlin.compiler.request import CompileOptions


def _source_members():
    import sys

    from merlin.compiler.backend import sourceproposal
    from merlin.compiler.ir import region

    return ((sys.modules[__name__], "_source_members"), (sys.modules[__name__], "_admit_cpu"),
            (Component, "discover"), *sourceproposal._source_members(), *region._source_members())


@dataclass(frozen=True)
class Component:
    binding: RegionBackend
    resource: str
    memory_domain: str
    validate_implementation: Callable
    target: object | None = None
    cpu_resource: str | None = None
    proposals: Callable | None = None
    complements: bool = False
    required_resource_kind: str | None = None

    def __post_init__(self):
        if type(self.binding) is not RegionBackend or any(type(s) is not str or not s for s in (
                self.resource, self.memory_domain)):
            raise ValueError("system components need a typed backend binding and explicit resource/memory")
        _validator_identity(self.validate_implementation)
        if self.proposals is not None:
            _validator_identity(self.proposals)
        if type(self.complements) is not bool or self.complements and self.proposals is None:
            raise ValueError("complement discovery requires an explicit bounded proposal callback")
        if self.target is not None and (self.proposals is not None or not self.cpu_resource):
            raise ValueError("selected-target discovery uses its provider hook and an explicit CPU resource")

    def discover(self, source, topology, provider, name, budget, cuts, *, preparation=None):
        from merlin.compiler.sourceguard import observe

        sources = observe("discovery", _source_members)
        if (self.required_resource_kind is not None and
                topology.resource(self.resource).kind != self.required_resource_kind):
            raise ValueError("component implementation kind differs from the selected execution resource")
        if preparation is not None:
            preparation.revalidate_ir(source)
        if self.target is not None:
            from merlin.compiler.planning.fusion import discover_fusion_candidates

            candidates, report = discover_fusion_candidates(
                source, provider=provider, target=self.target, topology=topology, resource=self.resource,
                memory_domain=self.memory_domain, cpu_resource=self.cpu_resource,
                validate_implementation=self.validate_implementation, budget=budget,
                **({key: preparation.record()[key] for key in
                    ("numerical_policy_sha256", "preparation_sha256")} if preparation is not None else {}),
            )
            sources.revalidate()
            report = {**report, "source_observation": sources.record()}
            return tuple(p for p in candidates if p.proposal.abi != "inlined"), report
        hook = self.proposals or getattr(provider, "propose", None)
        if not callable(hook):
            raise ValueError(f"selected component {name!r} has no admitted source proposal hook")
        hook_binding = _validator_identity(hook)
        validator = _validator_identity(self.validate_implementation)
        def custody():
            sources.revalidate()
            if preparation is not None:
                preparation.revalidate_ir(source)
            source.revalidate()
            topology.revalidate()
            current = self.proposals or getattr(provider, "propose", None)
            if any(a is not b for a, b in zip(_validator_identity(current), hook_binding, strict=True)):
                raise ValueError("component source proposal hook changed during discovery/admission")
            if any(a is not b for a, b in zip(
                    _validator_identity(self.validate_implementation), validator, strict=True)):
                raise ValueError("original implementation admission changed")
        kwargs = {"resource": self.resource, "memory_domain": self.memory_domain}
        if self.complements:
            kwargs.update(cuts=cuts, budget=budget)
        if preparation is not None:
            import inspect

            kwargs["preparation"] = preparation
            try:
                inspect.signature(hook).bind(source, **kwargs)
            except (TypeError, ValueError) as error:
                raise ValueError("component proposal hook does not support explicit staged preparation") from error
        custody()
        offered = hook(source, **kwargs)
        custody()
        if type(offered) is not tuple or len(offered) > budget.max_candidates:
            raise ValueError("component proposals must be a bounded immutable tuple")
        accepted, reasons = [], []
        def check(bound, proposal):
            custody()
            if self.validate_implementation(bound, proposal) is not None:
                raise ValueError("implementation admission must return None or raise its refusal")
            custody()
        for proposal in offered:
            if (type(proposal) is not SourceProposal or proposal.provider != name or
                    proposal.resource != self.resource or proposal.memory_domain != self.memory_domain):
                raise ValueError("component proposal changed its selected provider/resource/memory identity")
            try:
                accepted.append(admit_proposal(source, proposal, topology, validate_implementation=check))
            except ValueError as error:
                reasons.append({"alternative": proposal.alternative.id, "status": "refused", "reason": str(error)})
            else:
                reasons.append({"alternative": proposal.alternative.id, "status": "admitted"})
            custody()
        return tuple(accepted), {"provider": name, "offers": reasons, "qualification": "not_established",
                                  "source_observation": sources.record(),
                                  "source_sha256": source.stamp.source_sha256,
                                  "scope": "bounded declared source proposals; actual lowering obligations remain"}


def _admit_cpu(source, proposal):
    """Original-caller applicability checks, independent of CPU provider claims.

    Exact source identity and live ports are checked by shared proposal admission.
    These checks grant no numerical tolerance, compiler equivalence or hardware
    qualification. Native storage/ABI obligations are deliberately still pending.
    """
    from xdsl.dialects import builtin

    from merlin.compiler.ir.semantics import region_semantic_attributes

    operations = tuple(source.operations[n] for n in proposal.alternative.nodes)
    if region_semantic_attributes(source.module, source.entry, operations):
        raise ValueError("built-in scalar admission cannot preserve annotated source numerical obligations")

    from merlin.compiler.backend.cabi import TensorPort
    from merlin.compiler.ir.boundary import live_values

    if proposal.alternative.implementation != "scalar.callable.v1" or proposal.alternative.scratch_bytes != 0:
        raise ValueError("CPU proposal differs from the selected conditional implementation envelope")
    metadata = dict(proposal.alternative.metadata)
    if metadata.get("native_abi_admission") != "required_before_control_or_link":
        raise ValueError("CPU proposal omitted its unresolved native workspace/ABI obligation")
    inputs, outputs = live_values(operations)
    if metadata.get("workspace_status") == "source_backed_native_admission_pending":
        import json

        from merlin.compiler.lower.llvm.workspace import plan

        limit = metadata.get("tensor_workspace_limit", "")
        if not limit.isdecimal() or str(int(limit)) != limit or int(limit) <= 0:
            raise ValueError("CPU source workspace needs an exact positive original ceiling")
        workspace, _ = plan(operations, outputs)
        expected = workspace.record() if workspace is not None else None
        if (metadata.get("source_workspace") != json.dumps(expected, sort_keys=True, separators=(",", ":"))
                or workspace is not None and workspace.bytes > int(limit)):
            raise ValueError("CPU source workspace differs from original source values or exceeds caller ceiling")
    elif (metadata.get("tensor_workspace_limit") != "0" or
          metadata.get("workspace_status") != "conditional_not_observed"):
        raise ValueError("CPU proposal omitted its unresolved native workspace/ABI obligation")
    if not outputs:
        raise ValueError("callable CPU region requires live tensor outcomes")
    for value in (*inputs, *outputs):
        TensorPort.from_type(value.type)
    for root in operations:
        for operation in root.walk():
            if (isinstance(operation, builtin.UnregisteredOp) or
                    operation.name.partition(".")[0] not in {"arith", "math", "tensor", "linalg", "scf", "cf"}):
                raise ValueError("CPU source applicability has unknown effects or operation semantics")
            if any(isinstance(v.type, (builtin.MemRefType, builtin.UnrankedMemRefType, builtin.VectorType))
                   for v in (*operation.operands, *operation.results)):
                raise ValueError("CPU source applicability requires immutable tensor/scalar values")
            flags = operation.properties.get("fastmath")
            if flags is not None and flags.data:
                raise ValueError("CPU system admission has no separately approved fast-math policy")


def scalar_cpu(*, resource, memory_domain, options: CompileOptions, entry="kernel"):
    """Select callable scalar CPU computation explicitly, without inventing a device.

    Existing scalar target/layout options remain caller selections. Object output
    and the internal region entry are consequences of this callable route.
    """
    from merlin.compiler.backend import scalar, scalaroffers

    if type(options) is not CompileOptions:
        raise ValueError("scalar CPU component needs explicit typed compiler options")
    parameters = dict(options.parameters)
    if parameters.get("scalar.emit", "object") != "object" or parameters.get("scalar.call-entry", entry) != entry:
        raise ValueError("scalar CPU component requires object output and its selected region entry")
    parameters.update({"scalar.emit": "object", "scalar.call-entry": entry})
    ceiling = parameters.get("scalar.workspace-bytes", 0)
    if type(ceiling) is not int or ceiling < 0:
        raise ValueError("scalar.workspace-bytes must be a nonnegative original caller ceiling")
    proposals = scalaroffers.WorkspaceProposals(ceiling).propose if ceiling else scalaroffers.propose
    return Component(RegionBackend(scalar, entry, replace(options, parameters=parameters)), resource, memory_domain,
                     _admit_cpu, proposals=proposals, complements=True, required_resource_kind="cpu")
