"""Plan and compile the actual post-frontend program through explicit components.

This ordinary backend binds source only after public capture/import/immutable
binding. Discovery, independent admission and bounded selection precede existing
partition/control/build stages. No precision reconsideration or runtime fallback
is implicit, and a failed conditional lowering obligation fails this compile.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from merlin.compiler.backend.providers import resolve
from merlin.compiler.backend.interface import CompilerInterface
from merlin.compiler.backend.source_proposals import (
    LoweringTicket,
    ProposalBudget,
    SourceProposal,
    _validator_identity,
    admit_proposal,
)
from merlin.compiler.build.image import CImageLink, link_selected_image
from merlin.compiler.build.regions import RegionProduct, control_entry, emit_selected_c
from merlin.compiler.compilation import (
    Artifact,
    Compilation,
    CompileOptions,
    CompileRequest,
    Source,
    invoke,
    request_identity,
    validate_request,
)
from merlin.compiler.partition.regions import partition_execution
from merlin.compiler.planning.cover_selection import SelectedExecution, select_execution
from merlin.compiler.transforms.certificates import ir_text
from merlin.compiler.transforms.pipeline_state import PipelineContext
from merlin.target.system import SelectedTopology

# ---- pipeline ----------------------------------------------------------------------------------
# Compile all selected regions through the public provider contract.
#
# The default product is a checked partition and its region artifacts. Explicit
# C control emission can consume admitted synchronous wrappers; an optional
# caller-selected image linker adds startup and a deployment-bound ELF. None of
# these production stages establishes numerical or device-execution correctness.


@dataclass(frozen=True)
class RegionBackend:
    """Bind one proposal provider to its selected compiler and input entry ABI."""

    backend: Any
    entry: str
    options: CompileOptions = field(default_factory=CompileOptions)

    def __post_init__(self):
        if type(self.entry) is not str or not self.entry.strip():
            raise ValueError("region backend requires an explicit source entry name")
        if type(self.options) is not CompileOptions:
            raise ValueError("region backend requires typed compile options")


class PlannedCompiler:
    """An explicit backend object for ``merlin.api.compile`` with a selected plan.

    Provider keys must exactly match selected proposal provider identities.
    CPU source remains inline. Every region is validated before any backend
    compiles; a failure stops the build without replanning or claiming fallback.
    """

    name = "merlin.planned"

    def __init__(
        self,
        selection: SelectedExecution,
        backends: dict[str, RegionBackend],
        *,
        image: CImageLink | None = None,
        validate_completion=None,
    ):
        if type(selection) is not SelectedExecution or type(backends) is not dict:
            raise ValueError("planned compilation requires a selection and explicit backend bindings")
        if any(type(key) is not str or type(value) is not RegionBackend for key, value in backends.items()):
            raise ValueError("planned backend bindings must name typed RegionBackend records")
        if image is not None and type(image) is not CImageLink:
            raise ValueError("planned image requires a typed CImageLink selection")
        self.selection = selection
        self.backends = dict(backends)
        self._selection = selection
        self._bindings = tuple(self.backends.items())
        self._options = self._option_identity()
        self.image = image
        self._image = image
        self._completion_validator = validate_completion
        self._completion_binding = None if validate_completion is None else _validator_identity(validate_completion)

    def _option_identity(self):
        return json.dumps(
            {key: {"entry": value.entry, "options": asdict(value.options)} for key, value in self.backends.items()},
            sort_keys=True,
            allow_nan=False,
        )

    def _revalidate(self):
        _check_completion_validator(self._completion_validator, self._completion_binding)
        if self.image is not self._image:
            raise ValueError("planned image binding changed")
        if self.image is not None:
            self.image.revalidate()
        if self.selection is not self._selection or len(self.backends) != len(self._bindings):
            raise ValueError("planned compiler binding changed")
        if any(self.backends.get(key) is not value for key, value in self._bindings):
            raise ValueError("planned backend binding changed")
        if self._option_identity() != self._options:
            raise ValueError("planned backend options changed")
        self.selection.revalidate()

    def validate(self, request):
        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

        self._revalidate()
        if request.source.kind != "mlir":
            raise ValueError("planned compilation currently requires imported MLIR")
        control = control_entry(
            request.options, self.selection.source.entry, self.image.deployment if self.image is not None else None
        )
        if control is not None and any(item.proposal.abi == "inlined" for item in self.selection.selected):
            raise ValueError("C control needs a compiled wrapper for every operation; inline CPU cannot be omitted")
        actual = parse_mlir_text(request.source.text)
        if not actual.is_structurally_equivalent(self.selection.source.module):
            raise ValueError("compilation source differs from the original selected program")
        expected = {item.proposal.provider for item in self.selection.selected if item.proposal.abi != "inlined"}
        if set(self.backends) != expected:
            raise ValueError("backend bindings must exactly cover selected non-inline providers")

    def compile(self, request):
        self.validate(request)
        control = control_entry(
            request.options, self.selection.source.entry, self.image.deployment if self.image is not None else None
        )
        if request.output.exists():
            raise FileExistsError(f"compilation requires fresh output: {request.output}")
        partition = partition_execution(self.selection)
        services = {}
        for key, binding in self.backends.items():
            name, provider = resolve(binding.backend, group="merlin.backends", methods=("validate", "compile"))
            if name != key:
                raise ValueError("resolved backend identity differs from its selected proposal provider")
            services[key] = (name, provider)
        work = []
        for index, region in enumerate(partition.regions):
            binding = self.backends[region.provider]
            serialized = region.request_source(binding.entry)
            source = Source(serialized.text, origin=f"{request.source.origin or '<input>'}#{region.symbol}")
            child = CompileRequest(source, request.output / "regions" / str(index), binding.options)
            name, provider = services[region.provider]
            item = next(item for item in self.selection.selected if item.alternative.id == region.alternative)
            ticket = None
            if item.proposal.lowering is not None:
                if any(
                    not callable(getattr(provider, method, None))
                    for method in ("validate_selected", "compile_selected")
                ):
                    raise ValueError("selected offer requires explicit validate_selected/compile_selected support")
                ticket = LoweringTicket.bind(partition, region, binding.entry, child)
            before = request_identity(child)
            validate_request(child, provider, selected=ticket)
            if request_identity(child) != before:
                raise ValueError("region validation changed its original request")
            self._revalidate()
            partition.revalidate()
            work.append((region, binding.entry, child, name, provider, ticket))
        # Publish stage products only after the full roster validates.
        request.output.mkdir(parents=True)
        if work:
            (request.output / "regions").mkdir()
        products, records, region_products = [], [], []
        for region, entry, child, name, provider, ticket in work:
            self._revalidate()
            partition.revalidate()
            compiled = invoke(child, name, provider, selected=ticket)
            self._revalidate()
            partition.revalidate()
            # A region ELF/binary is not the whole-program executable. Retain its
            # original role in the nested manifest and qualify the aggregate role.
            products.extend(Artifact(a.path, a.kind, "region." + a.role, a.sha256) for a in compiled.artifacts)
            products.append(Artifact.from_file(compiled.manifest, kind="json", role="region.manifest"))
            region_products.append(RegionProduct(entry, compiled))
            records.append(
                {
                    "alternative": region.alternative,
                    "provider": region.provider,
                    "partition_entry": region.symbol,
                    "backend_entry": entry,
                    "source_ordinals": list(region.source_ordinals),
                    "inputs": list(region.inputs),
                    "outputs": list(region.outputs),
                    "lowering": ticket.record() if ticket is not None else None,
                    "lowering_binding": "selected_offer" if ticket is not None else "legacy_source_and_options",
                    "compilation": compiled.manifest.relative_to(request.output).as_posix(),
                }
            )
        stage = "compiled_regions"
        remaining = [
            "control_lowering",
            "region_abi_adaptation",
            "link",
            "deployment_admission",
            "backend_correctness_verification",
            "execution_evidence",
        ]
        if control is not None:
            description, files = emit_selected_c(
                partition,
                tuple(region_products),
                name=control,
                directory=request.output,
                validate_completion=self._completion_validator,
            )
            control_directory = request.output / "control"
            control_directory.mkdir()
            for member, text in files.items():
                path = control_directory / member
                path.write_text(text, encoding="utf-8")
                products.append(
                    Artifact.from_file(path, kind="c-source" if path.suffix == ".c" else "c-header", role="control")
                )
            description["link_inputs"] = [f"control/{control}.c", *description["link_inputs"]]
            path = control_directory / "control.json"
            path.write_text(json.dumps(description, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            products.append(Artifact.from_file(path, kind="json", role="control_plan"))
            stage = "c_dispatch"
            remaining = ["link", "deployment_admission", "backend_correctness_verification", "execution_evidence"]
            if self.image is not None:
                products.extend(
                    link_selected_image(
                        self.image, partition, tuple(region_products), description, directory=request.output
                    )
                )
                self._revalidate()
                stage = "linked_image"
                remaining = ["platform_startup_qualification", "backend_correctness_verification", "execution_evidence"]
        source_path = request.output / "partition.mlir"
        source_path.write_text(ir_text(partition.module), encoding="utf-8")
        report_path = request.output / "plan.json"
        report = {
            "selection": self.selection.record(),
            "partition": asdict(partition.verification),
            "source_observation": partition.source_observation(),
            "emission": partition.emission.receipt(),
            "regions": records,
            "stage": stage,
            "execution_verified": False,
            "remaining": remaining,
        }
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        products.extend(
            (
                Artifact.from_file(source_path, kind="mlir", role="partitioned_source"),
                Artifact.from_file(report_path, kind="json", role="execution_plan"),
            )
        )
        partition.revalidate()
        self._revalidate()
        result = Compilation(
            request.output,
            self.name,
            request.source.sha256,
            tuple(products),
            deployment=self.image.deployment if self.image is not None else None,
            metadata={"stage": stage, "execution_verified": False},
        )
        result.verify()
        return result


# ---- components --------------------------------------------------------------------------------
# Explicit source-discovery components for the ordinary system compiler.
#
# Compiler claims pass through original-caller source/target admission. A selected
# component supplies no application accuracy policy or independent hardware truth.


def _source_members():
    import sys

    from merlin.compiler.backend import source_proposals as sourceproposal
    from merlin.compiler.backend import interface
    from merlin.compiler.ir import region
    from merlin.compiler.planning import interface_candidates

    return (
        (sys.modules[__name__], "_source_members"),
        (sys.modules[__name__], "_admit_cpu"),
        (Component, "discover"),
        *sourceproposal._source_members(),
        *interface._source_members(),
        *region._source_members(),
        *interface_candidates._source_members(),
    )


def _check_completion_validator(callback, binding):
    current = None if callback is None else _validator_identity(callback)
    if (
        (current is None) != (binding is None)
        or current is not None
        and any(a is not b for a, b in zip(current, binding, strict=True))
    ):
        raise ValueError("original completion admission changed")


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
        if type(self.binding) is not RegionBackend or any(
            type(s) is not str or not s for s in (self.resource, self.memory_domain)
        ):
            raise ValueError("system components need a typed backend binding and explicit resource/memory")
        _validator_identity(self.validate_implementation)
        if self.proposals is not None:
            _validator_identity(self.proposals)
        if type(self.complements) is not bool or self.complements and self.proposals is None:
            raise ValueError("complement discovery requires an explicit bounded proposal callback")
        if self.target is not None and (self.proposals is not None or not self.cpu_resource):
            raise ValueError("selected-target discovery uses its provider hook and an explicit CPU resource")

    def discover(self, source, topology, provider, name, budget, cuts, *, preparation=None):
        from merlin.compiler.compilation import observe

        sources = observe("discovery", _source_members)
        if (
            self.required_resource_kind is not None
            and topology.resource(self.resource).kind != self.required_resource_kind
        ):
            raise ValueError("component implementation kind differs from the selected execution resource")
        if preparation is not None:
            preparation.revalidate_ir(source)
        if self.proposals is None and getattr(provider, "input_ir", None) == "merlin.interface.v1":
            from merlin.compiler.planning.interface_candidates import discover_interface_candidates

            candidates, report = discover_interface_candidates(
                source,
                provider=provider,
                name=name,
                topology=topology,
                resource=self.resource,
                memory_domain=self.memory_domain,
                entry=self.binding.entry,
                validate_implementation=self.validate_implementation,
                budget=budget,
                target=self.target,
                preparation=preparation,
            )
            sources.revalidate()
            return candidates, {**report, "source_observation": sources.record()}
        if self.target is not None:
            from merlin.compiler.planning.fusion_admission import discover_fusion_candidates

            candidates, report = discover_fusion_candidates(
                source,
                provider=provider,
                target=self.target,
                topology=topology,
                resource=self.resource,
                memory_domain=self.memory_domain,
                cpu_resource=self.cpu_resource,
                validate_implementation=self.validate_implementation,
                budget=budget,
                **(
                    {key: preparation.record()[key] for key in ("numerical_policy_sha256", "preparation_sha256")}
                    if preparation is not None
                    else {}
                ),
            )
            sources.revalidate()
            report = {**report, "source_observation": sources.record()}
            return tuple(p for p in candidates if p.proposal.abi != "inlined"), report
        hook = self.proposals or getattr(provider, "propose", None)
        if not callable(hook):
            raise ValueError(f"selected component {name!r} has no admitted source proposal hook")
        compiler_interface = CompilerInterface(provider, name)
        hook_binding = _validator_identity(hook)
        validator = _validator_identity(self.validate_implementation)

        def custody():
            compiler_interface.revalidate()
            sources.revalidate()
            if preparation is not None:
                preparation.revalidate_ir(source)
            source.revalidate()
            topology.revalidate()
            current = self.proposals or getattr(provider, "propose", None)
            if any(a is not b for a, b in zip(_validator_identity(current), hook_binding, strict=True)):
                raise ValueError("component source proposal hook changed during discovery/admission")
            if any(
                a is not b for a, b in zip(_validator_identity(self.validate_implementation), validator, strict=True)
            ):
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
            compiler_interface.validate_recipe(proposal.lowering)
            if self.validate_implementation(bound, proposal) is not None:
                raise ValueError("implementation admission must return None or raise its refusal")
            custody()

        for proposal in offered:
            if (
                type(proposal) is not SourceProposal
                or proposal.provider != name
                or proposal.resource != self.resource
                or proposal.memory_domain != self.memory_domain
            ):
                raise ValueError("component proposal changed its selected provider/resource/memory identity")
            try:
                accepted.append(admit_proposal(source, proposal, topology, validate_implementation=check))
            except ValueError as error:
                reasons.append({"alternative": proposal.alternative.id, "status": "refused", "reason": str(error)})
            else:
                reasons.append({"alternative": proposal.alternative.id, "status": "admitted"})
            custody()
        report = {
            "provider": name,
            "offers": reasons,
            "qualification": "not_established",
            "source_observation": sources.record(),
            "source_sha256": source.stamp.source_sha256,
            "scope": "bounded declared source proposals; actual lowering obligations remain",
            "compiler_interface": compiler_interface.record(),
        }
        custody()
        return tuple(accepted), report


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

    from merlin.compiler.backend.c_abi import TensorPort
    from merlin.compiler.ir.boundary import live_values

    if proposal.alternative.implementation != "scalar.callable.v1" or proposal.alternative.scratch_bytes != 0:
        raise ValueError("CPU proposal differs from the selected conditional implementation envelope")
    metadata = dict(proposal.alternative.metadata)
    if metadata.get("native_abi_admission") != "required_before_control_or_link":
        raise ValueError("CPU proposal omitted its unresolved native workspace/ABI obligation")
    inputs, outputs = live_values(operations)
    if metadata.get("workspace_status") == "source_backed_native_admission_pending":
        import json

        from merlin.compiler.lower.native.caller_buffers import plan

        limit = metadata.get("tensor_workspace_limit", "")
        if not limit.isdecimal() or str(int(limit)) != limit or int(limit) <= 0:
            raise ValueError("CPU source workspace needs an exact positive original ceiling")
        workspace, _ = plan(operations, outputs)
        expected = workspace.record() if workspace is not None else None
        if (
            metadata.get("source_workspace") != json.dumps(expected, sort_keys=True, separators=(",", ":"))
            or workspace is not None
            and workspace.bytes > int(limit)
        ):
            raise ValueError("CPU source workspace differs from original source values or exceeds caller ceiling")
    elif (
        metadata.get("tensor_workspace_limit") != "0" or metadata.get("workspace_status") != "conditional_not_observed"
    ):
        raise ValueError("CPU proposal omitted its unresolved native workspace/ABI obligation")
    if not outputs:
        raise ValueError("callable CPU region requires live tensor outcomes")
    for value in (*inputs, *outputs):
        TensorPort.from_type(value.type)
    for root in operations:
        for operation in root.walk():
            if isinstance(operation, builtin.UnregisteredOp) or operation.name.partition(".")[0] not in {
                "arith",
                "math",
                "tensor",
                "linalg",
                "scf",
                "cf",
            }:
                raise ValueError("CPU source applicability has unknown effects or operation semantics")
            if any(
                isinstance(v.type, (builtin.MemRefType, builtin.UnrankedMemRefType, builtin.VectorType))
                for v in (*operation.operands, *operation.results)
            ):
                raise ValueError("CPU source applicability requires immutable tensor/scalar values")
            flags = operation.properties.get("fastmath")
            if flags is not None and flags.data:
                raise ValueError("CPU system admission has no separately approved fast-math policy")


def scalar_cpu(*, resource, memory_domain, options: CompileOptions, entry="kernel"):
    """Select callable scalar CPU computation explicitly, without inventing a device.

    Existing scalar target/layout options remain caller selections. Object output
    and the internal region entry are consequences of this callable route.
    """
    from merlin.compiler.backend import scalar
    from merlin.compiler.backend import source_proposals as scalaroffers

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
    return Component(
        RegionBackend(scalar, entry, replace(options, parameters=parameters)),
        resource,
        memory_domain,
        _admit_cpu,
        proposals=proposals,
        complements=True,
        required_resource_kind="cpu",
    )


# ---- system ------------------------------------------------------------------------------------


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _write(path, record):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class SystemCompiler:
    """Source-driven composition for the same public ``compile`` call.

    Each component has one explicit provider/resource/memory binding. The
    provider preference tuple must name exactly that roster. The current policy
    is bounded first-compatible cover, not latency optimization; missing costs
    stay unknown. Source callbacks cannot approve their own accuracy thresholds
    or replace caller-selected topology/target facts.

    Optional ``validate_source(module, entry, source_options)`` is selected by
    the original caller. It must check the approved source contract and return
    exact typed coverage from ``source_semantic_attributes``. It cannot change
    the IR or approve a backend implementation by acknowledging annotations.
    """

    name = "merlin.target.system"

    def __init__(
        self,
        *,
        topology,
        components,
        preferred_providers,
        entry=None,
        image=None,
        context=None,
        budget=None,
        max_states=4096,
        validate_preparation=None,
        validate_source=None,
        source_options=None,
        validate_completion=None,
    ):
        if (
            type(topology) is not SelectedTopology
            or type(components) is not tuple
            or not components
            or any(type(component) is not Component for component in components)
        ):
            raise ValueError("system compilation requires an original selected topology and immutable typed components")
        if image is not None and type(image) is not CImageLink:
            raise ValueError("system image construction requires an explicit CImageLink")
        if entry is not None and (type(entry) is not str or not entry):
            raise ValueError("system entry must be an explicit function name or uniquely inferred")
        context = PipelineContext() if context is None else context
        budget = ProposalBudget() if budget is None else budget
        if type(context) is not PipelineContext or type(budget) is not ProposalBudget:
            raise ValueError("system compilation requires typed context identities and discovery bounds")
        if type(max_states) is not int or not 0 < max_states <= 100000:
            raise ValueError("system selection requires a bounded positive state limit")
        if validate_preparation is not None:
            _validator_identity(validate_preparation)
        self._preparation_validator = validate_preparation
        self._preparation_validator_owner = (
            None if validate_preparation is None else _validator_identity(validate_preparation)
        )
        if validate_source is None and source_options is not None:
            raise ValueError("source semantic options require original-caller validation")
        self._source_validator, self._source_options = validate_source, source_options
        self._source_validator_owner = None if validate_source is None else _validator_identity(validate_source)
        self._source_options_owner = source_options
        self._completion_validator = validate_completion
        self._completion_binding = None if validate_completion is None else _validator_identity(validate_completion)
        self.topology, self.components = topology, components
        self.entry, self.image, self.context, self.budget = entry, image, context, budget
        self.preferred_providers, self.max_states = preferred_providers, max_states
        self._services = tuple(
            resolve(component.binding.backend, group="merlin.backends", methods=("validate", "compile"))
            for component in components
        )
        names = tuple(name for name, _ in self._services)
        if (
            len(set(names)) != len(names)
            or type(preferred_providers) is not tuple
            or len(set(preferred_providers)) != len(preferred_providers)
            or set(preferred_providers) != set(names)
        ):
            raise ValueError("system provider preference must exactly name unique component bindings")
        self._owners = (topology, components, image, context, budget)
        self._callbacks = tuple(
            (
                _validator_identity(c.validate_implementation),
                _validator_identity(c.proposals) if c.proposals is not None else None,
            )
            for c in components
        )
        self._service_owners = self._services
        self._settings = self._current()
        self._revalidate()

    def _current(self):
        return _json(
            {
                "entry": self.entry,
                "policy": self.preferred_providers,
                "max_states": self.max_states,
                "context": asdict(self.context),
                "budget": asdict(self.budget),
                "source_options": self._source_options,
                "components": [
                    {
                        "entry": c.binding.entry,
                        "options": asdict(c.binding.options),
                        "binding_owner": id(c.binding),
                        "backend_owner": id(c.binding.backend),
                        "resource": c.resource,
                        "memory_domain": c.memory_domain,
                        "complements": c.complements,
                        "kind": c.required_resource_kind,
                        "target_owner": id(c.target),
                        "cpu_resource": c.cpu_resource,
                    }
                    for c in self.components
                ],
            }
        )

    def _revalidate(self):
        if any(
            left is not right
            for left, right in zip(
                (self.topology, self.components, self.image, self.context, self.budget), self._owners, strict=True
            )
        ):
            raise ValueError("system compiler original configuration owner changed")
        _check_completion_validator(self._completion_validator, self._completion_binding)
        if self._services is not self._service_owners:
            raise ValueError("system provider resolution changed after selection")
        if self._current() != self._settings:
            raise ValueError("system compiler options, policy or component selection changed")
        if self._source_options is not self._source_options_owner:
            raise ValueError("original source semantic options owner changed")
        if self._source_validator_owner is None:
            if self._source_validator is not None:
                raise ValueError("original source semantic admission changed")
        elif any(
            a is not b
            for a, b in zip(_validator_identity(self._source_validator), self._source_validator_owner, strict=True)
        ):
            raise ValueError("original source semantic admission changed")
        if self._preparation_validator_owner is None:
            if self._preparation_validator is not None:
                raise ValueError("original preparation admission changed")
        elif any(
            a is not b
            for a, b in zip(
                _validator_identity(self._preparation_validator), self._preparation_validator_owner, strict=True
            )
        ):
            raise ValueError("original preparation admission changed")
        self.topology.revalidate()
        if self.image is not None:
            self.image.revalidate()
        for component, (name, provider), (validator, proposals) in zip(
            self.components, self._services, self._callbacks, strict=True
        ):
            if getattr(provider, "name", name) != name:
                raise ValueError("selected system provider identity changed")
            self.topology.resource(component.resource)
            self.topology.memory(component.memory_domain)
            if any(
                a is not b
                for a, b in zip(_validator_identity(component.validate_implementation), validator, strict=True)
            ):
                raise ValueError("original system implementation admission changed")
            if proposals is None and component.proposals is not None:
                raise ValueError("system proposal mechanism changed after component selection")
            if proposals is not None and any(
                a is not b for a, b in zip(_validator_identity(component.proposals), proposals, strict=True)
            ):
                raise ValueError("original system source discovery callback changed")

    def _entry(self, module):
        from xdsl.dialects import func

        definitions = [op for op in module.body.block.ops if type(op) is func.FuncOp and op.body.blocks]
        matches = [fn for fn in definitions if self.entry is None or fn.sym_name.data == self.entry]
        if len(matches) != 1:
            raise ValueError("system source must bind one complete entry after frontend preparation")
        return matches[0].sym_name.data

    def validate_preparation(self, request):
        """Require original-caller semantic admission, not policy/hash self-approval."""
        self._revalidate()
        if request.preparation is None:
            return
        if self._preparation_validator is None:
            raise ValueError("system compiler has no original-caller staged preparation admission")
        original = request_identity(request)
        if self._preparation_validator(request.source, request.preparation) is not None:
            raise ValueError("preparation admission must return None or raise refusal")
        self._revalidate()
        if request_identity(request) != original:
            raise ValueError("preparation admission changed the original request or selected stage")

    def validate(self, request):
        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

        self._revalidate()
        if request.source.kind != "mlir":
            raise ValueError("system compiler requires realized imported tensor MLIR")
        module = parse_mlir_text(request.source.text)
        module.verify()
        entry = self._entry(module)
        control_entry(request.options, entry, self.image.deployment if self.image is not None else None)

    def compile(self, request):
        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
        from merlin.compiler.lower.device.plan import bind_source

        self.validate(request)
        if request.preparation is not None:
            self.validate_preparation(request)
        original = request_identity(request)
        module = parse_mlir_text(request.source.text)
        source = bind_source(
            module,
            entry=self._entry(module),
            context=self.context,
            graph_mode="ssa",
            max_operations=self.budget.max_source_operations,
            validate_source=self._source_validator,
            source_options=self._source_options,
        )
        candidates, discoveries = [], []
        # Ordinary OOT offers establish cut points first. Explicit complement
        # providers may then offer source regions around those same alternatives;
        # outlining remains a materialization step after global selection.
        order = sorted(range(len(self.components)), key=lambda n: self.components[n].complements)
        for index in order:
            component = self.components[index]
            name, provider = self._services[index]
            cuts = tuple(
                sorted({n for item in candidates for n in (item.alternative.nodes[0], item.alternative.nodes[-1] + 1)})
            )
            found, report = component.discover(
                source, self.topology, provider, name, self.budget, cuts, preparation=request.preparation
            )
            source.revalidate()
            self._revalidate()
            if request_identity(request) != original:
                raise ValueError("system source discovery changed the original compile request")
            candidates.extend(found)
            if len(candidates) > self.budget.max_candidates:
                raise ValueError("system discovery exceeded the selected total candidate budget")
            discoveries.append(
                {
                    "provider": name,
                    "resource": component.resource,
                    "memory_domain": component.memory_domain,
                    "report": report,
                }
            )
        if not candidates:
            raise ValueError("system discovery produced no independently admitted source implementations")
        available = {item.proposal.provider for item in candidates}
        preference = tuple(name for name in self.preferred_providers if name in available)
        selection = select_execution(
            source, tuple(candidates), preferred_providers=preference, max_states=self.max_states
        )
        selected_names = {item.proposal.provider for item in selection.selected}
        bindings = {
            name: component.binding
            for component, (name, _) in zip(self.components, self._services, strict=True)
            if name in selected_names
        }
        planned = PlannedCompiler(selection, bindings, image=self.image, validate_completion=self._completion_validator)
        record = {
            "schema": "merlin.system_compilation.v1",
            "request_source_sha256": request.source.sha256,
            "realized_ir_sha256": source.stamp.source_sha256,
            "entry": source.entry,
            "topology_sha256": self.topology.sha256,
            "context": asdict(self.context),
            "policy": {
                "kind": "bounded-first-compatible",
                "within_provider_order": "larger admitted regions, then stable identity; no cost comparison",
                "provider_order": list(self.preferred_providers),
                "cost_comparison": "not_performed; missing costs remain unknown",
            },
            "discovery": discoveries,
            "selection": selection.record(),
            "selected_obligations": [
                {"alternative": item.alternative.id, "requirements": dict(item.alternative.metadata)}
                for item in selection.selected
            ],
            **({"source_semantics": source.semantic_admission()} if self._source_validator is not None else {}),
            **({"preparation": request.preparation.record()} if request.preparation is not None else {}),
            "numerical_verification": "not_established",
            "hardware_qualification": "not_established",
        }
        try:
            compiled = planned.compile(request)
            self._revalidate()
            source.revalidate()
            if request_identity(request) != original:
                raise ValueError("system lowering changed the original compilation request")
        except Exception as error:
            if request.output.is_dir():
                _write(
                    request.output / "system-failure.json",
                    {
                        **record,
                        "status": "failed",
                        "reason": str(error),
                        "error_type": type(error).__name__,
                        "fallback": "not_attempted; selected lowering obligations were not discharged",
                    },
                )
            raise
        path = request.output / "system.json"
        _write(
            path,
            {
                **record,
                "status": "compiled",
                "stage": compiled.metadata["stage"],
                "obligation_evidence": "selected region ABI/native receipts and control admission; see plan.json",
            },
        )
        result = replace(
            compiled,
            backend=self.name,
            artifacts=(*compiled.artifacts, Artifact.from_file(path, kind="json", role="system_plan")),
            metadata={
                **compiled.metadata,
                "system": {"record": path.name, "realized_ir_sha256": source.stamp.source_sha256},
            },
        )
        result.verify()
        return result


__all__ = [
    "Component",
    "SystemCompiler",
    "scalar_cpu",
]
