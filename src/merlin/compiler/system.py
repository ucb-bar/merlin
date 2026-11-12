"""Plan and compile the actual post-frontend program through explicit components.

This ordinary backend binds source only after public capture/import/immutable
binding. Discovery, independent admission and bounded selection precede existing
partition/control/build stages. No precision reconsideration or runtime fallback
is implicit, and a failed conditional lowering obligation fails this compile.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from merlin.compiler.artifact import Artifact
from merlin.compiler.backend.discovery import ProposalBudget
from merlin.compiler.backend.service import resolve
from merlin.compiler.backend.sourceproposal import _validator_identity
from merlin.compiler.build.control import control_entry
from merlin.compiler.build.image import CImageLink
from merlin.compiler.components import Component, scalar_cpu
from merlin.compiler.invocation import request_identity
from merlin.compiler.pipeline import PlannedCompiler
from merlin.compiler.planning.execution import select_execution
from merlin.compiler.transforms.pipeline_state import PipelineContext
from merlin.target.system.topology import SelectedTopology

__all__ = ["Component", "SystemCompiler", "scalar_cpu"]


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
        from merlin.compiler.lower.model.offload_plan import bind_source

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
        planned = PlannedCompiler(selection, bindings, image=self.image)
        record = {
            "schema": "merlin.system_compilation.v1",
            "request_source_sha256": request.source.sha256,
            "realized_ir_sha256": source.stamp.source_sha256,
            "entry": source.entry,
            "topology_sha256": self.topology.sha256,
            "context": asdict(self.context),
            "policy": {
                "kind": "bounded-first-compatible",
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
