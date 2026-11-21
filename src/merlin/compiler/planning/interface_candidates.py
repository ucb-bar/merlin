"""Discover source regions in Merlin, then query OOT Interface implementations.

Shared rules expose dense contractions and bounded unique-consumer frontiers.
The source-to-Interface pass proves each actual fusion before a backend sees it.
Physical storage, transfers and performance are not inferred. Target legality
remains the original caller's independent admission.
"""

from __future__ import annotations

import json
from math import prod

from merlin.compiler.backend.boundary import INTERFACE_IR, interface_source
from merlin.compiler.backend.interface import CompilerInterface
from merlin.compiler.backend.source_proposals import (
    LoweringRecipe,
    SourceProposal,
    _validator_identity,
    admit_proposal,
    proposal_boundary,
    source_numerical_identity,
)
from merlin.compiler.compilation import Source
from merlin.compiler.ir.region import preview_region
from merlin.compiler.partition.global_plan import CycleInterval, RegionAlternative
from merlin.compiler.transforms.certificates import ir_text


def _record(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_members():
    import sys

    from merlin.compiler.backend import boundary
    from merlin.compiler.ir import interface_contract, readout
    from merlin.compiler.lower import interface
    from merlin.compiler.lower import interface_readout
    from merlin.compiler.lower.quant import requantization
    from merlin.compiler.lower.quant import readout_match

    owner = sys.modules[__name__]
    return (
        *(
            (owner, name)
            for name in ("_source_members", "_record", "_regions", "_prepare_region", "discover_interface_candidates")
        ),
        (boundary, "interface_source"),
        *(
            (interface, name)
            for name in (
                "to_interface",
                "_metadata",
                "_initializer",
                "_contraction_maps",
                "_plain_tensor",
                "match_readout",
                "match_integer_gemm",
                "_canonical_body",
            )
        ),
        *(
            (interface_contract, name)
            for name in (
                "interface_text",
                "validate_interface",
                "parse_interface",
                "_validate_numerical",
                "_validate_attribute",
            )
        ),
        *(
            (interface_readout, name)
            for name in (
                "_tensor_spec",
                "match_integer_saturating_cast",
                "_parameters",
                "match_f32_readout",
                "match_readout",
            )
        ),
        *(
            (readout, name)
            for name in (
                "scale_bits",
                "readout_values",
                "readout_contract",
                "constant_parameter",
                "readout_parameters",
                "parameter_readout_contract",
            )
        ),
        (interface_readout, "match"),
        *(
            (readout_match, name)
            for name in ("_tensor_spec", "match_f32_readout", "constant_parameter", "match_integer_gemm")
        ),
        (requantization, "match_integer_gemm"),
        (requantization, "_canonical_body"),
        (requantization, "prove_scale"),
        (requantization, "synthesize_bias"),
    )


def _regions(source, max_operations):
    """Discover raw and linear fused groups without approving a rewrite.

    Only local initializer dependencies may be absorbed automatically. Every
    candidate still passes full source coverage, numerical and live-port checks
    in to_interface. A fork keeps its observers and stops this linear frontier.
    """
    indices = {operation: index for index, operation in enumerate(source.operations)}
    helpers = {"arith.constant", "tensor.empty", "tensor.splat", "linalg.fill"}

    def include_helpers(selected, values):
        pending = [value.owner for value in values]
        while pending:
            operation = pending.pop()
            if operation in selected or operation not in indices or operation.name not in helpers:
                continue
            selected.add(operation)
            pending.extend(value.owner for value in operation.operands)

    for root in source.operations:
        if root.name not in {"linalg.generic", "linalg.matmul"}:
            continue
        selected = {root}
        include_helpers(selected, root.operands[2:] if len(root.operands) == 3 else ())
        tail = root
        while True:
            nodes = tuple(sorted(indices[operation] for operation in selected))
            if len(selected) > max_operations or nodes == tuple(range(nodes[0], nodes[-1] + 1)):
                yield nodes
            if len(selected) > max_operations:
                break
            users = {use.operation for value in tail.results for use in value.uses}
            if len(users) != 1:
                break
            successor = next(iter(users))
            if successor not in indices or successor in selected or successor.name != "linalg.generic":
                break
            selected.add(successor)
            include_helpers(selected, successor.operands)
            tail = successor


def _prepare_region(source, nodes, *, entry):
    operations = tuple(source.operations[index] for index in nodes)
    preview = preview_region(source.module, source.entry, operations, symbol=entry)
    # Visibility is an outlining policy, not part of tensor arithmetic. The final
    # partition performs this same public-entry materialization in request_source.
    preview.module.body.block.first_op.properties.pop("sym_visibility", None)
    return interface_source(Source(ir_text(preview.module)))


def discover_interface_candidates(
    source,
    *,
    provider,
    name,
    topology,
    resource,
    memory_domain,
    entry,
    validate_implementation,
    budget,
    target=None,
    preparation=None,
):
    """Return admitted original-source alternatives and an inspectable work report.

    Discovery never sends linalg to a backend. Implementations are queried only
    after the common pass has materialized a complete Interface region. No
    fallback, precision change or retry after effects is introduced here.
    """
    hook = getattr(provider, "propose_interface", None)
    identity = getattr(provider, "discovery_identity", None)
    if not callable(hook) or not callable(identity):
        raise ValueError("Interface components need implementation proposals and source identity")
    interface = CompilerInterface(provider, name)
    if interface.description is None or interface.description.source_domain.ir != INTERFACE_IR:
        raise ValueError("Interface components require a canonical compiler description")
    hook_identity, identity_binding = _validator_identity(hook), _validator_identity(identity)
    validator_binding = _validator_identity(validate_implementation)
    implementation = identity()
    description = _record(interface.record())
    target_record = _record(target.load_contract()) if target is not None else None

    def custody():
        source.revalidate()
        topology.revalidate()
        interface.revalidate()
        for current, original in (
            (getattr(provider, "propose_interface", None), hook_identity),
            (getattr(provider, "discovery_identity", None), identity_binding),
            (validate_implementation, validator_binding),
        ):
            if not callable(current) or any(
                a is not b for a, b in zip(_validator_identity(current), original, strict=True)
            ):
                raise ValueError("Interface discovery or original admission callback changed")
        if identity() != implementation or _record(interface.record()) != description:
            raise ValueError("Interface provider source or declarations changed")
        if target is not None and _record(target.load_contract()) != target_record:
            raise ValueError("selected target contract changed during Interface discovery")
        if preparation is not None:
            preparation.revalidate_ir(source)

    custody()
    members = tuple(source.module.walk())
    if len(members) > budget.max_source_operations or any(
        prod(shape) > budget.max_tensor_elements
        for operation in members
        for value in (*operation.operands, *operation.results)
        if (shape := getattr(value.type, "get_shape", lambda: ())())
    ):
        raise ValueError("Interface discovery exceeds the source/tensor budget")
    accepted, rows, visits, sessions, exhausted = [], [], 0, 0, False
    for nodes in _regions(source, budget.max_region_operations):
        if (
            visits >= budget.max_match_visits
            or sessions + 2 > budget.max_proof_sessions
            or len(accepted) >= budget.max_candidates
        ):
            exhausted = True
            break
        visits += 1
        if len(nodes) > budget.max_region_operations:
            exhausted = True
            rows.append({"nodes": nodes, "status": "refused", "reason": "region operation budget"})
            continue
        sessions += 1
        try:
            prepared = _prepare_region(source, nodes, entry=entry)
        except ValueError as error:
            rows.append({"nodes": nodes, "status": "refused", "reason": str(error)})
            continue
        custody()
        try:
            recipes = hook(prepared, resource=resource, memory_domain=memory_domain)
        except ValueError as error:
            custody()
            rows.append({"nodes": nodes, "status": "refused", "reason": str(error)})
            continue
        custody()
        if type(recipes) is not tuple or any(type(recipe) is not LoweringRecipe for recipe in recipes):
            raise ValueError("Interface proposals must be an immutable tuple of typed lowering recipes")
        if len(recipes) + len(accepted) > budget.max_candidates or sessions + len(recipes) > budget.max_proof_sessions:
            raise ValueError("Interface implementation proposals exceed the selected budget")
        inputs, outputs = proposal_boundary(source, nodes)
        for index, recipe in enumerate(recipes):
            interface.validate_recipe(recipe)
            if recipe.implementation_sha256 != implementation or recipe.abi != "logical_tensor_v1":
                raise ValueError("Interface recipe changed its source identity or logical tensor ABI")
            alternative = RegionAlternative(
                id=f"{name}/interface/{nodes[0]:06d}/{index}",
                nodes=nodes,
                implementation=recipe.pipeline,
                placement=resource,
                cycles=CycleInterval.unknown("no comparable full-region Interface implementation cost"),
                inputs=inputs,
                outputs=outputs,
                metadata=(("interface_ir", INTERFACE_IR), ("pattern", "shared-interface-region")),
            )
            proposal = SourceProposal(
                alternative,
                name,
                resource,
                memory_domain,
                recipe.abi,
                source.stamp.source_sha256,
                tuple(source.operation_ordinals[node] for node in nodes),
                source_numerical_identity(source, nodes),
                lowering=recipe,
            )

            def check(bound, offered, *, expected_sha=prepared.sha256):
                custody()
                # Recompute shared conversion from the actual bound source; an
                # OOT capability declaration cannot admit its own numerical change.
                actual = _prepare_region(bound, offered.alternative.nodes, entry=entry)
                if actual.sha256 != expected_sha:
                    raise ValueError("selected source no longer produces the proposed Interface program")
                interface.validate_recipe(offered.lowering)
                if validate_implementation(bound, offered) is not None:
                    raise ValueError("implementation admission must return None or raise refusal")
                custody()

            sessions += 1
            try:
                admitted = admit_proposal(source, proposal, topology, validate_implementation=check)
            except ValueError as error:
                rows.append({"nodes": nodes, "status": "refused", "reason": str(error)})
            else:
                accepted.append(admitted)
                rows.append(
                    {
                        "nodes": nodes,
                        "status": "admitted",
                        "interface_sha256": prepared.sha256,
                        "pipeline": recipe.pipeline,
                    }
                )
            custody()
    return tuple(accepted), {
        "schema": "merlin.interface_discovery.v1",
        "provider": name,
        "source_sha256": source.stamp.source_sha256,
        "compiler_interface": interface.record(),
        "offers": rows,
        "work": {"matcher_visits": visits, "proof_sessions": sessions},
        "status": "budget_exhausted" if exhausted else "complete",
        "scope": "bounded dense/unique-consumer frontiers; shared source proof and backend admission both required",
        "qualification": "not_established",
        "costs": "unknown",
    }
