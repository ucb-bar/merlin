"""Admit OOT-discovered dense/readout alternatives before global selection.

Backends match the implementations they actually lower. Shared code checks the
offered source, the selected target contract and exact readout theorem. It does
not rewrite to a greedy fused partition or choose between overlapping offers.
The original caller retains independent target-compute implementation admission.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from merlin.compiler.backend.sourceproposal import (
    SourceProposal,
    _validator_identity,
    admit_original_cpu,
    admit_proposal,
)


def _record(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _target_identity(target):
    return (
        target.name,
        target.kind,
        target.base.resolve(),
        target.capability_contract_path.resolve(),
        target.contract_path.resolve(),
        target.facts_path.resolve(),
        target.backend,
    )


def _serialized_readout(root, operations, output_path):
    """Check the existing serialized requantization family against selected semantics."""
    from merlin.llvmlower.requantization_match import match
    from merlin.targetgen.output_path import validate
    from merlin.transforms.dense_ir import _trace_attributes

    readout = validate(output_path)
    if readout.get("accumulator", {}).get("dtype") != "i32" or not readout.get("exits", {}).get("requant"):
        raise ValueError("selected output path has no i32 to requantized exit")
    kinds = {stage["kind"]: stage for stage in readout.get("readout", ())}
    sequence = tuple(stage["kind"] for stage in readout.get("readout", ()) if stage["kind"] != "pool")
    if (
        sequence not in {("convert", "scale", "requant"), ("activation", "convert", "scale", "requant")}
        or len(kinds) != len(readout.get("readout", ()))
        or ("activation" in kinds and kinds["activation"].get("domain") != "accumulator")
        or kinds.get("convert") != {"kind": "convert", "to": "f32", "rounding": "half_even"}
        or kinds.get("scale", {}).get("dtype") != "f32"
        or kinds.get("scale", {}).get("rounding") != "half_even"
        or kinds.get("scale", {}).get("granularity") not in {"per_command", "per_tensor", "per_channel"}
        or kinds.get("requant") != {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]}
    ):
        raise ValueError("selected output path does not implement the proved f32 nearest-even saturating readout")
    candidates = [op for op in operations if op is not root and root.results[0] in op.operands]
    if len(candidates) != 1:
        raise ValueError("serialized requantization has no unique source consumer")
    proved = match(candidates[0])
    epilogue = proved.epilogue
    if not _trace_attributes(epilogue.attributes) or set(epilogue.properties) != {
        "indexing_maps",
        "iterator_types",
        "operandSegmentSizes",
    }:
        raise ValueError("serialized readout has unsupported shell attributes or properties")
    properties = {
        "arith.constant": {"value"},
        "arith.sitofp": set(),
        "arith.mulf": {"fastmath"},
        "arith.addf": {"fastmath"},
        "math.roundeven": {"fastmath"},
        "arith.maximumf": {"fastmath"},
        "arith.minimumf": {"fastmath"},
        "arith.fptosi": set(),
        "linalg.yield": set(),
    }
    for scalar in epilogue.regions[0].block.ops:
        if (
            scalar.name not in properties
            or not _trace_attributes(scalar.attributes)
            or not set(scalar.properties) <= properties[scalar.name]
            or scalar.regions
            or scalar.successors
        ):
            raise ValueError("serialized readout has unsupported scalar effects or properties")
        fastmath = scalar.properties.get("fastmath")
        if fastmath is not None and fastmath.data:
            raise ValueError("serialized readout has fast-math assumptions")
    if proved.contraction is not root:
        raise ValueError("serialized requantization proof belongs to another contraction")
    if proved.relu and kinds.get("activation", {}).get("function") != "relu":
        raise ValueError("selected output path cannot implement the source activation")
    if proved.integer_bias is not None:
        preloads = [stage for stage in readout.get("accumulator_stages", ()) if stage["kind"] == "preload"]
        if len(preloads) != 1 or preloads[0].get("dtype") != "i32":
            raise ValueError("selected output path cannot preload the proved integer bias")
        bounds = preloads[0].get("range", ())
        if len(bounds) != 2 or any(not bounds[0] <= value <= bounds[1] for value in proved.integer_bias):
            raise ValueError("proved integer bias exceeds the selected preload range")
    return proved.epilogue, {"cut": "requant", "proof": proved.proof, "matcher": "requantization_match"}


def discover_fusion_candidates(
    source,
    *,
    provider,
    target,
    topology,
    resource: str,
    memory_domain: str,
    cpu_resource: str,
    validate_implementation,
    budget=None,
    cursor=None,
    numerical_policy_sha256=None,
    preparation_sha256=None,
):
    """Return admitted overlapping offers and original CPU alternatives plus reasons.

    ``target`` is the existing selected TargetInfo, independently of the compiler
    provider. Its capability contract is reopened at admission and every later
    proposal revalidation. An authored contract alone is not hardware evidence;
    the caller's validator must check target computation/overflow obligations
    independently. Provider-local output-path JSON is never read as authority.
    Initial shared proof scope: zero-initialized signed i8/i32 dense products,
    optionally followed by the complete exact requantization epilogue. Costs stay
    unknown and the existing selector decides the complete program cover.
    """
    from merlin.compiler.backend.discovery import ProposalBudget, ProposalContext, ProposalPage
    from merlin.targetgen.target_registry import TargetInfo
    from merlin.verification.lean_contract import document_digest

    if type(target) is not TargetInfo:
        raise ValueError("fusion discovery needs the existing explicitly selected TargetInfo")
    source.revalidate()
    path = target.capability_contract_path.resolve(strict=True)
    selected_bytes = path.read_bytes()
    target_identity = _target_identity(target)
    contract = target.load_contract()
    contract_record = _record(contract)
    output_path = (contract.get("output_epilogue") or {}).get("output_path")
    propose = getattr(provider, "propose", None)
    provider_name = getattr(provider, "name", None)
    if not callable(propose) or not isinstance(provider_name, str) or not provider_name:
        raise ValueError("selected compiler has no implementation proposal hook")
    validator_binding = _validator_identity(validate_implementation)
    topology.revalidate()
    page_hook = getattr(provider, "propose_page", None)
    discovery = None
    context = None
    provider_binding = None
    if callable(page_hook):
        describe, identity = getattr(provider, "describe", None), getattr(provider, "discovery_identity", None)
        if not callable(describe) or not callable(identity):
            raise ValueError("paged source proposals require implemented patterns and provider source identity")
        context = ProposalContext(
            source.stamp.source_sha256,
            document_digest(
                {
                    "selected_target": tuple(str(v) for v in target_identity),
                    "bytes_sha256": hashlib.sha256(selected_bytes).hexdigest(),
                    "contract": contract,
                }
            ),
            (
                document_digest(
                    {"context": asdict(source.stamp.context), "selected_numerics": contract.get("numerical_semantics")}
                )
                if numerical_policy_sha256 is None
                else numerical_policy_sha256
            ),
            (
                document_digest({"context": asdict(source.stamp.context), "preparation": "not_supplied"})
                if preparation_sha256 is None
                else preparation_sha256
            ),
            topology.sha256,
            provider_name,
            identity(),
            describe(),
            resource,
            memory_domain,
            ProposalBudget() if budget is None else budget,
        )
        provider_binding = (page_hook, describe, identity)
        page = page_hook(source, context, cursor=cursor)
        if type(page) is not ProposalPage:
            raise ValueError("compiler paged proposal hook must return a typed ProposalPage")
        offered, discovery = page.proposals, page.discovery
        if len(offered) > context.budget.max_candidates or discovery.get("discovery_context_sha256") != document_digest(
            context.record()
        ):
            raise ValueError("compiler proposal page differs from its bound context or output budget")
        work = discovery.get("work", {})
        if (
            type(work.get("matcher_visits")) is not int
            or not 0 <= work["matcher_visits"] <= context.budget.max_match_visits
            or type(work.get("provider_proof_sessions")) is not int
            or not 0 <= work["provider_proof_sessions"] + len(offered) <= context.budget.max_proof_sessions
        ):
            raise ValueError("compiler proposal page exceeds its declared matcher/proof session budget")
        if any(
            p.lowering is None
            or p.lowering.implementation_sha256 != context.implementation_sha256
            or p.lowering.pipeline not in {pattern.pipeline for pattern in context.patterns}
            for p in offered
        ):
            raise ValueError("compiler proposal page changed its described implementation or recipe")
    else:
        if budget is not None or cursor is not None:
            raise ValueError("selected compiler has no bounded source proposal hook")
        offered = propose(source, resource=resource, memory_domain=memory_domain)
    source.revalidate()
    if (
        _target_identity(target) != target_identity
        or path.read_bytes() != selected_bytes
        or _record(target.load_contract()) != contract_record
    ):
        raise ValueError("compiler proposal hook changed the selected target capability contract")
    if type(offered) is not tuple:
        raise ValueError("compiler proposal hook must return an immutable tuple")
    accepted, reasons = [], []

    def check_provider():
        if context is not None and (
            (
                getattr(provider, "propose_page", None),
                getattr(provider, "describe", None),
                getattr(provider, "discovery_identity", None),
            )
            != provider_binding
            or provider.name != context.provider
            or provider.describe() != context.patterns
            or provider.discovery_identity() != context.implementation_sha256
        ):
            raise ValueError("selected provider implementation changed; rediscover proposals")
        topology.revalidate()

    check_provider()

    def check(current_source, proposal):
        from merlin.compiler.ir.contraction_shapes import contraction_extents, zero_initialised
        from merlin.transforms.dense_ir import _trace_attributes, static_zero_integer_matmul_geometry
        from merlin.xdsl_dialects.lowering import compute_groups as CG
        from merlin.xdsl_dialects.lowering import output_fusion as OF

        check_provider()
        if (
            _target_identity(target) != target_identity
            or path.read_bytes() != selected_bytes
            or _record(target.load_contract()) != contract_record
        ):
            raise ValueError("selected target capability contract changed; rediscover proposals")
        if proposal.provider != provider_name:
            raise ValueError("proposal is not owned by the selected compiler")
        operations = tuple(current_source.operations[n] for n in proposal.alternative.nodes)
        roots = [op for op in operations if (stage := CG.classify(op)) is not None and stage.kind == CG.CONTRACTION]
        if len(roots) != 1:
            raise ValueError("initial fusion proof requires exactly one dense contraction")
        root = roots[0]
        if root.name != "linalg.generic":
            raise ValueError("initial fusion proof requires an explicit proved Generic signed dense body")
        static_zero_integer_matmul_geometry(root)
        if [str(v.type.element_type) for v in root.operands] != ["i8", "i8", "i32"] or not zero_initialised(root):
            raise ValueError("initial fusion proof requires signed i8 inputs and a zero i32 initializer")
        metadata = dict(proposal.alternative.metadata)
        pattern = metadata.get("pattern")
        required = {root}
        if pattern == "dense-requant":
            if output_path is None:
                raise ValueError("selected target contract has no output path; fused readout is unknown")
            values = OF.Values()
            chain = OF.source_chain(root, values)
            decision = OF.decide(chain, output_path, values, contraction_extents(current_source.module))
            if decision.cut == "requant":
                required.update(decision.members)
                expected = _record(decision.record())
            else:
                epilogue, proof = _serialized_readout(root, operations, output_path)
                required.add(epilogue)
                expected = _record(proof)
            if metadata.get("numerical_proof") != expected:
                raise ValueError("provider readout arithmetic differs from the independently checked selected contract")
        elif pattern != "dense":
            raise ValueError("offered pattern has no shared numerical checker")
        # Dependencies may include immutable tensor/scalar producers, never an
        # unaccounted arithmetic/effect operation hidden inside a fused region.
        helpers = {"arith.constant", "tensor.empty", "tensor.splat", "linalg.fill"}
        if not required <= set(operations) or any(op not in required and op.name not in helpers for op in operations):
            raise ValueError("offered region drops proved stages or absorbs unrelated source operations")
        for operation in operations:
            if not _trace_attributes(operation.attributes):
                raise ValueError("offered region has unsupported source attributes")
            if operation not in required:
                allowed = {
                    "arith.constant": {"value"},
                    "tensor.empty": set(),
                    "tensor.splat": set(),
                    "linalg.fill": {"operandSegmentSizes"},
                }[operation.name]
                if set(operation.properties) != allowed:
                    raise ValueError("offered helper has unknown execution properties")
        # Shared source proposal admission checks every live tensor input/output;
        # the original caller checks selected target arithmetic and implementation.
        if any(
            left is not right
            for left, right in zip(_validator_identity(validate_implementation), validator_binding, strict=True)
        ):
            raise ValueError("original target implementation validator changed")
        if validate_implementation(current_source, proposal) is not None:
            raise ValueError("target implementation validator must raise on refusal and otherwise return None")
        if any(
            left is not right
            for left, right in zip(_validator_identity(validate_implementation), validator_binding, strict=True)
        ):
            raise ValueError("target implementation validator changed during admission")
        if (
            _target_identity(target) != target_identity
            or path.read_bytes() != selected_bytes
            or _record(target.load_contract()) != contract_record
        ):
            raise ValueError("selected target capability contract changed during admission")
        check_provider()

    for proposal in offered:
        if type(proposal) is not SourceProposal:
            raise ValueError("compiler returned an untyped implementation proposal")
        try:
            admitted = admit_proposal(source, proposal, topology, validate_implementation=check)
        except ValueError as exc:
            category = (
                "missing_selected_output_path"
                if output_path is None and dict(proposal.alternative.metadata).get("pattern") == "dense-requant"
                else "source_or_target_admission"
            )
            reasons.append(
                {"alternative": proposal.alternative.id, "status": "refused", "category": category, "reason": str(exc)}
            )
        else:
            accepted.append(admitted)
            reasons.append(
                {
                    "alternative": proposal.alternative.id,
                    "status": "admitted",
                    "pattern": dict(proposal.alternative.metadata).get("pattern"),
                }
            )
        # Custody corruption is a hard failure, never an optimization refusal.
        source.revalidate()
        check_provider()
        if (
            _target_identity(target) != target_identity
            or path.read_bytes() != selected_bytes
            or _record(target.load_contract()) != contract_record
        ):
            raise ValueError("selected target capability contract changed during admission")
    cpu = tuple(
        admit_original_cpu(source, index, topology, resource=cpu_resource, memory_domain=memory_domain)
        for index in range(len(source.operations))
    )
    report = {
        "schema": "merlin.fusion_discovery.v1",
        "source_sha256": source.stamp.source_sha256,
        "target": target.name,
        "target_contract_sha256": hashlib.sha256(selected_bytes).hexdigest(),
        "provider": provider_name,
        "offers": reasons,
        "cpu_alternatives": len(cpu),
        "selection": "not_performed",
        "hardware_qualification": "not_established",
        "costs": "unknown; no complete resource-envelope model",
        "planning_scope": "one shared coherent memory domain and serial completion",
        "discovery": discovery,
        "discovery_context": context.record() if context is not None else None,
        "next_cursor": discovery.get("next_cursor") if discovery is not None else None,
        "status": ("budget_exhausted" if discovery.get("scope_incomplete") else "complete")
        if discovery is not None
        else "legacy_unpaged",
        "numerical_policy_sha256": context.numerical_policy_sha256 if context is not None else None,
        "preparation_sha256": context.preparation_sha256 if context is not None else None,
    }
    if discovery is not None:
        discovery["work"]["shared_admission_sessions"] = len(offered)
    return (*accepted, *cpu), report
