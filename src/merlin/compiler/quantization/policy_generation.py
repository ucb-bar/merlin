"""Graph-derived numerical requirements matched to provider family declarations.

A match selects an implementation claim. It does not discharge fusion premises,
prove runtime domains, authorize new arithmetic or qualify model accuracy. Export
schema decoding and receipt authentication remain at the frontend boundary.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from hashlib import sha256

from merlin.common.digest import is_sha256
from merlin.compiler.backend.interface import CompilerDescription
from merlin.compiler.semantic.model import KernelRequest


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return sha256(_json(value).encode()).hexdigest()


def _names(values):
    return (type(values) is tuple and bool(values)
            and all(type(value) is str and value.strip() for value in values)
            and len(set(values)) == len(values))


class PolicyGenerationError(ValueError):
    """A source requirement has no unambiguous declared numerical implementation."""


@dataclass(frozen=True)
class NumericalFamily:
    """Numerical extension of one existing ImplementationFamily/SourcePattern.

    ``constraints`` compares canonical JSON values at paths in the requirement
    record, e.g. ("result", "dtype") or ("attributes", "rounding"). Paths cannot
    name node IDs. An empty set leaves shape legality to the existing backend
    admission; matching is never a substitute for that admission.
    ``excluded_constraints`` is an optional conjunction: when all its predicates
    match, the family is excluded. It distinguishes structural alternatives
    without provider-order priority or model/node-name routing.

    ``policy_fields`` is a compatibility projection, declared by the provider,
    of only this family's authorization fields. JSON strings keep declarations
    immutable. Shared fields must agree across selected families.
    ``numerical_contract`` records the provider's structured numerical claims
    and obligations; publishing them does not prove or authorize them.
    """

    family: str
    pattern: str
    source_policies: tuple[str, ...]
    implementation_policy: str
    constraints: tuple[tuple[tuple[str, ...], str], ...] = ()
    policy_fields: tuple[tuple[str, str], ...] = ()
    precision_change: str | None = None
    numerical_contract: tuple[tuple[str, str], ...] = ()
    excluded_constraints: tuple[tuple[tuple[str, ...], str], ...] = ()

    def __post_init__(self):
        if not _names((self.family,)) or not _names((self.pattern,)):
            raise ValueError("numerical declarations require named family and pattern")
        if not _names(self.source_policies) or not _names((self.implementation_policy,)):
            raise ValueError("numerical declarations require explicit source and implementation policies")
        if self.precision_change is not None and not _names((self.precision_change,)):
            raise ValueError("precision changes require explicit authorization identities")
        if type(self.constraints) is not tuple or type(self.policy_fields) is not tuple:
            raise ValueError("numerical declaration fields must be immutable")
        self._validate_constraints(self.constraints)
        self._validate_constraints(self.excluded_constraints)
        self._validate_fields(self.policy_fields)
        self._validate_fields(self.numerical_contract)

    @staticmethod
    def _validate_constraints(constraints):
        if type(constraints) is not tuple:
            raise ValueError("numerical constraints must be immutable")
        paths = []
        for entry in constraints:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("constraints require immutable path/value pairs")
            path, value = entry
            if (type(path) is not tuple or not path
                    or any(type(key) is not str or not key for key in path)
                    or path[0] not in {"operation", "operand_count", "result", "operands", "attributes", "index_maps"}
                    or any(key.startswith("prov.") for key in path)
                    or type(value) is not str or _json(json.loads(value)) != value):
                raise ValueError("constraints require structural paths and canonical JSON values")
            paths.append(path)
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate numerical constraint")
    @staticmethod
    def _validate_fields(fields):
        if type(fields) is not tuple:
            raise ValueError("numerical contract fields must be immutable")
        keys = []
        for entry in fields:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("policy fields require immutable name/value pairs")
            key, value = entry
            if not _names((key,)) or type(value) is not str or _json(json.loads(value)) != value:
                raise ValueError("policy fields require names and canonical JSON values")
            keys.append(key)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate compatibility policy field")


def graph_requirements(source: KernelRequest):
    """Inspect every computation in the actual graph, preserving operand order.

    Do not prune by model/site names or output reachability. Even an unused pure
    node in the compiled graph is a required operation until a source transform
    removes it. Storage/constant bytes stay bound by KernelRequest.digest().
    """
    if type(source) is not KernelRequest:
        raise ValueError("policy generation requires a typed source graph")
    source.__post_init__()
    nodes = {node.id: node for node in source.nodes}
    return tuple({
        "node": node.id,
        "inputs": list(node.inputs),
        "operation": node.op,
        "operand_count": len(node.inputs),
        "result": node.type.record(),
        "operands": [{**nodes[key].type.record(), "effect": nodes[key].effect,
                      "producer": {
                          "operation": nodes[key].op,
                          "attributes": {k: v for k, v in nodes[key].attrs if not k.startswith("prov.")},
                          "index_maps": [value.record() for value in nodes[key].index_maps],
                      }} for key in node.inputs],
        "attributes": {key: value for key, value in node.attrs if not key.startswith("prov.")},
        "index_maps": [value.record() for value in node.index_maps],
    } for node in source.nodes if node.effect == "pure")


def _matches(requirement, declaration):
    policies = {requirement["result"]["numerical_policy"],
                *(operand["numerical_policy"] for operand in requirement["operands"])}
    if not policies <= set(declaration.source_policies):
        return False
    return (_satisfies_constraints(requirement, declaration.constraints)
            and not (declaration.excluded_constraints
                     and _satisfies_constraints(requirement, declaration.excluded_constraints)))


def _satisfies_constraints(requirement, constraints):
    for path, expected in constraints:
        value = requirement
        try:
            for key in path:
                if isinstance(value, list):
                    if not key.isascii() or not key.isdecimal() or str(int(key)) != key:
                        return False
                    value = value[int(key)]
                else:
                    value = value[key]
        except (KeyError, IndexError, ValueError, TypeError):
            return False
        if _json(value) != expected:
            return False
    return True


def generate_policy(source: KernelRequest, description: CompilerDescription,
                    numerical_families: tuple[NumericalFamily, ...], *,
                    original_source_sha256: str, preparation_sha256: str,
                    authorized_precision_changes: tuple[str, ...] = (),
                    selected_family_ids: tuple[str, ...] | None = None):
    """Generate requirements and deterministic family claims for a checked graph.

    Call after authenticated capture staging/translation and before existing
    proposal discovery. Family availability supplies no precision authorization.
    Competing matches refuse rather than changing arithmetic by provider order.
    ``selected_family_ids`` explicitly restricts available implementations; it
    grants no precision change and cannot bypass numerical matching.
    Returned compatibility fields are separate from the source-bound match
    record; callers must still use original proposal admission/lowering recipes.
    """
    if not is_sha256(original_source_sha256) or not is_sha256(preparation_sha256):
        raise ValueError("policy generation requires authenticated source and preparation SHA-256")
    if type(description) is not CompilerDescription:
        raise ValueError("policy generation requires a typed compiler description")
    description.__post_init__()
    if (type(numerical_families) is not tuple or not numerical_families
            or any(type(row) is not NumericalFamily for row in numerical_families)):
        raise ValueError("policy generation requires immutable numerical family declarations")
    if authorized_precision_changes and not _names(authorized_precision_changes):
        raise ValueError("precision authorization identities must be immutable and unique")
    if type(authorized_precision_changes) is not tuple:
        raise ValueError("precision authorization identities must be immutable")
    families = {row.id: row for row in description.families}
    patterns = {row.id: row for row in description.patterns}
    if selected_family_ids is not None and (
            not _names(selected_family_ids) or not set(selected_family_ids) <= families.keys()):
        raise ValueError("selected families require immutable unique declared family identities")
    identities = set()
    for row in numerical_families:
        row.__post_init__()
        identity = (row.family, row.pattern)
        if identity in identities:
            raise ValueError("duplicate numerical family/pattern declaration")
        identities.add(identity)
        family = families.get(row.family)
        if family is None or row.pattern not in family.patterns:
            raise ValueError("numerical contract names an undeclared family/pattern")
        if patterns[row.pattern].pipeline != family.pipeline:
            raise ValueError("numerical family and pattern pipelines differ")
    requirements = graph_requirements(source)
    for node in source.nodes:
        if node.type.dtype not in description.source_domain.types:
            raise PolicyGenerationError(f"{node.id}: type is outside the declared source domain")
    for requirement in requirements:
        if requirement["operation"] not in description.source_domain.operations:
            raise PolicyGenerationError(f'{requirement["node"]}: operation is outside the declared source domain')
    selected, fields = [], {}
    for requirement in requirements:
        matches = [row for row in numerical_families
                   if (selected_family_ids is None or row.family in selected_family_ids)
                   and requirement["operation"] in patterns[row.pattern].root_operations
                   and _matches(requirement, row)
                   and (row.precision_change is None or row.precision_change in authorized_precision_changes)]
        if len(matches) != 1:
            raise PolicyGenerationError(
                f'{requirement["node"]}: {requirement["operation"]} requires one numerical family; '
                f'found {len(matches)} authorized matches')
        row = matches[0]
        family = families[row.family]
        selected.append({
            "node": requirement["node"], "family": row.family, "pattern": row.pattern,
            "pipeline": family.pipeline, "implementation_policy": row.implementation_policy,
            "precision_change": row.precision_change,
            "fusion_preconditions": dict(family.fusion_preconditions),
            "numerical_contract": {key: json.loads(value) for key, value in row.numerical_contract},
            "status": "implementation_claim_pending_admission",
        })
        for key, value in row.policy_fields:
            if key in fields and fields[key] != value:
                raise PolicyGenerationError(f"selected families disagree on compatibility field {key}")
            fields[key] = value
    return {
        "schema": "merlin.generated_precision_policy.v1",
        "source_sha256": original_source_sha256,
        "graph_sha256": source.digest(),
        "preparation_sha256": preparation_sha256,
        "target_identity": source.target_identity,
        "compiler_description_sha256": _digest(description.record()),
        "numerical_families_sha256": _digest([asdict(row) for row in numerical_families]),
        "authorized_precision_changes": list(authorized_precision_changes),
        "selected_family_ids": list(selected_family_ids) if selected_family_ids is not None else None,
        "requirements": list(requirements), "selected_families": selected,
        "compatibility_policy_fields": {key: json.loads(value) for key, value in sorted(fields.items())},
    }


def generate_capture_policy(bound, capture_dir, description, numerical_families, *,
                            target_identity, source_operation_translators=(),
                            authorized_precision_changes=(), selected_family_ids=None,
                            bind_external_parameters=False, application_state=None,
                            max_state_bytes=64 * 1024 * 1024):
    """Generate family claims from receipt-checked IR, never a saved graph answer.

    Producer decoding remains outside Merlin. ``bound`` comes from that
    producer's explicit schema join. The source translator may declare target
    operations through the existing frontend protocol. Lifted literals are
    loaded from staged, authenticated bytes; parameters and mutable state keep
    their original input roles. The preparation identity names the staged IR.
    """
    from merlin.compiler.frontend.inputs import load_lifted_argument_bytes
    from merlin.compiler.frontend.policy_inputs import stage_policy_inputs
    from merlin.compiler.semantic.linalg.linalg_bridge import translate_linalg_text
    from merlin.compiler.semantic.source_preparation import prepare_source, verify_source_preparation

    if type(bind_external_parameters) is not bool:
        raise ValueError("external parameter identity selection must be Boolean")
    # The target bounds the lifted constants and captured state read for identity.
    if type(max_state_bytes) is not int or max_state_bytes < 0:
        raise ValueError("lifted state byte budget must be a nonnegative integer")
    with stage_policy_inputs(bound, capture_dir) as (directory, staged):
        from merlin.compiler.lower.c.parameters import StateDeclaration

        declaration = StateDeclaration.from_record(application_state, staged) if application_state is not None else None
        state_arguments = {port.argument_index for port in declaration.ports} if declaration else set()
        argument_bytes = load_lifted_argument_bytes(
            directory, staged, kinds=("constant", "buffer") if declaration else ("constant",),
            max_bytes=max_state_bytes,
        )
        constants = {index: payload for index, payload in argument_bytes.items() if index not in state_arguments}
        source = translate_linalg_text(
            staged.program, entry=staged.entry, target_identity=target_identity,
            lowering_policy="diagnostic", argument_constants=constants,
            source_operation_translators=source_operation_translators,
        ).request
        if bind_external_parameters:
            from merlin.compiler.lower.c.parameters import parameter_source_identity

            source = replace(source, source_identity=parameter_source_identity(source.source_identity, staged))
        if declaration is not None:
            source = replace(source, source_identity=sha256(
                (source.source_identity + ":state:" + declaration.compilation_identity(argument_bytes)).encode()
            ).hexdigest())
        translated_digest = source.digest()
        source, preparation = prepare_source(source)
        if verify_source_preparation(source, preparation).digest() != translated_digest:
            raise PolicyGenerationError("source preparation does not replay to the authenticated graph")
        authenticated_inputs = {
            "export_manifest_sha256": bound.export_manifest_sha256,
            "export_policy_sha256": bound.export_policy_sha256,
            "fx_manifest_sha256": bound.fx_manifest_sha256,
            "native_program_sha256": bound.native_program_sha256,
            "capture_receipt_sha256": bound.capture_receipt_sha256,
            "quantization_manifest_sha256": bound.quantization_manifest_sha256,
            "weights_sha256": staged.weights_sha256,
            "constants_sha256": staged.constants_sha256,
        }
        generated = generate_policy(
            source, description, numerical_families,
            original_source_sha256=staged.original_source_sha256,
            preparation_sha256=_digest({"program_sha256": staged.program_sha256,
                                        "source_preparation": preparation,
                                        "authenticated_inputs": authenticated_inputs}),
            authorized_precision_changes=authorized_precision_changes,
            selected_family_ids=selected_family_ids,
        )
        return {**generated, "authenticated_inputs": authenticated_inputs,
                "parameter_mode": "external" if bind_external_parameters else "source_inputs",
                "application_state_sha256": declaration.digest() if declaration else None,
                "capture_program_sha256": staged.program_sha256,
                "translated_graph_sha256": translated_digest, "source_preparation": preparation}
