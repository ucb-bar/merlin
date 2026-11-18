"""Protected automatic component selection wired into ordinary generation.

The live minimal software issuer supplies correspondences. Strict original
graph replay derives interaction classes; a fixed generic source factory makes
fresh bounded programs. Saved receipts recheck selected source bytes and the
derivation, but cannot issue hardware/software authority or complete UNKNOWNs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from merlin.common.paths import module_source_path
from merlin.development.targetgen.sources.frontend_use_def import original_use_def_semantics
from merlin.development.testgen.components import automatic_plan as P
from merlin.development.testgen.components.component_execution_budget import validate as validate_budget
from merlin.development.testgen.components.component_generation import digest
from merlin.development.testgen.components.coverage_plan import (
    BasisSource,
    ComponentCoveragePlan,
    ComponentSemanticBasis,
)
from merlin.development.testgen.intake.software import (
    REVIEW_SCHEMA,
    IndependentSoftwareIntake,
    _bindings,
    validate_minimal_software,
)

# ---- component_automatic -----------------------------------------------------------------------

SCHEMA = "merlin.component_automatic_policy.v1"


EFFECT_POLICY_SCHEMA = "merlin.component_automatic_policy.v2"


ARITHMETIC_POLICY_SCHEMA = "merlin.component_automatic_policy.v3"


LOGICAL_POLICY_SCHEMA = "merlin.component_automatic_policy.v4"


TYPED_POLICY_SCHEMA = "merlin.component_automatic_policy.v5"


PACKING_POLICY_SCHEMA = "merlin.component_automatic_policy.v6"


UNIFIED_POLICY_SCHEMA = "merlin.component_automatic_policy.v7"


ORIGINAL_POLICY_SCHEMA = "merlin.component_automatic_policy.v8"


LINEAR_POLICY_SCHEMA = "merlin.component_automatic_policy.v9"


POINTWISE_POLICY_SCHEMA = "merlin.component_automatic_policy.v10"


TRANSPOSE_POLICY_SCHEMA = "merlin.component_automatic_policy.v11"


BROADCAST_POLICY_SCHEMA = "merlin.component_automatic_policy.v12"


SCALAR_BINARY_POLICY_SCHEMA = "merlin.component_automatic_policy.v13"


INTEGER_SCALAR_POLICY_SCHEMA = "merlin.component_automatic_policy.v14"


METADATA_POLICY_SCHEMA = "merlin.component_automatic_policy.v15"


TRIANGULAR_POLICY_SCHEMA = "merlin.component_automatic_policy.v16"


RESHAPE_POLICY_SCHEMA = "merlin.component_automatic_policy.v17"


REDUCTION_POLICY_SCHEMA = "merlin.component_automatic_policy.v18"


RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v1"


EFFECT_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v2"


ARITHMETIC_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v3"


LOGICAL_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v4"


TYPED_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v5"


PACKING_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v6"


UNIFIED_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v7"


ORIGINAL_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v8"


LINEAR_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v9"


POINTWISE_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v10"


TRANSPOSE_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v11"


BROADCAST_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v12"


SCALAR_BINARY_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v13"


INTEGER_SCALAR_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v14"


METADATA_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v15"


TRIANGULAR_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v16"


RESHAPE_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v17"


REDUCTION_RECEIPT_SCHEMA = "merlin.component_automatic_derivation.v18"


_FIELDS = {
    "schema",
    "status",
    "hardware",
    "software_spec_sha256",
    "numerical_semantics_sha256",
    "semantic_basis_sha256",
    "budget",
    "execution_budget",
}


def _original_source_version(policy):
    return {
        ORIGINAL_POLICY_SCHEMA: 1,
        LINEAR_POLICY_SCHEMA: 2,
        POINTWISE_POLICY_SCHEMA: 3,
        TRANSPOSE_POLICY_SCHEMA: 4,
        BROADCAST_POLICY_SCHEMA: 5,
        SCALAR_BINARY_POLICY_SCHEMA: 6,
        INTEGER_SCALAR_POLICY_SCHEMA: 7,
        METADATA_POLICY_SCHEMA: 8,
        TRIANGULAR_POLICY_SCHEMA: 9,
        RESHAPE_POLICY_SCHEMA: 10,
        REDUCTION_POLICY_SCHEMA: 11,
    }[policy["schema"]]


def _pin(path, role):
    path = Path(path).absolute()
    if any(member.is_symlink() for member in (path, *path.parents)) or not path.is_file():
        raise ValueError("automatic selection requires ordinary explicit source files")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "role": role}


def _read(pin):
    selected = _pin(pin["path"], pin["role"])
    if selected != pin:
        raise ValueError("automatic derivation selected source bytes or identity changed")
    return Path(pin["path"]).read_bytes()


def _closed_policy(policy):
    additions = {
        EFFECT_POLICY_SCHEMA: {"operator_schema_intake_sha256"},
        TYPED_POLICY_SCHEMA: {"operator_schema_intake_sha256"},
        ARITHMETIC_POLICY_SCHEMA: {"arithmetic_intake_sha256"},
        PACKING_POLICY_SCHEMA: {"packing_intake_sha256"},
        ORIGINAL_POLICY_SCHEMA: {
            "operator_schema_intake_sha256",
            "arithmetic_intake_sha256",
            "packing_intake_sha256",
            "original_source_budget",
        },
        UNIFIED_POLICY_SCHEMA: {
            "operator_schema_intake_sha256",
            "arithmetic_intake_sha256",
            "packing_intake_sha256",
        },
    }
    additions[LINEAR_POLICY_SCHEMA] = additions[ORIGINAL_POLICY_SCHEMA]
    additions[POINTWISE_POLICY_SCHEMA] = additions[ORIGINAL_POLICY_SCHEMA]
    additions[TRANSPOSE_POLICY_SCHEMA] = additions[ORIGINAL_POLICY_SCHEMA]
    additions[BROADCAST_POLICY_SCHEMA] = additions[ORIGINAL_POLICY_SCHEMA]
    additions[SCALAR_BINARY_POLICY_SCHEMA] = additions[ORIGINAL_POLICY_SCHEMA]
    additions[INTEGER_SCALAR_POLICY_SCHEMA] = additions[ORIGINAL_POLICY_SCHEMA]
    additions[METADATA_POLICY_SCHEMA] = additions[ORIGINAL_POLICY_SCHEMA]
    additions[TRIANGULAR_POLICY_SCHEMA] = additions[ORIGINAL_POLICY_SCHEMA]
    additions[RESHAPE_POLICY_SCHEMA] = additions[ORIGINAL_POLICY_SCHEMA]
    additions[REDUCTION_POLICY_SCHEMA] = additions[ORIGINAL_POLICY_SCHEMA]
    fields = _FIELDS | additions.get(policy.get("schema") if isinstance(policy, dict) else None, set())
    if isinstance(policy, dict) and policy.get("schema") in {
        LOGICAL_POLICY_SCHEMA,
        TYPED_POLICY_SCHEMA,
        PACKING_POLICY_SCHEMA,
    }:
        fields |= set(policy) & {"arithmetic_intake_sha256", "operator_schema_intake_sha256"}
    if (
        isinstance(policy, dict)
        and policy.get("schema") == ARITHMETIC_POLICY_SCHEMA
        and "operator_schema_intake_sha256" in policy
    ):
        fields |= {"operator_schema_intake_sha256"}
    if (
        not isinstance(policy, dict)
        or set(policy) != fields
        or policy["schema"]
        not in {
            SCHEMA,
            EFFECT_POLICY_SCHEMA,
            ARITHMETIC_POLICY_SCHEMA,
            LOGICAL_POLICY_SCHEMA,
            TYPED_POLICY_SCHEMA,
            PACKING_POLICY_SCHEMA,
            UNIFIED_POLICY_SCHEMA,
            ORIGINAL_POLICY_SCHEMA,
            LINEAR_POLICY_SCHEMA,
            POINTWISE_POLICY_SCHEMA,
            TRANSPOSE_POLICY_SCHEMA,
            BROADCAST_POLICY_SCHEMA,
            SCALAR_BINARY_POLICY_SCHEMA,
            INTEGER_SCALAR_POLICY_SCHEMA,
            METADATA_POLICY_SCHEMA,
            TRIANGULAR_POLICY_SCHEMA,
            RESHAPE_POLICY_SCHEMA,
            REDUCTION_POLICY_SCHEMA,
        }
        or policy["status"] != "reviewed"
    ):
        raise ValueError(
            "automatic coverage requires the closed reviewed preauthor policy without authored obligations"
        )
    validate_budget(policy["execution_budget"])
    if policy["schema"] in {
        ORIGINAL_POLICY_SCHEMA,
        LINEAR_POLICY_SCHEMA,
        POINTWISE_POLICY_SCHEMA,
        TRANSPOSE_POLICY_SCHEMA,
        BROADCAST_POLICY_SCHEMA,
        SCALAR_BINARY_POLICY_SCHEMA,
        INTEGER_SCALAR_POLICY_SCHEMA,
        METADATA_POLICY_SCHEMA,
        TRIANGULAR_POLICY_SCHEMA,
        RESHAPE_POLICY_SCHEMA,
        REDUCTION_POLICY_SCHEMA,
    }:
        from merlin.development.testgen.originals.original_call_sources import (
            validate_budget as validate_original_budget,
        )

        validate_original_budget(policy["original_source_budget"])
    budget = policy["budget"]
    if (
        not isinstance(budget, dict)
        or set(budget) != {"max_members", "max_interaction_cells"}
        or any(type(value) is not int or value < 1 for value in budget.values())
    ):
        raise ValueError("automatic coverage needs explicit finite member and interaction budgets")
    return policy


def _selected_effects(policy):
    return policy["schema"] == EFFECT_POLICY_SCHEMA or (
        policy["schema"]
        in {
            ARITHMETIC_POLICY_SCHEMA,
            LOGICAL_POLICY_SCHEMA,
            TYPED_POLICY_SCHEMA,
            PACKING_POLICY_SCHEMA,
            UNIFIED_POLICY_SCHEMA,
            ORIGINAL_POLICY_SCHEMA,
            LINEAR_POLICY_SCHEMA,
            POINTWISE_POLICY_SCHEMA,
            TRANSPOSE_POLICY_SCHEMA,
            BROADCAST_POLICY_SCHEMA,
            SCALAR_BINARY_POLICY_SCHEMA,
            INTEGER_SCALAR_POLICY_SCHEMA,
            METADATA_POLICY_SCHEMA,
            TRIANGULAR_POLICY_SCHEMA,
            RESHAPE_POLICY_SCHEMA,
            REDUCTION_POLICY_SCHEMA,
        }
        and "operator_schema_intake_sha256" in policy
    )


def _selected_arithmetic(policy):
    return policy["schema"] == ARITHMETIC_POLICY_SCHEMA or (
        policy["schema"]
        in {
            LOGICAL_POLICY_SCHEMA,
            TYPED_POLICY_SCHEMA,
            PACKING_POLICY_SCHEMA,
            UNIFIED_POLICY_SCHEMA,
            ORIGINAL_POLICY_SCHEMA,
            LINEAR_POLICY_SCHEMA,
            POINTWISE_POLICY_SCHEMA,
            TRANSPOSE_POLICY_SCHEMA,
            BROADCAST_POLICY_SCHEMA,
            SCALAR_BINARY_POLICY_SCHEMA,
            INTEGER_SCALAR_POLICY_SCHEMA,
            METADATA_POLICY_SCHEMA,
            TRIANGULAR_POLICY_SCHEMA,
            RESHAPE_POLICY_SCHEMA,
            REDUCTION_POLICY_SCHEMA,
        }
        and "arithmetic_intake_sha256" in policy
    )


def _policy(raw, *, evidence, basis):
    policy = _closed_policy(yaml.safe_load(raw))
    if basis is None or policy["semantic_basis_sha256"] != basis.source.sha256:
        raise ValueError("automatic coverage requires the exact selected independent semantic basis")
    expected = {key: evidence.derivation_identity[key] for key in ("contract_sha256", "raw_facts_sha256")}
    sources = [row for row in evidence.source_snapshots if row.role == "software-spec"]
    if (
        policy["hardware"] != expected
        or len(sources) != 1
        or policy["software_spec_sha256"] != sources[0].sha256
        or policy["numerical_semantics_sha256"] != digest(evidence.software_spec["numerical_semantics"])
    ):
        raise ValueError("automatic coverage selected hardware/software/numerical identity changed")
    return policy


def _relations(basis):
    declaration = json.loads(basis.declaration_json)
    return [
        (member["id"], original_use_def_semantics(json.loads(Path(source.path).read_bytes())))
        for member, source in zip(declaration["members"], basis.graph_sources, strict=True)
    ]


def require_basis_selection(path, *, recipe, software_intake):
    """Reject substituted graph selection before opening any example source."""
    document = yaml.safe_load(Path(path).read_bytes())
    if not isinstance(document, dict) or document.get("schema") not in {
        SCHEMA,
        EFFECT_POLICY_SCHEMA,
        ARITHMETIC_POLICY_SCHEMA,
        LOGICAL_POLICY_SCHEMA,
        TYPED_POLICY_SCHEMA,
        PACKING_POLICY_SCHEMA,
        UNIFIED_POLICY_SCHEMA,
        ORIGINAL_POLICY_SCHEMA,
        LINEAR_POLICY_SCHEMA,
        POINTWISE_POLICY_SCHEMA,
        TRANSPOSE_POLICY_SCHEMA,
        BROADCAST_POLICY_SCHEMA,
        SCALAR_BINARY_POLICY_SCHEMA,
        INTEGER_SCALAR_POLICY_SCHEMA,
        METADATA_POLICY_SCHEMA,
        TRIANGULAR_POLICY_SCHEMA,
        RESHAPE_POLICY_SCHEMA,
        REDUCTION_POLICY_SCHEMA,
    }:
        return
    if type(software_intake) is not IndependentSoftwareIntake:
        raise ValueError("automatic coverage needs the live protected minimal software intake")
    software_intake.verify()
    review_pin = next(pin for pin in software_intake.source_pins if pin.role == "protected-minimal-review")
    review = yaml.safe_load(Path(review_pin.path).read_bytes())
    selected = (yaml.safe_load(Path(recipe).read_bytes()) or {}).get("semantic_basis")
    if not isinstance(selected, dict) or set(selected) != {"path", "sha256"}:
        raise ValueError("automatic coverage needs the exact protected example roster selection")
    chosen = Path(selected["path"])
    chosen = chosen if chosen.is_absolute() else Path(recipe).absolute().parent / chosen
    expected = Path(review["semantic_basis"]["path"])
    expected = expected if expected.is_absolute() else Path(review_pin.path).parent / expected
    if chosen.absolute() != expected.absolute() or selected["sha256"] != review["semantic_basis"]["sha256"]:
        raise ValueError("automatic example roster differs from the protected source selection")


def resolve(
    path,
    *,
    evidence,
    semantic_basis,
    hardware_intake,
    software_intake,
    output_root,
    operator_schema_intake=None,
    arithmetic_intake=None,
    packing_intake=None,
):
    """Resolve old explicit plans or the new independently derived normal v2 plan."""
    raw = Path(path).read_bytes()
    selected = yaml.safe_load(raw)
    if not isinstance(selected, dict) or selected.get("schema") not in {
        SCHEMA,
        EFFECT_POLICY_SCHEMA,
        ARITHMETIC_POLICY_SCHEMA,
        LOGICAL_POLICY_SCHEMA,
        TYPED_POLICY_SCHEMA,
        PACKING_POLICY_SCHEMA,
        UNIFIED_POLICY_SCHEMA,
        ORIGINAL_POLICY_SCHEMA,
        LINEAR_POLICY_SCHEMA,
        POINTWISE_POLICY_SCHEMA,
        TRANSPOSE_POLICY_SCHEMA,
        BROADCAST_POLICY_SCHEMA,
        SCALAR_BINARY_POLICY_SCHEMA,
        INTEGER_SCALAR_POLICY_SCHEMA,
        METADATA_POLICY_SCHEMA,
        TRIANGULAR_POLICY_SCHEMA,
        RESHAPE_POLICY_SCHEMA,
        REDUCTION_POLICY_SCHEMA,
    }:
        if arithmetic_intake is not None or operator_schema_intake is not None or packing_intake is not None:
            raise ValueError("independent source observations require an explicit versioned automatic policy")
        return ComponentCoveragePlan.load(path, evidence=evidence, semantic_basis=semantic_basis), None
    if type(software_intake) is not IndependentSoftwareIntake or software_intake.hardware is not hardware_intake:
        raise ValueError("automatic coverage needs the live protected minimal software and identical hardware intake")
    software_intake.verify()
    policy = _policy(raw, evidence=evidence, basis=semantic_basis)
    receipt = json.loads(software_intake.receipt_json)
    if receipt["semantic_basis_sha256"] != semantic_basis.source.sha256:
        raise ValueError("automatic coverage example basis differs from protected minimal software review")
    review_pin = next(pin for pin in software_intake.source_pins if pin.role == "protected-minimal-review")
    review = yaml.safe_load(Path(review_pin.path).read_bytes())
    spec = software_intake.public_facts()
    _bindings(review, spec, semantic_basis)
    protected_graphs = {pin.path for pin in software_intake.source_pins if pin.role == "independent-example-graph"}
    if protected_graphs != {source.path for source in semantic_basis.graph_sources}:
        raise ValueError("automatic graph sources differ from exact protected independent example membership")
    relations = _relations(semantic_basis)
    effects, schema_record = None, None
    if _selected_effects(policy):
        from merlin.development.testgen.intake.operator_schemas import IndependentOperatorSchemaIntake

        if (
            type(operator_schema_intake) is not IndependentOperatorSchemaIntake
            or operator_schema_intake.software is not software_intake
        ):
            raise ValueError("automatic effect selection needs the identical live independent schema/software intake")
        schema_record = operator_schema_intake.record()
        if policy["operator_schema_intake_sha256"] != operator_schema_intake.sha256:
            raise ValueError("automatic effect selection differs from protected native schema observations")
        effects = [
            (member, operator_schema_intake.effects(graph_path=source.path))
            for (member, _), source in zip(relations, semantic_basis.graph_sources, strict=True)
        ]
    elif operator_schema_intake is not None:
        raise ValueError("operator effects require the explicit versioned automatic policy")
    arithmetic_record = None
    if _selected_arithmetic(policy):
        from merlin.development.testgen.intake.arithmetic_intake import IndependentArithmeticIntake

        if (
            type(arithmetic_intake) is not IndependentArithmeticIntake
            or arithmetic_intake.hardware is not hardware_intake
        ):
            raise ValueError("automatic arithmetic needs the identical live independent hardware intake")
        arithmetic_record = arithmetic_intake.record()
        if policy["arithmetic_intake_sha256"] != arithmetic_intake.sha256:
            raise ValueError("automatic arithmetic differs from protected actual typed SSA observations")
    elif arithmetic_intake is not None:
        raise ValueError("local arithmetic requires the explicit versioned automatic policy")
    packing_record = None
    if policy["schema"] in {
        PACKING_POLICY_SCHEMA,
        UNIFIED_POLICY_SCHEMA,
        ORIGINAL_POLICY_SCHEMA,
        LINEAR_POLICY_SCHEMA,
        POINTWISE_POLICY_SCHEMA,
        TRANSPOSE_POLICY_SCHEMA,
        BROADCAST_POLICY_SCHEMA,
        SCALAR_BINARY_POLICY_SCHEMA,
        INTEGER_SCALAR_POLICY_SCHEMA,
        METADATA_POLICY_SCHEMA,
        TRIANGULAR_POLICY_SCHEMA,
        RESHAPE_POLICY_SCHEMA,
        REDUCTION_POLICY_SCHEMA,
    }:
        from merlin.development.testgen.intake.packing_intake import IndependentPackingIntake

        if type(packing_intake) is not IndependentPackingIntake or packing_intake.hardware is not hardware_intake:
            raise ValueError("automatic packing needs the identical live independent hardware intake")
        packing_record = packing_intake.record()
        if policy["packing_intake_sha256"] != packing_intake.sha256:
            raise ValueError("automatic packing differs from protected actual typed SSA observations")
    elif packing_intake is not None:
        raise ValueError("local packing requires the explicit versioned automatic policy")
    typed_record = None
    if policy["schema"] in {
        TYPED_POLICY_SCHEMA,
        UNIFIED_POLICY_SCHEMA,
        ORIGINAL_POLICY_SCHEMA,
        LINEAR_POLICY_SCHEMA,
        POINTWISE_POLICY_SCHEMA,
        TRANSPOSE_POLICY_SCHEMA,
        BROADCAST_POLICY_SCHEMA,
        SCALAR_BINARY_POLICY_SCHEMA,
        INTEGER_SCALAR_POLICY_SCHEMA,
        METADATA_POLICY_SCHEMA,
        TRIANGULAR_POLICY_SCHEMA,
        RESHAPE_POLICY_SCHEMA,
        REDUCTION_POLICY_SCHEMA,
    }:
        from merlin.development.testgen.originals import typed_add_sources as T

        typed_record = T.observe(
            schema_record=schema_record,
            basis=semantic_basis,
            numerical_semantics=spec["numerical_semantics"],
            destination=Path(output_root) / "coverage" / "automatic-typed-add",
        )
    original_record = None
    if policy["schema"] in {
        ORIGINAL_POLICY_SCHEMA,
        LINEAR_POLICY_SCHEMA,
        POINTWISE_POLICY_SCHEMA,
        TRANSPOSE_POLICY_SCHEMA,
        BROADCAST_POLICY_SCHEMA,
        SCALAR_BINARY_POLICY_SCHEMA,
        INTEGER_SCALAR_POLICY_SCHEMA,
        METADATA_POLICY_SCHEMA,
        TRIANGULAR_POLICY_SCHEMA,
        RESHAPE_POLICY_SCHEMA,
        REDUCTION_POLICY_SCHEMA,
    }:
        from merlin.development.testgen.originals import original_call_sources as O

        original_record = O.observe(
            schema_record=schema_record,
            basis=semantic_basis,
            numerical_semantics=spec["numerical_semantics"],
            budget=policy["original_source_budget"],
            destination=Path(output_root) / "coverage" / "automatic-original-calls",
            version=_original_source_version(policy),
        )
    declaration, unknowns = P.derive(
        policy,
        spec=spec,
        review=review,
        basis=semantic_basis,
        relations=relations,
        effects=effects,
        arithmetic=arithmetic_record["facts"] if arithmetic_record is not None else None,
        logical_interactions=policy["schema"]
        in {
            LOGICAL_POLICY_SCHEMA,
            TYPED_POLICY_SCHEMA,
            PACKING_POLICY_SCHEMA,
            UNIFIED_POLICY_SCHEMA,
            ORIGINAL_POLICY_SCHEMA,
            LINEAR_POLICY_SCHEMA,
            POINTWISE_POLICY_SCHEMA,
            TRANSPOSE_POLICY_SCHEMA,
            BROADCAST_POLICY_SCHEMA,
            SCALAR_BINARY_POLICY_SCHEMA,
            INTEGER_SCALAR_POLICY_SCHEMA,
            METADATA_POLICY_SCHEMA,
            TRIANGULAR_POLICY_SCHEMA,
            RESHAPE_POLICY_SCHEMA,
            REDUCTION_POLICY_SCHEMA,
        },
        typed_add=T.forms(typed_record, basis=semantic_basis) if typed_record is not None else None,
        packing=packing_record["facts"] if packing_record is not None else None,
        retain_historical_gaps=policy["schema"]
        in {
            UNIFIED_POLICY_SCHEMA,
            ORIGINAL_POLICY_SCHEMA,
            LINEAR_POLICY_SCHEMA,
            POINTWISE_POLICY_SCHEMA,
            TRANSPOSE_POLICY_SCHEMA,
            BROADCAST_POLICY_SCHEMA,
            SCALAR_BINARY_POLICY_SCHEMA,
            INTEGER_SCALAR_POLICY_SCHEMA,
            METADATA_POLICY_SCHEMA,
            TRIANGULAR_POLICY_SCHEMA,
            RESHAPE_POLICY_SCHEMA,
            REDUCTION_POLICY_SCHEMA,
        },
    )
    if original_record is not None:
        unknowns = O.merge_unknowns(unknowns, original_record, basis=semantic_basis, unknown=P._unknown)
    destination = Path(output_root) / "coverage" / "automatic-selection"
    if any(member.is_symlink() for member in (destination, *destination.parents)):
        raise ValueError("automatic selection output must have an ordinary explicit path")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    plan_path = destination / "derived-plan.json"
    plan_path.write_text(json.dumps(declaration, sort_keys=True, indent=2) + "\n")
    plan_path.chmod(0o600)
    sources = [
        _pin(path, "automatic-policy"),
        _pin(review_pin.path, "minimal-review"),
        _pin(software_intake.source.path, "minimal-software"),
    ]
    sources += [_pin(row["path"], row["role"]) for row in semantic_basis.sources()]
    sources += [
        _pin(module_source_path(name), "automatic-reader")
        for name in (
            __name__,
            P.__name__,
            "merlin.development.targetgen.sources.frontend_use_def",
            "merlin.development.targetgen.sources.frontend_trace",
            "merlin.development.testgen.components.coverage_plan",
            "merlin.development.testgen.intake.software",
            "merlin.development.testgen.components.automatic_plan",
        )
    ]
    if packing_record is not None:
        sources.extend(
            _pin(module_source_path(name), "packing-source-reader")
            for name in (
                "merlin.development.testgen.components.automatic_plan",
                "merlin.development.targetgen.corpus.binding",
            )
        )
    if typed_record is not None:
        sources += [
            _pin(member, "typed-add-observation")
            for member in sorted((Path(output_root) / "coverage" / "automatic-typed-add").rglob("*"))
            if member.is_file()
        ]
        sources += [
            _pin(module_source_path(name), "typed-add-reader")
            for name in (
                "merlin.development.targetgen.sources.frontend_typed_add",
                "merlin.development.targetgen.sources.schema_defaults_observer",
                "merlin.development.testgen.originals.typed_add_sources",
                "merlin.development.testgen.originals.original_schema_defaults",
            )
        ]
    if original_record is not None:
        sources += [
            _pin(member, "original-call-observation")
            for member in sorted((Path(output_root) / "coverage" / "automatic-original-calls").rglob("*"))
            if member.is_file()
        ]
        sources += [
            _pin(module_source_path(name), "original-call-reader")
            for name in O.reader_modules(_original_source_version(policy))
        ]
    record = {
        "schema": (
            REDUCTION_RECEIPT_SCHEMA
            if policy["schema"] == REDUCTION_POLICY_SCHEMA
            else RESHAPE_RECEIPT_SCHEMA
            if policy["schema"] == RESHAPE_POLICY_SCHEMA
            else TRIANGULAR_RECEIPT_SCHEMA
            if policy["schema"] == TRIANGULAR_POLICY_SCHEMA
            else METADATA_RECEIPT_SCHEMA
            if policy["schema"] == METADATA_POLICY_SCHEMA
            else INTEGER_SCALAR_RECEIPT_SCHEMA
            if policy["schema"] == INTEGER_SCALAR_POLICY_SCHEMA
            else SCALAR_BINARY_RECEIPT_SCHEMA
            if policy["schema"] == SCALAR_BINARY_POLICY_SCHEMA
            else BROADCAST_RECEIPT_SCHEMA
            if policy["schema"] == BROADCAST_POLICY_SCHEMA
            else TRANSPOSE_RECEIPT_SCHEMA
            if policy["schema"] == TRANSPOSE_POLICY_SCHEMA
            else POINTWISE_RECEIPT_SCHEMA
            if policy["schema"] == POINTWISE_POLICY_SCHEMA
            else LINEAR_RECEIPT_SCHEMA
            if policy["schema"] == LINEAR_POLICY_SCHEMA
            else ORIGINAL_RECEIPT_SCHEMA
            if policy["schema"] == ORIGINAL_POLICY_SCHEMA
            else UNIFIED_RECEIPT_SCHEMA
            if policy["schema"] == UNIFIED_POLICY_SCHEMA
            else PACKING_RECEIPT_SCHEMA
            if policy["schema"] == PACKING_POLICY_SCHEMA
            else TYPED_RECEIPT_SCHEMA
            if policy["schema"] == TYPED_POLICY_SCHEMA
            else LOGICAL_RECEIPT_SCHEMA
            if policy["schema"] == LOGICAL_POLICY_SCHEMA
            else ARITHMETIC_RECEIPT_SCHEMA
            if arithmetic_record is not None
            else EFFECT_RECEIPT_SCHEMA
            if schema_record is not None
            else RECEIPT_SCHEMA
        ),
        "hardware_intake_sha256": hardware_intake.sha256,
        "software_intake_sha256": software_intake.sha256,
        "sources": sources,
        "derived_plan": _pin(plan_path, "automatic-derived-plan"),
        "relation_semantics": [{"member": member, **relation.public_semantics()} for member, relation in relations],
        "required_unknowns": unknowns,
        "qualification": "fresh bounded source classes; no original dimensions/topology or physical resource grants",
    }
    if schema_record is not None:
        record["operator_schema_intake"] = schema_record
        record["operator_effect_semantics"] = [
            {"member": member, **effect.public_semantics()} for member, effect in effects
        ]
    if arithmetic_record is not None:
        record["arithmetic_intake"] = arithmetic_record
    if typed_record is not None:
        record["typed_add_sources"] = typed_record
    if packing_record is not None:
        record["packing_intake"] = packing_record
    if original_record is not None:
        record["original_call_sources"] = original_record
    record["sha256"] = digest(record)
    receipt_path = destination / "derivation.json"
    receipt_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    receipt_path.chmod(0o600)
    plan = (
        ComponentCoveragePlan.load(plan_path, evidence=evidence, semantic_basis=semantic_basis)
        if declaration["obligations"]
        else ComponentCoveragePlan(
            str(plan_path),
            hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            json.dumps(declaration, sort_keys=True, separators=(",", ":")),
            (),
        )
    )
    return plan, record
