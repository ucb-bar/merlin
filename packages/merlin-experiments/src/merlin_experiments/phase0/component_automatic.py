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
from merlin.targetgen.frontend_use_def import original_use_def_semantics

from . import component_automatic_plan as P
from .component_coverage_plan import ComponentCoveragePlan
from .component_execution_budget import validate as validate_budget
from .component_generation import digest
from .component_semantic_basis import BasisSource, ComponentSemanticBasis
from .minimal_software import validate_minimal_software
from .software_intake import REVIEW_SCHEMA, IndependentSoftwareIntake, _bindings

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
    }:
        from .original_call_sources import validate_budget as validate_original_budget

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
        from .operator_schema_intake import IndependentOperatorSchemaIntake

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
        from .arithmetic_intake import IndependentArithmeticIntake

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
    }:
        from .packing_intake import IndependentPackingIntake

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
    }:
        from . import typed_add_sources as T

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
    }:
        from . import original_call_sources as O

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
            "merlin.targetgen.frontend_use_def",
            "merlin.targetgen.frontend_trace",
            "merlin_experiments.phase0.component_semantic_basis",
            "merlin_experiments.phase0.minimal_software",
            "merlin_experiments.phase0.component_arithmetic_obligations",
        )
    ]
    if packing_record is not None:
        sources.extend(
            _pin(module_source_path(name), "packing-source-reader")
            for name in (
                "merlin_experiments.phase0.component_packing_sources",
                "merlin.targetgen.corpus_spec",
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
                "merlin.targetgen.frontend_typed_add",
                "merlin.targetgen.torch_schema_defaults_observer",
                "merlin_experiments.phase0.typed_add_sources",
                "merlin_experiments.phase0.original_schema_defaults",
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
            METADATA_RECEIPT_SCHEMA
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


def verify(record, *, report, verify_sources=True):
    """Reopen all selected originals and recompute the complete required roster."""
    if record.get("schema") not in {
        RECEIPT_SCHEMA,
        EFFECT_RECEIPT_SCHEMA,
        ARITHMETIC_RECEIPT_SCHEMA,
        LOGICAL_RECEIPT_SCHEMA,
        TYPED_RECEIPT_SCHEMA,
        PACKING_RECEIPT_SCHEMA,
        UNIFIED_RECEIPT_SCHEMA,
        ORIGINAL_RECEIPT_SCHEMA,
        LINEAR_RECEIPT_SCHEMA,
        POINTWISE_RECEIPT_SCHEMA,
        TRANSPOSE_RECEIPT_SCHEMA,
        BROADCAST_RECEIPT_SCHEMA,
        SCALAR_BINARY_RECEIPT_SCHEMA,
        INTEGER_SCALAR_RECEIPT_SCHEMA,
        METADATA_RECEIPT_SCHEMA,
    } or digest({k: v for k, v in record.items() if k != "sha256"}) != record.get("sha256"):
        raise ValueError("automatic component derivation identity changed")
    identity = report["generation_identity"]
    if identity.get("automatic_derivation_sha256") != digest(record) or any(
        record[key] != identity.get(key) for key in ("hardware_intake_sha256", "software_intake_sha256")
    ):
        raise ValueError("automatic derivation independent generation binding changed")
    if not verify_sources:
        raise ValueError(
            "automatic derivation requires its explicit original source replay; relocated closure is unqualified"
        )
    sources = record["sources"]
    for source in sources:
        _read(source)

    def one(role):
        rows = [row for row in sources if row["role"] == role]
        if len(rows) != 1:
            raise ValueError("automatic derivation requires exact selected source membership")
        return rows[0]

    policy = _closed_policy(yaml.safe_load(_read(one("automatic-policy"))))
    if (policy["schema"] == LOGICAL_POLICY_SCHEMA) != (record["schema"] == LOGICAL_RECEIPT_SCHEMA):
        raise ValueError("logical interaction derivation requires its explicit original versioned policy")
    if (policy["schema"] == TYPED_POLICY_SCHEMA) != (record["schema"] == TYPED_RECEIPT_SCHEMA):
        raise ValueError("typed add derivation requires its explicit original versioned policy")
    if (policy["schema"] == PACKING_POLICY_SCHEMA) != (record["schema"] == PACKING_RECEIPT_SCHEMA):
        raise ValueError("packing derivation requires its explicit original versioned policy")
    if (policy["schema"] == UNIFIED_POLICY_SCHEMA) != (record["schema"] == UNIFIED_RECEIPT_SCHEMA):
        raise ValueError("unified derivation requires its explicit original versioned policy")
    if (policy["schema"] == ORIGINAL_POLICY_SCHEMA) != (record["schema"] == ORIGINAL_RECEIPT_SCHEMA):
        raise ValueError("original call derivation requires its explicit original versioned policy")
    if (policy["schema"] == LINEAR_POLICY_SCHEMA) != (record["schema"] == LINEAR_RECEIPT_SCHEMA):
        raise ValueError("linear original source derivation requires its explicit versioned policy")
    if (policy["schema"] == POINTWISE_POLICY_SCHEMA) != (record["schema"] == POINTWISE_RECEIPT_SCHEMA):
        raise ValueError("pointwise original source derivation requires its explicit versioned policy")
    if (policy["schema"] == BROADCAST_POLICY_SCHEMA) != (record["schema"] == BROADCAST_RECEIPT_SCHEMA):
        raise ValueError("broadcast original source derivation requires its explicit versioned policy")
    if (policy["schema"] == METADATA_POLICY_SCHEMA) != (record["schema"] == METADATA_RECEIPT_SCHEMA):
        raise ValueError("metadata original source derivation requires its explicit versioned policy")
    if (policy["schema"] == INTEGER_SCALAR_POLICY_SCHEMA) != (record["schema"] == INTEGER_SCALAR_RECEIPT_SCHEMA):
        raise ValueError("integer scalar original source derivation requires its explicit versioned policy")
    if (policy["schema"] == SCALAR_BINARY_POLICY_SCHEMA) != (record["schema"] == SCALAR_BINARY_RECEIPT_SCHEMA):
        raise ValueError("scalar binary original source derivation requires its explicit versioned policy")
    if (policy["schema"] == TRANSPOSE_POLICY_SCHEMA) != (record["schema"] == TRANSPOSE_RECEIPT_SCHEMA):
        raise ValueError("transpose original source derivation requires its explicit versioned policy")
    basis_pin = one("semantic-basis-roster")
    basis = ComponentSemanticBasis.load(
        _read(basis_pin),
        source=BasisSource(basis_pin["path"], basis_pin["sha256"], basis_pin["role"]),
        parent=Path(basis_pin["path"]).parent,
        routing={},
    )
    if {row["path"] for row in sources if row["role"] == "semantic-basis-graph"} != {
        row.path for row in basis.graph_sources
    }:
        raise ValueError("automatic derivation independent graph membership changed")
    review = yaml.safe_load(_read(one("minimal-review")))
    if (
        not isinstance(review, dict)
        or set(review) != {"schema", "target", "source", "semantic_basis", "numerical_choices", "operation_basis"}
        or review["schema"] != REVIEW_SCHEMA
    ):
        raise ValueError("automatic derivation protected review schema changed")
    spec = validate_minimal_software(yaml.safe_load(_read(one("minimal-software"))), target=review["target"])
    if (
        policy["hardware"] != report["hardware"]
        or policy["software_spec_sha256"] != one("minimal-software")["sha256"]
        or policy["semantic_basis_sha256"] != basis.source.sha256
        or policy["numerical_semantics_sha256"] != digest(spec["numerical_semantics"])
    ):
        raise ValueError("automatic derivation selected semantic identity changed")
    if (
        review["source"]["sha256"] != one("minimal-software")["sha256"]
        or review["semantic_basis"]["sha256"] != basis.source.sha256
    ):
        raise ValueError("automatic derivation protected source review binding changed")
    _bindings(review, spec, basis)
    relations = _relations(basis)
    effects = None
    selected_effects = _selected_effects(policy)
    if selected_effects:
        from merlin.targetgen.frontend_operator_effects import original_operator_effects

        from .operator_schema_intake import verify_record

        if record["schema"] not in {
            EFFECT_RECEIPT_SCHEMA,
            ARITHMETIC_RECEIPT_SCHEMA,
            LOGICAL_RECEIPT_SCHEMA,
            TYPED_RECEIPT_SCHEMA,
            PACKING_RECEIPT_SCHEMA,
            UNIFIED_RECEIPT_SCHEMA,
            ORIGINAL_RECEIPT_SCHEMA,
            LINEAR_RECEIPT_SCHEMA,
            POINTWISE_RECEIPT_SCHEMA,
            TRANSPOSE_RECEIPT_SCHEMA,
            BROADCAST_RECEIPT_SCHEMA,
            SCALAR_BINARY_RECEIPT_SCHEMA,
            INTEGER_SCALAR_RECEIPT_SCHEMA,
            METADATA_RECEIPT_SCHEMA,
        } or not {
            "operator_schema_intake",
            "operator_effect_semantics",
        }.issubset(record):
            raise ValueError("automatic derivation lost its selected original operator effect observation")
        schema_record = verify_record(record["operator_schema_intake"])
        if (
            policy["schema"]
            not in {
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
            }
            or hashlib.sha256(
                (json.dumps(schema_record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            ).hexdigest()
            != policy["operator_schema_intake_sha256"]
            or schema_record["software_intake_sha256"] != record["software_intake_sha256"]
        ):
            raise ValueError("automatic effect receipt differs from its exact protected intake identity")
        schema_members = {row["graph_path"]: row for row in schema_record["members"]}
        if set(schema_members) != {source.path for source in basis.graph_sources}:
            raise ValueError("automatic effects lost protected original graph membership")
        effects = [
            (
                member,
                original_operator_effects(
                    json.loads(Path(source.path).read_bytes()),
                    json.loads(Path(schema_members[source.path]["observation"]).read_bytes()),
                    tensor_arguments=(
                        json.loads(Path(schema_members[source.path]["tensor_arguments"]["observation"]).read_bytes())
                        if "tensor_arguments" in schema_members[source.path]
                        else None
                    ),
                    zero_returns=(
                        json.loads(Path(schema_members[source.path]["zero_returns"]["observation"]).read_bytes())
                        if "zero_returns" in schema_members[source.path]
                        else None
                    ),
                ),
            )
            for (member, _), source in zip(relations, basis.graph_sources, strict=True)
        ]
        if record["operator_effect_semantics"] != [
            {"member": member, **effect.public_semantics()} for member, effect in effects
        ]:
            raise ValueError("automatic effects differ from native source/argument/result replay")
    elif policy["schema"] not in {
        SCHEMA,
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
    } or set(record) & {
        "operator_schema_intake",
        "operator_effect_semantics",
    }:
        raise ValueError("historical automatic policy cannot acquire new effect authority")
    arithmetic = None
    if _selected_arithmetic(policy) and record["schema"] in {
        ARITHMETIC_RECEIPT_SCHEMA,
        LOGICAL_RECEIPT_SCHEMA,
        TYPED_RECEIPT_SCHEMA,
        PACKING_RECEIPT_SCHEMA,
        UNIFIED_RECEIPT_SCHEMA,
        ORIGINAL_RECEIPT_SCHEMA,
        LINEAR_RECEIPT_SCHEMA,
        POINTWISE_RECEIPT_SCHEMA,
        TRANSPOSE_RECEIPT_SCHEMA,
        BROADCAST_RECEIPT_SCHEMA,
        SCALAR_BINARY_RECEIPT_SCHEMA,
        INTEGER_SCALAR_RECEIPT_SCHEMA,
        METADATA_RECEIPT_SCHEMA,
    }:
        from .arithmetic_intake import verify_record

        if "arithmetic_intake" not in record:
            raise ValueError("automatic derivation lost its selected original arithmetic observation")
        selected = verify_record(record["arithmetic_intake"])
        if (
            policy["schema"]
            not in {
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
            }
            or hashlib.sha256(
                (json.dumps(selected, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            ).hexdigest()
            != policy["arithmetic_intake_sha256"]
            or selected["hardware_intake_sha256"] != record["hardware_intake_sha256"]
        ):
            raise ValueError("automatic arithmetic receipt differs from exact protected original hardware")
        arithmetic = selected["facts"]
    elif _selected_arithmetic(policy) or "arithmetic_intake" in record:
        raise ValueError("historical automatic policy cannot acquire local arithmetic authority")
    packing = None
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
    }:
        from .packing_intake import verify_record

        if "packing_intake" not in record:
            raise ValueError("automatic derivation lost its selected original packing observation")
        selected = verify_record(record["packing_intake"])
        if (
            hashlib.sha256(
                (json.dumps(selected, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            ).hexdigest()
            != policy["packing_intake_sha256"]
            or selected["hardware_intake_sha256"] != record["hardware_intake_sha256"]
        ):
            raise ValueError("automatic packing receipt differs from exact protected original hardware")
        if {row["path"] for row in sources if row["role"] == "packing-source-reader"} != {
            str(module_source_path(name))
            for name in (
                "merlin_experiments.phase0.component_packing_sources",
                "merlin.targetgen.corpus_spec",
            )
        }:
            raise ValueError("automatic packing lost its exact source preparation reader")
        packing = selected["facts"]
    elif "packing_intake" in record or any(row["role"] == "packing-source-reader" for row in sources):
        raise ValueError("historical automatic policy cannot acquire local packing authority")
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
    }:
        from . import typed_add_sources as T

        if "typed_add_sources" not in record:
            raise ValueError("typed add derivation lost its actual original native source observation")
        typed_record = T.verify(
            record["typed_add_sources"],
            schema_record=schema_record,
            basis=basis,
            numerical_semantics=spec["numerical_semantics"],
        )
    elif "typed_add_sources" in record:
        raise ValueError("historical automatic policy cannot acquire typed add source premises")
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
    }:
        from . import original_call_sources as O

        if "original_call_sources" not in record:
            raise ValueError("original call derivation lost its complete actual native/source roster")
        original_record = O.verify(
            record["original_call_sources"],
            schema_record=schema_record,
            basis=basis,
            numerical_semantics=spec["numerical_semantics"],
        )
        expected_schema = {
            1: O.SCHEMA,
            2: O.LINEAR_SCHEMA,
            3: O.POINTWISE_SCHEMA,
            4: O.TRANSPOSE_SCHEMA,
            5: O.BROADCAST_SCHEMA,
            6: O.SCALAR_BINARY_SCHEMA,
            7: O.INTEGER_SCALAR_SCHEMA,
            8: O.METADATA_SCHEMA,
        }[_original_source_version(policy)]
        if original_record["schema"] != expected_schema:
            raise ValueError("original source factory version differs from the explicitly selected policy")
        if original_record["budget"] != policy["original_source_budget"] or {
            row["path"] for row in sources if row["role"] == "original-call-reader"
        } != {str(module_source_path(name)) for name in O.reader_modules(_original_source_version(policy))}:
            raise ValueError("original call derivation changed its explicit source budget or readers")
    elif "original_call_sources" in record or any(row["role"].startswith("original-call-") for row in sources):
        raise ValueError("historical automatic policy cannot acquire original typed source preparation")
    declaration, unknowns = P.derive(
        policy,
        spec=spec,
        review=review,
        basis=basis,
        relations=relations,
        effects=effects,
        arithmetic=arithmetic,
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
        },
        typed_add=T.forms(typed_record, basis=basis) if typed_record is not None else None,
        packing=packing,
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
        },
    )
    if original_record is not None:
        unknowns = O.merge_unknowns(unknowns, original_record, basis=basis, unknown=P._unknown)
    if (
        yaml.safe_load(_read(record["derived_plan"])) != declaration
        or report["declaration"] != declaration
        or record["required_unknowns"] != unknowns
        or record["relation_semantics"]
        != [{"member": member, **relation.public_semantics()} for member, relation in relations]
    ):
        raise ValueError("automatic derivation disagrees with original graph replay or source factory")
    expected = [row["id"] for row in declaration["obligations"]] + [row["id"] for row in unknowns]
    if [row["id"] for row in report["obligations"]] != expected:
        raise ValueError("automatic derivation lost an original required obligation")
    for actual, wanted in zip(
        report["obligations"][len(declaration["obligations"]) :], P.required_unknown_rows(unknowns), strict=True
    ):
        if actual != wanted:
            raise ValueError("automatic unsupported/resource obligation cannot acquire a fabricated witness")
    return record
