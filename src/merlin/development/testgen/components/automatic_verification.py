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


def verify(record, *, report, verify_sources=True):
    """Reopen all selected originals and recompute the complete required roster."""
    from merlin.development.testgen.components.automatic_selection import (
        ARITHMETIC_POLICY_SCHEMA,
        ARITHMETIC_RECEIPT_SCHEMA,
        BROADCAST_POLICY_SCHEMA,
        BROADCAST_RECEIPT_SCHEMA,
        EFFECT_POLICY_SCHEMA,
        EFFECT_RECEIPT_SCHEMA,
        INTEGER_SCALAR_POLICY_SCHEMA,
        INTEGER_SCALAR_RECEIPT_SCHEMA,
        LINEAR_POLICY_SCHEMA,
        LINEAR_RECEIPT_SCHEMA,
        LOGICAL_POLICY_SCHEMA,
        LOGICAL_RECEIPT_SCHEMA,
        METADATA_POLICY_SCHEMA,
        METADATA_RECEIPT_SCHEMA,
        ORIGINAL_POLICY_SCHEMA,
        ORIGINAL_RECEIPT_SCHEMA,
        PACKING_POLICY_SCHEMA,
        PACKING_RECEIPT_SCHEMA,
        POINTWISE_POLICY_SCHEMA,
        POINTWISE_RECEIPT_SCHEMA,
        RECEIPT_SCHEMA,
        REDUCTION_POLICY_SCHEMA,
        REDUCTION_RECEIPT_SCHEMA,
        RESHAPE_POLICY_SCHEMA,
        RESHAPE_RECEIPT_SCHEMA,
        SCALAR_BINARY_POLICY_SCHEMA,
        SCALAR_BINARY_RECEIPT_SCHEMA,
        SCHEMA,
        TRANSPOSE_POLICY_SCHEMA,
        TRANSPOSE_RECEIPT_SCHEMA,
        TRIANGULAR_POLICY_SCHEMA,
        TRIANGULAR_RECEIPT_SCHEMA,
        TYPED_POLICY_SCHEMA,
        TYPED_RECEIPT_SCHEMA,
        UNIFIED_POLICY_SCHEMA,
        UNIFIED_RECEIPT_SCHEMA,
        _closed_policy,
        _original_source_version,
        _read,
        _relations,
        _selected_arithmetic,
        _selected_effects,
    )

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
        TRIANGULAR_RECEIPT_SCHEMA,
        RESHAPE_RECEIPT_SCHEMA,
        REDUCTION_RECEIPT_SCHEMA,
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
    if (policy["schema"] == REDUCTION_POLICY_SCHEMA) != (record["schema"] == REDUCTION_RECEIPT_SCHEMA):
        raise ValueError("reduction original source derivation requires its explicit versioned policy")
    if (policy["schema"] == RESHAPE_POLICY_SCHEMA) != (record["schema"] == RESHAPE_RECEIPT_SCHEMA):
        raise ValueError("reshape original source derivation requires its explicit versioned policy")
    if (policy["schema"] == TRIANGULAR_POLICY_SCHEMA) != (record["schema"] == TRIANGULAR_RECEIPT_SCHEMA):
        raise ValueError("triangular original source derivation requires its explicit versioned policy")
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
        from merlin.development.targetgen.sources.operator_effects import original_operator_effects
        from merlin.development.testgen.intake.operator_schemas import verify_record

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
            TRIANGULAR_RECEIPT_SCHEMA,
            RESHAPE_RECEIPT_SCHEMA,
            REDUCTION_RECEIPT_SCHEMA,
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
                TRIANGULAR_POLICY_SCHEMA,
                RESHAPE_POLICY_SCHEMA,
                REDUCTION_POLICY_SCHEMA,
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
        TRIANGULAR_POLICY_SCHEMA,
        RESHAPE_POLICY_SCHEMA,
        REDUCTION_POLICY_SCHEMA,
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
        TRIANGULAR_RECEIPT_SCHEMA,
        RESHAPE_RECEIPT_SCHEMA,
        REDUCTION_RECEIPT_SCHEMA,
    }:
        from merlin.development.testgen.intake.arithmetic_intake import verify_record

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
                TRIANGULAR_POLICY_SCHEMA,
                RESHAPE_POLICY_SCHEMA,
                REDUCTION_POLICY_SCHEMA,
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
        TRIANGULAR_POLICY_SCHEMA,
        RESHAPE_POLICY_SCHEMA,
        REDUCTION_POLICY_SCHEMA,
    }:
        from merlin.development.testgen.intake.packing_intake import verify_record

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
                "merlin.development.testgen.components.automatic_plan",
                "merlin.development.targetgen.corpus.binding",
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
        TRIANGULAR_POLICY_SCHEMA,
        RESHAPE_POLICY_SCHEMA,
        REDUCTION_POLICY_SCHEMA,
    }:
        from merlin.development.testgen.originals import typed_add_sources as T

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
        TRIANGULAR_POLICY_SCHEMA,
        RESHAPE_POLICY_SCHEMA,
        REDUCTION_POLICY_SCHEMA,
    }:
        from merlin.development.testgen.originals import original_call_sources as O

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
            9: O.TRIANGULAR_SCHEMA,
            10: O.RESHAPE_SCHEMA,
            11: O.REDUCTION_SCHEMA,
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
            TRIANGULAR_POLICY_SCHEMA,
            RESHAPE_POLICY_SCHEMA,
            REDUCTION_POLICY_SCHEMA,
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
            TRIANGULAR_POLICY_SCHEMA,
            RESHAPE_POLICY_SCHEMA,
            REDUCTION_POLICY_SCHEMA,
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
