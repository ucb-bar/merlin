"""Observe original typed calls and construct independently bounded source forms.

Every original call stays in the private denominator. Fresh loaders use fixed
guard/transfer extents rather than example dimensions. Their existence grants
no numerical capsule coverage, software correspondence or target capability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.common.jsonio import canonical_json
from merlin.targetgen.frontend_original_call import call_contracts
from merlin.targetgen.original_operator_sources import (
    ADD_FORM_SCHEMA,
    FORM_SCHEMA,
    MATMUL_FORM_SCHEMA,
    add_source,
    conv2d_forms,
    conv2d_source,
    matmul_forms,
    matmul_source,
    original_add_forms,
    policy_compatibility,
)

from . import original_schema_defaults as D

SCHEMA = "merlin.original_call_sources.v1"
LINEAR_SCHEMA = "merlin.original_call_sources.v2"
POINTWISE_SCHEMA = "merlin.original_call_sources.v3"
TRANSPOSE_SCHEMA = "merlin.original_call_sources.v4"
BROADCAST_SCHEMA = "merlin.original_call_sources.v5"
SCALAR_BINARY_SCHEMA = "merlin.original_call_sources.v6"
INTEGER_SCALAR_SCHEMA = "merlin.original_call_sources.v7"
METADATA_SCHEMA = "merlin.original_call_sources.v8"
TRIANGULAR_SCHEMA = "merlin.original_call_sources.v9"
BUDGET_SCHEMA = "merlin.original_call_source_budget.v1"
READER_MODULES = (
    __name__,
    "merlin_experiments.phase0.original_schema_defaults",
    "merlin.targetgen.frontend_original_call",
    "merlin.targetgen.original_operator_sources",
    "merlin.targetgen.frontend_typed_add",
    "merlin.targetgen.torch_schema_defaults_observer",
)
_LIMITS = {
    "max_sources",
    "max_tensor_elements",
    "max_scalar_products",
    "max_source_bytes",
    "max_total_tensor_elements",
    "max_total_scalar_products",
    "max_total_source_bytes",
}
_COHORTS = (("functional_guard", 1), ("functional_guard", 2), ("withheld_transfer", 3))


def _same(actual, expected, *, version):
    # Preserve historical equality exactly. The new literal source record
    # compares validated finite JSON with scalar types and signed zero intact.
    return canonical_json(actual) == canonical_json(expected) if version >= 6 else actual == expected


def required_source_cohorts():
    """Ordered original teaching-source membership; never hardware tail evidence.

    This is the same fixed roster used by historical source records. A later
    reference/IR join must preserve every slot rather than choose its own subset.
    """
    return _COHORTS


def reader_modules(version):
    if type(version) is not int or version not in {1, 2, 3, 4, 5, 6, 7, 8, 9}:
        raise ValueError("original source readers need an explicit supported factory version")
    return (
        READER_MODULES
        + (("merlin.targetgen.original_pointwise_sources",) if version >= 3 else ())
        + (("merlin.targetgen.original_transpose_sources",) if version >= 4 else ())
        + (("merlin.targetgen.original_broadcast_add_sources",) if version >= 5 else ())
        + (("merlin.targetgen.original_scalar_binary_sources",) if version >= 6 else ())
        + (
            (
                "merlin.targetgen.original_metadata_sources",
                "merlin_experiments.phase0.zero_return_intake",
                "merlin.targetgen.torch_zero_return_observer",
            )
            if version >= 8
            else ()
        )
        + (("merlin.targetgen.original_triangular_sources",) if version >= 9 else ())
    )


def validate_budget(budget):
    if (
        not isinstance(budget, dict)
        or set(budget) != {"schema", *_LIMITS}
        or budget["schema"] != BUDGET_SCHEMA
        or any(type(budget[key]) is not int or budget[key] < 1 for key in _LIMITS)
    ):
        raise ValueError("original source forms need complete explicit finite construction budgets")
    return budget


def _forms(trace, schemas, defaults, *, numerical_semantics, version, tensor_bindings=None, zero_returns=None):
    add_forms = original_add_forms
    if version >= 5:
        from merlin.targetgen.original_broadcast_add_sources import broadcast_add_forms

        add_forms = broadcast_add_forms
    factories = [conv2d_forms] if version == 1 else [conv2d_forms, matmul_forms, add_forms]
    if version >= 3:
        from merlin.targetgen.original_pointwise_sources import pointwise_forms

        factories.append(pointwise_forms)
    if version >= 4:
        from merlin.targetgen.original_transpose_sources import transpose_forms

        factories.append(transpose_forms)
    forms = [
        form
        for factory in factories
        for form in factory(trace, schemas, defaults, numerical_semantics=numerical_semantics)
    ]
    if version >= 6:
        from merlin.targetgen.original_scalar_binary_sources import scalar_binary_forms

        forms.extend(
            scalar_binary_forms(
                trace,
                schemas,
                defaults,
                numerical_semantics=numerical_semantics,
                version=2 if version >= 7 else 1,
                tensor_bindings=tensor_bindings,
            )
        )
    if version >= 8:
        from merlin.targetgen.original_metadata_sources import metadata_forms

        forms.extend(
            metadata_forms(trace, schemas, defaults, numerical_semantics=numerical_semantics, zero_returns=zero_returns)
        )
    if version >= 9:
        from merlin.targetgen.original_triangular_sources import triu_forms

        forms.extend(triu_forms(trace, schemas, defaults, numerical_semantics=numerical_semantics))
    return forms


def _tensor_bindings(schema_record, graph_path, version):
    if version not in {7, 8, 9}:
        return None
    # Missing native selection remains a factory refusal in every original
    # cohort. This record is data; live schema ownership is replayed separately.
    if schema_record.get("schema") not in {
        "merlin.independent_operator_schema_intake.v2",
        "merlin.independent_operator_schema_intake.v3",
    }:
        return None
    rows = [row for row in schema_record["members"] if row["graph_path"] == graph_path]
    if len(rows) != 1:
        raise ValueError("integer scalar sources require the exact original graph's native bindings")
    return rows[0]["tensor_bindings"]


def _zero_returns(schema_record, graph_path, trace, schemas, version):
    if version not in {8, 9} or schema_record.get("schema") != "merlin.independent_operator_schema_intake.v3":
        return None
    from .zero_return_intake import verify_returns

    rows = [row for row in schema_record["members"] if row["graph_path"] == graph_path]
    if len(rows) != 1:
        raise ValueError("metadata sources lost their complete original zero-return graph member")
    return verify_returns(
        trace=trace,
        schema_observation=schemas,
        getter=schema_record["zero_return_getter"],
        member=rows[0]["zero_returns"],
    )


def _sources(calls, forms, *, budget, total, requested, version=1):
    """Derive the entire requested source roster before any loader allocation."""
    pointwise = {}
    if version >= 3:
        from merlin.targetgen.original_pointwise_sources import FORM_SCHEMA as POINTWISE_FORM_SCHEMA
        from merlin.targetgen.original_pointwise_sources import pointwise_source

        pointwise = {POINTWISE_FORM_SCHEMA: pointwise_source}
    if version >= 4:
        from merlin.targetgen.original_transpose_sources import FORM_SCHEMA as TRANSPOSE_FORM_SCHEMA
        from merlin.targetgen.original_transpose_sources import transpose_source

        pointwise[TRANSPOSE_FORM_SCHEMA] = transpose_source
    if version >= 5:
        from merlin.targetgen.original_broadcast_add_sources import FORM_SCHEMA as BROADCAST_FORM_SCHEMA
        from merlin.targetgen.original_broadcast_add_sources import broadcast_add_source

        pointwise[BROADCAST_FORM_SCHEMA] = broadcast_add_source
    if version >= 6:
        from merlin.targetgen.original_scalar_binary_sources import FORM_SCHEMA as SCALAR_BINARY_FORM_SCHEMA
        from merlin.targetgen.original_scalar_binary_sources import INTEGER_FORM_SCHEMA, scalar_binary_source

        pointwise[SCALAR_BINARY_FORM_SCHEMA if version == 6 else INTEGER_FORM_SCHEMA] = scalar_binary_source
    if version >= 8:
        from merlin.targetgen.original_metadata_sources import ASSERTION_SCHEMA, CAST_SCHEMA, metadata_source

        pointwise.update({CAST_SCHEMA: metadata_source, ASSERTION_SCHEMA: metadata_source})
    if version >= 9:
        from merlin.targetgen.original_triangular_sources import FORM_SCHEMA as TRIANGULAR_FORM_SCHEMA
        from merlin.targetgen.original_triangular_sources import triu_source

        pointwise[TRIANGULAR_FORM_SCHEMA] = triu_source
    indexed = {form["node"]: form for form in forms}
    result = []
    for call in calls:
        for cohort, extent in required_source_cohorts():
            row = {"node": call["node"], "target": call["target"], "cohort": cohort, "extent": extent}
            try:
                if requested > budget["max_sources"]:
                    raise ValueError("complete original call source roster exceeds its declared member budget")
                form = indexed.get(call["node"])
                if form is None:
                    raise ValueError("original operator has no implemented typed original-form source factory")
                factory = (
                    conv2d_source
                    if version == 1
                    else {
                        FORM_SCHEMA: conv2d_source,
                        MATMUL_FORM_SCHEMA: matmul_source,
                        ADD_FORM_SCHEMA: add_source,
                        **pointwise,
                    }[form["form_schema"]]
                )
                if version >= 2 and form["status"] != "supported":
                    raise ValueError(form["reason"])
                source = factory(form, extent=extent, max_tensor_elements=budget["max_tensor_elements"])
                metadata = source.metadata()
                costs = {key: metadata[key] for key in ("tensor_elements", "scalar_products")}
                costs["source_bytes"] = len(source.loader.encode())
                if any(costs[key] > budget["max_" + key] for key in costs):
                    raise ValueError("original typed source exceeds its explicit per-member construction budget")
                # Retain failed requested members without charging a loader
                # that will not be constructed or executing any tensor code.
                if any(total[key] + costs[key] > budget["max_total_" + key] for key in costs):
                    raise ValueError("original typed source exceeds its complete-roster construction budget")
                for key, count in costs.items():
                    total[key] += count
                row.update(
                    status="source_constructed",
                    metadata=metadata,
                    costs=costs,
                    source_sha256=hashlib.sha256(source.loader.encode()).hexdigest(),
                )
                result.append((row, source.loader))
            except (KeyError, TypeError, ValueError) as error:
                row.update(status="unknown", reason=str(error))
                result.append((row, None))
    return result


def observe(*, schema_record, basis, numerical_semantics, budget, destination, version=1):
    """Write source-only original forms through the selected normal observer."""
    validate_budget(budget)
    if type(version) is not int or version not in {1, 2, 3, 4, 5, 6, 7, 8, 9}:
        raise ValueError("original source observation requires an explicit supported factory version")
    destination = Path(destination)
    rows = D.observe_members(schema_record=schema_record, basis=basis, destination=destination, version=2)
    for ordinal, row in enumerate(rows):
        trace, schemas, defaults = D.verify_member(row, schema_record=schema_record, version=2)
        zero_returns = _zero_returns(schema_record, row["graph_path"], trace, schemas, version)
        calls = call_contracts(trace, schemas, defaults, zero_returns=zero_returns)
        forms = _forms(
            trace,
            schemas,
            defaults,
            numerical_semantics=numerical_semantics,
            version=version,
            tensor_bindings=_tensor_bindings(schema_record, row["graph_path"], version),
            zero_returns=zero_returns,
        )
        row.update(
            calls=calls,
            forms=forms,
            policy_compatibility=[
                {"node": form["node"], **policy_compatibility(form, numerical_semantics)} for form in forms
            ],
            source_members=[],
        )
    total = dict.fromkeys(("tensor_elements", "scalar_products", "source_bytes"), 0)
    requested = sum(len(row["calls"]) for row in rows) * len(required_source_cohorts())
    for ordinal, row in enumerate(rows):
        for index, (member, loader) in enumerate(
            _sources(row["calls"], row["forms"], budget=budget, total=total, requested=requested, version=version)
        ):
            if loader is not None:
                path = destination / str(ordinal) / ("source-" + str(index) + ".py")
                path.write_text(loader)
                path.chmod(0o600)
                member["source"] = {"path": str(path), "sha256": member["source_sha256"]}
            row["source_members"].append(member)
    record = {
        "schema": {
            1: SCHEMA,
            2: LINEAR_SCHEMA,
            3: POINTWISE_SCHEMA,
            4: TRANSPOSE_SCHEMA,
            5: BROADCAST_SCHEMA,
            6: SCALAR_BINARY_SCHEMA,
            7: INTEGER_SCALAR_SCHEMA,
            8: METADATA_SCHEMA,
            9: TRIANGULAR_SCHEMA,
        }[version],
        "budget": budget,
        "members": rows,
    }
    return verify(record, schema_record=schema_record, basis=basis, numerical_semantics=numerical_semantics)


def verify(record, *, schema_record, basis, numerical_semantics):
    """Reconstruct every original binding and fresh loader from actual defaults."""
    if (
        not isinstance(record, dict)
        or set(record) != {"schema", "budget", "members"}
        or record["schema"]
        not in {
            SCHEMA,
            LINEAR_SCHEMA,
            POINTWISE_SCHEMA,
            TRANSPOSE_SCHEMA,
            BROADCAST_SCHEMA,
            SCALAR_BINARY_SCHEMA,
            INTEGER_SCALAR_SCHEMA,
            METADATA_SCHEMA,
            TRIANGULAR_SCHEMA,
        }
    ):
        raise ValueError("original call sources require their closed observation version")
    budget = validate_budget(record["budget"])
    version = {
        SCHEMA: 1,
        LINEAR_SCHEMA: 2,
        POINTWISE_SCHEMA: 3,
        TRANSPOSE_SCHEMA: 4,
        BROADCAST_SCHEMA: 5,
        SCALAR_BINARY_SCHEMA: 6,
        INTEGER_SCALAR_SCHEMA: 7,
        METADATA_SCHEMA: 8,
        TRIANGULAR_SCHEMA: 9,
    }[record["schema"]]
    if [row["graph_path"] for row in record["members"]] != [source.path for source in basis.graph_sources]:
        raise ValueError("original call sources changed their complete protected graph membership")
    total = dict.fromkeys(("tensor_elements", "scalar_products", "source_bytes"), 0)
    requested = sum(len(row["calls"]) for row in record["members"]) * len(required_source_cohorts())
    for row in record["members"]:
        if set(row) != {
            "graph_path",
            "request",
            "observation",
            "invocation",
            "calls",
            "forms",
            "policy_compatibility",
            "source_members",
        }:
            raise ValueError("original call source member fields changed")
        trace, schemas, defaults = D.verify_member(row, schema_record=schema_record, version=2)
        zero_returns = _zero_returns(schema_record, row["graph_path"], trace, schemas, version)
        calls = call_contracts(trace, schemas, defaults, zero_returns=zero_returns)
        forms = _forms(
            trace,
            schemas,
            defaults,
            numerical_semantics=numerical_semantics,
            version=version,
            tensor_bindings=_tensor_bindings(schema_record, row["graph_path"], version),
            zero_returns=zero_returns,
        )
        compatibility = [{"node": form["node"], **policy_compatibility(form, numerical_semantics)} for form in forms]
        expected = _sources(calls, forms, budget=budget, total=total, requested=requested, version=version)
        if not all(
            _same(actual, expected, version=version)
            for actual, expected in (
                (row["calls"], calls),
                (row["forms"], forms),
                (row["policy_compatibility"], compatibility),
            )
        ):
            raise ValueError("original typed bindings/forms/policy differ from actual original schema replay")
        if len(row["source_members"]) != len(expected):
            raise ValueError("original source preparation lost a requested guard or private transfer member")
        for index, (member, (wanted, loader)) in enumerate(zip(row["source_members"], expected, strict=True)):
            if loader is not None:
                pin = member.get("source")
                if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
                    raise ValueError("original typed source lost its exact loader identity")
                path = Path(pin["path"])
                if path != Path(row["observation"]).parent / ("source-" + str(index) + ".py"):
                    raise ValueError("original typed source changed its exact construction owner path")
                if any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file():
                    raise ValueError("original typed source requires an ordinary explicit loader path")
                if path.read_bytes() != loader.encode() or pin["sha256"] != wanted["source_sha256"]:
                    raise ValueError("original typed source differs from its independently reconstructed loader")
                wanted["source"] = pin
            if not _same(member, wanted, version=version):
                raise ValueError("original typed source metadata or missing member differs from original replay")
    return record


def required_unknowns(record, *, basis, unknown):
    """Source construction never removes original admission requirements."""
    result = []
    for original, row in zip(json.loads(basis.declaration_json)["members"], record["members"], strict=True):
        by_node = {}
        for member in row["source_members"]:
            by_node.setdefault(member["node"], []).append(member)
        for call in row["calls"]:
            selector = {"member": original["id"], "node": call["node"], "target": call["target"]}
            if call["status"] != "bound":
                result.append(unknown("original_call_binding", selector, call["reason"]))
            for member in by_node.get(call["node"], []):
                if member["status"] == "unknown":
                    result.append(
                        unknown(
                            "original_operator_factory",
                            {**selector, "cohort": member["cohort"], "extent": member["extent"]},
                            member["reason"],
                        )
                    )
            result.append(
                unknown(
                    "original_operator_admission",
                    selector,
                    "original typed source needs protected owner correspondence, independent complete reference "
                    "and comparison; source construction grants no capsule coverage",
                )
            )
    return result


def merge_unknowns(historical, record, *, basis, unknown):
    """Retain historical scopes and each exact original call's missing admission."""
    rows = {row["id"]: row for row in historical}
    for row in required_unknowns(record, basis=basis, unknown=unknown):
        if row["id"] in rows and rows[row["id"]] != row:
            raise ValueError("original call source scopes disagree on a required missing obligation")
        rows[row["id"]] = row
    return sorted(rows.values(), key=lambda row: row["id"])
