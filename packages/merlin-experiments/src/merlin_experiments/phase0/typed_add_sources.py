"""Actual public-schema defaults bound to original typed add source premises.

This observation extends an already issued original operator-schema intake;
it issues no software, numerical, effect or hardware capability.
"""

from __future__ import annotations

import json

from merlin.targetgen.frontend_typed_add import add_forms

from . import original_schema_defaults as D

ENVIRONMENT = D.ENVIRONMENT
SCHEMA = "merlin.original_typed_add_sources.v1"


def _observer():
    return D.observer()


def observe(*, schema_record, basis, numerical_semantics, destination):
    """Run the fixed reader over every exact protected original schema member."""
    rows = D.observe_members(schema_record=schema_record, basis=basis, destination=destination, version=1)
    for row in rows:
        trace, schemas, defaults = D.verify_member(row, schema_record=schema_record, version=1)
        row["forms"] = add_forms(trace, schemas, defaults, numerical_semantics=numerical_semantics)
    return verify(
        {"schema": SCHEMA, "members": rows},
        schema_record=schema_record,
        basis=basis,
        numerical_semantics=numerical_semantics,
    )


def verify(record, *, schema_record, basis, numerical_semantics):
    """Recompute exact source forms from terminal native observations."""
    if not isinstance(record, dict) or set(record) != {"schema", "members"} or record["schema"] != SCHEMA:
        raise ValueError("typed add sources require their explicit closed observation version")
    if [row["graph_path"] for row in record["members"]] != [source.path for source in basis.graph_sources]:
        raise ValueError("typed add sources lost the complete original protected graph roster")
    for row in record["members"]:
        if set(row) != {"graph_path", "request", "observation", "invocation", "forms"}:
            raise ValueError("typed add source member observation fields changed")
        trace, schemas, defaults = D.verify_member(row, schema_record=schema_record, version=1)
        if row["forms"] != add_forms(trace, schemas, defaults, numerical_semantics=numerical_semantics):
            raise ValueError("typed add source premises differ from original typed SSA/default replay")
    return record


def forms(record, *, basis):
    return [
        (member["id"], row["forms"])
        for member, row in zip(json.loads(basis.declaration_json)["members"], record["members"], strict=True)
    ]
