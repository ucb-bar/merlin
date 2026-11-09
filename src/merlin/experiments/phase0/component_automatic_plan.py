"""Shape-free source relations select fixed independent component programs.

Original examples select semantic classes only. This owner never copies their
dimensions, topology, fanout, frequencies, constants or precision policies.
Reviewed software declares arithmetic; the ordinary writer supplies the oracle.
"""

from __future__ import annotations

from .component_coverage_plan import BUDGETED_PLAN_SCHEMA
from .component_generation import digest

_SOURCE_FORMS = {
    "aten.matmul.default": "contraction",
    "aten.mm.default": "contraction",
    "aten.clone.default": "movement",
}
_LOGICAL_COPY_INTERACTIONS = {"shared_producer_multiple_consumers", "publication_and_further_use"}


def _program(factory):
    def axis(name):
        return {"axis": name}

    inputs = [{"name": "A", "role": "input", "shape": [axis("M"), axis("K")], "dtype": "operand"}]
    if factory in _LOGICAL_COPY_INTERACTIONS:
        # Independent bounded semantic graphs; original class presence selects
        # them, never the original graph's dimensions, fanout or topology.
        last_input = "P" if factory == "shared_producer_multiple_consumers" else "C0"
        nodes = [
            {"name": "P", "op": "copy", "inputs": ["A"]},
            {"name": "C0", "op": "copy", "inputs": ["P"]},
            {"name": "C1", "op": "copy", "inputs": [last_input]},
        ]
        outputs = [{"name": name, "value": value} for name, value in (("Yproducer", "P"), ("Y0", "C0"), ("Y1", "C1"))]
    elif factory == "may_alias_result":
        # A logical identity view is an independent possibility admitted by
        # may-alias annotations. It does not exercise arbitrary view shapes
        # or require physical pointer equality from a downstream compiler.
        nodes = [{"name": "P", "op": "alias", "inputs": ["A"]}, {"name": "C", "op": "copy", "inputs": ["P"]}]
        outputs = [{"name": "Yinput", "value": "A"}, {"name": "Yview", "value": "P"}, {"name": "Ycopy", "value": "C"}]
    elif factory == "movement":
        nodes = [{"name": "P", "op": "copy", "inputs": ["A"]}]
        outputs = [{"name": "Y", "value": "P"}]
    elif factory == "elementwise_add":
        inputs.append({"name": "B", "role": "input", "shape": [axis("M"), axis("K")], "dtype": "operand"})
        nodes = [{"name": "P", "op": "add", "inputs": ["A", "B"]}]
        outputs = [{"name": "Y", "value": "P"}]
    else:
        count = 2 if factory == "shared_input_multiple_consumers" else 1
        inputs += [
            {"name": "W" + str(index), "role": "weight", "shape": [axis("K"), axis("N")], "dtype": "operand"}
            for index in range(count)
        ]
        nodes = [
            {"name": "P" + str(index), "op": "matmul", "inputs": ["A", "W" + str(index)]} for index in range(count)
        ]
        outputs = [{"name": "Y" + str(index), "value": "P" + str(index)} for index in range(count)]
    return {"inputs": inputs, "nodes": nodes, "outputs": outputs}


def _factory(owner, *, typed=False):
    selected = {"movement": "copy", "contraction": "matmul"}
    if typed and "add" in owner.get("ops", []):
        selected["elementwise_add"] = "add"
    eligible = [
        family
        for family, operation in selected.items()
        if family in owner.get("families", []) or operation in owner.get("ops", [])
    ]
    return eligible[0] if len(eligible) == 1 else None


def _unknown(kind, selector, reason):
    return {
        "id": "auto_missing_" + digest({"kind": kind, "selector": selector}),
        "kind": kind,
        "selector": selector,
        "reason": reason,
    }


def derive(
    policy,
    *,
    spec,
    review,
    basis,
    relations,
    effects=None,
    arithmetic=None,
    logical_interactions=False,
    typed_add=None,
    packing=None,
    retain_historical_gaps=False,
):
    """Construct a complete required class roster without invented permissions.

    One and two are fresh bounded semantic extents, independent of target or
    example shape. Private transfer sources use three instead. These are not
    aligned/tail, capacity, tiling or address-boundary witnesses. Those require
    an independently qualified hardware role/axis map and remain mandatory.
    """
    owners = {row["id"]: row for row in spec["operations"]}
    owner_links = {}
    for link in review["operation_basis"]:
        owner_links.setdefault(link["owner"], []).append(link)
    unknowns, cases, present = [], {}, set()
    examples = {row["id"]: row for row in basis.semantics()}
    typed_forms = dict(typed_add or [])
    for member, relation in relations:
        present.update(relation.interaction_classes)
        for operation in relation.operations:
            # Review is member-specific; another example's occurrence cannot
            # manufacture a correspondence for this source call family.
            selected = {
                link["owner"]
                for link in review["operation_basis"]
                if link["member"] == member and operation in link["operations"]
            }
            if not selected:
                unknowns.append(
                    _unknown(
                        "operation", operation, "original operation has no protected reviewed software correspondence"
                    )
                )
            for owner in selected:
                factory = _factory(owners[owner], typed=typed_add is not None)
                if factory == "elementwise_add":
                    forms = [form for form in typed_forms.get(member, []) if form["target"] == operation]
                    signature = owners[owner].get("signature", {})

                    def dtype(value):
                        return (
                            "int" + value[1:]
                            if isinstance(value, str) and value.startswith("i") and value[1:].isdigit()
                            else value
                        )

                    inputs = [dtype(value) for value in signature.get("ordered_operand_dtypes", [])]
                    outputs = [dtype(value) for value in signature.get("ordered_result_dtypes", [])]
                    if not forms or any(
                        form["status"] != "supported"
                        or form.get("operand_dtypes") != inputs
                        or form.get("result_dtypes") != outputs
                        or signature.get("broadcasting") != form.get("broadcasting")
                        for form in forms
                    ):
                        unknowns.append(
                            _unknown(
                                "source_operator_form",
                                operation,
                                "original typed add premises or reviewed ordered signature are unsupported",
                            )
                        )
                        continue
                    unknowns.append(
                        _unknown(
                            "numeric_domain",
                            operation,
                            "bounded cases leave the original input-value and overflow domain unqualified",
                        )
                    )
                if factory is None:
                    unknowns.append(
                        _unknown(
                            "semantic_family", owner, "reviewed semantics have no supported independent source factory"
                        )
                    )
                else:
                    cases.setdefault((factory, owner), set()).add(member)
                    if factory != "elementwise_add" and _SOURCE_FORMS.get(operation) != factory:
                        unknowns.append(
                            _unknown(
                                "source_operator_form",
                                operation,
                                "a generic semantic-family source does not exercise this original operator form",
                            )
                        )
        for effect in examples[member]["effect_semantics"]:
            unknowns.append(
                _unknown(
                    "effect", effect["kind"], "automatic derivation has no independently implemented effect witness"
                )
            )
    contraction = {owner for factory, owner in cases if factory == "contraction"}
    movement = {owner for factory, owner in cases if factory == "movement"}
    if arithmetic is not None:
        from .component_arithmetic_obligations import required_unknowns

        unknowns.extend(required_unknowns(arithmetic, spec=spec, contraction_owners=contraction))
    for interaction in present:
        if interaction == "shared_input_multiple_consumers" and len(contraction) == 1:
            owner = next(iter(contraction))
            cases[(interaction, owner)] = {
                member for member, relation in relations if interaction in relation.interaction_classes
            }
        elif logical_interactions and interaction in _LOGICAL_COPY_INTERACTIONS and len(movement) == 1:
            owner = next(iter(movement))
            cases[(interaction, owner)] = {
                member for member, relation in relations if interaction in relation.interaction_classes
            }
            unknowns.append(
                _unknown(
                    "physical_interaction",
                    interaction,
                    "logical copy use-def sources do not establish physical reuse, layout, lifetime or completion",
                )
            )
        else:
            unknowns.append(
                _unknown("interaction", interaction, "interaction lacks a unique reviewed compatible source factory")
            )
    if effects is not None:
        movement = {owner for factory, owner in cases if factory == "movement"}
        for member, effect in effects:
            for unknown in effect.unknowns():
                unknowns.append(_unknown("operator_effect", unknown["target"], unknown["reason"]))
            for kind in effect.effect_classes:
                eligible = {
                    link["owner"]
                    for link in review["operation_basis"]
                    if link["member"] == member
                    and any(
                        witness["target"] in link["operations"] and witness["kind"] == kind
                        for witness in effect.witnesses()
                    )
                } & movement
                if kind == "may_alias_result" and len(eligible) == 1:
                    owner = next(iter(eligible))
                    cases.setdefault((kind, owner), set()).add(member)
                    unknowns.append(
                        _unknown(
                            "physical_effect",
                            kind,
                            "logical alias sources do not establish physical ownership, layout, lifetime or completion",
                        )
                    )
                else:
                    unknowns.append(
                        _unknown(
                            "effect",
                            kind,
                            "observed source effect has no uniquely reviewed supported logical source factory",
                        )
                    )
    unknowns.append(
        _unknown(
            "effect_domain",
            "original_operator_effects",
            "source use-def relations do not independently establish operator alias/mutation effect semantics",
        )
    )
    unknowns.append(
        _unknown(
            "resource_role",
            "rtl_boundary_axis_mapping",
            "structural RTL resources do not establish a semantic allocation/axis role map",
        )
    )
    obligations = []
    for factory, owner in sorted(cases):
        # Every owner correspondence is selected from the protected review,
        # including the independent relation that motivated the generic class.
        links = [
            {
                "member": link["member"],
                "operations": [{"source": op, "owner": owner} for op in link["operations"]],
                "effects": [],
            }
            for link in owner_links[owner]
        ]
        for cohort in ("functional_guard", "withheld_transfer"):
            guard = cohort == "functional_guard"
            extents = [1, 2] if guard else [3]
            axes = {
                "M": {"kind": "extent", "values": extents},
                "K": {"kind": "extent", "values": [1] if guard else [2]},
            }
            if factory == "elementwise_add":
                axes["K"]["values"] = [1, 2] if guard else [2]
            elif factory not in {"movement", "may_alias_result"} | _LOGICAL_COPY_INTERACTIONS:
                axes["N"] = {"kind": "extent", "values": extents}
            obligations.append(
                {
                    "id": "auto_" + factory + "_" + digest(owner)[:16] + "_" + cohort,
                    "mandatory": True,
                    "cohort": cohort,
                    "operations": [owner],
                    "effects": [],
                    "expectation": "admitted_program",
                    "frontend": "mlir",
                    "base": {"op": "component_program", "kind": "model_slice", "program": _program(factory)},
                    "axes": axes,
                    "interactions": [],
                    "semantic_basis": links,
                }
            )
    if packing is not None:
        from .component_packing_sources import append_sources

        append_sources(
            facts=packing,
            spec=spec,
            movement_owners=movement,
            owner_links=owner_links,
            program=_program,
            unknown=_unknown,
            obligations=obligations,
            unknowns=unknowns,
        )
    unique_unknowns = {row["id"]: row for row in unknowns}
    if retain_historical_gaps:
        # A selected combined policy prepares additional bounded source forms;
        # it cannot erase missing owners or numeric/physical obligations from
        # its independently selected logical, typed and packing scopes. Replay
        # each scope over these same originals, never a saved missing-row list.
        if not logical_interactions or effects is None or arithmetic is None or typed_add is None or packing is None:
            raise ValueError("unified source preparation needs every original selected facet")
        for selected_typed, selected_packing in ((None, None), (typed_add, None), (None, packing)):
            _, missing = derive(
                policy,
                spec=spec,
                review=review,
                basis=basis,
                relations=relations,
                effects=effects,
                arithmetic=arithmetic,
                logical_interactions=True,
                typed_add=selected_typed,
                packing=selected_packing,
            )
            for row in missing:
                if row["id"] in unique_unknowns and unique_unknowns[row["id"]] != row:
                    raise ValueError("unified source facets disagree on an original missing obligation")
                unique_unknowns[row["id"]] = row
    declaration = {
        key: value
        for key, value in policy.items()
        if key
        not in {
            "schema",
            "operator_schema_intake_sha256",
            "arithmetic_intake_sha256",
            "packing_intake_sha256",
            "original_source_budget",
        }
    }
    declaration.update(schema=BUDGETED_PLAN_SCHEMA, effects=[], obligations=obligations)
    return declaration, sorted(unique_unknowns.values(), key=lambda row: row["id"])


def required_unknown_rows(unknowns):
    return [
        {
            "id": row["id"],
            "mandatory": True,
            "cohort": "functional_guard",
            "expectation": "admitted_program",
            "declaration_sha256": digest(row),
            "coverage": {"status": "incomplete", "reason": row["reason"], "missing_cells": None},
            "resource_boundaries": {},
            "errors": [row["reason"]],
            "state": "unavailable",
            "members": [],
        }
        for row in unknowns
    ]
