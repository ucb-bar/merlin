"""Regenerate an operand sidecar from an explicit target recipe and original sources.

The recipe selects semantic roles and source wires. Field widths, offsets, local
address flags and load-state mappings are re-derived from the selected FIRRTL.
No reserved slot is promoted to an observed field, and original facts stay exact.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from merlin.targetgen.rtl.firrtl_operand_slices import derive_operand_layouts, source_flag_offsets
from merlin.targetgen.rtl.source_relocations import relocate_operand_sources

from . import selected_operand_protocol as P
from .semantics import BoundSemantics, bundle_layouts


def _pin(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError("operand recipe needs canonical ordinary selected source files")
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def _bound_bundles(spec):
    pairs, projected, names = [], set(), set()
    for row in spec.get("fields", ()):
        if "operand" in row and "bundle" in row:
            pair = (row["operand"], row["bundle"])
            if pair not in pairs:
                pairs.append(pair)
            for name in row.get("names", ()):
                projected.add((*pair, name))
            if "field" in row:
                projected.add((*pair, row["field"]))
        names.update(row.get("names", ()))
        if "name" in row:
            names.add(row["name"])
    return pairs, projected, names


def _project_layout_fields(classes, observed, class_bundles):
    """Project every derived field of each bundle a class reads, under its own RTL name.

    A class that reads one operand register through a bundle observes the whole
    bundle; leaving its other derived fields out would report a declared subset
    as the full roster. Interpretation stays the plain integer slice.
    """
    derived = {}
    for row in observed["fields"]:
        if row.get("status") == "derived":
            derived.setdefault(row["bundle"], []).append(row["name"])
    if type(class_bundles) is not dict:
        raise ValueError("operand recipe class bundles must be a mapping")
    for klass, operands in class_bundles.items():
        if klass not in classes or type(operands) is not dict:
            raise ValueError("operand recipe binds a bundle to an undeclared class")
        for operand, bundle in operands.items():
            if operand not in ("rs1", "rs2") or bundle not in derived:
                raise ValueError("operand recipe class bundle is not a derived selected layout")
    for klass, spec in classes.items():
        pairs, projected, names = _bound_bundles(spec)
        pairs += [pair for pair in (class_bundles.get(klass) or {}).items() if pair not in pairs]
        shared = {bundle for _, bundle in pairs if sum(other == bundle for _, other in pairs) > 1}
        for operand, bundle in pairs:
            for field in derived.get(bundle, ()):
                if (operand, bundle, field) in projected:
                    continue
                name = field if bundle not in shared and field not in names else operand + "_" + field
                if name in names:
                    raise ValueError("operand recipe cannot name a derived layout field uniquely")
                names.add(name)
                projected.add((operand, bundle, field))
                spec["fields"].append({"name": name, "operand": operand, "bundle": bundle, "field": field})


#: Field kinds a recipe may declare. They mirror the decoded-field kinds the
#: machine evidence consumer accepts; anything else is refused, not mapped.
FIELD_KINDS = ("stride", "extent", "scale", "activation", "output_dtype", "address")
_CONSUMER_STATE = ("reg", "regreset", "output")


def _field_kinds(classes, selections, firrtl, entries):
    """Verify each declared kind's RTL consumer in the selected FIRRTL; return kinds and citations.

    An entry names a decoded ``CLASS.field`` label, its kind and the consuming
    state element (``module``/``sink``). The field must be a bundle slice the
    protocol projects; the selected FIRRTL must connect exactly that bundle
    field wire into the named register or output port of the module the wire
    lives in. The kind itself is the reviewed reading of that consumer; a field
    with no verified consumer is never declared and its kind stays unknown.
    """
    from merlin.targetgen.rtl import firrtl_struct as F

    if type(entries) is not list or len(entries) > 256:
        raise ValueError("operand recipe field kinds must be a bounded list")
    sources = {(row["bundle"], row["name"]): row for row in selections}
    modules, kinds, citations = {}, {}, []
    for entry in entries:
        if (
            type(entry) is not dict
            or set(entry) != {"label", "kind", "consumer"}
            or type(entry["consumer"]) is not dict
            or set(entry["consumer"]) != {"module", "sink"}
            or entry["kind"] not in FIELD_KINDS
            or type(entry["label"]) is not str
            or entry["label"] in kinds
        ):
            raise ValueError("operand recipe field kind entry is malformed or duplicated")
        klass, dot, name = entry["label"].partition(".")
        rows = [row for row in (classes.get(klass) or {}).get("fields", ()) if row.get("name") == name]
        if not dot or len(rows) != 1 or "bundle" not in rows[0] or "field" not in rows[0]:
            raise ValueError("operand recipe field kind names no projected bundle field")
        source = sources.get((rows[0]["bundle"], rows[0]["field"]))
        consumer = entry["consumer"]
        if source is None or source["module"] != consumer["module"]:
            raise ValueError("operand recipe field kind consumer is outside the field's selected module")
        if consumer["module"] not in modules:
            path = Path(firrtl["path"])
            if _pin(path)["sha256"] != firrtl["sha256"]:
                raise ValueError("selected FIRRTL bytes changed")
            bodies = F.read_module_bodies(path, [consumer["module"]])
            if consumer["module"] not in bodies:
                raise ValueError("operand recipe field kind consumer module is absent")
            modules[consumer["module"]] = F.parse_module(consumer["module"], bodies[consumer["module"]])
        module = modules[consumer["module"]]
        sink, wire = F.parse_expr(consumer["sink"]), F.parse_expr(source["field"])
        declaration = module.decls.get(getattr(sink, "root", None))
        if not isinstance(sink, F.Ref) or declaration is None or declaration[0] not in _CONSUMER_STATE:
            raise ValueError("operand recipe field kind consumer is not a register or output port")
        lines = sorted(c.line for c in module.connects if c.lhs == sink and c.rhs == wire)
        if not lines:
            raise ValueError("selected FIRRTL does not connect the field into its declared consumer")
        kinds[entry["label"]] = entry["kind"]
        citations.append(
            dict(
                label=entry["label"],
                kind=entry["kind"],
                bundle=rows[0]["bundle"],
                field=rows[0]["field"],
                module=consumer["module"],
                source=source["field"],
                sink=consumer["sink"],
                sink_declaration=dict(keyword=declaration[0], line=declaration[2]),
                firrtl_lines=lines,
            )
        )
    return dict(sorted(kinds.items())), citations


def layout_field_roster(selected):
    """Derived fields of selected layouts that no protocol class projects."""
    classes = selected["protocol"]["roles"]["classes"]
    projected = set()
    for spec in classes.values():
        projected |= {(bundle, field) for _, bundle, field in _bound_bundles(spec)[1]}
    rows = [row for row in selected["layouts"]["fields"] if row.get("status") == "derived"]
    missing = [
        dict(bundle=row["bundle"], field=row["name"]) for row in rows if (row["bundle"], row["name"]) not in projected
    ]
    return dict(
        status="complete" if not missing else "declared_subset", derived_fields=len(rows), unprojected_fields=missing
    )


def derive(
    original: BoundSemantics, *, original_facts_bytes: bytes, recipe_path: Path, source_relocations=None
) -> dict:
    """Derive and verify a detached sidecar; selecting it for a release is separate."""
    if type(original) is not BoundSemantics or original.operand_observer is not None:
        raise ValueError("operand recipe requires unchanged original bound semantics")
    recipe_path = Path(recipe_path)
    recipe_pin = _pin(recipe_path)
    recipe = yaml.safe_load(recipe_path.read_text())
    required = {
        "schema",
        "semantic_declaration",
        "aliases",
        "config_subtype_sources",
        "isa_sibling_sources",
        "bundles",
        "fields",
        "address",
        "load_state_selection",
    }
    # Optional: reviewed class-to-bundle bindings for consumers whose declared
    # roles bind no RTL bundle yet. Each binding cites the consuming RTL.
    # field_kinds: reviewed kinds, each verified against its FIRRTL consumer.
    optional = {"class_operand_bundles", "field_kinds"}
    if (
        type(recipe) is not dict
        or not required <= set(recipe)
        or set(recipe) - required - optional
        or recipe["schema"] != "merlin.operand_selection_recipe.v1"
    ):
        raise ValueError("operand recipe has missing or unsupported fields")
    facts = json.loads(original_facts_bytes)
    if facts != original._facts:
        raise ValueError("operand recipe differs from original facts")
    original_facts = copy.deepcopy(facts)
    relocation_snapshot = copy.deepcopy(source_relocations)
    facts = relocate_operand_sources(facts, source_relocations)
    inputs = facts["inputs"]
    isa_path = Path(inputs["isa_path"])
    isa_pin = _pin(isa_path)
    if isa_pin["sha256"] != inputs["isa_sha256"]:
        raise ValueError("selected ISA source bytes changed")
    # Literal Scala UInt declarations only. Expressions or duplicate ownership
    # refuse rather than supplying a numerical fallback from the semantic cache.
    from merlin.model_check.binding_sources import scala_tokens

    constants, declared = {}, set()
    tokens = scala_tokens(isa_path.read_text())
    for index, token in enumerate(tokens):
        if token.text != "val" or index and tokens[index - 1].line == token.line:
            continue
        declaration = tokens[index:index + 6]
        if (len(declaration) != 6 or declaration[1].kind != "ident"
                or [item.text for item in declaration[2:3]] != ["="]
                or declaration[3].kind != "number" or not declaration[3].text.isdecimal()
                or [item.text for item in declaration[4:]] != [".", "U"]):
            continue
        name, value = declaration[1].text, int(declaration[3].text)
        if name in declared:
            raise ValueError("selected ISA has duplicate literal ownership")
        declared.add(name)
        if (index + 6 < len(tokens) and tokens[index + 6].line == declaration[-1].line
                and tokens[index + 6].text not in (";", "}")):
            continue  # A required nonliteral field will fail the exact roster check below.
        constants[name] = value
    raw_classes = original.isa_constants(original._target)["FUNCT_CLASS"]
    if any(constants.get(name) != code for code, name in raw_classes.items()):
        raise ValueError("original instruction roster has no exact selected ISA literals")
    aliases = recipe["aliases"]
    if type(aliases) is not dict or set(aliases) - set(raw_classes.values()):
        raise ValueError("operand recipe aliases an unselected original instruction")
    declaration_path = (recipe_path.parent / recipe["semantic_declaration"]).resolve()
    declaration_pin = _pin(declaration_path)
    declaration = yaml.safe_load(declaration_path.read_text())
    declared_roles = declaration["rocc_operand_roles"]
    subtypes = {}
    for source, label in recipe["config_subtype_sources"].items():
        code = constants.get(source)
        if type(code) is not int or declaration["encoding"]["config_subtype"].get(code) != label:
            raise ValueError("configuration role differs from selected ISA and semantic declaration")
        if code in subtypes:
            raise ValueError("configuration roles have duplicate source ownership")
        subtypes[code] = label
    protocol = {
        "version": 1,
        "encoding": {
            "semantic_class": {code: aliases.get(name, name) for code, name in raw_classes.items()},
            "config_subtype": subtypes,
        },
        "roles": {
            "classes": copy.deepcopy(declared_roles["classes"]),
            "config": {key: copy.deepcopy(declared_roles["config"][key]) for key in ("class", "selector")},
        },
        "address_field": {"bundle": recipe["address"]["bundle"], "field": recipe["address"]["name"]},
    }
    # A load-state field is selected from the original declared bundle, not
    # synthesized from the load opcode's nominal class name.
    if recipe["load_state_selection"] is not None:
        config_class = recipe["load_state_selection"]["config_class"]
        protocol["roles"]["classes"][config_class]["fields"].append(
            {
                "name": "state_id",
                "operand": "rs1",
                "bundle": recipe["load_state_selection"]["config_field"]["bundle"],
                "field": recipe["load_state_selection"]["config_field"]["name"],
            }
        )
    selections = copy.deepcopy(recipe["fields"])
    layouts = bundle_layouts(facts)
    for bundle, source in recipe["bundles"].items():
        for name in layouts[bundle]["fields"]:
            selections.append(
                {
                    "bundle": bundle,
                    "name": name,
                    "module": source["module"],
                    "field": source["field"] + "." + name,
                    "input": source["input"],
                    **({"packing_context": source["packing_context"]} if "packing_context" in source else {}),
                }
            )
    firrtls = inputs["firrtl_inputs"]
    if len(firrtls) != 1:
        raise ValueError("operand recipe requires one explicitly selected original FIRRTL")
    observed = derive_operand_layouts(
        firrtls[0]["path"], expected_sha256=firrtls[0]["sha256"], selections=selections, word_bits=original._width
    )
    address = recipe["address"]
    flags = source_flag_offsets(observed, bundle=address["bundle"], name=address["name"], flags=address["flags"])
    # Global address roles are valid only when all selected address operands
    # independently show exactly the same packing and one-bit flags.
    for other in address["agree"]:
        other_flags = source_flag_offsets(observed, bundle=other["bundle"], name=other["name"], flags=address["flags"])
        if other_flags != flags:
            raise ValueError("selected local address flag roles disagree across operands")
    protocol["roles"]["local_address_flags"] = flags
    _project_layout_fields(protocol["roles"]["classes"], observed, recipe.get("class_operand_bundles") or {})
    kinds, kind_sources = _field_kinds(
        protocol["roles"]["classes"], selections, firrtls[0], recipe.get("field_kinds") or []
    )
    pins = [recipe_pin, declaration_pin, isa_pin]
    for source in recipe["isa_sibling_sources"]:
        if type(source) is not str or Path(source).name != source:
            raise ValueError("operand semantic source must be an explicitly named ISA sibling")
        pins.append(_pin(isa_path.parent / source))
    selected = {
        "schema": P.SCHEMA,
        "target": original._target,
        "original_facts_sha256": hashlib.sha256(original_facts_bytes).hexdigest(),
        "source_pins": pins,
        "layouts": observed,
        "protocol": protocol,
    }
    if recipe["load_state_selection"] is not None:
        selected["load_state_selection"] = copy.deepcopy(recipe["load_state_selection"])
    if source_relocations is not None:
        selected["source_relocations"] = copy.deepcopy(source_relocations)
    if kinds:
        selected["operand_field_kinds"] = kinds
        selected["operand_field_kind_sources"] = kind_sources
    selected["operand_layout_roster"] = layout_field_roster(selected)
    blob = json.dumps(selected, sort_keys=True, allow_nan=False).encode()
    P.bind_original_operand_selection(
        original,
        original_facts_bytes=original_facts_bytes,
        selection_bytes=blob,
        expected_selection_sha256=hashlib.sha256(blob).hexdigest(),
    )
    if (
        source_relocations != relocation_snapshot
        or relocate_operand_sources(original_facts, source_relocations) != facts
        or original._facts != original_facts
    ):
        raise ValueError("operand recipe original sources changed before publication")
    return selected


def main():
    import argparse

    from .semantics import bind

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blob = args.facts.read_bytes()
    facts = json.loads(blob)
    original = bind(target=facts["inputs"]["target"], contract=yaml.safe_load(args.contract.read_text()), facts=facts)
    selected = derive(original, original_facts_bytes=blob, recipe_path=args.recipe.resolve())
    # Never replace a previously selected or sealed sidecar.
    with args.output.open("x") as stream:
        stream.write(json.dumps(selected, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": _pin(args.output.resolve())["sha256"],
                "derived_fields": sum(row["status"] == "derived" for row in selected["layouts"]["fields"]),
                "unavailable_fields": sum(row["status"] != "derived" for row in selected["layouts"]["fields"]),
                "field_roster": layout_field_roster(selected),
            }
        )
    )


if __name__ == "__main__":
    main()
