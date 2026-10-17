"""Regenerate an operand sidecar from an explicit target recipe and original sources.

The recipe selects semantic roles and source wires. Field widths, offsets, local
address flags and load-state mappings are re-derived from the selected FIRRTL.
No reserved slot is promoted to an observed field, and original facts stay exact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

from merlin.targetgen.rtl.firrtl_operand_slices import derive_operand_layouts, source_flag_offsets

from . import selected_operand_protocol as P
from .semantics import BoundSemantics, bundle_layouts


def _pin(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError("operand recipe needs canonical ordinary selected source files")
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def derive(original: BoundSemantics, *, original_facts_bytes: bytes, recipe_path: Path) -> dict:
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
    if type(recipe) is not dict or set(recipe) != required or recipe["schema"] != "merlin.operand_selection_recipe.v1":
        raise ValueError("operand recipe has missing or unsupported fields")
    facts = json.loads(original_facts_bytes)
    if facts != original._facts:
        raise ValueError("operand recipe differs from original facts")
    inputs = facts["inputs"]
    isa_path = Path(inputs["isa_path"])
    isa_pin = _pin(isa_path)
    if isa_pin["sha256"] != inputs["isa_sha256"]:
        raise ValueError("selected ISA source bytes changed")
    # Literal Scala UInt declarations only. Expressions or duplicate ownership
    # refuse rather than supplying a numerical fallback from the semantic cache.
    constants = {}
    for name, value in re.findall(r"(?m)^\s*val\s+(\w+)\s*=\s*(\d+)\.U\b", isa_path.read_text()):
        if name in constants:
            raise ValueError("selected ISA has duplicate literal ownership")
        constants[name] = int(value)
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
    blob = json.dumps(selected, sort_keys=True, allow_nan=False).encode()
    P.bind_original_operand_selection(
        original,
        original_facts_bytes=original_facts_bytes,
        selection_bytes=blob,
        expected_selection_sha256=hashlib.sha256(blob).hexdigest(),
    )
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
            }
        )
    )


if __name__ == "__main__":
    main()
