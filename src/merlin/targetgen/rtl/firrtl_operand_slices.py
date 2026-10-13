"""Source-bound operand fields from explicit elaborated FIRRTL wires.

The trusted target declaration selects wires and raw register roots. Only unique
unguarded aliases and bit extractions establish a layout. This observes packing,
not command acceptance, instruction roles, address legality or sequencer state.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from . import firrtl_struct as F

SCHEMA = "merlin.firrtl_operand_layouts.v1"


def _source(path, expected):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path or path.is_symlink()
            or not path.is_file() or type(expected) is not str or len(expected) != 64):
        raise ValueError("operand layout needs an ordinary selected FIRRTL source and byte pin")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError("selected operand FIRRTL bytes changed")
    return path


def _type(module, reference, modules=None):
    declaration = module.decls.get(reference.root)
    if declaration is not None and declaration[0] == "inst" and not reference.dynamic:
        callee = (modules or {}).get(declaration[1])
        if callee is None or not reference.segments or type(reference.segments[0]) is not str:
            raise ValueError("operand instance root has no original selected port type")
        return F.field_type(callee.port_type(reference.segments[0]), reference.segments[1:])
    if (declaration is None or declaration[0] not in {"wire", "input", "output"}
            or reference.dynamic):
        raise ValueError("operand source is not an explicit ground wire or interface")
    return F.field_type(F.parse_type(declaration[1]), reference.segments)


def _leaves(typ, path=()):
    if typ is None:
        raise ValueError("operand field has no elaborated type")
    if typ[0] == "uint" and type(typ[1]) is int and typ[1] >= 0:
        return [] if typ[1] == 0 else [(path, typ[1])]
    if typ[0] == "bundle":
        names = [name for name, _flip, _child in typ[1]]
        if len(names) != len(set(names)) or any(flip for _name, flip, _child in typ[1]):
            raise ValueError("operand bundle has ambiguous or flipped fields")
        return [leaf for name, _flip, child in typ[1] for leaf in _leaves(child, (*path, name))]
    raise ValueError("operand field has unresolved, signed or unsupported elaborated type")


def _slice(module, expression, source, word_bits, *, active=None):
    active = set() if active is None else active
    if expression == source:
        return 0, word_bits, ()
    if len(active) >= 128 or repr(expression) in active:
        raise ValueError("operand slice has a cyclic or oversized source cone")
    active = {*active, repr(expression)}
    if isinstance(expression, F.Ref):
        if expression.dynamic:
            raise ValueError("operand source uses a dynamic index")
        if expression.root in module.nodes and not expression.segments:
            rhs, line, _locator = module.nodes[expression.root]
        else:
            declaration = module.decls.get(expression.root)
            if declaration is None or declaration[0] != "wire":
                raise ValueError("operand source reaches state or an unrelated interface")
            drivers = [c for c in module.connects if c.lhs == expression]
            if len(drivers) != 1 or drivers[0].guards:
                raise ValueError("operand source has missing, multiple or conditional drivers")
            rhs, line = drivers[0].rhs, drivers[0].line
        offset, width, lines = _slice(module, rhs, source, word_bits, active=active)
        if expression.root not in module.nodes and F.type_width(_type(module, expression)) != width:
            raise ValueError("operand alias changes its actual declared width")
        return offset, width, (*lines, line)
    if isinstance(expression, F.Prim):
        if expression.op == "bits" and len(expression.args) == 3:
            value, high, low = expression.args
            if not isinstance(high, F.IntArg) or not isinstance(low, F.IntArg):
                raise ValueError("operand extraction has unresolved bounds")
            offset, width, lines = _slice(module, value, source, word_bits, active=active)
            if not 0 <= low.value <= high.value < width:
                raise ValueError("operand extraction exceeds its actual source width")
            return offset + low.value, high.value - low.value + 1, lines
        if expression.op == "asUInt" and len(expression.args) == 1:
            return _slice(module, expression.args[0], source, word_bits, active=active)
    raise ValueError("operand source is not an exact alias or bit extraction")


def _field(module, row, word_bits, modules):
    source, field = F.parse_expr(row["input"]), F.parse_expr(row["field"])
    if not isinstance(source, F.Ref) or not isinstance(field, F.Ref):
        raise ValueError("operand field selection requires explicit reference paths")
    if F.type_width(_type(module, source, modules)) != word_bits:
        raise ValueError("operand register root has another actual width")
    typ = _type(module, field)
    leaves = _leaves(typ)
    if not leaves or len(leaves) > word_bits:
        raise ValueError("operand field is empty or exceeds its register budget")
    projections, lines = [], []
    for tail, declared_width in leaves:
        reference = F.Ref(field.root, (*field.segments, *tail))
        offset, width, evidence = _slice(module, reference, source, word_bits)
        if width != declared_width:
            raise ValueError("operand extraction differs from its actual declared field width")
        projections.append({"field": ".".join(map(str, tail)), "offset": offset, "width": width})
        lines.extend(evidence)
    projections.sort(key=lambda item: item["offset"])
    offset, end = projections[0]["offset"], projections[0]["offset"]
    for projection in projections:
        if projection["offset"] != end:
            raise ValueError("operand aggregate has gaps or overlapping source bits")
        end += projection["width"]
    if end - offset != F.type_width(typ):
        raise ValueError("operand aggregate does not preserve every actual field bit")
    return {"status": "derived", "offset": offset, "width": end - offset,
            "subfields": projections, "source_lines": sorted(set(lines))}


def derive_operand_layouts(firrtl, *, expected_sha256, selections, word_bits=64):
    """Export actual packed field widths; unknown relationships remain unavailable.

    Each explicit selection binds a destination ``bundle``/``name`` to one
    ``module``/``field`` and its original ``input`` register. No reserved packing
    slot or numerical output participates in the observation.
    """
    if (word_bits not in {32, 64} or type(selections) is not list
            or not selections or len(selections) > 256):
        raise ValueError("operand layout selection exceeds its bounded roster")
    selections = copy.deepcopy(selections)
    identities = set()
    for row in selections:
        if (type(row) is not dict or set(row) != {"bundle", "name", "module", "field", "input"}
                or any(type(value) is not str or not value for value in row.values())
                or not row["bundle"].isidentifier() or not row["name"].isidentifier()
                or (row["bundle"], row["name"]) in identities):
            raise ValueError("operand layout selection has malformed or duplicate fields")
        identities.add((row["bundle"], row["name"]))
    path = _source(firrtl, expected_sha256)
    names = {row["module"] for row in selections}
    bodies = F.read_module_bodies(path, sorted(names))
    preliminary = {name: F.parse_module(name, lines) for name, lines in bodies.items()}
    for row in selections:
        reference = F.parse_expr(row["input"])
        module = preliminary.get(row["module"])
        declaration = module.decls.get(reference.root) if module is not None and isinstance(reference, F.Ref) else None
        if declaration is not None and declaration[0] == "inst":
            names.add(declaration[1])
    if names != set(bodies):
        bodies = F.read_module_bodies(path, sorted(names))
    for lines in bodies.values():
        seen = set()
        for _number, raw in lines[1:]:
            words = F.split_locator(raw)[0].strip().split(None, 1)
            if words and words[0] in {"node", "wire", "input", "output", "reg", "regreset"}:
                name = words[1].split("=", 1)[0].split(":", 1)[0].strip()
                if name in seen:
                    raise ValueError("operand module has duplicate source definitions")
                seen.add(name)
    modules = {name: F.parse_module(name, lines) for name, lines in bodies.items()}
    fields = []
    for row in selections:
        module = modules.get(row["module"])
        if module is None:
            result = {"status": "unavailable", "reason": "selected elaborated module is absent"}
        else:
            try:
                result = _field(module, row, word_bits, modules)
            except (ValueError, F.FirrtlParseError) as exc:
                result = {"status": "unavailable", "reason": str(exc)}
        fields.append({**row, **result})
    _source(path, expected_sha256)
    return {"schema": SCHEMA, "source": str(path), "source_sha256": expected_sha256,
            "word_bits": word_bits, "selections": selections, "fields": fields,
            "scope": "original elaborated combinational field packing only",
            "command_acceptance_and_semantic_roles": "unproven"}


def apply_selected_layouts(facts, selection):
    """Re-derive a detached observation against the facts' original FIRRTL pin.

    Only existing unresolved fields can gain actual widths. Unknown observations
    remain unresolved; another original width/offset is a refusal, never overwritten.
    """
    if type(selection) is not dict or selection.get("schema") != SCHEMA:
        raise ValueError("operand layout record is not an original selection")
    declared = (facts.get("inputs") or {}).get("firrtl_inputs") or []
    if not any(row.get("path") == selection.get("source")
               and row.get("sha256") == selection.get("source_sha256") for row in declared):
        raise ValueError("operand layout belongs to another selected FIRRTL source")
    observed = derive_operand_layouts(
        selection["source"], expected_sha256=selection["source_sha256"],
        selections=selection["selections"], word_bits=selection["word_bits"],
    )
    if observed != selection:
        raise ValueError("operand layout record differs from its actual source observation")
    result = copy.deepcopy(facts)
    interfaces = result.get("facts", {}).get("interfaces", [])
    rosters = [item for item in interfaces if item.get("name") == "register_bundle_layouts"]
    if len(rosters) != 1:
        raise ValueError("operand facts have no unique original bundle roster")
    bundles = rosters[0].get("bundles") or {}
    for row in observed["fields"]:
        if row["status"] != "derived":
            continue
        bundle = bundles.get(row["bundle"])
        field = (bundle.get("fields") or {}).get(row["name"]) if isinstance(bundle, dict) else None
        if not isinstance(field, dict) or bundle.get("width") != observed["word_bits"]:
            raise ValueError("operand field is not part of the original selected bundle")
        if field.get("offset") != row["offset"] or field.get("width") not in {None, row["width"]}:
            raise ValueError("actual operand layout contradicts the original source declaration")
        field["width"] = row["width"]
    result["inputs"]["operand_layout_selection"] = observed
    return result


def source_flag_offsets(selection, *, bundle, name, flags):
    """Project explicitly selected one-bit source fields into address-relative flags.

    ``flags`` maps caller-selected role names to actual original subfield paths.
    It does not establish those roles' effects or authorize a protocol implicitly.
    The detached target protocol must still select and bind every meaning.
    """
    observed = derive_operand_layouts(
        selection["source"], expected_sha256=selection["source_sha256"],
        selections=selection["selections"], word_bits=selection["word_bits"],
    )
    if observed != selection or type(flags) is not dict or not flags:
        raise ValueError("operand flags require an exact original source observation")
    rows = [row for row in observed["fields"] if row["bundle"] == bundle and row["name"] == name]
    if len(rows) != 1 or rows[0]["status"] != "derived":
        raise ValueError("operand flags have no complete selected address field")
    row = rows[0]
    subfields = {field["field"]: field for field in row["subfields"]}
    result = {}
    for role, source in flags.items():
        field = subfields.get(source)
        if type(role) is not str or not role.isidentifier() or field is None or field["width"] != 1:
            raise ValueError("operand flag is not an original selected one-bit source field")
        result[role] = {"msb_offset": row["offset"] + row["width"] - field["offset"] - 1}
    return result
