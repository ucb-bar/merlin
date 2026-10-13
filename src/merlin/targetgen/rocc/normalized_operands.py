"""Detached, optional operand protocol projections over selected source RTL facts.

Raw command identities remain owned by BoundSemantics. These projections do not
execute code, discover a provider, select a target, or qualify numerical results.
A reserved packing slot is never substituted for an unresolved actual RTL width.
"""
from __future__ import annotations

import copy

from . import semantics as S


def _keys(value, required, optional=()):
    if type(value) is not dict or set(value) - set(required) - set(optional) or not set(required) <= set(value):
        raise ValueError("operand protocol has missing or unsupported fields")


def _strict_layouts(facts, word_bits):
    layouts = copy.deepcopy(S.bundle_layouts(facts))
    for bundle in layouts.values():
        if type(bundle) is not dict or type(bundle.get("fields")) is not dict:
            raise ValueError("operand protocol requires source bundle field mappings")
        occupied = 0
        projections = {}
        for name, field in bundle["fields"].items():
            if type(name) is not str or not name.isidentifier():
                raise ValueError("operand field has no unique original source name")
            if type(field) is not dict:
                raise ValueError("operand protocol field layout is malformed")
            width, offset = field.get("width"), field.get("offset")
            reserved = field.pop("slot_width", None)
            if width is None:
                continue
            if (type(width) is not int or width <= 0 or type(offset) is not int or offset < 0
                    or offset + width > word_bits or (type(reserved) is int and width > reserved)):
                raise ValueError("actual operand field width does not fit its source register layout")
            mask = ((1 << width) - 1) << offset
            if occupied & mask:
                raise ValueError("actual operand fields overlap")
            occupied |= mask
            if "source_subfields" in field:
                children = field["source_subfields"]
                if type(children) is not list or not children or len(children) > word_bits:
                    raise ValueError("operand subfield roster is not an actual bounded source projection")
                end = offset
                for child in children:
                    _keys(child, {"field", "offset", "width"})
                    path, start, child_width = child["field"], child["offset"], child["width"]
                    if (type(path) is not str or not path or not all(part.isidentifier() for part in path.split("."))
                            or type(start) is not int or start != end or type(child_width) is not int
                            or child_width <= 0 or start + child_width > offset + width):
                        raise ValueError("operand subfields have gaps, overlaps or unsupported source paths")
                    key = name + "." + path
                    if key in projections or key in bundle["fields"]:
                        raise ValueError("operand subfield has ambiguous original ownership")
                    projections[key] = {"offset": start, "width": child_width}
                    end += child_width
                if end != offset + width:
                    raise ValueError("operand subfields do not preserve the complete original packed field")
        # These are views of the original field, not new disjoint register fields.
        bundle["fields"].update(projections)
    return layouts


class BoundOperandObserver:
    """Immutable selected normalized decode program; no ambient manifest/facts fallback."""

    def __init__(self, *, target, protocol, facts, raw_isa, word_bits):
        _keys(protocol, {"version", "encoding", "roles"}, {"address_field"})
        if type(protocol["version"]) is not int or protocol["version"] != 1:
            raise ValueError("unsupported operand protocol version")
        if type(word_bits) is not int or word_bits not in {32, 64}:
            raise ValueError("operand observer requires the selected register width")
        self._target, self._word_bits = target, word_bits
        protocol = copy.deepcopy(protocol)
        encoding, roles = protocol["encoding"], protocol["roles"]
        _keys(encoding, {"semantic_class"}, {"config_subtype"})
        _keys(roles, {"classes"}, {"config", "local_address_flags"})
        classes = S._numeric_code_map(encoding["semantic_class"], field="protocol.semantic_class", width=7)
        if set(classes) != set(raw_isa["FUNCT_CLASS"]):
            raise ValueError("normalized protocol must bind the exact selected raw instruction roster")
        if any(not label.isidentifier() or label == "UNKNOWN" for label in classes.values()):
            raise ValueError("normalized protocol requires explicit unique class labels")
        layouts = _strict_layouts(facts, word_bits)
        config, selector, selector_slot, subtypes = roles.get("config"), {}, None, {}
        if config is not None:
            _keys(config, {"class", "selector"})
            selector = config["selector"]
            _keys(selector, {"operand", "bundle", "field"})
            if selector["operand"] not in {"rs1", "rs2"} or config["class"] not in classes.values():
                raise ValueError("configuration selector does not bind a selected class and register")
            selector_slot = S.field_slot(layouts.get(selector["bundle"]), selector["field"])
            # An unresolved selector stays UNKNOWN at decode, rather than reading reserved bits.
            subtypes = S._numeric_code_map(encoding.get("config_subtype"), field="protocol.config_subtype",
                                          width=selector_slot[1] if selector_slot else word_bits)
            if set(subtypes.values()) & set(classes.values()):
                raise ValueError("configuration subtypes collide with base instruction classes")
        elif "config_subtype" in encoding:
            raise ValueError("configuration subtypes require a source-bound selector")
        known = set(classes.values()) | set(subtypes.values())
        if type(roles["classes"]) is not dict or set(roles["classes"]) - known:
            raise ValueError("operand fields name an unselected normalized class")
        for spec in roles["classes"].values():
            _keys(spec, {"fields"}, {"unresolved"})
            if type(spec["fields"]) is not list:
                raise ValueError("operand protocol fields must be a list")
            spec["unresolved"] = "null"
            names = []
            for field in spec["fields"]:
                _keys(field, set(), {"name", "names", "operand", "bundle", "field", "as", "const", "from",
                                    "equals", "flag", "set", "clear", "all"})
                if "const" in field and type(field["const"]) is not str:
                    raise ValueError("protocol constants are symbolic labels, not invented numerical fields")
                if field.get("as") in {"const", "equals"}:
                    raise ValueError("constant/comparison projections require explicit const/from declarations")
                if "all" in field:
                    _keys(field, {"name", "all"}, {"set", "clear"})
                    if (type(field["all"]) is not list or not 1 <= len(field["all"]) <= 32
                            or ("set" in field) != ("clear" in field)):
                        raise ValueError("operand conjunction requires a bounded original predicate roster")
                    for predicate in field["all"]:
                        _keys(predicate, {"from", "equals"})
                        if predicate["from"] not in names or type(predicate["equals"]) not in {bool, int, str}:
                            raise ValueError("operand conjunction must reference previously decoded scalar fields")
                    if "set" in field and any(type(field[key]) is not str or not field[key]
                                              for key in ("set", "clear")):
                        raise ValueError("operand conjunction choices must be explicit symbolic labels")
                if "names" in field:
                    _keys(field, {"names", "operand", "bundle"})
                    if field["operand"] not in {"rs1", "rs2"}:
                        raise ValueError("operand field projection has no selected register")
                    if type(field["names"]) is not list or not all(type(n) is str for n in field["names"]):
                        raise ValueError("operand field names must be explicit strings")
                    names.extend(field["names"])
                else:
                    names.append(field.get("name"))
            if any(type(n) is not str or not n.isidentifier() for n in names) or len(names) != len(set(names)):
                raise ValueError("operand protocol field names are missing or duplicated")
        address_width = None
        if "address_field" in protocol:
            address = protocol["address_field"]
            _keys(address, {"bundle", "field"})
            slot = S.field_slot(layouts.get(address["bundle"]), address["field"])
            address_width = slot[1] if slot else None
        flag_specs = roles.get("local_address_flags", {})
        if type(flag_specs) is not dict:
            raise ValueError("local address flag roles must be a mapping")
        for spec in flag_specs.values():
            _keys(spec, {"msb_offset"})
            if type(spec["msb_offset"]) is not int or not 0 <= spec["msb_offset"] < word_bits:
                raise ValueError("local address flag offset is not bounded")
        if flag_specs and "address_field" not in protocol:
            raise ValueError("local address flags require a source-bound address field")
        flags = S.local_address_flags(roles, address_width) if address_width else {name: 0 for name in flag_specs}
        programs = S._compile_classes(roles, layouts, flags)
        if address_width is None:
            for program in programs.values():
                for step in program["fields"]:
                    if step["as"] in {"flag", "flag_choice"}:
                        step.update(offset=None, width=None)
        self._isa = {
            "CUSTOM_OPCODE": raw_isa["CUSTOM_OPCODE"], "FUNCT3": raw_isa["FUNCT3"], "FUNCT_CLASS": classes,
            "CONFIG_CLASS": config["class"] if config else None,
            "CONFIG_SELECTOR": {"operand": selector["operand"], "offset": selector_slot[0],
                                "width": selector_slot[1]} if selector_slot else None,
            "CONFIG_SUBTYPE": subtypes, "OPERAND_FIELDS": programs,
        }

    @property
    def word_bits(self):
        return self._word_bits

    def isa_constants(self, target):
        if target != self._target:
            raise ValueError("operand observer belongs to a different selected target")
        return copy.deepcopy(self._isa)

    def decode_instruction(self, funct, rs1, rs2, isa):
        if isa != self._isa:
            raise ValueError("operand observer requires its own selected decode program")
        observations = []
        for operand in (rs1, rs2):
            if type(operand) is not dict:
                raise ValueError("operand observation must retain its original identity")
            observation = copy.deepcopy(operand)
            raw = observation.get("raw")
            if raw is not None:
                if type(raw) is not int or not -(1 << (self.word_bits - 1)) <= raw < (1 << self.word_bits):
                    raise ValueError("observed register value exceeds selected word width")
                if raw < 0:
                    observation["raw"] = raw + (1 << self.word_bits)
            observations.append(observation)
        return S.decode_instruction(funct, *observations, self._isa)
