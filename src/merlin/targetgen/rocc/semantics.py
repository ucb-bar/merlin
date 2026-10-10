"""Decode explicit operand bindings using the selected RTL register layouts.

Contract labels assign software roles; facts supply instruction codes and field
positions. Neither implies an implementation, schedule or hardware qualification.
"""

from __future__ import annotations

import copy


def _interface(facts, name):
    body = facts.get("facts", {})
    interfaces = body.get("interfaces", []) if type(body) is dict else []
    if type(interfaces) is not list:
        raise ValueError("RTL interfaces must be a list")
    matches = [item for item in interfaces if type(item) is dict and item.get("name") == name]
    if len(matches) != 1:
        raise ValueError("selected RTL interface is missing or ambiguous")
    return matches[0]


def _uint(value, width):
    return type(value) is int and 0 <= value < (1 << width)


class BoundSemantics:
    """A detached data selection, with no provider imports or file generation."""

    def __init__(self, *, target, contract, facts):
        if type(target) is not str or not target or type(contract) is not dict or contract.get("name") != target:
            raise ValueError("semantics require the selected named contract")
        if type(facts) is not dict:
            raise ValueError("semantics require selected RTL facts")
        self._target = target
        self._facts = copy.deepcopy(facts)
        table = _interface(facts, "funct_decode_table")
        codes = table.get("legal_funct")
        if type(codes) is not list or not codes or any(not _uint(code, 7) for code in codes):
            raise ValueError("RoCC instruction codes must be explicit funct7 values")
        if len(set(codes)) != len(codes):
            raise ValueError("RoCC instruction codes must be unique")
        if not _uint(table.get("custom_opcode"), 7) or not _uint(table.get("funct3"), 3):
            raise ValueError("RoCC transport encoding must be explicit in facts")
        roles = contract.get("rocc_operand_roles")
        if type(roles) is not dict or set(roles) != {"version", "word_bits", "instructions"}:
            raise ValueError("RoCC operand roles require a complete versioned declaration")
        if (
            type(roles["version"]) is not int
            or roles["version"] != 1
            or type(roles["word_bits"]) is not int
            or roles["word_bits"] not in (32, 64)
        ):
            raise ValueError("unsupported operand declaration version or register width")
        self._width = roles["word_bits"]
        instructions = roles["instructions"]
        if type(instructions) is not list or not instructions or len(instructions) > 128:
            raise ValueError("operand roles require a bounded instruction roster")
        self._instructions = {}
        layouts = None
        for instruction in instructions:
            if type(instruction) is not dict or set(instruction) != {"funct", "class", "operands"}:
                raise ValueError("malformed instruction role binding")
            code, label, operands = instruction["funct"], instruction["class"], instruction["operands"]
            if not _uint(code, 7) or code not in codes or code in self._instructions:
                raise ValueError("instruction role lacks a unique RTL code")
            if type(label) is not str or not label.isidentifier() or label in {"UNKNOWN", "FENCE"}:
                raise ValueError("instruction role requires a distinct semantic label")
            if type(operands) is not dict or set(operands) - {"rs1", "rs2"}:
                raise ValueError("instruction operands must bind original register slots")
            fields = {}
            for slot, binding in operands.items():
                if type(binding) is not dict or set(binding) != {"bundle"} or type(binding["bundle"]) is not str:
                    raise ValueError("operand must select one extracted register bundle")
                if layouts is None:
                    layouts = _interface(facts, "register_bundle_layouts").get("bundles")
                bundle = layouts.get(binding["bundle"]) if type(layouts) is dict else None
                if (
                    type(bundle) is not dict
                    or type(bundle.get("width")) is not int
                    or bundle.get("width") != self._width
                ):
                    raise ValueError("selected operand bundle has an unknown or mismatched width")
                roster = bundle.get("fields")
                if type(roster) is not dict or not roster or len(roster) > self._width:
                    raise ValueError("operand bundle requires explicit bounded fields")
                occupied = 0
                fields[slot] = {}
                for name, field in roster.items():
                    if type(name) is not str or not name.isidentifier() or type(field) is not dict:
                        raise ValueError("malformed operand field")
                    offset, width = field.get("offset"), field.get("width")
                    if type(offset) is not int or type(width) is not int or offset < 0 or width <= 0:
                        raise ValueError("operand field layout must be exactly resolved")
                    if offset + width > self._width:
                        raise ValueError("operand field exceeds the selected register width")
                    mask = ((1 << width) - 1) << offset
                    if occupied & mask:
                        raise ValueError("operand fields overlap")
                    occupied |= mask
                    fields[slot][name] = (offset, width)
            self._instructions[code] = (label, fields)
        self._isa = {
            "CUSTOM_OPCODE": table["custom_opcode"],
            "FUNCT3": table["funct3"],
            "FUNCT_CLASS": {code: value[0] for code, value in self._instructions.items()},
        }
        self.legal_codes = frozenset(codes)
        self.complete_decode = table.get("scope") == "complete_rocc_funct7" and table.get("complete_isa") is True
        from merlin.targetgen.rtl_checks_generic import BoundChecks

        self.rtl_checks = BoundChecks(self, contract.get("rtl_checks"))

    def isa_constants(self, target):
        if target != self._target:
            raise ValueError("semantics selection belongs to a different target")
        return copy.deepcopy(self._isa)

    def _selected(self, isa):
        if isa != self._isa:
            raise ValueError("instruction decode must use its selected ISA data")

    def decode_instruction(self, funct, rs1, rs2, isa):
        self._selected(isa)
        instruction = self._instructions.get(funct) if _uint(funct, 7) else None
        if instruction is None:
            return "UNKNOWN", {}
        label, fields = instruction
        decoded = {}
        for slot, operand in (("rs1", rs1), ("rs2", rs2)):
            if type(operand) is not dict:
                raise ValueError("operand observation must retain its register slot")
            raw = operand.get("raw")
            if type(raw) is int and -(1 << (self._width - 1)) <= raw < 0:
                raw += 1 << self._width
            if raw is not None and not _uint(raw, self._width):
                raise ValueError("operand observation exceeds the selected register width")
            for name, (offset, width) in fields.get(slot, {}).items():
                decoded[f"{slot}.{name}"] = None if raw is None else (raw >> offset) & ((1 << width) - 1)
        return label, decoded

    def instruction_funct(self, name, rs1, isa):
        self._selected(isa)
        matches = [code for code, (label, _) in self._instructions.items() if label == name]
        if len(matches) != 1:
            raise ValueError("instruction label has no unique selected encoding")
        if not (_uint(rs1, self._width) or type(rs1) is int and -(1 << (self._width - 1)) <= rs1 < 0):
            raise ValueError("assembly operand exceeds the selected register width")
        return matches[0]


def bind(*, target, contract, facts):
    """Bind one selected contract/facts snapshot without native execution."""
    return BoundSemantics(target=target, contract=contract, facts=facts)
