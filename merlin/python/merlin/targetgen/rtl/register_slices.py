"""Conservative input-slice recovery from conditional elaborated register updates.

Only combinational mux/bitwise/extract logic is interpreted. Unknown controls
explore both branches; an unknown data branch prevents a field claim. This is
not a sequencer or arithmetic proof. Input and register identities are supplied
by the trusted target adapter, never by the compiler under examination.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product

from merlin.perf.hw_counters import _module_lines, _operand_refs


def register_definitions(hw_text: str, module: str) -> dict[str, str]:
    lines, error = _module_lines(hw_text, module)
    if lines is None:
        raise ValueError(error)
    definitions = {}
    for line in lines:
        line = line.strip()
        if line.startswith("%") and " = " in line:
            ref, body = line.split(" = ", 1)
            if ref in definitions:
                raise ValueError("duplicate hardware SSA definition")
            definitions[ref] = body
    return definitions


def derive_register_slices(hw_text: str, *, module: str, registers: list[str],
                           selector: str, selector_value: int,
                           inputs: dict[str, tuple[str, int]]) -> dict:
    """Recover one input slice/constant for each possible non-hold update.

    Missing or unsupported data is UNKNOWN, including cross-register updates.
    A selector with only hold paths produces ``no_non_hold_update``. Register
    widths and slice offsets come solely from the supplied elaborated HW.
    """
    definitions = register_definitions(hw_text, module)
    if (type(selector_value) is not int or not registers or len(set(registers)) != len(registers)
            or selector in inputs or not inputs
            or any(not name or type(width) is not int or width <= 0 for name, width in inputs.values())):
        raise ValueError("malformed register slice request")
    active = set()

    @lru_cache(None)
    def values(ref):
        if ref == selector:
            return frozenset([selector_value])
        if ref in inputs:
            label, width = inputs[ref]
            return frozenset([("slice", label, 0, width)])
        if ref in active or len(active) >= 128:
            return frozenset([("unknown", ref)])
        active.add(ref)
        try:
            return evaluate(ref)
        except (ValueError, IndexError, KeyError, TypeError):
            return frozenset([("unknown", ref)])
        finally:
            active.remove(ref)

    def evaluate(ref):
        body = definitions.get(ref, "")
        if body.startswith("hw.constant "):
            token = body.split()[1]
            if token in ("true", "false"):
                return frozenset([int(token == "true")])
            width = int(body.rsplit(" : i", 1)[1])
            return frozenset([int(token) % (1 << width)])
        if body.startswith("seq.firreg "):
            return frozenset([("state", ref)])
        refs = _operand_refs(body, "comb.mux")
        if refs:
            condition = values(refs[0])
            result = set()
            if any(v != 0 for v in condition):
                result.update(values(refs[1]))
            if any(v != 1 for v in condition):
                result.update(values(refs[2]))
            return frozenset(result)
        if body.startswith("comb.extract "):
            source, rest = body.removeprefix("comb.extract ").split(" from ", 1)
            offset, width = int(rest.split()[0]), int(body.rsplit("-> i", 1)[1])
            result = set()
            for item in values(source):
                if type(item) is int:
                    result.add((item >> offset) & ((1 << width) - 1))
                elif item[0] == "slice" and 0 <= offset and offset + width <= item[3]:
                    result.add(("slice", item[1], item[2] + offset, width))
                else:
                    result.add(("unknown", ref))
            return frozenset(result)
        if body.startswith("comb.icmp "):
            predicate, rest = body.removeprefix("comb.icmp ").removeprefix("bin ").split(" ", 1)
            refs = [r.strip() for r in rest.split(" :", 1)[0].split(",")]
            result = set()
            for a, b in product(*(values(r) for r in refs)):
                if type(a) is int and type(b) is int and predicate in ("eq", "ne"):
                    result.add(int((a == b) == (predicate == "eq")))
                else:
                    result.update((0, 1))
            return frozenset(result)
        for operation in ("and", "or", "xor"):
            refs = _operand_refs(body, "comb." + operation)
            if not refs:
                continue
            width = int(body.rsplit(" : i", 1)[1])
            choices = [values(r) for r in refs]
            combinations = 1
            for choice in choices:
                combinations *= len(choice)
            if combinations > 256:
                return frozenset([("unknown", ref)])
            result = set()
            for operands in product(*choices):
                if operation == "and" and 0 in operands:
                    result.add(0)
                elif operation == "or" and width == 1 and 1 in operands:
                    result.add(1)
                elif all(type(v) is int for v in operands):
                    value = operands[0]
                    for operand in operands[1:]:
                        if operation == "and": value &= operand
                        elif operation == "or": value |= operand
                        else: value ^= operand
                    result.add(value)
                elif width == 1:
                    result.update((0, 1))
                else:
                    result.add(("unknown", ref))
            return frozenset(result)
        return frozenset([("unknown", ref)])

    rows = []
    for register in registers:
        body = definitions.get(register, "")
        if not body.startswith("seq.firreg "):
            rows.append({"register": register, "status": "UNKNOWN", "reason": "not a register"})
            continue
        if " reset " in body:
            rows.append({"register": register, "status": "UNKNOWN", "reason": "reset update not modeled"})
            continue
        width = int(body.rsplit(" : i", 1)[1])
        possible = set(values(body.split()[1])) - {("state", register)}
        row = {"register": register, "register_width": width}
        if not possible:
            row["status"] = "no_non_hold_update"
        elif len(possible) == 1:
            value = next(iter(possible))
            if type(value) is int and 0 <= value < 1 << width:
                row.update(status="derived", constant=value)
            elif isinstance(value, tuple) and value[0] == "slice" and value[3] == width:
                row.update(status="derived", input=value[1], offset=value[2], width=value[3])
            else:
                row.update(status="UNKNOWN", reason="unsupported data update")
        else:
            row.update(status="UNKNOWN", reason="multiple possible non-hold data updates")
        rows.append(row)
    return {"schema": "conditional_register_slices_v1", "module": module,
            "selector_value": selector_value, "registers": rows,
            "scope": "all possible non-hold updates under selector; acceptance not proved",
            "arithmetic_and_sequencer_progress": "UNPROVEN"}


def decode_register_slices(layout: dict, operands: dict[str, dict]) -> list[dict]:
    """Apply trusted recovered slices; symbolic pointers retain truncation.

    This does not establish that an ABI address is in range or that a stride
    counts bytes/elements. Such units are downstream target semantics.
    """
    if layout.get("schema") != "conditional_register_slices_v1":
        raise ValueError("unsupported register layout")
    rows = []
    for field in layout["registers"]:
        if field["status"] == "no_non_hold_update":
            continue
        row = dict(field)
        if field["status"] == "derived":
            if "constant" in field:
                row["value"] = field["constant"]
            else:
                offset, width = field["offset"], field["width"]
                if type(offset) is not int or type(width) is not int or offset < 0 or width <= 0:
                    raise ValueError("invalid recovered bit slice")
                operand = operands.get(field["input"], {})
                if operand.get("kind") == "const" and type(operand.get("raw")) is int:
                    row["value"] = (operand["raw"] >> offset) & ((1 << width) - 1)
                elif (operand.get("kind") == "argbase" and type(operand.get("arg_index")) is int
                      and operand["arg_index"] >= 0 and type(operand.get("offset")) is int):
                    row["symbolic_value"] = {"argument": operand["arg_index"], "byte_offset": operand["offset"],
                        "extract_offset": offset, "extract_width": width,
                        "address_range_validated": False}
                else:
                    row.update(status="UNKNOWN", reason="emitted operand unresolved")
        rows.append(row)
    return rows
