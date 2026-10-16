"""Explicit original-source load-state banks; no command acceptance authority."""

from dataclasses import dataclass

from . import firrtl_struct as F
from .firrtl_operand_slices import _source


@dataclass(frozen=True)
class LoadStateMapping:
    source: str
    source_sha256: str
    config_class: str
    ports: tuple[tuple[str, int], ...]
    states: tuple[int, ...]
    stride_width: int

    def port(self, klass):
        return dict(self.ports).get(klass)


def derive_load_state_mapping(layouts, selection, isa):
    """Re-derive every selected state from one actual bank and selector cone."""
    fields = {
        "module",
        "command",
        "state",
        "config_state",
        "config_field",
        "config_class",
        "stride_register",
        "stride_input",
        "dram_address",
        "row_index",
        "stride_sink",
        "load_classes",
    }
    if type(selection) is not dict or set(selection) != fields:
        raise ValueError("load-state selection has missing or unsupported fields")
    if any(type(selection[key]) is not str or not selection[key] for key in fields - {"load_classes", "config_field"}):
        raise ValueError("load-state selection requires exact original source references")
    classes = selection["load_classes"]
    if (
        type(classes) is not list
        or not 1 <= len(classes) <= 32
        or any(type(label) is not str or not label for label in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("load-state selection has an ambiguous command roster")
    field = selection["config_field"]
    if type(field) is not dict or set(field) != {"bundle", "name"}:
        raise ValueError("load-state configuration lacks its original operand field")
    matches = [row for row in layouts["fields"] if (row["bundle"], row["name"]) == (field["bundle"], field["name"])]
    if (
        len(matches) != 1
        or matches[0]["status"] != "derived"
        or matches[0]["module"] != selection["module"]
        or matches[0]["field"] != selection["config_state"]
    ):
        raise ValueError("load-state configuration differs from selected operand source")
    programs = isa["OPERAND_FIELDS"]
    config = programs.get(selection["config_class"], {}).get("fields", [])
    state_fields = [step for step in config if step["name"] == "state_id"]
    stride_fields = [step for step in config if step["name"] == "stride"]
    if (
        len(state_fields) != 1
        or state_fields[0].get("operand") != "rs1"
        or state_fields[0].get("width") != matches[0]["width"]
        or state_fields[0].get("offset") != matches[0]["offset"]
        or len(stride_fields) != 1
        or stride_fields[0].get("operand") != "rs2"
        or stride_fields[0].get("as") != "raw"
        or selection["config_class"] not in isa["CONFIG_SUBTYPE"].values()
    ):
        raise ValueError("load-state configuration has no matching selected decoder")
    command_codes = {label: code for code, label in isa["FUNCT_CLASS"].items()}
    if len(command_codes) != len(isa["FUNCT_CLASS"]) or any(label not in command_codes for label in classes):
        raise ValueError("load-state commands differ from the original class roster")
    for label in classes:
        dram = [step for step in programs.get(label, {}).get("fields", []) if step["name"] == "dram"]
        if len(dram) != 1 or dram[0].get("as") != "operand" or dram[0].get("operand") != "rs1":
            raise ValueError("load-state class has no selected DRAM input role")
    path = _source(layouts["source"], layouts["source_sha256"])
    bodies = F.read_module_bodies(path, [selection["module"]])
    if selection["module"] not in bodies:
        raise ValueError("load-state source module is absent")
    lines = bodies[selection["module"]]
    # Invalidations are not ordinary drivers. An invalidated field may not
    # enter any proved cone; unrelated reserved/spacer fields grant nothing.
    seen = set()
    invalidated = []
    for _number, raw in lines[1:]:
        statement = F.split_locator(raw)[0].strip()
        words = statement.split(None, 1)
        if words and (words[0] == "invalidate" or statement.endswith(" is invalid")):
            ref = F.parse_expr(words[1] if words[0] == "invalidate" else statement.removesuffix(" is invalid"))
            if not isinstance(ref, F.Ref) or ref.dynamic:
                raise ValueError("load-state source invalidation has unknown identity")
            invalidated.append(ref)
        if words and words[0] in {"node", "wire", "input", "output", "reg", "regreset"}:
            name = words[1].split("=", 1)[0].split(":", 1)[0].strip()
            if name in seen:
                raise ValueError("load-state source has duplicate definitions")
            seen.add(name)
    module = F.parse_module(selection["module"], lines)
    refs = {
        key: F.parse_expr(selection[key])
        for key in ("command", "state", "config_state", "stride_input", "dram_address", "row_index", "stride_sink")
    }
    if any(not isinstance(ref, F.Ref) or ref.dynamic for ref in refs.values()):
        raise ValueError("load-state references must be exact static source paths")
    input_ref = F.parse_expr(matches[0]["input"])
    if (
        not isinstance(input_ref, F.Ref)
        or input_ref.dynamic
        or not input_ref.segments
        or input_ref.segments[-1] != "rs1"
        or refs["dram_address"] != input_ref
        or refs["stride_input"] != F.Ref(input_ref.root, (*input_ref.segments[:-1], "rs2"))
    ):
        raise ValueError("load-state stride/address inputs differ from the original command registers")
    register = selection["stride_register"]
    declaration = module.decls.get(register)
    typ = F.parse_type(declaration[1]) if declaration is not None and declaration[0] == "reg" else None
    if (
        typ is None
        or typ[0] != "vec"
        or typ[1][0] != "uint"
        or type(typ[1][1]) is not int
        or not 0 < typ[1][1] <= layouts["word_bits"]
        or not 0 < typ[2] <= 32
    ):
        raise ValueError("load-state stride bank has no actual bounded unsigned vector type")
    bits, count = typ[1][1], typ[2]
    bank_read = F.Ref(register, (None,), (refs["state"],))

    def expand(expr, *, keep=(), active=()):
        if isinstance(expr, F.Ref) and any(
            expr.root == ref.root and expr.segments[: len(ref.segments)] == ref.segments for ref in invalidated
        ):
            raise ValueError("load-state source uses invalidated state or address data")
        if expr in keep or isinstance(expr, F.Lit | F.IntArg):
            return expr
        if len(active) >= 128 or expr in active:
            raise ValueError("load-state source cone is cyclic or oversized")
        if isinstance(expr, F.Ref):
            if expr in {
                bank_read,
                refs["command"],
                refs["config_state"],
                refs["stride_input"],
                refs["dram_address"],
                refs["row_index"],
            }:
                return expr
            if expr.root in module.nodes and not expr.segments:
                rhs = module.nodes[expr.root][0]
            else:
                drivers = [c for c in module.connects if c.lhs == expr]
                if len(drivers) != 1 or drivers[0].guards:
                    raise ValueError("load-state source has ambiguous or conditional drivers")
                rhs = drivers[0].rhs
            return expand(rhs, keep=keep, active=(*active, expr))
        if isinstance(expr, F.Prim):
            return F.Prim(expr.op, tuple(expand(arg, keep=keep, active=(*active, expr)) for arg in expr.args))
        raise ValueError("load-state source has an unsupported expression")

    state_expression = expand(refs["state"])

    def evaluate(expr, code, config_state):
        if expr == refs["command"]:
            return code
        if expr == refs["config_state"]:
            return config_state
        if isinstance(expr, F.Lit) and expr.kind == "UInt":
            return expr.value
        if isinstance(expr, F.Prim) and expr.op == "eq" and len(expr.args) == 2:
            operands = [evaluate(arg, code, config_state) for arg in expr.args]
            if any(type(value) is not int for value in operands):
                raise ValueError("load-state selector depends on unknown operand data")
            return int(operands[0] == operands[1])
        if isinstance(expr, F.Prim) and expr.op == "mux" and len(expr.args) == 3:
            condition = evaluate(expr.args[0], code, config_state)
            if type(condition) is not int or condition not in {0, 1}:
                raise ValueError("load-state selector lacks a defined Boolean condition")
            return evaluate(expr.args[1 if condition else 2], code, config_state)
        raise ValueError("load-state selector has an unsupported or unbound condition")

    config_code = command_codes.get(isa["CONFIG_CLASS"])
    if config_code is None or any(evaluate(state_expression, config_code, i) != i for i in range(count)):
        raise ValueError("load-state configuration does not select the same actual bank")
    ports = tuple((label, evaluate(state_expression, command_codes[label], None)) for label in classes)
    if any(type(index) is not int or not 0 <= index < count for _label, index in ports):
        raise ValueError("load-state command selects an unknown or outside bank index")
    writes = [c for c in module.connects if c.lhs.root == register]
    if len(writes) != 1 or writes[0].lhs != bank_read or writes[0].rhs != refs["stride_input"]:
        raise ValueError("load-state stride bank has an unproved writer")
    guards = []
    for guard in writes[0].guards:
        if guard.positive:
            try:
                guards.append(expand(guard.cond))
            except ValueError:
                pass  # Additional readiness predicates cannot grant configuration.
    command_guard = F.Prim("eq", (refs["command"], F.Lit("UInt", config_code.bit_length() or 1, config_code)))
    if command_guard not in guards:
        raise ValueError("load-state stride writer is not confined to configuration commands")
    expected_address = F.Prim("add", (refs["dram_address"], F.Prim("mul", (refs["row_index"], bank_read))))
    address = expand(refs["stride_sink"])
    if isinstance(address, F.Prim) and address.op == "tail" and address.args == (expected_address, F.IntArg(1)):
        # FIRRTL add has one carry bit. Dropping exactly that bit retains the
        # original unsigned address width, provided the stride product fits it.
        row = module.decls.get(refs["row_index"].root)
        row_type = F.parse_type(row[1]) if row is not None and row[0] in {"reg", "regreset"} else None
        if (
            refs["row_index"].segments
            or row_type is None
            or row_type[0] != "uint"
            or type(row_type[1]) is not int
            or row_type[1] + bits > layouts["word_bits"]
        ):
            raise ValueError("load-state address carry truncation lacks exact unsigned widths")
        address = address.args[0]
    if address != expected_address:
        raise ValueError("load-state DRAM address does not use the selected stride bank")
    _source(path, layouts["source_sha256"])
    return LoadStateMapping(
        str(path), layouts["source_sha256"], selection["config_class"], ports, tuple(range(count)), bits
    )
