"""Generic RoCC operand semantics, driven entirely by a target's RTL facts and contract.

This is the host-side ``rocc_semantics`` capability :mod:`merlin.targetgen.rocc.decode` requires
(``isa_constants``, ``decode_instruction``, ``instruction_funct``), written once for every RoCC
accelerator instead of once per support package. Nothing here names an accelerator:

* the funct-code -> class vocabulary is the contract's ``encoding.semantic_class`` (and the CONFIG
  refinement ``encoding.config_subtype``), each code range-checked against the RTL-derived field width;
* the custom opcode, funct3 and the systolic DIM are RTL facts (``funct_decode_table``,
  ``arrays[mesh]``);
* every operand field's bit position and width is an RTL fact: the ``register_bundle_layouts``
  interface CIRCT extracted from the target's own Scala Bundles;
* which bundle field each decoded value comes from, and how it is interpreted (raw integer, float32
  bits, a local-address flag, an all-ones sentinel), is the contract's ``rocc_operand_roles`` block —
  the human-reviewed ABI annotation the RTL cannot ground on its own.

A field whose bundle or role is not declared is left UNKNOWN (omitted or ``None``, per the class's
``unresolved`` policy); no field is ever decoded at an assumed offset.

``rtl_checks`` (an attribute of this module) is :mod:`merlin.targetgen.rtl_checks_generic`, the
structural-check capability :func:`merlin.targetgen.rtl_checks.selected_checks` resolves as
``rocc_semantics.rtl_checks``.
"""

from __future__ import annotations

import struct
from typing import Any

from merlin.common.facts_view import interface as _facts_interface

#: Width of the R-type ``funct7`` field that carries a RoCC instruction's identity. This is the RISC-V
#: RoCC transport itself (``.insn r opcode, funct3, funct7, rd, rs1, rs2``), not a property of any
#: accelerator; it is used only when the facts record carries no decoder-observed width of its own.
ROCC_FUNCT7_WIDTH = 7

#: Contract keys this module reads.
OPERAND_ROLES_KEY = "rocc_operand_roles"

_OPERANDS = ("rs1", "rs2")


class OperandRolesError(ValueError):
    """The contract's ``rocc_operand_roles`` block is malformed."""


# --------------------------------------------------------------------------------------------- inputs
def _load_manifest(target: str):
    # Looked up through the module attribute at call time, so a caller may substitute the manifest.
    from merlin.targetgen import target_experiment

    return target_experiment.load_capability_manifest(target)


def _load_facts(target: str) -> dict:
    from merlin.targetgen.rtl import facts

    return facts.load_facts(target)


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def bundle_layouts(facts: dict) -> dict:
    """``{bundle name: layout}`` from the facts' ``register_bundle_layouts`` interface ({} if none)."""
    body = facts.get("facts", facts) if isinstance(facts, dict) else {}
    block = _facts_interface(body, "register_bundle_layouts") or {}
    bundles = block.get("bundles")
    return bundles if isinstance(bundles, dict) else {}


def field_slot(layout: dict | None, field: str) -> tuple[int, int] | None:
    """``(offset, width)`` of ``field`` in a CIRCT-extracted bundle layout, or ``None`` (UNKNOWN).

    A field whose RTL width is a generator parameter carries the software ABI's ``slot_width`` (the
    bits the operand reserves for it); that slot is what an encoder packs and so what a decoder reads.
    """
    if not isinstance(layout, dict):
        return None
    spec = (layout.get("fields") or {}).get(field)
    if not isinstance(spec, dict):
        return None
    offset = spec.get("offset")
    width = spec.get("width")
    if not isinstance(width, int):
        width = spec.get("slot_width")
    if type(offset) is int and type(width) is int and offset >= 0 and width > 0:
        return offset, width
    return None


def _numeric_code_map(declared: object, *, field: str, width: object) -> dict[int, str]:
    """Import a code-indexed contract map across YAML/JSON boundaries.

    JSON requires object keys to be strings, whereas authored YAML uses integer keys. Accept the two
    unambiguous representations, then use the RTL-derived field width to reject codes the instruction
    cannot carry. Malformed or colliding spellings never become aliases, and labels are unique because
    :func:`instruction_funct` inverts these maps.
    """
    if type(width) is not int or width <= 0:
        raise ValueError(f"{field} has no positive RTL-derived code width")
    if not isinstance(declared, dict) or not declared:
        raise ValueError(f"{field} must be a nonempty code-to-class mapping")
    normalized: dict[int, str] = {}
    for source, label in declared.items():
        if type(source) is int:
            code = source
        elif type(source) is str:
            try:
                code = int(source)
            except ValueError as exc:
                raise ValueError(f"{field} has a nonnumeric code {source!r}") from exc
            if str(code) != source:
                raise ValueError(f"{field} has a noncanonical decimal code {source!r}")
        else:
            raise ValueError(f"{field} has a noninteger code {source!r}")
        if code < 0 or code >= (1 << width):
            raise ValueError(f"{field} code {code} exceeds its {width}-bit field")
        if code in normalized:
            raise ValueError(f"{field} has colliding spellings for code {code}")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{field} code {code} has no class label")
        normalized[code] = label
    if len(set(normalized.values())) != len(normalized):
        raise ValueError(f"{field} has duplicate class labels")
    return normalized


# ------------------------------------------------------------------------------- operand-role program
def operand_roles(contract: dict | None) -> dict:
    """The contract's ``rocc_operand_roles`` block, or ``{}`` when the target declares none."""
    block = (contract or {}).get(OPERAND_ROLES_KEY)
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise OperandRolesError(f"{OPERAND_ROLES_KEY} must be a mapping")
    return block


def local_address_flags(roles: dict, addr_len: object) -> dict[str, int]:
    """``{flag role: bit mask}`` for the local-address flag bits, counted down from the address MSB."""
    flags = roles.get("local_address_flags") or {}
    if not flags:
        return {}
    if type(addr_len) is not int or addr_len <= 0:
        raise OperandRolesError("local_address_flags need a positive encoding.addr_len")
    out: dict[str, int] = {}
    for name, spec in flags.items():
        offset = (spec or {}).get("msb_offset") if isinstance(spec, dict) else None
        if type(offset) is not int or not 0 <= offset < addr_len:
            raise OperandRolesError(f"local_address_flags.{name} needs an msb_offset inside addr_len")
        out[str(name)] = 1 << (addr_len - 1 - offset)
    return out


def _named_constants(roles: dict, flags: dict[str, int]) -> dict[str, int]:
    """``rocc_operand_roles.isa_constants``: names composed from flags or a float32 scalar."""
    out: dict[str, int] = {}
    for name, spec in (roles.get("isa_constants") or {}).items():
        if not isinstance(spec, dict):
            raise OperandRolesError(f"isa_constants.{name} must be a mapping")
        if "flags" in spec:
            value = 0
            for flag in spec["flags"] or ():
                if flag not in flags:
                    raise OperandRolesError(f"isa_constants.{name} names undeclared flag {flag!r}")
                value |= flags[flag]
            out[str(name)] = value
        elif "f32_bits" in spec:
            out[str(name)] = _f32_bits(spec["f32_bits"])
        else:
            raise OperandRolesError(f"isa_constants.{name} declares neither flags nor f32_bits")
    return out


def encoding_fields(declared: dict, *, target: str | None = None, contract: dict | None = None) -> dict:
    """Complete ``declared`` (a contract ``encoding`` block) with ``readout_bits`` derived from the
    contract's local-address flags, without mutating it. Returns it unchanged when the target's
    contract (``contract`` or the one ``target`` resolves) declares no ``readout_bits`` composition."""
    encoding = dict(declared)
    if "readout_bits" in encoding or encoding.get("addr_len") is None:
        return encoding
    if contract is None:
        if target is None:
            return encoding
        contract = _load_manifest(target).contract
    roles = operand_roles(contract)
    names = roles.get("readout_bits") or {}
    if not names:
        return encoding
    constants = _named_constants(roles, local_address_flags(roles, int(encoding["addr_len"])))
    missing = [v for v in names.values() if v not in constants]
    if missing:
        raise OperandRolesError(f"readout_bits names undeclared isa_constants {missing}")
    encoding["readout_bits"] = {str(k): constants[v] for k, v in names.items()}
    return encoding


def derived_readout_bits(addr_len: int, *, target: str | None = None, contract: dict | None = None) -> dict[str, int]:
    """``encoding.readout_bits`` for an ``addr_len``-bit local address, composed from the contract.

    The values are the contract's ``rocc_operand_roles.readout_bits`` names evaluated over its declared
    local-address flags (bit offsets from the address MSB) and float32 scalars, for the given width.
    Nothing is assumed about which flags exist: a contract (``contract``, or the one ``target``
    resolves) that declares no ``readout_bits`` composition yields ``{}``.
    """
    if type(addr_len) is not int or addr_len <= 0:
        raise OperandRolesError(f"addr_len must be a positive integer, got {addr_len!r}")
    if contract is None:
        if target is None:
            raise OperandRolesError("derived_readout_bits needs the target (or its contract) whose roles compose it")
        contract = _load_manifest(target).contract
    completed = encoding_fields({"addr_len": addr_len}, contract=contract)
    return dict(completed.get("readout_bits") or {})


_INTERPRETATIONS = {"int", "operand", "raw", "f32", "flag", "flag_choice", "all_ones", "equals", "const"}


def _compile_field(cls: str, spec: dict, layouts: dict, flags: dict[str, int]) -> list[dict]:
    """One declared field -> zero or more executable steps with RTL-resolved offsets/widths."""
    if not isinstance(spec, dict):
        raise OperandRolesError(f"{cls}: a field entry must be a mapping")
    if "names" in spec:  # every listed bundle field the RTL layout carries, each as a plain int
        bundle = spec.get("bundle")
        steps = []
        for name in spec["names"] or ():
            slot = field_slot(layouts.get(bundle), str(name))
            if slot is not None:
                steps.append(
                    {
                        "name": str(name),
                        "as": "int",
                        "operand": spec.get("operand"),
                        "offset": slot[0],
                        "width": slot[1],
                        "omit_unresolved": True,
                    }
                )
        return steps
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise OperandRolesError(f"{cls}: a field entry has no name")
    if "const" in spec:
        return [{"name": name, "as": "const", "value": spec["const"]}]
    if "from" in spec:
        return [{"name": name, "as": "equals", "from": str(spec["from"]), "value": spec.get("equals")}]
    kind = spec.get("as", "int")
    if kind not in _INTERPRETATIONS:
        raise OperandRolesError(f"{cls}.{name}: unknown interpretation {kind!r}")
    operand = spec.get("operand")
    if operand not in _OPERANDS:
        raise OperandRolesError(f"{cls}.{name}: operand must be one of {_OPERANDS}")
    step: dict[str, Any] = {"name": name, "as": kind, "operand": operand}
    if kind in ("operand", "raw"):
        return [step]
    slot = field_slot(layouts.get(spec.get("bundle")), str(spec.get("field")))
    if slot is None:
        # The RTL layout does not carry this field: the value stays UNKNOWN, never an assumed offset.
        step.update(offset=None, width=None)
    else:
        step.update(offset=slot[0], width=slot[1])
    if kind in ("flag", "flag_choice"):
        flag = spec.get("flag")
        if flag not in flags:
            raise OperandRolesError(f"{cls}.{name}: names undeclared local-address flag {flag!r}")
        step["mask"] = flags[flag]
        if kind == "flag_choice":
            step["set"], step["clear"] = spec.get("set"), spec.get("clear")
    return [step]


def _compile_classes(roles: dict, layouts: dict, flags: dict[str, int]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cls, spec in (roles.get("classes") or {}).items():
        if not isinstance(spec, dict):
            raise OperandRolesError(f"classes.{cls} must be a mapping")
        policy = spec.get("unresolved", "omit")
        if policy not in ("omit", "null"):
            raise OperandRolesError(f"classes.{cls}.unresolved must be omit|null")
        steps: list[dict] = []
        for field in spec.get("fields") or ():
            steps.extend(_compile_field(str(cls), field, layouts, flags))
        out[str(cls)] = {"unresolved": policy, "fields": steps}
    return out


# ------------------------------------------------------------------------------------- ISA constants
def isa_constants(target: str) -> dict:
    """Derive the RoCC ISA constants for ``target`` from its RTL facts + contract. No target is baked
    in — the caller passes the target it is grading; the decoder holds no default."""
    manifest = _load_manifest(target)
    enc = dict(manifest.encoding)
    roles = operand_roles(manifest.contract)
    facts = _load_facts(target)["facts"]
    mesh = next((a for a in facts.get("arrays", []) if a.get("name") == "mesh"), {})
    fdt = _facts_interface(facts, "funct_decode_table") or {}
    layouts = bundle_layouts(facts)

    width = fdt.get("width") if type(fdt.get("width")) is int else ROCC_FUNCT7_WIDTH
    funct_class = _numeric_code_map(enc.get("semantic_class"), field="semantic_class", width=width)

    config = roles.get("config") or {}
    selector = config.get("selector") or {}
    selector_slot = field_slot(layouts.get(selector.get("bundle")), str(selector.get("field")))
    config_subtype: dict[int, str] = {}
    if enc.get("config_subtype") is not None or config:
        config_subtype = _numeric_code_map(
            enc.get("config_subtype"), field="config_subtype", width=selector_slot[1] if selector_slot else None
        )

    flags = local_address_flags(roles, enc.get("addr_len"))
    out: dict[str, Any] = {"DIM": mesh.get("rows")}
    out.update(_named_constants(roles, flags))
    out.update(
        {
            "CUSTOM_OPCODE": fdt.get("custom_opcode"),
            "FUNCT3": fdt.get("funct3"),
            "FUNCT_CLASS": funct_class,
            "CONFIG_SUBTYPE": config_subtype,
        }
    )
    for key, bundle in (config.get("publish_layouts") or {}).items():
        out[str(key)] = layouts.get(bundle)

    sentinel = roles.get("retain_sentinel")
    if isinstance(sentinel, dict) and sentinel.get("value") == "all_ones":
        slot = field_slot(layouts.get(sentinel.get("bundle")), str(sentinel.get("field")))
        if slot is not None:
            out["RETAIN_SENTINEL"] = (1 << slot[1]) - 1

    # The executable decode program: which bundle slot each decoded field reads, resolved against THIS
    # facts record. Carried in the ISA dict so decode_instruction needs no second lookup.
    out["CONFIG_CLASS"] = config.get("class")
    out["CONFIG_SELECTOR"] = (
        {"operand": selector.get("operand", "rs1"), "offset": selector_slot[0], "width": selector_slot[1]}
        if selector_slot
        else None
    )
    out["OPERAND_FIELDS"] = _compile_classes(roles, layouts, flags)
    return out


# --------------------------------------------------------------------------------------------- decode
def _extract(raw: int | None, step: dict) -> int | None:
    if raw is None or step.get("offset") is None:
        return None
    return (raw >> step["offset"]) & ((1 << step["width"]) - 1)


def _decode_fields(program: dict, rs1: dict, rs2: dict) -> dict:
    ops = {"rs1": rs1, "rs2": rs2}
    raws = {k: (v.get("raw") if v.get("kind") == "const" else None) for k, v in ops.items()}
    keep_none = program["unresolved"] == "null"
    dec: dict = {}
    for step in program["fields"]:
        kind, name = step["as"], step["name"]
        if kind == "const":
            dec[name] = step["value"]
            continue
        if kind == "operand":
            dec[name] = dict(ops[step["operand"]])
            continue
        if kind == "equals":
            src = dec.get(step["from"])
            value = (src == step["value"]) if src is not None else None
        elif kind == "raw":
            value = raws[step["operand"]]
        else:
            bits = _extract(raws[step["operand"]], step)
            if bits is None:
                value = None
            elif kind == "int":
                value = bits
            elif kind == "f32":
                value = _f32_from_bits(bits)
            elif kind == "flag":
                value = bool(bits & step["mask"])
            elif kind == "flag_choice":
                value = step["set"] if (bits & step["mask"]) else step["clear"]
            else:  # all_ones
                value = bits == (1 << step["width"]) - 1
        if value is not None or (keep_none and not step.get("omit_unresolved")):
            dec[name] = value
    return dec


def decode_instruction(funct: int, rs1: dict, rs2: dict, isa: dict) -> tuple[str, dict]:
    """Return (class, decoded-fields) for one ``.insn`` given resolved operands and the target ``isa``."""
    base = isa["FUNCT_CLASS"].get(funct, "UNKNOWN")
    programs = isa.get("OPERAND_FIELDS") or {}
    if base != "UNKNOWN" and base == isa.get("CONFIG_CLASS"):
        selector = isa.get("CONFIG_SELECTOR")
        operand = {"rs1": rs1, "rs2": rs2}.get((selector or {}).get("operand"), rs1)
        raw = operand.get("raw") if operand.get("kind") == "const" else None
        code = _extract(raw, selector) if selector else None
        sub = isa["CONFIG_SUBTYPE"].get(code) if code is not None else None
        if sub is None:
            return "UNKNOWN", {}
        program = programs.get(sub)
        return sub, (_decode_fields(program, rs1, rs2) if program else {})
    program = programs.get(base)
    if program is None:
        return base, {}
    return base, _decode_fields(program, rs1, rs2)


def _class_to_funct(isa: dict) -> dict[str, int]:
    """``class-name -> func7`` from FUNCT_CLASS, plus the CONFIG subtypes (which share the CONFIG funct)."""
    inv = {v: k for k, v in isa["FUNCT_CLASS"].items()}
    config = inv.get(isa.get("CONFIG_CLASS")) if isa.get("CONFIG_CLASS") else None
    if config is not None:
        for sub in isa["CONFIG_SUBTYPE"].values():
            inv[sub] = config
    return inv


def instruction_funct(name: str, rs1: int, isa: dict) -> int:
    """The funct7 for class ``name``; a CONFIG subtype also requires ``rs1`` to select it."""
    classes = _class_to_funct(isa)
    funct = classes.get(name)
    if funct is None:
        raise ValueError(f"unknown instruction class {name!r}; legal classes: {sorted(classes)}")
    want = next((bits for bits, sub in isa["CONFIG_SUBTYPE"].items() if sub == name), None)
    if want is not None:
        selector = isa.get("CONFIG_SELECTOR")
        if not selector:
            raise ValueError(f"{name} is a CONFIG subtype but the RTL facts carry no selector field layout")
        mask = (1 << selector["width"]) - 1
        got = (rs1 >> selector["offset"]) & mask
        expr = f"(rs1 & {mask:#x})" if selector["offset"] == 0 else f"((rs1 >> {selector['offset']}) & {mask:#x})"
        if got != want:
            raise ValueError(
                f"{name} requires {expr} == {want}; got rs1={rs1} ({expr} == {got}). "
                f"Set the low {selector['width']} bits of rs1 to select the subtype."
                if selector["offset"] == 0
                else f"{name} requires {expr} == {want}; got rs1={rs1} ({expr} == {got})."
            )
    return funct


def __getattr__(name: str):
    if name == "rtl_checks":
        from merlin.targetgen import rtl_checks_generic

        return rtl_checks_generic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
