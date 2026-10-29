"""Complete current straight-line Kernel projection for configuration deletion.

Tokens preserve the full typed Kernel structure. Effects come independently
from current selected ISA encodings and declarations. Unknown/dynamic/async
calls are retained barriers; no physical or instruction implementation theorem
is inferred from successful structural decoding.
"""

from __future__ import annotations

import dataclasses
import struct

from merlin.sched.ir import Call, Kernel, parse_kernel
from merlin.sched.ir.kernel import check_structure

from .configuration_elision_bridge import Unsupported, _encoding, canonical, digest
from .configuration_hoist_bridge import Unsupported as UnknownCall
from .configuration_hoist_bridge import decode_effect


def tree(value):
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return {"float64_bits": struct.pack(">d", value).hex()}
    if type(value) is tuple:
        return {"tuple": [tree(item) for item in value]}
    name = type(value).__module__ + "." + type(value).__name__
    if dataclasses.is_dataclass(value) and type(value).__module__ in ("merlin.sched.ir.kernel", "merlin.sched.ir.expr"):
        return {
            "class": name,
            "fields": {field.name: tree(getattr(value, field.name)) for field in dataclasses.fields(value)},
        }
    if name == "merlin.sched.ir.kernel._Null":
        return {"class": name, "fields": {}}
    raise Unsupported("unsupported complete Kernel field type: " + name)


def program(source, *, isa, effects):
    kernel = parse_kernel(source)
    if type(kernel) is not Kernel or check_structure(kernel) or kernel.text() != source:
        raise Unsupported("exact complete typed canonical Kernel source required")
    if any(type(statement) is not Call for statement in kernel.body):
        raise Unsupported("configuration Kernel elision supports complete straight-line Call bodies")
    frame = tree(kernel)["fields"]
    frame.pop("body")
    rows = []
    instructions = {row["name"]: row for row in isa["instructions"]}
    for call in kernel.body:
        token = tree(call)
        try:
            effect = decode_effect(token, isa, effects)
            if effect is not None:
                # Independent exact encoding uniqueness/round-trip check,
                # separate from the effect-field projection above.
                values = {key: value.value for key, value in call.args}
                _encoding(isa, instructions[call.instr], values)
        except (UnknownCall, AttributeError):
            effect = None
        rows.append(
            {
                "token": canonical(token),
                "effect": effect,
                "eraseAllowed": effect is not None and effect["pureWrite"],
                "decoded": None,
            }
        )
    return {"frame": canonical(frame), "entries": [{"name": kernel.name, "header": canonical(frame), "code": rows}]}


def project(source, final, *, isa, effects):
    before, after = (program(text, isa=isa, effects=effects) for text in (source, final))
    fields = sorted(
        {
            field
            for p in (before, after)
            for e in p["entries"]
            for atom in e["code"]
            if atom["effect"] is not None
            for field in [
                *(x[0] for x in atom["effect"]["reads"]),
                *(x[0] for x in atom["effect"]["writes"]),
                *atom["effect"]["clobbers"],
            ]
        }
    )
    if len(fields) > 256:
        raise Unsupported("configuration actual Kernel field registry exceeds bounded domain")
    for p in (before, after):
        for e in p["entries"]:
            for atom in e["code"]:
                effect = atom["effect"]
                if effect is not None:
                    for key in ("reads", "writes"):
                        effect[key] = [[fields.index(field), value] for field, value in effect[key]]
                    effect["clobbers"] = [fields.index(field) for field in effect["clobbers"]]
    original = before["entries"][0]["code"]
    for i, atom in enumerate(original):
        atom["id"] = i
    cursor = 0
    for i, atom in enumerate(after["entries"][0]["code"]):
        found = next(
            (
                j
                for j in range(cursor, len(original))
                if all(original[j][key] == atom[key] for key in ("token", "effect", "eraseAllowed"))
            ),
            None,
        )
        atom["id"] = len(original) + i if found is None else found
        if found is not None:
            cursor = found + 1
    return {
        "schema": "merlin.configuration-elision.projection.v1",
        "fields": fields,
        "before": before,
        "after": after,
        "source_sha256": digest(source),
        "final_sha256": digest(final),
        "representation": "current_complete_straightline_kernel",
        "projection_is_trusted_python": True,
        "hardware_model_approved": False,
        "proof_authority": False,
    }
