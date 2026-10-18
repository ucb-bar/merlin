"""Bind device spill capabilities to selected resource and software contracts.

Reuse the existing spill planner and actual-byte provenance checker. Only
addressable device byte registers and independently qualified synchronous,
bit-preserving transfers enter that planner. No host register allocation or
latency estimate is inferred from a resource declaration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from merlin.sched.ir import Call
from merlin.targetgen.isa_spec import resolve_fact_integer

from .spill_conformance import SpillClass, validate_spill_classes


def _fields(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"device spill {label}: expected exactly {sorted(keys)}")
    return value


@dataclass(frozen=True)
class SelectedSpillContract:
    declarations_json: str

    def bind(self, adapter, banks, *, emitters, qualification_identity):
        """Admit supported classes; retain a diagnostic for unavailable routes."""
        declarations = json.loads(self.declarations_json)
        if emitters is None:
            emitters = {}
        if (type(emitters) is not dict or set(emitters) - set(declarations)
                or any(not callable(emit) for emit in emitters.values())):
            raise ValueError("device spill emitters must name declared classes and be callable")
        if qualification_identity is not None and (
            type(qualification_identity) is not str or not qualification_identity.strip()
        ):
            raise ValueError("device spill qualification requires an independently selected nonempty identity")
        emitters = dict(emitters)
        bank_map = {bank.name: bank for bank in banks}
        classes, unavailable = [], {}
        for selector, row in declarations.items():
            reason = row["unavailable"]
            if qualification_identity is None:
                reason = "missing_execution_qualification"
            elif selector not in emitters:
                reason = "missing_physical_emitter"
            elif row["width_bits"] is None or row["alignment"] is None:
                reason = "unknown_RTL_width_or_alignment"
            if reason is not None:
                unavailable[selector] = reason
                continue
            if any(name not in bank_map for name in (row["register_bank"], row["spill_bank"])):
                raise ValueError("selected device spill banks are missing from the actual allocation model")

            def emit(read, write, *, selected=selector, declared=row):
                direction = (read.space, write.space)
                routes = {(declared["register_bank"], declared["spill_bank"]): declared["store"],
                          (declared["spill_bank"], declared["register_bank"]): declared["reload"]}
                instruction = routes.get(direction)
                if (instruction is None or read.writes or not write.writes
                        or read.size != declared["width_bits"] // 8 or write.size != read.size):
                    raise ValueError("device spill request exceeds its selected class or transfer direction")
                call = emitters[selected](read, write)
                if type(call) is not Call or call.instr != instruction:
                    raise ValueError("device spill emitter returned an undeclared store/reload instruction")
                effect = adapter.project(call)
                if (effect is None or not effect.exact_copy or effect.completion != "synchronous"
                        or effect.accesses != (read, write)):
                    raise ValueError("selected device spill instruction changed exact operands or completion")
                return call

            classes.append(SpillClass(row["register_bank"], row["spill_bank"], row["width_bits"] // 8,
                                      row["alignment"], emit, qualification_identity))
        result = tuple(classes)
        validate_spill_classes(result, banks)
        return result, {"schema": "merlin.selected_device_spill_binding.v1",
                        "qualification_identity": qualification_identity,
                        "selected_classes": [name for name in declarations if name not in unavailable],
                        "unavailable_classes": unavailable,
                        "cost_policy": "longest-unused-gap; timing UNKNOWN"}


def resolve_spill_contract(declaration, facts, isa, storage):
    """Resolve explicit device roles and exact ISA transfers over selected banks."""
    row = _fields(declaration, ("schema", "classes"), "contract")
    if row["schema"] != "merlin.device_spill_contract.v1" or storage is None:
        raise ValueError("selected device spills require a versioned contract and selected storage effects")
    if type(row["classes"]) is not dict or len(row["classes"]) > 256:
        raise ValueError("device spill classes must be a bounded mapping")
    effects = json.loads(storage.declarations_json)
    banks = {window.resource: window for window in storage.banks}
    normalized, register_banks = {}, set()
    for selector, item in row["classes"].items():
        if type(selector) is not str or not selector:
            raise ValueError("device spill class selectors must be nonempty strings")
        _fields(item, ("register_bank", "spill_bank", "width_bits", "alignment", "store", "reload", "execution"),
                "class")
        if any(type(item[key]) is not str or item[key] not in banks for key in ("register_bank", "spill_bank")):
            raise ValueError("device spills require declared private device register and spill banks")
        source, destination = banks[item["register_bank"]], banks[item["spill_bank"]]
        if source.backing == destination.backing or item["register_bank"] in register_banks:
            raise ValueError("device spills require distinct physical backings and one class per register bank")
        register_banks.add(item["register_bank"])
        if type(item["execution"]) is not str or not item["execution"]:
            raise ValueError("device spill execution must be explicit or unknown")
        reason = (None if item["execution"] == "synchronous_bit_preserving_bytes"
                  else "unsupported_or_unknown_execution")
        bits = (resolve_fact_integer(item["width_bits"], facts, label="device register bits", minimum=1)
                if item["width_bits"] is not None else None)
        alignment = (resolve_fact_integer(item["alignment"], facts, label="device spill alignment", minimum=1)
                     if item["alignment"] is not None else None)
        if bits is not None and bits % 8:
            raise ValueError("device register spills require complete byte representations")
        for name, read, write in ((item["store"], source.resource, destination.resource),
                                  (item["reload"], destination.resource, source.resource)):
            isa.instruction(name)
            effect = effects.get(name)
            if effect is None or not effect["exact_copy"] or effect["nonmemory_effects"]:
                raise ValueError("device spill instructions require complete exact-copy software effects")
            if effect["completion"] != "synchronous":
                reason = "unsupported_transfer_completion"
            accesses = effect["accesses"]

            def possible(access, space):
                binding = access["space"]
                return space == binding.get("resource") or space in {c["resource"] for c in binding.get("choices", ())}

            if (len(accesses) != 2 or accesses[0]["writes"] or not accesses[1]["writes"]
                    or not possible(accesses[0], read) or not possible(accesses[1], write)):
                raise ValueError("device spill store/reload cannot realize its declared memory direction")
        normalized[selector] = {**item, "width_bits": bits, "alignment": alignment, "unavailable": reason}
    return SelectedSpillContract(json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False))
