"""Pinned LoopMatmul register observations, not loop semantic equivalence.

This edge names the target's state and queued command ports. Every selector,
field offset and width is recovered from host-pinned facts/elaborated hardware.
No candidate descriptor or header packing expression supplies the layout.
"""
from __future__ import annotations

import hashlib
import json

from merlin.targetgen.address_space import derive_address_space
from merlin.targetgen.rtl.register_slices import (
    decode_register_slices, derive_register_slices, register_definitions,
)

_FIELDS = frozenset((
    "max_i", "max_j", "max_k", "pad_i", "pad_j", "pad_k",
    "a_dram_addr", "b_dram_addr", "c_dram_addr", "d_dram_addr",
    "a_dram_stride", "b_dram_stride", "c_dram_stride", "d_dram_stride",
    "a_transpose", "b_transpose", "full_c", "low_d", "ex_accumulate", "act",
    "a_ex_spad_id", "b_ex_spad_id", "inc_acc_addr", "spad_only", "c_spad_addr",
    "lda_started", "ldb_started", "ldd_started", "ex_started", "st_started",
))


def _physical_type(store) -> dict | None:
    """Translate one RTL-derived store element type without guessing signedness."""
    dtype, width = store.element_dtype, store.element_bits
    if not isinstance(dtype, str) or type(width) is not int or width <= 0:
        return None
    if dtype.startswith("i"):
        encoding = "signed_integer"
    elif dtype.startswith("u"):
        encoding = "unsigned_integer"
    elif dtype.startswith("f"):
        encoding = "floating_point"
    else:
        return None
    return {"encoding": encoding, "width_bits": width, "rtl_dtype": dtype}


def _readout_types(facts: dict) -> dict:
    """Widths of the two LoopMatmul store modes, derived from this facts artifact."""
    body = facts.get("facts") if isinstance(facts, dict) else None
    target = body.get("target") if isinstance(body, dict) else None
    if not isinstance(target, str) or not target:
        return {"full_c_0": None, "full_c_1": None,
                "status": "UNKNOWN", "reason": "facts do not identify their target"}
    space = derive_address_space(target, facts=facts)
    narrow, full = space.store("scratchpad"), space.store("accumulator")
    narrow_type = _physical_type(narrow) if narrow is not None else None
    full_type = _physical_type(full) if full is not None else None
    status = "derived" if narrow_type is not None and full_type is not None else "UNKNOWN"
    return {
        "full_c_0": narrow_type,
        "full_c_1": full_type,
        "status": status,
        "reason": (None if status == "derived" else
                   "operand/accumulator store element types are not both grounded in RTL facts"),
        "sources": dict(space.sources),
    }


def derive_layouts(*, facts_text: str, hardware_text: str) -> dict:
    facts = json.loads(facts_text)
    hardware_sha = hashlib.sha256(hardware_text.encode()).hexdigest()
    if hardware_sha != facts["inputs"]["core_hw_sha256"]:
        raise ValueError("elaborated hardware does not match host facts")
    tables = [row for row in facts["facts"]["interfaces"] if row.get("name") == "funct_decode_table"]
    if len(tables) != 1:
        raise ValueError("ambiguous instruction selector table")
    table = tables[0]
    definitions = register_definitions(hardware_text, "LoopMatmul")
    registers = []
    labels = {}
    for ref, body in definitions.items():
        if not ref.startswith("%loops_") or not body.startswith("seq.firreg "):
            continue
        slot, sep, field = ref.removeprefix("%loops_").partition("_")
        if sep and slot.isdigit() and field in _FIELDS:
            registers.append(ref)
            labels[ref] = {"slot": int(slot), "field": field}
    if not registers:
        raise ValueError("no selected loop descriptor registers in hardware")
    # Port widths must come from actual elaborated operands, not machine word defaults.
    inputs = {}
    for ref, label in (("%cmd_q.io_deq_bits_cmd_rs1", "rs1"), ("%cmd_q.io_deq_bits_cmd_rs2", "rs2")):
        widths = set()
        for body in definitions.values():
            if body.startswith("comb.extract " + ref + " from "):
                widths.add(int(body.rsplit(" : (i", 1)[1].split(")", 1)[0]))
        if len(widths) != 1:
            raise ValueError("unknown or inconsistent queued operand width")
        inputs[ref] = (label, widths.pop())
    layouts = {}
    for code, name in table["names"].items():
        if name != "LOOP_WS" and not name.startswith("LOOP_WS_CONFIG_"):
            continue
        selector = int(code)
        if selector not in table["legal_funct"]:
            raise ValueError("header-only selector is not an RTL fact")
        layout = derive_register_slices(hardware_text, module="LoopMatmul", registers=registers,
            selector="%cmd_q.io_deq_bits_cmd_inst_funct", selector_value=selector, inputs=inputs)
        for row in layout["registers"]:
            row.update(labels[row["register"]])
        layouts[str(selector)] = {"instruction_name": name, "layout": layout}
    return {"schema": "loop_matmul_register_layouts_v1", "layouts": layouts,
        "binding": {"facts_sha256": hashlib.sha256(facts_text.encode()).hexdigest(), "hardware_sha256": hardware_sha},
        "physical_readout_types": _readout_types(facts),
        "selected_fields": sorted(_FIELDS),
        "absent_selected_fields": sorted(_FIELDS - {item["field"] for item in labels.values()}),
        "unexamined_state_and_semantics": ["is_resadd global state", "on-chip initial address state",
            "stride units and generated DMA address arithmetic", "accumulator arithmetic and readout",
            "command acceptance and inter-loop sequencing"],
        "scope": "conditional selected register updates; unknown controls overapproximated",
        "descriptor_arithmetic_equivalence": "UNKNOWN", "sequencer_progress": "UNPROVEN"}


def observe_instructions(instructions: list[dict], *, layouts: dict) -> dict:
    """Observe actual decoded SSA operands, keeping pointer ranges unresolved."""
    if layouts.get("schema") != "loop_matmul_register_layouts_v1":
        raise ValueError("unsupported loop register layout")
    observations = []
    for instruction in instructions:
        entry = layouts["layouts"].get(str(instruction.get("funct")))
        if entry is None:
            continue
        observations.append({"instruction_index": instruction["index"],
            "instruction_name": entry["instruction_name"],
            "fields": decode_register_slices(entry["layout"],
                {"rs1": instruction.get("rs1", {}), "rs2": instruction.get("rs2", {})})})
    return {"schema": "loop_matmul_register_observations_v1", "binding": layouts["binding"],
        "instructions": observations, "address_ranges_validated": False,
        "descriptor_arithmetic_equivalence": "UNKNOWN", "timing_calibration_admissible": False}


def _consensus_field(fields: list[dict], name: str) -> tuple[dict | int | None, str | None]:
    """Recover a field when every possible descriptor slot receives the same exact value."""
    selected = [row for row in fields if row.get("field") == name]
    if not selected:
        return None, f"hardware layout exposes no {name} field"
    values = []
    for row in selected:
        if row.get("status") != "derived":
            return None, f"{name} update is not derived for every possible descriptor slot"
        if type(row.get("value")) is int:
            value: dict | int = row["value"]
        elif isinstance(row.get("symbolic_value"), dict):
            symbolic = row["symbolic_value"]
            argument, byte_offset = symbolic.get("argument"), symbolic.get("byte_offset")
            if type(argument) is not int or argument < 0 or type(byte_offset) is not int:
                return None, f"{name} symbolic destination is incomplete"
            value = {"arg_index": argument, "byte_offset": byte_offset}
        else:
            return None, f"{name} emitted value is unresolved"
        values.append(value)
    canonical = {json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values}
    if len(canonical) != 1:
        return None, f"{name} differs across possible descriptor slots"
    return values[0], None


def derive_writebacks(instructions: list[dict], *, layouts: dict) -> dict:
    """Recover fused-loop DRAM writebacks from exact emitted operands and exact RTL layouts.

    A CONFIG_ADDRS_DC observation supplies the target C address and the following LOOP_WS supplies
    the physical ``full_c`` selector.  Both are accepted only when the RTL slice derivation gives the
    same value for every possible hardware descriptor slot.  This avoids assuming which slot the
    sequencer allocated while still proving the representation of every emitted loop writeback.
    """
    observed = observe_instructions(instructions, layouts=layouts)
    readout_types = layouts.get("physical_readout_types")
    readout_types = readout_types if isinstance(readout_types, dict) else {}
    latest_destination: dict | int | None = None
    latest_destination_error: str | None = "no preceding C-address descriptor"
    writebacks = []
    unresolved = []
    covered = []
    for instruction in observed["instructions"]:
        index = instruction["instruction_index"]
        name = instruction["instruction_name"]
        covered.append(index)
        if name == "LOOP_WS_CONFIG_ADDRS_DC":
            latest_destination, latest_destination_error = _consensus_field(
                instruction["fields"], "c_dram_addr")
            continue
        if name != "LOOP_WS":
            continue
        full_c, full_c_error = _consensus_field(instruction["fields"], "full_c")
        if full_c_error is not None or latest_destination_error is not None:
            unresolved.append({"instruction_index": index,
                               "reason": full_c_error or latest_destination_error})
            continue
        if type(latest_destination) is int:
            if latest_destination != 0:
                unresolved.append({"instruction_index": index,
                                   "reason": "C address is a non-null absolute value, not a kernel ABI argument"})
            # A null C address suppresses the LoopMatmul store in the elaborated controller.
            continue
        if not isinstance(latest_destination, dict) or full_c not in (0, 1):
            unresolved.append({"instruction_index": index,
                               "reason": "destination or physical full_c selector is unresolved"})
            continue
        physical_type = readout_types.get(f"full_c_{full_c}")
        if readout_types.get("status") != "derived" or not isinstance(physical_type, dict):
            unresolved.append({"instruction_index": index,
                               "reason": readout_types.get("reason") or
                               "physical readout type is not derived from RTL facts"})
            continue
        writebacks.append({
            "instruction_index": index,
            "instruction_name": name,
            "destination": dict(latest_destination),
            "physical_readout": {
                "encoding": physical_type["encoding"],
                "width_bits": physical_type["width_bits"],
            },
            "selector": {"field": "full_c", "value": full_c},
        })
    return {
        "schema": "loop_matmul_writeback_evidence_v1",
        "binding": dict(layouts["binding"]),
        "covered_instruction_indices": covered,
        "writebacks": writebacks,
        "unresolved_writebacks": unresolved,
        "coverage_status": "complete" if not unresolved else "UNKNOWN",
        "address_ranges_validated": False,
        "scope": "emitted descriptor representation and physical output width; execution timing excluded",
    }
