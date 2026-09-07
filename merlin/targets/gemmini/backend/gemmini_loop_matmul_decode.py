"""Pinned LoopMatmul register observations, not loop semantic equivalence.

This edge names the target's state and queued command ports. Every selector,
field offset and width is recovered from host-pinned facts/elaborated hardware.
No candidate descriptor or header packing expression supplies the layout.
"""
from __future__ import annotations

import hashlib
import json

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
