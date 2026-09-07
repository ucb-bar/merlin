"""Extract exact command motifs from compiler-owned full-model task ranges.

This adapter is gated on a target's derived command ISA and on actual LLVM ownership attributes.
It compares instruction payloads and operand relationships rather than operation labels. The
result deliberately distinguishes instruction equivalence from timing-context equivalence: a
matching sequence does not prove matching memory contention, pointer alignment, or cache state.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation

from .activity_schedule import ActivityEvent
from .mechanism_probe import MechanismSignature, derive_mechanism_signature


@dataclass(frozen=True)
class InstructionMotif:
    signature: MechanismSignature | None
    task_index: int
    instruction_indices: tuple[int, ...]
    artifact_digest: str
    missing: tuple[str, ...]
    timing_context_missing: tuple[str, ...]
    facts: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "extracted_instruction_motif_v1",
                "signature": self.signature.to_dict() if self.signature else None,
                "task_index": self.task_index, "instruction_indices": list(self.instruction_indices),
                "artifact_digest": self.artifact_digest, "missing": list(self.missing),
                "timing_context_missing": list(self.timing_context_missing), "facts": dict(self.facts),
                "licence": "exact emitted instruction motif; timing-context qualification is separate"}


def _owner(op: Any) -> int | None:
    attr = op.attributes.get("merlin.global_task")
    return getattr(getattr(attr, "value", None), "data", None)


def _operand(value: Mapping[str, Any], bindings: Mapping[int, int]) -> dict[str, int | str]:
    kind = value.get("kind")
    if kind == "const" and isinstance(value.get("raw"), int):
        return {"kind": "const", "value": value["raw"]}
    if kind == "argbase" and value.get("arg_index") in bindings:
        return {"kind": "operand", "operand": bindings[value["arg_index"]],
                "byte_offset": value["offset"]}
    raise ValueError("unresolved or non-task instruction operand")


def extract_task_instruction_motif(*, artifact: bytes, command_buffer: Mapping[str, Any],
                                    target: str, task_index: int,
                                    parsed_module: Any = None) -> InstructionMotif:
    """Extract actual task ownership, decoded payloads, extents, def-use and capacity facts.

    Only straight-line custom-command tasks are supported. A host segment is not silently reduced
    to its inline assembly. Unknown instruction payloads, non-task pointers, inherited execution
    configuration, and unresolved store geometry refuse a signature. All opcode/flag/geometry
    constants come from the target's existing decoder and address-space derivation.
    """
    from merlin.perf.deps import rocc as dependencies
    from merlin.targetgen import address_space, memory_regime
    from merlin.targetgen.rocc import decode

    digest = hashlib.sha256(artifact).hexdigest()
    missing: list[str] = []
    context: list[str] = []
    facts: dict[str, Any] = {}

    def result(signature=None, indices=()):
        return InstructionMotif(signature, task_index, tuple(indices), digest,
                                tuple(missing), tuple(context), facts)

    plan = command_buffer.get("params", {}).get("global_program_plan", {})
    task = next((row for row in plan.get("tasks", []) if row.get("task_index") == task_index), None)
    if not task or task.get("kind") != "contraction":
        missing.append("task is absent or is not a command-only contraction")
        return result()
    module = parsed_module if parsed_module is not None else decode._parse_module(artifact.decode())
    if module is None:
        missing.append("artifact does not parse structurally")
        return result()
    owned = [op for op in module.walk() if _owner(op) == task_index]
    if not owned:
        missing.append("no LLVM operations carry this task ownership")
        return result()
    # Restrict to the address/constant plumbing understood by the actual decoder. Host arithmetic,
    # loads, stores or control flow would change the timed task and must remain visible.
    allowed = {"llvm.inline_asm", "llvm.mlir.constant", "llvm.ptrtoint", "llvm.add"}
    unexpected = sorted({op.name for op in owned if op.name not in allowed})
    if unexpected:
        missing.append(f"task contains non-command work: {unexpected}")
        return result()
    trace = decode.decode_module(module, target=target)
    asm = [op for op in module.walk() if op.name == "llvm.inline_asm"]
    indices = [i for i, op in enumerate(asm) if _owner(op) == task_index]
    rows = [trace["instructions"][i] for i in indices]
    if not rows:
        missing.append("task emits no decoded instructions")
        return result()
    if indices != list(range(indices[0], indices[-1] + 1)):
        missing.append("task instruction ownership is not contiguous")
    if any(row["class"] == "UNKNOWN" for row in rows):
        missing.append("task contains unknown instruction encoding")
    isa = decode.isa_constants(target)
    config_classes = set(isa.get("CONFIG_SUBTYPE", {}).values())
    emitted_classes = {row["class"] for row in rows}
    if not config_classes or not config_classes.issubset(emitted_classes):
        missing.append("task inherits configuration; standalone configuration equivalence not established")
    before = trace["instructions"][:indices[0]]
    after = trace["instructions"][indices[-1] + 1:]
    opened = rows[0]["class"] == "FENCE" or bool(before and before[-1]["class"] == "FENCE")
    closed = rows[-1]["class"] == "FENCE" or bool(after and after[0]["class"] == "FENCE")
    facts["device_boundary_fences"] = {"entry": opened, "exit": closed}
    if not opened or not closed:
        context.append("task lacks an observed device-drained entry/exit boundary")
    context.extend(("runtime operand base alignment and memory-bank mapping are not bound",
                    "external memory/cache state and concurrent port traffic are not bound"))

    names = list(dict.fromkeys([*task.get("reads", []), *task.get("writes", [])]))
    abi = command_buffer.get("kernel_abi", {}).get("args", [])
    bindings = {i: names.index(arg["tensor"]) for i, arg in enumerate(abi)
                if arg.get("tensor") in names}
    semantics = []
    try:
        for row in rows:
            item = {"class": row["class"]}
            if row.get("funct") is not None:
                item["selector"] = row["funct"]
                item["rs1"] = _operand(row["rs1"], bindings)
                item["rs2"] = _operand(row["rs2"], bindings)
            semantics.append(item)
    except ValueError as error:
        missing.append(str(error))
    instructions, effects = dependencies.instructions_and_effects(
        {"instructions": rows}, flag_masks=dependencies.flag_masks_for(target))
    unresolved = [reason for effect in effects for reason in effect.unresolved]
    missing.extend(unresolved)
    # Preserve actual physical addresses and every mode bit in the instruction payload. The footprint
    # report is conservative (all touched rows), so it cannot claim liveness-aware reuse or overlap.
    touched: dict[str, set[int]] = {}
    for effect in effects:
        for access in (*effect.defs, *effect.uses):
            if access.file == "spad" and access.slot == decode.GARBAGE:
                continue  # the decoded ABI's retain-staged-operand sentinel is not a store row
            slot = access.slot
            if access.file == "acc":
                namespace = isa.get("ACC_I8")
                if not isinstance(namespace, int):
                    missing.append("accumulator namespace tag not derived")
                    continue
                slot &= ~namespace
            touched.setdefault(access.file, set()).add(slot)
    for row in rows:
        decoded = row.get("decoded", {})
        if "c_addr" not in decoded:
            continue
        payload = row.get("rs2", {})
        if payload.get("kind") != "const":
            missing.append("accumulator tile extent not decoded")
            continue
        extent = decode._pack_fields(payload["raw"])
        mode_bits = isa["ACC_I8"] | isa["ACC_ACCUM"] | isa["FULL_C_BIT"]
        base = decoded["c_addr"] & ~mode_bits
        touched.setdefault("acc", set()).update(range(base, base + extent["rows"]))
    space = address_space.derive_address_space(target)
    stores = [store for store in space.stores if store.row_bytes and store.total_rows]
    stores.sort(key=lambda store: store.row_bytes)
    if len(stores) != 2 or stores[0].row_bytes == stores[1].row_bytes:
        missing.append("decoded operand/accumulator files cannot be uniquely bound to physical stores")
        return result(indices=indices)
    files = {"spad": stores[0], "acc": stores[1]}
    capacity = {}
    for file, slots in touched.items():
        store = files.get(file)
        if store is None:
            missing.append(f"no physical store for decoded state file {file}")
            continue
        # The highest addressed row checks address validity; touched row count alone misses holes.
        if slots and (min(slots) < 0 or max(slots) >= store.total_rows):
            missing.append(f"decoded {file} address exceeds derived physical capacity")
        capacity[store.name] = memory_regime.classify(len(slots), len(slots), store.total_rows)
    facts["physical_footprint"] = {file: {"touched_rows": len(slots),
                                          "max_row": max(slots, default=None)}
                                   for file, slots in touched.items()}
    facts["physical_stores"] = [store.to_dict() for store in stores]
    facts["def_use"] = [{"defs": [(a.file, a.slot) for a in effect.defs],
                         "uses": [(a.file, a.slot) for a in effect.uses]}
                        for effect in effects]
    tensors = command_buffer.get("tensors", {})
    dtype = stores[0].element_dtype
    if not dtype:
        missing.append("operand dtype not derived from the physical store")
    reps = []
    for name in names:
        tensor = tensors.get(name, {})
        current_dtype = tensor.get("dtype")
        if name in task.get("reads", []) and current_dtype != dtype:
            missing.append("task operand dtype disagrees with derived command datapath")
        if not current_dtype:
            missing.append("task tensor dtype is absent")
            continue
        reps.append(ValueRepresentation("external-pointer", "decoded-stride-and-offset",
                                        current_dtype, "exact-decoded-command-payload",
                                        quantization="decoded-config-state"))
    extents = sorted({(row.get("decoded", {}).get("rows"), row.get("decoded", {}).get("cols"))
                      for row in rows if "rows" in row.get("decoded", {})})
    facts["transfer_extents"] = [list(extent) for extent in extents]
    if not extents or any(not isinstance(v, int) or v <= 0 for extent in extents for v in extent):
        missing.append("transfer extents are not completely decoded")
    if missing:
        return result(indices=indices)
    movement_bytes = 0
    movement_commands = 0
    for row in rows:
        decoded = row.get("decoded", {})
        if "dram" not in decoded:
            continue
        bits = address_space.element_bits(decoded.get("readout") or dtype)
        if bits is None or "rows" not in decoded or "cols" not in decoded:
            missing.append("physical transfer byte volume cannot be derived")
            return result(indices=indices)
        movement_bytes += decoded["rows"] * decoded["cols"] * bits / 8
        movement_commands += 1
    signature = derive_mechanism_signature(
        representations=reps,
        events=[ActivityEvent("task", "whole_motif", "compute", 0,
                              movement_bytes=movement_bytes, movement_commands=movement_commands)],
        capacity_regime=capacity, tile_shape=[v for extent in extents for v in extent],
        edge_cases=[f"exact-transfer-extent:{r}x{c}" for r, c in extents],
        repetition_semantics="one complete issued command motif; no inter-motif overlap inferred",
        instruction_semantics=semantics)
    return result(signature, indices)


def compare_instruction_motifs(model: InstructionMotif, probe: InstructionMotif) -> dict[str, Any]:
    matched = (model.signature is not None and probe.signature is not None
               and model.signature == probe.signature)
    context = sorted(set(model.timing_context_missing + probe.timing_context_missing))
    return {"instruction_equivalent": matched,
            "timing_calibration_admissible": matched and not context,
            "missing": sorted(set(model.missing + probe.missing)),
            "timing_context_missing": context,
            "model_artifact_digest": model.artifact_digest,
            "probe_artifact_digest": probe.artifact_digest,
            "licence": "matching instruction sequences alone do not establish matching elapsed cycles"}


def initialized_compute_primitives(trace: Mapping[str, Any], *, target: str) -> list[dict[str, Any]]:
    """Find overwrite compute pairs with explicit operand initialization and execution config.

    These small motifs can calibrate an isolated compute engine only after the probe runner drains
    initialization outside its measured interval. The model's producer issue order proves data
    dependence, not completion or absence of queued memory traffic. Consequently this extraction
    never prices an in-context task or grants overlap. Retained-weight and accumulate-onto pairs
    are refused here until their initial state has its own witnessed initialization.
    """
    from merlin.perf.deps.rocc import INHERITS_DESTINATION
    from merlin.perf.mechanism_probe import _digest
    from merlin.targetgen.rocc import decode

    isa = decode.isa_constants(target)
    config_classes = set(isa.get("CONFIG_SUBTYPE", {}).values())
    rows = trace.get("instructions", [])
    latest_execution = None
    live: dict[int, dict[str, Any]] = {}
    out = []
    for index, row in enumerate(rows):
        cls = row.get("class")
        payload = row.get("decoded", {})
        known_state_effect = (
            cls in config_classes or cls in INHERITS_DESTINATION
            or cls in set(INHERITS_DESTINATION.values()) or cls == "FENCE"
            or "spad_addr" in payload or "acc_addr" in payload)
        if not known_state_effect:
            # An instruction the adapter cannot model may overwrite staging or configuration.
            # Keeping the old state would manufacture initialization evidence across a blind spot.
            live.clear()
            latest_execution = None
        # The existing structural decoder exposes subtype EX, independent of selector encoding.
        if cls in config_classes and payload.get("subtype") == "EX":
            try:
                latest_execution = {"selector": row["funct"],
                                    "rs1": _operand(row["rs1"], {}),
                                    "rs2": _operand(row["rs2"], {})}
            except ValueError:
                latest_execution = None
        if "spad_addr" in payload:
            address, count = payload["spad_addr"], payload.get("rows")
            if isinstance(address, int) and isinstance(count, int) and count > 0:
                for slot in range(address, address + count):
                    live[slot] = {"writer_index": index, "base": address,
                                  "rows": count, "cols": payload.get("cols")}
        stager = INHERITS_DESTINATION.get(cls)
        if stager is None or index == 0:
            continue
        prior = rows[index - 1]
        previous = prior.get("decoded", {})
        missing = []
        if prior.get("class") != stager:
            missing.append("compute is not adjacent to its decoded destination stager")
        if previous.get("accumulate") is not False:
            missing.append("accumulator initial value is required but not established")
        if latest_execution is None:
            missing.append("active execution configuration is unresolved")
        a, b = payload.get("a_spad"), previous.get("weight_spad")
        inputs = []
        for operand, address in (("activation", a), ("weight", b)):
            if address == decode.GARBAGE:
                missing.append(f"{operand} uses retained internal state without an initialization witness")
                continue
            state = live.get(address)
            if state is None:
                missing.append(f"{operand} staging address has no observed producer")
                continue
            # Verify every row of the named loaded tile still refers to the same producer.
            if any(live.get(slot) != state for slot in range(address, address + state["rows"])):
                missing.append(f"{operand} staging tile was partially overwritten")
                continue
            inputs.append({"operand": operand, "address": address, "rows": state["rows"],
                           "cols": state["cols"], "producer_index": state["writer_index"]})
        semantics = []
        for command in (prior, row):
            try:
                semantics.append({"class": command["class"], "selector": command["funct"],
                                  "rs1": _operand(command["rs1"], {}),
                                  "rs2": _operand(command["rs2"], {})})
            except ValueError:
                missing.append("primitive payload contains unresolved or external operands")
        # Writer index is provenance, not hardware semantics: different program positions may
        # initialize exactly the same operand extents and physical addresses.
        domain = {"target_isa_digest": _digest(isa),
                  "execution_configuration": latest_execution, "instructions": semantics,
                  "initialized_operands": [{k: v for k, v in entry.items() if k != "producer_index"}
                                           for entry in inputs],
                  "accumulator_initialization": "overwrite",
                  "calibration_context": "operand producers completed; all other engines drained"}
        if any(entry["cols"] is None for entry in inputs):
            missing.append("initialized operand extent is unresolved")
        out.append({"schema": "initialized_compute_primitive_v1",
                    "instruction_indices": [index - 1, index],
                    "domain": domain, "domain_digest": _digest(domain) if not missing else None,
                    "initialization_provenance": inputs, "missing": missing,
                    "calibration_admissible": False,
                    "required_probe_instrumentation": [
                        "finish operand initialization before measurement",
                        "drain all competing engines before measurement",
                        "measure only the primitive and its completion using warm 1 + measured 1"],
                    "unpriced_behavior": [
                        "input-value dependent timing, unless excluded by target facts or bounded by probes",
                        "in-context memory and execution contention"],
                    "in_context_cycles": None,
                    "licence": "isolated engine primitive; queued producer/neighbor contention remains UNKNOWN"})
    return out
