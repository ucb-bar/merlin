"""Target-neutral diagnostics over one emitted whole-program command buffer.

This module reports what the compiler actually declared about representations and activity.  It
does not infer a hardware timeline from command order: command buffers do not state resource
occupancy or overlap, so those fields remain explicitly UNKNOWN until an adapter supplies events or
a warm measured run supplies counters.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def representation_activity(command_buffer: Mapping[str, Any]) -> dict[str, Any]:
    """Inventory boundary encodings and representation-affecting commands without guessing."""
    raw_tensors = command_buffer.get("tensors")
    raw_commands = command_buffer.get("commands")
    tensors = raw_tensors if isinstance(raw_tensors, Mapping) else {}
    commands = (raw_commands if isinstance(raw_commands, Sequence)
                and not isinstance(raw_commands, (str, bytes)) else ())

    tensor_rows: dict[str, dict[str, Any]] = {}
    for raw_name, raw_spec in sorted(tensors.items(), key=lambda item: str(item[0])):
        name = str(raw_name)
        spec = _mapping(raw_spec)
        physical_present = "physical" in spec
        tensor_rows[name] = {
            "dtype": spec.get("dtype"),
            "shape": list(spec.get("shape") or ()) if isinstance(spec.get("shape"), Sequence)
            and not isinstance(spec.get("shape"), (str, bytes)) else None,
            "role": spec.get("role"),
            "physical": spec.get("physical") if physical_present else None,
            "physical_status": "declared" if physical_present else "UNKNOWN",
        }

    opcodes: Counter[str] = Counter()
    directives: list[dict[str, Any]] = []
    malformed: list[str] = []
    for index, raw_command in enumerate(commands):
        if not isinstance(raw_command, Mapping):
            malformed.append(f"commands[{index}] is not a mapping")
            continue
        opcode = str(raw_command.get("opcode") or "")
        if not opcode:
            malformed.append(f"commands[{index}] has no opcode")
            continue
        opcodes[opcode] += 1
        attributes = _mapping(raw_command.get("attributes"))
        represented = {key: attributes[key] for key in (
            "layout", "output_dtype", "epilogue", "acc_scale", "requant_shift")
                       if key in attributes}
        if represented:
            directives.append({
                "index": index,
                "opcode": opcode,
                "operands": dict(sorted(
                    (str(key), str(value))
                    for key, value in _mapping(raw_command.get("operands")).items())),
                "attributes": represented,
            })

    missing: list[str] = []
    if not isinstance(raw_tensors, Mapping):
        missing.append("tensor declarations are absent")
    if not (isinstance(raw_commands, Sequence)
            and not isinstance(raw_commands, (str, bytes))):
        missing.append("command sequence is absent")
    missing.extend(malformed)
    missing.extend((
        "command-buffer order does not declare per-resource busy intervals",
        "encoding directives do not prove the emitted instruction stream performed each conversion",
    ))
    params = _mapping(command_buffer.get("params"))
    raw_placement = params.get("lane_placement")
    placement_rows = (raw_placement if isinstance(raw_placement, Sequence)
                      and not isinstance(raw_placement, (str, bytes)) else ())
    lane_counts: Counter[str] = Counter()
    family_lane_counts: Counter[str] = Counter()
    lane_sequence: list[str] = []
    for raw in placement_rows:
        if not isinstance(raw, Mapping):
            continue
        lane = str(raw.get("lane") or "UNKNOWN")
        family = str(raw.get("family") or "UNKNOWN")
        lane_counts[lane] += 1
        family_lane_counts[f"{family}:{lane}"] += 1
        lane_sequence.append(lane)
    lane_transitions = sum(left != right for left, right in zip(lane_sequence, lane_sequence[1:]))
    declined = command_buffer.get("declined")
    declined = declined if isinstance(declined, Mapping) else None
    return {
        "schema": "command_buffer_representation_activity_v1",
        "tensors": tensor_rows,
        "command_counts": dict(sorted(opcodes.items())),
        "representation_directives": directives,
        "representation_directive_count": len(directives),
        "occupancy": {
            "status": "UNKNOWN",
            "compute_utilization": None,
            "movement_compute_overlap_cycles": None,
            "latency_hiding_efficiency": None,
            "reason": "command-buffer order has no resource timeline; use a target event adapter or warm counters",
        },
        "emitted_encoding_transitions": {
            "status": "UNKNOWN",
            "count": None,
            "declared_directives": len(directives),
            "reason": "verify conversions in the emitted instruction trace; declarations are intent, not execution",
        },
        "placement": {
            "status": "declared" if placement_rows else "UNKNOWN",
            "region_count": sum(lane_counts.values()) if placement_rows else None,
            "lane_counts": dict(sorted(lane_counts.items())),
            "family_lane_counts": dict(sorted(family_lane_counts.items())),
            "adjacent_lane_transitions": lane_transitions if placement_rows else None,
            "note": ("lane transitions are structural placement boundaries, not measured copies or "
                     "cycles"),
        },
        "lowering": {
            "status": "declined" if declined else "emitted",
            "declined": ({key: declined.get(key) for key in ("op", "reason", "shape")
                          if key in declined} if declined else None),
        },
        "missing": missing,
    }
