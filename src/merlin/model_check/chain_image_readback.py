"""Decode complete-chain transport without granting placement/hardware credit."""

from __future__ import annotations

import json
import math
import struct

from merlin.common import invocation_record
from merlin.llvmlower.c_runtime import DT_BYTES
from merlin.llvmlower.compilation_recipe import verify_completed_recipe
from merlin.runtime.execution_memory import MemoryMapBinding, MemoryReservation, admit_execution_memory
from merlin.runtime.out_bin import binary_console_diagnostics, parse_binary_console_details

from .chain_group_profile import reconcile as reconcile_profiles
from .chain_group_profile import separate as separate_profiles
from .chain_image_counter import verify as verify_counter_support
from .chain_image_objects import payload, reopen
from .chain_image_profile import verify as verify_profile_support


def verify_image(record):
    image = json.loads(payload(record))
    if (
        image.get("schema") != "merlin.post_freeze.firesim.chain_image.v1"
        or image.get("execution_scope") != "complete_chain"
    ):
        raise ValueError("not a complete-chain image manifest")
    verify_completed_recipe(
        reopen(image["recipe"]), executable=reopen(image["boot_elf"]), expected_recipe_sha256=image["recipe"]["sha256"]
    )
    for field in ("selection", "workload_contract", "producer"):
        reopen(image[field])
    selection = json.loads(payload(image["selection"]))
    declared_map = selection["memory_map"]
    binding = MemoryMapBinding(reopen(declared_map), declared_map["sha256"], selection["execution_identity"])
    memory = admit_execution_memory(
        reopen(image["boot_elf"]),
        binding,
        (MemoryReservation("activation_arena", **selection["layout"]["arena"], alignment=64),),
    )
    if memory != image["memory_admission"]:
        raise ValueError("complete-chain execution memory admission changed")
    invocation_record.verify(reopen(image["boot"]["invocation"]))
    for record in (*image["plan"]["source_pins"], *image["reference"]["source_pins"], *image["producer_sources"]):
        reopen(record)
    for program in image["namespace"]:
        for record in (*program["objects"], program["mapping"]):
            reopen(record)
        for record in program["invocations"]:
            invocation_record.verify(reopen(record))
    for record in image.get("group_profile", {}).get("support", []):
        verify_profile_support(record)
    verify_counter_support(image.get("counter_support"))
    reopen(image["plan"]["input_blob"])
    reopen(image["reference"]["blob"])
    expected_steps = [{"program": p["index"], "step": i} for p in image["plan"]["programs"] for i in range(p["steps"])]
    if [{k: row[k] for k in ("program", "step")} for row in image["plan"]["schedule"]] != expected_steps:
        raise ValueError("chain manifest does not schedule its complete declared program roster")
    roster = [
        {"stage": p["name"], "step": i, "index": n, "shape": shape, "dtype": dtype}
        for p in image["plan"]["programs"]
        for i in range(p["steps"])
        for n, (shape, dtype) in enumerate(p["outputs"])
    ]
    if roster != image["output_roster"] or len(expected_steps) != image["chain_steps"]:
        raise ValueError("chain manifest output roster differs from its actual declared schedule")
    return image


def _integer(token):
    value = int(token)
    if str(value) != token or not 0 <= value < 1 << 64:
        raise ValueError("noncanonical chain counter/extent")
    return value


def _hex(token):
    if len(token) != 16 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("noncanonical chain bit-pattern/digest")
    return int(token, 16)


def _double(token):
    value = struct.unpack("<d", struct.pack("<Q", _hex(token)))[0]
    if math.isnan(value) or value < 0:
        raise ValueError("invalid nonnegative chain error summary")
    return value if math.isfinite(value) else None


def _fnv(data):
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return value


def decode(image, console):
    """Require every scheduled counter/array exactly once in temporal order.

    Compact acceptance comes from the pinned guest's all-element comparison,
    never digest/sample equality. This pure decoder supplies no execution origin.
    The caller must bind the actually consumed image and native placement gates.
    """
    if image["transport"] == "timing_only_v1":
        from .chain_image_timing import decode_timing

        return decode_timing(image, console)
    compact = image["transport"] == "all_element_comparison_v1"
    if not compact and image["transport"] != "full_arrays_v1":
        raise ValueError("unknown chain readback transport")
    if compact:
        if b"OUT_BIN_" in console:
            raise ValueError("mixed complete-chain transports")
        text, frames, outputs = console, {}, {}
    else:
        outputs, _, frames = parse_binary_console_details(console)
        text = binary_console_diagnostics(console)
    ordinary, profiles = separate_profiles(text.decode("ascii"))
    rows = [line.split() for line in ordinary.splitlines() if line.startswith("CHAIN_") or line == "DONE"]
    at, comparisons, cycles, mismatches = 0, [], [], 0
    reference = reopen(image["reference"]["blob"])
    with reference.open("rb") as expected:
        for ordinal, step in enumerate(image["plan"]["schedule"]):
            program, index = image["plan"]["programs"][step["program"]], step["step"]
            if at >= len(rows):
                raise ValueError("truncated complete-chain step transport")
            row = rows[at]
            at += 1
            if (
                row[:2] != ["CHAIN_STEP", "v1"]
                or len(row) != 6
                or list(map(_integer, row[2:4])) != [step["program"], index]
            ):
                raise ValueError("missing/duplicate/reordered chain step counter")
            cycles.append(
                {
                    "program": program["name"],
                    "step": index,
                    "cycles": _integer(row[4]),
                    "memref_rank_mismatches": _integer(row[5]),
                }
            )
            for output, (shape, dtype) in enumerate(program["outputs"]):
                elements = math.prod(shape)
                binding = {"stage": program["name"], "step": index, "index": output, "dtype": dtype, "shape": shape}
                if compact:
                    if at >= len(rows):
                        raise ValueError("truncated complete-chain array transport")
                    row = rows[at]
                    at += 1
                    if (
                        len(row) < 13
                        or row[:2] != ["CHAIN_ARRAY", "v1"]
                        or list(map(_integer, row[2:6])) != [step["program"], index, output, elements]
                    ):
                        raise ValueError("missing/duplicate/reordered compact array")
                    failed, finite, samples = _integer(row[6]), _integer(row[7]), _integer(row[12])
                    if (
                        failed > elements
                        or finite not in (0, 1)
                        or (not finite and not failed)
                        or samples != min(failed, 4)
                        or len(row) != 13 + 3 * samples
                    ):
                        raise ValueError("inconsistent complete-array comparison summary")
                    locations = list(map(_integer, row[13::3]))
                    if len(set(locations)) != len(locations) or any(i >= elements for i in locations):
                        raise ValueError("invalid compact mismatch sample positions")
                    sample_values = [(_hex(row[i]), _hex(row[i + 1])) for i in range(14, len(row), 3)]
                    if any(max(pair) >= 1 << (8 * DT_BYTES[dtype]) for pair in sample_values):
                        raise ValueError("compact sample exceeds its declared storage width")
                    expected.seek(image["reference"]["offsets"][ordinal][output])
                    payload = expected.read(elements * DT_BYTES[dtype])
                    if len(payload) != elements * DT_BYTES[dtype] or _hex(row[11]) != _fnv(payload):
                        raise ValueError("guest compared a different selected reference array")
                    maximum, square = _double(row[8]), _double(row[9])
                    comparisons.append(
                        {
                            **binding,
                            "elements": elements,
                            "mismatched": failed,
                            "finite": bool(finite),
                            "max_abs_error": maximum if finite else None,
                            "rms_error": math.sqrt(square / elements) if finite and square is not None else None,
                            "actual_fnv1a64": row[10],
                            "reference_fnv1a64": row[11],
                            "samples": [
                                {"index": i, "actual_bits": a, "reference_bits": b}
                                for i, (a, b) in zip(locations, sample_values, strict=True)
                            ],
                            "pass": failed == 0 and finite == 1,
                        }
                    )
                    _hex(row[10])
                    mismatches += failed
                else:
                    name = f"p{step['program']}_s{index}_out{output}"
                    if name not in frames:
                        raise ValueError("complete-chain raw output array is missing")
                    frame = frames[name]
                    columns = shape[-1] if shape else 1
                    if (frame.rows, frame.cols, frame.word_bytes, frame.signed) != (
                        elements // columns,
                        columns,
                        DT_BYTES[dtype],
                        dtype in {"i8", "i16", "i32", "i64"},
                    ):
                        raise ValueError("chain raw output has another storage ABI")
                    comparisons.append({**binding, "elements": elements, "frame": name, "pass": None})
    if (
        len(rows) != at + 2
        or rows[at] != ["CHAIN_END", "v1", str(len(cycles)), str(len(comparisons)), str(mismatches)]
        or rows[-1] != ["DONE"]
    ):
        raise ValueError("complete-chain terminal counts are absent or inconsistent")
    if not compact and set(outputs) != {r["frame"] for r in comparisons}:
        raise ValueError("full readback added or omitted an array")
    group_profiles = reconcile_profiles(
        profiles, image["plan"]["schedule"], cycles,
        enabled=image.get("group_profile", {}).get("enabled", False),
    )
    return {
        "schema": "merlin.chain_image.readback.v1",
        "transport": image["transport"],
        "complete_chain_steps": len(cycles),
        "output_array_count": len(comparisons),
        "comparisons": comparisons,
        "step_cycles": cycles,
        **({"group_profiles": group_profiles} if image.get("group_profile", {}).get("enabled", False) else {}),
        "runtime_status": "pass" if all(row["memref_rank_mismatches"] == 0 for row in cycles) else "fail",
        "numerical_status": ("pass" if not mismatches else "fail") if compact else "pending_all_output_checker",
        "execution_origin_verified": False,
        "scientific_qualification": False,
        "scope": "Pinned guest transport only; actual consumed-image/hardware/source-placement gates remain separate.",
    }
