"""Strict single-ELF Tiny semantic segment admission before device creation."""
import hashlib
import json
import struct

from bundle import runtime_program, verify_kernel_receipt
from placement import placement
from segment_boundary import SegmentChain
import kernel_library  # noqa: E402,F401
from atlas_program import assembly
from chipyard.arena import DEFAULT_ARENA, relocate_assembly
import tiny_full_contract as full

require = full.require
REGION_SYMBOLS = {"phase": "tiny_full_segment_phase", "control": "tiny_full_segment_control"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def span(image, address, size, writable):
    require(size > 0 and any(s["address"] <= address and address + size <= s["address"] + s["memory_bytes"]
        and bool(s["flags"] & 2) == writable for s in image.segments), "segment region protection/extent")
    if not writable:
        image.bytes_at(address, size)
    return (address, size)


def symbol(image, name, size, writable):
    item = image.symbols[name]
    require(item["size"] == size, "segment ABI symbol extent: " + name)
    return span(image, item["address"], size, writable)


def layout(image, data):
    address, _ = symbol(image, "tiny_full_compiled_live_layout", 64, False)
    values = struct.unpack("<8Q", image.bytes_at(address, 64))
    version, workspace_bytes, h0, h1, hidden_bytes, bank_bytes, phase_bytes, state_bytes = values
    require(version == 1 and hidden_bytes == 524288 and bank_bytes == 1802240
        and phase_bytes == 32 and state_bytes == 48 and h1 == h0 + hidden_bytes
        and 0 < h0 < h1 and h1 + hidden_bytes <= workspace_bytes <= 24 * 1024 * 1024,
        "segment compiled live layout ABI")
    workspace, _ = symbol(image, "tiny_full_segment_workspace", 24 * 1024 * 1024, True)
    result = []
    def add(name, address, size):
        span(image, address, size, True)
        result.append(dict(name=name, address=address, bytes=size))
    for field in ("keys", "values"):
        base, _ = symbol(image, "tiny_full_" + field, bank_bytes * 2, True)
        for bank in range(2):
            add(f"{field}_{bank}", base + bank * bank_bytes, bank_bytes)
    for bank, offset in enumerate((h0, h1)):
        add(f"hidden_{bank}", workspace + offset, hidden_bytes)
    for name, size in (("phase", 32), ("control", 64)):
        add(name, *symbol(image, REGION_SYMBOLS[name], size, True))
    for item in data["reference_buffers"]:
        add(item["actual_symbol"], *symbol(image, item["actual_symbol"], item["bytes"], True))
    require(len(result) == 145, "complete Tiny semantic live inventory")
    for i, left in enumerate(result):
        require(left["address"] % (2 if left["name"].startswith(("keys_", "values_", "hidden_")) else 4) == 0,
                "segment live alignment")
        require(all(full.disjoint((left["address"], left["bytes"]), (r["address"], r["bytes"]))
                    for r in result[:i]), "segment live alias")
    return result, values


def runtime_sites(image):
    ready = {pc: length for pc, length, word in image.instructions("tiny_full_segment_ready")
             if length == 4 and word == 0x10500073}
    require(len(ready) == 1, "exactly one named segment ready WFI")
    stops = full.stop_sites(image)
    success = {pc: length for pc, length, word in image.instructions("complete")
               if length == 4 and word == 0x10500073}
    require(len(success) == 1 and set(success) <= set(stops) and not set(ready) & set(stops),
            "distinct named segment completion/ready")
    require(image.symbols["main"]["size"] > 0 and image.entry == image.symbols["_start"]["address"],
            "segment normal startup entry")
    return ready, stops, success


def admit_public_source(bundle, item):
    source = assembly(*item["physical_mkn"], item["output_offset"],
                      item["rescale"], traffic_reuse=True)
    source = relocate_assembly(source, DEFAULT_ARENA, placement(bundle.build)["arena_base"])
    require(bundle.read(item["source"]) == source.encode(), "segment selected public reuse source")


def validate_image(bundle, image, assemble):
    require(callable(assemble), "pinned assembler required before segment admission")
    data = full.validate_bundle(bundle, image)
    require(data["decode_count"] == 2 and bundle.kernel_count == 840 and bundle.build.get("segment_mode") is True
        and bundle.build.get("result_name") == "tiny-segment", "explicit full Tiny segment build")
    regions, compiled = layout(image, data)
    ready, stops, success = runtime_sites(image)
    # Every immutable pointer must refer to read-only file-backed bytes.
    symbol(image, "tiny_full_panels", 120 * 840, False)
    for i, item in enumerate(data["kernels"]):
        panel = struct.unpack("<6QIi8Q", image.bytes_at(image.symbols["tiny_full_panels"]["address"] + 120 * i, 120))
        span(image, panel[8], len(bundle.read(item["weights"])), False)
        span(image, panel[10], len(bundle.read(item["original"])), False)
    for item in data["host_parameters"].values():
        symbol(image, item["symbol"], item["bytes"], False)
    for item in data["reference_buffers"]:
        symbol(image, item["expected_symbol"], item["bytes"], False)
    controls = []
    for name, size in (("tiny_full_state_global", 48), ("tiny_full_projection_state", 112),
                       ("tiny_full_host", 112), ("tiny_full_mailbox", 128), ("tohost", 8), ("fromhost", 8),
                       ("tiny_full_segment_request", 8)):
        controls.append(symbol(image, name, size, True))
    allocations = [(image.symbols["tiny_full_segment_workspace"]["address"], 24 * 1024 * 1024)]
    allocations += [(r["address"], r["bytes"]) for r in regions if not r["name"].startswith("hidden_")]
    stack = image.symbols["__stack_bottom"]["address"], image.symbols["__stack_top"]["address"]
    require(stack[1] - stack[0] == 65536 and stack[1] % 16 == 0, "segment fresh stack extent")
    allocations += controls + [span(image, stack[0], stack[1] - stack[0], True)]
    for i, left in enumerate(allocations):
        require(all(full.disjoint(left, right) for right in allocations[:i]), "segment allocation/control alias")
    state = struct.unpack("<5QII", image.bytes_at(image.symbols["tiny_full_state_global"]["address"], 48))
    banks = [image.symbols["tiny_full_" + field]["address"] + bank * 1802240
             for field in ("keys", "values") for bank in range(2)]
    require(state == (*banks, 0, 0, 0), "segment initial KV pointer/state ABI")
    context = struct.unpack("<9Q", image.bytes_at(image.symbols["tiny_full_projection_state"]["address"], 72))
    require(image.bytes_at(image.symbols["tiny_full_projection_state"]["address"] + 72, 40) == bytes(40),
            "segment legacy attention context must be zero")
    require(context == (image.symbols["tiny_full_panels"]["address"], 840, 0, placement(bundle.build)["arena_base"],
        max(k["total_dram_bytes"] for k in data["kernels"]), image.symbols["transport"]["address"], 0, 0, 3),
        "segment immutable projection context ABI")
    verified = set()
    for item in data["kernels"]:
        key = item["source"], item["original"]
        if key not in verified:
            if data.get("program_policy") == "traffic-reuse-v1":
                admit_public_source(bundle, item)
            runtime_program(bundle.read(key[0]).decode(), bundle.read(key[1]), assemble)
            verified.add(key)
    return data, regions, compiled


def validate_bundle(bundle, image, assemble, model_profile):
    data, regions, compiled = validate_image(bundle, image, assemble)
    chain = SegmentChain(bundle.root, "tiny-full")
    chain.admit_layout(regions)
    require(chain.kernel_count == 840 and chain.placement == placement(bundle.build)
        and chain.model_profile == model_profile and chain.data["elf"] == "host/host.elf"
        and chain.data["contract"] == "contract.json" and chain.data["capture_receipt"] == "segments/receipt.json",
        "segment chain image/placement/model binding")
    expected = [0] + [v for step in range(3) for v in
                     [*(280 * step + 12 * layer for layer in range(1, 23)), 280 * (step + 1)]]
    require([b["launch_cursor"] for b in chain.boundaries] == expected, "full69-segment launch partition")
    preparation = json.loads(chain.read("segments/receipt.json"))
    require(preparation.get("schema") == "atlas.tiny_boundary_preparation.v1"
        and preparation.get("status") == "PREPARED_REFERENCE_BOUNDARIES_NOT_EXECUTED"
        and preparation.get("fresh_instruction_execution") is False and preparation.get("reference_only") is True
        and all(preparation.get(k) is True for k in ("source_bookends_verified", "all_actual_predecessors_exact",
            "all22_KV_all_steps_exact", "all_logits_greedy_exact", "full_step_and_phases_exact", "scratch_reset_each_phase"))
        and preparation.get("segments") == 69 and preparation.get("boundaries") == 70
        and preparation.get("physical_panels") == 840 and preparation.get("projection_calls") == 465,
        "complete native reference preparation gate")
    layout_dict = preparation["compiled_live_layout"]
    require(tuple([layout_dict["version"], layout_dict["workspace_bytes"], *layout_dict["hidden_offset"],
        layout_dict["hidden_bytes"], layout_dict["cache_bank_bytes"], layout_dict["phase_bytes"],
        layout_dict["state_bytes"]]) == compiled, "native/RV64 compiled live layout equality")
    native_gate = json.loads(chain.read("segments/native-gate.json"))
    require(native_gate["capture_sha256"] == bundle.build["capture_sha256"] == preparation["capture_sha256"]
        and chain.sums["segments/native-gate.json"] == bundle.build["gate_sha256"]
        and native_gate["ledger"] == preparation["gate"]["ledger"], "native gate/capture/build binding")
    source_pins = preparation.get("source_sha256")
    snapshots = preparation.get("source_snapshots")
    require(isinstance(source_pins, dict) and source_pins and isinstance(snapshots, dict) and snapshots,
            "native preparation source snapshots required")
    for path, member in snapshots.items():
        require(path in source_pins and chain.sums.get("segments/" + member) == source_pins[path],
                "native source snapshot binding")
    for name in ("tiny_full_model.c", "tiny_full_model.h"):
        h = chain.sums["host/" + name]
        require({v for p, v in source_pins.items() if p.rsplit("/", 1)[-1] == name} == {h},
                "native/RV64 arithmetic source equality")
    expected_names = {r["name"] for r in regions}
    require(set(preparation["semantic_regions"]) == expected_names and all(
        preparation["semantic_regions"][r["name"]] == r["bytes"] for r in regions), "native live inventory equality")
    refs = {r["actual_symbol"]: r for r in data["reference_buffers"]}
    require(set(chain.data["final_outputs"]) == set(refs), "all137final references in chain")
    for name, item in refs.items():
        require(chain.sums[chain.boundaries[-1]["regions"][name]] == item["sha256"], "final chain reference binding")
    for i, boundary in enumerate(chain.boundaries):
        control = struct.unpack("<8Q", chain.read(boundary["regions"]["control"]))
        cursor = boundary["launch_cursor"]
        step, rem = divmod(cursor, 280)
        layer = rem // 12
        require(rem == 0 or rem <= 264 and rem % 12 == 0, "semantic boundary cursor")
        position = 0 if step == 0 else 127 + step
        token = 0 if step == 0 else data["greedy_tokens"][step - 1]
        require(control == (step, layer, position, step % 2, token, 128 if step == 0 else 1,
                            step * 155 + layer * 7, cursor), "semantic control/phase partition")
        phase = struct.unpack("<Q6I", chain.read(boundary["regions"]["phase"]))
        if rem:
            require(phase == (position, step % 2, token, layer, 128 if step == 0 else 1, 1, 0),
                    "semantic active continuation")
        else:
            require(phase[5:] == (0, 0), "semantic inactive continuation")
    return chain, data


def verify_kernel(summary, item, index):
    return verify_kernel_receipt(summary, item, index, full.CYCLE_CAP)
