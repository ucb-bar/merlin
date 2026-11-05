"""Admit complete Smol semantic boundaries before any fresh device is created."""
import json
import struct

from bundle import runtime_program, verify_kernel_receipt
from placement import placement
from segment_boundary import SegmentChain
from smol_full_plan import calls
from smol_full_phases import execution_profile, extra_live
import smol_full_contract as full

require = full.require
COMPILED_LAYOUT = (1, 40, 28, 16, 16, 184, 144, 80, 112, 2915, 1421, 200)



def partitions():
    """Derive the fixed 24 vision, 16 prefix and 160 expert layer intervals."""
    result = []
    for call in calls():
        site, layer = call["site"], call["layer"]
        if site == 7:
            key = (0, call["image"], min(layer, 11))
        elif site in (0, 5):
            key = (1, 0, layer if site == 5 else 0)
        else:
            key = (2, call["step"], layer if site == 6 else 15 if site == 4 else 0)
        if not result or result[-1]["identity"] != key:
            result.append(dict(index=len(result), identity=key, call_start=call["index"], call_end=call["index"]+1))
        else:
            result[-1]["call_end"] = call["index"]+1
    require(len(result) == 200 and result[0]["call_start"] == 0
            and result[-1]["call_end"] == 1421, "complete fixed semantic partition")
    return result


def span(image, address, size, writable):
    require(type(address) is int and type(size) is int and size > 0 and any(
        s["address"] <= address and address+size <= s["address"]+s["memory_bytes"]
        and bool(s["flags"] & 2) == writable for s in image.segments), "phase memory protection/extent")
    if not writable:
        full.readonly_span(image, address, size)
    return address, size


def symbol(image, name, size, writable):
    row = image.symbols[name]
    require(row["size"] == size, "phase symbol extent: "+name)
    return span(image, row["address"], size, writable)


def runtime_sites(image):
    def wfi(name):
        sites = {pc: length for pc, length, word in image.instructions(name)
                 if length == 4 and word == 0x10500073}
        require(len(sites) == 1, "one named phase WFI: "+name)
        return sites
    ready = {pc: length for pc, length, word in image.instructions("smol_full_segment_ready")
             if length == 4 and word == 0x13}
    require(len(ready) == 1, "one returning phase load hook NOP")
    require(not any(length == 4 and word == 0x10500073
                    for _, length, word in image.instructions("smol_full_segment_ready")),
            "phase ready hook must not halt a debugger-free full run")
    success = wfi("smol_full_segment_done")
    stops = dict(full.stop_sites(image))
    stops.update(success)
    stops.update(wfi("smol_full_segment_failed"))
    require(not set(ready) & set(stops), "distinct phase load/completion sites")
    return ready, stops, success


def layout(image, data):
    profile = execution_profile(data)
    compiled_layout = (*COMPILED_LAYOUT[:9], profile.launch_count, *COMPILED_LAYOUT[10:])
    address, size = symbol(image, "smol_full_compiled_phase_layout", 96, False)
    require(struct.unpack("<12Q", image.bytes_at(address, size)) == compiled_layout, "compiled phase ABI")
    address, size = symbol(image, "smol_full_driver_observation", 56, False)
    observation = struct.unpack("<7Q", image.bytes_at(address, size))
    parameters = data["host_parameters"]
    require(observation[:6] == tuple(image.symbols[parameters["observation."+name]["symbol"]]["address"]
        for name in ("images", "state", "noise", "image_masks", "language_mask", "language_tokens"))
        and 0 < observation[6] < 1 << 32, "compiled driver observation pointers")
    expected = extra_live(profile)
    for row in data["reference_buffers"]:
        require(row["actual_symbol"] not in expected, "duplicate semantic symbol")
        expected[row["actual_symbol"]] = row["bytes"]
    require(len(expected) == 56, "complete fixed Smol semantic inventory")
    regions = [dict(name=name, address=symbol(image, name, size, True)[0], bytes=size)
               for name, size in expected.items()]
    controls = [symbol(image, name, size, True) for name, size in (
        ("smol_full_raw", 184), ("smol_full_host", 112), ("atlas_workload_mailbox", 128),
        ("tohost", 8), ("fromhost", 8), ("smol_full_segment_request", 8),
        ("smol_full_segment_observations", profile.cycle_bytes))]
    if data.get('fusion_set','none')!='none':controls.append(symbol(image,'state',104,True))
    stack = image.symbols["__stack_bottom"]["address"], image.symbols["__stack_top"]["address"]
    require(stack[1]-stack[0] == 65536 and stack[1] % 16 == 0, "fresh phase stack")
    allocations = [(r["address"], r["bytes"]) for r in regions] + controls + [span(
        image, stack[0], stack[1]-stack[0], True)]
    for i, allocation in enumerate(allocations):
        require(all(full.disjoint(allocation, before) for before in allocations[:i]), "phase allocation alias")
    request, _ = symbol(image, "smol_full_segment_request", 8, True)
    require(image.bytes_at(request, 8) == bytes([255])*8, "default full-policy selector")
    return regions


def embedded_boundaries(chain, image, phases, regions):
    """Bind guest self-check tables to every admitted captured byte."""
    address, size = symbol(image, "smol_full_phase_plan", 200*32, False)
    plan = image.bytes_at(address, size)
    for index, phase in enumerate(phases):
        require(struct.unpack_from("<8I", plan, 32*index) == (
            *phase["identity"], phase["launch_start"], phase["launch_end"],
            phase["call_start"], phase["call_end"], 0), "compiled phase plan")
    address, size = symbol(image, "smol_full_phase_regions", 56*24, False)
    rows = image.bytes_at(address, size)
    for index, region in enumerate(regions):
        actual, count, name = struct.unpack_from("<3Q", rows, 24*index)
        text = (region["name"]+"\0").encode()
        full.readonly_span(image, name, len(text))
        require((actual, count) == (region["address"], region["bytes"])
                and image.bytes_at(name, len(text)) == text, "compiled semantic region table")
    address, size = symbol(image, "smol_full_phase_expected", 201*56*8, False)
    pointers = image.bytes_at(address, size)
    for index, boundary in enumerate(chain.boundaries):
        for ordinal, region in enumerate(regions):
            pointer, = struct.unpack_from("<Q", pointers, 8*(56*index+ordinal))
            raw = chain.read(boundary["regions"][region["name"]])
            full.readonly_span(image, pointer, len(raw))
            require(image.bytes_at(pointer, len(raw)) == raw, "compiled exact semantic boundary")


def validate_image(bundle, image, assemble):
    require(callable(assemble), "pinned assembler required before phase admission")
    data = full.validate_bundle(bundle, image, assemble=assemble)
    require(bundle.build.get("segment_mode") is True and bundle.build.get("result_name") == "smol-full"
            and bundle.build.get("segment_result_name") == "smol-segment",
            "explicit dual-mode full policy/phase build")
    regions = layout(image, data)
    runtime_sites(image)
    verified = set()
    for item in [*data["kernels"], *data.get("attention_kernels", [])]:
        key = item["source"], item["original"]
        if key not in verified:
            runtime_program(bundle.read(key[0]).decode(), bundle.read(key[1]), assemble)
            verified.add(key)
    return data, regions


def validate_bundle(bundle, image, assemble, model_profile):
    data, regions = validate_image(bundle, image, assemble)
    profile = execution_profile(data)
    compiled_layout = (*COMPILED_LAYOUT[:9], profile.launch_count, *COMPILED_LAYOUT[10:])
    chain = SegmentChain(bundle.root, "smol-full")
    chain.admit_layout(regions)
    require(chain.kernel_count == profile.launch_count and chain.placement == placement(bundle.build)
            and chain.model_profile == model_profile and chain.data["elf"] == "host/host.elf"
            and chain.data["contract"] == bundle.provenance["contract"]
            and chain.data["capture_receipt"] == "segments/receipt.json",
            "phase chain image/placement/model binding")
    phases = partitions()
    for entry in phases:
        indices = [row["index"] for row in data["kernels"]
                   if entry["call_start"] <= row["call"] < entry["call_end"]]
        require(indices and indices == list(range(indices[0], indices[-1]+1)), "contiguous phase launches")
        entry.update(launch_start=indices[0]+(entry["index"] if profile.attention_count else 0),
                     launch_end=indices[-1]+1+(entry["index"]+1 if profile.attention_count else 0))
    require([b["launch_cursor"] for b in chain.boundaries] == [0]+[p["launch_end"] for p in phases],
            "all200 semantic phase boundaries")
    embedded_boundaries(chain, image, phases, regions)
    preparation = json.loads(chain.read("segments/receipt.json"))
    require(preparation.get("schema") == "atlas.smol_boundary_preparation.v1"
            and preparation.get("status") == "PREPARED_REFERENCE_BOUNDARIES_NOT_EXECUTED"
            and preparation.get("reference_only") is True and preparation.get("guest_execution") is False
            and preparation.get("device_execution") is False and preparation.get("segments") == 200
            and preparation.get("boundaries") == 201 and preparation.get("projection_calls") == 1421
            and preparation.get("physical_panels") == 2715
            and preparation.get("compiled_phase_layout") == list(compiled_layout), "complete native phase gate")
    for name in ("source_bookends_verified", "all_actual_predecessors_exact", "all_final_references_exact",
                 "full_and_phases_exact", "scratch_reset_each_phase", "arena_reset_each_phase",
                 "canonical_zero_observations", "live_corruption_rejected"):
        require(preparation.get(name) is True, "native phase gate missing: "+name)
    require(preparation.get("source_reference_sha256") == data["source_reference_sha256"], "phase capture identity")
    sources, snapshots = preparation.get("source_sha256"), preparation.get("source_snapshots")
    require(isinstance(sources, dict) and sources and isinstance(snapshots, dict) and snapshots,
            "phase source snapshots required")
    for path, member in snapshots.items():
        require(path in sources and chain.sums.get("segments/"+member) == sources[path], "phase source snapshot")
    required = {"smol_full_driver.c", "smol_full_driver.h", "smol_policy_graph.c", "smol_policy_graph.h",
                "smol_vision_graph.c", "smol_vision_graph.h"}
    if data.get('fusion_set','none')!='none':
        required.update(('smol_fusion.c','smol_fusion.h','smol_policy_raw.c','smol_policy_raw.h'))
    for name in required:
        require({value for path, value in sources.items() if path.rsplit("/", 1)[-1] == name}
                == {chain.sums["host/"+name]}, "native/RV64 shared arithmetic/driver source: "+name)
    require(preparation.get("semantic_regions") == {r["name"]: r["bytes"] for r in regions},
            "native/RV64 complete live inventory")
    refs = {row["actual_symbol"]: row for row in data["reference_buffers"]}
    require(set(chain.data["final_outputs"]) == set(refs), "all47 final references in phase chain")
    for name, row in refs.items():
        require(chain.sums[chain.boundaries[-1]["regions"][name]] == row["sha256"], "final phase reference identity")
    for index, boundary in enumerate(chain.boundaries):
        cursor = struct.unpack("<10I", chain.read(boundary["regions"]["smol_full_phase_state"]))
        expected = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0) if index == 0 else (
            1, index, *phases[index-1]["identity"], phases[index-1]["launch_end"]-(index if profile.attention_count else 0),
            phases[index-1]["call_end"], phases[index-1]["launch_end"], 0, 0)
        require(cursor == expected, "phase pointer-free cursor/order")
        require(chain.read(boundary["regions"]["smol_full_cycles"]) == bytes(profile.cycle_bytes),
                "canonical boundary observations must be zero")
    return chain, data


def verify_kernel(summary, item, index):
    return verify_kernel_receipt(summary, item, index, full.CYCLE_CAP)
