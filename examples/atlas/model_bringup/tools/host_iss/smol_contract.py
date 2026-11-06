"""Strict selected SmolVLA layer admission and live guest completion checks."""

import hashlib
import json
import struct

CYCLE_CAP = 100000000
ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
INPUT_WIDTHS = (960, 960, 960, 960, 960, 960, 2560)
OUTPUT_WIDTHS = (960, 320, 320, 960, 2560, 2560, 960)
FINAL = (("expected_output", "output", 339840),
         ("expected_key_cache", "rotary_key", 113280),
         ("expected_value_cache", "value", 113280))


def require(condition, message):
    if not condition:
        raise ValueError("Smol contract: " + message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def symbol(image, name, size):
    value = image.symbols[name]
    require(value["size"] == size, "symbol extent " + name)
    return value["address"]


def blob_symbol(image, name, size):
    # Assembly incbin labels have zero st_size; bytes_at still enforces the
    # complete file-backed extent, and callers compare every authenticated byte.
    value = image.symbols[name]
    require(value["size"] in (0, size), "embedded symbol extent " + name)
    return value["address"]


def contract(bundle):
    provenance = bundle.provenance
    require(provenance.get("workload") == "smol" and bundle.kernel_count == 7, "workload or launch count")
    name = provenance.get("contract")
    require(isinstance(name, str), "missing source contract")
    raw = bundle.read(name)
    require(digest(raw) == provenance.get("contract_sha256"), "source contract digest")
    value = json.loads(raw)
    require(value.get("schema") == "atlas.smol_selected_layer1.epoch3_launch_data.v1", "source schema")
    return value


def stop_sites(image):
    result = {}
    for name in ("atlas_workload_main", "atlas_block_trap_report"):
        sites = {pc: length for pc, length, word in image.instructions(name)
                 if length == 4 and word == 0x10500073}
        require(bool(sites), "missing completion/trap WFI in " + name)
        result.update(sites)
    return result


def validate_bundle(bundle, image):
    value = contract(bundle)
    build = bundle.build
    require(build.get("status") == "passed_RV64GC_LP64D_ELF_build_only"
            and build.get("ELF_executed") is False and build.get("source_bookends_unchanged") is True,
            "original build status or bookends")
    elf_hash = bundle.sums["host/host.elf"]
    require(build.get("elf", {}).get("sha256") == elf_hash == value["ELF"]["sha256"], "ELF identity")
    require(digest(image.raw) == elf_hash, "parsed ELF identity")
    require(image.entry == 0x80000000, "ELF entry")
    for name, recorded in value["symbols"].items():
        observed = image.symbols[name]
        require(all(observed[key] == recorded[key] for key in ("address", "size", "section")),
                "source symbol disagreement " + name)
    symbol(image, "smol_workload_mailbox", 120)
    symbol(image, "smol_selected_state", 32)
    symbol(image, "tohost", 8)
    extents = dict(input=339840, pad=177, attention_boundaries=177, input_norm_weight=1920,
                   post_norm_weight=1920, cosine=1048704, sine=1048704, expected_output=339840,
                   expected_key_cache=113280, expected_value_cache=113280)
    require(set(value["assets"]) == set(extents), "asset set")
    parameters = value["constant_parameters"]
    require(parameters["input_epsilon"] == parameters["post_epsilon"] == 1e-5
            and parameters["first_table_position"] == -1 and parameters["table_rows"] == 8193,
            "selected-layer parameters")
    for name, asset in value["assets"].items():
        raw = bundle.read("data/" + name + ".bin")
        record = asset["file"]
        require(len(raw) == record["bytes"] == extents[name] and digest(raw) == record["sha256"], "asset identity " + name)
        address = blob_symbol(image, asset["ELF_constant_symbol"], len(raw))
        require(address == asset["ELF_constant_address"] and image.bytes_at(address, len(raw)) == raw,
                "embedded asset disagreement " + name)
    for name, actual, size in FINAL:
        asset = value["assets"][name]
        require(asset["actual_C_buffer_symbol"] == actual, "actual buffer identity " + name)
        require(symbol(image, actual, size) == asset["actual_C_buffer"]["address"]
                and asset["actual_C_buffer"]["size"] == size, "actual buffer extent " + name)
    projections = value["ordered_projections"]
    require(len(projections) == 7, "source projection count")
    for index, (projection, role, width_in, width_out) in enumerate(zip(projections, ROLES, INPUT_WIDTHS, OUTPUT_WIDTHS)):
        require(projection["index"] == index and projection["role"] == role, "projection order")
        require((projection["physical_rows"], projection["native_rows"], projection["input_width"],
                 projection["output_width"]) == (192, 177, width_in, width_out), "projection dimensions")
        paths = bundle.kernel_paths(index)
        require(paths["dram_base"] == 0x100000000 and paths["output_offset"] == projection["output_offset"]
                and paths["requested_dma"] == projection["requested_dma"], "launch descriptor")
        for key, source in (("source", "program_S"), ("original", "original_program_bin"),
                            ("initial", "initial_dram"), ("expected", "expected_physical_output")):
            raw = bundle.read(paths[key])
            require(len(raw) == projection[source]["bytes"] and digest(raw) == projection[source]["sha256"],
                    "projection payload " + role + "/" + key)
        initial = bundle.read(paths["initial"])
        require(len(initial) == projection["dram_bytes"], "projection DRAM extent")
        program = bundle.read(paths["original"])
        address = blob_symbol(image, projection["ELF_original_program_symbol"], len(program))
        require(image.bytes_at(address, len(program)) == program, "embedded projection program")
        activation_bytes = 192 * width_in
        weight_bytes = width_in * width_out
        require(paths["output_offset"] == activation_bytes + weight_bytes + 256
                and len(initial) == paths["output_offset"] + 192 * width_out * 2 + 256, "projection layout")
        weights = initial[activation_bytes:activation_bytes + weight_bytes]
        address = blob_symbol(image, projection["ELF_weight_symbol"], weight_bytes)
        require(address == projection["ELF_weight_address"] and digest(weights) == projection["weights"]["sha256"]
                and image.bytes_at(address, weight_bytes) == weights, "embedded projection weights")
    stop_sites(image)


def verify_completion(remote, image, device, bundle, out):
    """Compare real guest output and resident caches with authenticated references."""
    validate_bundle(bundle, image)
    mailbox = remote.read(symbol(image, "smol_workload_mailbox", 120), 120)
    (out / "mailbox.bin").write_bytes(mailbox)
    require(len(mailbox) == 120, "mailbox extent")
    fields = struct.unpack_from("<23I", mailbox)
    require(fields[:6] == (0x534d4f4c, 1, 0, 7, 6, 0), "mailbox status")
    require(fields[6:13] == tuple(range(1, 8)), "mailbox sequence")
    require(all(fields[13:20]), "zero device cycle counter")
    require(fields[20:23] == (0, 0, 0), "mailbox output/cache mismatch")
    require(mailbox[92:120] == bytes(28), "padding or trap record nonzero")
    tohost_raw = remote.read(symbol(image, "tohost", 8), 8)
    require(len(tohost_raw) == 8, "tohost extent")
    tohost = int.from_bytes(tohost_raw, "little")
    require(tohost == 1, "tohost completion")
    state = remote.read(symbol(image, "smol_selected_state", 32), 32)
    (out / "state.bin").write_bytes(state)
    require(state == struct.pack("<8I", 1, 7, 6, 0, 1, 1, 0, 0), "block completion/cache state")
    completed = device.completed
    require(len(completed) == 7, "device launch count")
    for index, record in enumerate(completed):
        require(record.get("index") == index and record.get("sequence") == index + 1
                and record.get("exact") is True and record.get("drained") is True
                and record.get("reset_count") == 1, "device launch ledger")
    outputs = {}
    for name, actual, size in FINAL:
        raw = remote.read(symbol(image, actual, size), size)
        (out / (actual + ".bin")).write_bytes(raw)
        expected = bundle.read("data/" + name + ".bin")
        require(len(raw) == size and raw == expected, "live output/cache differs " + actual)
        outputs[actual] = dict(bytes=size, sha256=digest(raw), exact=True)
    return dict(mailbox_words=fields, tohost=tohost, all_outputs_exact=True, all_caches_exact=True,
                outputs=outputs, block_state_words=struct.unpack("<8I", state), workload="smol",
                scope_selected_layer_only=True)
