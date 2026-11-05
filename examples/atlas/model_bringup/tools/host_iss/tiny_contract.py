"""Admission and final guest-state checks for Tiny block-zero prefill/decode.

The fixture contract supplies references, never successor host activations.
These functions do not execute a model and make no full-inference claim.
"""

import hashlib
import json
import struct

from bundle import EPILOGUE

CYCLE_CAP = 200_000_000
KERNEL_COUNT = 24
MAGIC = 0x54494E59424C4B31
# Exact BF16 extents for block zero, 128-token prefill and one decode token.
FINAL_BUFFERS = {
    "prefill_output": (0, "tiny_expected_output", 128 * 2048 * 2),
    "decode_output": (1, "tiny_expected_output", 2048 * 2),
    "prefill_keys": (0, "tiny_expected_keys", 4 * 128 * 64 * 2),
    "decode_keys": (1, "tiny_expected_keys", 4 * 129 * 64 * 2),
    "prefill_values": (0, "tiny_expected_values", 4 * 128 * 64 * 2),
    "decode_values": (1, "tiny_expected_values", 4 * 129 * 64 * 2),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _contract(bundle):
    provenance = bundle.provenance
    assert provenance["workload"] == "tiny" and bundle.kernel_count == KERNEL_COUNT
    raw = bundle.read(provenance["contract"])
    assert digest(raw) == provenance["contract_sha256"], "Tiny contract identity"
    return json.loads(raw)


def _pointer(image, name, index=0):
    symbol = image.symbols[name]
    assert symbol["size"] >= (index + 1) * 8, ("Tiny pointer table extent", name)
    return struct.unpack("<Q", image.bytes_at(symbol["address"] + index * 8, 8))[0]


def stop_sites(image):
    """Include the startup trap loop, whose trap_entry symbol has size zero."""
    sites = {pc: length for pc, length, word in image.instructions("main")
             if length == 4 and word == 0x10500073}
    assert sites, "Tiny completion WFI missing"
    # Startup's fixed twelve-instruction trap recorder ends in c.j 0. Validate
    # that loop from its bytes rather than attempting to scan a zero-size symbol.
    trap = image.symbols["trap_entry"]["address"] + 56
    assert image.bytes_at(trap, 2) == b"\x01\xa0", "Tiny trap loop changed"
    assert trap not in sites
    sites[trap] = 2
    return sites


def validate_bundle(bundle, image):
    """Bind the supplied bundle, retained contract and actual ELF byte for byte."""
    data = _contract(bundle)
    build = json.loads(bundle.read("host/build.json"))
    assert build["status"] == "compiled_unexecuted_Tiny_firstblock_prefill_decode_RV64_ELF"
    assert build["actual_returncode"] == 0 and build["source_bookends_verified"] is True
    assert build["host_elf_executed"] is False and build["platform_executed"] is False
    assert digest(image.raw) == build["elf"]["sha256"] == data["ELF"]["sha256"]
    assert data["finite_model_cycle_cap_per_start"] == CYCLE_CAP
    assert data["mailbox"]["symbol"] == "atlas_test_mailbox"
    assert image.symbols["atlas_test_mailbox"]["size"] == 128
    assert image.symbols["tohost"]["size"] == 8
    assert image.symbols["tiny_panels"]["size"] == 120 * KERNEL_COUNT
    expected_sites = {x["pc"]: x["length"] for x in data["completion_WFI_sites"] + data["trap_loop_sites"]}
    assert stop_sites(image) == expected_sites, "Tiny stop sites disagree with contract"
    assert len(data["programs"]) == KERNEL_COUNT
    for index, item in enumerate(data["programs"]):
        assert item["index"] == index and item["sequence"] == index + 1
        phase = index // 12
        assert item["phase"] == phase
        geometry, memory = item["geometry"], item["memory"]
        assert geometry["rows"] == (1 if phase else 128)
        assert geometry["physical_rows"] == (32 if phase else 128)
        paths = bundle.kernel_paths(index)
        for key, name in (("source", "program.S"), ("original", "program.bin"),
                          ("initial", "initial-dram.bin"), ("expected", "expected-bf16-tiles.bin")):
            assert bundle.sums[paths[key]] == item["files"][name]["sha256"], ("Tiny kernel payload identity", index, name)
        assert paths["dram_base"] == memory["dram_base"] == 0x100000000
        assert paths["output_offset"] == memory["output_offset"]
        assert paths["requested_dma"] == item["requested_dma"]
        initial, expected = bundle.read(paths["initial"]), bundle.read(paths["expected"])
        assert len(initial) == memory["total_dram_bytes"]
        assert len(expected) == memory["output_bytes"] == item["requested_dma"]["store_bytes"]
        original = bundle.read(paths["original"])
        assert original[-4:] == struct.pack("<I", 0x73)
        assert digest(original[:-4] + EPILOGUE) == item["runtime_IMEM_sha256"]
        assert len(original) // 4 + 5 == item["runtime_IMEM_words"]
        # The RV64 ABI lays out tiny_panel as six Q, unsigned, signed,
        # then eight Q. Check its actual descriptor and embedded program.
        panel = struct.unpack("<6QIi8Q", image.bytes_at(image.symbols["tiny_panels"]["address"] + index * 120, 120))
        assert panel[:8] == tuple(geometry[k] for k in ("rows", "physical_rows", "k", "n", "column", "width", "logical", "a_scale"))
        assert panel[11:] == (len(original) // 4, memory["output_offset"], memory["total_dram_bytes"],
                              item["requested_dma"]["load_bytes"], item["requested_dma"]["store_bytes"])
        assert image.bytes_at(panel[10], len(original)) == original
        assert image.bytes_at(_pointer(image, "tiny_panel_expected", index), len(expected)) == expected
    assets = data["final_actual_C_output_KV_assets"]
    assert len(assets) == len(FINAL_BUFFERS)
    assert {x["actual_symbol"] for x in assets} == set(FINAL_BUFFERS)
    for item in assets:
        name = item["actual_symbol"]
        phase, table, size = FINAL_BUFFERS[name]
        assert (item["phase"], item["expected_pointer_table"], item["bytes"]) == (phase, table, size)
        assert image.symbols[name]["size"] == size
        address = _pointer(image, table, phase)
        assert address == image.symbols[item["expected_blob_symbol"]]["address"] == item["expected_ELF_address"]
        assert digest(image.bytes_at(address, size)) == item["expected_blob"]["sha256"]
    return data


def verify_completion(remote, image, device, bundle, out):
    """Require every launch and all six actual guest outputs and cache buffers."""
    data = _contract(bundle)
    mailbox = remote.read(image.symbols["atlas_test_mailbox"]["address"], 128)
    (out / "mailbox.bin").write_bytes(mailbox)
    assert len(mailbox) == 128, "Tiny mailbox extent"
    words = struct.unpack("<16Q", mailbox)
    assert words == (MAGIC, 2, 24, 23, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0), ("Tiny completion mailbox", words)
    tohost_raw = remote.read(image.symbols["tohost"]["address"], 8)
    assert len(tohost_raw) == 8, "Tiny tohost extent"
    tohost = int.from_bytes(tohost_raw, "little")
    assert tohost == 1, ("Tiny tohost", tohost)
    assert len(device.completed) == KERNEL_COUNT, "Tiny completed start count"
    assert [x["index"] for x in device.completed] == list(range(KERNEL_COUNT)), "Tiny launch order"
    assert [x["sequence"] for x in device.completed] == list(range(1, KERNEL_COUNT + 1)), "Tiny completion sequences"
    assert all(x.get("exact") is True and x.get("drained") is True and x.get("reset_count") == 1
               for x in device.completed), "Tiny device launch ledger"
    results = []
    for item in data["final_actual_C_output_KV_assets"]:
        name = item["actual_symbol"]
        phase, table, size = FINAL_BUFFERS[name]
        expected = image.bytes_at(_pointer(image, table, phase), size)
        assert digest(expected) == item["expected_blob"]["sha256"], ("Tiny final reference identity", name)
        actual = remote.read(image.symbols[name]["address"], size)
        (out / (name + ".bin")).write_bytes(actual)
        assert actual == expected, ("Tiny final guest buffer mismatch", name)
        results.append(dict(symbol=name, bytes=size, BF16_words=size // 2, actual_sha256=digest(actual),
                            expected_sha256=digest(expected), exact=True))
    return dict(tohost=tohost, mailbox_words=words, expected_starts=KERNEL_COUNT,
                prefill_tokens=128, decode_tokens=1, final_guest_buffers=results,
                all_final_guest_buffers_exact=True)
