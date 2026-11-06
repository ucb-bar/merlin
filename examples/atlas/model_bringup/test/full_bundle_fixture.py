"""Small checksum-bound full/kernel bundles shared by provider and SoC tests."""
import json
import struct

import kernel_library  # noqa: E402,F401
from atlas_tooling import sha
from bundle_payload_fixture import write_bundle_payloads


def full_bundle_fixture(root, mutate=None, *, kernel_only=False):
    activation, weights, expected = b"a" * 1024, b"w" * 1024, b"e" * 2048
    initial = activation + weights + b"\xa5" * 256 + b"\xcd" * 2048 + b"\x5a" * 256
    descriptor = dict(index=0, source="assets/program.S", original="assets/program.bin",
                      activation="assets/a.bin", weights="assets/w.bin", expected="assets/out.bin",
                      physical_mkn=[32, 32, 32], initial_sha256=sha(initial),
                      output_offset=2304, total_dram_bytes=4608, dram_base=0x100000000,
                      requested_dma=dict(load_bytes=2048, store_bytes=2048))
    provenance = dict(schema="atlas.full_kernel_assets.v1" if kernel_only else "atlas.full_bundle.v1",
                      workload="tiny-full", kernel_count=2,
                      kernels=[descriptor, dict(descriptor, index=1)], contract="contract.json",
                      contract_sha256=sha(b"{}"))
    if mutate is not None:
        mutate(provenance)
    metadata = "kernels.json" if kernel_only else "provenance.json"
    files = {metadata: json.dumps(provenance).encode(),
             "assets/program.S": b"ecall\n", "assets/program.bin": struct.pack("<I", 0x73),
             "assets/a.bin": activation, "assets/w.bin": weights, "assets/out.bin": expected}
    if not kernel_only:
        files.update({"host/host.elf": b"bound elf", "host/build.json": b"{}", "contract.json": b"{}"})
    write_bundle_payloads(root, files)
    return root, initial
