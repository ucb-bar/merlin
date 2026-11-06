"""Checksum-bound inputs for the fixed TinyLlama and SmolVLA block hosts."""

import json
import struct

from atlas_host_layout import ATLAS_DRAM_BASE, ATLAS_IMEM_BYTES
from bundle import ChecksumBundle


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid block descriptor integer")
    return value


class BlockBundle(ChecksumBundle):
    def __init__(self, root, workload):
        super().__init__(root)
        self.provenance = json.loads(self.read("provenance.json"))
        self.build = json.loads(self.read("host/build.json"))
        counts = {"tiny": 24, "smol": 7}
        if (workload not in counts or self.provenance.get("schema") != "atlas.block_bundle.v1"
            or self.provenance.get("workload") != workload
            or self.provenance.get("kernel_count") != counts[workload]):
            raise ValueError("unsupported fixed block bundle")
        self.workload = workload
        self.kernel_count = counts[workload]
        self.kernels = self.provenance.get("kernels")
        if not isinstance(self.kernels, list) or len(self.kernels) != self.kernel_count:
            raise ValueError("block launch count differs")
        if "host/host.elf" not in self.sums:
            raise ValueError("block ELF missing from checksums")
        for descriptor in self.kernels:
            self.validate_kernel(descriptor)

    def validate_kernel(self, item):
        if not isinstance(item, dict):
            raise ValueError("invalid block launch descriptor")
        for key in ("source", "original", "initial", "expected"):
            if not isinstance(item.get(key), str) or item[key] not in self.sums:
                raise ValueError("unlisted block kernel payload: " + key)
        original = self.read(item["original"])
        if (not original or len(original) % 4 or len(original) + 20 > ATLAS_IMEM_BYTES
            or original[-4:] != struct.pack("<I", 0x73)):
            raise ValueError("block program must fit IMEM and end in ECALL")
        initial, expected = self.read(item["initial"]), self.read(item["expected"])
        offset = integer(item.get("output_offset"))
        base = integer(item.get("dram_base"))
        if base != ATLAS_DRAM_BASE:
            raise ValueError("fixed block staging address differs")
        if not expected or len(expected) % 2 or offset + len(expected) > len(initial) or len(initial) > 64 * 1024 * 1024:
            raise ValueError("block output or staging extent differs")
        requested = item.get("requested_dma")
        if (not isinstance(requested, dict) or not {"load_bytes", "store_bytes"} <= requested.keys()
            or requested.keys() - {"load_bytes", "store_bytes", "response_beats"}):
            raise ValueError("invalid requested DMA counters")
        for value in requested.values():
            integer(value)
        if requested["store_bytes"] != len(expected):
            raise ValueError("DMA store extent differs from output")

    def kernel_paths(self, index):
        integer(index)
        if index >= self.kernel_count:
            raise ValueError("block launch index exceeds contract")
        # Return a copy so device-side bookkeeping cannot alter admission state.
        return dict(self.kernels[index], requested_dma=dict(self.kernels[index]["requested_dma"]))
