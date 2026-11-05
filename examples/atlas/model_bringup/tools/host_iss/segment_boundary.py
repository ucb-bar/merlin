"""Checksum-bound captured live state and exact segment-chain comparisons.

This format is preparation, not an execution receipt. Workload harnesses must
admit the complete live-region inventory and the actual segment entry code.
"""

import hashlib
import json
import re

from bundle import ChecksumBundle
from placement import placement


def integer(value, minimum=0):
    if type(value) is not int or value < minimum or value >= 1 << 64:
        raise ValueError("invalid segment integer")
    return value


def identity(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid segment digest")
    return value


def regions(layout, physical):
    if not isinstance(layout, list) or not layout:
        raise ValueError("segment live-region inventory required")
    names, spans = set(), []
    for item in layout:
        if not isinstance(item, dict) or set(item) != {"name", "address", "bytes"}:
            raise ValueError("invalid segment live region")
        name = item["name"]
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None or name in names:
            raise ValueError("invalid or duplicate segment region name")
        start, size = integer(item["address"]), integer(item["bytes"], 1)
        end = start + size
        if not physical["dram_base"] <= start < end <= physical["dram_base"] + physical["dram_bytes"]:
            raise ValueError("segment live region outside physical DRAM")
        if any(start < hi and lo < end for lo, hi in spans):
            raise ValueError("overlapping segment live regions")
        names.add(name)
        spans.append((start, end))
    return {item["name"]: dict(item) for item in layout}


class SegmentChain(ChecksumBundle):
    """Immutable ordered boundaries, bound to one ELF and workload contract."""

    def __init__(self, root, workload):
        super().__init__(root)
        self.data = data = json.loads(self.read("segments.json"))
        if (data.get("schema") != "atlas.segment_chain.v1"
                or workload not in ("tiny-full", "smol-full") or data.get("workload") != workload
                or data.get("qualification") != "REFERENCE_BOUNDARIES_ONLY"):
            raise ValueError("unsupported segment-chain schema or qualification")
        for key in ("elf", "contract", "capture_receipt"):
            if self.sums.get(data.get(key)) != identity(data.get(key + "_sha256")):
                raise ValueError("segment-chain identity differs: " + key)
        self.kernel_count = integer(data.get("kernel_count"), 1)
        self.placement = placement(data)
        self.model_profile = data.get("model_profile")
        if (not isinstance(self.model_profile, dict)
                or set(self.model_profile) != {"revision", "backend", "curve_sha256"}
                or self.model_profile["revision"] != "9e6313bddecb8ba936206eb347f5cd4e5335282c"
                or self.model_profile["backend"] != "curve"):
            raise ValueError("segment chain requires the pinned model and curve backend")
        identity(self.model_profile["curve_sha256"])
        self.layout = regions(data.get("live_regions"), self.placement)
        self.layout_admitted = False
        self.boundaries = data.get("boundaries")
        if not isinstance(self.boundaries, list) or len(self.boundaries) < 2:
            raise ValueError("at least two captured boundaries required")
        if len(self.boundaries) > self.kernel_count + 1:
            raise ValueError("more boundaries than launches")
        previous = -1
        for index, boundary in enumerate(self.boundaries):
            if not isinstance(boundary, dict) or boundary.get("index") != index or type(boundary["index"]) is not int:
                raise ValueError("segment boundary order differs")
            cursor = integer(boundary.get("launch_cursor"))
            if cursor <= previous or cursor > self.kernel_count or index == 0 and cursor != 0:
                raise ValueError("segment launch partition differs")
            previous = cursor
            payloads = boundary.get("regions")
            if not isinstance(payloads, dict) or set(payloads) != set(self.layout):
                raise ValueError("segment boundary live-region coverage differs")
            for name, member in payloads.items():
                if len(self.read(member)) != self.layout[name]["bytes"]:
                    raise ValueError("segment boundary extent differs: " + name)
        if previous != self.kernel_count:
            raise ValueError("segment chain does not cover the final launch")
        outputs = data.get("final_outputs")
        if not isinstance(outputs, list) or not outputs or len(outputs) != len(set(outputs)):
            raise ValueError("segment final-output inventory required")
        if any(name not in self.layout for name in outputs):
            raise ValueError("segment final output is not captured live state")

    def admit_layout(self, expected):
        """The workload contract supplies this inventory, never the snapshot."""
        if regions(expected, self.placement) != self.layout:
            raise ValueError("segment live inventory differs from workload contract")
        self.layout_admitted = True

    def require_layout(self):
        if not self.layout_admitted:
            raise ValueError("workload live-region admission required before execution")

    def restore(self, index, write):
        integer(index)
        self.require_layout()
        if index >= len(self.boundaries) - 1:
            raise ValueError("segment start index outside chain")
        for name, member in self.boundaries[index]["regions"].items():
            write(self.layout[name]["address"], self.read(member))

    def compare(self, index, read):
        """Compare the actual guest end state with the next captured boundary."""
        integer(index)
        self.require_layout()
        if index >= len(self.boundaries) - 1:
            raise ValueError("segment result index outside chain")
        result = []
        for name, member in self.boundaries[index + 1]["regions"].items():
            region = self.layout[name]
            actual = read(region["address"], region["bytes"])
            expected = self.read(member)
            if actual != expected:
                offset = next((i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
                              min(len(actual), len(expected)))
                raise ValueError(f"segment {index} region {name} differs at byte {offset}")
            result.append(dict(name=name, bytes=len(actual), sha256=hashlib.sha256(actual).hexdigest(), exact=True))
        return result

    def verify_receipts(self, receipts):
        """Require fresh live execution evidence for every link, in order."""
        self.require_layout()
        if not isinstance(receipts, list) or len(receipts) != len(self.boundaries) - 1:
            raise ValueError("incomplete segment execution chain")
        cycles = 0
        for index, receipt in enumerate(receipts):
            start, end = self.boundaries[index:index + 2]
            if (receipt.get("schema") != "atlas.segment_execution.v1"
                    or receipt.get("status") != "PASS_SEGMENT_WITH_MODEL_DEVICE"
                    or receipt.get("segmented") is not True or receipt.get("continuous") is not False
                    or receipt.get("host_elf_executed") is not True or receipt.get("fresh_device") is not True
                    or receipt.get("source_bookends_equal") is not True
                    or receipt.get("chain_sha256sums_sha256") != self.checksums_sha256
                    or receipt.get("segment_index") != index or type(receipt.get("segment_index")) is not int
                    or receipt.get("launch_range") != [start["launch_cursor"], end["launch_cursor"]]
                    or receipt.get("elf_sha256") != self.data["elf_sha256"]
                    or receipt.get("contract_sha256") != self.data["contract_sha256"]
                    or receipt.get("model_profile") != self.model_profile):
                raise ValueError("segment execution binding differs")
            observed = receipt.get("boundaries")
            expected = {name: self.sums[member] for name, member in end["regions"].items()}
            if not isinstance(observed, dict) or observed != expected:
                raise ValueError("segment execution boundary coverage differs")
            kernels = receipt.get("kernels")
            if not isinstance(kernels, list) or len(kernels) != end["launch_cursor"] - start["launch_cursor"]:
                raise ValueError("segment execution launch count differs")
            for ordinal, kernel in enumerate(kernels, start["launch_cursor"]):
                if (type(kernel.get("index")) is not int or kernel["index"] != ordinal
                        or kernel.get("exact") is not True or kernel.get("drained") is not True):
                    raise ValueError("segment kernel is not ordered, exact and drained")
                cycles += integer(kernel.get("cycles"), 1)
        self.verify_bookends()
        return dict(status="PASS_SEGMENTED_CHAIN", segmented=True, continuous=False,
                    segments=len(receipts), launches=self.kernel_count, measured_kernel_cycles=cycles,
                    final_outputs=self.data["final_outputs"], hardware_execution=False)
