"""Segment-chain controls: missing state, byte corruption, gaps and profile mixing."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS = Path(__file__).absolute().parents[1] / "tools"
sys.path[:0] = [str(TOOLS), str(TOOLS / "host_iss")]
from placement import arena_extent, placement
from segment_boundary import SegmentChain
from bundle_payload_fixture import write_bundle_payloads


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class SegmentTest(unittest.TestCase):
    def fixture(self, mutate=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        files = {"host.elf": b"ELF fixture", "contract.json": b"{}", "capture.json": b"{}",
                 "start.bin": b"abcd", "middle.bin": b"efgh", "final.bin": b"ijkl"}
        data = dict(schema="atlas.segment_chain.v1", workload="tiny-full", qualification="REFERENCE_BOUNDARIES_ONLY",
                    elf="host.elf", elf_sha256=digest(files["host.elf"]), contract="contract.json",
                    contract_sha256=digest(files["contract.json"]), capture_receipt="capture.json",
                    capture_receipt_sha256=digest(files["capture.json"]), kernel_count=3,
                    live_regions=[dict(name="kv", address=0x81000000, bytes=4)], final_outputs=["kv"],
                    model_profile=dict(revision="9e6313bddecb8ba936206eb347f5cd4e5335282c", backend="curve",
                                       curve_sha256="1" * 64),
                    boundaries=[dict(index=i, launch_cursor=cursor, regions={"kv": member})
                                for i, (cursor, member) in enumerate(((0, "start.bin"), (2, "middle.bin"),
                                                                     (3, "final.bin")))])
        if mutate:
            mutate(data)
        files["segments.json"] = json.dumps(data).encode()
        write_bundle_payloads(root, files)
        return root, data

    def chain(self):
        root, data = self.fixture()
        chain = SegmentChain(root, "tiny-full")
        chain.admit_layout(data["live_regions"])
        return chain

    def receipts(self, chain):
        result = []
        for index, (begin, end) in enumerate(zip(chain.boundaries, chain.boundaries[1:])):
            result.append(dict(schema="atlas.segment_execution.v1", status="PASS_SEGMENT_WITH_MODEL_DEVICE",
                segmented=True, continuous=False, host_elf_executed=True, fresh_device=True,
                source_bookends_equal=True, chain_sha256sums_sha256=chain.checksums_sha256,
                segment_index=index, launch_range=[begin["launch_cursor"], end["launch_cursor"]],
                elf_sha256=chain.data["elf_sha256"], contract_sha256=chain.data["contract_sha256"],
                model_profile=copy.deepcopy(chain.model_profile),
                boundaries={name: chain.sums[member] for name, member in end["regions"].items()},
                kernels=[dict(index=i, exact=True, drained=True, cycles=10)
                         for i in range(begin["launch_cursor"], end["launch_cursor"])]))
        return result

    def test_live_state_inventory_is_admitted_before_restore_or_check(self):
        root, _ = self.fixture()
        chain = SegmentChain(root, "tiny-full")
        with self.assertRaisesRegex(ValueError, "admission"):
            chain.restore(0, lambda *_: None)
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            chain.admit_layout([dict(name="kv", address=0x81000000, bytes=3)])
        chain.admit_layout([dict(name="kv", address=0x81000000, bytes=4)])
        memory = {}
        chain.restore(0, lambda address, raw: memory.update({address: raw}))
        self.assertEqual(memory, {0x81000000: b"abcd"})

    def test_actual_guest_boundary_compares_every_byte(self):
        chain = self.chain()
        self.assertTrue(chain.compare(0, lambda *_: b"efgh")[0]["exact"])
        with self.assertRaisesRegex(ValueError, "byte 3"):
            chain.compare(0, lambda *_: b"efgX")
        with self.assertRaisesRegex(ValueError, "byte 3"):
            chain.compare(0, lambda *_: b"efg")

    def test_order_coverage_and_payload_tampering_fail_closed(self):
        for mutate in (lambda d: d["boundaries"][0].update(launch_cursor=1),
                       lambda d: d["boundaries"][1].update(launch_cursor=0),
                       lambda d: d["boundaries"][-1].update(launch_cursor=2),
                       lambda d: d["boundaries"][1].update(regions={}),
                       lambda d: d.update(final_outputs=["uncaptured"])):
            root, _ = self.fixture(mutate)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                SegmentChain(root, "tiny-full")
        chain = self.chain()
        (chain.root / "final.bin").write_bytes(b"ijkX")
        with self.assertRaisesRegex(ValueError, "changed bundle payload"):
            chain.compare(1, lambda *_: b"ijkl")

    def test_only_complete_same_profile_live_receipts_sum_cycles(self):
        chain = self.chain()
        receipts = self.receipts(chain)
        result = chain.verify_receipts(receipts)
        self.assertEqual((result["launches"], result["measured_kernel_cycles"]), (3, 30))
        for mutate in (lambda r: r.pop(), lambda r: r[0].update(continuous=True),
                       lambda r: r[1]["model_profile"].update(curve_sha256="2" * 64),
                       lambda r: r[0]["kernels"][1].update(index=2),
                       lambda r: r[1].update(boundaries={}),
                       lambda r: r[0]["kernels"][0].update(exact=False)):
            altered = copy.deepcopy(receipts)
            mutate(altered)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                chain.verify_receipts(altered)

    def test_placement_rejects_boolean_cross_window_and_outside_dram(self):
        self.assertEqual(placement({})["arena_base"], 0x100000000)
        for value in (dict(arena_base=True, dram_base=0x80000000, dram_bytes=1 << 31),
                      dict(arena_base=0x90000001, dram_base=0x80000000, dram_bytes=1 << 31),
                      dict(arena_base=0x100000000, dram_base=0x80000000, dram_bytes=1 << 31)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                placement(dict(placement=value))
        metadata = dict(placement=dict(arena_base=0xffffffc0, dram_base=0x80000000, dram_bytes=1 << 32))
        with self.assertRaisesRegex(ValueError, "low32"):
            arena_extent(metadata, 128)


if __name__ == "__main__":
    unittest.main()
