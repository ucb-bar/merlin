"""CPU-only Tiny completion and retained-fixture binding controls."""
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest

sys.path[:0] = [str(Path(__file__).absolute().parents[1] / "tools" / "host_iss"),
                str(Path(__file__).absolute().parents[1] / "tools")]
import tiny_contract as tiny
from bundle import EPILOGUE


class Image:
    raw = b"test ELF identity"

    def __init__(self):
        self.symbols, self.memory, self.next = {}, {}, 0x80000000

    def add(self, name, raw, size=None):
        address = self.next
        self.next += len(raw) + 128
        self.symbols[name] = dict(address=address, size=len(raw) if size is None else size)
        self.memory[address] = raw
        return address

    def bytes_at(self, address, count):
        for start, raw in self.memory.items():
            if start <= address and address + count <= start + len(raw):
                return raw[address-start:address-start+count]
        raise ValueError("unmapped test range")

    def instructions(self, name):
        assert name == "main"
        return [(0x80001000, 4, 0x10500073)]


class Bundle:
    kernel_count = 24

    def __init__(self):
        self.files, self.sums, self.paths = {}, {}, []
        self.provenance = dict(workload="tiny", contract="contract.json")

    def add(self, name, raw):
        self.files[name], self.sums[name] = raw, tiny.digest(raw)

    def read(self, name):
        raw = self.files[name]
        assert tiny.digest(raw) == self.sums[name]
        return raw

    def kernel_paths(self, index):
        return self.paths[index]

    def contract(self, data):
        self.add("contract.json", json.dumps(data).encode())
        self.provenance["contract_sha256"] = self.sums["contract.json"]


def fixture():
    image, bundle = Image(), Bundle()
    image.add("trap_entry", bytes(56) + b"\x01\xa0", size=0)
    image.add("atlas_test_mailbox", struct.pack("<16Q", tiny.MAGIC, 2, 24, 23, 0, 0, 1, *([0]*9)))
    image.add("tohost", struct.pack("<Q", 1))
    original, expected = struct.pack("<2I", 0x13, 0x73), b"\xab\xcd"
    program_address = image.add("program", original)
    expected_address = image.add("panel_expected", expected)
    image.add("tiny_panel_expected", struct.pack("<24Q", *([expected_address]*24)))
    panels, programs = [], []
    for index in range(24):
        phase = index // 12
        geometry = dict(rows=1 if phase else 128, physical_rows=32 if phase else 128,
                        k=2048, n=2048, column=0, width=2048, logical=0, a_scale=-6)
        memory = dict(dram_base=0x100000000, output_offset=8, total_dram_bytes=16, output_bytes=2)
        dma = dict(load_bytes=8, store_bytes=2)
        paths = dict(source=f"{index}/program.S", original=f"{index}/program.bin", initial=f"{index}/initial.bin",
                     expected=f"{index}/expected.bin", output_offset=8, dram_base=0x100000000, requested_dma=dma)
        files = {}
        for key, name, raw in (("source", "program.S", b"nop\necall\n"), ("original", "program.bin", original),
                               ("initial", "initial-dram.bin", bytes(16)), ("expected", "expected-bf16-tiles.bin", expected)):
            bundle.add(paths[key], raw)
            files[name] = dict(sha256=tiny.digest(raw))
        bundle.paths.append(paths)
        programs.append(dict(index=index, phase=phase, sequence=index+1, geometry=geometry, memory=memory,
                             files=files, requested_dma=dma, runtime_IMEM_sha256=tiny.digest(original[:-4]+EPILOGUE),
                             runtime_IMEM_words=7))
        panels.append(struct.pack("<6QIi8Q", *geometry.values(), 0, 0, program_address, 2, 8, 16, 8, 2))
    image.add("tiny_panels", b"".join(panels))
    assets, pointers = [], {}
    for index, (name, (phase, table, size)) in enumerate(tiny.FINAL_BUFFERS.items()):
        raw = bytes([index+1])*size
        image.add(name, raw)
        blob = "reference_"+name
        address = image.add(blob, raw)
        pointers.setdefault(table, [None, None])[phase] = address
        assets.append(dict(actual_symbol=name, phase=phase, expected_pointer_table=table, bytes=size,
                           expected_blob_symbol=blob, expected_ELF_address=address, expected_blob=dict(sha256=tiny.digest(raw))))
    for name, addresses in pointers.items():
        image.add(name, struct.pack("<2Q", *addresses))
    data = dict(ELF=dict(sha256=tiny.digest(image.raw)), finite_model_cycle_cap_per_start=tiny.CYCLE_CAP,
                mailbox=dict(symbol="atlas_test_mailbox"), programs=programs,
                completion_WFI_sites=[dict(pc=0x80001000, length=4)],
                trap_loop_sites=[dict(pc=image.symbols["trap_entry"]["address"]+56, length=2)],
                final_actual_C_output_KV_assets=assets)
    bundle.add("host/build.json", json.dumps(dict(status="compiled_unexecuted_Tiny_firstblock_prefill_decode_RV64_ELF",
        actual_returncode=0, source_bookends_verified=True, host_elf_executed=False, platform_executed=False,
        elf=dict(sha256=tiny.digest(image.raw)))).encode())
    bundle.contract(data)
    return image, bundle, data, types.SimpleNamespace(completed=[dict(index=i, sequence=i+1, exact=True, drained=True, reset_count=1) for i in range(24)])


class TinyContractTests(unittest.TestCase):
    def setUp(self):
        self.image, self.bundle, self.data, self.device = fixture()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.remote = types.SimpleNamespace(read=self.image.bytes_at)

    def verify(self):
        return tiny.verify_completion(self.remote, self.image, self.device, self.bundle, self.out)

    def test_complete_admission_and_all_guest_buffers(self):
        tiny.validate_bundle(self.bundle, self.image)
        self.assertTrue(self.verify()["all_final_guest_buffers_exact"])
        self.assertEqual(len(list(self.out.glob("*.bin"))), 7)

    def test_all_mailbox_words_including_traps_and_reserved(self):
        address = self.image.symbols["atlas_test_mailbox"]["address"]
        original = self.image.memory[address]
        for index in range(16):
            with self.subTest(word=index):
                words = list(struct.unpack("<16Q", original))
                words[index] ^= 1
                self.image.memory[address] = struct.pack("<16Q", *words)
                with self.assertRaises(AssertionError): self.verify()
        self.image.memory[address] = original

    def test_every_final_output_and_cache_compared(self):
        for name in tiny.FINAL_BUFFERS:
            with self.subTest(buffer=name):
                address = self.image.symbols[name]["address"]
                raw = self.image.memory[address]
                self.image.memory[address] = bytes([raw[0]^1])+raw[1:]
                with self.assertRaises(AssertionError): self.verify()
                self.image.memory[address] = raw

    def test_tohost_and_launch_ledger(self):
        address = self.image.symbols["tohost"]["address"]
        self.image.memory[address] = struct.pack("<Q", 3)
        with self.assertRaises(AssertionError): self.verify()
        self.image.memory[address] = struct.pack("<Q", 1)
        original = self.device.completed
        for completed in (original[:-1], list(reversed(original)), [dict(index=i, sequence=i+2) for i in range(24)]):
            self.device.completed = completed
            with self.assertRaises(AssertionError): self.verify()

    def test_short_tohost_and_unqualified_device_ledger_refused(self):
        address = self.image.symbols["tohost"]["address"]
        self.remote.read = lambda observed, size: b"\x01" if observed == address else self.image.bytes_at(observed, size)
        with self.assertRaises(AssertionError): self.verify()
        self.remote.read = self.image.bytes_at
        for change in (dict(exact=False), dict(drained=False), dict(reset_count=2)):
            original = dict(self.device.completed[0])
            self.device.completed[0].update(change)
            with self.assertRaises(AssertionError): self.verify()
            self.device.completed[0] = original

    def test_contract_digest_refused(self):
        self.bundle.provenance["contract_sha256"] = "0"*64
        with self.assertRaises(AssertionError): tiny.validate_bundle(self.bundle, self.image)

    def test_phase_dma_program_and_reference_admission(self):
        for mutate in (lambda d: d["programs"][12]["geometry"].update(rows=128),
                       lambda d: d["programs"][0].update(runtime_IMEM_sha256="0"*64),
                       lambda d: d["programs"][0]["requested_dma"].update(load_bytes=999),
                       lambda d: d["programs"][0]["files"]["program.S"].update(sha256="0"*64),
                       lambda d: d["final_actual_C_output_KV_assets"].pop()):
            data = copy.deepcopy(self.data)
            mutate(data)
            self.bundle.contract(data)
            with self.assertRaises(AssertionError): tiny.validate_bundle(self.bundle, self.image)

    def test_actual_elf_and_trap_bytes_refused(self):
        self.image.raw += b"changed"
        with self.assertRaises(AssertionError): tiny.validate_bundle(self.bundle, self.image)
        self.image.raw = b"test ELF identity"
        address = self.image.symbols["trap_entry"]["address"]
        self.image.memory[address] = bytes(58)
        with self.assertRaises(AssertionError): tiny.stop_sites(self.image)

    def test_embedded_program_and_final_reference_refused(self):
        for name in ("program", "reference_decode_values"):
            address = self.image.symbols[name]["address"]
            raw = self.image.memory[address]
            self.image.memory[address] = bytes(len(raw))
            with self.assertRaises(AssertionError): tiny.validate_bundle(self.bundle, self.image)
            self.image.memory[address] = raw


if __name__ == "__main__":
    unittest.main()
