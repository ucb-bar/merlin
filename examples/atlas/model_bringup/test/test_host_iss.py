"""CPU-only controls for the portable ResNet ISS bridge; no QEMU/Core runs."""

import hashlib
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock

TOOLS = Path(__file__).absolute().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "host_iss"))
from bundle import Bundle, EPILOGUE, relative_name, runtime_program, verify_completion
from elf_image import memory_instruction, mmio_sites
from gdb_remote import Remote, decode_response, frame
from model_device import Device
from bundle_payload_fixture import write_bundle_payloads


class BundleTest(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        files = {"host/host.elf": b"fixed elf",
            "provenance.json": json.dumps({"schema": "atlas.full_resnet_bundle.v3", "kernel_count": 54}).encode()}
        files["host/build.json"] = json.dumps(dict(returncode=0, source_bookends_equal=True, kernel_count=54,
            elf_sha256=hashlib.sha256(files["host/host.elf"]).hexdigest())).encode()
        for index in range(54):
            for name in ("receipt.json", "program.S", "program.bin", "initial-dram.bin", "expected-bf16-tiles.bin"):
                files[f"kernels/{index:02d}/model/{name}"] = b"test payload"
        write_bundle_payloads(root, files)
        return root

    def test_relocation_and_changed_payload(self):
        root = self.fixture()
        bundle = Bundle(root)
        self.assertEqual(bundle.read("host/host.elf"), b"fixed elf")
        (root / "host/host.elf").write_bytes(b"changed elf")
        with self.assertRaises(ValueError):
            bundle.read("host/host.elf")
        with self.assertRaises(ValueError):
            bundle.verify_bookends()

    def test_duplicate_or_changed_checksums(self):
        root = self.fixture()
        bundle = Bundle(root)
        sums = root / "SHA256SUMS"
        old = sums.read_text()
        sums.write_text(old + old.splitlines()[0] + "\n")
        with self.assertRaises(ValueError):
            Bundle(root)
        with self.assertRaises(ValueError):
            bundle.verify_bookends()

    def test_paths_rejected_before_access(self):
        for name in ("../x", "/x", "a//x", "./x", "a/./b", "a\\b", "a:b", "a\nx"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                relative_name(name)

    def test_symlink_payload_refused(self):
        root = self.fixture()
        path = root / "host/host.elf"
        path.unlink()
        path.symlink_to(root / "provenance.json")
        with self.assertRaises(ValueError):
            Bundle(root)

    def test_mutated_manifest_and_missing_payload(self):
        root = self.fixture()
        sums = root / "SHA256SUMS"
        manifest = root / "provenance.json"
        manifest.write_text(json.dumps({"schema": "atlas.full_resnet_bundle.v3", "kernel_count": 53}))
        lines = sums.read_text().splitlines()
        sums.write_text("\n".join((hashlib.sha256(manifest.read_bytes()).hexdigest() + "  provenance.json"
            if line.endswith("  provenance.json") else line) for line in lines) + "\n")
        with self.assertRaises(ValueError):
            Bundle(root)
        (root / "kernels/00/model/program.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            Bundle(root)

    def test_traversal_in_checksum_is_refused(self):
        root = self.fixture()
        (root / "SHA256SUMS").write_text("0" * 64 + "  ../outside\n")
        with self.assertRaises(ValueError):
            Bundle(root)

    def test_unlisted_payload_and_missing_case(self):
        root = self.fixture()
        bundle = Bundle(root)
        with self.assertRaises(ValueError):
            bundle.read("other.bin")
        sums = root / "SHA256SUMS"
        sums.write_text("\n".join(l for l in sums.read_text().splitlines() if "kernels/53/model/program.S" not in l) + "\n")
        with self.assertRaises(ValueError):
            Bundle(root)


class ProgramTest(unittest.TestCase):
    def test_exact_epilogue_and_mismatch(self):
        original = struct.pack("<2I", 0x13, 0x73)
        program = Mock()
        program.assemble.return_value = struct.unpack("<7I", original[:-4] + EPILOGUE)
        assemble = Mock(return_value=program)
        result, words = runtime_program("nop\necall\n", original, assemble)
        self.assertIs(result, program)
        self.assertEqual(words, original[:-4] + EPILOGUE)
        self.assertEqual(assemble.call_args.args[0].getvalue(),
            "nop\nnop\ncsrrs x31, 0xc11, x0\nnop\ncsrrw x0, 0xc10, x31\nnop\necall\n")
        program.assemble.return_value = [0x73]
        with self.assertRaises(ValueError):
            runtime_program("ecall", original, assemble)
        for source, raw in [("nop", original), ("ecall", b"bad"), ("ecall", struct.pack("<I", 0x13))]:
            with self.subTest(source=source, raw=raw), self.assertRaises(ValueError):
                runtime_program(source, raw, assemble)

    def test_real_instruction_decode_and_site_count(self):
        lw = memory_instruction(0x0005A503, 4)
        self.assertEqual(lw, dict(store=False, base=11, reg=10, offset=0, signed=True))
        self.assertFalse(memory_instruction(0x0005E503, 4)["signed"])
        self.assertEqual(memory_instruction(0x00C5A023, 4)["reg"], 12)
        image = Mock()
        image.instructions.side_effect = [[(0x100, 4, 0x0005A503)], [(0x200, 4, 0x00C5A023)]]
        self.assertEqual(set(mmio_sites(image)), {0x100, 0x200})
        image.instructions.side_effect = [[(0x100, 4, 0x0005A503)] * 2]
        with self.assertRaises(AssertionError):
            mmio_sites(image)

    def test_mmio_imem_and_csr_refusals(self):
        device = Device.__new__(Device)
        device.imem = bytearray(128 * 1024)
        device.written = set()
        device.core = Mock()
        device.core.arch_state.halted = True
        device.write32(0x20000, 0x1234)
        self.assertEqual(device.read32(0x20000), 0x1234)
        self.assertEqual(device.written, {0})
        device.write32(0x40010, 7)
        device.core.arch_state.write_csrf.assert_called_with(0xC10, 7)
        for address, value in ((0x20001, 0), (0x40018, 0), (0x4000C, 0), (0x40010, 1 << 32)):
            with self.subTest(address=address), self.assertRaises(AssertionError):
                device.write32(address, value)
        device.core.arch_state.halted = False
        with self.assertRaises(AssertionError):
            device.write32(0x20000, 0)
        with self.assertRaises(AssertionError):
            device.read32(0x50000)

    def test_complete_mailbox_and_every_failure_boundary(self):
        mailbox = bytearray(928)
        struct.pack_into("<10I", mailbox, 0, 0x41544C53, 3, 0, 0, 54, 0, 0, 0, 0xFFFFFFFF, 54)
        struct.pack_into("<54I", mailbox, 40, *range(1, 55))
        completed = [dict(index=i, sequence=i + 1) for i in range(54)]
        logits = bytes(4000)
        verify_completion(mailbox, 1, logits, logits, completed)
        for offset in (0, 4, 8, 12, 16, 24, 28, 32, 36, 40, 252, 904, 927):
            bad = bytearray(mailbox)
            bad[offset] ^= 1
            with self.subTest(offset=offset), self.assertRaises(AssertionError):
                verify_completion(bad, 1, logits, logits, completed)
        for tohost, actual, done in ((0, logits, completed), (1, b"x" + logits[1:], completed), (1, logits, completed[:-1]),
            (1, logits, list(reversed(completed)))):
            with self.assertRaises(AssertionError):
                verify_completion(mailbox, tohost, actual, logits, done)


class ProtocolTest(unittest.TestCase):
    def remote(self):
        # In-memory duplex transport keeps protocol checks independent of local
        # socket permissions while retaining fragmentation and finite waits.
        class Endpoint:
            def __init__(self):
                self.buffer = bytearray()
                self.timeout = None

            def sendall(self, data):
                self.peer.buffer.extend(data)

            def recv(self, count):
                if not self.buffer:
                    raise TimeoutError("bounded transport read")
                data = bytes(self.buffer[:count])
                del self.buffer[:count]
                return data

            def settimeout(self, timeout):
                self.timeout = timeout

        client, server = Endpoint(), Endpoint()
        client.peer, server.peer = server, client
        remote = Remote.__new__(Remote)
        remote.socket = client
        remote.trace = None
        remote.timeout = 1
        return remote, server

    def test_frame_escape_rle_and_buffered_ack(self):
        self.assertEqual(frame(b"?"), b"$?#3f")
        self.assertEqual(decode_response(b"0*}"), b"0" * 97)
        self.assertEqual(decode_response(b"}\x03}\x04}]}\x0a"), b"#$}*")
        remote, server = self.remote()
        server.sendall(b"+" + frame(b"0000008000000000"))
        self.assertEqual(remote.reg(32), 0x80000000)
        expected = frame(b"p20") + b"+"
        actual = b""
        while len(actual) < len(expected):
            actual += server.recv(len(expected) - len(actual))
        self.assertEqual(actual, expected)

    def test_checksum_error_and_deadline(self):
        remote, server = self.remote()
        server.sendall(b"$OK#00")
        with self.assertRaises(ValueError):
            remote.receive()
        self.assertEqual(server.recv(1), b"-")
        remote.deadline = time.monotonic() - 1
        remote._buffer = b"+"
        remote._offset = 0
        self.assertEqual(remote.byte(), b"+")
        with self.assertRaises(TimeoutError):
            remote.receive()
        with self.assertRaises(TimeoutError):
            remote.send(b"?")

    def test_partial_packet_wait_is_bounded(self):
        remote, server = self.remote()
        remote.deadline = time.monotonic() + 0.03
        server.sendall(b"$OK")
        with self.assertRaises(TimeoutError):
            remote.receive()
        self.assertGreater(remote.socket.timeout, 0)
        self.assertLessEqual(remote.socket.timeout, 0.03)

    def test_malformed_escape_or_rle(self):
        for raw in (b"* ", b"0*", b"}", b"0*\x1d"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_response(raw)


if __name__ == "__main__":
    unittest.main()
