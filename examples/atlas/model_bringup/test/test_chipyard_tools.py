"""Fast, synthetic controls for the portable SoC build/run/result tools.

Compiler and simulator fixtures exercise plumbing only; these tests make no
hardware execution or numerical qualification claim.
"""

import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

TOOLS = Path(__file__).absolute().parents[1] / "tools"
sys.path.insert(0, str(TOOLS / "chipyard"))
import resnet_kernels as kernels
import run_sim
from atlas_host_build import elf_sections
import kernel_library  # noqa: E402,F401
from atlas_tooling import sha


class TemporaryTree(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="atlas-soc-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, path, raw):
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(raw, str):
            path.write_text(raw)
        else:
            path.write_bytes(raw)
        return path

    def json(self, path, value):
        return self.write(path, json.dumps(value))


class KernelToolsTest(TemporaryTree):
    def setUp(self):
        super().setUp()
        self.name = "kernel-00-tiny"
        self.kernel = self.root / self.name
        self.program = struct.pack("<2I", 0x13, 0x73)
        self.expected = b"expected"
        self.initial = b"\xa5" * 256 + b"\xcd" * 8 + b"\x5a" * 256
        for name, raw in (("program.bin", self.program), ("initial-dram.bin", self.initial),
                          ("expected-bf16-tiles.bin", self.expected)):
            self.write(self.kernel / "model" / name, raw)
        self.compiler = self.write(self.root / "fixture-cc", b"synthetic compiler input")

    def compile_fixture(self, command, **kwargs):
        # A deterministic stand-in for linking; no cross compiler is required.
        out = Path(command[command.index("-o") + 1])
        payload = b"".join((out.parent / name).read_bytes() for name in
                           ("program.bin", "initial-dram.bin", "expected.bin"))
        out.write_bytes(b"synthetic ELF fixture\0" + payload)
        self.assertEqual(kwargs["env"]["TMPDIR"], str(out.parent))
        return SimpleNamespace(returncode=0)

    def build(self, out):
        with mock.patch.object(kernels.subprocess, "run", side_effect=self.compile_fixture):
            return kernels.build_kernel(self.kernel, out, self.compiler, max_polls=7, wall_seconds=2)

    def test_real_compiler_reproducibility_and_elf_loadmem(self):
        compiler = os.environ.get("RISCV_CC") or shutil.which("riscv64-unknown-elf-gcc")
        if compiler is None:
            self.skipTest("set RISCV_CC or provide riscv64-unknown-elf-gcc for real ELF checks")
        outputs = [self.root / name for name in ("real-first", "real-second")]
        receipts = [kernels.build_kernel(self.kernel, out, compiler, max_polls=7, wall_seconds=30)
                    for out in outputs]
        for receipt in receipts:
            self.assertEqual(receipt["returncode"], 0, (Path(receipt["command"][-1]).parent / "build.log").read_text())
        first, second = [(out / "host.elf").read_bytes() for out in outputs]
        self.assertEqual(first, second)
        self.assertEqual(receipts[0]["elf_sha256"], receipts[1]["elf_sha256"])
        self.assertEqual(json.loads((outputs[0] / "build.json").read_text()), receipts[0])
        self.assertIn("#define PAYLOAD_MAX_POLLS 7u", (outputs[0] / "payload.h").read_text())
        self.assertEqual(receipts[0]["output_offset"], 256)
        self.assertEqual(receipts[0]["output_bytes"], len(self.expected))
        _, sections, symbols = elf_sections(first)
        dram = sections[".atlas_dram"]
        self.assertEqual((dram[1], dram[3]), (1, kernels.ATLAS_DRAM_BASE))
        self.assertEqual(first[dram[4]:dram[4] + dram[5]], self.initial)
        shoff, = struct.unpack_from("<Q", first, 40)
        shsize, = struct.unpack_from("<H", first, 58)
        for symbol, raw in (("atlas_program", self.program), ("atlas_expected", self.expected)):
            address, _, index = symbols[symbol]
            section = struct.unpack_from("<IIQQQQIIQQ", first, shoff + index * shsize)
            offset = section[4] + address - section[3]
            self.assertEqual(first[offset:offset + len(raw)], raw)

    def test_build_refuses_invalid_program_bounds_and_reused_output(self):
        with mock.patch.object(kernels.subprocess, "run") as compiler:
            for value in (0, 100000001):
                with self.subTest(polls=value), self.assertRaises(ValueError):
                    kernels.build_kernel(self.kernel, self.root / "out", self.compiler, max_polls=value)
            self.write(self.kernel / "model/program.bin", struct.pack("<2I", 0x73, 0x73))
            with self.assertRaises(ValueError):
                kernels.build_kernel(self.kernel, self.root / "out", self.compiler)
            compiler.assert_not_called()
        self.write(self.kernel / "model/program.bin", self.program)
        self.build(self.root / "out")
        with self.assertRaises(ValueError):
            self.build(self.root / "out")

    def test_build_source_change_and_deadline_cannot_report_success(self):
        def changed(command, **kwargs):
            result = self.compile_fixture(command, **kwargs)
            self.compiler.write_bytes(b"changed compiler")
            return result
        with mock.patch.object(kernels.subprocess, "run", side_effect=changed):
            receipt = kernels.build_kernel(self.kernel, self.root / "changed", self.compiler)
        self.assertFalse(receipt["source_bookends_equal"])
        self.assertNotEqual(receipt["returncode"], 0)
        self.assertNotIn("elf_sha256", receipt)
        with mock.patch.object(kernels.subprocess, "run", side_effect=subprocess.TimeoutExpired("cc", 1)):
            receipt = kernels.build_kernel(self.kernel, self.root / "deadline", self.compiler, wall_seconds=1)
        self.assertEqual(receipt["returncode"], 124)
        self.assertNotIn("elf_sha256", receipt)

    def test_output_guard_extent_uniqueness_and_alignment(self):
        self.assertEqual(kernels.output_region(self.initial, 8), 256)
        for raw, size in ((self.initial * 2, 8), (b"x" + self.initial, 8), (self.initial, 0),
                          (self.initial, 7), (self.initial[:-1], 8)):
            with self.subTest(size=size, length=len(raw)), self.assertRaises(ValueError):
                kernels.output_region(raw, size)

    def test_batch_manifest_and_result_rejection_matrix(self):
        build_root, runs = self.root / "build", self.root / "runs"
        receipt = self.build(build_root / self.name)
        self.json(build_root / "batch.json", {"kernels": [self.name], "kernel_count": 1})
        run = runs / self.name
        metadata = dict(elf_sha256=receipt["elf_sha256"], returncode=0, timed_out=False)
        good = f"ATLAS_KERNEL_PASS name={self.name} device_cycles=123\n"
        cases = [("exact", good, 0, metadata, True), ("missing-pass", "", 0, metadata, False),
                 ("duplicate", good * 2, 0, metadata, False),
                 ("other-kernel", good.replace(self.name, "kernel-other"), 0, metadata, False),
                 ("failure", good + "ATLAS_KERNEL_FAIL phase=bad\n", 0, metadata, False),
                 ("exit", good, 1, metadata, False),
                 ("elf", good, 0, dict(metadata, elf_sha256="0" * 64), False),
                 ("status", good, 0, dict(metadata, returncode=130), False),
                 ("timeout", good, 0, dict(metadata, timed_out=True), False),
                 ("missing-metadata", good, 0, None, False)]
        for label, log, code, status, exact in cases:
            with self.subTest(case=label):
                self.write(run / "sim.log", log)
                self.write(run / "exit-status.txt", str(code) + "\n")
                if status is None:
                    (run / "run.json").unlink(missing_ok=True)
                else:
                    self.json(run / "run.json", status)
                self.assertEqual(kernels.kernel_result(self.name, run, receipt)["exact"], exact)
                report = kernels.collect_results(build_root, runs)
                self.assertEqual(report["status"], "PASS" if exact else "FAIL")
                self.assertEqual(report["exact_kernel_count"], int(exact))
        self.write(build_root / self.name / "host.elf", b"mutated")
        with self.assertRaisesRegex(ValueError, "ELF changed"):
            kernels.load_batch(build_root)
        self.json(build_root / "batch.json", {"kernels": [self.name, self.name], "kernel_count": 2})
        with self.assertRaises(ValueError):
            kernels.load_batch(build_root)


class SimulatorRunnerTest(TemporaryTree):
    def setUp(self):
        super().setUp()
        self.elf = self.write(self.root / "fixture.elf", b"fixture ELF")
        self.simulator = self.write(self.root / "fixture-sim", '#!/bin/sh\nprintf "fixture ran\\n"\n')
        self.simulator.chmod(0o755)

    def test_small_executable_records_command_identity_and_success(self):
        out = self.root / "success"
        result = run_sim.run_simulator(self.simulator, self.elf, out, wall_seconds=2, max_cycles=17,
                                       extra_args=["+fixture=1"])
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["elf_sha256"], sha(self.elf.read_bytes()))
        self.assertEqual((out / "sim.log").read_text(), "fixture ran\n")
        self.assertEqual((out / "exit-status.txt").read_text(), "0\n")
        self.assertEqual(json.loads((out / "run.json").read_text()), result)
        self.assertIn("+max-cycles=17", result["command"])
        self.assertIn("+loadmem=" + str(self.elf), result["command"])
        with self.assertRaises(ValueError):
            run_sim.run_simulator(self.simulator, self.elf, out)

    def test_timeout_interrupt_and_forced_cleanup_record_failure(self):
        for label, first, cleanup, code, signals in (
                ("timeout", subprocess.TimeoutExpired("sim", 1), None, 124, [signal.SIGINT]),
                ("interrupt", KeyboardInterrupt(), None, 130, [signal.SIGINT]),
                ("forced", subprocess.TimeoutExpired("sim", 1), subprocess.TimeoutExpired("sim", 20),
                 124, [signal.SIGINT, signal.SIGKILL])):
            with self.subTest(case=label):
                process = mock.MagicMock(pid=123, returncode=-2)
                process.__enter__.return_value = process
                process.wait.side_effect = [first, cleanup, None] if cleanup else [first, None]
                with mock.patch.object(run_sim.subprocess, "Popen", return_value=process) as popen, \
                        mock.patch.object(run_sim, "signal_group") as kill:
                    result = run_sim.run_simulator(self.simulator, self.elf, self.root / label, wall_seconds=1)
                self.assertEqual(result["returncode"], code)
                self.assertEqual(result["timed_out"], code == 124)
                self.assertEqual([call.args[1] for call in kill.call_args_list], signals)
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                self.assertEqual(json.loads((self.root / label / "run.json").read_text())["returncode"], code)
                self.assertEqual((self.root / label / "exit-status.txt").read_text(), str(code) + "\n")

    def test_reserved_plusargs_rejected_before_launch(self):
        with mock.patch.object(run_sim.subprocess, "Popen") as popen:
            for arg in ("+loadmem=other", "+max-cycles=0", "+permissive", "+dramsim_ini_dir=other", "--bad"):
                with self.subTest(arg=arg), self.assertRaises(ValueError):
                    run_sim.run_simulator(self.simulator, self.elf, self.root / "bad", extra_args=[arg])
            popen.assert_not_called()
        self.assertFalse((self.root / "bad").exists())




if __name__ == "__main__":
    unittest.main()
