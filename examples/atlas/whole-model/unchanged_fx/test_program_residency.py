"""Real generic C policy, persistent physical IMEM and actual trampoline execution."""
import os
from pathlib import Path
import tempfile
import unittest

from program_residency import NativeProgramResidency

ROOT = Path(__file__).resolve().parents[5]
ATLAS = ROOT / "atlas-mlir"


def raw(*words):
    return b"".join(word.to_bytes(4, "little") for word in words)


class NativeResidencyTest(unittest.TestCase):
    def test_exact_bytes_capacity_fallback_busy_and_closed_session(self):
        a, b, c = [raw(word, 0x73) for word in (0x100093, 0x200093, 0x300093)]
        absolute = raw(0x17, 0x73)
        with tempfile.TemporaryDirectory() as directory:
            active = [False]
            bridge = NativeProgramResidency(ATLAS, [a, b, c, absolute], directory,
                                           capacity=2, busy=lambda: active[0])
            first, one = bridge.prepare(a)
            second, two = bridge.prepare(b)
            repeated, three = bridge.prepare(bytes(bytearray(a)))
            self.assertEqual(three["body_loads"], 2)
            self.assertEqual(three["hits"], 1)
            self.assertEqual(one["entry_word"], three["entry_word"])
            self.assertEqual(first[8:16], repeated[8:16])
            self.assertEqual(second[16:24], b)
            active[0] = True
            with self.assertRaisesRegex(AssertionError, "preparation failed"):
                bridge.prepare(c)
            active[0] = False
            _, replacement = bridge.prepare(c)
            self.assertEqual(replacement["evictions"], 1)
            fallback, placement = bridge.prepare(absolute)
            self.assertTrue(placement["fallback"])
            self.assertEqual(fallback[:8], absolute)
            _, new = bridge.prepare(a)
            self.assertEqual(new["body_loads"], 4)
            with self.assertRaisesRegex(AssertionError, "publication"):
                bridge.prepare(raw(0x400093, 0x73))
            result = bridge.finish()
            self.assertEqual(result["trampoline_words_written"], 10)
            self.assertEqual(result["fallback_words_written"], 2)
            with self.assertRaisesRegex(ValueError, "closed"):
                bridge.prepare(a)

    def test_functional_executes_complete_physical_imem_from_reset_pc(self):
        from resident_runtime import ResidentFunctionalModelRuntime, SelectedDelayInterpreter
        model_root = os.environ.get("GATES_MODEL_ROOT")
        if not model_root:
            self.skipTest("selected instruction-model source required")
        a = raw(0x02a00093, 0x73)
        b = raw(0x00700093, 0x73)
        from hashlib import sha256
        plan = dict(schema="atlas.layout_device.v1", input_addresses=[0x90000000],
                    input_payload_bytes=[32], output_addresses=[0x90000020], output_bytes=32)
        with tempfile.TemporaryDirectory() as directory:
            runtime = ResidentFunctionalModelRuntime(ATLAS, Path(model_root), [a, b], Path(directory),
                image_plans={sha256(value).hexdigest(): plan for value in (a, b)},
                memory_bytes=128, trace_payloads=False)
            for image, expected in ((a, 42), (b, 7), (a, 42)):
                for index in range(len(image) // 4):
                    runtime.write(None, 0x20000 + index * 4,
                                  int.from_bytes(image[index * 4:index * 4 + 4], "little"))
                runtime.write(None, 0x40018, 1)
                self.assertEqual(runtime.read(None, 0x40008), 0)
                self.assertEqual(runtime.read(None, 0x40008), 5)
                mapped = runtime.mapped_programs[runtime.launches[-1]["residency"]["mapped_image_sha256"]]
                actual = SelectedDelayInterpreter(runtime.functional_config).run(mapped)
                self.assertEqual(actual.state.read_xrf(1), expected)
                self.assertEqual(actual.instructions_executed, 4)
            result = runtime.finish()
            self.assertEqual(result["program_residency"]["body_loads"], 2)
            self.assertEqual(result["program_residency"]["hits"], 1)
            self.assertEqual(result["delay_slot_sentinel"], dict(x10=2, x12=1, x11=3))
            self.assertEqual(result["total_instructions"], 12)


if __name__ == "__main__":
    unittest.main()
