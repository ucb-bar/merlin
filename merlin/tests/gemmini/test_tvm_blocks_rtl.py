"""Trust-boundary checks for observable block RTL verification."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from merlin.common.paths import build_dir, repo_root


class BlockRTLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = repo_root() / "examples/gemmini/comparisons/tvm/verify_blocks_rtl.py"
        spec = importlib.util.spec_from_file_location("block_rtl_verifier", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_instrumentation_binds_actual_manifest_calls(self):
        source = "int run(){\n  if (cpu(a)) return -3;\n  if (device(b)) return -3;\n  return 0;\n}\n"
        calls = [{"symbol": "cpu"}, {"symbol": "device"}]
        result, placement = self.module.instrument_graph(source, calls, {"device"})
        self.assertEqual(placement, [0, 1])
        self.assertIn("rtl_trace(1, 0);\n  if (device(b)) return -3;\n  rtl_trace(1, 1);", result)
        for incorrect in (calls[:1], list(reversed(calls)), calls + [{"symbol": "extra"}]):
            with self.assertRaises(ValueError):
                self.module.instrument_graph(source, incorrect, {"device"})

    def test_binary_payload_is_length_delimited_even_with_marker_bytes(self):
        payload = b"\x00RTL_OUTPUT index=99 bytes=0\n\xff"
        raw = b"RTL_MODEL_END\nRTL_OUTPUT index=0 bytes=" + str(len(payload)).encode() + b"\n" + payload + b"\nRTL_OUTPUT_END\nRTL_BLOCK_PASS calls=1\n"
        records, text = self.module.read_output_records(raw)
        self.assertEqual(records, [{"index": 0, "payload": payload}])
        self.assertIn("RTL_BLOCK_PASS calls=1", text)
        self.assertNotIn("index=99", text)

    def test_incomplete_payload_does_not_produce_an_output(self):
        records, text = self.module.read_output_records(b"RTL_MODEL_END\nRTL_OUTPUT index=0 bytes=8\n123")
        self.assertFalse(records)
        self.assertIn("RTL_MODEL_END", text)
        with self.assertRaises(ValueError):
            self.module.read_output_records(b"RTL_OUTPUT index=0 bytes=1\nx\nWRONG_END\n")

    def test_success_requires_guest_marker_and_simulator_completion(self):
        text = "RTL_BLOCK_PASS calls=1\nTestDriver.v:158: Verilog $finish\n"
        self.assertEqual(self.module.classify(text, 0, False), "passed")
        for code, timeout in ((255, False), (0, True)):
            self.assertNotEqual(self.module.classify(text, code, timeout), "passed")
        self.assertNotEqual(self.module.classify("RTL_BLOCK_PASS calls=1", 0, False), "passed")

    def test_interrupted_output_keeps_completed_records_and_simulator_failure(self):
        raw = (b"RTL_OUTPUT index=0 bytes=1\nx\nRTL_OUTPUT_END\nRTL_OUTPUT index=1 bytes=16\n"
               b"*** FAILED *** (timeout) after 850001 simulation cycles\n")
        with self.assertRaises(ValueError):
            self.module.read_output_records(raw)
        records, text = self.module.read_output_records(raw, allow_incomplete=True)
        self.assertEqual(records, [{"index": 0, "payload": b"x"}])
        self.assertIn("(timeout)", text)
        self.assertEqual(self.module.classify(text, 255, False), "cycle_limit")

    def test_success_requires_every_output_exactly_once(self):
        self.assertTrue(self.module.complete_output_indices([{"index": 1}, {"index": 0}], 2))
        for records in ([], [{"index": 0}], [{"index": 0}, {"index": 0}], [{"index": 0}, {"index": 2}]):
            self.assertFalse(self.module.complete_output_indices(records, 2))

    def test_negative_requires_real_numerical_failure_and_exact_decoded_guest_code(self):
        text = "RTL_NUMERICAL_FAILURE\nAssertion failed: *** FAILED *** (exit code =         22)\n"
        self.assertEqual(self.module.classify(text, 255, False, True), "negative_control_passed")
        for changed in (text.replace("22", "12"), text.replace("RTL_NUMERICAL_FAILURE", "missing")):
            self.assertNotEqual(self.module.classify(changed, 255, False, True), "negative_control_passed")
        self.assertNotEqual(self.module.classify(text, 0, False, True), "negative_control_passed")

    @unittest.skipUnless(shutil.which("cc"), "Requires an explicit native C compiler")
    def test_compiled_word_stack_scan_preserves_exact_first_changed_byte(self):
        scratch = build_dir() / "test-tvm-blocks-rtl"
        scratch.mkdir(exist_ok=True)
        code = "#include <stdint.h>\n" + self.module.stack_scanner_source() + r'''
int main(void){
  uint8_t bytes[144] __attribute__((aligned(8)));
  uint64_t pattern=UINT64_C(0x55aa55aa55aa55aa);
  for(unsigned offset=0;offset<8;++offset)for(unsigned length=1;length<=128;++length){
    uint8_t*bottom=bytes+offset;uint8_t*top=bottom+length;
    for(unsigned i=0;i<length;++i)bottom[i]=(uint8_t)(pattern>>(i%8*8));
    if(rtl_stack_used(bottom,top)!=0)return 1;
    for(unsigned index=0;index<length;++index){bottom[index]^=1;
      if(rtl_stack_used(bottom,top)!=length-index)return 2;
      bottom[index]^=1;
    }
  }return 0;
}
'''
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            source, executable = Path(directory) / "scan.c", Path(directory) / "scan"
            source.write_text(code)
            subprocess.run([shutil.which("cc"), "-std=c11", "-O2", str(source), "-o", str(executable)],
                           check=True, capture_output=True, timeout=15)
            subprocess.run([str(executable)], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
