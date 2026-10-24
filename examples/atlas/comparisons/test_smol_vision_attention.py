"""CPU16 vision flash host arithmetic controls, never Atlas execution evidence."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from source_relocation import physical_path


class VisionAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        root = physical_path(Path(__file__).absolute().parent)
        cls.out = physical_path(tempfile.mkdtemp(prefix="smol-vision-attention-"))
        cls.source_files = [root / name for name in ("smol_vision_attention.c", "smol_vision_attention.h", "smol_host_bits.h")]
        cls.source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in cls.source_files}
        cls.library = cls.out / "attention.so"
        shim = cls.out / "attention-test.c"
        shim.write_text('#include "smol_vision_attention.c"\n'
                        'void smol_test_prior_exp(const float *x, float *y, float *native, size_t count) {\n'
                        '  for (size_t i=0; i<count; ++i) { y[i]=vision_prior_exp(x[i]); native[i]=expf(x[i]); }\n'
                        '}\n')
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-O2", "-fPIC", "-shared",
                        "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                        "-I", str(root), str(shim), "-lm",
                        "-o", str(cls.library)], check=True, timeout=90, capture_output=True)
        cls.loaded = ctypes.CDLL(str(cls.library))
        cls.function = cls.loaded.smol_vision_attention_cpu16_bf16
        cls.pointer = ctypes.POINTER(ctypes.c_uint16)
        cls.function.argtypes = [cls.pointer, cls.pointer, cls.pointer,
                                 ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, cls.pointer]
        cls.function.restype = ctypes.c_int

    def run_c(self, q, k, v):
        output = np.full_like(q, 0xdead)
        pointer = lambda array: array.ctypes.data_as(self.pointer)
        result = self.function(pointer(q), pointer(k), pointer(v), q.shape[0], k.shape[0], q.shape[1], pointer(output))
        return result, output

    def check_torch(self, rows, keys, seed):
        generator = torch.Generator().manual_seed(seed)
        q = torch.randn((1, 1, rows, 64), generator=generator).to(torch.bfloat16)
        k = torch.randn((1, 1, keys, 64), generator=generator).to(torch.bfloat16)
        v = torch.randn((1, 1, keys, 64), generator=generator).to(torch.bfloat16)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=0.125)
        arrays = [tensor[0].transpose(0, 1).contiguous().view(torch.uint16).numpy().copy() for tensor in (q, k, v)]
        result, actual = self.run_c(*arrays)
        self.assertEqual(result, 0)
        np.testing.assert_array_equal(actual, expected[0].transpose(0, 1).contiguous().view(torch.uint16).numpy())

    def test_negative_fp32_prior_exp_matches_native_libm_bits(self):
        generator = np.random.default_rng(731)
        random_bits = generator.integers(0, 0x42d00000, 131072, dtype=np.uint32) | np.uint32(0x80000000)
        incoming = np.concatenate((np.linspace(-104, 0, 262144, dtype=np.float32),
                                   random_bits.view(np.float32), np.array([-np.inf, -0.0, 0.0], np.float32)))
        actual, expected = np.empty_like(incoming), np.empty_like(incoming)
        pointer = ctypes.POINTER(ctypes.c_float)
        function = self.loaded.smol_test_prior_exp
        function.argtypes = [pointer, pointer, pointer, ctypes.c_size_t]
        function.restype = None
        function(*[array.ctypes.data_as(pointer) for array in (incoming, actual, expected)], incoming.size)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    @unittest.skipUnless(os.environ.get("SMOL_RV64_CC") and os.environ.get("SMOL_RV64_SUPPORT"),
                         "external freestanding toolchain/support fixture required")
    def test_actual_freestanding_headers_accept_attention(self):
        compiler = physical_path(os.environ["SMOL_RV64_CC"])
        support = physical_path(os.environ["SMOL_RV64_SUPPORT"])
        resource = subprocess.run([str(compiler), "-print-resource-dir"], check=True,
                                  capture_output=True, text=True, timeout=30).stdout.strip()
        includes = physical_path(Path(resource) / "include")
        root = physical_path(Path(__file__).absolute().parent)
        result = subprocess.run([str(compiler), "--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d",
                        "-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math", "-ffreestanding",
                        "-fno-builtin", "-nostdinc", "-isystem", str(includes), "-I", str(support / "include"),
                        "-Wall", "-Wextra", "-Werror", "-c", str(root / "smol_vision_attention.c"),
                        "-o", str(self.out / "attention-rv64.o")], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_one_key_block_matches_cpu_sdpa(self):
        self.check_torch(2, 16, 8)

    def test_two_key_blocks_match_cpu_sdpa(self):
        self.check_torch(1, 1024, 9)

    def test_invalid_geometry_nonfinite_and_output_alias_refused(self):
        q, k, v = np.zeros((1, 1, 64), np.uint16), np.zeros((16, 1, 64), np.uint16), np.zeros((16, 1, 64), np.uint16)
        q[0, 0, 0] = 0x7fc0
        result, output = self.run_c(q, k, v)
        self.assertEqual(result, -1)
        self.assertTrue(np.all(output == 0xdead))
        q.fill(0)
        pointer = lambda array: array.ctypes.data_as(self.pointer)
        self.assertEqual(self.function(pointer(q), pointer(k), pointer(v), 1, 16, 1, pointer(q)), -1)
        self.assertEqual(self.function(pointer(q), pointer(k), pointer(v), 0, 16, 1, pointer(q)), -1)

    def check_authenticated_boundary(self, prefix, status, tag):
        path = physical_path(os.environ[prefix + "_RECEIPT"])
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), os.environ[prefix + "_SHA256"])
        receipt = json.loads(raw)
        self.assertEqual(receipt["status"], status)
        if tag == "native":
            self.assertTrue(receipt["all_archived_fields_raw_exact"])
        else:
            self.assertTrue(receipt["input_state_rng_bookends_equal"])
            self.assertTrue(receipt["projection_reference"]["source_bookends_equal"])
            self.assertFalse(receipt["projection_reference"]["fresh_model_executed"])
        assets = [receipt["calls"][index]["output"] for index in (1, 2, 3)]
        assets.append(receipt["calls"][4]["input"])
        arrays = []
        for item in assets:
            self.assertEqual(item["shape"], [1, 1024, 768])
            self.assertEqual(item["dtype"], "torch.bfloat16")
            payload = physical_path(path.parent / item["file"]).read_bytes()
            self.assertEqual(len(payload), item["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
            arrays.append(np.frombuffer(payload, dtype="<u2").copy().reshape(1024, 12, 64))
        rows = int(os.environ.get("SMOL_VISION_TEST_ROWS", "16"))
        self.assertTrue(0 < rows <= 1024)
        result, actual = self.run_c(arrays[0][:rows].copy(), arrays[1], arrays[2])
        expected = arrays[3][:rows]
        mismatches = int(np.count_nonzero(actual != expected))
        record = dict(status="PASS" if result == 0 and mismatches == 0 else "FAIL",
                      scope="C host attention boundary preparation only", atlas_executed=False,
                      source_sha256=self.source_hashes,
                      source_receipt_sha256=hashlib.sha256(raw).hexdigest(), rows=rows,
                      compared_words=int(expected.size), mismatches=mismatches, returncode=result,
                      actual_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                      expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest())
        (self.out / ("boundary-" + tag + "-result.json")).write_text(json.dumps(record, indent=2) + "\n")
        self.assertEqual({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in self.source_files}, self.source_hashes)
        self.assertEqual(result, 0)
        self.assertEqual(mismatches, 0, str(self.out / ("boundary-" + tag + "-result.json")))

    @unittest.skipUnless(os.environ.get("SMOL_NATIVE_CAPTURE_RECEIPT"), "external authenticated native vision fixture required")
    def test_authenticated_first_native_vision_boundary(self):
        self.check_authenticated_boundary("SMOL_NATIVE_CAPTURE", "PASS_FRESH_VISION_NATIVE_POLICY_CAPTURE", "native")

    @unittest.skipUnless(os.environ.get("SMOL_ATLAS_CAPTURE_RECEIPT"), "external authenticated Atlas-arithmetic vision fixture required")
    def test_authenticated_first_atlas_arithmetic_vision_boundary(self):
        self.check_authenticated_boundary("SMOL_ATLAS_CAPTURE", "PASS_ATLAS_ARITHMETIC_FULL_POLICY_REPLAY", "atlas")


if __name__ == "__main__":
    unittest.main()
