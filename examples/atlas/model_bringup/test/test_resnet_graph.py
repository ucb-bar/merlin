"""Small independent geometry/math controls; no model assets required."""

import copy
import _ctypes
import ctypes as c
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import generate_resnet_graph as generate
from atlas_host_build import GRAPH_PROGRAM_WORDS
import kernel_library  # noqa: E402,F401
from atlas_tile_arithmetic import f32_to_bf16_bits, pack_bf16, tensor_scale_exponent


def fixture():
    weight = {"padded_shape": [64, 32], "raw_fp32": {"shape": [49, 3]}, "b_scale_exponent": -5,
        "fp8_tiled": {"path": "weight", "sha256": "fixture"}}
    nodes = [{"node": "input", "operator": "placeholder", "output_shape": [1, 1, 7, 7], "partition": "host", "inputs": []},
        {"node": "conv", "target": "conv", "operator": "Conv2d", "output_shape": [1, 3, 1, 1], "partition": "device",
        "inputs": ["input"], "geometry": {"kernel_size": [7, 7], "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1]},
        "weight_shape": [3, 1, 7, 7], "matrix_mkn": [1, 49, 3], "padded_mkn": [32, 64, 32], "macs": 147, "weight": weight},
        {"node": "output", "operator": "output", "output_shape": [1, 3, 1, 1], "partition": "host", "inputs": ["conv"]}]
    template = {"site": "conv", "matrix_mkn": [1, 49, 3], "padded_mkn": [32, 64, 32], "weight": weight,
        "b_scale_exponent": -5, "dram_base": 1 << 32, "input_bytes": 4096, "output_offset": 4352, "output_bytes": 2048,
        "total_dram_bytes": 6656, "program_words": 4, "rescale_patch_word_index": 1}
    return {"nodes": nodes}, [template]


class GraphAdmissionTest(unittest.TestCase):
    def validate(self, plan, templates):
        return generate.validate_graph(plan, templates, lambda item: bytes(2048), require_full=False)

    def test_tiny_kernel7_admitted(self):
        p, t = fixture()
        k, _ = self.validate(p, t)
        self.assertEqual(k[0]["groups_proven"], 1)

    def test_groups_and_weight_channel_proof(self):
        for explicit in (False, True):
            p, t = fixture()
            if explicit:
                p["nodes"][1]["geometry"]["groups"] = 2
            else:
                p["nodes"][1]["weight_shape"][1] = 2
            with self.subTest(explicit=explicit), self.assertRaises(ValueError):
                self.validate(p, t)

    def test_wrong_logical_padded_macs_and_output(self):
        for key, value in [("matrix_mkn", [1, 48, 3]), ("padded_mkn", [32, 32, 32]), ("macs", 148),
            ("output_shape", [1, 3, 2, 1])]:
            p, t = fixture()
            p["nodes"][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(p, t)

    def test_arity_topology_and_duplicate_names(self):
        for mode in range(3):
            p, t = fixture()
            if mode == 0:
                p["nodes"][1]["inputs"] = ["input", "input"]
            if mode == 1:
                p["nodes"][1]["inputs"] = ["output"]
            if mode == 2:
                p["nodes"][1]["node"] = "input"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.validate(p, t)

    def test_template_physical_and_patch_refusals(self):
        for key, value in [("output_offset", 4351), ("total_dram_bytes", 6655), ("dram_base", 0),
            ("rescale_patch_word_index", 3), ("program_words", GRAPH_PROGRAM_WORDS + 1)]:
            p, t = fixture()
            t[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(p, t)

    def test_unary_residual_flatten_and_maxpool_refusals(self):
        for op in ("ReLU", "residual_add", "flatten", "MaxPool2d"):
            p, t = fixture()
            node = {"node": "bad", "operator": op, "partition": "host", "inputs": ["conv"], "output_shape": [1, 3, 2, 1]}
            if op == "residual_add":
                node["inputs"] = ["conv", "input"]
            if op == "flatten":
                node["output_shape"] = [1, 4]
            if op == "MaxPool2d":
                node["host_params"] = {"kernel": [2, 2], "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1],
                    "ceil_mode": False}
            p["nodes"].insert(2, node)
            p["nodes"][-1]["inputs"] = ["bad"]
            p["nodes"][-1]["output_shape"] = node["output_shape"]
            with self.subTest(op=op), self.assertRaises(ValueError):
                self.validate(p, t)

    def test_batchnorm_parameter_extent_and_nonfinite(self):
        p, t = fixture()
        bn = {"node": "bn", "operator": "BatchNorm2d", "partition": "host", "inputs": ["conv"], "output_shape": [1, 3, 1, 1],
            "host_params": {"channels": 3, "arithmetic": "FP32 multiply then FP32 add, no FMA",
            "scale_then_bias": {"path": "bn"}}}
        p["nodes"].insert(2, bn)
        p["nodes"][-1]["inputs"] = ["bn"]
        for params in (bytes(20), struct.pack("<6f", 1, 1, 1, 0, 0, float("nan"))):
            with self.subTest(length=len(params)), self.assertRaises(ValueError):
                generate.validate_graph(p, t, lambda item: bytes(2048) if item["path"] == "weight" else params,
                    require_full=False)


class GraphMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        out = Path(cls.directory.name) / "math.so"
        source = Path(__file__).absolute().parent.parent / "runtime/atlas_resnet_graph_math.c"
        subprocess.run(["/usr/bin/cc", "-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math", "-Wall", "-Wextra",
            "-Werror", "-pedantic", "-shared", "-fPIC", str(source), "-o",
            str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cls.lib = c.CDLL(str(out))
        cls.lib.atlas_graph_pack.argtypes = [c.POINTER(c.c_float), c.c_uint, c.c_uint, c.c_uint, c.c_uint, c.c_uint,
            c.c_uint, c.POINTER(c.c_uint8), c.c_size_t, c.POINTER(c.c_int)]
        cls.lib.atlas_graph_decode.argtypes = [c.POINTER(c.c_uint8), c.c_size_t, c.POINTER(c.c_float), c.c_uint, c.c_uint,
            c.c_uint]

    @classmethod
    def tearDownClass(cls):
        # NFS cannot remove the temporary directory while math.so is loaded.
        # No function pointer is used after the class finishes its checks.
        _ctypes.dlclose(cls.lib._handle)
        del cls.lib
        cls.directory.cleanup()


    def test_graph_affine_and_residual_in_place(self):
        ptr = c.POINTER(c.c_float)
        self.lib.atlas_graph_affine.argtypes = [ptr, ptr, ptr, ptr, c.c_uint, c.c_size_t]
        self.lib.atlas_graph_add.argtypes = [ptr, ptr, ptr, c.c_size_t]
        self.lib.atlas_graph_relu.argtypes = [ptr, ptr, c.c_size_t]
        data = (c.c_float * 4)(-2.0, 1.0, 2.0, 3.0)
        scale, bias = (c.c_float * 2)(2.0, -1.0), (c.c_float * 2)(1.0, 2.0)
        self.assertEqual(self.lib.atlas_graph_affine(data, data, scale, bias, 2, 2), 0)
        self.assertEqual(self.lib.atlas_graph_relu(data, data, 4), 0)
        self.assertEqual(list(data), [0.0, 3.0, 0.0, 0.0])
        identity = (c.c_float * 4)(-1.0, 0.5, 2.0, -2.0)
        self.assertEqual(self.lib.atlas_graph_add(data, identity, data, 4), 0)
        self.assertEqual(self.lib.atlas_graph_relu(data, data, 4), 0)
        self.assertEqual(list(data), [0.0, 3.5, 2.0, 0.0])
        self.assertEqual(self.lib.atlas_graph_affine(data, data, scale, bias, 2, 0), 1)

    def packed(self, values, h, w, kernel, stride, padding):
        data = (c.c_float * len(values))(*values)
        oh = (h + 2 * padding - kernel) // stride + 1
        ow = (w + 2 * padding - kernel) // stride + 1
        m, k = oh * ow, kernel * kernel
        mp, kp = (m + 31) // 32 * 32, (k + 31) // 32 * 32
        out = (c.c_uint8 * (mp * kp + 2))(*([0xCC] * (mp * kp + 2)))
        shift = c.c_int()
        pointer = c.cast(c.byref(out, 1), c.POINTER(c.c_uint8))
        self.assertEqual(self.lib.atlas_graph_pack(data, 1, h, w, kernel, stride, padding, pointer, mp * kp, c.byref(shift)),
            0)
        rows = []
        for row in range(m):
            vals = []
            for kr in range(kernel):
                for kc in range(kernel):
                    ir = row // ow * stride + kr - padding
                    ic = row % ow * stride + kc - padding
                    vals.append(f32_to_bf16_bits(data[ir * w + ic]) if 0 <= ir < h and 0 <= ic < w else 0)
            rows.append(vals)
        exponent = tensor_scale_exponent([v for row in rows for v in row])
        expected = bytearray()
        for mt in range(0, mp, 32):
            for kt in range(0, kp, 32):
                for r in range(32):
                    for col in range(32):
                        expected.append(pack_bf16(rows[mt + r][kt + col], exponent + 127) if mt + r < m and kt + col < k
                            else 0)
        self.assertEqual(out[0], 0xCC)
        self.assertEqual(out[-1], 0xCC)
        self.assertEqual(bytes(out)[1:-1], bytes(expected))
        self.assertEqual(shift.value, exponent)
        return shift.value

    def test_kernel7_stride2_padding_and_positive_zero_tails(self):
        self.packed([(i - 24) / 16 for i in range(49)], 7, 7, 7, 2, 3)

    def test_stride2_1x1_excludes_unused_huge_maximum(self):
        values = [1048576.0] * 9
        for i in (0, 2, 6, 8):
            values[i] = 1.0
        self.assertEqual(self.packed(values, 3, 3, 1, 2, 0), -8)

    def test_decode_crops_both_padded_dimensions_and_preserves_canaries(self):
        channels, m = 33, 33
        physical = bytearray(struct.pack("<H", 0x7FC1) * (64 * 64))
        expected = []
        for col in range(channels):
            for row in range(m):
                value = row / 4 + col
                expected.append(value)
                bits = f32_to_bf16_bits(value)
                offset = (((row // 32) * 2 + col // 32) * 2048 + ((col % 32) // 16) * 1024 + (row % 32) * 32 + (col % 16)
                    * 2)
                struct.pack_into("<H", physical, offset, bits)
        raw = (c.c_uint8 * len(physical)).from_buffer_copy(physical)
        out = (c.c_float * (len(expected) + 2))(*([12345.0] * (len(expected) + 2)))
        ptr = c.cast(c.byref(out, 4), c.POINTER(c.c_float))
        self.assertEqual(self.lib.atlas_graph_decode(raw, len(physical), ptr, channels, 3, 11), 0)
        self.assertEqual(list(out)[1:-1], expected)
        self.assertEqual(out[0], 12345)
        self.assertEqual(out[-1], 12345)

    def test_pack_nonfinite_bad_geometry_and_capacity_refused(self):
        src = (c.c_float * 49)(*([1.0] * 49))
        out = (c.c_uint8 * 2048)()
        shift = c.c_int()
        for kernel, stride, pad, cap in ((7, 1, 0, 2047), (2, 1, 0, 2048), (7, 3, 0, 2048), (7, 1, 4, 2048)):
            self.assertNotEqual(self.lib.atlas_graph_pack(src, 1, 7, 7, kernel, stride, pad, out, cap, c.byref(shift)), 0)
        src[3] = float("nan")
        self.assertNotEqual(self.lib.atlas_graph_pack(src, 1, 7, 7, 7, 1, 0, out, 2048, c.byref(shift)), 0)

    def test_affine_separate_fp32_rounding(self):
        f = self.lib.atlas_graph_affine
        f.argtypes = [c.POINTER(c.c_float)] * 4 + [c.c_uint, c.c_size_t]
        a = (c.c_float * 1)(1.0000001192092896)
        b = (c.c_float * 1)(1.0000001192092896)
        bias = (c.c_float * 1)(-1.000000238418579)
        out = (c.c_float * 1)()
        self.assertEqual(f(a, out, b, bias, 1, 1), 0)
        self.assertEqual(struct.pack("<f", out[0]), bytes(4))


if __name__ == "__main__":
    unittest.main()
