"""Resource admission and numerical reporting for isolated compiler workloads."""

import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

from merlin.common.paths import build_dir, repo_root


class BlockReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = repo_root() / "examples/gemmini/comparisons/tvm"
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("tvm_block_verifier", root / "verify_blocks.py")
        cls.verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.verifier)
        import block_workloads
        cls.workloads = block_workloads
        import numpy
        cls.np = numpy
        cls.scratch = build_dir() / "test-tvm-blocks"
        cls.scratch.mkdir(exist_ok=True)

    def test_cosine_distinguishes_quantization_and_compiler_error(self):
        source = self.np.array([1.0, 2.0, -3.0], dtype="float32")
        quantized = self.np.array([1.0, 2.1, -2.8], dtype="float32")
        compiler = self.verifier.metrics(quantized.copy(), quantized, self.np)
        quality = self.verifier.metrics(quantized, source, self.np)
        self.assertTrue(compiler["exact_equal"])
        self.assertAlmostEqual(compiler["cosine"], 1.0)
        self.assertLess(quality["cosine"], 1.0)
        self.assertGreater(quality["relative_l2"], 0.0)

    def test_metrics_reject_nonfinite_and_different_tensor_shapes(self):
        with self.assertRaises(ValueError):
            self.verifier.metrics(self.np.ones((2, 2)), self.np.ones((4,)), self.np)
        with self.assertRaises(ValueError):
            self.verifier.metrics(self.np.array([float("nan")]), self.np.ones(1), self.np)

    def test_zero_vector_and_opposite_direction_are_explicit(self):
        for actual, expected, cosine in (([0, 0], [0, 0], 1), ([0, 0], [1, 0], 0), ([-1, 0], [1, 0], -1)):
            result = self.verifier.metrics(self.np.array(actual), self.np.array(expected), self.np)
            self.assertEqual(result["cosine"], cosine)

    def test_preflight_reads_only_header_and_refuses_unknown_geometry(self):
        header = {"model.action_out_proj.weight": {"shape": [32, 720], "dtype": "F32", "data_offsets": [0, 92160]},
                  "model.action_out_proj.bias": {"shape": [32], "dtype": "F32", "data_offsets": [92160, 92288]}}
        with tempfile.TemporaryDirectory(dir=self.scratch) as directory:
            path = Path(directory) / "weights.safetensors"
            def write():
                raw = json.dumps(header).encode()
                path.write_bytes(struct.pack("<Q", len(raw)) + raw)
            write()
            result = self.workloads.preflight("smolvla-action", path)
            self.assertFalse(result["full_checkpoint_load"])
            self.assertEqual(result["selected_fp32_bytes_upper_bound"], 92288)
            one = self.workloads.preflight("smolvla-action", path, action_chunk_size=1)
            self.assertEqual(one["contraction_macs_per_call"], 720 * 32)
            self.assertEqual(result["contraction_macs_per_call"], 50 * one["contraction_macs_per_call"])
            reduced = self.workloads.preflight("smolvla-action", path, action_hidden_size=360)
            self.assertEqual(reduced["selected_fp32_bytes_upper_bound"], (32 * 360 + 32) * 4)
            self.assertEqual(reduced["contraction_macs_per_call"], result["contraction_macs_per_call"] // 2)
            with self.assertRaises(ValueError):
                self.workloads.preflight("smolvla-action", path, action_chunk_size=2)
            header["model.action_out_proj.weight"]["shape"] = [32000, 720]
            write()
            with self.assertRaises(ValueError):
                self.workloads.preflight("smolvla-action", path)

    def test_reduced_geometry_refuses_ignored_or_oversized_overrides(self):
        for kind, options in (("smolvla-action", {"tiny_hidden_size": 1024}),
                              ("tinyllama-head-ffn", {"action_hidden_size": 360}),
                              ("smolvla-block-action", {"action_hidden_size": 360}),
                              ("tinyllama-head-ffn", {"tiny_intermediate_size": 5633}),
                              ("tinyllama-head-ffn", {"tiny_hidden_size": 63}),
                              ("smolvla-action", {"action_hidden_size": 0}),
                              ("smolvla-action", {"action_hidden_size": 360.5}),
                              ("tinyllama-head-ffn", {"tokens": 0})):
            with self.subTest(kind=kind, options=options), self.assertRaises(ValueError):
                self.workloads.workload_geometry(kind, **options)

    def test_tiny_prefix_resource_accounting_uses_requested_shapes(self):
        prefix = "model.layers.0."
        header = {}
        for name in ("input_layernorm", "post_attention_layernorm"):
            header[prefix + name + ".weight"] = {"shape": [2048], "dtype": "BF16"}
        for name, shape in (("q", [2048, 2048]), ("k", [256, 2048]),
                            ("v", [256, 2048]), ("o", [2048, 2048])):
            header[prefix + "self_attn." + name + "_proj.weight"] = {"shape": shape, "dtype": "BF16"}
        for name, shape in (("gate", [5632, 2048]), ("up", [5632, 2048]), ("down", [2048, 5632])):
            header[prefix + "mlp." + name + "_proj.weight"] = {"shape": shape, "dtype": "BF16"}
        with tempfile.TemporaryDirectory(dir=self.scratch) as directory:
            path = Path(directory) / "weights.safetensors"
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw)
            reduced = self.workloads.preflight("tinyllama-head-ffn", path, tokens=1,
                                               tiny_hidden_size=1024, tiny_intermediate_size=2816)
            elements = 3 * 1024 * 2816 + 4 * 1024 * 64
            self.assertEqual(reduced["selected_fp32_bytes_upper_bound"], (elements + 2 * 1024) * 4)
            self.assertEqual(reduced["contraction_macs_per_call"], elements + 2 * 64)
            self.assertFalse(reduced["full_checkpoint_load"])

    def test_preflight_rejects_truncated_header(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as directory:
            path = Path(directory) / "weights.safetensors"
            path.write_bytes(struct.pack("<Q", 128) + b"{}")
            with self.assertRaises(ValueError):
                self.workloads.tensor_directory(path)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires the explicitly selected TVM build")
    def test_public_offset_specialization_executes_and_rejects_explicit_strides(self):
        import tvm
        from static_buffer_abi import specialize_zero_offsets
        tir = tvm.tir
        a, b = tir.Var("a", "handle"), tir.Var("b", "handle")
        source = tir.decl_buffer((2, 3), "float32", name="A", elem_offset=tir.Var("a_offset", "int32"))
        target = tir.decl_buffer((2, 3), "float32", name="B", elem_offset=tir.Var("b_offset", "int32"))
        i = tir.Var("i", "int32")
        body = tir.For(i, 0, 6, tir.ForKind.SERIAL,
                       tir.BufferStore(target, source[i // 3, i % 3] + 1.0, [i // 3, i % 3]))
        function = tir.PrimFunc([a, b], body, buffer_map={a: source, b: target}).with_attr("global_symbol", "offset_copy")
        specialized, changes = specialize_zero_offsets(tvm.IRModule({"offset_copy": function}), tvm)
        self.assertEqual(len(changes), 1)
        self.assertEqual(len(changes[0]["buffers"]), 2)
        native = tvm.build(specialized, target="llvm")
        inputs = self.np.arange(6, dtype="float32").reshape(2, 3)
        result = tvm.nd.empty((2, 3), "float32")
        native["offset_copy"](tvm.nd.array(inputs), result)
        self.np.testing.assert_array_equal(result.numpy(), inputs + 1)
        bad = tir.decl_buffer((2, 3), "float32", strides=(3, 1), name="strided")
        refused = tir.PrimFunc([a], tir.Evaluate(0), buffer_map={a: bad})
        with self.assertRaises(ValueError):
            specialize_zero_offsets(tvm.IRModule({"refused": refused}), tvm)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires the explicitly selected TVM build")
    def test_row_schedule_preserves_softmax_and_bounds_private_scratch(self):
        import tvm
        from cpu_row_schedule import localize_private_producers
        a = tvm.te.placeholder((1, 15, 50, 54), name="a", dtype="float32")
        output = tvm.topi.nn.softmax(a, axis=-1)
        original = tvm.IRModule({"main": tvm.te.create_prim_func([a, output])})
        localized, record = localize_private_producers(original, tvm)
        self.assertTrue(record["changes"])
        self.assertFalse(record["refusals"])
        primitive = localized["main"]
        buffers = []
        tvm.tir.stmt_functor.post_order_visit(primitive.body,
            lambda node: buffers.extend(node.alloc_buffers) if isinstance(node, tvm.tir.Block) else None)
        sizes = [int(self.np.prod([int(dim) for dim in buffer.shape])) * 4 for buffer in buffers]
        self.assertLessEqual(max(sizes), 54 * 4)
        native = tvm.build(localized, target="llvm")
        values = self.np.linspace(-3, 4, 1 * 15 * 50 * 54, dtype="float32").reshape(1, 15, 50, 54)
        expected = self.np.exp(values - values.max(axis=-1, keepdims=True))
        expected /= expected.sum(axis=-1, keepdims=True)
        result = tvm.nd.empty(values.shape, "float32")
        native(tvm.nd.array(values), result)
        self.np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-7)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires the explicitly selected TVM build")
    def test_single_hart_schedule_removes_threadpool_dependency(self):
        import tvm
        from cpu_row_schedule import serialize_parallel_loops
        tir = tvm.tir
        a, b = tir.Var("a", "handle"), tir.Var("b", "handle")
        source = tir.decl_buffer((6,), "float32", name="A")
        target = tir.decl_buffer((6,), "float32", name="B")
        i = tir.Var("i", "int32")
        body = tir.For(i, 0, 6, tir.ForKind.PARALLEL, tir.BufferStore(target, source[i] * 2, [i]))
        function = tir.PrimFunc([a, b], body, buffer_map={a: source, b: target}).with_attr("global_symbol", "parallel_copy")
        serial, record = serialize_parallel_loops(tvm.IRModule({"parallel_copy": function}), tvm)
        self.assertEqual(record["serialized_loops"], 1)
        native = tvm.build(serial, target="llvm")
        self.assertNotIn("__TVMBackendParallelLaunch", native.get_source("ll"))
        inputs = self.np.arange(6, dtype="float32")
        output = tvm.nd.empty((6,), "float32")
        native["parallel_copy"](tvm.nd.array(inputs), output)
        self.np.testing.assert_array_equal(output.numpy(), inputs * 2)


if __name__ == "__main__":
    unittest.main()
