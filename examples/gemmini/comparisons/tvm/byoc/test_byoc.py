#!/usr/bin/env python3
"""Bounded numerical/partition/schedule regression suite, including rejections.

Host execution uses an explicitly labelled scalar intrinsic implementation to
test compiler transforms. It supplies no Gemmini hardware evidence; verify_byoc
provides independent Spike and RTL execution of the real primitive.
"""

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import tvm
from tvm import relax, tir
from tvm.relax.backend.contrib import gemmini_byoc as backend
from verify_byoc import fixture, budget, identity

HOST_PRIMITIVE = r'''
#include <stdint.h>
void tvm_gemmini_tile_update_i8_i32(const int8_t*a,const int8_t*b,int32_t*c,int64_t sa,int64_t sb,int64_t sc){
  for(int i=0;i<16;++i)for(int j=0;j<8;++j){int64_t v=c[i*sc+j];
    for(int k=0;k<16;++k)v+=(int64_t)a[i*sa+k]*b[k*sb+j];c[i*sc+j]=(int32_t)v;}}
'''


def graph(dtype="int8", shape=(3, 17, 5), epilogue=True, leak=False):
    m, k, n = shape
    a = relax.Var("a", relax.TensorStructInfo((m, k), dtype))
    b = relax.Var("b", relax.TensorStructInfo((k, n), dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [a, b]):
        with bb.dataflow():
            mm = bb.emit(relax.op.matmul(a, b, out_dtype="int32" if dtype == "int8" else dtype))
            value = mm
            if epilogue:
                value = bb.emit(relax.op.add(value, relax.const(np.arange(n, dtype="int32" if dtype == "int8" else dtype))))
                value = bb.emit(relax.op.nn.relu(value))
            value = bb.emit_output(relax.Tuple([mm, value]) if leak else value)
        bb.emit_func_output(value)
    return bb.get()


class BackendTests(unittest.TestCase):
    def test_trace_replays_and_exact_arithmetic(self):
        rng = np.random.default_rng(29)
        for m, n, k in ((1, 1, 1), (16, 8, 16), (17, 19, 33), (32, 24, 64), (1, 1, 131071)):
            for order in ("ijk", "jik", "sample"):
                with self.subTest(shape=(m, n, k), order=order):
                    result = backend.schedule_matmul(m, n, k, order=order)
                    replay = tir.Schedule(result.semantic_mod, debug_mask="all")
                    result.trace.apply_to_schedule(replay, remove_postproc=False)
                    tvm.ir.assert_structural_equal(result.scheduled_mod, replay.mod)
                    a = rng.integers(-128, 128, (m, k), dtype="int8")
                    b = rng.integers(-128, 128, (k, n), dtype="int8")
                    if k == 131071:
                        a.fill(-128)
                        b.fill(-128)
                    expected = a.astype("int64") @ b.astype("int64")
                    fn = tvm.build(result.scheduled_mod, target="llvm")
                    out = tvm.nd.empty((m, n), "int32")
                    fn(tvm.nd.array(a), tvm.nd.array(b), out)
                    np.testing.assert_array_equal(out.numpy(), expected)

    def test_longest_composite_and_leak(self):
        partition = backend.partition_for_gemmini(graph())
        self.assertIn('"Composite": "gemmini.matmul_bias_relu"', partition.script())
        leaked = backend.partition_for_gemmini(graph(leak=True)).script()
        self.assertNotIn('"Composite": "gemmini.matmul_bias_relu"', leaked)
        self.assertIn('"Composite": "gemmini.matmul"', leaked)

    def test_fallback_and_partition_boundary(self):
        for dtype in ("float32", "int32", "uint8"):
            mod = graph(dtype=dtype, epilogue=False)
            partition = backend.partition_for_gemmini(mod)
            self.assertNotIn('"Codegen"', partition.script())
        # Lowering accepts only partitions. It does not silently offload an
        # eligible unpartitioned matmul or rewrite unrelated compiler regions.
        mod = graph(epilogue=False)
        lowered = backend.LowerGemminiBYOC()(mod)
        self.assertNotIn("tvm_gemmini_", lowered.script())
        k = tir.Var("k", "int64")
        dynamic = graph(shape=(3, k, 5), epilogue=False)
        self.assertNotIn('"Codegen"', backend.partition_for_gemmini(dynamic).script())
        self.assertNotIn('"Codegen"', backend.partition_for_gemmini(graph(shape=(1, 131072, 1), epilogue=False)).script())

    def test_other_backend_is_preserved(self):
        mod = graph(epilogue=False)
        other = mod["main"].with_attr("Codegen", "another_backend")
        mod = tvm.IRModule({"main": other})
        tvm.ir.assert_structural_equal(backend.LowerGemminiBYOC()(mod), mod)

    def test_cpu_fallback_math(self):
        x = relax.Var("x", relax.TensorStructInfo((2, 3), "float32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x]):
            with bb.dataflow():
                a = bb.emit(relax.op.subtract(x, relax.const(2.0, "float32")))
                a = bb.emit(relax.op.negative(a))
                a = bb.emit(relax.op.multiply(a, relax.const(3.0, "float32")))
                a = bb.emit(relax.op.divide(a, relax.const(2.0, "float32")))
                mean = bb.emit(relax.op.mean(a, axis=[1]))
                total = bb.emit(relax.op.sum(a, axis=[0]))
                y = bb.emit_output(relax.Tuple([mean, total]))
            bb.emit_func_output(y)
        lower = backend.prepare_gemmini_byoc(bb.get())
        self.assertNotIn("tvm_gemmini_", lower.script())
        vm = relax.VirtualMachine(relax.build(lower, target="llvm"), tvm.cpu())
        value = np.arange(6, dtype="float32").reshape(2, 3)
        result = vm["main"](tvm.nd.array(value))
        expected = -(value - 2) * 3 / 2
        np.testing.assert_array_equal(result[0].numpy(), expected.mean(axis=1))
        np.testing.assert_array_equal(result[1].numpy(), expected.sum(axis=0))

    def test_shape_rejections(self):
        for shape in ((0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 131072), (True, 1, 1), (1, 1 << 30, 1)):
            with self.assertRaises(ValueError):
                backend.schedule_matmul(*shape)
        with self.assertRaises(ValueError):
            backend.schedule_matmul(1, 1, 1, order="bad")

    def test_graph_numerics(self):
        for case in ("tiny", "tails", "bmm", "conv"):
            for order in ("ijk", "jik"):
                with self.subTest(case=case, order=order):
                    mod, a, _, _, expected = fixture(case)
                    records = []
                    lower = backend.prepare_gemmini_byoc(mod, order=order, records=records)
                    self.assertTrue(relax.analysis.well_formed(lower))
                    self.assertTrue(records)
                    for _, record in records:
                        replay = tir.Schedule(record.semantic_mod, debug_mask="all")
                        record.trace.apply_to_schedule(replay, remove_postproc=False)
                        tvm.ir.assert_structural_equal(replay.mod, record.scheduled_mod)
                    if case in ("tiny", "tails"):
                        calls = [binding.value for block in lower["main"].body.blocks for binding in block.bindings if isinstance(binding.value, relax.Call)]
                        self.assertEqual(len(calls), 1)
                    vm = relax.VirtualMachine(relax.build(lower, target="llvm"), tvm.cpu())
                    result = vm["main"](tvm.nd.array(a)).numpy()
                    np.testing.assert_array_equal(result, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    budget()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "host-primitive.c"
    source.write_text(HOST_PRIMITIVE)
    library = out / "host-primitive.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-O2", source, "-o", library], check=True, timeout=10)
    handle = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    report = {"status": "passed" if result.wasSuccessful() else "failed", "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "resources": budget(),
              "backend": identity(backend.__file__), "test_source": identity(__file__), "hardware_execution": False}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
