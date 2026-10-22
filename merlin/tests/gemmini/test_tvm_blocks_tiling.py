"""Shape/resource policy and real TensorIR reuse checks for block experiments."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from merlin.common.paths import build_dir, repo_root


class BlockTilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(repo_root() / "examples/gemmini/comparisons/tvm"))
        import block_tiling
        cls.policy = block_tiling
        cls.facts = {"dim": 16, "spad_rows": 16384, "acc_rows": 1024}

    def test_action_and_one_token_shapes_choose_useful_reuse(self):
        for shape, expected in (((50, 32, 360), [4, 2]), ((1, 32, 720), [1, 2]),
                                ((1, 64, 1024), [1, 4]), ((1, 2816, 1024), [1, 4]),
                                ((50, 720, 720), [4, 4])):
            with self.subTest(shape=shape):
                self.assertEqual(self.policy.choose_tiles(shape, self.facts)["tiles"], expected)
        row = self.policy.choose_tiles((50, 32, 360), self.facts)
        self.assertEqual(row["primitive_API_counts"], {"loads_a": 92, "loads_b": 46,
                                                      "computes": 184, "stores": 8, "admissions": 1})

    def test_macro_fill_distinguishes_unavoidable_micro_tile_tail(self):
        row = self.policy.choose_tiles((1, 64, 17), self.facts)
        self.assertEqual(row["macro_grid_fill"], 1)
        self.assertEqual(row["micro_tile_output_fill"], 1 / 16)
        row = self.policy.choose_tiles((50, 720, 17), self.facts)
        self.assertGreaterEqual(row["macro_grid_fill"], 0.9)
        # Three column tiles cannot fill a four-slot group at the default limit.
        self.assertEqual(self.policy.choose_tiles((17, 33, 19), self.facts)["tiles"], [2, 1])

    def test_capacities_and_bad_shapes_fail_closed(self):
        facts = dict(self.facts, acc_rows=16, spad_rows=32)
        self.assertEqual(self.policy.choose_tiles((64, 64, 16), facts)["tiles"], [1, 1])
        for shape, facts, fill in (((0, 64, 16), self.facts, 90),
                                   ((1, 64, 16), dict(self.facts, acc_rows=1), 90),
                                   ((1, 64, 16), self.facts, 101),
                                   ((1.0, 64, 16), self.facts, 90)):
            with self.subTest(shape=shape, facts=facts, fill=fill), self.assertRaises(ValueError):
                self.policy.choose_tiles(shape, facts, fill)

    def test_header_binding_refuses_changed_or_ambiguous_parameters(self):
        import hashlib
        scratch = build_dir() / "test-tvm-block-tiling"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            header = Path(directory) / "gemmini_params.h"
            header.write_text("#define DIM 16\n#define BANK_NUM 4\n#define BANK_ROWS 4096\n#define ACC_ROWS 1024\n")
            digest = hashlib.sha256(header.read_bytes()).hexdigest()
            receipt = Path(directory) / "receipt.json"
            receipt.write_text(json.dumps({"headers": {"include/gemmini_params.h": {"sha256": digest}},
                                           "resolved_headers": {str(header): digest}}))
            self.assertEqual(self.policy.hardware_facts(receipt)["acc_rows"], 1024)
            header.write_text(header.read_text() + "#define ACC_ROWS 512\n")
            with self.assertRaises(ValueError):
                self.policy.hardware_facts(receipt)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires the explicitly selected TVM build")
    def test_semantic_relax_selection_reaches_actual_device_attributes(self):
        import tvm
        from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
        a = tvm.relax.Var("a", tvm.relax.TensorStructInfo((1, 64), "int8"))
        b = tvm.relax.Var("b", tvm.relax.TensorStructInfo((64, 80), "int8"))
        builder = tvm.relax.BlockBuilder()
        with builder.function("main", [a, b]):
            with builder.dataflow():
                result = builder.emit_output(tvm.relax.op.matmul(a, b, out_dtype="int32"))
            builder.emit_func_output(result)
        semantic = builder.get()
        choices, rows = self.policy.select_from_relax(semantic, tvm, self.facts)
        self.assertEqual(choices, {(1, 80, 64): (1, 1)})
        # At five column tiles, 1x4 would leave 3/8 macro slots empty.
        self.assertEqual(rows[0]["macro_grid_fill"], 1)
        device = prepare_gemmini_graph(semantic, tile_choices=choices, route="legacy")
        selected = [f for f in device.functions.values() if isinstance(f, tvm.tir.PrimFunc) and f.attrs and "gemmini.m" in f.attrs]
        self.assertEqual(len(selected), 1)
        self.assertEqual(int(selected[0].attrs["gemmini.tile_j"]), 1)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires the explicitly selected TVM build")
    def test_large_tail_schedule_preserves_independent_integer_math(self):
        import numpy as np
        import tvm
        from tvm.relax.backend.contrib.gemmini_schedule import make_gemmini_matmul
        shape = (50, 720, 19)
        selection = self.policy.choose_tiles(shape, self.facts)
        schedule = make_gemmini_matmul(*shape, *selection["tiles"])
        self.assertEqual((schedule.metadata["tile_i"], schedule.metadata["tile_j"]), (4, 4))
        native = tvm.build(schedule.semantic_mod, target="llvm")
        generator = np.random.default_rng(20261020)
        a = generator.integers(-127, 128, (shape[0], shape[2]), dtype=np.int8)
        b = generator.integers(-127, 128, (shape[2], shape[1]), dtype=np.int8)
        result = tvm.nd.empty((shape[0], shape[1]), "int32")
        native(tvm.nd.array(a), tvm.nd.array(b), result)
        np.testing.assert_array_equal(result.numpy(), a.astype("int64") @ b.astype("int64"))
        self.assertEqual(schedule.metadata["compute"], selection["primitive_API_counts"]["computes"])
        self.assertEqual(schedule.metadata["load_a"], selection["primitive_API_counts"]["loads_a"])
        self.assertEqual(schedule.metadata["load_b"], selection["primitive_API_counts"]["loads_b"])


if __name__ == "__main__":
    unittest.main()
