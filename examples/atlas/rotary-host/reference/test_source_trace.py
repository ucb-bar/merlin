"""Shared trace reconstruction and rotary qualification slice checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

import rotary_exact
import rotary_source
import source_trace
import whole_reference


def tensor(shape, dtype):
    return dict(kind="tensor", shape=shape, dtype=dtype)


def node(key):
    return {"node_id": key}


class SourceTraceTests(unittest.TestCase):
    def test_constant_roles_tuple_results_and_source_arguments(self):
        records = [dict(id="value", op="placeholder", target="value", args=[], kwargs={},
                        input_kind="CONSTANT_TENSOR", input_target="source.value",
                        results=[tensor([2], "float32")]),
                   dict(id="split", op="call_function", target="aten.unbind.int",
                        args=[node("value"), 0], kwargs={}, results=[tensor([], "float32")] * 2),
                   dict(id="first", op="call_function", target="<built-in function getitem>",
                        args=[node("split"), 0], kwargs={}, results=[tensor([], "float32")]),
                   dict(id="converted", op="call_function", target="aten._to_copy.default",
                        args=[node("first")], kwargs={
                            "dtype": {"kind": "dtype", "value": "torch.bfloat16"},
                            "layout": {"kind": "layout", "value": "torch.strided"},
                            "device": {"kind": "device", "value": "cpu"}},
                        results=[tensor([], "bfloat16")]),
                   dict(id="output", op="output", target="output", args=[[node("converted")]],
                        kwargs={}, results=[])]
        value = torch.tensor([1.0, 2.0])
        program, users = source_trace.rebuild_program(records, lambda record: value)
        self.assertEqual(users, [])
        self.assertEqual(program.state_dict, {})
        self.assertIs(program.constants["source.value"], value)
        graph = torch.fx.GraphModule(torch.nn.Module(), program.graph)
        self.assertEqual(graph(value)[0].item(), 1.0)
        self.assertEqual(graph(value)[0].dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "source constant metadata differs"):
            source_trace.rebuild_program(records, lambda record: value.to(torch.bfloat16))

    def test_rotary_slice_and_qualification_extent_keep_source_words(self):
        records = []

        def add(key, target, args, shape, dtype="float32", **extra):
            records.append(dict(id=key, op="call_function", target=target, args=args,
                                kwargs={}, results=[tensor(shape, dtype)], **extra))

        records.append(dict(id="frequency", op="placeholder", target="frequency", args=[], kwargs={},
                            input_kind="BUFFER", input_target="model.model.rotary_emb.inv_freq",
                            results=[tensor([32], "float32")]))
        add("positions", "aten.arange.default", [64], [64], "int64")
        add("batch", "aten.unsqueeze.default", [node("positions"), 0], [1, 64], "int64")
        add("column", "aten.unsqueeze.default", [node("batch"), 1], [1, 1, 64], "int64")
        add("floating", "aten.to.dtype", [node("column"),
            {"kind": "dtype", "value": "torch.float32"}], [1, 1, 64])
        add("frequency_batch", "aten.unsqueeze.default", [node("frequency"), 0], [1, 32])
        add("frequency_column", "aten.unsqueeze.default", [node("frequency_batch"), -1], [1, 32, 1])
        add("product", "aten.bmm.default", [node("frequency_column"), node("floating")], [1, 32, 64])
        add("angles", "aten.transpose.int", [node("product"), 1, 2], [1, 64, 32])
        add("doubled", "aten.cat.default", [[node("angles"), node("angles")], -1], [1, 64, 64])
        for name in ("cos", "sin"):
            add(name, "aten." + name + ".default", [node("doubled")], [1, 64, 64])
            add(name + "_scaled", "aten.mul.Tensor", [node(name), 1.0], [1, 64, 64])
            add(name + "_bf16", "aten.to.dtype", [node(name + "_scaled"),
                {"kind": "dtype", "value": "torch.bfloat16"}], [1, 64, 64], "bfloat16")
        # Unrelated source inputs must not enter the rotary dependency slice.
        records.insert(0, dict(id="unused", op="placeholder", target="unused", args=[], kwargs={},
                               input_kind="USER_INPUT", results=[tensor([1], "int64")]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "frontend-trace.json"
            trace.write_text(json.dumps({"graphs": {"quantized": {"nodes": records}}}))
            frequency = rotary_exact.inv_freq().view(np.float32)
            save_file({"model.model.rotary_emb.inv_freq": torch.from_numpy(frequency.copy())},
                      str(root / "weights.safetensors"))
            for extent in (None, 3):
                with self.subTest(extent=extent):
                    outputs, provenance, interpreter = rotary_source.replay(
                        trace, whole_reference.primitives.POLICIES, positions_end=extent)
                    expected = rotary_exact.prepare(np.arange(extent or 64, dtype=np.int64)[None, :])
                    for name in ("cos", "sin"):
                        np.testing.assert_array_equal(outputs[name].raw, expected[name + "_bf16"])
                    self.assertEqual(provenance["source_node_count"], 16)
                    self.assertEqual(provenance["qualification_arange_extent"], extent)
                    self.assertEqual(interpreter.env, {})
            positions = next(record for record in records if record['id'] == 'positions')
            positions.update(op="placeholder", target="position_ids", args=[],
                             input_kind="USER_INPUT", input_target=None)
            trace.write_text(json.dumps({"graphs": {"quantized": {"nodes": records}}}))
            with self.assertRaisesRegex(ValueError, "rotary source dependency is not captured inverse frequency"):
                rotary_source.program_from_trace(trace)
            positions.update(op="call_function", target="aten.arange.default", args=[64])
            trace.write_text(json.dumps({"graphs": {"quantized": {"nodes": records}}}))
            frequency[0] = 0.5
            save_file({"model.model.rotary_emb.inv_freq": torch.from_numpy(frequency.copy())},
                      str(root / "weights.safetensors"))
            with self.assertRaisesRegex(ValueError, "source lifted inverse-frequency bytes differ"):
                rotary_source.program_from_trace(trace)


if __name__ == "__main__":
    unittest.main()
