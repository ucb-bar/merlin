"""Independent whole-source cache replay and poisoned-cache refusal checks."""
from pathlib import Path
from unittest.mock import patch
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

import cache_decode_reference as reference


class Source(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.arange(16, dtype=torch.bfloat16).reshape(8, 2))

    def forward(self, ids, positions, mask, *past):
        hidden = torch.nn.functional.embedding(ids, self.weight)
        current = hidden.reshape(1, 1, ids.shape[1], 2)
        if past:
            return hidden, torch.cat((past[0], current), dim=2), torch.cat((past[1], current), dim=2)
        return hidden, current.clone(), current.clone()


def capture(folder, past=0, query=2):
    folder.mkdir()
    ids = torch.arange(past, past + query, dtype=torch.int64).reshape(1, query)
    positions = ids.clone()
    keep = np.arange(past + query)[None, :] <= np.arange(past, past + query)[:, None]
    mask = torch.from_numpy(np.where(keep, 0, 0xff7f).astype(np.uint16)).view(torch.bfloat16)[None, None]
    args = (ids, positions, mask)
    if past:
        poison = torch.full((1, 1, past, 2), 99, dtype=torch.bfloat16)
        args += (poison.clone(), poison.clone())
    program = torch.export.export(Source(), args)
    specs = {spec.arg.name: spec for spec in program.graph_signature.input_specs}
    def encode(value):
        if isinstance(value, torch.fx.Node):
            return {"node_id": value.name}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, torch.dtype):
            return {"kind": "dtype", "value": str(value)}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        return value
    nodes = []
    for node in program.graph.nodes:
        value = node.meta.get("val")
        results = [{"kind": "tensor", "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch.")}] if isinstance(value, torch.Tensor) else []
        row = dict(id=node.name, target=str(node.target), op=node.op,
                   args=encode(node.args), kwargs=encode(node.kwargs), results=results)
        if node.op == "placeholder":
            spec = specs[node.name]
            row.update(input_kind=spec.kind.name, input_target="model." + spec.target if spec.target else None)
        nodes.append(row)
    reference.write(folder / "frontend-trace.json", {"graphs": {"quantized": {"nodes": nodes}}})
    save_file({"model." + key: value for key, value in {**program.state_dict, **program.constants}.items()},
              str(folder / "weights.safetensors"))
    np.savez(folder / "inputs.npz", **{f"in{i}": value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()
                                      for i, value in enumerate(args)})


class WholeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = reference.reference.primitives.POLICIES
        self.export = self.root / "export"
        self.export.mkdir()
        save_file(Source().state_dict(), str(self.export / "model.safetensors"))
        reference.write(self.export / "policy.json", {})
        self.publish_export_manifest()

    def publish_export_manifest(self):
        reference.write(self.export / "manifest.json", {"files": {
            name: {"sha256": reference.digest(self.export / name)}
            for name in ("model.safetensors", "policy.json")}})

    def test_embedding_and_decode_use_independent_expected_cache_words(self):
        capture(self.root / "prefill")
        prefill = reference.run(self.root / "prefill", self.root / "prefill-reference", self.policy, export_dir=self.export)
        capture(self.root / "decode", past=2, query=1)
        decode = reference.run(self.root / "decode", self.root / "decode-reference", self.policy, export_dir=self.export,
                               parent=self.root / "prefill-reference")
        self.assertEqual(len(prefill["outputs"]), 3)
        self.assertEqual(prefill["original_export"]["authenticated_tensor_count"], 1)
        self.assertEqual(prefill["original_export"]["model_safetensors_sha256"],
                         reference.digest(self.export / "model.safetensors"))
        self.assertTrue(decode["cache_prefix_preserved"])
        with np.load(self.root / "decode-reference/expected.npz") as arrays:
            expected = torch.arange(6, dtype=torch.bfloat16).view(torch.uint16).numpy().reshape(1, 1, 3, 2)
            np.testing.assert_array_equal(arrays["output_1"], expected)
            np.testing.assert_array_equal(arrays["output_2"], expected)

    def test_greedy_parent_uses_expected_logits_instead_of_captured_ids(self):
        capture(self.root / "prefill")
        reference.run(self.root / "prefill", self.root / "parent", self.policy, export_dir=self.export)
        capture(self.root / "decode", past=2, query=1)
        result = reference.run(self.root / "decode", self.root / "greedy", self.policy,
                               export_dir=self.export, parent=self.root / "parent", greedy_parent=True)
        # Parent logits are [[0, 1], [2, 3]], so vocabulary index 1 wins.
        selected = np.array([[1]], dtype=np.int64)
        self.assertEqual(result["inputs"]["input_0"], reference.reference.info(selected))
        self.assertEqual(result["greedy_selection"]["token_ids"], [[1]])
        self.assertEqual(result["greedy_selection"]["source"], "independent_parent_expected_logits")
        self.assertIsNone(result["explicit_tokens_sha256"])
        with np.load(self.root / "greedy/expected.npz") as arrays:
            expected = Source().weight[1].view(torch.uint16).numpy().reshape(1, 1, 2)
            np.testing.assert_array_equal(arrays["output_0"], expected)
        capture(self.root / "decode2", past=3, query=1)
        chained = reference.run(self.root / "decode2", self.root / "greedy2", self.policy,
                                export_dir=self.export, parent=self.root / "greedy", greedy_parent=True)
        self.assertEqual(chained["greedy_selection"]["token_ids"], [[1]])
        self.assertEqual(chained["parent_reference_sha256"], reference.digest(self.root / "greedy/reference.json"))

    def test_greedy_last_row_ties_signed_values_and_nan(self):
        words = np.array([[[0x7f80, 0, 0, 0], [0xbf80, 0x4000, 0x4000, 0x3f80]]], dtype=np.uint16)
        np.testing.assert_array_equal(reference.greedy_tokens(words), [[1]])
        np.testing.assert_array_equal(reference.greedy_tokens(np.array([[[0x8000, 0]]], dtype=np.uint16)), [[0]])
        words[0, 0, 0] = 0x7fc1
        with self.assertRaisesRegex(ValueError, "NaN"):
            reference.greedy_tokens(words)
        with self.assertRaisesRegex(ValueError, "BF16"):
            reference.greedy_tokens(words.astype(np.float32))

    def test_greedy_requires_parent_and_excludes_explicit_tokens(self):
        capture(self.root / "decode", past=2, query=1)
        _, users = reference.program_from_trace(self.root / "decode")
        for parent, tokens in ((None, None), (self.root / "parent", self.root / "tokens.npy")):
            with self.assertRaisesRegex(ValueError, "requires --parent-reference and excludes --tokens"):
                reference.decode_inputs(self.root / "decode", users, parent, tokens, greedy_parent=True)

    def test_captured_weight_mutation_refuses_before_interpretation(self):
        capture(self.root / "prefill")
        weights = Source().weight.clone()
        weights.view(torch.uint16).reshape(-1)[0] = 1
        save_file({"model.weight": weights}, str(self.root / "prefill/weights.safetensors"))
        with patch.object(reference, "Interpreter") as interpreter:
            with self.assertRaisesRegex(ValueError, "differs from original export"):
                reference.run(self.root / "prefill", self.root / "refused", self.policy, export_dir=self.export)
            interpreter.assert_not_called()
        self.assertFalse((self.root / "refused").exists())

    def test_original_export_file_mutation_refuses(self):
        capture(self.root / "prefill")
        (self.export / "policy.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "original export file hash differs"):
            reference.run(self.root / "prefill", self.root / "refused", self.policy, export_dir=self.export)

    def test_capture_archive_extra_state_refuses(self):
        capture(self.root / "prefill")
        weights = {"model.weight": Source().weight, "model.extra": torch.zeros(1)}
        save_file(weights, str(self.root / "prefill/weights.safetensors"))
        with self.assertRaisesRegex(ValueError, "archive and source state roster differ"):
            reference.run(self.root / "prefill", self.root / "refused", self.policy, export_dir=self.export)

    def test_original_export_dtype_and_shape_refuse(self):
        capture(self.root / "prefill")
        for weight in (Source().weight.float(), Source().weight.reshape(2, 8)):
            with self.subTest(dtype=weight.dtype, shape=weight.shape):
                save_file({"weight": weight}, str(self.export / "model.safetensors"))
                self.publish_export_manifest()
                with self.assertRaisesRegex(ValueError, "differs from original export"):
                    reference.run(self.root / "prefill", self.root / "refused", self.policy, export_dir=self.export)

    def test_generated_parent_receipt_refuses(self):
        capture(self.root / "decode", past=2, query=1)
        parent = self.root / "generated"
        parent.mkdir()
        reference.write(parent / "reference.json", dict(status="EXPECTED", oracle_mode="generated_execution"))
        _, users = reference.program_from_trace(self.root / "decode")
        with self.assertRaisesRegex(ValueError, "independent whole-cache"):
            reference.decode_inputs(self.root / "decode", users, parent)

    def test_parent_word_mutation_refuses(self):
        capture(self.root / "prefill")
        reference.run(self.root / "prefill", self.root / "parent", self.policy, export_dir=self.export)
        with np.load(self.root / "parent/expected.npz") as saved:
            arrays = {key: saved[key].copy() for key in saved}
        arrays["output_1"].flat[0] ^= np.uint16(1)
        np.savez(self.root / "parent/expected.npz", **arrays)
        capture(self.root / "decode", past=2, query=1)
        _, users = reference.program_from_trace(self.root / "decode")
        with self.assertRaisesRegex(ValueError, "authenticated independent"):
            reference.decode_inputs(self.root / "decode", users, self.root / "parent")

    def test_missing_parent_and_noncausal_mask_refuse(self):
        capture(self.root / "decode", past=2, query=1)
        _, users = reference.program_from_trace(self.root / "decode")
        with self.assertRaisesRegex(ValueError, "requires an independent parent"):
            reference.decode_inputs(self.root / "decode", users)
        capture(self.root / "prefill")
        with np.load(self.root / "prefill/inputs.npz") as saved:
            arrays = {key: saved[key].copy() for key in saved}
        arrays["in2"].fill(0)
        np.savez(self.root / "prefill/inputs.npz", **arrays)
        _, users = reference.program_from_trace(self.root / "prefill")
        with self.assertRaisesRegex(ValueError, "causal append"):
            reference.decode_inputs(self.root / "prefill", users)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
