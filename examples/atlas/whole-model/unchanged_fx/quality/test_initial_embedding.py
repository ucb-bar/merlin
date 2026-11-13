"""Authenticate input overrides without trusting an activation receipt alone."""
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import initial_embedding as reference


class InitialEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.root = Path(self.directory.name)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.nn.Embedding(4, 3, dtype=torch.bfloat16)
                with torch.no_grad():
                    self.table.weight.copy_(torch.arange(12).reshape(4, 3))
            def forward(self, tokens):
                return self.table(tokens)
        program = torch.export.export(Model(), (torch.zeros((1, 2), dtype=torch.int64),))
        torch.export.save(program, self.root / "model.pt2")
        model_hash = sha256((self.root / "model.pt2").read_bytes()).hexdigest()
        (self.root / "fx_manifest.json").write_text(json.dumps({"files": {"model.pt2": {"sha256": model_hash}}}))
        self.manifest = {"source_model_sha256": model_hash}
        self.token_input_name = next(spec.arg.name for spec in program.graph_signature.input_specs if spec.kind.name == "USER_INPUT")
        tokens = np.array([[0, 2]], np.int64)
        words = program.state_dict["table.weight"].view(torch.uint16).numpy()
        self.activation = words[tokens].copy()
        np.save(self.root / "tokens.npy", tokens, allow_pickle=False)
        np.savez(self.root / "initial-inputs.npz", input_0=self.activation)
        token_hash = sha256(tokens.tobytes()).hexdigest()
        self.preparation = {"schema": "atlas.unchanged_embedding_input.v1", "status": "PREPARED",
                            "source_model_sha256": model_hash, "source_boundary": "embedding",
                            "operation": "raw BF16 word gather by token index", "weight_cast_or_requantization": False,
                            "source_parameter": "table.weight", "source_parameter_dtype": "bf16",
                            "source_parameter_shape": list(words.shape), "source_parameter_sha256": sha256(words.tobytes()).hexdigest(),
                            "tokens_sha256": token_hash, "window": {"input_sha256": token_hash, "input_length": 2},
                            "input_0": reference.array_info(self.activation)}
        self.capsule = {"source_boundaries": ["embedding"], "input_keys": ["input_0"],
                        "tensors": {"input_0": {"dtype": "torch.bfloat16", "shape": [1, 2, 3]}}}

    def tearDown(self):
        self.directory.cleanup()

    def authenticate(self):
        (self.root / "preparation.json").write_text(json.dumps(self.preparation))
        return reference.authenticate(self.root / "initial-inputs.npz", self.root / "preparation.json",
                                      self.root / "tokens.npy", self.root, self.manifest, self.capsule)

    def test_verified_gather_preserves_raw_words(self):
        value, record = self.authenticate()
        np.testing.assert_array_equal(value, self.activation)
        self.assertTrue(record["independent_raw_gather_verified"])
        self.assertFalse(record["source_graph_numerical_execution"])

    def test_driver_authentication_uses_source_gather_without_arithmetic_engines(self):
        import sys
        from types import SimpleNamespace
        from whole_model import authenticate_initial_inputs

        (self.root / "preparation.json").write_text(json.dumps(self.preparation))
        args = SimpleNamespace(initial_inputs=self.root / "initial-inputs.npz",
            initial_preparation=self.root / "preparation.json", initial_tokens=self.root / "tokens.npy",
            source_bundle=self.root, selection_manifest=None)
        before = set(sys.modules)
        value, receipt = authenticate_initial_inputs(args, self.manifest, self.capsule)
        np.testing.assert_array_equal(value, self.activation)
        self.assertTrue(receipt["independent_raw_gather_verified"])
        self.assertFalse(receipt["source_graph_numerical_execution"])
        added = set(sys.modules) - before
        self.assertFalse(any(name in {"whole_reference", "layer_reference", "fx_layer_reference"}
                             or name == "npu_model" or name.startswith("npu_model.") for name in added))

    def test_forged_receipt_cannot_bless_arbitrary_activation(self):
        changed = self.activation.copy()
        changed[0, 0, 0] ^= np.uint16(1)
        np.savez(self.root / "initial-inputs.npz", input_0=changed)
        self.preparation["input_0"] = reference.array_info(changed)
        with self.assertRaisesRegex(ValueError, "unchanged source embedding gather"):
            self.authenticate()

    def test_other_parameter_and_extra_override_are_rejected(self):
        self.preparation["source_parameter"] = "other.weight"
        with self.assertRaisesRegex(ValueError, "different source parameter"):
            self.authenticate()
        self.preparation["source_parameter"] = "table.weight"
        np.savez(self.root / "initial-inputs.npz", input_0=self.activation, input_1=self.activation)
        with self.assertRaisesRegex(ValueError, "only layer0 input_0"):
            self.authenticate()

    def test_token_and_table_identities_are_checked(self):
        self.preparation["tokens_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "tokens differ"):
            self.authenticate()
        self.preparation["tokens_sha256"] = self.preparation["window"]["input_sha256"]
        self.preparation["source_parameter_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "table identity"):
            self.authenticate()

    def test_other_preparation_cannot_depend_on_changed_token_values(self):
        self.capsule["source_boundaries"].append(self.token_input_name)
        with self.assertRaisesRegex(ValueError, "depends on the overridden source token values"):
            self.authenticate()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
