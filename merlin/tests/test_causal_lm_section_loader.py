"""Small loader contract tests; no checkpoint download or model execution."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

_LOADER_PATH = Path(__file__).resolve().parents[2] / "examples/workloads/causal_lm_section/loader.py"
_SPEC = importlib.util.spec_from_file_location("causal_lm_section_test_loader", _LOADER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
loader = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(loader)


class _FakeLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=16, _commit_hash="test", tie_word_embeddings=True)
        self.decoder = nn.Module()
        self.decoder.config = self.config
        self.decoder.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.head = nn.Linear(4, 16)
        self.embedding = nn.Embedding(16, 4)

    def get_decoder(self) -> nn.Module:
        return self.decoder

    def get_output_embeddings(self) -> nn.Module:
        return self.head

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def set_input_embeddings(self, embedding: nn.Module) -> None:
        self.embedding = embedding


class CausalLMSectionBatchTest(unittest.TestCase):
    def _load(self, *, batch: str | None = None, entry: str = "ids"):
        env = {"CAUSAL_LM_MODEL": "fake-checkpoint", "CAUSAL_LM_ENTRY": entry, "M2M_SEQ": "8"}
        if batch is not None:
            env["M2M_BATCH"] = batch
        fake_transformers = SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeLM())
        )
        with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {"transformers": fake_transformers}):
            return loader.get_model_and_inputs()

    def test_default_keeps_single_sequence_abi_and_provenance(self) -> None:
        model, (ids,) = self._load()
        self.assertEqual(tuple(ids.shape), (1, 8))
        self.assertEqual(ids.dtype, torch.long)
        self.assertNotIn("batch_size", model.session_provenance)

    def test_eight_independent_token_id_sequences(self) -> None:
        model, (ids,) = self._load(batch="8")
        self.assertEqual(tuple(ids.shape), (8, 8))
        self.assertEqual(model.session_provenance["batch_size"], 8)
        self.assertEqual(model.session_provenance["entry"], "ids")

    def test_embedded_entry_preserves_batch_and_sequence_axes(self) -> None:
        model, (embeddings,) = self._load(batch="8", entry="embeds")
        self.assertEqual(tuple(embeddings.shape), (8, 8, 4))
        self.assertEqual(model.session_provenance["batch_size"], 8)
        self.assertEqual(model.lm.get_input_embeddings().num_embeddings, 1)

    def test_rejects_zero_batch_before_loading_checkpoint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "M2M_BATCH must be positive"):
            self._load(batch="0")


if __name__ == "__main__":
    unittest.main()
