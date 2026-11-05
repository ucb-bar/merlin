"""One-token execution of the iteration decoder with explicit K/V state."""

import importlib.util
import sys
from pathlib import Path

import torch


class DecoderStep(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, tokens, key_cache, value_cache, positions, mask):
        hidden = self.decoder.embedding(tokens)
        query, key, value = self.decoder.qkv(self.decoder.norm(hidden)).chunk(3, dim=-1)
        query = query.reshape(1, 1, 4, 8).transpose(1, 2)
        key = key.reshape(1, 1, 4, 8).transpose(1, 2)
        value = value.reshape(1, 1, 4, 8).transpose(1, 2)
        keys = torch.index_copy(key_cache, 2, positions, key)
        values = torch.index_copy(value_cache, 2, positions, value)
        scores = (query @ keys.transpose(-2, -1)) / (8**0.5)
        probabilities = torch.softmax(scores + mask, dim=-1)
        attention = (probabilities @ values).transpose(1, 2).reshape(1, 1, 32)
        hidden = hidden + self.decoder.projection(attention)
        normalized = self.decoder.norm(hidden)
        hidden = hidden + self.decoder.down(
            torch.nn.functional.silu(self.decoder.gate(normalized)) * self.decoder.up(normalized)
        )
        return self.decoder.head(hidden), keys, values, probabilities


def get_model_and_inputs():
    # Load the unchanged prefill model so both interfaces use identical weights.
    source = Path(__file__).with_name("loader.py")
    spec = importlib.util.spec_from_file_location("_iteration_decoder_prefill", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    decoder, _ = module.get_model_and_inputs()
    keys = torch.zeros(1, 4, 35, 8)
    values = torch.zeros_like(keys)
    mask = torch.full((1, 1, 1, 35), -torch.inf)
    mask[..., 0] = 0
    return DecoderStep(decoder).eval(), (
        torch.tensor([[0]]),
        keys,
        values,
        torch.tensor([0]),
        mask,
    )
