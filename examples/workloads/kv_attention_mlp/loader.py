"""One decoding step with explicit fixed-capacity K/V state and an MLP."""

import torch


class KVAttentionMLP(torch.nn.Module):
    def __init__(self, width=32, hidden=64):
        super().__init__()
        self.width = width
        for ordinal, (name, shape) in enumerate(
            (
                ("query", (width, width)),
                ("key", (width, width)),
                ("value", (width, width)),
                ("projection", (width, width)),
                ("up", (width, hidden)),
                ("down", (hidden, width)),
            )
        ):
            cells = torch.arange(shape[0] * shape[1]).reshape(shape)
            weight = (((cells * 7 + ordinal * 11) % 17) - 8).float() / 128
            self.register_parameter(name, torch.nn.Parameter(weight, requires_grad=False))

    def forward(self, values, key_cache, value_cache, positions, mask):
        query = values @ self.query
        key = (values @ self.key).unsqueeze(1)
        value = (values @ self.value).unsqueeze(1)
        keys = torch.index_copy(key_cache, 1, positions, key)
        cached_values = torch.index_copy(value_cache, 1, positions, value)
        scores = (query.unsqueeze(1) @ keys.transpose(1, 2)) / (self.width**0.5)
        probabilities = torch.softmax(scores + mask, dim=-1)
        attention = (probabilities @ cached_values).squeeze(1)
        hidden = values + attention @ self.projection
        output = hidden + torch.relu(hidden @ self.up) @ self.down
        return output, keys, cached_values, probabilities


def get_model_and_inputs():
    width, capacity = 32, 35
    values = ((torch.arange(width).reshape(1, width) * 3) % 17 - 8).float() / 32
    keys = torch.zeros(1, capacity, width)
    cached_values = torch.zeros_like(keys)
    mask = torch.full((1, 1, capacity), -torch.inf)
    mask[..., 0] = 0
    return KVAttentionMLP(width).eval(), (values, keys, cached_values, torch.tensor([0]), mask)
