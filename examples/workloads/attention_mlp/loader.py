"""A complete single-head attention block followed by a two-layer MLP.

The two results expose the MLP activation and final output. Parameters are
ordinary dense PyTorch tensors; capture externalizes them for compilation.
"""

import math
import os

import torch


class AttentionMLP(torch.nn.Module):
    def __init__(self, width=32, hidden=32):
        super().__init__()
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.fc1 = torch.nn.Linear(width, hidden, bias=False)
        self.fc2 = torch.nn.Linear(hidden, width, bias=False)
        self.scale = 1.0 / math.sqrt(width)

    def forward(self, tokens):
        query, key, value = self.query(tokens), self.key(tokens), self.value(tokens)
        probabilities = torch.softmax((query @ key.transpose(-2, -1)) * self.scale, dim=-1)
        activation = torch.relu(self.fc1(probabilities @ value))
        return activation, self.fc2(activation)


def get_model_and_inputs():
    hidden = int(os.environ.get("ATTENTION_MLP_HIDDEN", "32"))
    if hidden <= 0:
        raise ValueError("ATTENTION_MLP_HIDDEN must be positive")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(912)
        model = AttentionMLP(hidden=hidden).eval()
    tokens = ((torch.arange(32 * 32).reshape(32, 32) % 29) - 14).to(torch.float32) / 16
    return model, (tokens,)
