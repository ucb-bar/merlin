"""Bias-free two-layer MLP with its intermediate activation as a second result."""

import os

import torch


class TwoLayerMLP(torch.nn.Module):
    def __init__(self, width, hidden, outputs):
        super().__init__()
        self.expand = torch.nn.Linear(width, hidden, bias=False)
        self.project = torch.nn.Linear(hidden, outputs, bias=False)

    def forward(self, tokens):
        activation = torch.relu(self.expand(tokens))
        return activation, self.project(activation)


def get_model_and_inputs():
    dimensions = tuple(int(os.environ.get("MLP_" + key, value)) for key, value in (
        ("ROWS", "64"), ("WIDTH", "128"), ("HIDDEN", "512"), ("OUTPUTS", "128"),
    ))
    if any(value <= 0 for value in dimensions):
        raise ValueError("MLP dimensions must be positive")
    rows, width, hidden, outputs = dimensions
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(912)
        model = TwoLayerMLP(width, hidden, outputs).eval()
    tokens = ((torch.arange(rows * width).reshape(rows, width) % 29) - 14).float() / 64
    return model, (tokens,)
