"""Independent softmax rows with a leading batch dimension and column tails."""

import torch


def get_model_and_inputs():
    scores = ((torch.arange(2 * 3 * 35).reshape(2, 3, 35) % 67) - 33).float() / 8
    return torch.nn.Softmax(dim=-1).eval(), (scores,)
