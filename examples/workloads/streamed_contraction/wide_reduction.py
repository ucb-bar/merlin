"""A projection whose K dimension requires exact accumulator carry between launches."""

import torch


class WideReduction(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.nn.Parameter(weights, requires_grad=False)

    def forward(self, values):
        return torch.relu(values @ self.weights)


def get_model_and_inputs():
    m, k, n = 3, 8193, 2
    values = torch.ones((m, k)) / 16
    values[1] *= 2
    values[:, 1] /= 16
    weights = torch.zeros((k, n))
    weights[0] = torch.tensor([-1.0, 1.0]) / 16
    weights[1] = torch.tensor([-1.0, 1.0]) / 128
    weights[-1] = torch.tensor([1.0, -1.0]) / 8
    return WideReduction(weights).eval(), (values,)
