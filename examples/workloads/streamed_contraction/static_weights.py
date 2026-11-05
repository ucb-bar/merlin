"""A projection with immutable weights larger than a small device window."""

import torch


class StaticProjection(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.nn.Parameter(weights, requires_grad=False)

    def forward(self, values):
        return torch.relu(values @ self.weights)


def get_model_and_inputs():
    m, k, n = 1, 128, 2049
    values = (torch.arange(m * k).reshape(m, k) % 3 + 1).float() / 16
    rows, columns = torch.arange(k)[:, None], torch.arange(n)[None, :]
    weights = torch.where(rows % 11 == columns % 11, 1.0, 0.0)
    weights *= torch.where(columns % 2 == 0, 1.0, -1.0) / 16
    return StaticProjection(weights).eval(), (values,)
