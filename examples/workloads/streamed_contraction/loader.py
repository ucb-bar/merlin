"""A matrix projection whose packed output exceeds a small device window."""

import torch


class StreamedContraction(torch.nn.Module):
    def forward(self, values, weights):
        return torch.relu(values @ weights)


def get_model_and_inputs():
    m, k, n = 257, 32, 513
    values = (torch.arange(m * k).reshape(m, k) % 3 + 1).float() / 16
    rows = torch.arange(k)[:, None]
    columns = torch.arange(n)[None, :]
    weights = torch.where(rows % 11 == columns % 11, 1.0, 0.0)
    weights *= torch.where(columns % 2 == 0, 1.0, -1.0) / 16
    return StreamedContraction().eval(), (values, weights)
