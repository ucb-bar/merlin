"""Static multi-axis reduction with two observable outputs."""

import torch


class TensorReduction(torch.nn.Module):
    def forward(self, value):
        summed = value.sum(dim=(0, 2))
        return summed, torch.relu(summed)


def get_model_and_inputs():
    value = ((torch.arange(2 * 3 * 35).reshape(2, 3, 35) % 29) - 14).float() / 4
    return TensorReduction().eval(), (value,)
