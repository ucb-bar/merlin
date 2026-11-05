"""A captured read-only scale buffer stays an explicit invocation input."""

import torch


class ScaledValues(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([0.25, 1.0, 2.0]).reshape(1, 3, 1))

    def forward(self, values):
        return values * self.scale


def get_model_and_inputs():
    values = ((torch.arange(2 * 3 * 35).reshape(2, 3, 35) * 7) % 23 - 11).float() / 16
    return ScaledValues().eval(), (values,)
