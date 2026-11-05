"""A captured attention core with multiple tiles and non-square tails."""

import torch


class AttentionCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = torch.nn.Parameter(torch.randn(65, 65) / 8)
        self.values = torch.nn.Parameter(torch.randn(65, 35) / 4)

    def forward(self, query):
        probabilities = torch.softmax(query @ self.keys, dim=-1)
        return probabilities, probabilities @ self.values


def get_model_and_inputs():
    return AttentionCore().eval(), (torch.randn(33, 65) / 4,)
