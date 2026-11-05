"""Temperature-one attention core and a bias-free MLP with non-square tails.

This public compiler input complements the projected attention block in
loader.py. It does not replace the required iteration models.
"""

import torch


class TiledAttentionMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = torch.nn.Parameter(torch.randn(65, 65) / 8)
        self.values = torch.nn.Parameter(torch.randn(65, 35) / 4)
        self.up = torch.nn.Parameter(torch.randn(35, 67) / 8)
        self.down = torch.nn.Parameter(torch.randn(67, 29) / 8)

    def forward(self, query):
        probabilities = torch.softmax(query @ self.keys, dim=-1)
        attention = probabilities @ self.values
        hidden = torch.relu(attention @ self.up)
        return probabilities, hidden, hidden @ self.down


def get_model_and_inputs():
    return TiledAttentionMLP().eval(), (torch.randn(33, 65) / 4,)
