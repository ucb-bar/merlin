"""Independent matrix products with distinct runtime weights in each batch."""

import torch


class BatchedMatmul(torch.nn.Module):
    def forward(self, lhs, rhs):
        return torch.bmm(lhs, rhs)


def get_model_and_inputs():
    lhs = ((torch.arange(2 * 3 * 35).reshape(2, 3, 35) * 7) % 23 - 11).float() / 64
    rhs = ((torch.arange(2 * 35 * 5).reshape(2, 35, 5) * 11) % 29 - 14).float() / 64
    return BatchedMatmul().eval(), (lhs, rhs)
