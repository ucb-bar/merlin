"""Broadcast additive-mask softmax with more than one column tile."""

import torch


class MaskedSoftmax(torch.nn.Module):
    def forward(self, scores, mask):
        return torch.softmax(scores + mask, dim=-1)


def get_model_and_inputs():
    scores = ((torch.arange(2 * 3 * 35).reshape(2, 3, 35) * 7) % 23 - 11).float() / 16
    rows, columns = torch.arange(3)[:, None], torch.arange(35)[None, :]
    keep = (columns % 3 != 1) & ((columns <= rows * 12) | (columns == 0))
    mask = torch.where(keep, 0.0, -torch.inf)
    return MaskedSoftmax().eval(), (scores, mask)
