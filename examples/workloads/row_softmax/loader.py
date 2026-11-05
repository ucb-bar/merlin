"""Row-wise PyTorch softmax with a static input."""

import torch


class RowSoftmax(torch.nn.Module):
    def forward(self, scores):
        return torch.softmax(scores, dim=-1)


def get_model_and_inputs():
    scores = ((torch.arange(33 * 65).reshape(33, 65) % 67) - 33).to(torch.float32) / 8
    return RowSoftmax().eval(), (scores,)
