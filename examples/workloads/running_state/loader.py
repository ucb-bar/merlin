"""A functional tensor state update with an independently returned output."""

import torch


class RunningState(torch.nn.Module):
    def forward(self, history, values):
        updated = history + values
        return torch.relu(updated), updated


def get_model_and_inputs():
    history = torch.zeros((2, 3, 35), dtype=torch.float32)
    values = ((torch.arange(history.numel()).reshape(history.shape) * 7) % 23 - 11).float() / 16
    return RunningState().eval(), (history, values)
