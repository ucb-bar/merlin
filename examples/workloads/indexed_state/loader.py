import torch


class IndexedState(torch.nn.Module):
    def forward(self, cache, update, positions):
        updated = torch.index_copy(cache, 1, positions, update)
        return updated, updated.sum(dim=1)


def get_model_and_inputs():
    cache = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4) / 16
    update = torch.full((2, 1, 4), 0.5)
    return IndexedState().eval(), (cache, update, torch.tensor([3], dtype=torch.int64))
