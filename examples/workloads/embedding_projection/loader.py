"""Dynamic token lookup and a projection with floating and integer outputs."""

import torch


class EmbeddingProjection(torch.nn.Module):
    def __init__(self, rows=17, width=7, projected=5):
        super().__init__()
        self.table = torch.nn.Embedding(rows, width)
        self.project = torch.nn.Linear(width, projected, bias=False)
        self.width, self.projected = width, projected

    def forward(self, indices):
        values = self.table(indices)
        projected = self.project(values.reshape(-1, self.width))
        return values, projected.reshape(*indices.shape, self.projected), indices


def get_model_and_inputs():
    model = EmbeddingProjection().eval()
    with torch.no_grad():
        for ordinal, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel()).reshape(parameter.shape)
            parameter.copy_(((values * 37 + ordinal * 53) % 251 - 125).float() / 512)
    return model, (torch.tensor([[16, 0, 7], [4, 9, 1]], dtype=torch.int64),)
