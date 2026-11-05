"""A small transformer encoder: a bring-your-own-model template for the Kodiak OPU route.

Two encoder layers over 64 tokens of width 128 (MLP 256). Every projection is a matmul with M = 64 tokens
and N >= 64, so each fills the OPU's 64x64 tile and is routed to it; the attention score/value products
(per head, 64x32) are smaller than one tile and stay on RVV. Copy this file, replace the model, keep
`get_model_and_inputs()`:

    examples/kodiak/run.sh capture my_model --loader path/to/my_model/loader.py

Weights are deterministic (no checkpoint download), so the capture and its golden are reproducible.
"""

import torch

TOKENS, WIDTH, HEADS, MLP, LAYERS = 64, 128, 4, 256, 2


def get_model_and_inputs():
    torch.manual_seed(0)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=WIDTH, nhead=HEADS, dim_feedforward=MLP, dropout=0.0, batch_first=True, norm_first=True
    )
    model = torch.nn.TransformerEncoder(layer, num_layers=LAYERS, enable_nested_tensor=False).eval()
    inputs = (torch.randn(1, TOKENS, WIDTH),)
    return model, inputs
