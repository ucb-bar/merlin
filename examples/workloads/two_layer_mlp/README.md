# Two-layer MLP

This PyTorch input computes a bias-free linear layer, ReLU, and a second linear
layer. It returns both the hidden activation and final output.

The default dimensions are input **64×128**, hidden **64×512**, and output
**64×128**: 8,388,608 multiply-accumulate steps across the two projections.
Weights are deterministic random values, not a trained checkpoint.

Set `MLP_ROWS`, `MLP_WIDTH`, `MLP_HIDDEN`, and `MLP_OUTPUTS` before capture to
choose other positive static dimensions. The compiler reads captured shapes;
these variables are only loader settings. A backend may reject dimensions or
operations outside its supported domain.

Capture and stage with the same commands as the
[attention + MLP example](../attention_mlp/README.md#capture), replacing its
loader path with `examples/workloads/two_layer_mlp/loader.py`. Target precision
selection and comparison against the original PyTorch outputs are separate
from capture.
