# Batched matrix multiplication

Two runtime operands, `[2,3,35]` and `[2,35,5]`, produce `[2,3,5]` through
`torch.bmm`. Each batch has different operands. The reduction and output shapes
exercise tails; no batch may use another batch's weights or accumulated state.

Capture `loader.py` with the same model2MLIR command as the other public
workloads. Stage compiler inputs separately from runtime tensors and references.
Any target precision change must be selected explicitly and checked separately
from original-model quality.
