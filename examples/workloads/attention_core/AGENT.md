# Attention-core capture

Authored PyTorch input for the shared capture pipeline. Keep generated captures,
weights, compiler output and evaluator tensors under an explicit artifact root.
The static keys and values are model parameters, never stored expected outputs.
This example is an attention core, not a complete transformer block or model.
