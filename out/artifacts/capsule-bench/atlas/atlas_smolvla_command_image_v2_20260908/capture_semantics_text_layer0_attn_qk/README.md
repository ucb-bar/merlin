# Real-capture Atlas qualification: `atlas_p0102`

This directory qualifies one physical SmolVLA partition, not the whole model.
The partition is the first text-layer attention QK contraction:
`matmul_101`, `B=15, M=113, K=64, N=113`. The tensors in
`capture_operands.npz` were tapped from the saved ExportedProgram after all 500
state tensors and all six inputs were replaced with the recorded capture
values. That reconstructed program's complete output is bit-identical to the
saved `golden.npy`; `capture_boundary.json` binds every source artifact and
tensor by SHA-256.

The raw matmul has exactly one immediate captured consumer: FX `mul_22`, the
same frontier recorded as MLIR `mul_32`, which scales by exactly 0.125 before
masking and softmax. The qualified publication boundary therefore folds that
fixed host scalar after BF16 dequantization. This boundary was selected from
the captured graph before evaluating GSIM output. The receipt also retains the
unscaled raw-matmul comparison as a non-acceptance diagnostic.

The fresh fixed-backend image has 30,986 words and executes the exact captured
operands on assertion-enabled elaborated-RTL GSIM in 8,700,444 cycles. The raw
spec contains no expected output. Independent NumPy references give:

- captured frontier: cosine 0.9992777705, max absolute error 0.1032150984;
- decoded FP8 quantization domain: cosine 0.9999981523, max absolute error
  0.0062534809.

Both pass the pre-existing fixed gate (`cosine >= 0.995`, `max abs <= 0.125`).
Changing the batched command to consume raw `W`, deleting its resident-weight
eviction, or perturbing the source reference all fail closed. The complete
receipt is `result.json`; raw words, preloads, GSIM stdout/stderr, device BF16
bytes, calibration, command dependencies, and assertion status are retained in
this directory.

Regeneration:

```sh
/scratch/agustin/projects/model2MLIR/.venv/bin/python extract_capture_batched_partition.py
MERLIN_ATLAS_GSIM_DIR=/scratch/agustin/tmp/gsim-atlas-core \
  ../../../../../.venv/bin/python run_capture_partition.py text_layer0_attn_qk
```

This promotes physical capture coverage from 3/391 to 4/391 and physically
qualified conversion events from 12/1,250 to 15/1,250. It does not qualify the
later mask/softmax chain, event/DMA runtime, whole-model correctness,
whole-model performance, or E2E execution.
