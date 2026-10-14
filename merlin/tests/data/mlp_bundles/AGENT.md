# AGENT.md — merlin/tests/data/mlp_bundles

## Purpose

Two tiny whole-model bundles captured by model2MLIR, the inputs of the external-kernel tests
(`fixtures/mlp_bundles.py` loads them): a 3-layer MLP 16 -> 32 -> 32 -> 8, same seed, weights and
input in both.

- `mlp_basic/` — plain PyTorch; every op lowered to linalg.
- `mlp_basic_triton/` — fc1+relu+fc2+relu is one Triton op, so `model.mlir` holds one
  `func.call @triton_exp__fused_mlp2` tagged `prov.kernel` (body linked from `src/merlin/kernels/triton_llvmir/`).

Each holds `model.mlir`, `weights.safetensors` (+ `.manifest.json`), `inputs.npz`, `input_order.json`
and `golden.npy` (`mlp_basic`'s PyTorch output, the oracle for both). `prov.weights_file` is the
bundle-relative `"weights.safetensors"`.

## Regenerating

They are a snapshot of model2MLIR's output, tracked so these tests need no model2MLIR checkout. Never
edit them by hand; re-capture with

    $MERLIN_M2M_VENV/bin/python build_tools/scripts/gen_mlp_kernel_fixtures.py \
        --m2m $MERLIN_M2M_DIR --out merlin/tests/data/mlp_bundles

`merlin/tests/infra/test_mlp_bundles_fresh.py` re-captures them with the configured model2MLIR and fails,
naming the files, when the snapshot is stale (skipped when no model2MLIR checkout is configured).
