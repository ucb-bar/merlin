# Recipe: SpacemiT X60 (Linux Board)

End-to-end steps to compile a model with Merlin and run it on a SpacemiT K1
board (X60 RISC-V cores) over SSH.

## Prerequisites

- SpacemiT K1 board accessible via SSH.
- SpacemiT toolchain installed on the build host.

## Steps

### 1. Build the SpacemiT runtime

```bash
conda run -n merlin-dev uv run tools/build.py \
    --profile spacemit \
    --config release
```

### 2. Compile the model

```bash
conda run -n merlin-dev uv run tools/compile.py \
    models/mlp/mlp.q.int8.mlir \
    --target spacemit_x60 \
    --quantized
```

### 3. Deploy to the board

Copy the runtime installation and compiled model artifacts to the board:

```bash
scp -r build/spacemit-merlin-release/install/ root@10.44.86.251:/opt/merlin/
scp build/compiled_models/mlp/*.vmfb root@10.44.86.251:/opt/merlin/models/
```

### 4. Run on the board

```bash
ssh root@10.44.86.251
/opt/merlin/install/bin/iree-benchmark-module \
    --module=/opt/merlin/models/mlp.q.int8.vmfb \
    --function=main
```

## Reference

See the samples in `samples/SpacemiTX60/` for more advanced usage patterns
including dual-model async execution and dispatch-level scheduling.
