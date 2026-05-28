# Multi-granularity demo (Part C / C7)

Tiny purpose-built model that exercises three kernel-embedding granularities
concurrently in one whole-model compile:

| Layer | Granularity  | Substitution                                              |
| ----- | ------------ | --------------------------------------------------------- |
| A     | MEGAKERNEL   | matmul -> bias_add -> relu chain (4 dispatches -> 1)      |
| B     | LAYER        | single 1024-elementwise mul (1 dispatch -> 1)             |
| C     | TILE         | 256x256 matmul tiled to 4x 64x64 sub-tiles (1 -> 4)       |

## Running

```bash
# 1. Compile the un-substituted baseline (golden output reference).
./merlin compile models/research/multi_granularity_demo/multi_granularity_demo.mlir \
    --target host --output-dir build/compiled_models/multi_granularity_demo/baseline

# 2. Auto-scaffold a kernel manifest stub via breakdown_vmfb.py.
tools/breakdown_vmfb.py \
    --output-dir build/compiled_models/multi_granularity_demo/baseline \
    --emit-kernel-manifest build/compiled_models/multi_granularity_demo/kernels \
    --kernel-source-lang c \
    --kernel-targets llvm-cpu-x86_64

# 3. Fill in the generated src/<name>.c stubs with kernel bodies (or paste
#    in Claude/KernelBlaster outputs). For this demo, drop the three
#    reference kernels from kernels/ into the corresponding stubs.

# 4. Compile with kernel embedding.
./merlin compile models/research/multi_granularity_demo/multi_granularity_demo.mlir \
    --target host \
    --kernels-dir build/compiled_models/multi_granularity_demo/kernels \
    --output-dir build/compiled_models/multi_granularity_demo/embedded

# 5. Verify byte-equality on a fixed input batch.
tests/granularity/run_byte_equality.sh \
    build/compiled_models/multi_granularity_demo/baseline/multi_granularity_demo.vmfb \
    build/compiled_models/multi_granularity_demo/embedded/multi_granularity_demo.vmfb
```

## Reference kernels

`kernels/` (the auto-scaffolded directory after step 2) gets populated with
empty stubs. The three reference implementations:

- `tile_matmul_64x64.c` — TILE kernel, computes a 64x64 sub-tile of a
  matmul. Plain f32 triple loop. Performance-not-the-goal; verify byte-
  identical output against the un-embedded path.
- `elementwise_mul_1024.c` — LAYER kernel, single elementwise mul.
- `matmul_bias_relu_1024.c` — MEGAKERNEL, fused matmul + bias_add +
  relu, plain f32 triple loop.

All kernels use `__attribute__((visibility("hidden")))` and the IREE
embedded-elf ABI. See `samples/custom_dispatch/cpu/embedded` for the
manual-template counterpart.

## Test matrix

The full granularity test matrix lives under `tests/granularity/`:

- `tile/<op>_<shape>_<dtype>.mlir` — TILE candidates
- `layer/<op>_<shape>_<dtype>.mlir` — LAYER candidates
- `megakernel/<chain>_<shape>_<dtype>.mlir` — MEGAKERNEL candidates

Each fixture has a paired golden output (computed by running the
un-embedded baseline) and a `kernels/` directory with the substitution
implementation. The test driver: substitute -> compile -> run ->
byte-compare.
