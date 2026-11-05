# Bounded model tile runner

`tools/run_model_tile.py` converts and packs source operands, assembles a fixed
schedule with the pinned model assembler, and executes decoded instruction
objects using `rtl` numerics. It retains raw instruction words for review; the
model executes the decoded objects.

The default `atlas_mxu0_fp8_unit_bf16acc_v1` recipe rounds source operands to
BF16 nearest-even and applies VPU FP8Pack at unit scale (E8M0 code 127). MXU0
rounds every MAC to BF16, overwrites the first K32 contribution, and accumulates
subsequent contributions. The original native output is retained as a diagnostic
reference.

`--recipe tensor-pow2` selects `atlas_mxu0_fp8_tensor_pow2_bf16acc_v1`. Each
complete BF16 operand tensor gets the smallest power-of-two scale whose
normalized maximum is at most 448; all-zero tensors use exponent zero. Exact
rational comparisons select exponents in `[-126,127]`. The scalar encoder and
pinned VPU converter must agree byte for byte. Scales remain fixed across K and
output tiles; VLI.ALL and VMUL.BF16 rescale both output halves after the pop.
Their delays count in model cycles. The product scale must be normal finite
BF16, output overflow is rejected, and signed zero follows VPU multiplication.

## Run

Use Python 3.14 with the model dependencies and physical paths without symlinks.
The model must be at `9e6313bddecb8ba936206eb347f5cd4e5335282c`, with unchanged
tracked sources and imports inside that checkout. Set
`PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1`.

```sh
python tools/run_model_tile.py --model-root /physical/pinned/npu-model \
  --input /physical/capture/operands.npz --manifest /physical/capture/manifest.json \
  --out /physical/new-artifact-directory
```

The NPZ contains finite float32 `A[32,K]`, `B[K,32]`, and `native_C[32,32]`,
with K a positive multiple of 32 up to 4096. BF16 captures use exact float32
views; optional uint16 `A_bf16`, `B_bf16`, and `native_C_bf16` authenticate
those views. ZIP members must be unique flat `.npy` names from that inventory
and total at most 32 MiB uncompressed. The manifest requires `schema_version:
1`, `npz_sha256`, `source_dtype` (`float32` or `bfloat16`), and nonempty
`source` and `site` metadata. The capture defines the operation boundary and
full reduction range.

`--smoke` replaces the input/manifest with a nonsymmetric synthetic K64 fixture.
Default runs use two independently initialized model instances and distinct
deterministic storage seeds. Completion requires ECALL, drained engines and DMA,
exact output bytes, preserved inputs/guards, and observed DMA bytes equal to
requested transfers. Smoke also checks omitted transpose, reset final-K
accumulation, and a wrong oracle; `--negative-controls` selects those checks for
another K>=64 fixture.

`receipt.json` binds inputs, runner, oracle, imported model sources, memory
curve, and outputs. It records fetch-to-ECALL cycles, issue/stall/engine
observations, DMA bytes, guards, and native-error diagnostics. `cycles-*.jsonl`
and `perfetto-*.json` retain event traces. Raw source, encoded operands, initial
DRAM, expected output, actual output, and instructions remain available on
failure. Source and input hashes are rechecked before success.

## Workload contractions

`third_party/atlas-kernels/kernels/projection_gemm/atlas_program.py` supplies assembly and input checks to
`tools/compile_workload_inputs.py` and `tools/run_workload_contraction.py`. The
latter accepts finite FP32 physical operands with positive M/N/K multiples of
32, K<=8192, at most 1,048,576 output cells, at most 64 MiB guarded DRAM, and at
most 8192 instruction words. The source archive limit remains 32 MiB. Callers
provide tail padding and a source-bound A/B/native_C archive and manifest.

The workload producer selects resident B when it fits one 256 KiB bank,
otherwise resident A when it fits, otherwise streams both. Its receipt records
resident/scratch bounds, the DMA low32 window under high base 1, whole-output
equality, traffic, guards, and drained ECALL. Five producer helpers are checked
and hashed before and after execution. For complete host integration, use the
[ResNet graph route](resnet-full-host.md).

## Arithmetic providers and checks

The default scalar oracle uses exact rational arithmetic with bounded caches.
The opt-in NumPy module `third_party/atlas-kernels/lib/golden/atlas_vector_arithmetic.py` exposes
`ordered_matmul(uint8[M,K], uint8[K,N]) -> uint16[M,N]`, `round_bf16(float64) ->
uint16`, and `rescale_bf16_array(uint16, exponent) -> uint16`. Callers
explicitly select and hash that provider. Both implementations visit K in order
and round each nonzero FP8 product update directly to BF16 nearest-even.
Binary64 intermediates are exact for the finite FP8 grid and bounded BF16
accumulator; the module docstring gives the proof.

```sh
PYTHONPATH=tools:test python -m unittest test_tile_arithmetic
python -m unittest discover -s third_party/atlas-kernels/lib/tests -p test_vector_arithmetic.py
```

These tests compare finite FP8 products, all finite BF16 decodes, ordered
chains, rounding boundaries, rescaling, signed zero, and physical output layout.
Run `test_workload_contraction` for workload geometry and physical payload
checks.

Model results describe target arithmetic and instruction-model execution. Native
error and underflow are diagnostic; they do not establish task accuracy,
calibration, host execution, or platform timing. Changed source pins, arithmetic
policy, or instruction images require fresh qualification at the affected
boundary.
