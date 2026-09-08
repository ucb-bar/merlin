# Canonical multi-model Phase-2 compiler checkpoint

This directory is the directly invocable compiler checkpoint assembled after q534. It combines
three orthogonal, target-general mechanisms in one immutable snapshot:

1. Standard LLVM `roundeven` and `fptosi` scalar lowering. This is enabled by default and is the
   mechanism measured by FireSim queue job 534.
2. Capability-, geometry-, layout-, and semantics-driven native `LOOP_CONV_WS` selection. This is
   enabled by default but fails closed when the current boundary cannot be represented exactly.
3. A dynamic-activation/static-per-channel-weight i8 contraction bridge. This changes source-f32
   arithmetic and is therefore disabled unless the explicit numeric contract is requested.

There are no model names, layer names, or fixed workload shapes in these mechanisms. The compiler
tree SHA-256 is `38e42aea90f29e271ddc510daca09901cc36970323e2926d1d9426a6e6ee1fdf`
over 42 files. The bundled Merlin support snapshot is
`0714adc43f25efb2421afc37519190f9a2b58c75ad75d7a08b9ff3537936498a`
over 1,016 files.

## What is qualified

The complete default four-model compile gate passes sequentially with a peak compiler RSS of
308,860 KiB:

| Workload | Compile wall time | Default outcome |
|---|---:|---|
| ResNet-50 W8A8 | 18.80 s | q534 target LLVM byte-identical; 53 audited native-conv refusals |
| TinyLLaMA | 1.39 s | byte-identical all-host fallback |
| LSTMNetViT W8A8 | 51.22 s | byte-identical mixed host/mesh target |
| SmolVLA W8A8 | 46.85 s | byte-identical mixed host/mesh target; `i1` storage qualified |

The unit/semantic gate is 31/31. Two reduced native-convolution programs execute exactly on local
Spike: 192 outputs in 63 proxy cycles and a forced two-descriptor K-reduction with 144 outputs in
79 proxy cycles. These proxy counters establish semantics and relative mechanism viability, not
FPGA performance.

ResNet's canonical target LLVM is byte-for-byte the q534 target. Therefore the q534 hardware result
transfers without another expensive run: 1,537,416,019 measured compute cycles, 1000/1000 exact
logits, 166,648,204 cycles saved relative to q530 (9.7795%, 1.1084x). Native convolution adds no
ResNet speedup yet because all 53 convolutions expose an `i32 NCHW` output while this hardware's
loop-convolution store is narrow and physically NHWC/HWNC. The selector records that dependency
and keeps the proven row-streamed fallback.

With the explicit dynamic-weight contract, TinyLLaMA moves 15 contractions to the mesh (45
accelerator commands, 3,424,256 declared MACs). The frozen numeric witness has 0/2,048 tolerance
violations, maximum absolute error 0.018019676, relative L2 0.007236148, cosine 0.999973894, and
unchanged top-1 for all 8 tokens. This is compiler-correctness evidence for that one frozen witness,
not source-f32 bit equivalence, dataset accuracy, hardware execution, or a performance claim.

Full machine-readable evidence is in
[`validation/canonical_receipt.json`](validation/canonical_receipt.json).

## Verify

From this directory:

```sh
./verify.py

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/compiler" \
/scratch/agustin/projects/oscar-merlin/.venv/bin/pytest -q tests
```

Expected: the canonical verifier passes and 31 tests pass. To rerun the two reduced native-
convolution executions:

```sh
PYTHONDONTWRITEBYTECODE=1 \
/scratch/agustin/projects/oscar-merlin/.venv/bin/python \
  validation/run_reduced_spike.py
```

## Compile a model directly

```sh
artifact=/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-bench/gemmini/development_phase2_canonical_multimodel_20260908
input=/absolute/path/to/model.mlir
output_dir=/absolute/path/to/output
mkdir -p "$output_dir"

MERLIN_PYTHON=/scratch/agustin/projects/oscar-merlin/.venv/bin/python \
  "$artifact/run-gemmini-opt" \
  --source-convolution \
  --convert-iface-to-gemmini \
  --emit-command-buffer="$output_dir/command_buffer.json" \
  --emit-target-artifact \
  -o "$output_dir/target.mlir" \
  "$input"
```

The command emits the compiler-owned command buffer and target LLVM-dialect MLIR. Native
object/ELF construction stays runner-owned because it must use the selected target's pinned LLVM,
ISA, linker script, harness ABI, and stack preflight.

For the qualified but non-bit-equivalent TinyLLaMA bridge, add:

```sh
--dynamic-weight-only-contract symmetric_per_output_channel_roundeven_v1
```

Do not enable that option silently for arbitrary models. It must be paired with a declared numeric
policy and a representative accuracy qualification.

## Current macro dependencies

The checkpoint is usable now, but it is not the end of the performance campaign. The high-value
remaining work is explicit:

- form an exact quantized/narrow epilogue and choose a globally consistent layout so native
  convolution can replace the 53 ResNet row-streamed im2col paths;
- fuse or eliminate host epilogue/residual boundaries while preserving source rounding order;
- reduce repeated im2col/addressing/synchronization work and physical traffic;
- quantify and move the remaining host contractions in LSTMNetViT and SmolVLA; and
- extend TinyLLaMA qualification from one frozen witness to representative accuracy evidence.

No new FireSim run was made for this assembly because its default ResNet target is already the
exact q534 target. A future hardware run is warranted only after one of those structural counters
moves materially, and it must use the queue with
`firesim kill -> firesim infrasetup -> firesim runworkload -> firesim kill`, one untimed warm run,
then a measured compute-only interval.
