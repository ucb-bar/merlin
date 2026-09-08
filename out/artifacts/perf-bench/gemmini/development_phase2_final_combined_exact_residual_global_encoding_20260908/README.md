# Final combined Phase-2 compiler

This is the directly invocable, content-addressed Phase-2 compiler artifact. It combines the
canonical native-scalar and affine-im2col improvements with exact ordered epilogue formation,
exact two-tensor residual fusion, and target-neutral global encoding/lifetime planning. It has no
model, layer, or fixed-shape dispatch.

Verify the complete release seal first:

```sh
artifact=/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-bench/gemmini/development_phase2_final_combined_exact_residual_global_encoding_20260908
PYTHONDONTWRITEBYTECODE=1 python "$artifact/verify.py"
```

Then compile a model:

```sh
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

Object and ELF construction remain runner-owned because they must use the selected target's pinned
compiler, ISA, linker script, harness ABI, and hardware facts. TinyLLaMA's numerically qualified,
non-bit-exact bridge is available only with the explicit option
`--dynamic-weight-only-contract symmetric_per_output_channel_roundeven_v1`.

The compiler tree SHA-256 is
`48694957d14c9608960f7ac2b6cd08b72b15c55e4a11ff26e1cf952e0cd7607e` over 45 files. The
65-test suite, two exact warm/reentrant witnesses, four-model compile gate, exact ownership and
encoding audits, and transferred ResNet Spike A/B are bound by `verify.py`.

ResNet's final target is byte-identical to the sealed residual child: 492,147,976 Spike proxy
cycles versus 506,265,226 for the affine parent (2.7885% reduction), with 1,000/1,000 logits exact.
This is not a new hardware claim. The latest honest full-Merlin hardware checkpoint remains q535 at
1,316,619,699 FireSim cycles. q536's 555,991,472 cycles are a hybrid native-convolution opportunity
bound with a TVM-owned host graph/arena, not a Merlin compiler speedup.

See [FINAL_COMBINED.md](FINAL_COMBINED.md) and
`validation/final_combined_receipt.json` for the detailed result. The reviewable compiler delta is
`final_combined_compiler.patch`.
