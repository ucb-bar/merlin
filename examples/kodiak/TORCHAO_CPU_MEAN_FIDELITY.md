# Captured TorchAO CPU mean order: diagnostic mode

The TorchAO W8A8 TinyLlama capture made with PyTorch
`2.16.0.dev20261104+cpu` (`torch.version.git_version` =
`7f24a19c0967db04f2d5ba44c1e56eca489932ec`) uses the AVX2 CPU
`cascade_sum` order for contiguous f32 `aten.mean.dim` rows of length 2048.
PyTorch divides that sum by 2048. The source order matters: a different f32
sum can cross the following signed-int8 activation quantizer's rounding
boundary. This mode replays that source CPU sum in the captured graph. It is
off by default and is a source-fidelity diagnostic, not a general replacement
for mean or an accelerator optimization.

## Select and verify the source

In the same capture Python environment and on the same CPU used for the
reference, check the wheel and dispatch first:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.git_version)
print(torch.backends.cpu.get_cpu_capability())
PY
```

The version must be `2.16.0.dev20261104+cpu`, the full commit must be the
one above, and CPU capability must be `AVX2`. The source operands must be
contiguous, finite f32 rows with round-to-nearest-even arithmetic. Other
PyTorch builds and CPU ISAs need their own order check.

Enable the pass for the existing **source TorchAO W8A8 capture** compile
command, including any child processes, with:

```bash
export MERLIN_PASSES=torchao-cpu-avx2-mean
export MERLIN_TORCHAO_CPU_SOURCE_COMMIT=7f24a19c0967db04f2d5ba44c1e56eca489932ec
merlin-compile --list-passes
# Run the normal source-capture compile command here, then the host/device
# harness on the exact same input and captured weights as Torch eager.
```

`MERLIN_TORCHAO_CPU_AVX2_MEAN=1` is the equivalent direct switch. The pass
requires the exact commit assertion and `prov.quantization =
"int8_dyn_act_int8_weight"`; it changes only provenance-marked
`aten.mean.dim` f32 sums over the final K=2048 axis with a positive-zero
initializer and the expected add region. It preserves the captured division.
Compilation refuses a different source commit. The source commit assertion is
supplied by the caller; the pass cannot prove the wheel or CPU dispatch from
MLIR alone. Record the number of `torchao_cpu_avx2_means` rewrites from
preprocessing along with the build receipt.

## Compare every output

Save **every** Torch eager output and every emitted-host output for the same
input and weights as arrays under matching keys in two `.npz` files. For a
single output, use the key `output` in both. Record the capture receipt,
compiler commit, input seed, and source wheel in the result. This comparison
checks every element; it does not rely on a selected logit, argmax, or a
compiler-generated golden. On a device run, use the emitted-host result as a
separate execution check, then compare device output against Torch independently.

```bash
python - torch_outputs.npz merlin_outputs.npz <<'PY'
import sys
import numpy as np

with np.load(sys.argv[1]) as source, np.load(sys.argv[2]) as emitted:
    if set(source.files) != set(emitted.files):
        raise SystemExit(f"output keys differ: {source.files} vs {emitted.files}")
    for name in sorted(source.files):
        a, b = source[name], emitted[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise SystemExit(f"{name}: shape/dtype differ: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise SystemExit(f"{name}: nonfinite output; inspect elementwise before scoring")
        delta = a.astype(np.float64) - b.astype(np.float64)
        denominator = np.linalg.norm(a.astype(np.float64).ravel())
        rel_l2 = np.linalg.norm(delta.ravel()) / denominator if denominator else np.linalg.norm(delta.ravel())
        mismatches = int(np.count_nonzero(a.view(np.uint8) != b.view(np.uint8)))
        print(name, "shape", a.shape, "byte_mismatches", mismatches,
              "max_abs", float(np.max(np.abs(delta), initial=0)), "rel_l2", float(rel_l2))
PY
```

Use a captured intermediate probe to locate the **first** mismatching op if
any output differs. Compare the same input tensor at each op, not only final
model output. The current evidence is narrow:

- For the first TinyLlama RMSNorm on eight captured K=2048 rows, the input
  `pow` tensor was 16,384/16,384 f32 values bit exact before the reduction.
  The previous serial Merlin sum yielded 0/8 bit-exact means. The emitted
  source-order reduction yielded 8/8 bit-exact sums and 8/8 bit-exact means
  against the captured Torch CPU result.
- A separate scalar reference of the pinned AVX2 order matched 100/100
  independent positive rows and 100/100 signed rows bit for bit. The emitted
  path matched three frozen Torch rows at both `-O0` and `-O2`; the same LLVM
  IR compiled to an RV64GCV object. The real TinyLlama MLIR had five eligible
  reductions, and the rewritten module verified.

This does **not** establish whole-model TorchAO parity, SpectFormer parity,
RISC-V numerical parity, or physical Kodiak behavior. Re-run the complete
source-framework comparison after the pass is integrated, report every output
and its first divergence, then test the resulting image on the chosen device.
