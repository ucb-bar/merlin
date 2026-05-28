# SaturnOPU Kernel Embedding — MLIR Phase Snapshots

This directory holds **what the SaturnOPU kernels look like inside a real
compile** at successive IREE pipeline phases. Mirrors the IREE canonical
sample at
[`third_party/iree_bar/samples/custom_dispatch/cpu/embedded/`](../../../third_party/iree_bar/samples/custom_dispatch/cpu/embedded/),
which keeps `example_transform.mlir`, `example_stream.mlir`, and
`example_hal.mlir` as reference artifacts of the same flow.

## Snapshots

Each subdirectory holds six MLIR files plus a `COVERAGE.txt` summary:

| File | Phase | What it shows |
|---|---|---|
| `0_input.mlir` | input | The user's MLIR before any kernel rewriting. |
| `1_transform_spec.mlir` | spec | The auto-generated transform-dialect spec from `tools/kernels/spec_gen.py`. |
| `2_after_preprocessing.mlir` | 3 | linalg ops rewritten into `flow.dispatch @kb_*` calls. |
| `3_flow.mlir` | 6 | Flow form: `util.call @call_*` wrappers + `hal.executable.source` with linked `.o`. |
| `4_stream.mlir` | 7 | Stream-dialect form (mirror of IREE's `example_stream.mlir`). |
| `5_hal.mlir` | 11 | HAL-dialect form (mirror of IREE's `example_hal.mlir`). |
| `COVERAGE.txt` | — | Human-readable summary: how many ops matched each kernel + what's left unmatched. |

## What's here

| Subdir | Source | Purpose |
|---|---|---|
| `add_f32/` | `tests/granularity/fixtures/embed_pipeline/add_input.mlir` | Smallest case — 1D dynamic elementwise add, no push constants. |
| `linear_f32/` | `tests/granularity/fixtures/embed_pipeline/matmul_input.mlir` | 2D matmul-transpose-B with M, K, N as push constants. |
| `dronet_partial/` | `models/dronet/dronet.mlir` | **Real model: full f32 dronet, partial coverage with current 3-kernel manifest.** |

## Reading `dronet_partial/` (multi-kernel demo)

`COVERAGE.txt` shows the current state:

```
--- Kernel call sites in flow phase (one line per actual rewrite) ---
      7 util.call @call_saturnopu_conv_2d_nchw_fchw_f32
      2 util.call @call_saturnopu_matmul_f32
      1 util.call @call_saturnopu_pooling_nchw_max_f32

--- Unmatched linalg ops (still in dispatch-creation) ---
     21 linalg.generic
      3 linalg.fill
```

**Three kernels firing simultaneously**, ten rewrites in dronet:

| Kernel | Times fired | Dronet ops covered |
|---|---:|---|
| `saturnopu_conv_2d_nchw_fchw_f32` | 7 | every `linalg.conv_2d_nchw_fchw` (the entire conv backbone) |
| `saturnopu_matmul_f32` | 2 | both classifier-head matmuls (`(1×6272) × (6272×1)`) |
| `saturnopu_pooling_nchw_max_f32` | 1 | the post-stem max pool |

This proves the embed pipeline supports **N kernels in a single manifest, all
matching at once**. Browse `3_flow.mlir` to see the rewritten util.call sites
(grep `util\.call @call_saturnopu`); each one corresponds to a
`hal.executable.source @kb_saturnopu_*` carrying the linked `.o`.

**What's still unmatched:** 21 `linalg.generic` (BatchNorm rsqrt + scale +
shift + ReLU activations, residual adds) and 3 `linalg.fill` (zero
initializers not consumed by an outlined kernel). Each can be added with
the same recipe — one `abi/<op>_workgroup.c`, one `match/<op>.match.mlir`,
one manifest entry. See `manifest.json` for the existing patterns.

To grow the coverage:

1. Add a kernel + match.mlir for each remaining op family. The existing
   pattern — `abi/<op>_workgroup.c` + `match/<op>.match.mlir` + a
   `manifest.json` entry — is the same for every kernel.
2. Re-run `./refresh_phase_dumps.sh dronet` and re-read `COVERAGE.txt`.
3. The same kernels can be exercised standalone on Spike via
   `tests/granularity/test_rvv_kernels_on_spike.py` to validate
   correctness independently of the embed flow.

## Refresh

```bash
benchmarks/SaturnOPU/kernels/phase_dumps/refresh_phase_dumps.sh         # all three
benchmarks/SaturnOPU/kernels/phase_dumps/refresh_phase_dumps.sh dronet  # only dronet
```

Run from any directory; the script resolves paths against the repo root.

## Compile commands the script wraps

For dronet specifically:

```bash
./merlin compile models/dronet/dronet.mlir \
  --target spacemit_x60 --hw RVV \
  --kernels-dir benchmarks/SaturnOPU/kernels \
  --dump-phases \
  --output-dir build/saturnopu_phase_dumps/dronet_partial \
  --iree-compile-arg='--iree-opt-data-tiling=false'
```

`--iree-opt-data-tiling=false` keeps `linalg.matmul` as the plain named op so
our match pattern applies. With data tiling enabled, the matmuls get wrapped
in `iree_encoding.set_encoding`/`unset_encoding` (visible by re-running with
the flag removed) — extending the manifest to match those is the next step.
