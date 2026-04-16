# ASPLOS27-OPU — Binary Archive Manifest

The paper-measurement binaries are too large for git. They live at
`/scratch2/agustin/artifacts/ASPLOS27-OPU/` and are referenced here
by sha256 so a mis-copy can be detected.

| File | Size | SHA256 | Contains |
| --- | ---: | --- | --- |
| `firesim_bench_binaries.tar.zst` | 1,475,170,392 B (~1.48 GB) | `ec200eb523826d8a2ddb6e7b49cc152aa7b2ce5dfade104f90cfd85fc1ccc2b2` | 67 `bench_model_*` ELFs used in the FireSim V128-D64 sweep — one per (model × variant) combination, including `*_opu`, `*_rvv`, `*_opu_prof` profile builds, and the microbench variants (`mt_*`). |
| `compiled_vmfb.tar.zst` | 495,780,163 B (~496 MB) | `719f8685fae989838a3e1169dffc8cac1e8abaf9aee64d63d6fb8da6483cbf50` | 9 IREE-compiled model directories under `build/compiled_models/opu_bench_suite/` — each contains the `.vmfb`, `files/` (.s / .ll / .bc / .o per-dispatch), `configs/`, `phases/`, `sources/`, `binaries/`, `benchmarks/` subdirs. |

Source of truth: `/scratch2/agustin/artifacts/ASPLOS27-OPU/sha256sums.txt`.

## Restore

See `ARCHIVE.md` for the exact `tar --zstd -xf` commands.

## What's NOT here

- The compiled merlin runtime (IREE samples, HAL driver, tooling).
  Rebuild with `tools/merlin.py build --profile firesim-merlin`
  from the pinned SHAs.
- The FireSim bitstream. Build from the pinned Chipyard SHA +
  hardware recipe in `build_tools/hardware/saturn_opu_u250.yaml`.
- The host iree-compile binary. Rebuild with
  `tools/merlin.py build --profile vanilla`.
