# CudaTile Pipeline: linalg.generic → GPU cubin

This document describes the full compilation flow from MLIR `linalg.generic`
matmul operations to executable GPU cubins via the NVIDIA cuda_tile dialect.

## Pipeline Overview

```
linalg.generic (matmul)
    │  iree-opt --merlin-linalg-to-cuda-tile-text
    ▼
cuda_tile dialect text (.mlir file on disk)
    │  cuda-tile-opt (optional: verify)
    ▼
cuda_tile dialect text (verified)
    │  cuda-tile-translate --mlir-to-cudatilebc --no-implicit-module --bytecode-version=13.1
    ▼
cuda_tile bytecode (.tilebc)
    │  tileiras --gpu-name sm_100
    ▼
GPU cubin (.cubin, ELF for NVIDIA)
    │  cuModuleLoad() at runtime
    ▼
GPU execution
```

## Tools

### `iree-opt` (with Merlin plugin)

Runs the `merlin-linalg-to-cuda-tile-text` pass. This pass:
- Pattern-matches `linalg.generic` ops that represent matmul (C += A * B)
- Emits textual cuda_tile-dialect IR to a file on disk
- Does NOT link against cuda-tile headers — the integration boundary is purely textual

**Invocation:**
```bash
iree-opt \
  --pass-pipeline="builtin.module(merlin-linalg-to-cuda-tile-text{output-path=out.mlir,tile-m=64,tile-n=64,tile-k=32})" \
  input.mlir
```

**Pass options:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `output-path` | string | `""` (stdout) | File path for generated cuda_tile IR |
| `tile-m` | int64 | 64 | Tile size for M dimension |
| `tile-n` | int64 | 64 | Tile size for N dimension |
| `tile-k` | int64 | 32 | Tile size for K dimension |

### `cuda-tile-opt`

MLIR opt tool for the cuda_tile dialect. Used to parse and verify generated IR.

```bash
cuda-tile-opt output.mlir -o /dev/null  # verify only
```

### `cuda-tile-translate`

Translates cuda_tile MLIR to bytecode (`.tilebc`).

```bash
cuda-tile-translate \
  --mlir-to-cudatilebc \
  --no-implicit-module \
  --bytecode-version=13.1 \
  output.mlir -o output.tilebc
```

**Important flags:**
- `--no-implicit-module`: Required because our IR uses `cuda_tile.module` not `builtin.module`
- `--bytecode-version=13.1`: Required to match the cuda-tile bytecode version

### `tileiras`

NVIDIA's tile assembler. Converts `.tilebc` bytecode to a GPU cubin.

```bash
tileiras --gpu-name sm_100 output.tilebc -o output.cubin
```

The `--gpu-name` selects the target GPU architecture (e.g., `sm_100` for Blackwell).

## Required Packages

Install via conda from the `nvidia` channel:

```bash
conda install -c nvidia cuda-tileiras cuda-nvvm cuda-nvcc
```

| Package | Provides |
|---------|----------|
| `cuda-tileiras` | `tileiras` binary |
| `cuda-nvvm` | NVVM IR libraries (used by tileiras) |
| `cuda-nvcc` | NVIDIA CUDA compiler tools |

The `cuda-tile-opt` and `cuda-tile-translate` tools are built from source
as part of the merlin build (`third_party/cuda-tile`).

## Tile Size Constraints

- **Divisibility**: M, N, K dimensions of the matmul MUST be divisible by
  tile-m, tile-n, tile-k respectively. The pass will emit an error if not.
- **Typical values**: tile-m=64, tile-n=64, tile-k=32 (defaults)
- **Larger tiles** (128x128x64) may improve performance on large matmuls
  but require the dimensions to be correspondingly larger.

## Example: Full Pipeline (single cubin)

```bash
# 1. Generate cuda_tile text from linalg matmul
iree-opt \
  --pass-pipeline="builtin.module(merlin-linalg-to-cuda-tile-text{output-path=/tmp/matmul.mlir})" \
  test/linalg_to_cuda_tile_text.mlir

# 2. Verify the generated IR
cuda-tile-opt /tmp/matmul.mlir -o /dev/null

# 3. Translate to bytecode
cuda-tile-translate \
  --mlir-to-cudatilebc --no-implicit-module --bytecode-version=13.1 \
  /tmp/matmul.mlir -o /tmp/matmul.tilebc

# 4. Assemble to cubin
tileiras --gpu-name sm_100 /tmp/matmul.tilebc -o /tmp/matmul.cubin

# 5. Inspect the cubin
file /tmp/matmul.cubin  # should show "ELF 64-bit LSB ..."
```

## Fatbin (multi-arch GPU binary)

To target multiple GPU architectures, compile the same `.tilebc` with different
`--gpu-name` values and combine with `fatbinary`:

```bash
# Assemble for two architectures
tileiras --gpu-name sm_100 matmul.tilebc -o matmul_sm100.cubin
tileiras --gpu-name sm_120 matmul.tilebc -o matmul_sm120.cubin

# Combine into a fatbin
fatbinary --create=matmul.fatbin \
  "--image3=kind=elf,sm=100,file=matmul_sm100.cubin" \
  "--image3=kind=elf,sm=120,file=matmul_sm120.cubin"
```

The CUDA runtime (`cuModuleLoad`) automatically selects the best cubin from a
fatbin at load time based on the GPU it's running on.

**Supported GPU targets** (tileiras 13.1): `sm_100`, `sm_103`, `sm_110`,
`sm_120`, `sm_121` (Blackwell-era and newer only).

**Required tool**: `fatbinary` ships with `cuda-nvcc` conda package.

## Dimension Mapping: linalg → cuda_tile

```
linalg.generic                              cuda_tile
───────────────────────────────────────────────────────────────
tensor<MxK>  (A input)            →  tensor_view<MxK, strides=[K,1]>
tensor<KxN>  (B input)            →  tensor_view<KxN, strides=[N,1]>
tensor<MxN>  (C output)           →  tensor_view<MxN, strides=[N,1]>

(no tiling concept)                →  partition_view A: tile=(tileM x tileK)
                                   →  partition_view B: tile=(tileK x tileN)
                                   →  partition_view C: tile=(tileM x tileN)

reduction iterator k               →  for %k in (0 to K/tileK)
arith.mulf + arith.addf            →  mmaf %a, %b, %acc
```

## Current Limitations

- **Matmul only**: the pass recognizes exactly `C[m,n] += A[m,k] * B[k,n]`
  with `{mulf, addf, yield}` body. It does not handle:
  - Convolutions (different indexing maps / iterator counts)
  - Batch matmul (4 iterators)
  - Fused ops (e.g., matmul+relu)
  - Integer types (muli/addi)
- **Static shapes only**: dynamic dimensions are skipped
- **Row-major only**: strides assume row-major (contiguous last dim)

## Building IREE with the Merlin Plugin

```bash
cd /path/to/merlin
cmake -G Ninja \
  -B build-iree-merlin \
  -S third_party/iree \
  -DCMAKE_INSTALL_PREFIX=build-iree-merlin/install \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_CMAKE_PLUGIN_PATHS=$(pwd) \
  -DIREE_ENABLE_LLD=ON \
  -DCMAKE_CXX_FLAGS="-Wno-error=cpp" \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build build-iree-merlin --target iree-opt -- -j$(nproc)
```
