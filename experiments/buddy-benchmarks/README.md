# Buddy-MLIR Gemmini Performance Benchmarks

Performance evaluation of [Buddy-MLIR](https://github.com/buddy-compiler/buddy-mlir)'s Gemmini dialect backend,
benchmarked against the Gemmini C reference implementation on Spike simulator.

For the full lowering pipeline and setup instructions, see [WORKFLOW.md](WORKFLOW.md).

## Performance Results

### Matmul Workloads

| Workload | Dataflow | Gemmini C cycles | Buddy cycles | Checksum Match | Speedup |
|----------|----------|------------------|--------------|----------------|---------|
| MLP2 (64x832) | WS | 2,528 | 409 | ✓ 252338 | 6.18x |
| MLP2 (64x832) | OS | 207,782 | 96,076 | ✓ 252338 | 2.16x |
| MLP1 (6-layer) | WS | 25,251 | 2,539 | ✓ 258664 | 9.95x |
| softmax matmul (31x30x66) | WS | 335 | 145 | ✓ 3860 | 2.31x |
| IGELU matmul (30x30x30) | WS | 133 | 133 | ✓ -23260 | 1.00x |

### Conv Workloads

After the conv-encoding fix ([buddy-compiler/buddy-mlir#689](https://github.com/buddy-compiler/buddy-mlir/pull/689)):

| Workload | CPU cycles | Gemmini C cycles | Buddy cycles | Checksum Match | Buddy vs Gemmini C |
|----------|-----------|------------------|--------------|----------------|---------------------|
| conv (17x17, k=3, stride=2) | 7,559,913 | 1,027 | 149 | ✓ 950 | 6.89x |
| conv_with_pool (17x17, k=3, pool=3) | 7,714,291 | 1,605 | 172 | ✓ 30827 | 9.33x |

### ResNet50 Layer Validation

| Layer | Gemmini C cycles | Buddy cycles | Checksum Match | Speedup |
|-------|------------------|--------------|----------------|---------|
| Conv1 (7x7, stride=2, pool) | 225,146 | 7,313 | ✓ 10206332 | 30.8x |

## Methodology

- **Simulator**: Spike ISA simulator with Gemmini extension (`dim=16`)
- **Cycle measurement**: `rdcycle` instruction around the accelerator call
  (between `gemmini_flush(0)` and `gemmini_fence()`)
- **Validation**: Output checksums compared between Buddy-MLIR and Gemmini C reference
- **Gemmini C reference**: `tiled_matmul_auto` / `tiled_conv_auto` from
  [gemmini-rocc-tests](https://github.com/ucb-bar/gemmini-rocc-tests)

### Important Caveats

The `rdcycle` counter measures **CPU instructions executed**, not wall-clock time or
Gemmini hardware execution time. Buddy-MLIR's compile-time loop unrolling reduces
host-side loop overhead (fewer `rdcycle` ticks for loop control), making the cycle
counts appear faster even when the underlying Gemmini hardware work is identical.

The speedup numbers reflect reduced host-side orchestration overhead, not necessarily
faster accelerator throughput.

### Buddy-MLIR Conv Encoding Fix

The conv benchmarks require the fix from [buddy-compiler/buddy-mlir#689](https://github.com/buddy-compiler/buddy-mlir/pull/689),
which corrects the `im2col` encoding for convolutions in the Gemmini lowering path.
Without this fix, conv outputs produce incorrect checksums.

## Directory Structure

```
experiments/buddy-benchmarks/
├── README.md                          # This file
├── scripts/
│   └── run_benchmark.sh               # Run all benchmarks on Spike
├── kernels/
│   ├── Makefile                        # Build all kernel benchmarks
│   ├── conv/
│   │   ├── conv-buddy.mlir            # 17x17 conv, k=3, stride=2
│   │   └── conv-buddy.c              # C harness
│   ├── conv-with-pool/
│   │   ├── conv-with-pool-buddy.mlir  # Conv + 3x3 maxpool
│   │   └── conv-with-pool-buddy.c
│   ├── mlp2/
│   │   ├── mlp2-buddy.mlir           # 2-layer MLP (WS)
│   │   ├── mlp2-buddy-os.mlir        # 2-layer MLP (OS)
│   │   └── mlp2-buddy.c
│   ├── mlp1/
│   │   ├── mlp1-buddy.mlir           # 6-layer MLP
│   │   └── mlp1-buddy.c
│   ├── softmax-matmul/
│   │   ├── softmax-matmul-buddy.mlir
│   │   └── softmax-matmul-buddy.c
│   └── igelu-matmul/
│       ├── igelu-matmul-buddy.mlir
│       └── igelu-matmul-buddy.c
├── resnet50/
│   ├── Makefile                        # Build + validate ResNet50 conv1
│   ├── conv1-buddy.mlir               # ResNet50 conv1 (7x7, stride=2, pool)
│   ├── conv1-buddy.c                  # Buddy C harness
│   ├── conv1-gemmini.c                # Gemmini C reference
│   ├── conv1-bad-buddy.mlir           # Intentional bad case (wrong stride)
│   └── conv1-bad-buddy.c
└── logs/                               # Reference Spike output logs
    ├── conv1-gemmini.log
    ├── conv1-buddy.log
    └── conv1-bad-buddy.log
```

## How to Reproduce

### Prerequisites

- RISC-V GNU toolchain (GCC cross-compiler for `riscv64-unknown-elf`)
- [Buddy-MLIR](https://github.com/buddy-compiler/buddy-mlir) built with Gemmini dialect
  (`buddy-opt`, `buddy-translate`, `buddy-llc`)
- [Spike](https://github.com/riscv-software-src/riscv-isa-sim) ISA simulator with Gemmini extension
- [gemmini-rocc-tests](https://github.com/ucb-bar/gemmini-rocc-tests) (for headers and baremetal runtime)

### Build and Run Kernel Benchmarks

```bash
cd experiments/buddy-benchmarks/kernels

# Set paths (adjust to your environment)
export RISCV=/path/to/riscv-toolchain
export BUDDY=/path/to/buddy-mlir/build/bin

# Build all benchmarks
make all

# Run all on Spike
make run-all

# Or run individual benchmarks
make run-conv
make run-mlp2
make run-mlp1
```

### Build and Run ResNet50 Validation

```bash
cd experiments/buddy-benchmarks/resnet50

# Build all (Gemmini C reference + Buddy + intentional bad case)
make all

# Run full validation suite (compares checksums automatically)
make validate
```

### Run Everything

```bash
cd experiments/buddy-benchmarks
./scripts/run_benchmark.sh
```
