# K1 VLA compiler profile — 2026-09-08

## Result

All measurements use one SpaceMiT K1 core, three independent launches, two untimed warmups and
five timed in-process iterations. Every reported latency passed the full-output host-golden gate.

| model | Merlin generated | Merlin + XNNPACK GEMM | kernel-swap result | correctness | run range |
|---|---:|---:|---:|---:|---:|
| BitVLA | 102.96 ms | 84.32 ms | XNNPACK 1.221x faster | 0.9999946 / 0.9999927 cosine | 2.03% / 2.87% |
| OpenVLA | 264.76 ms | 36.91 ms | XNNPACK 7.174x faster | 0.99999994 / 0.99999994 cosine | 0.56% / 10.50% |
| RDT2 | 6.664 s | 312.24 ms | XNNPACK 21.342x faster | 1.0 / 1.0000001 cosine | 0.17% / 8.45% |

The hybrid is genuinely “our compiler with their kernels”: it retains Merlin's applicable graph
passes and swaps only eligible FP32 matmuls to XNNPACK's RVV GEMM. Both arms apply
`prepack_weight_layout`, `fuse_transpose_b`, and `erase_self_copy`; the native arm additionally uses
the per-model search-selected Merlin schedule. An earlier comparison omitted prepacking from the
hybrid and therefore measured runtime weight-layout work rather than an honest kernel swap; those
stale OpenVLA (434.12 ms) and RDT2 (16.512 s) values must not be used.

## Where Merlin's generated runtime spends time

| model | layout/copy | contraction | elementwise | reduction/softmax | profile status |
|---|---:|---:|---:|---:|---|
| BitVLA | 34.07 ms (32.0%) | 38.43 ms (36.1%) | 5.72 ms (5.4%) | 2.51 ms (2.4%) | trusted: 0.503% perturbation, 96.90% coverage |
| OpenVLA | 3.95 ms (1.5%) | 253.69 ms (96.1%) | 5.39 ms (2.0%) | 0.48 ms (0.2%) | trusted: 0.821% perturbation, 100.50% coverage |
| RDT2 | 11.39 ms (0.2%) | 6.690 s (99.4%) | 24.02 ms (0.4%) | 2.73 ms | trusted: 0.756% perturbation, 100.19% coverage |

The general AOT layout rewrite pretransposes 26 immutable OpenVLA weights and 23 immutable RDT2
weights, removing 17.7 MiB and 287.3 MiB of movement per inference respectively. It deliberately
retains RDT2's four transposed runtime inputs. This cuts native OpenVLA by 2.22x and RDT2 by 3.03x
versus the prior accepted rows, while full-output correctness remains gated. The new profiles make
the next issue clear: OpenVLA and RDT2 now spend 96.1% and 99.4% of their native wall in contractions.
The remaining gap is native GEMM codegen, not graph layout. On BitVLA, layout/copy remains material
and immutable fake-quantized weights still need a general constant-DAG fold before they can use the
same prepacking path.

## Scope and caveats

- These are deterministic, reduced-depth FP32 architecture captures with generated parameters,
  not pretrained task-quality benchmarks.
- A fresh standalone ExecuTorch column is intentionally absent: available older artifacts use
  different depths and/or unrecoverable weights (for example, old BitVLA used 30 LLM layers while
  this capture uses 2), so plotting them together would be invalid.
- OpenVLA's wide MLIR C-interface boundary is now forced inline by ABI width. This fixes the LLVM
  23 RISC-V register-scavenger failure at production `-O2` without a model-specific workaround.
- OpenVLA's profile artifact checks 40% of the output prefix, while its paired latency artifact
  checks the full output. RDT2 checks the full output in both artifacts.
- No FPGA or FireSim resource was used.

## Reproduction

- Figure: `python figures/gen_k1_vla_profile_comparison.py`
- Frozen figure data: `figures/k1_vla_profile_comparison_20260907.json`
- Whole-model driver: `build_tools/scripts/k1_e2e_xnnpack.py`
- Per-op driver: `build_tools/scripts/k1_op_profile.py`
