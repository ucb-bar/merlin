# K1 INT8 performance status — 2026-09-07

We do not yet beat ExecuTorch across the four full-model workloads. The strongest result is
TinyLlama: Merlin is now only 1.047x slower at one core (1.235 s vs 1.180 s) and 1.078x slower at
eight cores (399.8 ms vs 370.9 ms), with the complete 256,000-element output matching on every
launch. Restoring 32-lane RVV accumulation produced a controlled 1.27x/1.18x improvement.

LSTMNetVIT improved from 433.0 to a 130.9 ms launch median at one core and from 86.9 to 49.6 ms at
eight cores by using direct grouped convolution, rejecting small OpenMP regions, prequantizing all
17 eligible constant contraction weights ahead of time, and folding 39 residual broadcast-to-add/mul
chains. The last fold is a controlled 1.033x/1.027x win; the separate raw-iteration medians are
131.1/49.5 ms. Every full output remained byte-identical. LSTMNetVIT is still 2.52x/3.23x behind
ExecuTorch, so global layout/quantization planning and multicore scheduling remain the main work.
ResNet-50 tail-panel
packing improved one-core latency to 2.253 s, but the eight-core result remains 1.251 s, confirming a
multicore scheduling bottleneck. smolVLA's first eight-core build exposed a BF16 runtime-ABI defect;
after correcting it, both traced and uninstrumented K1 executions match the host golden exactly.
That accepted 150.3 s run is a mixed W8-dequant/BF16 diagnostic, not W8A8, and remains about 6.50x
slower than ExecuTorch qd8. The newer v2 capture is genuinely W8A8 (140 INT8 contractions, zero
BF16 matmuls); its first session build reaches lowering and stops at 30 masked attention tails for
sequence length 113. We are implementing full-tile plus tail lowering while preserving MR4/NR8,
rather than compiling by regressing every attention kernel to MR=NR=1.

The post-all17 LSTMNetVIT profiler is not directly suitable for choosing passes. Besides 2.70% wall
perturbation, its markers prevent a normal broadcast fold and attribute 4.86 ms to work absent from
the uninstrumented binary. Intersecting the profile with the release IR found 52 add and 10 multiply
candidates at input; ordinary fusion removed 23 and the new pass explicitly folded the remaining
31+8. Its lowered object changed and the uninstrumented interleaved board A/B won at both core counts.
This release-survivor check remains mandatory for subsequent profile-guided candidates.

The older slide's bitVLA win is real but narrower: it compares FP32 matmul kernels routed inside the
same Merlin graph/runtime (148.3 ms Merlin vs 167.3 ms XNNPACK kernels), not the full standalone
ExecuTorch flow. It shows that Merlin can generate a winning kernel for a favorable shape; the
current project is closing the harder whole-model scheduling, memory, and runtime gap.

See `k1_executorch_status.pdf` for the current plots. All headline INT8 comparisons use one- and
eight-core results separately, full-output gates, and identity-matched ExecuTorch qd8 references
where available.
