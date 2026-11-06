# TinyLlama TorchAO source fidelity after the first mean

The bounded two-layer TinyLlama source capture has 15 native signed i8 × i8 →
i32 contractions. Its first observed numerical divergence from eager TorchAO
was the K=2048 float32 mean in layer 0 RMSNorm. The optional, source-pinned
AVX2 reduction pass reproduces that captured mean for all eight rows. A full
host run with the pass has **not** completed, so the first divergence *after*
that correction is not yet known.

This probe uses saved same-input arrays from the managed
`full-tinyllama-opu-gate` task. The bounded source capture receipt is
`75634410bbfda3a89b7280586cd0cc49614acfd2ad5c17b9ff1582e2773683a3`.
The saved Merlin attention arrays in `.tmp/l2_merlin_attention_probe.npz`
have SHA256 `aec92d348f42f459c6ffa8c51a73579fc21d0502c62f5c3724d962daf9725b4c`.
Torch was `2.16.0.dev20261104+cpu`, source commit
`7f24a19c0967db04f2d5ba44c1e56eca489932ec`, AVX2 dispatch, one CPU thread.

For each operation, Torch received the **same saved Merlin operands**:

| Layer-0 attention operation | Saved operands → saved result | Elements | Bit mismatches | Maximum absolute error | Relative L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Q × K BMM | `v40`, `v43` → `v45` | 2,048 | 0 | 0 | 0 |
| Masked-logit softmax | `v50` → `v61` | 2,048 | 499 | `1.1920929e-7` | `4.4792373e-8` |
| P × V BMM | `v64`, `v67` → `v69` | 16,384 | 6,096 | `7.4505806e-9` | `4.1698321e-8` |

The saved `v64` is bitwise the reshape of `v61`, so the P × V comparison
also isolates its BMM arithmetic with the same already-rounded probabilities.
The Q × K result is exactly equal at this probe point; the subsequent softmax
and P × V each have small same-input float differences. These counts do **not**
prove either is the first mismatch after the mean pass or that either makes the
complete model fail. The earlier RMSNorm difference already changed the inputs
to this attention block. No quantized operands, golden, tolerance, or compiler
code were changed for this probe.

Next, run the complete bounded source host gate with
`MERLIN_PASSES=torchao-cpu-avx2-mean` and
`MERLIN_TORCHAO_CPU_SOURCE_COMMIT=7f24a19c0967db04f2d5ba44c1e56eca489932ec`
against the unchanged eager TorchAO golden. Record every output and the first
same-input mismatching operation. The prepared exact OPU/twin Spike replay must
also wait for its 55 GiB memory and 175 GiB scratch guards; that result is a
separate target-arithmetic gate, not TorchAO framework parity. Full 22-layer
model and FireSim claims remain open.
