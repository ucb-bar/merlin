# Moving non-matmul work onto Gemmini: reformulations, exactness and declarations

Analysis and proposal, 2026-10-16. Nothing described here is implemented unless a sentence says so.

This note is for engineers building a Gemmini implementation of the workloads in
[QUANTIZATION.md](QUANTIZATION.md). It asks which non-matmul work can move onto Gemmini, and at what
numerical price. Each technique gets a derivation, an exactness class, a command sketch, a cost and an
owner. Section 10 proposes declarations. "§4.2" and similar refer to
[model_check.md](../../../docs/reference/model_check.md). Numbers come from numpy experiments (Appendix A).

## 1. Three classes

| Class | Meaning | Does the INT8 reference change? | Judged by |
|---|---|---|---|
| **R**, recipe reformulation | The recipe defines the model as the reformulated function, so the INT8 reference computes it | Yes: new recipe digest, new capture | Reference admission against FP (§4.5); kernels then bit-exact (§4.2) |
| **E**, exact compiler rewrite | A rewrite that is bit-identical to the reference under stated, checkable conditions | No | Bit-exact (§4.2); the conditions are a proof obligation |
| **A**, approximation | A different function from the reference, with a stated error bound | No | Ratio rule (§4.3) on declared kernels, plus the model gates (§4.4) |

Any A becomes an R when the recipe adopts the approximate function before calibration. Only the place
of judgement moves. "Exact in real arithmetic" is not class E, because the policy compares bits (§4).

## 2. Target facts

The target is Chipyard's `GemminiRocketConfig` (wrapped by `FireSimGemminiRocketConfig`) with the default
Gemmini config, at Gemmini commit `8c3f9923` (Chipyard 1.14.0). Scala files are in `src/main/scala/gemmini/`;
`gemmini.h` and `gemmini_params.h` are in `software/gemmini-rocc-tests/include/`. The FireSim build's
Scala differs only by one disabled debug assertion; its `gemmini_params.h` is identical.

| Fact | Detail | Source |
|---|---|---|
| Array | 16×16 int8 × int8, weight- and output-stationary dataflows, 20-bit PE carriers | `Configs.scala:23-38`; Phase 1 RTL facts `arrays`, `compute_datapaths` |
| Partial sums | One 16-deep pass sums at most 16·128·128 = 2^18 < 2^19, so the 20-bit carrier cannot wrap | `Configs.scala:29`; software spec `internal_arithmetic` |
| Accumulator | int32, 1024 rows of 16 (64 KiB); accumulation is a wrapping 32-bit add | `gemmini_params.h:12`; `AccumulatorMem.scala:70` |
| Bias | An int32 seed (int8 with `low_D`) is loaded into the accumulator before the matmul; the readout has no bias operand | `gemmini.h:551-562`; contract `stage_routes`, site `accumulator_seed` |
| i8 readout | Activation, then scale, then saturate (below) | `Configs.scala:117-153`; `AccumulatorScale.scala:118-138`; `Arithmetic.scala:122-128` |
| Scale granularity | One f32 `acc_scale` and one activation per `CONFIG_ST`; the `CONFIG_EX` pair applies when the mesh reads the accumulator | `StoreController.scala:236-238`; `ExecuteController.scala:468-476,551-555` |
| Load scaling | int8 loads compute `clamp_i8(rne(f32(x)·s))`; int32 accumulator loads are unscaled | `Configs.scala:75-114`; `gemmini_params.h:71-74` |
| Accumulate on load | Two int8 loads can meet in accumulator rows, one overwriting and one accumulating (Gemmini's residual add) | `gemmini.h:3132-3173` (`resadd_cpu`, `sp_tiled_resadd`); contract `operand_sum` |
| Readout max pool | Available on the i8 readout | `Configs.scala:52`; `StoreController.scala:73` |
| ReLU order | RTL: before the scale. Spike's model and the C reference: after scaling and clamping. They differ when `s < 0` | `AccumulatorScale.scala:118-138`; `software/libgemmini/gemmini.cc:262-263`; `gemmini.h:878-907` |
| Activations | `has_nonlinear_activations = true`, but only ReLU is wired: `has_normalizations` defaults to false and the default config leaves it | `Configs.scala:53`; `GemminiConfigs.scala:92`; `AccumulatorScale.scala:61-62,127-129` |

The i8 readout, step by step:

```
a  = accumulator value (int32: seed + products)
a' = relu(a) if the activation is RELU, else a      # ReLU comes BEFORE the scale
f  = f32(a')            # int32 -> f32, round to nearest even; exact only for |a'| <= 2^24
p  = f32(f * s)         # s = acc_scale (f32); product rounded to nearest even
n  = rne_to_int32(p)    # saturates at the int32 bounds
q  = clamp(n, -128, 127)
```

Merlin models this path as `Tensor.requant_acc_scale` (`src/merlin/runtime/tensor.py:158-184`). The
target contract records that model as RTL-certified bit-exact
([target_contract.yaml](../target/contracts/target_contract.yaml), `readout_semantics`). The Phase 1
release payload carries the same readout section. Two consequences matter below. First, for `s < 0`
the RTL gives `s·relu(a) ≤ 0` but Spike gives `relu(s·a) ≥ 0`, so a needed sign belongs elsewhere.
Second, a per-row or per-column scale needs an int32 readout and a host epilogue, or one `CONFIG_ST`
per readout command.

## 3. Requantization epilogue fused into the readout

**Statement.** The reference epilogue ([QUANTIZATION.md](QUANTIZATION.md) §2), then the device's:

```
y_ref = clamp(rne(relu?(f32(f32(f32(f32(acc)·s_x)·s_w) + b)) · r_y), qlo, 127),   r_y = f32(1/s_y)
q     = clamp(rne(f32(relu?(acc + D)) · M), qlo, 127)                            # device
```

`relu?` is optional, and `qlo` is 0 with ReLU, −128 without. The device has one integer seed `D` per
channel and one f32 scale `M`.

**Class E: when is it bit-exact?** Both are monotone non-decreasing step functions of the integer `acc`.
They agree on an interval iff their 255 transition points coincide (127 with ReLU). Let `S_q` be the
first accumulator value with `y_ref ≥ q`, and `T_q` the first `t` with `clamp(rne(f32(t)·M)) ≥ q`.
Exactness needs one integer `D = T_q − S_q` for every `q`. `group_exact_float.prove` checks exactly
this, through `requantization.synthesize_bias`
(`src/merlin/xdsl_dialects/lowering/group_exact_float.py:71-131`,
`src/merlin/llvmlower/requantization.py:253-339`). Preconditions (`group_exact_float.py:27-68,84-112`):
stages are binary32, ordered cast < scale < bias < ReLU < quantize. The output is signed i8 with zero
point 0. Scales are positive, weights stored i8, the bias F32. The domain is `[−B, B]`, with
`B = 128 · max_j Σ_k |W_kj| < 2^31`. It covers every int8 input, so the proof needs no sample data.

**How often does it succeed?** One synthetic layer: K = 768, 64 channels, int8 weights of std 30.
The scales are `s_x = 0.02`, `s_w = 0.0021` and `s_y = 0.031`.

| Bias | Channels proved |
|---|---|
| zero | 64/64 |
| f32 bias on the `s_x·s_w` grid | 33/64 |
| free f32 bias (std 0.05) | 0/64 (0–2/64 with ReLU or K = 3072) |

A real float bias is almost never on the accumulator grid. Class E therefore applies mainly to
bias-free layers. TinyLlama's linears are bias-free, but their outputs feed float operations anyway.

**Class R: the recipe defines the single-scale epilogue.** It states the epilogue as the device computes
it: `D = rne(b / s_A)` with `s_A = f32(s_x·s_w)`, and `M = f32(s_A · r_y)`. The device is then exact by
construction. Two things change against the two-multiply form. The bias moves to the accumulator grid,
with error at most `s_A/2`. The scale product is rounded once. Measured on Gaussian accumulators over
±40 output steps, 1M values per row:

| s_A/s_y | Outputs differing, free bias | Bias already on the grid | Prediction s_A/(4·s_y) |
|---|---|---|---|
| 1e-5 to 1e-4 | 8.0e-6 | 1e-6 | 7.9e-6 |
| 1e-4 to 1e-3 | 7.2e-5 | 7e-6 | 7.9e-5 |
| 1e-3 to 1e-2 | 9.4e-4 | 1e-6 | 7.9e-4 |
| 1e-2 to 1e-1 | 9.6e-3 | 0 | 7.9e-3 |

Every difference is 1 LSB, and the bias rounding explains them. A uniform error in `±s_A/2` crosses a
rounding boundary with probability `(s_A/4)/s_y`.

**Max pool comes free.** A quantizer with a positive scale is monotone, so `Q(max v) = max Q(v)`
exactly. With the epilogue on the device, the readout max pool is class E on either side of the quantize.

**Commands and cost.**

```
CONFIG_ST(act=RELU, acc_scale=M)                 once per layer
MVIN bias row -> accumulator, stride 0           (repeating_bias)
PRELOAD/COMPUTE with the accumulate bit; MVOUT i8, at most 64 columns per command
```

There are no extra MACs, and readout traffic falls from 4 bytes to 1 per element. The host epilogue
pass disappears: int32 read, two multiplies, add, ReLU, quantize, int8 write.

**Owner.** R: the recipe and the capture's integerizer. E: Merlin's compute grouping, which already runs
the proof. The device compiler uses an i8 readout only on proved or R-defined groups. ResNet50 gains
most: 33 of its 53 convs feed bias → ReLU → quantize, and 20 feed residual adds (§7). Transformer
linears feed float or bf16 operations, so an int8 readout rarely helps there.

## 4. Norm folding: LayerNorm or RMSNorm into the next linear

**Identity.** With `x̂ = (x − μ)/σ` per token:

```
W·(γ ⊙ x̂ + β) + b  =  (W·diag(γ))·x̂ + (W·β + b)
```

**Step 1: fold γ and β (class R).** The identity is exact in reals, but it changes what is quantized.
The activation becomes `x̂` instead of `γ ⊙ x̂ + β`, and the weight `W·diag(γ)` instead of `W`. γ's
per-channel spread moves into the weights, as in SmoothQuant. This is a recipe choice, like PT2E's
BatchNorm folding, and its accuracy must be measured.

**Step 2: cancel 1/σ under per-token dynamic quantization.** Let `Q_t(v) = rne(127 · v / absmax(v))`
per row. For any `c > 0`, `Q_t(c·v) = Q_t(v)` in exact arithmetic. Hence:

```
q_t  = Q_t(x̂_t) = Q_t(x_t − μ_t)                     LayerNorm
q_t  = Q_t(x_t / rms_t) = Q_t(x_t)                    RMSNorm: the codes of the raw input
y_tj = s_t · s_Wj · acc_tj + (W·β + b)_j,             s_t = absmax(x_t − μ_t) / (127 · σ_t)
```

σ (or rms) moves into the per-row output scale. The host still needs μ, σ and `absmax(x − μ)` per
row. One read-only pass yields `Σx`, `Σx²`, `max x` and `min x`; `absmax(x − μ)` is
`max(max x − μ, μ − min x)`. (Use Welford's update when |μ| ≫ σ.) A second pass writes the codes. The
normalize pass and the absmax pass over its result disappear. Per-row scales force an int32 readout,
so the output epilogue was a host pass already.

**Floating point.** The test computes in f32 as a host would, on 20 × 1024 × 768 heavy-tailed rows.
Of 15.7M codes, 3 differ (1.9e-7), each by 1 LSB, for both norms: near-ties flip in f32. So the
reference must define the codes this way (R); otherwise the rewrite is a tiny A.

**Where it does not fold.**

- **Per-tensor static scales.** In `q = rne(x̂ / s)`, the per-row factor 1/σ_t changes the codes.
  Quantizing `x − μ` statically needs a range that covers the widest token, so narrow tokens lose
  resolution. The readout cannot apply 1/σ_t either. No fold.
- **Per-16-token block scales.** 1/σ_t varies inside a block and does not cancel. Folding becomes the
  R choice "block scale on `x − μ`, then 1/σ_t per row". Take a log-normal per-token scale (σ 0.6).
  The SQNR of `x̂` is then 27.2 dB per token, 24.9 dB with block-16 on `x̂`, 13.2 dB with block-16 on
  `x − μ`. No fold.

**Owner.** The recipe defines steps 1 and 2; Merlin's host code implements them. The matmul is unchanged.

## 5. Softmax row sums through a ones column

**Statement.** `P·[V | 1] = [P·V | P·1]`. Let `u_ts = rne(127 · exp(z_ts − max_s z_ts))`, so each
row's maximum is exactly 127. The device returns `Σ_s u_ts·v_s` and `Σ_s u_ts` as exact int32 sums.
The sum is at most `127·S = 130048` for S = 1024, and at most 2032 per 16-deep pass. The host computes
`o_t = s_V · (Σ u v) / (Σ u)`.

**It is a different function.** Under `activation_contraction` ([QUANTIZATION.md](QUANTIZATION.md) §5),
the reference quantizes `p = e/Σe` with one static scale: `o_ref = s_P · s_V · Σ rne(p/s_P)·v`.
Pinning the row maximum at 127 is a per-row P scale. Dividing by `Σ u` differs again from dividing by
`Σ e`. So the ones column is class R if the recipe defines the denominator as `Σ u`, and A otherwise.

Relative RMS error of `P·V` against float; V stays float to isolate P (256 rows, scores N(0, τ²)):

| Keys S | τ | Static s_P = 1/127 | Σu denominator | Σe denominator |
|---|---|---|---|---|
| 113 | 1 | 0.178 | 0.014 | 0.014 |
| 113 | 3 | 0.026 | 0.024 | 0.012 |
| 1024 | 1 | 0.739 | 0.025 | 0.025 |
| 1024 | 3 | 0.057 | 0.076 | 0.021 |

Both per-row variants beat the static scale on flat rows. On peaky rows, dividing by `Σ u` moves the
mass lost to rounding onto the surviving keys. There, `Σ e` was more accurate.

**Commands and cost.** Append a column of int8 ones to V, the weight slot, and read out int32. With
head dimension 64, 65 columns take 5 column tiles instead of 4: +25 % PV MACs. A separate `P·1` matmul
also costs a 16-wide tile and a second pass over P. The host's `exp` and quantize pass touches every
element anyway, so summing `u` or `e` there costs one add per element.

**Recommendation.** Do not use the ones column here. Choose the denominator in the recipe and sum in
the existing host pass. The column pays only when the device makes the numerators, which needs the
disabled softmax unit (§8).

## 6. GELU as a sum of ReLU hinges

**Statement.** `GELU(x) ≈ g(x) = d + Σ_{k=1..K} a_k · relu(x − b_k)`, with `Σ a_k = 1`. The slope for
large x is then 1. A linear term `c·x` is a hinge at the domain's lower edge, so it needs no extra form.

**Device mapping (M1).**

1. Widen fc1 K times: `W1' = [W1 | … | W1]`. Copy k's seed is `D_kj = rne((b1_j − b_k) / s_acc)`,
   with `s_acc = s_in · s_W1`. Its rounding error, `s_acc/2`, is negligible.
2. Read copy k out with ReLU and `acc_scale_k = |a_k| · s_acc / s_h`. That takes one `CONFIG_ST`, or
   one loop command, per copy. All copies share one activation scale `s_h`.
3. Widen fc2's input K times. Copy k uses `sign(a_k) · W2`, exact because weights lie in [−127, 127].
   A negative `acc_scale` would work on the RTL but not on Spike (§2).
4. Add `d · W2 · 1` to fc2's bias, offline. fc2's reduction over `K·F` inputs then sums the copies:
   `y_i = s_h · s_W2 · Σ_k Σ_j sign(a_k) · W2_ij · q_kj`.

**Fit and error.** Knots come from Nelder–Mead, slopes from weighted least squares on [−6, 6]. **The
input distribution is assumed** (N(0, 1) or N(0, 2)); SigLIP fc1 statistics were not measured. The
baseline is GELU in float, then one int8 quantization at `s_g = 6/127`. The hinge scale is
`s_h = max_k |a_k|·(6 − b_k)/127`. RMS errors in units of `s_g`, N(0, 1) fit:

| K | Max abs error of g on [−6, 6], absolute | s_h/s_g | Rounding RMS | Total RMS | Baseline RMS |
|---|---|---|---|---|---|
| 4 | 0.044 | 0.62 | 0.24 | 0.43 | 0.31 |
| 6 | 0.013 | 0.36 | 0.18 | 0.21 | 0.31 |
| 8 | 0.012 | 0.29 | 0.17 | 0.19 | 0.31 |
| 12 | 0.006 | 0.20 | 0.15 | 0.15 | 0.31 |

N(0, 2) weighting gives totals of 0.42, 0.21, 0.23 and 0.15, against a 0.29 baseline. The fits are
local optima; reruns move K = 8 between 0.19 and 0.24 (its fit is in Appendix A). Each copy spans part
of the range, so `s_h < s_g`, and K int8 values carry more information than one. Roundings act only on
active copies: `|e| ≤ K·s_h/2`, with RMS about `s_h·sqrt(n_active/12)`.

**Class and cost.** A against the GELU reference; R if the recipe adopts g before calibration. The
cost is K times the fc1 and fc2 MACs. Take D = 768 and T = 1024 from the patch embedding in
[QUANTIZATION.md](QUANTIZATION.md) §4, and assume F = 3072. Then fc1 + fc2 is 4.83 GMAC per layer.
The extra is (K − 1)·4.83 GMAC, or 6·(K − 1) cycles per GELU element at the 256 MAC/cycle peak. Three
host passes disappear: fc1's epilogue, GELU, and fc2's input quantize. fc1-to-fc2 traffic grows K times.

**A hypothesis, not a fact.** The hand-written SmolVLA run reportedly spent about 0.2 % of its cycles
in Gemmini compute. Trading host passes for MACs may therefore pay. Host cycles per element and mesh
utilization decide it; measure both.

**Cheaper variants, not verified on RTL.**

- **M2.** Run fc1 once and read its accumulator rows K times. Between readouts, add the constant
  `rne((b_{k−1} − b_k)/s_acc)` with a stride-0 accumulate-on-load. This removes the K-fold fc1 MACs,
  provided a readout leaves the accumulator intact.
- **M3.** Sum the K hinge codes in the accumulator by accumulate-on-load at scale 1.0, which is exact.
  Read out once at `s_g`, so fc2 runs once. The extra rounding costs accuracy: total RMS is 0.34–0.38
  `s_g` for K = 6–12, above the 0.31 baseline.

**Alternative: a lookup table.** Read fc1 out as int8 at a scale `s_1` (§3). Map each code to fc2's
input code with a 256-entry table. That is one cheap host pass and no extra MACs. It adds a rounding
of fc1's output, so it is class R. Measure it against M1 before choosing.

## 7. Residual add in the accumulator

**Statement.** `y = Q_y(s_A·acc + b + s_r·r)`, with `s_A = s_x·s_w`. Seed the producing matmul's
accumulator with `D = rne(r·s_r/s_A) + rne(b/s_A)`. Read out with `acc_scale = s_A/s_y`.

**Exactness.** Against a single-rounding reference, the seed is exact iff `m = s_r/s_A` is an integer,
so that `D = m·r`. Otherwise `|D·s_A − s_r·r| ≤ s_A/2`. The code flips only within `s_A/2` of a
rounding boundary, with probability `s_A/(4·s_y)` and size 1 LSB. Measured on 2M elements, the ratio
jittered ±10 % (an exact integer reciprocal makes values hit exact ties and doubles the rate):

| s_A/s_y | Flips measured | Predicted |
|---|---|---|
| 1e-4 | 2.3e-5 | 2.5e-5 |
| 1e-3 | 2.3e-4 | 2.5e-4 |
| 1e-2 | 2.6e-3 | 2.5e-3 |
| 1e-1 | 2.6e-2 | 2.5e-2 |

**Which reference?** ResNet50's r53b reference quantizes the conv output at `s_c` first, then adds
(`operand_sum`). Against that two-rounding reference, the preloaded form differs on about 25 % of
outputs, with `s_c = s_y`. The fused block is therefore class R: the recipe must define conv, residual
and ReLU with one rounding. The seed rounding is then class E (integer m) or a small A.

A recipe makes m an integer by quantizing the skip tensor at `s_r := m·s_A`, `m = rne(s_r,obs / s_A)`.
That changes the scale by at most `1/(2m)`, which is small because `s_A ≪ s_r`. The host still writes
`m·r` as int32: accumulator loads are unscaled, and int8 load scaling saturates at ±127.

- **ResNet (int8 residual operands).** Add, ReLU and quantize move into the conv's readout. The host
  residual and conv epilogue passes disappear; a seed pass (1 byte read, 4 written) is added.
- **Transformers (float residual stream).** The result returns to the host's float stream anyway.
  Seeding `rne(r/s_A)` saves one add in an existing pass but adds a pass and 4 bytes of DMA per element.
  `|r|/s_A` can approach 2^31 on outlier channels, and `f32(acc + D)` rounds above 2^24. Not recommended.

## 8. No path on this configuration

- **SiLU-gated MLP.** `down(SiLU(gate·x) ⊙ (up·x))` multiplies two run-time tensors element-wise. The
  mesh multiplies only inside a matmul. A diagonal-weight emulation costs 16 MACs per product, plus a
  weight preload per 16 elements per token. SiLU is not in the readout either.
- **RoPE.** `x' = x ⊙ cos θ_t + rot(x) ⊙ sin θ_t`. `rot` is a signed permutation P, so `P·W_q` is
  exact in int8. One widened matmul then yields `q` and `rot(q)`, at twice the q/k MACs. The cos/sin
  factors vary per row and per column, so no readout scale can apply them. They stay on the host.
- **I-BERT units.** LAYERNORM, IGELU and SOFTMAX exist (`Activation.scala:5-12`,
  `AccumulatorScale.scala:376-422`). With `has_normalizations` false (`GemminiConfigs.scala:92`), they
  are compiled out. See `AccumulatorScale.scala:61-62` (assertion), `Scratchpad.scala:620` (pass-through
  normalizer) and `GemminiConfigs.scala:502-505` (no `HAS_NORMALIZATIONS` in the header).
  Enabling them needs a new bitstream, RTL facts and contract. They are class A against float.

## 9. Summary

| Technique | Class | Condition | Device cost | Host passes | Owner | Recommendation |
|---|---|---|---|---|---|---|
| Requantization in the readout | E or R | E: one seed and one scale match every transition (bias-free layers). R: recipe defines integer bias and one scale; s_A/(4·s_y) of codes change | none; readout 4 → 1 byte | −1 | recipe (R); Merlin proof (E) | R for CNNs |
| Norm folding | R | per-token dynamic scales; f32 near-ties at 1.9e-7 | none | −2 | recipe; Merlin host code | yes with per-token scales; no with per-tensor or block scales |
| Softmax ones column | R or A | recipe defines the Σu denominator | +25 % PV MACs | 0 | recipe | no; sum in the existing exp pass |
| GELU hinge sum | A (R if the recipe adopts g) | fitted K, knots, slopes; error table in §6 | K× fc1 and fc2 MACs | −3 | device compiler (A); recipe (R) | measure against the lookup table |
| Residual in the accumulator | R, then E or A | one rounding defined; integer m for E; else flips at s_A/(4·s_y) | none | ResNet −2/+1; transformers +1 | recipe; Merlin | ResNet yes; transformers no |
| SiLU gate, RoPE, I-BERT units | none | no path, or disabled in this build | — | — | — | host |

## 10. PROPOSAL: declaring and judging the three classes

### 10.1 What exists today

| Artifact | What it declares | Source |
|---|---|---|
| Recipe `quant_recipe_v1` | `families`; a digest over everything but `why`; opt-in families such as `activation_contraction` change it | `src/merlin/targetgen/quant_recipe.py:51-65,156-159,336-376` |
| Recipe additions | `software_admission` and `software_numerical_engine`, added before the digest | `src/merlin/targetgen/quantization_spec.py:340-362` |
| Software spec `merlin.software_spec.v1` | [software-spec.yaml](../target/software-spec.yaml); validated in code, with no file in `merlin/schemas/`; unknown top-level keys are ignored | `src/merlin/targetgen/software_spec.py:169-314` |
| Numerical contracts | only `operand_sum_exhaustive_i8_v1`, a witness with `bound_lsb` and `max_error_lsb` | `software_spec.py:21,214-218,522-551` |
| Spec identity | the file's sha256, carried by Phase 0 evidence and the release corpus manifest | `software_spec.py:346-354` |
| Phase 2 exactness contract | `mode: exact` or `bounded` with `max_abs_lsb`; `forms: []` | [exactness.yaml](../phase2/exactness.yaml) |
| model_check | submission rows `{id, opcodes, description, reason}`, judged by §4.3; any declaration fails the model gate as "not wired" | `src/merlin/model_check/approximations.py:15-50`; `commands.py:575-579` |

Nothing links a declared approximation to a reviewed statement by the target. No approximation kind
states an error bound.

### 10.2 Class R: declared in the recipe

Add a `reformulations` list to `quant_recipe_v1`, inside the digest. Each row names a generic kind:

```json
"reformulations": [
  {"kind": "readout_single_scale_epilogue", "bias": "accumulator_grid", "rounding": "half_even"},
  {"kind": "norm_affine_fold", "norms": ["layer_norm", "rms_norm"], "activation_granularity": "token"},
  {"kind": "fused_residual_single_rounding", "skip_scale": "integer_multiple_of_accumulator_unit"}
]
```

Capture and preparation apply each kind, so the INT8 reference is the reformulated program. An
epilogue then reads `seed add (i32) → f32 cast → one f32 scale → rne → clamp`. The digest keeps
captures apart. Reference admission (§4.5) judges the result once against FP, and kernels stay
bit-exact (§4.2). The Phase 1 compiler sees an ordinary program. A kind is admissible only where the
software spec says the readout can hold it, which `software_admission` already carries.

### 10.3 Class E: no per-use declaration, a complete readout statement, a proof record

E needs no opt-in, because the output is bit-identical. Two additions make it checkable. First, the
software spec states the readout arithmetic completely, so a proof is about declared semantics.
Proposed additions to `numerical_semantics.readout`, beside its five existing keys:

```yaml
  readout:
    acc_scale_arithmetic: {to_float: i32_to_f32_rne, multiply: f32_rne, to_integer: rne_saturate_i32}
    activation_stage: before_scale             # the functional model applies it after the scale
    acc_scale_sign: positive                   # the two orders differ for a negative scale
    acc_scale_granularity: per_readout_command
    bias_route: accumulator_seed_i32
```

Second, a fused readout carries a proof record, `numerical_contract: readout_transition_proof_v1`,
named like the existing witness. It holds the accumulator interval, transitions checked, integer seeds,
scale, and the sha256 of the source scales and bias; `group_exact_float.prove` returns these already.
The grader judges E kernels bit-exact (§4.2), as now, and may re-run the proof from record and weights.

### 10.4 Class A: a reviewed catalogue in the software spec, opted into by the submission

The software spec lists admissible kinds with their math and bounds. A submission opts in per opcode.
Keys are generic, and only the values come from the target. A new section is safer than rows under
`operations`, whose dict-valued `numerical_contract` passes validation unchecked. The GELU hinge entry:

```yaml
approximations:
  status: reviewed
  kinds:
    relu_hinge_sum_v1:
      status: reviewed
      replaces: {families: [elementwise_map], ops: [gelu, gelu_tanh]}
      composed_with: [contraction]
      math: "g(x) = d + sum_{k=1..K} a_k * relu(x - b_k), with sum_k a_k = 1"
      realization:
        producer: "contraction widened K times; copy k seeded with rne((b - b_k) / s_acc)"
        readout: "relu, then acc_scale_k = |a_k| * s_acc / s_h; one positive scale per readout command"
        consumer: "consumer contraction input widened K times; copy k uses sign(a_k) * W"
        constant: "d * W * 1 added to the consumer's bias"
      parameters:
        K: {min: 2, max: 16}
        per_call_site: [domain, knots, slopes, constant]     # recorded by the submission
      error_bound:
        function: "max |g - f| over the domain, recomputed from the recorded fit"
        rounding: "K half-even roundings of step s_h: |e| <= K * s_h / 2"
        saturation: "inputs beyond the domain saturate, as with one int8 quantization"
      judged_by: ratio_rule                    # model_check §4.3, then the model gates §4.4
      evidence: {status: reviewed, basis: "numerical_semantics.readout: relu, then one positive f32 scale"}
```

The submission manifest references the kind:

```yaml
numerics:
  approximations:
    - id: vision_mlp_hinges
      kind: relu_hinge_sum_v1                      # proposed key: a reviewed kind of the bound spec
      opcodes: [MLP_FC1_HINGE, MLP_FC2_HINGE]
      parameters: {K: 8, fit: fits/gelu_k8.json}   # proposed key
      description: "GELU as 8 ReLU hinges"
      reason: "no GELU on the readout; removes three host passes per element"
```

The grader checks that `kind` is reviewed in the software spec whose sha256 the bundle records. It
checks `parameters` against the kind's ranges and recomputes the function bound from the fit. It judges
the listed opcodes by §4.3 and the model by §4.4, and reports the bound beside the measured ratio.

Required changes: a fail-closed validator for the section; `kind` and `parameters` in
`approximations.py`; the spec's sha256 in the bundle; the model replay that `commands.py` lacks; a
contract version bump. The Phase 2 exactness contract can reuse the kind ids for `bounded` forms.

### 10.5 What each party sees

| | R | E | A |
|---|---|---|---|
| Declared in | recipe `reformulations`, inside the digest | nothing per use; readout semantics in the software spec | software spec `approximations.kinds`, plus the submission manifest |
| INT8 reference | changes (new capture) | unchanged | unchanged |
| Phase 1 compiler | an ordinary program | may fuse only with a proof record | may emit a kind only for opted-in opcodes |
| Grader | admission §4.5, then bit-exact §4.2 | bit-exact §4.2; may re-check the proof | reviewed kind, parameters in range, ratio rule §4.3, model gates §4.4 |

## Appendix A: how the numbers were produced

numpy only, with f32 wherever a host computes in f32 and float64 elsewhere. Seeds are fixed.

| Section | Inputs and settings |
|---|---|
| §3, proof | `requantization.synthesize_bias` from this repository; weights N(0, 30²) rounded and clipped to ±127; f32 bias N(0, 0.05²) |
| §3, flip rates | accumulators N(0, (40·s_y/s_A)²); `s_y` log-uniform in [0.01, 0.1]; bias N(0, (20·s_y)²); 10 draws per row, half with ReLU |
| §4 | Student-t rows (5 degrees of freedom) times a per-token `exp(N(0, 0.6²))`, plus a per-token mean N(0, 0.5²); 4 channels scaled by 10–40; `s = max(absmax/127, 1e-5)`, `q = rne(v·f32(1/s))`, eps 1e-6 |
| §5 | scores N(0, τ²); V N(0, 1); head dimension 64; 256 rows per setting |
| §6 | GELU-tanh; weighted least squares on 6001 points of [−6, 6]; knots start at `linspace(−3, 3, K)`; 2M samples clipped to [−6, 6] |
| §6, K = 8 fit | knots [−2.73, −2.07, −0.82, −0.38, −0.01, 0.37, 0.80, 2.32]; slopes [−0.050, −0.062, 0.194, 0.269, 0.289, 0.270, 0.199, −0.109]; d = −0.004 |
| §7 | `s_y = 0.05`; `s_r = s_y·U(0.5, 2)`; uniform int8 `r`; accumulators N(0, (30·s_y/s_A)²); float64 |
