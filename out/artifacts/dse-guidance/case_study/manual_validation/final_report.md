# Recovering a structural workload contract for accelerator DSE from real model captures

**Final report (P24, conference round).** This document is the canonical, renders-anywhere summary of the
study. It draws three lines crisply and refuses to cross them:

1. **Structural contract recovery + the loop-preserving capability are the contribution** — final-quality,
   verified, numerically faithful to the model's real computation.
2. **Deployment magnitudes are reported by config-composition** (real config; exact for layer-identical
   stacks) — *decoupled* from the reduced-config captures and labelled as such on every figure.
3. **There is no performance claim** — the hardware is unknown by design, so the tool emits
   hardware-INDEPENDENT requirements (arithmetic intensity + a ridge-point regime partition), never a
   latency/speedup/throughput for any built or hypothetical chip.

The adversarial self-critique is in [`threats_to_validity.md`](threats_to_validity.md); read it alongside
this report. Publication figures are in [`figures/`](figures/).

---

## 1. What the tool recovers (the contract)

From each model's real capture (`model2MLIR` MLIR), Merlin recovers a **structural workload contract**: the
op graph and GEMM shapes, the multi-rate **loop structure** (trip count K, loop-carried state, KV cache, the
repeated region), region **roles**, operand **locality/residency**, the dtype/**numerical contract**, and a
HW/SW-boundary search space — **each fact tagged with an evidence tier** (A = IR/measured exact, B = recovered
+ recompute-checked, C = config/assumed, D = unavailable).

This is **not** a DSE optimizer and makes **no** cost/cycle/area/energy/optimality claim.

### Loop-preserving capture (the keystone)
`model2MLIR` lowers `torch.while_loop → scf.for`, so the K-step decode/denoise loop survives export instead of
being unrolled away. The repeated region, its trip count K, and the loop-carried state (latent / KV) become
first-class IR. Roles are then attributed **structurally** from the `scf.for` boundary (authoritative over the
fqn the HOP body strips), not by a fqn heuristic.

| workload | K (IR) | repeated-region ops | loop-carried operands | KV bytes (IR) |
|---|---:|---:|---|---:|
| molmoact | 8 | 1256 | 5 | 262 144 |
| openvla | 7 | 597 | 5 | 221 184 |
| pi05 | 10 | 6208 | 2 | n/a (flow latent) |
| smolvla | 10 | 5179 | 2 | n/a (flow latent) |
| groot_n1d7 | 4 | 5107 | 2 | n/a (flow latent) |
| rdt | 5 | 641 | 3 | n/a (flow latent) |
| rdt2 | 5 | 1155 | 2 | n/a (flow latent) |
| xr0 | 5 | 849 | 2 | n/a (flow latent) |
| bitvla | 7 | 742 | 5 | 79 872 |
| tiny_llama | 7 | 754 | 5 | 61 440 |
| small_llama* | 7 | 557 | 5 | 30 720 |

\* small_llama is a synthetic toy (fully-known config); it is **excluded from the analyzed capture corpus**
(its functional-weight loop wrapper lowered GEMMs to `linalg.generic` → 0 `linalg.matmul`) and shown only
where the config — not the capture — is the source. Source: `loop_aware_contract.csv`.

**Every loop wrapper is numerically exact** vs the eager unrolled loop (cos ≈ 1.0 / bit-exact; per-model
audits in `*_source_to_mlir_audit.md`), so the recovered structure provably equals the model's real
computation — value-independently (shapes, not weights, drive the contract).

---

## 2. Deployment magnitudes — by config-composition, decoupled from the captures

The captures give **structure**, not deployment **scale** (openVLA's capture is reduced to 2 layers; pi0.5 is
full-depth). We therefore report absolute magnitudes from a **deployment config-composition** —
`embed + Σ per-layer × real n_layers`, exact for layer-identical transformer stacks — and tag every magnitude
figure *deployment-composition* vs *captured-config*. (`real_config_magnitudes.csv`, figure
`deployment_magnitude.png`.)

| workload | layers | GEMM params | GEMM MACs/token | source |
|---|---:|---:|---:|---|
| molmoact | 48 | 12.28 B | 11.19 G | MolmoActLlmConfig() defaults |
| openvla | 32 | 6.74 B | 6.48 G | **= Llama-2-7B (exact anchor)** |
| pi05 | 36 | 2.82 B | 0.31 G | Pi0Config(pi05=True) |
| tiny_llama | 22 | 1.10 B | 0.97 G | **= TinyLlama-1.1B (exact anchor)** |
| smolvla | 60 | 0.58 B | 0.10 G | SmolVLA instantiated config |
| rdt2 | 14 | 0.47 B | 0.47 G | RDT2 post_train.yaml (real `nn.Linear` shapes) |
| small_llama | 2 | 0.46 M | 0.40 M | synthetic toy (negligible — visibly the outlier) |

Two **exact external anchors** cross-check the composition (openVLA = Llama-2-7B = 6.74 B; tiny_llama =
TinyLlama-1.1B = 1.10 B). **bitvla, rdt, groot, xr0** are **omitted** from deployment-magnitude composition
wherever a field would require a guess — fewer networks, **no fabricated values** (see threats T9).

---

## 3. The honest roofline — hardware-INDEPENDENT (no chip assumed)

Given an unknown target, the only honest roofline uses the part that belongs to the **workload, not a chip**:
the roofline x-axis, **arithmetic intensity** = MACs ÷ weight-bytes-moved. (`arithmetic_intensity.csv`, figure
`arithmetic_intensity_roofline.png`.)

- **Non-resident** (weights reloaded every step): AI = 1 / dtype_bytes — the floor (0.5 MAC/byte at bf16).
- **Resident** (weights loaded once, reused across the once-prefix + K-step head):
  AI = (prefix + repeated·K) / ((prefix + repeated)·dtype).
- **Residency gain** = (prefix + repeated·K)/(prefix + repeated) — *how much residency raises AI*, a
  workload-specific, fully HW-free DSE finding:

| workload | AI resident (MAC/byte) | residency gain | regime |
|---|---:|---:|---|
| molmoact | 3.69 | **7.4×** | decode-heavy → residency helps most |
| openvla | 3.38 | 6.8× | decode-heavy |
| tiny_llama | 3.14 | 6.3× | decode-heavy |
| rdt2 | 2.50 | 5.0× | mid |
| smolvla | 1.28 | 2.6× | prefix-heavy → residency helps least |
| pi05 | 1.00 | 2.0× | prefix-heavy |

- **Ridge-point regime partition** (parameterized, *not* a chip): a machine with compute:bandwidth balance B
  (MAC/byte) is compute-bound on the resident workload iff AI > B. We report AI and let **any** B be compared
  — partitioning the space of possible machines, committing to none.
- **Absolute latency / "speed-of-light seconds" is deliberately NOT emitted as a point** — it would require a
  peak FLOPs/bandwidth, i.e. a specific chip.

The verifier re-derives AI_nonres == 1/dtype and the residency-gain formula and **asserts the artifact
contains no peak/bandwidth/latency/cycle/GHz number**.

### Measured legs are sanity anchors only (never a perf product)
FireSim cycles (6 models incl. the xr0 silicon datum), W8A8 accuracy (5 models), host-dispatch counts (3) are
kept with their caveats as independent anchors. The matmul-only cost model is crude (xr0 is a 4.7× outlier);
none of these is a prediction.

---

## 4. Low-bit datapath — honest corpus-wide tiering

Per-workload tier from capture-file presence, with int8 candidates ratified by the **measured** W8A8 accuracy
gate; fp8/int4 stay `unavailable` (never assumed). (`low_bit_visibility.csv`.)

| tier | workloads | what is visible |
|---|---|---|
| **native** | bitvla | packed int2 + absmean scale + the named `quant_ext.unpack_int2` op (opt-in, provably-inert recognizer) — accuracy **measured-pass** |
| **qdq_int8** | openvla, pi05, rdt | i8 storage + per-channel scale (dequant-before-matmul). A torchao stand-in, *not* the native scheme |
| **dequant_only** | molmoact, smolvla, groot, xr0, rdt2, tiny_llama | f32 capture; low-bit abstractions blocked (honest) |

Native packed fp8/int4 for the rest needs model-specific quant exports — **scoped, not faked**.

---

## 5. Evidence-tier honesty

Across the recovered facts: **Tier A ≈ 24%** (IR/measured), **B ≈ 40%** (recovered + recompute-checked),
**C ≈ 34%** (config/assumed), **D < 2%** (unavailable). Every fact and figure carries its tier; a C is never
rounded up to A. (`cross_workload_provenance.csv`, figures `evidence_type_by_*`.)

---

## 6. Verification

- `verify_implementation.py`: **all checks pass** (630/630 at the roofline+low-bit round), re-deriving every
  pinned number independently from the artifacts.
- `pytest merlin/python/tests/test_dse_guidance.py`: green.
- `reproduce_case_study.sh`: byte-stable regeneration of the committed `case_study/`.
- Figures rendered to `figures/*.png` at ≥150 dpi, every text element ≥ 8 pt, every figure stamped with its
  evidence tier + scale source + a one-line caption; no stacked-bars-on-log; dense heatmaps trimmed to top-N.

## 7. Bottom line for a reviewer

The structural contract recovery and the loop-preserving capability are the result, and they are verified and
numerically faithful. Magnitudes are deployment-real by config-composition (exact for layer-identical stacks),
clearly separated from the reduced-config captures. Performance is intentionally absent — the roofline is
hardware-independent, and nothing is claimed for any chip. The honest residuals (random-init values; K as the
captured decode length; low-bit native only for bitvla; attention not loop-level-structured) each carry the
exact input that would close them — see [`threats_to_validity.md`](threats_to_validity.md).
