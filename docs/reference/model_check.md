---
title: "Full-model check contract (model_check v1)"
kind: reference
status: current
owner: core
last_verified: 2026-10-15
related: [contracts, lowering_pipeline]
code_refs: [src/merlin/model_check/cli.py, src/merlin/model_check/report.py, src/merlin/runtime/linalg_numpy.py, merlin/schemas/model_check_report.schema.yaml, examples/gemmini/numerics/models.yaml]
---

# Full-model check contract (model_check v1)

Contract version **1.4**. A change bumps the version and adds a changelog line at the end; nothing here
changes silently. Owner: branch `cc/model-check`. Consumer: the Phase 1 harness. Progress notes for the
consumer: `/work/merlin/tasks/model-check/TASK.md`. Read this file from another checkout with
`git -C /work/merlin/repos/merlin.git show cc/model-check:docs/reference/model_check.md`.

`merlin.model_check` builds, runs and judges complete quantized models for a device **submission**
(a package with `manifest.yaml`) on Spike, and replays single device kernels on Spike or gSIM. It is
infrastructure: it never writes device-kernel code and never edits a submission. It names no model and
no target in code; both come from data (`--models` file, target contract).

## 1. Models and programs (data: `examples/<target>/numerics/models.yaml`)

The models file declares, per model: its programs in chain order, the chain kind, the capture stage of
each program, the outputs that are judged, and the metric kind. v1 declares:

| model | programs (chain order) | chain kind | judged outputs |
|---|---|---|---|
| `resnet50` | `model` | `single` | logits [1, 1000] |
| `smolvla` | `prefix_encode`, `flow_denoise`, `action_decode` | `flow`: prefix once → flow ×10 (t = 1.0, dt = −0.1, state carried) → action_decode | actions [1, 50, 32], dims 0–5 (the checkpoint's action shape is [6]; 6–31 padding) |
| `tiny_llama` | `prefill`, `decode` | `causal`: prefill 128 tokens → decode ×K teacher-forced on the input's next K tokens (K = 32 official) | prefill last-position logits + K decode logits |

Each program is compiled once per submission. Per input, only the embedded-input object is regenerated
and relinked; nothing is re-lowered.

## 2. CLI

Every subcommand takes `--submission DIR`, optional `--models FILE` (default: the target's
`examples/<target>/numerics/models.yaml`), optional `--bundle DIR` (default
`/work/merlin/private/<target>/numerics/current`, a symlink to the frozen version) and optional
`--work DIR` (default under the repo's `out/` root). Every subcommand writes one JSON report to `--out`.

```
python -m merlin.model_check check --submission S --model M [--program P]
        --split {visible|sealed|all} [--mode {fast|full}] [--decode-steps K] [--jobs N]
        [--timeout-s T] [--differential] [--agent-view] --out REPORT.json
python -m merlin.model_check replay-kernel --submission S --model M --program P
        (--group SOURCE_REGION_ID | --all) --split {visible|sealed|all}
        [--engine {spike|gsim}] [--timeout-s T] [--agent-view] --out REPORT.json
python -m merlin.model_check census --submission S --model M [--program P] --out REPORT.json
python -m merlin.model_check binding --out RECEIPT.json
python -m merlin.model_check bundle-verify [--bundle DIR] --out REPORT.json
python -m merlin.model_check bundle-build --captures DIR --model M [--program P] [--keep {all|sample}]
        --host-package DIR --contract FILE --rtl-facts FILE [--bundle DIR] --out REPORT.json
```

`check` and `replay-kernel` also take `--host-package` (the host package whose compilers and `-march`
build the replay image), `--contract` and `--rtl-facts` (they bind the simulator engines) and
`--dram-base`/`--dram-bytes`; each defaults to what the bundle was built with (`MANIFEST.json`
`engine`). `bundle-build` prepares every program of a model, writes its catalog and census, and records
the device calls of the capture's own inputs as the input `visible/capture`; its `consistency` gate
fails when the device-rewritten program does not reproduce the INT8 reference bit for bit (a Merlin
defect, never charged to a submission).

`check` has two modes. **`fast` is the default and the inner loop; it never builds or simulates a
whole model:**

1. Device grouping is Merlin's, not the submission's, so every model's group interface modules, kernel
   ABI and INT8 boundary tensors are precomputed once into the frozen bundle (§8).
2. For each **unique** kernel (distinct interface module) the submission's package lowers it once; a
   minimal replay ELF runs it on Spike on a sample of its groups and the split's boundary inputs
   (`fast`: up to 3 groups × 2 inputs per unique kernel; `--jobs` runs kernels in parallel).
   Every output is compared bit for bit (§4.2), or by §4.3 for a declared approximate kernel.
3. Model gate: when every kernel is exact, the integrated model computes exactly the INT8 reference, so
   the model metrics are the bundle's precomputed INT8-reference metrics, reported as such. Declared
   approximate kernels are the only case that runs a model replay: the INT8 program on the host
   (`linalg_numpy`), with just those groups answered by the submission's kernel on Spike, then §4.4.
4. Census from the bundle's grouping plus each kernel's outcome (lowered, or refused by the package).

Target: minutes per model on one core per unique kernel. **`full`** additionally builds every program's
linked ELF and runs the whole chain per input on Spike: the integration check (host code, ABI, memory
layout, linking) for the official grade or a scheduled job, never the inner loop. `--differential`
(full only) adds an offload-disabled build and requires bit-identical outputs.
`--decode-steps K` below 32 marks the TinyLlama gate `partial: true`; a partial gate is never an
official pass.

- `replay-kernel`: the kernel step alone, for any group or all groups; `--engine gsim` is the RTL check
  per unique kernel, `spike` the fast functional one.
- `census`: grouping and per-kernel lowering outcome only; no execution (§6).
- `binding`: the Spike↔RTL↔FireSim receipt (§7).
- `--agent-view`: omit every per-input value of the `sealed` split, keeping only aggregate status per
  gate. Anything shown to an agent must be produced with it.

Exit codes: `0` every gate passed; `1` a gate failed (report complete); `2` usage or configuration error;
`3` infrastructure error (tool crash, engine timeout, missing bundle). Codes 2 and 3 still write a report
with `status: "error"` when `--out` is writable.

## 3. INT8 reference

The **INT8 reference** of a program is its *prepared quantized program*: the capture's `model.mlir`
after Merlin's quantization preparation (`zephyr_model._prepare_model_mlir`; captured quantization
scope respected; passes `contraction_int8`), taken **before** any device grouping or rewrite, evaluated
by `merlin.runtime.linalg_numpy.evaluate`. That interpreter reads each op's semantics from the IR and
shares no code with any lowering:

- integers wrap at their declared width; int8 × int8 contractions are summed exactly;
- `f32` arithmetic is float32; `bf16` is computed in float32 and rounded to nearest-even per op;
- float contractions sum through BLAS, so float results are an oracle up to reassociation.

The reference states the program's own epilogue order (e.g. f32 scale multiply, f32 bias add,
round-to-nearest-even quantize). A device kernel that folds an epilogue into the accelerator's readout
must reproduce that order exactly (Merlin proves this per group: `llvmlower/group_exact_float.py`) or
leave the epilogue on the host.

The capture's `integer-reference.json` (a PyTorch fake-quant interpreter) is **not** the reference.

The **FP reference** is the original float model (PyTorch eager, checkpoint weights, no quantization)
on the same inputs and saved noise, chained as in §1.

## 4. Numerics policy

`y_fp` = FP reference, `y_ref` = INT8 reference, `y` = the submission's result, on the same input.

### 4.1 Metrics
- `rel_rmse(y, y_fp) = ‖y − y_fp‖₂ / ‖y_fp‖₂`
- `worst(y, y_fp) = max|y − y_fp| / s`; `s` = the tensor's output quantization scale for int8 tensors,
  `max|y_fp|` for float tensors
- `kl(p_fp ‖ p) = Σ p_fp (log p_fp − log p)` over the vocabulary, softmax in float64
- every metric is per input; a gate uses the **worst input** of the split

### 4.2 Exact kernels (default)
Every device kernel not declared approximate (§5) must, on every frozen boundary input of the split,
produce output **bit-identical** to the INT8 reference's boundary output.

### 4.3 Ratio rule (declared approximate kernels, host float code, model outputs)
```
metric(y, y_fp) ≤ max(1.1 × metric(y_ref, y_fp), floor)     for every metric of the gate
```
Floors: int8 tensors — `worst` 0.5 (LSB), `rel_rmse` 2⁻²⁰; float tensors — both 2⁻²⁰; `kl` 1e-6 nats.

For a **kernel**, `y_fp` is its real-valued result on the same frozen (already quantized) inputs: the
group's ops in float64 with every rounding and quantize step removed, in the output's dequantized units.

### 4.4 Model gates
| model | gate (worst input of the split; all must hold) |
|---|---|
| `resnet50` | §4.3 on `rel_rmse` and `worst` of the logits; top-1 and top-5 agreement with FP ≥ the INT8 reference's |
| `tiny_llama` | §4.3 on `kl`, worst distribution; top-1 agreement with FP over all distributions ≥ the INT8 reference's |
| `smolvla` | §4.3 on `rel_rmse` and `worst`, **per action dim** (0–5), over the 50-step chunk |

### 4.5 Reference admission (bundles, not submissions)
A bundle may be frozen only if its INT8 reference is close enough to FP on the visible inputs.
Proposed (pending approval; the approved values are recorded in v1.1): ResNet50 top-1 agreement
≥ 15/16 and logits cosine ≥ 0.99; TinyLlama mean `kl` ≤ 0.05 nats and top-1 agreement ≥ 90 %;
SmolVLA per-dim `rel_rmse` ≤ 5 %. Calibration data is disjoint from every evaluation input.

### 4.6 Why ratios
Bounding error relative to a reference implementation's own error, rather than by one absolute
tolerance, follows established practice: FlashAttention's tests require its max error to be at most 2×
that of the PyTorch implementation in the same precision, both against an fp32 reference
(https://github.com/Dao-AILab/flash-attention); GPTQ fits quantization to layer-output error on a small
calibration set (https://arxiv.org/abs/2210.17323); MLPerf Inference sets accuracy targets relative to
the FP32 reference. The 1.1× factor is our engineering choice, not a value validated by those works.

### 4.7 Where the policy applies: every phase, every engine
The numerics policy is not a Phase 1 gate only. It is the definition of a correct result wherever a
result is produced:

- **Engines.** A kernel's output is judged by §4.2/§4.3 whether it came from Spike (`spike`) or from
  the RTL-equivalent simulator (`gsim`), through the same judging code; only the engine that produced
  the bytes differs. A Spike pass is not evidence for gSIM or the reverse: each engine's result is
  judged on its own.
- **Phase 1** (functional compiler): `check` is the correctness gate; capsule exactness stays as is.
- **Phase 2** (performance): a cycle count, speedup or any other performance number counts only when
  the SAME build (same submission tree digest, same ELF digest for whole-model runs) carries a passing
  numerics report. An optimization that changes any number must be declared under §5 and stay inside
  §4.3; an undeclared change is a correctness failure, not a performance result. Performance reports
  embed the numerics report's digest.

## 5. Declaring approximate kernels (submission `manifest.yaml`)

```yaml
numerics:
  approximations:
    - id: attn_split_planes          # unique, [a-z0-9_]+
      opcodes: [ATTENTION_QK]        # interface opcodes it applies to
      description: "f32 operands split into 3 int8 planes, 9 pair products"
      reason: "no native float path on the accelerator"
```
Kernels whose interface opcode is listed are judged by §4.3; all others by §4.2. A missing `numerics`
key means no approximations; unknown keys are an error.

## 6. Offload census

Denominator: every contraction in the **original captured** program (`linalg.matmul`,
`linalg.batch_matmul`, `linalg.conv_*`, and `linalg.generic` whose body multiplies and reduces), any
element type, attention included. Each is classified once:

- `device` — inside a routed group (`group`, `symbol`, `source_region_id`)
- `host:<reason>` — `not_quantized_by_recipe`, `dtype`, `shape_unsupported`, `policy_refused`,
  `compiler_refused`, `approximate_declined`

Counts and MAC-weighted fractions per program. The census is a measurement; it never passes or fails.

## 7. Engines and binding

- `spike`: functional ISA model with the accelerator extension. Numbers only; cycles are labelled
  `functional`. `--isa` and `-m` come from the ELF's declared board, never Spike defaults.
- `gsim`: RTL-equivalent native simulator; used by `replay-kernel --engine gsim`.
- `binding` receipt: sha256 of Spike's accelerator params header, of the params header from the
  elaboration behind the selected FIRRTL, of the spike binary and of the extension library; the ISA
  string; Spike `-m`; the FireSim target's DRAM size. A mismatch, or a program footprint that does not
  fit, fails the receipt.

## 8. Inputs and bundle layout

Inputs (committed, small, reproducible): `examples/<target>/numerics/`
```
README.md   MANIFEST.json   models.yaml
inputs/<model>/visible/<id>/...      # 8 per model
calibration/<model>.json             # calibration item ids + sha256, disjoint from inputs
```
Sealed inputs (8 per model, from different source items) are produced by the same generator but stored
privately at `/work/merlin/private/<target>/numerics/inputs/<model>/sealed/<id>/...`; a public example
never holds a holdout. The public `MANIFEST.json` commits to them by sha256 only, so the sealed set is
verifiable when it is released.
Frozen bundle (generated, private): `/work/merlin/private/<target>/numerics/<version>/`
```
MANIFEST.json                        # sha256 of every file; generator/merlin/model2mlir commits;
                                     # capture hashes; policy constants
<model>/<split>/<id>/fp.npz          # FP outputs per program and chain step
<model>/<split>/<id>/int8.npz        # INT8-reference outputs per program and chain step
<model>/<split>/<id>/boundaries/<program>/<step>/<key>.npz
                                     # key = <source_region_id>@<normalized_source_sha256[:12]>
                                     # arrays: in0..inN (device-call layout), out, fp_out
<model>/kernels/<program>/<key>/interface.mlir   # the group's interface module (package input)
<model>/kernels/<program>/<key>/abi.json         # symbol, call buffers, argument access, prepack
<model>/kernels/unique.json                      # interface-module sha256 -> groups sharing it
<model>/<split>/<id>/metrics_int8.json           # INT8-reference model metrics vs FP (§4.4)
```

**Sealed data is never granted to an agent sandbox.** `/work/merlin/private/**` must appear in no
grant; the harness tests its bwrap arguments for it.

## 9. Report

Schema: `merlin/schemas/model_check_report.schema.yaml` (`merlin.model_check.report.v1`). Top level:
`schema`, `command`, `status` (`pass|fail|error`), `tool`, `submission`, `model`, `split`, `bundle`,
`binding`, `gates` (name → `{status, partial, detail}`), `programs[]` (build, census, kernels[],
runs[]), `model_metrics`, `failures[]`, `timing`.

## Changelog
- 1.0 (2026-10-15): initial contract.
- 1.1 (2026-10-15): `check --mode fast` (default) judges kernels by replay and takes model metrics
  from the reference when every kernel is exact; whole-model Spike runs move to `--mode full`.
- 1.2 (2026-10-15): §4.7: the policy judges Spike and gSIM results alike and applies to Phase 2 —
  a performance number counts only with a passing numerics report for the same build.
- 1.3 (2026-10-15): `bundle-build`; `check`/`replay-kernel` take `--host-package`, `--contract`,
  `--rtl-facts`, `--dram-base`, `--dram-bytes`, defaulting to the bundle's. Not yet wired: `--mode full`,
  `--engine gsim`, the model replay for declared approximations, `binding`, `bundle-verify`.
- 1.4 (2026-10-15): SmolVLA is judged on action dims 0–5: `smolvla_base`'s action feature shape is [6], so
  dims 6–13 carry no trained prediction (their FP values are ~0 and a relative metric there is undefined).
