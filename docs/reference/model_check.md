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

Contract version **1.7**. A change bumps the version and adds a changelog line at the end; nothing here
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
python -m merlin.model_check binding [--contract C] [--rtl-facts F] [--host-package H] [--bundle B]
        [--board NAME] [--spec FILE] [--image ELF]... [--dram-base A] [--dram-bytes N] --out RECEIPT.json
python -m merlin.model_check kernels-export --bundle B --model M [--program P] [--split visible]
        --out-dir DIR --out REPORT.json
python -m merlin.model_check variants-build --model M [--program P] --seed S [--bundle B] --out REPORT.json
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
- `--split sealed|all` on `check` and `replay-kernel` also replays every sealed shape variant of each
  program (§8.1), all of them even in `fast` mode, judged by §4.2; the `build` and `kernel_exact` gates
  include them.
- `census`: grouping and per-kernel lowering outcome only; no execution (§6).
- `binding`: the Spike↔RTL↔FireSim receipt (§7). Missing inputs come from the `--bundle` manifest, then
  `MERLIN_TARGET_CONTRACT`/`MERLIN_RTL_FACTS`; the spec defaults to `binding.yaml` next to the models file.
- `kernels-export`: per model, every kernel's interface module and call ABI plus an index of the forms
  (`DIR/index.md`, `DIR/<model>/{index.md,kernels.json}`,
  `DIR/<model>/kernels/<program>/<symbol>/{interface.mlir,abi.json}`). It reads only catalogs, sidecars
  and censuses, carries no tensor data and no paths, and accepts only the visible split: it is what a
  harness stages into an agent sandbox.
- `variants-build`: seeds and stores the sealed shape variants (§8.1); operator only.
- Cache: `check` and `replay-kernel` reuse kernel objects (keyed by package content, interface sha256,
  compile flags, tool identities, target contract and facts, and Merlin's build sources) and replay
  consoles (keyed by image sources, operand bytes, linked objects, toolchain and simulator) under
  `build_dir()/model_check/cache`. `--no-cache` disables both; **an official grade passes `--no-cache`**.
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
on the same inputs and saved noise, chained as in §1. It runs with the numerics a sealed capture has:
`ATEN_CPU_CAPABILITY=default` (ATen's portable kernels) on one thread (`OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`), recorded per model as `fp_numerics_env`. Vectorized ATen kernels round bf16
differently: SmolVLA's final flow state moves by ~1% between two valid hosts otherwise.

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
**Approved (1.7): ResNet50 top-1 agreement ≥ 15/16 and the logits cosine's mean over the inputs ≥ 0.99**,
the worst image reported (one image's cosine is noisy). Proposed (pending approval): TinyLlama mean `kl`
≤ 0.05 nats and top-1 agreement ≥ 90 %. **Approved (1.6): SmolVLA per action dim, `rel_rmse` ≤ 2 × the
float noise of that dim**, worst input reported. Float noise = `rel_rmse` between the FP reference (§3)
and the same float model run with vectorized ATen kernels, on the same inputs. A bf16 model's own float
noise is several percent (SmolVLA: up to 6.8 % on the worst input), so a fixed 5 % bar sat at that floor.
Calibration data is disjoint from every evaluation input.

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
- `binding` receipt (schema `merlin.model_check.binding.v1`): sha256 of the spike binary and the
  extension library against the contract's pins; Spike's accelerator params header against the one
  generated by the elaboration behind the selected FIRRTL; mesh, scratchpad, accumulator and scale types
  of that header against the RTL facts and the contract; the DTS ISA against Spike's `--isa`, the host
  package's `-march` and each image's recorded ISA; the board's DRAM against Spike `-m` and the FireSim
  target's memory (read from the config sources); each image's footprint. Every check is listed by name;
  a mismatch or an `unknown` fails the receipt. Which file plays which role, with structured citations
  the receipt re-reads, is data: `examples/<target>/numerics/binding.yaml`.
- **Cited partial extensions.** An extension Spike enables but the hardware implements only in part (e.g.
  counters without the `time` CSR) passes only while its citations hold, and every checked image
  (`--image`, plus a replay probe image the receipt builds) is disassembled and must access none of the
  missing CSRs, read by name from the disassembly.
- Reports of `check`/`replay-kernel` copy the bundle's receipt as `binding {receipt_sha256, status,
  receipt}`.

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
MANIFEST.json                        # generator/merlin/model2mlir commits, capture hashes, engine inputs
                                     # (host package, contract, RTL facts), binding {receipt, sha256, status},
                                     # per model sealed_variants {rule, programs{p: {variants, sha256}}}
binding.json                         # the binding receipt (§7)
<model>/programs/<program>/catalog.json          # stage, prepared programs, routing, interfaces by symbol
<model>/programs/<program>/ref/                  # the prepared INT8 program (§3)
<model>/programs/<program>/dev/device_signatures.json
                                     # sidecar: signatures, routed groups, call buffers by symbol
<model>/programs/<program>/census.json           # §6, per contraction of the original capture
<model>/blobs/                       # content-addressed arrays (a weight shared by every call is stored once)
<model>/<split>/<id>/<program>/calls.json        # the program's device calls (first step): symbol,
                                                 # ordinal, roles, operand and expected blob keys
<model>/<split>/<id>/fp.npz          # FP judged outputs (§3)
<model>/<split>/<id>/int8.npz        # INT8-reference judged outputs
<model>/<split>/<id>/metrics_int8.json           # INT8-reference model metrics vs FP (§4.4)
<model>/reference_quality.json       # per-input reference metrics and fp_numerics_env
<model>/sealed_variants/{blobs/, <program>/variants.json, <program>/calls.json}   # §8.1
```
Replays of sealed variants build in `<bundle>.sealed_variants-work/`, beside the private bundle, never
in the shared build directory.

**Sealed data is never granted to an agent sandbox.** `/work/merlin/private/**` must appear in no
grant; the harness tests its bwrap arguments for it.

### 8.1 Sealed shape variants (rule v1)
For every unique device kernel of every program, the sealed split adds one seeded variant in each of four
classes; a class a kernel cannot reach within the caps is recorded as skipped. A variant keeps the
kernel's form (op, epilogue stages and order, convolution kernel size, stride, padding and dilation,
stationary operand, operand/accumulator/output dtypes, host-call roles); only sizes change. Bounds come
from the target's RTL facts: E = array edge, S_op and S_acc = operand-store and accumulator-store
capacity in elements.

- `sub_tile`: every non-unit extent below E (a convolution: channel counts and output pixel count; its
  output stays at least 2×2 when the kernel's is).
- `aspect`: each extent moves by a non-zero offset smaller than E (a spatial extent down by at most half),
  then all scale down together until the work caps hold.
- `accumulator_spill`: the result holds 1.25–2× S_acc elements per batch.
- `operand_spill`: the live operands hold 1.25–2× S_op elements.

In every class a non-unit extent is never a multiple of E and never the kernel's own; a unit extent stays
1; a convolution's input is non-square and larger than its padding, and its output pixel count is not a
multiple of E; a batched form gets a different batch count in [2, 2B+1]. Each variant does at most 2²⁴
multiply-accumulates over at most 2²² operand elements. A declared `acc_scale` moves by 0.5–3 % (random
sign), rounded to 12 significant bits. Operands are uniform over the operand dtype's full range with both
extremes present, ranges capped so the exact accumulator stays within half its range (narrowed so a
typical result is about a quarter of the output range when the output is narrower than the accumulator).
Expected results are the interface program evaluated exactly. Variants and the seed are never released.

## 9. Report

Schema: `merlin/schemas/model_check_report.schema.yaml` (`merlin.model_check.report.v1`). Top level:
`schema`, `command`, `status` (`pass|fail|error`), `exit_code`, `tool`, `submission`, `model`, `split`,
`agent_view`, `bundle`, `binding`, `gates` (name → `{status, partial, detail}`), `programs[]`,
`model_metrics`, `failures[]`, `timing`.

- `programs[]`: `program`; `build {seconds, built, refused, cache {enabled, hits, misses, stored}}`
  (`full` adds `status: linked`); `census {denominator, device, refused, device_mac_fraction}` (the
  bundle's census with the submission's refused kernels moved off the device); `kernels[]`; `images[]`
  (`cases`, `engine`, `seconds`, `cached`, `elf`); `full` adds `runs[]` (`engine`, `status`, `split`).
  A sealed variant row has `split: sealed`, `source: sealed_variants`, kernels with `variant_of` and
  `variant_class`; the agent view reduces it to `summary {status: count}`.
- `kernels[]`: `symbol`, `form`, `status` (`pass|fail|refused`), `cases`, `exact`, `mismatches`,
  `source_region_id` (the first of `source_region_ids`), `interface_sha256`, `shape_sha256` (two kernels
  with both equal are the same kernel), and on failure `diagnosis` (`mismatch`: `results[]` with the
  first wrong element `{index, expected, got, position}`, counts, rows/cols hit, a 32-cell map, a `hint`
  when elements were never written; `refused`: the package's `refusal`). A diagnosis from a non-visible
  call sits under `kernels[].sealed`, which the agent view drops.
- `model_metrics`: `source` (`int8_reference` when every kernel is exact), `inputs`, `per_input[]`
  (`split`, `id`, the §4.1 metrics), `worst` (per metric over `per_input`: the minimum of an agreement,
  the maximum of an error) and `mean` (per metric; a per-input agreement of 0 or 1 averages to the
  split's agreement rate, which the agreement gates of §4.4 compare).

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
- 1.5 (2026-10-15): `binding` wired (§7, cited partial extensions, CSR scan); `kernels-export`;
  `variants-build` and sealed shape variants (§8.1) replayed by `--split sealed|all`; kernel-object and
  console cache with `--no-cache`; report fields for kernel identity, census, diagnosis and model metrics
  (§9); FP reference pinned to capture-time numerics (§3); §8 states the bundle layout as built. Not yet
  wired: `--mode full`, `--engine gsim`, the model replay for declared approximations, `bundle-verify`.
- 1.6 (2026-10-15): §4.5: SmolVLA's admission bar is 2 × its float noise per action dim (approved).
- 1.7 (2026-10-15): §4.5: ResNet50's cosine bar is on the mean over inputs (approved).
