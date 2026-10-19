---
title: "Full-model check contract (model_check v1)"
kind: reference
status: current
owner: core
last_verified: 2026-10-19
related: [contracts, lowering_pipeline]
code_refs: [src/merlin/model_check/cli.py, src/merlin/model_check/report.py, src/merlin/model_check/references.py, src/merlin/runtime/linalg_numpy.py, merlin/schemas/model_check_report.schema.yaml, examples/gemmini/numerics/models.yaml]
---

# Full-model check contract (model_check v1)

Contract version **1.17**. A change bumps the version and adds a changelog line at the end; nothing here
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
        [--timeout-s T] [--differential] [--agent-view]
        [--full-comparison host-twin|quantized-reference]
        [--quantized-reference-atol A] [--quantized-reference-rtol R] --out REPORT.json
python -m merlin.model_check replay-kernel --submission S --model M --program P
        (--group SOURCE_REGION_ID | --all) --split {visible|sealed|all}
        [--engine {spike|gsim}] [--timeout-s T] [--agent-view] --out REPORT.json
python -m merlin.model_check census --submission S --model M [--program P] --out REPORT.json
python -m merlin.model_check binding [--contract C] [--rtl-facts F] [--host-package H] [--bundle B]
        [--board NAME] [--spec FILE] [--image ELF]... [--dram-base A] [--dram-bytes N] --out RECEIPT.json
python -m merlin.model_check kernels-export --bundle B --model M [--program P] [--split visible]
        --out-dir DIR --out REPORT.json
python -m merlin.model_check variants-build --model M [--program P] --seed S [--bundle B] --out REPORT.json
python -m merlin.model_check bundle-admission --model M [--bundle DIR] --out REPORT.json
python -m merlin.model_check bundle-seal [--bundle DIR] [--jobs N] --out REPORT.json
python -m merlin.model_check bundle-verify [--bundle DIR] [--jobs N] --out REPORT.json
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
layout, linking) for the official grade or a scheduled job, never the inner loop. Its integration gate
is the `differential`: each step's outputs must be bit-identical to the host twin's (the same recipe built
with offload disabled) on the same bound inputs, read from the bundle's twin cache (`bundle-twins`) or run
on a miss. A model whose `full.differential` is `cached` (SmolVLA, TinyLlama: their twins take hours per
input on Spike) runs no twin: a step without a cached one is `pending` (not run, not failed) and so is
the gate unless a step differs. Its grade is then every kernel exact and the whole chain completed, with
`host_drift` reported; the per-call check will close the gate.
`--full-comparison host-twin` is the unchanged default. Explicitly selecting
`--full-comparison quantized-reference` instead runs the complete compiled chain on Spike and compares
**every element of every declared judged output** directly with the sealed `int8.npz`. Its gate is
`quantized_reference`, alongside the required exact accelerator-kernel replay and build gates. Integer
and boolean outputs are exact; float outputs must satisfy `abs(actual-reference) <= atol +
rtol*abs(reference)` at every element, with fixed defaults **atol = 2e-6, rtol = 1e-5**, inherited from
Merlin's host-region qualifier. Both tensors must be finite; output arity, shape and dtype must match.
The explicit tolerance flags and recorded `comparison.float_policy` select the policy before execution.

Direct comparison validates the reference and model-configuration bytes against the bundle seal before
execution and rechecks them afterward. It reuses protected, sealed payload aliases by their bound names
and bytes. Full NumPy reference recomputation and CPU-only host twins are not selected. The report has
`comparison.host_twin = {selected: false, status: not_run}` and no passing `differential` gate.
`reference_consistent` is `null` because the NumPy chain was not recomputed; `reference_validation`
records the sealed file/configuration/manifest/sums identities, strict output arity and unchanged reads.
Existing FP-baseline quality data is informational, including when absent or unparseable; it cannot
pass or fail the selected comparison. A failed exact-kernel gate prevents expensive full execution.
Any missing program step, selected-output mismatch, nonfinite output or partial chain prevents an
unqualified full-model pass. `--decode-steps` retains its explicit partial behavior.

Full-mode INT8 references share a constant/weight cache across recurrent steps. Persisted reference
steps are reused only under a versioned identity of the full chain, counts, input bytes, capture and
prepared-reference closure, evaluator sources and dependencies. Each step also verifies its actual
bound inputs, output arity, shapes, dtypes and stored bytes. Unkeyed old NPZ files are ignored; an
input/source mutation during evaluation publishes no reference entries. A frozen judged-output
comparison requires the complete output list. Before building, full mode logs any missing declared
cached-twin coverage; this diagnostic never turns a pending differential into a pass.

`--decode-steps K` below 32 marks the TinyLlama gate `partial: true`; a partial gate is never an
official pass.

The Phase 1 harness uses `check --mode fast --split sealed` at the freeze, running the models in
parallel under the shared compute pool. It requires both the original kernel replays and every selected
sealed shape variant, with a complete offload census. All required whole programs must still compile
and link. The reviewed Phase 1 descriptor omits the legacy whole-model execution gate; `--mode full
--split sealed` runs separately on the frozen submission after the experiment, outside its clock.
The official report records this integration validation as `deferred_post_experiment`. Public capsule
L3 gSIM qualification and timing collection remain required; model-kernel gSIM replay is not requested.

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
- `bundle-admission` re-judges a built bundle's reference admission (§4.5) under the models file's current
  rules, from each case's stored `metrics_int8.json` and `metrics_baseline.json`; nothing is recomputed.
  It rewrites `reference_quality.json`, so a sealed bundle must be sealed again.
- `bundle-seal` freezes a bundle: `SHA256SUMS` lists a sha256 of every file, and the manifest's `seal`
  records that file's digest, the file count and the size. `bundle-verify` recomputes every digest and
  fails on any changed, missing or extra file, on a sums file that differs from the manifest's, and on a
  model whose catalogs or censuses are missing or a binding receipt whose digest differs from the
  manifest's. Its report carries the manifest's own sha256 (`bundle.verify.manifest_sha256`), which a
  release pins. A bundle changed after its seal fails until sealed again; an unsealed bundle is refused.
- Cache: `check` and `replay-kernel` reuse kernel objects (keyed by package content, interface sha256,
  compile flags, tool identities, target contract and facts, and Merlin's build sources) and replay
  consoles (keyed by image sources, operand bytes, linked objects, toolchain and simulator) under
  `build_dir()/model_check/cache`. A stored object's linked symbol is named by its compiled code and
  objects link in content order, so a package edit that leaves a kernel's code unchanged rebuilds it into
  the same bytes and its replay consoles are reused. `--no-cache` disables both; **an official grade
  passes `--no-cache`**.
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
- named and non-contraction generic floating reductions carry the original initial accumulator
  through reduction indices in source lexicographic order, rounding each body operation at its
  declared type. An f32/f64 combiner's explicit `reassoc` flag permits NumPy's reduction tree;
  f16/bf16 retain per-operation rounding because reassociation does not permit promotion;
- float contractions sum through BLAS, so float results are an oracle up to reassociation.

New reference derivations record `merlin.linalg_numpy.numeric.v2.strict_float_reductions`, the
evaluator source digest and the declared numerical policy. Before this version, recognized floating
reductions used NumPy's pairwise summation and added the initial accumulator after the fold. Their
sealed references retain their historical identity. Correcting this requires a separately derived
and sealed successor bundle; never replace their reference files or reinterpret them as v2 evidence.
The same captures, weights, inputs and existing FP references may be reused. Recompute affected
quantized chains, device-call boundaries and metrics with fresh reference scratch; unchanged legacy
cases must retain their original policy attribution. Neither new captures nor a new FP inference is
required for this interpreter correction. Floating contractions retain their separately documented
bounded BLAS policy; their results do not establish exact agreement at a later integer boundary.

The reference states the program's own epilogue order (e.g. f32 scale multiply, f32 bias add,
round-to-nearest-even quantize). A device kernel that folds an epilogue into the accelerator's readout
must reproduce that order exactly (Merlin proves this per group: `xdsl_dialects/lowering/group_exact_float.py`) or
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
A bundle may be frozen only if its INT8 reference is close enough to FP on the visible inputs. The rules
are data: `judged.admission` in the models file, one `{metric, aggregate (mean|max|min), bound}` per rule,
where the bound is `max`, `min` or `max_ratio_to_baseline`. A baseline-relative rule needs
`judged.reference: {precision, baseline}`: the FP reference is run at `precision` (e.g. `float32`) and
the float model at the `baseline` precision (e.g. `native`, the checkpoint's own) is judged against it on
the same inputs. Building the references evaluates every rule per split into `reference_quality.json`
(`admission.<split>.{status, checks[]}`); a rule with no data is `unknown`, never a pass. The bars below
restate the target's models file.
**Approved (1.7): ResNet50 top-1 agreement ≥ 15/16 and the logits cosine's mean over the inputs ≥ 0.99**,
the worst image reported (one image's cosine is noisy). **Approved (1.15): TinyLlama mean `kl` ≤ 0.075
nats** (exp(0.075) − 1 ≈ +7.8 % perplexity, measured against the FP model's own next-token distributions)
**and top-1 agreement ≥ 90 %**. Set after the launch recipe was measured (8 inputs per split: visible 0.043,
sealed 0.066). The split-to-split spread of 0.02 is as large as the gap the earlier proposal of 0.05 left.
**Approved (1.9): SmolVLA within 2 × the bf16 model's own error,
both against an fp32 execution** (FlashAttention's rule, §4.6). Statistic: per input, the worst action dim's
`rel_rmse` (dims 0–5, the 50-step chunk) against the **fp32 reference** = the same checkpoint with every
weight upcast and every op in float32 (`ATEN_CPU_CAPABILITY=default`, one thread). `E_bf16` = that
statistic for the stock bf16 model. **Since 1.13 every input is judged under 4 initial-noise draws**
(`reference.seeds`: draw 0 is the stored noise, draws 1–3 are seeded from the input's identity), so a split
of 8 inputs gives 32 chains: the 10-step flow is chaotic, and one chain's worst dim swings about 2× under
perturbations that do not change the recipe. The INT8 reference passes when the **mean** and the **90th
percentile** over the chains are each ≤ 2 × the same statistic of `E_bf16`; the maximum is reported, not
gated. Reported beside it, never gating: the same statistic against the bf16 FP reference, and 2 × the bf16
"float noise" (the disagreement of two valid bf16 kernel sets).
Why fp32: a bar must measure distance to the true function. Two bf16 implementations disagreeing only
estimates bf16's error indirectly and depends on which second implementation is chosen; the fp32 run
measures it directly. A per-dim form is not used: with 8 inputs one dim's error is too poorly estimated.
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
README.md   MANIFEST.json   models.yaml   sources.yaml
inputs/<model>/visible.npz           # 8 per model: one array per input and field, key <id>.<field>
inputs/<model>/visible.json          # {<id>: metadata}; a model without arrays has only this file
calibration/<model>.json             # calibration item ids + sha256, disjoint from inputs
```
Sealed inputs (8 per model, from different source items) are produced by the same generator but stored
privately at `/work/merlin/private/<target>/numerics/stored/<model>/sealed.{npz,json}`, in the same
format; a public example never holds a holdout. The public `MANIFEST.json` commits to them by sha256
only, so the sealed set is verifiable when it is released. `merlin.model_check.inputs.stored` reads
either split. The model-ready arrays stay private and one file per input
(`derived/<model>/<split>/<id>/`).
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
<model>/<split>/<id>/fp_baseline.npz, metrics_baseline.json
                                     # only with judged.reference.baseline: the float model at the baseline
                                     # precision, and its own error against the FP reference (§4.5)
<model>/reference_quality.json       # per-input metrics, fp_numerics_env, fp_runs, admission per split
<model>/sealed_variants/{blobs/, <program>/variants.json, <program>/calls.json}   # §8.1
```
Replays of sealed variants build in `<bundle>.sealed_variants-work/`, beside the private bundle, never
in the shared build directory.

The references (`merlin.model_check.references.build`) are built with one `chain.ChainCache` per
process: each program is parsed once, each stage's weights and lifted constants are read once (read-only
arrays), and `linalg_numpy.evaluate` does work that reads only constants once (an op is hoisted only when
no value its regions read is defined outside it). `workers=k` runs the INT8 cases in k single-threaded
processes (`merlin.model_check.chain_worker`), each claiming the next unclaimed case and binding inputs
in its own scratch (`<model>/.chain_work/workers/<k>/`), and splits the FP reference over k processes;
rows, logs and `calls.json` are written in case order, and the count is bounded by the host's available
memory. None of this changes a byte of the bundle.

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
`model_metrics`, `failures[]`, `timing`. A gate's status is `pass` or `fail`, or `pending` for a gate the
models file defers (§2); a pending gate neither passes nor fails the report.

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
- 1.8 (2026-10-15): §4.5: SmolVLA's 2 × float-noise bar is stated on the per-input worst dim (mean and max).
- 1.9 (2026-10-16): §4.5: SmolVLA's admission is measured against an fp32 execution: within 2 × the bf16
  model's own error (FlashAttention's rule); the bf16-noise comparison is reported, not gated.
- 1.10 (2026-10-16): §4.5: admission rules are data (`judged.admission`, `judged.reference`), judged
  per split when references are built; §8 adds the baseline files.
- 1.11 (2026-10-16): §8: the stored inputs of a model and split are one `<split>.npz` (key `<id>.<field>`)
  and one `<split>.json`, public and sealed alike; the inputs and model-ready arrays are unchanged.
- 1.12 (2026-10-16): `bundle-seal` and `bundle-verify` (§2): a sealed bundle carries a sha256 of every file.
- 1.13 (2026-10-16): §4.5: SmolVLA is judged over 4 noise draws per input, on the mean and 90th percentile
  (approved); admission rules take `p<NN>` aggregates and `report_only`; `reference.seeds` in the models file.
- 1.14 (2026-10-16): §2, §9: `full.differential: cached` (SmolVLA, TinyLlama; approved): full mode
  compares with cached host twins only and records the rest `pending`; a pending gate does not fail a report.
- 1.15 (2026-10-16): §4.5: TinyLlama's KL bar is 0.075 nats (approved); `bundle-admission` re-judges a built
  bundle's admission from its stored metrics.

- 1.16 (2026-10-19): explicit `--full-comparison quantized-reference` reuses sealed quantized outputs,
  requires complete compiled chains and exact accelerator-kernel replay, checks all declared judged
  outputs under a fixed recorded float policy with exact integers, and selects no CPU-only host twins
  or NumPy chain recomputation. Host-twin remains the default; FP-baseline quality is report-only in
  the new mode. Reports distinguish the selected gate and sealed-reference validation explicitly.
- 1.17 (2026-10-19): reference interpreter numeric policy v2 preserves source order and initial
  accumulators for non-contraction floating reductions without explicit reassociation. Fresh reference
  quality/case records and cache identities state the policy. Existing sealed references remain
  historical evidence and require an explicitly derived successor for the new policy.
