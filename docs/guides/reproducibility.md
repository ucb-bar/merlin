---
title: Reproducibility & core workflows — the master guide
kind: guide
status: current
owner: targetgen
last_verified: 2026-07-22
related: [getting_started, adding_a_target, targetgen, kernel_mining, beam_search, dse, dse_guidance,
          design_pressure, model2mlir, rvv_e2e, zephyr, gemmini_experiment, integrations, llvm_integration]
code_refs: [merlin/python/merlin, build_tools/scripts]
---

# Reproducibility & core workflows — the master guide

**Start here.** This is the master reproduction guide: it maps every other guide by *what you want to
do*, then gives the exact end-to-end CLI + output location for each core workflow. All generated output
lives under the single root `out/` (`out/runs/`, `out/artifacts/`, `out/build/` — always via
`merlin.common.artifacts`, never a hand-built path). Every workflow is deterministic and re-runnable:
same inputs → same `out/…` products; run dirs + versioned products carry a `run_record.json` /
`manifest.yaml` (git sha, timestamp, version) as the source of truth. Environment/CLI setup:
[Getting started](getting_started.md). Full flat doc index: the generated hub
[docs/README.md](../README.md).

## Guide map — by intent

| I want to… | Guide(s) | Detailed workflow below |
|---|---|---|
| **compile (+build/run/verify) a workload with ONE command** | this guide | **§0** |
| set up the environment / see the CLI surface | [getting_started](getting_started.md), [CLI](../reference/cli.md) | — |
| generate a new target dialect from a contract | [targetgen](targetgen.md), [adding_a_target](adding_a_target.md) | §1 |
| improve the compiler from expert kernels (mine → beam) | [kernel_mining](kernel_mining.md), [beam_search](beam_search.md), [rvv_kernel_mining_methodology](../reference/rvv_kernel_mining_methodology.md) | §2 |
| run the autonomous whole-model beam (rediscover e2e gains) | [beam_search](beam_search.md), [CCA beam design](../design/beam_cca_architecture.md) | §5 |
| do DSE / design-pressure analysis | [dse](dse.md), [dse_guidance](dse_guidance.md), [design_pressure](design_pressure.md) | §3 |
| bring in a new model (capture bundle) | [model2mlir](model2mlir.md) | §4 |
| run a model end-to-end on the K1 board | [rvv_e2e](rvv_e2e.md) | §7 |
| run a model on Zephyr / FireSim / spike | [zephyr](zephyr.md) | §7 |
| run the Gemmini target-gen experiment (arms→cert→perf→publish) | [gemmini_experiment](gemmini_experiment.md) | §6 |
| publish a certified champion to `<target>-mlir` | [target_publishing](../design/target_publishing.md) | §6, §8 |
| understand the compiler internals | [architecture](../reference/architecture.md), [lowering_pipeline](../reference/lowering_pipeline.md), [llvm_integration](llvm_integration.md), [compilation_strategies](compilation_strategies.md) | — |
| find where code/tests/docs/output live | [repo_structure](../reference/repo_structure.md), [merlin_layout](../reference/merlin_layout.md) | — |
| compare against external frameworks (TVM/ExecuTorch/…) | [integrations](integrations.md) | §7 |

## 0. Compile a workload with one command (`merlin-compile`)

**Purpose.** The single front door over the whole compile pipeline — name a workload + target and it
handles the rest (resolve/auto-capture the bundle → lower → cross-compile → optionally run → gate vs
golden). Use this when you just want "compile model X"; drop to the detailed workflows below only when
you need to customize a stage.

```bash
# RVV — compile a whole captured model, build, run on the K1, verify vs golden (auto-captures if absent):
merlin-compile --workload bitvla --dtype int8 --target rvv --run k1        # --run none = compile only
# Gemmini — build the OOT backend package + run a capsule on spike, three-way gated:
merlin-compile --target gemmini --workload A2_single_tile_matmul --run spike
```

`--target rvv` treats the workload as a captured MODEL (`lower_model_file → build_k1_binary →
run_on_k1 → gate`); `--target gemmini` treats it as a capsule run through an OOT package
(`oot_runner` build + certify) — the accelerator runs kernels, not whole VLA models. `--run`
∈ `none|host|k1|spike|verilator` (default rvv→k1, gemmini→spike); `--no-capture` fails with the capture
command instead of auto-capturing; `--json` for machine output. Fail-closed: a missing toolchain/board/
sim reports a clear `status`, never a fake pass. Prereqs: [getting_started](getting_started.md) (the RVV
run needs the K1 board or spike; the Gemmini run needs the sim toolchain). This CLI only orchestrates
the same API the detailed workflows use — see §4 (capture), §6 (gemmini), §7 (run on hardware).

## 1. Generate a new target dialect

**Purpose.** Turn a target contract into a reviewable codegen package + generated dialect skeleton
(deterministic scaffold generation with validation gates — not "RTL → correct dialect").

```bash
merlin-targetgen build --target-name <name> \
  --source-dir merlin/targets/<name>/docs \
  --examples-dir merlin/targets/<name>/examples \
  --out out/build/generated/merlin-target-<name> \
  --emit xdsl,mlir,zephyr,llvm-plan,runtime
merlin-targetgen inspect --target out/build/generated/merlin-target-<name>
```

**Outputs.** The generated OOT repo skeleton under `out/build/generated/merlin-target-<name>/`; the
hand-authored contract that drives it lives in-tree at
`merlin/targets/<name>/contracts/target_contract.yaml`. See [Adding a target](adding_a_target.md)
and [Target generation](targetgen.md).

## 2. Improve a dialect / compiler from expert kernels

**Purpose.** Mine what expert kernels do, diff it against our compiler's output, route each
divergence to a default-off compiler feature, search feature combinations, and certify every change
against the frozen baseline (correctness first, cycles last).

```bash
merlin-rvv-mine      ...   # decode → CCA → divergences → typed actions/policies
merlin-rvv-autotune  ...   # beam over features; certify forks (spike + K1 silicon)
merlin-rvv-report    ...   # render the evidence report
```

**Outputs.** Mining artifacts, forks, and the evidence report under
`out/artifacts/kernel-mining/<target>/` (forks also under `out/artifacts/targets/<target>/`). The
full mechanics — CCA lift, the validity gate, the beam engines, and the K-ladder cos-gate — are in
the [RVV kernel-mining methodology](../reference/rvv_kernel_mining_methodology.md); the framing is in
[Kernel abstraction mining](kernel_mining.md).

## 3. DSE analysis

**Purpose.** Recover the workload contract (temporal / numerical / memory / runtime-interface),
derive hardware-independent design requirements, and emit a DSE-ready contract-analysis package —
without picking a design or claiming a speedup.

```bash
merlin-design-pressure ...                 # cut-point pressures → candidate contracts
merlin-dse             ...                  # variant cost-model comparison + exploitability
merlin-dse-guidance    --case-study ...     # workload-contract analysis package (per workload)
```

**Outputs.** Concern-first under `out/artifacts/`: `out/artifacts/design-pressure/<workload>/`,
`out/artifacts/dse/<workload>/`, `out/artifacts/dse-guidance/<workload>/`. See
[Design pressure](design_pressure.md), [Design-space exploration](dse.md), and
[DSE guidance](dse_guidance.md).

## 4. Support a new model

**Purpose.** Bring a new PyTorch model in as a framework-neutral **capture bundle** — the shared
input every baseline ingests.

```bash
build_tools/scripts/setup_model2mlir.sh                                   # m2m setup
# in the model2MLIR capture venv:
.venv/bin/python $MODEL2MLIR_DIR/workloads/capture.py <model> --formats fp32 int8
# then point Merlin at the resulting bundle (resolve() / recaptures_dir())
```

**Outputs.** The bundle lands under `out/artifacts/recaptures/<model>_<variant>_consistent/`
(`model.mlir` + `weights.safetensors` + `inputs.npz` + `golden.npy` + `extra.npz`). Quantization is
applied **in model2MLIR, not Merlin**; int8 (W8A8) is the only measured-working format today, fp8 /
int4 are a documented plan (`unavailable`). Full details — bundle layout, the honest torchAO status,
and ingestion — in [model2MLIR frontend](model2mlir.md).

## 5. Autonomous beam experiment (reproduce the e2e gains without hand-feeding)

**Purpose.** Have the beam **rediscover** the whole-model optimizations from the frozen `hand_v0`
seed — the run, the CCA extraction, and the improvements all autonomous — and compare what it found
against the manual `ours_best` and the XNNPACK expert. Architecture + rationale:
[CCA beam design](../design/beam_cca_architecture.md); mechanics: [beam search](beam_search.md).

Prereqs (board-gated; run on a QUIET board — the K1 noise floor is ≥1.9%, ≥4.3% under contention):
- the K1 reachable (`MERLIN_K1_HOST`, key), whole-model bundles under `out/artifacts/recaptures/`,
  per-dtype expert fixtures in `merlin/tests/data/cca_asm/`;
- the clean four-way baseline first, so `ours_best` (manual) and the XNNPACK expert walls exist:

```bash
# (a) clean four-way baseline — current-vs-XNNPACK + expert walls the beam compares to
MERLIN_COMPILE_TIMEOUT_S=3600 .venv/bin/python build_tools/scripts/k1_e2e_xnnpack.py \
  --model out/artifacts/recaptures/rdt2_fp32_consistent -n 3 \
  --configs baseline,ours_wholemodel_vf,ours_best,xnnpack_kernels,openblas_kernels

# (b) one beam cell, from frozen hand_v0, whole-model objective, whole-model proposer
MERLIN_COMPILE_TIMEOUT_S=3600 merlin-rvv-beam \
  --model-dir out/artifacts/recaptures/bitvla_fp32_consistent \
  --expert-objdump merlin/tests/data/cca_asm/xnnpack_f32_gemm_rvv.objdump \
  --expert-wall-ns <xnnpack whole-model wall ns> \
  --proposer wholemodel --targets k1 --width 5 --depth 2

# (c) the full autonomous matrix + auto comparison (beam-discovered vs manual vs XNNPACK)
#     the driver bounds the per-fork compile timeout to 900s by default (pathological-fork guard: a
#     fork whose schedule makes clang spin fails-closed fast instead of blocking the board); export
#     MERLIN_COMPILE_TIMEOUT_S to override.
.venv/bin/python \
  build_tools/scripts/run_autonomous_beam_experiment.py \
  --cells fp32:bitvla,fp32:openvla,fp32:rdt2,int8:bitvla,int8:openvla,int8:rdt2 --width 5 --depth 2
```

> **K1 SSH note.** The board sits on the Berkeley-IoT WiFi; the campus path filters inbound `:22` to
> that segment (ICMP + high ports pass, `:22` is dropped). The board's `ssh.socket` therefore also
> listens on **2222**, and `.env` sets `MERLIN_K1_SSH_PORT=2222` (honored by `rvvgen/k1.py` across all
> ssh/scp). If `k1.available()` is False but the board pings, it is not down — recheck the port.

**Outputs.** Beam runs under `out/runs/rvv/beam/<op>/<TS>_cca_beam_.../` (`beam_tree.yaml` = the full
per-fork record: discovered levers, real K1 speedup, `attainment_vs_expert`, gate). The experiment
summary is `out/artifacts/kernel-mining/rvv/bench/autonomous_beam_experiment.json`
(beam-discovered features + speedup + attainment vs the manual `ours_best`, per cell). Frozen
`hand_v0` is asserted byte-unchanged pre/post — the seed the beam forks from and measures against.

**Honesty invariants.** Correctness gates before any wall (fail-closed → `not_run`); a speedup is
credited only on real K1 silicon above the noise margin; an `inert` fork (emitted code == parent's)
is never promoted; whole-model int8/fp16 have no XNNPACK e2e column (harness limitation, stated).

## 6. Gemmini target-dialect-generation experiment (agentic case study)

**Purpose.** The case study for the target-gen tool: how well an agent authors a correct,
RTL-conformant Gemmini MLIR OOT backend under increasing Merlin help, in a cheat-proof sandbox — then
certify and publish it. Four arms (raw C++ → +Merlin infra → +xDSL tooling → +CIRCT checks), all graded
to 20/20 public capsules.

```bash
S=merlin/experiments/gemmini_capsule_bench_v0/scripts
.venv/bin/python $S/test_sandbox.py --arm merlin_rtlchecks   # MANDATORY pre-spend gate (21/21 GO)
.venv/bin/python $S/verify_no_cheat.py                        # static cheat-clean gate
.venv/bin/python $S/launch_ab_batch.py --tag <tag> --arms baseline,cpp_merlininfra,merlin,merlin_rtlchecks --mode sequential
.venv/bin/python merlin/experiments/gemmini_cert/run.py --simulators spike,verilator   # RTL conformance C0-C5
```
Do NOT set a tight `--round-timeout` (default 4h; a short cap is net-detrimental). The rate-limit
watchdog + `--resume` carry it across session limits. **Full detail** — arms, sandbox mechanics, cert,
perf-bench, publish, honesty invariants — in [Gemmini experiment](gemmini_experiment.md).

## 7. Run a model on real hardware / simulators + external baselines

**Purpose.** Execute a captured model end-to-end and (optionally) compare against external frameworks.

- **K1 board (RVV):** [rvv_e2e](rvv_e2e.md) — lower a bundle → SpacemiT-toolchain ELF → run on the K1.
  ⚠️ The board's SSH is on port **2222** (`MERLIN_K1_SSH_PORT`), not `:22` (campus IoT filter); a board
  that pings but hangs on `:22` is not down — recheck the port.
- **Zephyr / FireSim / spike:** [zephyr](zephyr.md) — SMP RVV-on-Saturn, spike + 2-tile FireSim.
- **External baselines (TVM / ExecuTorch / Buddy / EXO / ggml):** [integrations](integrations.md) — the
  same capture bundle run through each framework on the K1, honestly profiled.

## 8. Publish a certified champion (any target)

**Purpose.** Ship a certified codegen package into the target's `<target>-mlir` repo, branch-per-version.

```bash
git init --bare /tmp/<target>-mlir.git    # verify against a LOCAL remote first (no GitHub)
.venv/bin/python -m merlin.targetgen.publish publish --target <rvv|gemmini> \
  --remote file:///tmp/<target>-mlir.git --execute
```
Remote from `merlin/targets/publish.yaml` (`ucb-bar/rvv-mlir`, `ucb-bar/gemmini-mlir`) or
`MERLIN_PUBLISH_REMOTE_<TARGET>`. Gate: rvv = `spike_verified`/`rtl_certified`/`k1_verified`; gemmini
(mlir_oot) = `rtl_certified` or an `oot_runner.certify` pass. Baseline → `baseline` branch; each champion
→ `stable/<package_id>` + `v<ver>-<pkg>` tag; idempotent by fingerprint. **A real GitHub push (drop the
`file://`) needs an explicit human go-ahead.** Design + branch model: [target_publishing](../design/target_publishing.md).

---

For everything else, the generated hub [docs/README.md](../README.md) is the complete index (by
kind, by owner, with per-doc status and last-verified date).
