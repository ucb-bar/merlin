---
title: Reproducibility & core workflows
kind: guide
status: current
owner: targetgen
last_verified: 2026-07-14
related: [getting_started, adding_a_target, kernel_mining, dse, model2mlir]
code_refs: [merlin/python/merlin, build_tools/scripts]
---

# Reproducibility & core workflows

A one-screen index of the four end-to-end workflows: what each is for, the exact CLI to run, and
where its outputs land under the single generated-output root `out/` (`out/runs/`, `out/artifacts/`,
`out/build/` — always via `merlin.common.artifacts`, never a hand-built path). For the full,
auto-generated documentation index start at the hub, [docs/README.md](../README.md); for
environment setup and the CLI surface see [Getting started](getting_started.md).

Every workflow is deterministic and re-runnable: same inputs → same `out/…` products. Run
directories and versioned products carry a `run_record.json` / `manifest.yaml` (git sha, timestamp,
version) as the source of truth; the folder name is convenience.

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
the [RVV kernel-mining methodology](../rvv_kernel_mining_methodology.md); the framing is in
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

---

For everything else, the generated hub [docs/README.md](../README.md) is the complete index (by
kind, by owner, with per-doc status and last-verified date).
