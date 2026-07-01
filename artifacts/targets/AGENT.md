# AGENT.md — artifacts/targets

## Purpose

Generated **target packages** (codegen products): per-target directories of schedules/knobs/dialects
minted by the target tools (`merlin-rvv-mine`, `merlin-rvv-autotune`, `merlin-targetgen`, the fork/beam
machinery in `merlin.rvvgen`). This is the codegen-package home under the three-root convention — it
replaces the retired top-level `generated_targets/` (a transition symlink `generated_targets ->
artifacts/targets` keeps pre-migration references resolving; do not commit it).

## Layout

```
artifacts/targets/<target>/<package_id>/
  rvv/hand_v0/            reference baseline (schedule.mlir + knobs.yaml + manifest.yaml + baseline_runs/)
  rvv/hand_v0_int8/       int8 reference baseline
  rvv/impr_tuned_*/       promoted champion(s)
  rvv/impr_auto_*, impr_rvv_*, rvv_tuned_*   generated forks (regenerable)
  gemmini/…, muon/…, saturn_vec/…            per-target packages / OOT repos
```

`<package_id>` carries provenance: `<type>_<target>_v<N>_<ts>` plus a lineage `manifest.yaml`
(parent / version / depth / lever). Full buildable OOT repos live under `build/generated/`, not here;
mining/autotune *analysis* lives under `artifacts/kernel-mining/<target>/`.

## What is tracked vs generated (git)

- **Tracked** (via `.gitignore` negations): the hand-authored reference baselines and promoted
  champions only — `rvv/hand_v0`, `rvv/hand_v0_int8`, `rvv/impr_tuned_*`. These are the fixed points
  the tools fork from and tests load.
- **Ignored / regenerable**: every other fork, all `mlir_oot/build/` trees, `__pycache__`, fork READMEs.
- Target packages are **tool-generated** — regenerate with `merlin-rvv-mine` / `merlin-rvv-autotune` /
  `merlin-targetgen`; do not hand-commit overfit forks.

## Invariants

- Producers/loaders resolve this root via `merlin.common.paths.repo_root() / "artifacts" / "targets"`
  (or the `artifacts/targets/<target>` fork-root default), never a hardcoded `generated_targets`.
- Adding a new tracked baseline/champion requires an explicit `.gitignore` negation (mirror the existing
  `rvv/hand_v0` lines) — otherwise it stays ignored by design.
