---
title: "Design: drive the core to zero target-specific literals"
kind: design
status: draft
owner: targetgen
last_verified: 2026-07-29
related: [target_resolution, capability_manifests, capsule_bench]
code_refs:
  - merlin/python/merlin/targetgen/capability_manifests.py
  - merlin/python/merlin/targetgen/target_registry.py
  - merlin/python/merlin/targetgen/rtl/facts.py
  - merlin/python/merlin/targetgen/synthesize/llvm_extension_plan.py
  - merlin/python/merlin/compile_cli.py
  - merlin/python/merlin/compare/spec.py
  - merlin/contract/mlir_oot_backend_contract.yaml
  - build_tools/scripts/check_structure.py
---

# Drive the core to zero target-specific literals

**Principle.** The core is target-agnostic; targets are user-supplied descriptors or live in
published per-target repos discovered via `MERLIN_TARGET_PATH`. A target NAME may legitimately appear
only in (a) examples in schemas/docs, (b) the user's descriptor (`target_experiment.yaml`,
`input_bundles/`), and (c) one central pointer file (`merlin/targets/publish.yaml`). Everything else
must read the name/fact/lever from the resolved contract / `facts.json` / the target registry.

The agnostic machinery already exists — `target_registry.resolve()` + discovery,
`capability_manifests.derive_manifest()`, `families.family_profile()`, the config-driven
`capsule_runner` + `RunnerConfig`, `generate_prompt.py` derived slots, `xdsl_dialects.targets.factory`,
and the `runtime/backends/base.py` registry. Each task below routes a residue of hardcoded literals
through those seams.

## Task ledger

Status legend: `[ ]` open · `[~]` in progress · `[x]` done (committed on this branch).

- [ ] **T1 — Retire hardcoded capability manifests into `derive_manifest`.**
  `capability_manifests.py` (+ `test_capability_manifests.py`, `pipeline.py:126`). Convert
  `rvv/mx_gemmini/radiance` builders to `derive_manifest(descriptor, facts, residual=…)` as
  `atlas_manifest` already does; move residuals to the target package. Accept: no per-target manifest
  dict in core; `MANIFESTS` iterates discovered targets; existing tests pass.
- [ ] **T2 — Remove `target="gemmini"` defaults and `_DEFAULT_BACKEND`.** RTL/check/capsule stack +
  `target_registry.py:55-60`. Make `target` a required arg; add `runtime.default_backend` to reference
  contracts; read it like the external path (`target_registry.py:157-159`). Accept: no `="gemmini"`
  default in core signatures; `backend_for` reads the contract.
- [ ] **T3 — Route the capsule_bench harness through the descriptor (24 files).** Replace hardcoded
  gemmini paths / `GemminiRocketConfig` with `_common.TARGET`/descriptor. NOTE: several files have
  concurrent uncommitted edits — coordinate before touching.
- [ ] **T4 — Derive the synthesize plans from family/contract, not `if name==`.**
  `synthesize/{llvm_extension_plan,runtime_adapter_plan,dialect_plan,target_contract,zephyr_plan}.py`.
- [ ] **T5 — `{target}`-parameterize the generic ABI contracts.**
  `merlin/contract/{mlir_oot_backend_contract,oracle_runner_contract}.yaml` + consumers.
- [ ] **T6 — De-hardcode build_tools gates.**
  `check_structure.py`, `check_standalone_install.py`, `check_repro_env.py` iterate the registry.
- [ ] **T7 — Derive kernel-mining features from the framework contract.**
  `kernels/features/{roles,dispatch,loops}.py`, `framework_contracts/`, `cca_contract.py`, `markers.py`,
  `cli_index.py`, `ingest/{autocomp,exo}.py`.
- [ ] **T8 — CLI + comparison seams enumerate discovered targets.**
  `compile_cli.py:243-284` (`--target` choices = `all_targets()`), `compare/spec.py:27`.
- [ ] **T9 — `{target}`-template the targetgen_evals method/skill prompts.**
  `methods/{v0,v2,v3,v5,v6}/prompt.md`, `skills/*/AGENT.md`.
- [ ] **T10 — Miscellaneous derivations.** `evidence/report.CONCEPT_KEYWORDS`,
  `generate/target_repo.py:44`, `rtl/mlc_bridge.py:1080`.
- [ ] **T11 — Reference-target eviction (E-reference).** Move the in-tree reference backends / eval
  suites / dialects to a published reference-target package; keep one neutral `toy_npu` example. Fix the
  D-FLAG tests to parametrize over ≥2 targets.
- [ ] **T12 — Dead-code / drift sweep.** `backends/__init__.py` stale docstring, `write_all` atlas
  drift, `chia_repeatability.py:132` inconsistency; unused-symbol pass over moved modules.

Ordered by blast radius. T1/T2/T4/T5/T6/T8/T10/T12 are software-verifiable; T3/T7/T9/T11 touch
harness/hardware/reference paths and need coordination or hardware to certify.
