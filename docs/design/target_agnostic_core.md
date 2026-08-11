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

**The machine-readable ledger is `merlin/contract/overfit_register.yaml`.** This document is the prose
narrative; the register is the enforced source of truth, checked by
`build_tools/scripts/check_overfit_register.py` on every commit. Two things it does that a prose ledger
cannot:

* it **re-measures the live tree** and fails when a module depends on a specific target without being
  declared, so new coupling has to be a decision with an owner and a removal condition rather than
  something absorbed silently;
* it distinguishes `status: triaged` (someone read the code and can say what the weld costs and what
  would remove it) from `status: untriaged` (a tool surfaced it and nobody has looked). Without that
  distinction a required-fields schema simply pressures people into inventing rationales, and a
  fabricated removal condition reads as rigour.

The task numbers below (T1–T12) track the *literal* residue. The register additionally tracks the
**coupling** residue that the name gate could not see until it learned to look at imports: 34 modules,
26 hard imports, of which 9 sat in four modules no allowlist knew about.

## Task ledger

Status legend: `[ ]` open · `[~]` in progress · `[x]` done (committed on this branch).

- [x] **T1 — Retire hardcoded capability manifests into `derive_manifest`.**
  `capability_manifests.py` (+ `test_capability_manifests.py`). DONE: the per-target `rvv/mx_gemmini/
  radiance/atlas` manifest dicts are gone; every manifest is built by `manifest_for(name)` =
  `derive_manifest({"target":name}, facts, residual)` where the residual is a hand-owned
  `out/artifacts/targets/<name>/contracts/residual.yaml` side-input (tracked via a `.gitignore`
  negation). `MANIFESTS` is discovered (module `__getattr__` over `discovered_targets()`, scanning
  those residuals); `write`/`write_all`/`write_oot_target` iterate discovery. atlas reproduces its
  prior contract byte-identically; the prototypes reproduce theirs field-for-field + the inert
  family-derived defaults the loader used to fill (asserted by `test_capability_manifests`).
- [ ] **T2 — Remove `target="gemmini"` defaults and `_DEFAULT_BACKEND`.** RTL/check/capsule stack +
  `target_registry.py:55-60`. Make `target` a required arg; add `runtime.default_backend` to reference
  contracts; read it like the external path (`target_registry.py:157-159`). Accept: no `="gemmini"`
  default in core signatures; `backend_for` reads the contract.
- [ ] **T3 — Route the capsule_bench harness through the descriptor (24 files).** Replace hardcoded
  gemmini paths / `GemminiRocketConfig` with `_common.TARGET`/descriptor. NOTE: several files have
  concurrent uncommitted edits — coordinate before touching.
- [x] **T4 — Derive the synthesize plans from family/contract, not `if name==`.**
  `synthesize/{llvm_extension_plan,runtime_adapter_plan,dialect_plan,target_contract,zephyr_plan}.py`.
  DONE: the `if name=="toy_npu"/"saturn"` branches + `CURATED_TARGETS={"saturn"}` + the per-name LLVM
  `_DEFAULTS` table are gone. The LLVM-fork posture rides `families.contract_endpoint_kind`
  (vector/scalar->maybe-fork; command_buffer/inline_asm_insn/external_backend->no fork); the runtime-
  adapter / zephyr concrete plans ride the contract's command-buffer tensor-resident features; the
  dialect plan rides the generic `_curated` (file existence) + tensor-resident generator. The neutral
  `toy_npu` example (`families.DEFAULT_EXAMPLE_TARGET`, the one sanctioned example — see T11) is the
  FAMILY DEFAULT, selected via that single constant, not a hardware-name branch. saturn's former
  bespoke `_saturn()` runtime-adapter plan is retired (untested; it now routes through the family
  default). Regression: `test_synthesize_family_derivation`.
- [ ] **T5 — `{target}`-parameterize the generic ABI contracts.**
  `merlin/contract/{mlir_oot_backend_contract,oracle_runner_contract}.yaml` + consumers.
- [ ] **T6 — De-hardcode build_tools gates.**
  `check_structure.py`, `check_standalone_install.py`, `check_repro_env.py` iterate the registry.
- [ ] **T7 — Derive kernel-mining features from the framework contract.**
  `kernels/features/{roles,dispatch,loops}.py`, `framework_contracts/`, `cca_contract.py`, `markers.py`,
  `cli_index.py`, `ingest/{autocomp,exo}.py`.
- [~] **T8 — CLI + comparison seams enumerate discovered targets.**
  `compile_cli.py`: DONE (`aeacba34`) — `--target` choices now derive from the registry (`all_targets()`);
  the gemmini branch generalized to `compile_oot(target=…)` with gemmini defaults preserved.
  `compare/spec.py:27` `_TARGETS` still pending.
- [ ] **T9 — `{target}`-template the targetgen_evals method/skill prompts.**
  `methods/{v0,v2,v3,v5,v6}/prompt.md`, `skills/*/AGENT.md`.
- [ ] **T10 — Miscellaneous derivations.** `evidence/report.CONCEPT_KEYWORDS`,
  `generate/target_repo.py:44`, `rtl/mlc_bridge.py:1080`.
- [ ] **T11 — Reference-target eviction (E-reference).** Move the in-tree reference backends / eval
  suites / dialects to a published reference-target package; keep one neutral `toy_npu` example. Fix the
  D-FLAG tests to parametrize over ≥2 targets.
- [~] **T12 — Dead-code / drift sweep.** DONE: `backends/__init__.py` stale docstring (`687f30ca`),
  `write_all` docstring drift (`583d01ec`). Skipped (would regress): `generate/target_repo.py` `camel()`
  `{"toy_npu":"ToyNPU"}` is acronym casing, not overfit — removing it degrades generated class names.
  Pending: `chia_repeatability.py:132` (harness, coordinate with T3), unused-symbol pass (with T11).

Ordered by blast radius. T1/T2/T4/T5/T6/T8/T10/T12 are software-verifiable; T3/T7/T9/T11 touch
harness/hardware/reference paths and need coordination or hardware to certify.

## Progress (branch `chore/target-agnostic-core`)

Completed + verified (targeted buckets green):
- T1 (`capability_manifests` -> residual-driven `manifest_for` + discovered `MANIFESTS`),
  T4 (`synthesize/*` family/contract-derived, no `if name==`), T8 (`compile_cli`), T12 drift fixes.

Intentionally deferred (NOT started as code): T2–T3, T5–T7, T9–T11. Each changes an emitted contract, a
hardware/sim-certified path (the gemmini path must re-certify byte-for-byte), the concurrently-edited
capsule_bench harness, or the reference-target eviction — none certifiable from a pure-Python
environment. They are fully specified above for execution behind the real test/hardware harness in
small per-task commits.
