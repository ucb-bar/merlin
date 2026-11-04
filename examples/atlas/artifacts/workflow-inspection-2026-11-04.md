# Atlas workflow inspection, November 4, 2026

This local check used Merlin `0a68cf0a4091eda9bd82cceeb297b70f824537c8` on `followup/conform-workflow`, before these documentation changes. It records configuration inspection and an unsuccessful diagnostic generation attempt, with no reviewed release, compiler qualification, or hardware run. The selected definition is [atlas-functional](../experiment.yaml), target `atlas`, and the descriptor is [target/descriptor.yaml](../target/descriptor.yaml).

The descriptor declares `EE290SimConfig`, source revision `426a862f97f98938660772bd8a5d8f41316d15c3`, and Atlas gitlink `0079c0541111197741a231c002e3843fa6f545b2`. No matching populated RTL facts or common-elaboration receipt was selected. Explicit `MERLIN_TARGET_PATH=<merlin>/examples/atlas/support` selected provider `atlas-support`, role `support`, from [SOURCE.yaml](../SOURCE.yaml): companion commit `392eb077c8a48bd3a21b253d9e488ba3d7ad74cf`, source tree `42315503deaf7d52b1196354abd9b24eabaf7dc4`, and historical normalized vendored tree `c68fb5508610f67372b73936231edd476c6fe2e0`. The current committed support tree is `14542f90f7370dd614506cee06727b5b7579d227`, which differs from that historical record; [program_emitter_migration.json](../support/program_emitter_migration.json) separately records the emitter relocation and contract delta. This check does not establish current byte identity to the historical companion. These are support identities, not a compiler artifact. No evaluated/generated functional compiler or frozen submission SHA-256 was supplied; `host_lane.commit` remains `UNKNOWN`. The selected support has no executable `plugin.backend`.

Exact input SHA-256 values from this checkout:

| Input | SHA-256 |
| --- | --- |
| `examples/atlas/experiment.yaml` | `6a471471c8dd5570d6fa7bb1e3c06924888a84472f31d4fd927fa8dbe56f5c70` |
| `examples/atlas/target/descriptor.yaml` | `7aa0dec64f4159f59b408f42a96b67f45961732e236111d8d974d93466b8fda5` |
| `examples/atlas/target/contracts/target_contract.yaml` | `603a8d5abadd19bc6252f0a838688b85c938b8a4741018f0c897ee9c6d3a587c` |
| `examples/atlas/support/provider.yaml` | `7b6dedebfafc410cfb525082839aab6c8bb097b8ca11d4f0395e19aed90cc2dc` |
| `examples/atlas/support/contracts/target_contract.yaml` | `1eefe1d5941869b3063a09e82944fbb765934f50a93f27b78ebde86bddd98af0` |
| `examples/atlas/SOURCE.yaml` | `2771014dafe10562cb35497114f2d3ff657a526bb447d2a767dd12aa13cbbab6` |

## Commands and results

Commands below start with `merlin experiment`; the actual invocation was the supplied Python 3.12.12 venv's `python -m merlin.cli experiment`, resolving core and experiments packages from this checkout through explicit `PYTHONPATH`. `MERLIN_REPO_ROOT` selected this checkout, `MERLIN_OUT_ROOT=<workspace>/out/merlin`, and `TMPDIR=<workspace>/tmp`. The environment selected no Model2MLIR, SpecIR, NPU model, compiler, or external RTL roots. A local Python traversal guard pruned contract-restricted names before recursive enumeration and denied direct access to those paths. This access guard did not supply or replace any workflow stage.

| Arguments after `merlin experiment` | Exit | Observed result |
| --- | --- | --- |
| `inspect atlas-functional --phase 0` | 0 | 140 selected input files; evidence `diagnostic`; 14 evidence diagnostics. |
| `preflight atlas-functional --phase 0` | 0 | `configuration_ready: true`, `engine_readiness: not_executed`, no errors, synthesis `absent`. |
| `inspect atlas-functional --phase 1` | 0 | EL4 `capsule_bench`, 234 input paths, reviewed corpus required. |
| `preflight atlas-functional --phase 1` | 2 | `configuration_ready: false`, `engine_readiness: not_executed`, 11 errors, including one repeated timing-path error. |
| `inspect model-portfolio-template --phase 2` | 0 | `kind: template`, target `OPERATOR_TARGET`, 4 input paths. |
| `inspect measured-claims-template --phase 2` | 0 | `kind: template`, target `OPERATOR_TARGET`, 9 input paths, placeholder functional identity and certificate hashes. |
| `run atlas-functional --phase 0 --phase0-evidence-mode diagnostic --run-dir <workspace>/out/phase0-diagnostic` | 1 | One frozen engine attempt, `execution_failed`; fresh digest-bound conformance and synthesis inputs required. |
| `corpus prepare <workspace>/out/phase0-diagnostic --generated-only --output <workspace>/out/merlin/artifacts/protocols/atlas/workflow-corpus` | 2 | Refused: Phase 0 has no successful immutable output receipt; derive a new corpus. |
| `corpus inspect <workspace>/out/merlin/artifacts/protocols/atlas/workflow-corpus` | 2 | Release path absent after preparation refusal; no corpus inspected successfully. |

An initial corpus prepare used `<workspace>/out/corpus-release` and exited 2 because releases must live below the configured artifact root; the corresponding inspect also exited 2 because that path was absent. The corrected attempt above reached the actual source-run gate. No release directory or seal was manufactured. The nine principal invocations above comprise five exit-0 inspections/configuration checks and four unsuccessful or refused invocations; the two initial path-error invocations are separate.

Raw stdout, stderr, invocation timestamps, and explicit environment are retained in the job's `<workspace>/out/`: `entrypoints.json`, `phase0-diagnostic-command.json`, `corpus-commands.json`, `corpus-artifact-root-commands.json`, and each command's named `.stdout`/`.stderr`. The engine's frozen plan, orchestration receipt, evidence manifest, and `phase0-attempt-0001.log` are under `out/phase0-diagnostic/`. These local receipts are not published corpus or compiler certificates.

## Unmet inputs observed by the commands

Phase 0 evidence diagnostics, retained without promoting them to admission:

- `application-inventory`: no selected detailed application inventory
- `residual`: unavailable: <merlin>/examples/atlas/support/contracts/residual.yaml
- `numerical-model`: <merlin>/examples/atlas/target/software-spec.yaml: selected model requires SPECIR_ROOT
- `support-contract`: selected compute unit 'mxu' declares scaling None, but the support provider declares 'block_e8m0'; review the provider against the selected RTL readout and freeze a fresh corpus
- `facts`: no populated RTL facts selected
- `isa-taxonomy`: ExternalPathUnset: "external path 'npu_model' unset; set MERLIN_EXT_NPU_MODEL in .env (copy .env.example). Known: []"
- `source-consistency`: no verified common-elaboration receipt selected
- `rtl-audit`: no rtl-source-audit report (validation.json) beside the selected facts
- `software-review`: selected numerical software semantics are not reviewed
- `framework-catalog`: selected capture framework operator catalog is unavailable or incomplete
- `required-hardware-field`: selected facts do not establish facts.arrays
- `required-hardware-field`: selected facts do not establish facts.memories
- `required-hardware-field`: selected facts do not establish facts.timing
- `required-hardware-field`: selected facts do not establish source_consistency

Phase 1 preflight errors, preserving the emitted denominator and repository-relative path spellings:

- this functional experiment requires a reviewed Phase 0 release; select its seal and release descriptor before starting Phase 1
- input is absent or is not a regular file/directory: <merlin>/merlin/experiments/capsule_bench/targets/atlas/.oracle_timing.json
- input is absent or is not a regular file/directory: <merlin>/merlin/experiments/capsule_bench/targets/atlas/.oracle_timing.json
- input directory contains a directory symlink: <merlin>/merlin/contract/capsules/conformance; name its closure explicitly
- input is absent or is not a regular file/directory: <merlin>/experiments/capsule_bench/targets/atlas/contracts/hwbringup_atlas_v0/isa_include/atlas_isa_green_card.md
- input is absent or is not a regular file/directory: <merlin>/experiments/capsule_bench/targets/atlas/contracts/hwbringup_atlas_v0/isa_include/isa_definition.py
- input is absent or is not a regular file/directory: <merlin>/third_party/llvm-install
- input is absent or is not a regular file/directory: <merlin>/experiments/capsule_bench/targets/atlas/contracts/hwbringup_atlas_v0
- input is absent or is not a regular file/directory: <merlin>/merlin/targets/atlas/contracts/rtl_facts
- phase 1 toolchain clang does not resolve to an executable file: 'clang-23' (checkout <merlin>); set MERLIN_CLANG or install the checkout's LLVM toolchain
- phase 1 EL4 target 'atlas' has no selected support plugin.backend; RTL checks cannot run from the metadata-only example. Select a reviewed OOT support provider.

## Stage boundary and next inputs

The diagnostic engine stopped with `derived-only Phase 0 requires fresh digest-bound conformance and synthesis inputs; run merlin experiment corpus derive first`. Provision the exact four independent iteration captures (`coverage_mlp`, `residual_cnn`, `causal_decoder`, and `multimodal_policy`), selected common-elaboration RTL facts and audit, independent numerical model/runtime, and explicit compiler/tool identities. Then use the existing `corpus derive`, inspect/preflight, and diagnostic generation entrypoints with those new inputs. Keep unresolved FP8 scale encoding/block size, provider scaling contradictions, software review, and required decoder-mode obligations explicit. The known npu_model revision in the bringup contract does not automatically select it for this experiment.

As in [Radiance Phase 0](../../radiance/phase0/README.md), a diagnostic generation is followed by independent review before release sealing; this unsuccessful attempt supplies no generated cohort. [Radiance Phase 1](../../radiance/phase1/README.md) likewise requires renewed bundles and reviewed admission. Its descriptor distinguishes worked-example, no-kernel, and pinned-library information treatments; Atlas currently leaves its model-specific library selection disabled because its kernels overlap the held-out validation roster. Neither materialization nor existing kit results establish a new functional compiler campaign. [Radiance Phase 2](../../radiance/phase2/README.md) and Atlas both hand off frozen functional compiler evidence to shared templates with separate performance cohorts and protocol prerequisites.

A new reviewed Phase 0 release, successful corpus preparation/inspection, human review and seal, verified Phase 1, and Phase 2 preflight/execution remain **PENDING**. The 16-ELF identity, 1,589-program census, CTest, and atlas-kernels `make test` gates were not run by this documentation task and remain **PENDING** for the orchestrator.
