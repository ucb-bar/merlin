# Agent instructions: EL4 TVM comparison

Read root `AGENT.md`/`CLAUDE.md`, this directory's [README.md](README.md),
[EL4_ASSETS.md](EL4_ASSETS.md), [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md), and the
companion TVM `apps/gemmini/{README,OWNERSHIP,TUNING,FIRESIM,AGENTS}.md`.
This directory is a trusted comparison edge, not the generated compiler candidate.

## Goal and current boundary

Prepare repeatable complete FireSim runs of **ResNet50, TinyLlama and SmolVLA**
under the shared [EL4 protocol at `0cc289de52a10bed256f32c90cfbb54a21b3c81f`](https://github.com/ucb-bar/merlin/tree/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol).
Use one selected `visible/v00` input per model, with explicitly recorded warmup
and measured repetitions. ResNet is one image/one call; TinyLlama is 32-token
prefill plus 32 teacher-forced decode calls (cache capacity 64); SmolVLA is
prefix + ten denoise calls + action decode. Check 1 / 99 / 33 output arrays,
respectively, including carried state and padded channels. The earlier 16+1
selection and exclusion of SmolVLA are superseded.

Model revisions, shapes, calibration counts, recipes and host-math policies are
already specified. Do not ask the user to choose them again. The 260 ResNet
training images are used to prepare scales once, not for each benchmark pass.
Stored visible inputs already exist in the repo. What remains is a portable,
shared numerical release, complete compiled sessions and actual qualification.
A previous capture is required only for reproducing that historical numerical
identity. Fresh preparation must freeze a new identity, with independently
admitted references, and must not inherit old qualification.

The delivery includes prepared `.pt2` export, TVM intake and
`build_package_session.py`/`read_package_session.py`. The bridge has small two-stage
Spike controls, including 9 GiB weights / 13 GiB arena in a 16 GiB map. Full-model
capacity, selected-reference agreement and FireSim loading remain unqualified.
The EL4 V26 record is a separate generated-compiler/capsule history; it records
`full_model_validation: NOT_RUN for V26`. Neither it nor the comparison controls
establishes the requested three-model FireSim result.

## Source isolation and ownership

- The 2026-10-23 fetch/pull confirms EL4 is still
  `0cc289de52a10bed256f32c90cfbb54a21b3c81f`, already contained in this branch.
  The selected workload/input definitions are unchanged from the user's `db2f524`.
  Record actual delivery/runtime, workload, model2MLIR and TVM hashes separately.
  Do not silently follow moving branches. TVM is a separate repository.
- Preserve original comparison/TVM worktrees and uncommitted changes. Use isolated
  branches; never reset another person's work.
- Keep target-specific comparison code here and ISA/schedule code in TVM contrib
  or the selected backend. Do not add target-name literals to Merlin core.
- Reuse the capture worker, independent reference producer, shared numerical
  checker, package-session bridge and post-freeze FireSim prepare/collect tools.
  Full-model integration is unfinished; a parallel runtime is not required.
- Keep trusted TVM schedules/references outside the generated compiler's grants.
  Preserve frozen Phase 1 identities and the stopped manual campaign; do not edit
  its deadline, baseline or `manual-stop.json` to reuse a runner.

## Practical sequence

1. **Resolve the shared contract.** Export/check the pinned protocol and select
   `v00` inputs with their existing hashes. Publish a resolved models configuration
   with ResNet `native`, TinyLlama `portable_exp_trig` and SmolVLA
   `portable_neural_f32`; the generic models file defaults to native. Preserve
   the existing tolerances and source-versus-quantized quality reporting policy.
2. **Make preparation portable.** Follow `EL4_ASSETS.md`. Verify the stored
   ResNet/TinyLlama derivations. Close SmolVLA's processed-observation hash gate;
   fix/adapt the ignored `--data` path and use the selected LeRobot 0.6 processors,
   rather than the generic setup helper's 0.5.1. Keep capture/calibration/evaluation
   selections distinct. Do not run `inputs-prepare` against the pinned public
   manifest in place: it writes the source-adjacent files and manifest.
3. **Freeze and publish the numerical package.** Reuse an admitted release or
   finish capture preparation from the specified public sources. In the latter
   case, implement TinyLlama's declared WikiText calibration and
   SmoothQuant/outlier integration; record any new numerical identity. Retain
   weights/scales/IR, prepared stage `.pt2` files, all-output references and
   environment pins. Give every member a shared artifact locator and checksums;
   do not make historical machine paths a prerequisite.
4. **Compile the selected programs.** Use `--save-exported-program` for the same
   conversion used by Merlin. Extract all TVM tasks with `--program`, record CPU
   exclusions and preset coverage, then compile every selected stage. Old factory
   semantic JSON and native-source diagnostics do not substitute for EL4 intake.
5. **Qualify the complete guest session.** Reuse `build_package_session.py` with
   explicit stages/routes and independent fixtures. Carry actual candidate state
   through all calls and reset each repetition. Check every output, exact integers
   and finite floats; retain positive/wrong-oracle controls. Preserve the prior
   SmolVLA diagnostic failure and establish the selected capture's correctness
   without widening tolerances. Audit all memory, instructions, symbols and final
   ELF segments. Hardware-loop FSM commands remain forbidden; raw-command OS/WS
   and software loops are legal under the comparison's device contract.
6. **Bind the actual deployment.** Fill the FireSim manager contract, compiler
   freeze and complete-chain asset manifests. EL4's native complete-chain builder
   already exists. TVM's session schema/ABI and UART protocol need an explicit
   adapter to its source-bound manifest and compact comparison, or a supported
   full-array manager integration; do not rename schemas to bypass validation.
   Qualify the actual driver, boot
   convention, memory span and high addresses. A different F2/U250 image is a
   different hardware identity even when its Gemmini params header matches.
7. **Prepare, execute and collect.** Follow the pinned
   [post-freeze workflow](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/README.md):
   one complete model at a time. Strict version-pinned requests return an
   `elf_upload` staging module, not old-helper launch arguments; the supplied
   strict manager publisher requires compact comparison and actual installation.
   Lightweight `booted_elf_hash_v1` uses the installed helper and actual boot-file
   evidence, also with compact comparison. Legacy full-array transport remains
   separate. Follow the mode table in README; do not require a strict upgrade
   merely to choose lightweight reproducibility. Retain original SSM/upload/boot
   evidence. Paid FPGA jobs are complete models; keep small controls on functional
   engines. The old queue is a separate deployment option.
8. **Complete validation and report.** For full arrays, export UART with
   `read_package_session.py`, bind the manager output roster and use
   `merlin.model_check.baseline_numerics`. Compact mode compares every element in
   the guest against admitted references and retains its full transcript. Both
   leave `model_validation_complete` false pending the independent source,
   placement, exact-kernel, instruction and hardware gates. Report complete
   target-cycle windows, cold/warm policy, stage counts, host/transfer work, memory
   and simulator wall time. Add per-stage TVM counters without dropping aggregate
   timing; native stage counters exclude transfers and cannot be summed as E2E.
   The bridge currently fixes one warm and one measured invocation. Do not
   substitute summed operator minima for E2E time.

## Testing and known environment constraints

Create outputs with `merlin.common.artifacts.new_product/new_measurement`.
Comparison tests live in `merlin/tests/gemmini`; use the selected Torch/ONNX/TVM
Python, `MERLIN_REPO_ROOT`, matching `TVM_LIBRARY_PATH` and the model2MLIR roots
required by the tests. The delivery's TinyLlama cache controls used Transformers
5.4.0; its older SmolVLA diagnostic environment used incompatible 4.57.1 APIs.
These are recorded probe environments, not a complete three-model environment
lock. Source-exact SiLU diagnostics require their pinned Torch/AVX512 settings;
keep those guards, and distinguish them from EL4's portable host-math policies.
The current broad comparison run also exposes a LeRobot 0.5.1/diffusers 0.35.2/
TorchAO 0.16.0 import conflict. Pin compatible per-model preparation environments;
do not treat a union of Python overlays as a validated team environment.

Run checks appropriate to the changes and `python build_tools/scripts/check_structure.py`;
report inherited structure failures separately. Example-local documentation needs
no docs frontmatter. New heavyweight/generated/protected payloads stay outside
Git; preserve the existing committed visible inputs. Never overwrite execution
evidence after code changes. No PR or force-push is authorized by this handoff.
