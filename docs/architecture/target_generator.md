# Target Generator

Merlin TargetGen is a planner-first control plane for hardware-spec-driven
target enablement.

It treats Merlin as:

`hardware capability spec -> normalized capability model -> decision engine -> support plan + task graph -> agentic implementation workflow`

## Why this exists

Merlin's current target knowledge is split across several surfaces:

- `models/*.yaml` for compile flags
- `build_tools/hardware/*.yaml` for deployment recipes
- compiler plugins and dialects for semantic recovery
- IREE/LLVM fork changes for ISA exposure and ukernel behavior
- runtime/HAL code for device-backed targets

TargetGen introduces one canonical source of truth for target capabilities and
an optional deployment overlay for environment-specific data. The planner then
derives the support plan, verification ladder, and execution-ready task graph.

## Canonical inputs

TargetGen uses two YAML inputs:

1. Capability spec
   Lives under `target_specs/examples/<target>/capability.yaml`.
   Captures hardware-facing facts such as execution model, ISA exposure,
   operation coverage, tiling, memory, numeric contract, runtime contract, and
   verification resources.

2. Deployment overlay
   Lives under `target_specs/examples/<target>/overlays/*.yaml`.
   Captures environment-specific pins such as Chipyard branch/SHA, build
   profile, simulator mode, compile target name, and current hardware recipe.

This split keeps Chipyard or board-specific data out of the core hardware spec
so future non-Chipyard targets can use the same planner.

## Current planner outputs

`merlin targetgen plan` emits:

- `support_plan.json`
- `task_graph.json`
- `verification_manifest.json`
- `compile_view.yaml`
- `deployment_view.yaml` when an overlay is provided

`merlin targetgen generate` builds on the planner outputs and emits:

- `generation_bundle.json`
- `generation_summary.md`
- `inputs/capability.yaml`
- `inputs/overlay.yaml` when an overlay is provided
- `generated/tree/...` containing non-live scaffold files at prospective repo paths

`merlin targetgen orchestrate` emits:

- `execution_bundle.json`
- `execution_state.json`
- `task_states/*.json`
- `briefs/*.md`
- `prompts/prompt_NNN.md` when a prompt backend is enabled
- `provider_backend.json` when `--prompt-backend provider` is used

The execution bundle is the agentic scaffold. It does not generate static
patches. Instead, it tells an implementation agent which evidence to read,
which layers are likely to change, and which acceptance checks must close the
task. The paired execution-state artifacts make that scaffold operational by
recording adapter selection, repo root, mutation policy, validation contracts,
and per-task lifecycle state.

`merlin targetgen stage-mutation` is the bridge between planning and live
edits. It still does not mutate repo-tracked files. Instead it emits:

- `mutation/mutation_bundle.json`
- `mutation/proposal_brief.md`
- `mutation/worktree_plan.md`
- `mutation/proposed_tree/...`

This gives Merlin a reviewable proposed tree before any branch switch, worktree
creation, or validation run.

## Execution backends

TargetGen now has two execution engines:

- `local`: the in-process executor that advances task state directly
- `ray`: a control-plane backend that persists the same execution bundle and
  submits the local executor as a Ray Job

This keeps TargetGen as the planning source of truth while letting Merlin move
the orchestration and scheduling path onto Ray without introducing a second
planner.

## Dynamic prompt layer

TargetGen now resolves prompts through a layered Markdown library instead of a
single hardcoded string.

The hierarchy is:

1. `tools/targetgen/prompt_library/base/`
2. `tools/targetgen/prompt_library/families/<family>/`
3. `tools/targetgen/prompt_library/phases/<phase>/`
4. `tools/targetgen/prompt_library/tasks/<task>.md`
5. `target_specs/<target>/prompts/target/`
6. `target_specs/<target>/prompts/overlays/<deployment>/`

Each fragment is a Markdown file with YAML frontmatter. Frontmatter declares:

- `section`: one of `system`, `goal`, `target_facts`,
  `implementation_focus`, `evidence`, `write_scope`, `acceptance`, or
  `response_contract`
- `merge`: `append`, `prepend`, or `replace`
- optional selectors such as `families`, `phases`, `task_ids`,
  `integration_styles`, `targets`, `deployment_profiles`,
  `prompt_backends`, and `tool_profiles`

Fragment bodies are rendered with Jinja2 in strict mode. That keeps prompt
authoring declarative while failing fast on misspelled or missing variables.

## Execution backends

`merlin targetgen orchestrate` supports three prompt backends:

- `none`: emit only the execution bundle and human-readable briefs
- `manualllm`: emit `prompt_NNN.md` packets that follow the file-based
  prompt/response convention already used by `projects/mlirAgent`
- `provider`: emit the same prompt packets plus provider metadata derived from
  `projects/mlirAgent/configs/agents/*.yaml`

Tasks in the `post_global_plugin` and `structured_text_isa` families default to
the `mlir_agent` tool profile. Other families default to a generic contextual
edit profile. This keeps MLIR-centric tasks aligned with the existing
`projects/mlirAgent` workflow without forcing every target family through the
same execution adapter.

## Execution adapters

Task nodes now carry direct execution metadata:

- `repo_root`
- `execution_adapter`
- `mutation_policy`
- `artifacts_in`
- `artifacts_out`
- `validation_commands`
- `credential_requirements`
- `handoff_contract`

Current adapters are:

- `merlin_local`
- `iree_submodule`
- `llvm_submodule`
- `runtime_hal`

This still is not a fully autonomous patch engine, but it makes the task graph
decision-complete for an execution layer that mutates Merlin, IREE, and LLVM
from one Merlin-owned control plane.

## Decision rules

The planner currently recognizes four integration styles:

- `llvm_ukernel`
- `post_global_plugin`
- `structured_text_isa`
- `runtime_hal`

Those styles are derived from the declared execution model, compiler recovery
stage, ISA exposure, and runtime requirements.

Examples:

- SpacemiT and Saturn route through `llvm_ukernel`
- Gemmini routes through `post_global_plugin` and `llvm_ukernel`
- NPU routes through `post_global_plugin` and `structured_text_isa`
- Radiance routes through `runtime_hal`

External commercial and open exemplar specs now include:

- `nvidia_vulkan_ada`
- `qualcomm_adreno_vulkan`
- `nvidia_cuda_sm89`
- `qualcomm_qnn_htp_snapdragon8elite`
- `ara_rvv_vlen256`
- `vortex_gpgpu_u250`

## Current status

This is the planner-first slice of TargetGen. It intentionally does not replace
existing handwritten target YAMLs or hardware recipes yet. The current planner
produces derived views and validation artifacts so exemplar parity can be
proven before downstream adoption.

The new `generate` and `stage-mutation` commands extend that posture without
crossing into live mutation:

- `generate` stages likely repo changes under `build/generated/targetgen/...`
- `stage-mutation` groups the mutating subset into a proposed tree
- neither command promotes files into the repository or runs validation

That keeps the system agentic-first and review-friendly while the actual
mutation and verification engines continue to mature.
