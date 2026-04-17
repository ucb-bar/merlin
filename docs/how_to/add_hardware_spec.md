# Add A Hardware Spec

TargetGen adds a canonical hardware-facing spec for target enablement planning.

## 1) Start from the schema templates

Use:

- `target_specs/schema/capability_spec.yaml`
- `target_specs/schema/deployment_overlay.yaml`

Copy them into:

- `target_specs/examples/<target>/capability.yaml`
- `target_specs/examples/<target>/overlays/<profile>.yaml`

## 2) Fill the capability spec

The capability spec should only capture hardware facts and directly-derived
compiler/runtime requirements:

- target identity and maturity
- platform and environment types
- execution model and compiler recovery stage
- ISA exposure and state model
- native operations
- geometry and preferred tiles
- memory spaces and layout/packing rules
- legal numeric type triples
- runtime contract
- verification resources
- access model, SDK requirements, credential requirements, and verification gates
- current in-repo evidence paths in `references`

Do not put Chipyard SHAs, FireSim recipe names, or board transport details into
the capability spec. Those belong in the deployment overlay.

## 3) Fill the deployment overlay

Use the overlay for environment-specific facts:

- build profile
- compile target name and hw key
- current hardware recipe path
- Chipyard repo/branch/SHA/config class
- simulator or board transport details
- local device, lab device, or cloud transport details

Targets can have multiple overlays. For example, a target may have separate
bare-metal and FireSim overlays while sharing one capability spec.

## 4) Validate and inspect

Validate the spec:

```bash
conda run -n merlin-dev uv run tools/merlin.py targetgen validate \
  target_specs/examples/<target>/capability.yaml \
  --overlay target_specs/examples/<target>/overlays/<profile>.yaml
```

Generate the support plan:

```bash
conda run -n merlin-dev uv run tools/merlin.py targetgen plan \
  target_specs/examples/<target>/capability.yaml \
  --overlay target_specs/examples/<target>/overlays/<profile>.yaml
```

Explain the classification:

```bash
conda run -n merlin-dev uv run tools/merlin.py targetgen explain \
  target_specs/examples/<target>/capability.yaml \
  --overlay target_specs/examples/<target>/overlays/<profile>.yaml
```

Generate the execution bundle:

```bash
conda run -n merlin-dev uv run tools/merlin.py targetgen orchestrate \
  target_specs/examples/<target>/capability.yaml \
  --overlay target_specs/examples/<target>/overlays/<profile>.yaml
```

Generate mlirAgent-compatible prompt packets:

```bash
conda run -n merlin-dev uv run tools/merlin.py targetgen orchestrate \
  target_specs/examples/<target>/capability.yaml \
  --overlay target_specs/examples/<target>/overlays/<profile>.yaml \
  --prompt-backend manualllm
```

Generate provider metadata from `projects/mlirAgent/configs/agents/*.yaml`:

```bash
conda run -n merlin-dev uv run tools/merlin.py targetgen orchestrate \
  target_specs/examples/<target>/capability.yaml \
  --overlay target_specs/examples/<target>/overlays/<profile>.yaml \
  --prompt-backend provider \
  --agent codex
```

## 5) Add target-specific prompt overrides when needed

The base TargetGen prompt library lives under:

- `tools/targetgen/prompt_library/base/`
- `tools/targetgen/prompt_library/families/`
- `tools/targetgen/prompt_library/phases/`
- `tools/targetgen/prompt_library/tasks/`

Targets can extend that library without changing planner code:

- `target_specs/examples/<target>/prompts/target/*.md`
- `target_specs/examples/<target>/prompts/overlays/<profile>/*.md`

Each prompt fragment must have YAML frontmatter:

```markdown
---
section: target_facts
merge: append
---
Your Markdown content here.
```

Supported sections are:

- `system`
- `goal`
- `target_facts`
- `implementation_focus`
- `evidence`
- `write_scope`
- `acceptance`
- `response_contract`

Supported merge modes are:

- `append`
- `prepend`
- `replace`

Optional selectors let one fragment apply only to certain task families,
phases, task IDs, prompt backends, tool profiles, targets, deployment profiles,
or integration styles.

## 6) Agentic execution metadata

`merlin targetgen orchestrate` now emits:

- `execution_bundle.json`
- `execution_state.json`
- `task_states/*.json`
- `briefs/*.md`
- `prompts/prompt_NNN.md` when prompt packets are enabled

Each generated task now includes:

- the primary repo root it should operate in
- the execution adapter to use
- the mutation policy for that task
- artifact inputs and outputs
- validation commands
- credential requirements
- a handoff contract for the executor

This is what lets Merlin own one agentic workflow even when the eventual work
lands in Merlin-local code, the IREE fork, or the LLVM fork.

## 7) What TargetGen does with the spec

The planner turns the spec into:

- a normalized capability model
- family classification
- a support plan
- a verification ladder
- an ordered task graph
- sectioned task briefs
- execution-state manifests
- optional prompt packets and provider metadata

The execution bundle is intentionally evidence-first. It points an agent at the
right Merlin, IREE, LLVM, or runtime surfaces instead of trying to emit a
static patch from YAML alone.
