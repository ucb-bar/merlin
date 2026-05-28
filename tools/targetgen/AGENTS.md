# `tools/targetgen/` — agent guide

## Mental model

The largest subcommand in the repo. `tools/targetgen/cli.py` exposes 13
subactions (`validate`, `plan`, `generate`, `explain`, `orchestrate`,
`execute`, `stage-mutation`, `answer`, `status`, `ingest`, `classify`,
`modification-map`, `mcp`) over the **planner-first hardware target
integration framework**: ingest a target's source tree, classify its
integration style (`runtime_hal` / `structured_text_isa` /
`post_global_plugin` / `llvm_ukernel`), plan the support workflow, and
optionally orchestrate the implementation through prompts to an LLM.

The framework lives in 13 modules (`planner`, `executor`, `generator`,
`loader`, `model`, `prompts`, `stage_map`, `target_routes`, `audit`,
`baseline`, `explore`, `orchestrator`) plus two subpackages: `intake/`
(10 source-tree scanners) and `prompt_library/` (21 markdown templates
that `prompts.py` composes for LLM packets).

`cli.py` keeps all `cmd_*` handlers; the underscore helpers it shared
across them (`_load_inputs`, `_target_out_dir`, `_write_json`,
`_render_*`, `_build_*_view`, `_build_evidence_graph`) were extracted to
sibling modules: `spec.py`, `paths.py`, `io.py`, `render.py`.

For the per-module map, read `__init__.py`.

## Pitfalls

- **Capability spec + deployment overlay are the input contract.** Every
  command takes a capability path (positional) and optional `--overlay`.
  Schemas live in `target_specs/schema/`. Adding a new top-level field
  requires updating the schema, `loader.py`, AND every command that
  surfaces it.
- **The MCP server is invoked TWO ways** for historical reasons:
  `./merlin targetgen mcp` (registered subaction; uses `cli.cmd_mcp`
  which delegates to `tools/mcp_servers/targetgen.py`) and `./merlin mcp
  targetgen` (uniform dispatcher path). Both route to the same registry;
  don't add a third entry point.
- **`prompt_library/` is data, not code.** The 21 markdown files are
  templates composed by `prompts.py:load_provider_config`. Renaming a
  template breaks `prompts.py`; renaming a directory breaks
  `prompts.PROMPT_LIBRARY_ROOT`.
- **`execute` and `stage-mutation` write files under
  `build/generated/targetgen/<target>/`** — this is the only place the
  package mutates the filesystem. Other commands (`plan`, `generate`,
  `explain`, `orchestrate`) are read-only / artifact-only.
- **`intake/` scanners are plugin-style** (one per source type). Adding
  a new source kind = new scanner file + register it in
  `intake/__init__.py`. Don't add scanner logic to `classifier.py`
  directly.

## Cross-references

- Consumes: `target_specs/<target>/capability.yaml` + optional
  `overlays/<env>.yaml`, plus an external target's source tree.
- Produces: `build/generated/targetgen/<target>/` with plan, mutations,
  prompts, execution-state JSON, evidence reports.
- MCP wrapper: `tools/mcp_servers/targetgen.py` exposes 11 tools
  (`ingest_source`, `classify_target`, `create_capability_draft`,
  `plan_target`, `get_modification_map`, `get_validation_commands`,
  `get_allowed_patch_surfaces`, `explore_target`, `propose_modifications`,
  `list_pipeline_stages`, `ingest_xpurt_feedback`).
- Related subcommand: `./merlin ray` for the Ray execution-engine path
  (`cli.cmd_execute --engine=ray` calls `ray.submit_job`).

## Update triggers

Re-read this file and update it in the same turn if you edit:

- `tools/targetgen/cli.py` (new subaction / `cmd_*` change) — refresh
  module map; touch `tools/mcp_servers/targetgen.py` if the externally-visible
  surface changed.
- `tools/targetgen/{planner,executor,generator,model,prompts}.py` (core
  framework changes) — refresh per-module list.
- `tools/targetgen/prompt_library/**.md` (template add/remove/rename) —
  Pitfalls section warns about this — update the count.
- `target_specs/schema/*.yaml` (schema field add/remove) — also touch
  `target_specs/AGENTS.md`.
