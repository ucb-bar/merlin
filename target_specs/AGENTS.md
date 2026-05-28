# `target_specs/` — agent guide

## Mental model

The **newer canonical capability-spec surface** that drives TargetGen
(`tools/targetgen/`). Not the same thing as `models/*.yaml`:

| `models/*.yaml` | `target_specs/<target>/capability.yaml` |
|---|---|
| `iree-compile` flag bundle | Hardware capability declaration |
| Consumed by `./merlin compile` | Consumed by `./merlin targetgen plan / explain / orchestrate` |
| Per-variant flag overrides | ISA / memory / numeric-type / SDK requirements |

Both can exist for the same target — the YAML is the runtime contract,
the capability spec is the planning input.

## Layout

```
target_specs/
├── schema/
│   ├── capability_spec.yaml       canonical schema template
│   └── deployment_overlay.yaml    canonical overlay schema
└── examples/
    └── <target>/
        ├── capability.yaml        the spec itself
        ├── overlays/<profile>.yaml   environment-specific facts
        └── prompts/               optional LLM-prompt overrides
            ├── target/*.md
            └── overlays/<profile>/*.md
```

## Pitfalls

- **Capability spec is hardware-only.** Chipyard SHAs, FireSim recipe
  names, board transport details → deployment overlay, not capability.
  See `docs/how_to/add_hardware_spec.md`.
- **Multiple overlays per target are normal.** A target can have a
  bare-metal overlay and a FireSim overlay sharing one capability spec.
- **`prompts/` directories are data, not code.** TargetGen's prompt
  composition (`tools/targetgen/prompts.py`) merges these on top of
  the base prompt library in `tools/targetgen/prompt_library/`.
- **Renaming the example folder name breaks the planner's path math.**
  `_target_out_dir` in `tools/targetgen/paths.py` derives output dirs
  from the spec path. Be careful with renames.

## Cross-references

- Consumed by: every `./merlin targetgen <action>` (see
  `tools/targetgen/AGENTS.md`).
- Schema validation: `tools/targetgen/loader.py` against
  `schema/capability_spec.yaml` / `schema/deployment_overlay.yaml`.
- How-to: `docs/how_to/add_hardware_spec.md` for the contributor flow.
- Distinct from but co-existing with `models/<target>.yaml` (compile-flag
  view; see `models/AGENTS.md`).

## Update triggers

Re-read this file and update it in the same turn if you:

- Edit `target_specs/schema/capability_spec.yaml` or `deployment_overlay.yaml`
  (new field / removed field) — touch `tools/targetgen/loader.py` and
  every command that surfaces that field.
- Add a new example target under `target_specs/examples/` — note in
  `docs/how_to/add_hardware_spec.md` if the schema diverges.
- Modify `prompts/` directory layout — `tools/targetgen/prompts.py:PROMPT_LIBRARY_ROOT`
  needs to track it.
