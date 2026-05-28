# `tools/archive/` — agent guide

## Mental model

Frozen, never-deleted snapshots of tools that were relocated out of
active use during reorganization passes. Per [[feedback_preserve_results]],
**nothing here is ever deleted** — every script that produced plots,
CSVs, or board-verified results is preserved so the outputs remain
reproducible.

The full index lives in `tools/archive/README.md`. This AGENTS.md adds
the **rules of engagement** for agents touching this subtree.

## Rules

1. **Read-only by default.** Don't edit scripts in `tools/archive/`.
   They're frozen for reproducibility. If you need the same logic,
   copy the relevant code into the active tool and reference the
   archived path in a comment.
2. **Don't relocate or rename anything inside `archive/`.** Every
   subfolder is referenced from a previous PR description, memory
   entry, or dev_blog. Renaming silently breaks those references.
3. **Don't add new code here unless you're archiving.** If the
   workflow is still in use, it belongs in active `tools/<subcmd>/` or
   `scripts/`.
4. **Adding a new archive entry** requires a `README.md` blurb at the
   top level of this folder explaining what was archived, why, and the
   date — see existing entries as templates.

## Inventory

See `tools/archive/README.md` for the per-folder index. Brief:

- `qnn_v2/` — bindings-based emitter that never reached v1 parity
- `qnn_e2e_demo/` — demo orchestrators from the QNN bring-up campaign
- `qnn_islands/` — HTA conv-island export & profiling
- `compile_internals/` — one-shot helpers from compile-flow refactors
- `gemmini_bug_a/` — debug fixtures from a specific investigation
- `mlir_helpers/`, `quant_debug/`, `scheduling/`, `yaml_duplicates/`,
  `compile_internals/`, `part_c/` — relocated during the
  2026-05-25 cleanup

## Update triggers

Re-read this file and update it in the same turn if you:

- Add a new archive subfolder (extend the Inventory table + add an entry
  to `tools/archive/README.md` with date and rationale).
- Discover that an archived script is still being invoked — note it
  in Pitfalls; **don't un-archive without explicit user approval**.
