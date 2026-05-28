# Use The Merlin TargetGen MCP Server With Claude Code

The TargetGen MCP server exposes the deterministic Merlin planner over the
Model Context Protocol. Claude Code (and any other MCP-aware client) can ask
TargetGen what files to read, what files to edit, and what `./merlin`
commands to run for a new hardware target — instead of guessing.

The server is **read-only**: it never edits repo-tracked files. Mutation
preparation continues to flow through `./merlin targetgen stage-mutation`.

## Launch

```bash
./merlin targetgen mcp
```

This starts an MCP server over stdio. Configure Claude Code (or any MCP
client) to spawn that command, e.g. in Claude Code settings:

```json
{
  "mcpServers": {
    "merlin-targetgen": {
      "command": "./merlin",
      "args": ["targetgen", "mcp"],
      "cwd": "/path/to/merlin"
    }
  }
}
```

For the CLI-based workflow that does not use MCP, see
[`bring_up_external_backend_with_targetgen.md`](bring_up_external_backend_with_targetgen.md).

## Tools exposed

| Tool | Purpose |
| --- | --- |
| `targetgen_ingest_source` | Walk source paths and emit a `SourceInventory`. |
| `targetgen_classify_target` | Map an inventory to integration styles + a capability draft. |
| `targetgen_plan_target` | Run the existing planner against a `capability.yaml`. |
| `targetgen_get_modification_map` | Per-stage patch surfaces for the nine pipeline stages. |
| `targetgen_get_allowed_patch_surfaces` | The allowed write paths + validation for one stage. |
| `targetgen_get_validation_commands` | All validation commands for the target (always `./merlin …`). |
| `targetgen_list_pipeline_stages` | Enumerate the canonical stage names. |

All tool results are JSON-serialisable. Schemas are advertised via
`tools/list`.

## Recommended Claude Code flow

Use the [`/merlin-targetgen`](../../.claude/commands/merlin-targetgen.md)
slash-command. It encodes the canonical procedure:

1. `targetgen_ingest_source` (with `target_name` + `source_paths`)
2. `targetgen_classify_target`
3. Promote `capability.draft.yaml` after operator review.
4. `targetgen_plan_target`
5. `targetgen_get_modification_map`
6. Pick a stage where `applies == true`.
7. `targetgen_get_allowed_patch_surfaces` for that stage.
8. Edit only paths in `allowed_write_paths`.
9. Run `validation_commands` via `./merlin`.
10. Surface unanswered `blocking_questions` to the operator.

## Hard rules

- The MCP server is read-only; do not script "live" edits through it.
- Never edit `third_party/iree_bar/` or LLVM submodules unless the stage's
  `allowed_write_paths` explicitly include them — the
  `forbidden_unless_approved` field flags the default protected roots.
- Always use `./merlin` for build/compile/validate. Direct `cmake`,
  `ninja`, or bare `python3 …` calls bypass the project's conda + uv
  environment guarantees.
- Use **dispatch** in any new docs/code; that matches the existing Merlin
  and IREE vocabulary.

## Troubleshooting

- *Tool returns `ToolError: missing or empty required argument …`* — the
  client did not supply the required field. Schemas list which fields are
  mandatory.
- *`Unknown stage` error* — only the nine names returned by
  `targetgen_list_pipeline_stages` are accepted.
- *Server fails to start* — confirm the conda env is active
  (`conda activate merlin-dev`). The `mcp` Python package ships with the
  Merlin dev environment.
