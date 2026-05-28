# `tools/mcp_servers/` — agent guide

## Mental model

This package is **not** a `./merlin` subcommand in the usual sense — it
holds the MCP (Model Context Protocol) servers that expose merlin's
daily-driver subcommands as structured tools for Claude Code and other
MCP clients. The pattern is unique in the repo:

```
tools/mcp_servers/
├── cli.py            ./merlin mcp <name> dispatcher
├── scaffold.py       shared stdio scaffold (make_stdio_server factory)
├── build.py          merlin-build MCP tool registry
├── compile.py        merlin-compile MCP tool registry
├── run.py            merlin-run MCP tool registry
├── perf.py           merlin-perf MCP tool registry
├── verify.py         merlin-verify MCP tool registry
└── targetgen.py      merlin-targetgen MCP tool registry
```

Each domain-registry module (`build.py`, `compile.py`, etc.) exports:
`TOOL_REGISTRY` (list of `ToolDefinition`), `dispatch_tool(name, args)`,
`list_tool_definitions()`, `ToolError`. The dispatcher (`cli.py`) loads
the requested registry by name and hands it + the shared
`scaffold.make_stdio_server` to start the stdio server.

`.mcp.json` registers all 6 servers via `./merlin mcp <name>`.

## Pitfalls

- **Tool descriptions are auto-matched against user intent** by the MCP
  client. They are not arbitrary docstrings — write them in user-intent
  language ("Use when the user asks…"). Vague descriptions cost match
  quality.
- **Schemas must match the underlying CLI exactly.** Every tool's
  `input_schema` mirrors flags on the wrapped `./merlin <subcmd>`. When
  you add a CLI flag, the MCP `input_schema` MUST gain the matching field
  or callers will fail with `unknown argument`.
- **Each registry module's `Depends on:` header is load-bearing.** Per
  the Maintenance Protocol in `AGENTS.md`, editing the wrapped subcommand
  requires grepping for it in the MCP registries to know whether the
  schema/description needs updating. Keep those headers honest.
- **Structured returns, not stdout tails.** The pattern is: shell out to
  `./merlin <subcmd>`, parse the produced artifacts (vmfb, uartlog CSV,
  hash text) in Python, and return a typed dict. A raw stdout-tail tool
  is barely better than `Bash` and defeats the purpose.
- **Lazy imports inside tool handlers** keep `./merlin --help` fast.
  Module-level imports of heavy SDK dependencies (qairt-converter,
  onnxruntime, …) are a regression.

## Cross-references

- Each domain registry shells out to `./merlin <subcmd>` and parses its
  outputs:
    - `build.py`  → `./merlin build` + reads `models/*.yaml`
                    + walks `compiler/`, `runtime/`, `build_tools/`
                    for `build_check_freshness`.
    - `compile.py` → `./merlin compile` + inspects
                    `build/compiled_models/<model>/<target>/`.
    - `run.py`    → `./merlin run <mode>` + parses stdout for makespan
                    / per-instance latency / hashes.
    - `perf.py`   → `./merlin perf-decompose` + parses the CSV output.
    - `verify.py` → `./merlin verify-output` + parses hashes from stdout.
    - `targetgen.py` → calls the TargetGen framework in `tools/targetgen/`
                       directly (no shell-out).
- Shared scaffold (`scaffold.py:make_stdio_server`) is the only piece
  reused across registries. To add a new MCP server, write
  `tools/mcp_servers/<name>.py` with the four expected exports and add the name
  to `cli.py:_PACKAGES` + `.mcp.json`.

## Update triggers

Re-read this file and update it in the same turn if you edit:

- `tools/mcp_servers/cli.py` or `scaffold.py` — the dispatcher / shared scaffold.
- `tools/mcp_servers/{build,compile,run,perf,verify,targetgen}.py` — the
  registries. Adding a tool requires touching its `TOOL_REGISTRY`,
  `dispatch_tool`, and `list_tool_definitions`.
- `.mcp.json` — the MCP server registration. Keep aligned with `_PACKAGES`.
- Any wrapped subcommand's CLI surface — the MCP `input_schema`
  must mirror it exactly.
