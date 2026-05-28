"""MCP servers package — one tool registry module per domain.

`./merlin mcp <name>` dispatches via `cli.py` into the chosen registry
(build, compile, run, perf, verify, targetgen). The shared stdio scaffold
lives in `scaffold.py`.

Adding a new MCP server:
    1. Create `tools/mcp_servers/<name>.py` with TOOL_REGISTRY, ToolError,
       dispatch_tool, list_tool_definitions (see existing modules).
    2. Add the name to `cli.py`'s `_PACKAGES` tuple.
    3. Add the entry to `.mcp.json`.
"""
