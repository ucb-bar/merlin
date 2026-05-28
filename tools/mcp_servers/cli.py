"""`./merlin mcp <name>` — start one of the Merlin MCP servers.

Each MCP server is a module under `tools/mcp_servers/<name>.py` containing a
`TOOL_REGISTRY` plus `dispatch_tool` / `list_tool_definitions` / `ToolError`.
The shared stdio scaffold (`tools/mcp_servers/scaffold.py`) wires them into a
real MCP server.

Example invocations (typically from `.mcp.json`):

    ./merlin mcp build
    ./merlin mcp compile
    ./merlin mcp run
    ./merlin mcp perf
    ./merlin mcp verify
    ./merlin mcp targetgen
"""

from __future__ import annotations

import argparse
import importlib

# One name per `tools/mcp_servers/<name>.py` module.
_PACKAGES = ("build", "compile", "run", "perf", "verify", "targetgen")


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "name",
        choices=_PACKAGES,
        help="Which MCP server to start. Each name N corresponds to tools/mcp_servers/N.py.",
    )


def main(args: argparse.Namespace) -> int:
    # Import the registry module + the shared scaffold, then start the server.
    try:
        registry = importlib.import_module(f"mcp_servers.{args.name}")
        scaffold = importlib.import_module("mcp_servers.scaffold")
    except ImportError as e:
        print(f"failed to import mcp_servers.{args.name}: {e}")
        return 2

    _, run_stdio = scaffold.make_stdio_server(
        server_name=f"merlin-{args.name}",
        logger_name=f"{args.name}.mcp",
        list_tool_definitions=registry.list_tool_definitions,
        dispatch_tool=registry.dispatch_tool,
        ToolError=registry.ToolError,
    )
    run_stdio()
    return 0
