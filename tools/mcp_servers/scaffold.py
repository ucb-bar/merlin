"""Shared MCP stdio scaffold used by every `tools/mcp_servers/<name>.py` package.

Each MCP package supplies a `ToolError` and a `dispatch_tool` / `list_tool_definitions`
pair (its `tools.py`). This module turns those into a working stdio MCP
server with one factory call:

    # tools/mcp_servers/<name>.pyserver.py
    from .tools import ToolError, dispatch_tool, list_tool_definitions
    from .scaffold import make_stdio_server

    build_server, run_stdio = make_stdio_server(
        server_name="merlin-<name>",
        logger_name="<name>.mcp",
        list_tool_definitions=list_tool_definitions,
        dispatch_tool=dispatch_tool,
        ToolError=ToolError,
    )

Keeps per-package `server.py` files trivial.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server


def make_stdio_server(
    *,
    server_name: str,
    logger_name: str,
    list_tool_definitions: Callable[[], list],
    dispatch_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
    ToolError: type[Exception],
) -> tuple[Callable[[], Server], Callable[[], None]]:
    """Return `(build_server, run_stdio)` for a stdio MCP server.

    `list_tool_definitions()` must yield objects with `name`,
    `description`, and `input_schema` attributes. `dispatch_tool(name,
    args)` runs the named tool and returns a JSON-serialisable dict.
    Any `ToolError` raised is surfaced as a `ToolError:` text response;
    other exceptions are caught and surfaced as `InternalError:` with
    full traceback logged.
    """
    logger = logging.getLogger(logger_name)

    def build_server() -> Server:
        server: Server = Server(server_name)

        @server.list_tools()
        async def _list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.input_schema,
                )
                for tool in list_tool_definitions()
            ]

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
            try:
                result = dispatch_tool(name, arguments or {})
            except ToolError as exc:
                logger.warning("%s tool %s: %s", server_name, name, exc)
                return [types.TextContent(type="text", text=f"ToolError: {exc}")]
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("%s tool %s crashed", server_name, name)
                return [types.TextContent(type="text", text=f"InternalError: {exc}")]
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, sort_keys=False, default=str),
                )
            ]

        return server

    async def _run_stdio_async() -> None:
        server = build_server()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    def run_stdio() -> None:
        """Entrypoint used by `./merlin mcp <name>` to start the server."""
        asyncio.run(_run_stdio_async())

    return build_server, run_stdio
