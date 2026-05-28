#!/usr/bin/env bash
# Verify that the .mcp.json registration of merlin-targetgen is healthy
# from a fresh shell. Run this from the repo root.
#
# Exits non-zero with a clear diagnostic if:
#   * .mcp.json is missing or unparseable.
#   * The advertised command does not launch a working MCP server.
#   * The expected 7 tools are not exposed.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ ! -f .mcp.json ]; then
    echo "::ERROR:: .mcp.json missing in $REPO_ROOT" >&2
    exit 2
fi

# Use the same conda env Merlin tests use.
if ! command -v conda >/dev/null 2>&1; then
    echo "::ERROR:: conda not on PATH; activate your shell init first" >&2
    exit 2
fi

EXPECTED_TOOLS=(
    targetgen_ingest_source
    targetgen_classify_target
    targetgen_create_capability_draft
    targetgen_plan_target
    targetgen_get_modification_map
    targetgen_get_allowed_patch_surfaces
    targetgen_get_validation_commands
    targetgen_list_pipeline_stages
)

# Probe the registered server through the same wire-protocol path
# Claude Code uses (mcp.client.stdio). `--live-stream` is required —
# without it `conda run` buffers and silently drops the heredoc's
# stdout, hiding both progress messages and exit-status diagnostics.
#
# We capture conda's exit code explicitly because `set -e` does not
# always propagate through `conda run`'s subprocess wrapper.
set +e
conda run --live-stream -n merlin-dev python - "${EXPECTED_TOOLS[@]}" <<'PY'
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

expected = set(sys.argv[1:])

cfg = json.load(open(".mcp.json"))
servers = cfg.get("mcpServers") or {}
if "merlin-targetgen" not in servers:
    print("::ERROR:: .mcp.json missing mcpServers.merlin-targetgen", file=sys.stderr)
    raise SystemExit(2)

server = servers["merlin-targetgen"]


def expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


command = expand(server["command"])
args = expand(server.get("args", []))
env = dict(os.environ)
env.update(expand(server.get("env", {})))

print(f"Launching: {command} {' '.join(args)}")


async def go():
    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_tools()
            advertised = {t.name for t in response.tools}
            missing = expected - advertised
            extra = advertised - expected
            print(f"Advertised tools: {sorted(advertised)}")
            if missing:
                print(f"::ERROR:: missing expected tools: {sorted(missing)}", file=sys.stderr)
                raise SystemExit(1)
            if extra:
                print(f"NOTE: server advertises additional tools: {sorted(extra)}")
            print("OK: all expected tools advertised")


asyncio.run(go())
PY
RC=$?
set -e

if [ $RC -ne 0 ]; then
    cat <<MSG >&2

::ERROR:: MCP server launch failed (rc=$RC).

Common causes:
  * The Merlin .venv is broken (root-owned, read-only) — uv cannot
    recreate it. Reinstall with: ./merlin setup --reset-venv
  * conda env merlin-dev is missing — see docs/getting_started.md.
  * The 'mcp' Python package is not installed in merlin-dev.

If you're a contributor running this for the first time, run the
standard install path documented in CLAUDE.md before retrying.
MSG
    exit $RC
fi

echo
echo "Registration verified. Restart your Claude Code session to pick up .mcp.json."
echo "Then run: /mcp   (the merlin-targetgen entry should be listed as 'connected')"
