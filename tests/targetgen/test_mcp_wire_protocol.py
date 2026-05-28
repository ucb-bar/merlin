"""Real MCP-over-stdio roundtrip test.

In-process ``dispatch_tool`` tests live in ``test_mcp_tools.py``; they don't
exercise the JSON-RPC wire layer, the async server runtime, or the stdio
streams. This test spawns ``./merlin targetgen mcp`` as a real subprocess and
talks to it through ``mcp.client.stdio`` — the same path Claude Code uses.

The subprocess is launched directly via the conda-env Python (skipping the
``./merlin`` bash wrapper) so the test does not depend on a writable
``.venv``. The wrapper itself is exercised separately in Tier 2.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from conftest import REPO_ROOT
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

EXAMPLES = REPO_ROOT / "target_specs" / "examples"
RADIANCE_CAPABILITY = EXAMPLES / "radiance_muon" / "capability.yaml"
FIXTURES = Path(__file__).parent / "fixtures"

EXPECTED_TOOL_NAMES = frozenset(
    {
        "targetgen_ingest_source",
        "targetgen_classify_target",
        "targetgen_create_capability_draft",
        "targetgen_plan_target",
        "targetgen_get_modification_map",
        "targetgen_get_allowed_patch_surfaces",
        "targetgen_get_validation_commands",
        "targetgen_list_pipeline_stages",
        "targetgen_explore_target",
        "targetgen_propose_modifications",
    }
)


def _server_params() -> StdioServerParameters:
    """Launch the MCP server via the current Python with PYTHONPATH=tools.

    Skips ``./merlin`` so the test does not require a writable .venv. The
    wrapper itself is exercised in Tier 2.
    """
    if shutil.which("python") is None and not Path(sys.executable).exists():
        pytest.skip("no usable Python interpreter for subprocess")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "tools") + os.pathsep + env.get("PYTHONPATH", "")
    return StdioServerParameters(
        command=sys.executable,
        args=[str(REPO_ROOT / "tools" / "merlin.py"), "targetgen", "mcp"],
        env=env,
    )


async def _roundtrip() -> dict:
    """Single async roundtrip; returns a dict of test results for assertion."""
    params = _server_params()
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_response = await session.list_tools()
            tool_names = {t.name for t in tools_response.tools}

            stages_call = await session.call_tool("targetgen_list_pipeline_stages", {})

            modmap_call = await session.call_tool(
                "targetgen_get_modification_map",
                {"capability_path": str(RADIANCE_CAPABILITY)},
            )

            patch_surface_call = await session.call_tool(
                "targetgen_get_allowed_patch_surfaces",
                {
                    "capability_path": str(RADIANCE_CAPABILITY),
                    "stage": "hal_driver",
                },
            )

            error_call = await session.call_tool(
                "targetgen_get_allowed_patch_surfaces",
                {
                    "capability_path": str(RADIANCE_CAPABILITY),
                    "stage": "not_a_stage",
                },
            )

            ingest_call = await session.call_tool(
                "targetgen_ingest_source",
                {
                    "target_name": "wire_smoke",
                    "source_paths": [str(FIXTURES / "external_mlir_cuda_tile")],
                },
            )

            return {
                "tool_names": tool_names,
                "tools_with_schemas": [{"name": t.name, "input_schema": t.inputSchema} for t in tools_response.tools],
                "stages_text": _first_text(stages_call),
                "modmap_text": _first_text(modmap_call),
                "patch_surface_text": _first_text(patch_surface_call),
                "error_text": _first_text(error_call),
                "ingest_text": _first_text(ingest_call),
            }


def _first_text(call_result) -> str:
    for content in call_result.content:
        if getattr(content, "type", None) == "text":
            return content.text
    pytest.fail(f"no text content returned: {call_result!r}")
    return ""  # pragma: no cover


@pytest.fixture(scope="module")
def wire_results() -> dict:
    return asyncio.run(_roundtrip())


def test_initialize_and_list_tools(wire_results: dict) -> None:
    assert wire_results["tool_names"] == EXPECTED_TOOL_NAMES


def test_each_tool_has_input_schema(wire_results: dict) -> None:
    for entry in wire_results["tools_with_schemas"]:
        schema = entry["input_schema"]
        assert isinstance(schema, dict), entry
        assert schema.get("type") == "object", entry


def test_list_pipeline_stages_via_wire(wire_results: dict) -> None:
    payload = json.loads(wire_results["stages_text"])
    assert len(payload["stages"]) == 9
    assert payload["stages"][-1] == "hal_driver"


def test_modification_map_via_wire(wire_results: dict) -> None:
    payload = json.loads(wire_results["modmap_text"])
    assert payload["target"] == "radiance_muon"
    assert payload["primary_integration"] == "runtime_hal"
    assert len(payload["stages"]) == 9


def test_allowed_patch_surfaces_via_wire(wire_results: dict) -> None:
    payload = json.loads(wire_results["patch_surface_text"])
    assert payload["stage"] == "hal_driver"
    assert payload["applies"] is True
    assert any("runtime/src/iree/hal/drivers/radiance_muon" in p for p in payload["allowed_write_paths"])
    assert "third_party/iree_bar/" in payload["forbidden_unless_approved"]


def test_unknown_stage_returns_error_via_wire(wire_results: dict) -> None:
    """Bad stage is rejected by either the input schema enum or the tool body."""
    text = wire_results["error_text"]
    assert "not_a_stage" in text, text
    assert text.startswith("Input validation error") or text.startswith("ToolError:"), text


def test_ingest_via_wire(wire_results: dict) -> None:
    payload = json.loads(wire_results["ingest_text"])
    assert payload["target"] == "wire_smoke"
    assert "mlir_dialect" in payload["detected_source_kinds"]
