from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from targetgen_mcp import (  # noqa: E402
    ToolError,
    build_server,
    dispatch_tool,
    list_tool_definitions,
)
from targetgen_mcp.tools import TOOL_REGISTRY  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = REPO_ROOT / "target_specs" / "examples"


def test_tool_registry_lists_expected_tools() -> None:
    names = {t.name for t in list_tool_definitions()}
    assert names == {
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


def test_tool_input_schemas_are_valid_dicts() -> None:
    for tool in TOOL_REGISTRY:
        assert tool.input_schema["type"] == "object"
        assert "properties" in tool.input_schema


def test_list_pipeline_stages_returns_nine_stages() -> None:
    result = dispatch_tool("targetgen_list_pipeline_stages", {})
    assert len(result["stages"]) == 9
    assert result["stages"][-1] == "hal_driver"


def test_ingest_source_returns_inventory() -> None:
    result = dispatch_tool(
        "targetgen_ingest_source",
        {
            "target_name": "fake_radiance",
            "source_paths": [str(FIXTURES / "radiance_gluon_gpu")],
        },
    )
    assert result["target"] == "fake_radiance"
    assert "hal_driver_source" in result["detected_source_kinds"]


def test_classify_target_from_source_paths() -> None:
    result = dispatch_tool(
        "targetgen_classify_target",
        {
            "target_name": "fake_radiance",
            "source_paths": [str(FIXTURES / "radiance_gluon_gpu")],
        },
    )
    assert result["primary_integration"] == "runtime_hal"
    assert "gpu_codegen_stack" in result["source_styles"]


def test_classify_target_requires_inventory_or_sources() -> None:
    with pytest.raises(ToolError, match="inventory_path"):
        dispatch_tool("targetgen_classify_target", {"target_name": "x"})


def test_plan_target_returns_support_plan() -> None:
    result = dispatch_tool(
        "targetgen_plan_target",
        {"capability_path": str(EXAMPLES / "radiance_muon" / "capability.yaml")},
    )
    assert result["target"] == "radiance_muon"
    assert "runtime_hal" in result["integration_styles"]


def test_get_modification_map_for_radiance() -> None:
    result = dispatch_tool(
        "targetgen_get_modification_map",
        {"capability_path": str(EXAMPLES / "radiance_muon" / "capability.yaml")},
    )
    assert result["primary_integration"] == "runtime_hal"
    assert len(result["stages"]) == 9


def test_get_allowed_patch_surfaces_returns_hal_writes() -> None:
    result = dispatch_tool(
        "targetgen_get_allowed_patch_surfaces",
        {
            "capability_path": str(EXAMPLES / "radiance_muon" / "capability.yaml"),
            "stage": "hal_driver",
        },
    )
    assert result["applies"] is True
    assert any("runtime/src/iree/hal/drivers/radiance_muon" in p for p in result["allowed_write_paths"])
    # The HAL stage is not editing IREE submodules so they remain forbidden.
    assert "third_party/iree_bar/" in result["forbidden_unless_approved"]


def test_get_allowed_patch_surfaces_rejects_unknown_stage() -> None:
    with pytest.raises(ToolError, match="Unknown stage"):
        dispatch_tool(
            "targetgen_get_allowed_patch_surfaces",
            {
                "capability_path": str(EXAMPLES / "radiance_muon" / "capability.yaml"),
                "stage": "not_a_stage",
            },
        )


def test_get_validation_commands_use_merlin_wrapper() -> None:
    result = dispatch_tool(
        "targetgen_get_validation_commands",
        {"capability_path": str(EXAMPLES / "radiance_muon" / "capability.yaml")},
    )
    for entry in result["validation_commands"]:
        for cmd in entry["commands"]:
            assert cmd.startswith("./merlin "), cmd


def test_create_capability_draft_writes_loadable_yaml(tmp_path: Path) -> None:
    out = tmp_path / "draft.yaml"
    result = dispatch_tool(
        "targetgen_create_capability_draft",
        {
            "target_name": "fake_radiance",
            "source_paths": [str(FIXTURES / "radiance_gluon_gpu")],
            "out_path": str(out),
        },
    )
    assert result["target"] == "fake_radiance"
    assert result["capability_path"] == str(out)
    assert result["loader_validated"] is True
    assert "next_step" in result
    assert out.exists()
    # Round-trip through the loader to confirm the contract holds outside
    # the MCP tool too.
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from targetgen import load_capability_spec  # noqa: E402

    caps = load_capability_spec(out)
    assert caps.identity.name == "fake_radiance"


def test_create_capability_draft_refuses_overwrite_by_default(tmp_path: Path) -> None:
    out = tmp_path / "draft.yaml"
    out.write_text("preexisting content\n")
    with pytest.raises(ToolError, match="overwrite=true"):
        dispatch_tool(
            "targetgen_create_capability_draft",
            {
                "target_name": "x",
                "source_paths": [str(FIXTURES / "radiance_gluon_gpu")],
                "out_path": str(out),
            },
        )
    # File untouched
    assert out.read_text() == "preexisting content\n"


def test_create_capability_draft_overwrite_true_replaces_file(tmp_path: Path) -> None:
    out = tmp_path / "draft.yaml"
    out.write_text("preexisting content\n")
    result = dispatch_tool(
        "targetgen_create_capability_draft",
        {
            "target_name": "fake_radiance",
            "source_paths": [str(FIXTURES / "radiance_gluon_gpu")],
            "out_path": str(out),
            "overwrite": True,
        },
    )
    assert result["loader_validated"] is True
    assert "preexisting" not in out.read_text()


def test_create_capability_draft_chains_into_plan_target(tmp_path: Path) -> None:
    out = tmp_path / "draft.yaml"
    dispatch_tool(
        "targetgen_create_capability_draft",
        {
            "target_name": "fake_chain",
            "source_paths": [str(FIXTURES / "radiance_gluon_gpu")],
            "out_path": str(out),
        },
    )
    plan = dispatch_tool("targetgen_plan_target", {"capability_path": str(out)})
    assert plan["target"] == "fake_chain"
    assert plan["integration_styles"]
    modmap = dispatch_tool("targetgen_get_modification_map", {"capability_path": str(out)})
    assert modmap["target"] == "fake_chain"
    assert len(modmap["stages"]) == 9


def test_dispatch_tool_unknown_name_raises() -> None:
    with pytest.raises(ToolError, match="Unknown tool"):
        dispatch_tool("targetgen_nonexistent", {})


def test_build_server_does_not_run() -> None:
    server = build_server()
    # Server is built lazily; we just confirm tool registration succeeded by
    # asking the server to enumerate request handlers.
    assert server is not None
    # The mcp Server stores request handlers on .request_handlers; ensure
    # ListToolsRequest and CallToolRequest are wired.
    handler_names = {cls.__name__ for cls in server.request_handlers}
    assert "ListToolsRequest" in handler_names
    assert "CallToolRequest" in handler_names


def test_mcp_server_does_not_mutate_repo(tmp_path: Path, monkeypatch) -> None:
    """Sanity: dispatching tools never touches repo-tracked paths."""
    # Use a synthetic source so the tools work without writing anywhere.
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "CMakeLists.txt").write_text("add_mlir_dialect(Foo foo)\n")
    result = dispatch_tool(
        "targetgen_ingest_source",
        {"target_name": "selftest", "source_paths": [str(src)]},
    )
    json.dumps(result)  # JSON-serialisable
    # No files should appear outside src/.
    assert {p.name for p in tmp_path.iterdir()} == {"src"}
