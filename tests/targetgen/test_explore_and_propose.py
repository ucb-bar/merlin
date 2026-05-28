"""Tests for the agent-driven exploration / proposal flow.

Covers ``targetgen_explore_target`` (raw evidence emission) and
``targetgen_propose_modifications`` (agent-supplied classification with
audit-against-evidence).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from targetgen.audit import audit_claim  # noqa: E402
from targetgen.explore import explore_target  # noqa: E402
from targetgen.intake import build_source_inventory  # noqa: E402
from targetgen_mcp import ToolError, dispatch_tool  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# explore_target
# ---------------------------------------------------------------------------


def test_explore_returns_directory_summary_and_findings() -> None:
    report = explore_target("smoke", [FIXTURES / "radiance_gluon_gpu"])
    assert report.target == "smoke"
    assert report.directory_summaries
    assert report.directory_summaries[0].total_files > 0
    assert report.findings, "expected at least one scanner finding on radiance fixture"
    # findings_by_kind aggregates the findings list
    total = sum(report.findings_by_kind.values())
    assert total == len(report.findings)


def test_explore_points_at_readme_when_present() -> None:
    report = explore_target("smoke", [FIXTURES / "radiance_gluon_gpu"])
    if report.key_files.readme is not None:
        assert Path(report.key_files.readme).is_file()


def test_explore_emits_actionable_next_steps() -> None:
    report = explore_target("smoke", [FIXTURES / "radiance_gluon_gpu"])
    joined = " ".join(report.next_steps)
    assert "targetgen_propose_modifications" in joined
    # The agent is told to use Read/Grep, not given a file dump.
    assert "Read" in joined or "Grep" in joined


def test_explore_handles_empty_dir(tmp_path: Path) -> None:
    report = explore_target("empty", [tmp_path])
    assert report.findings == []
    assert report.findings_by_kind == {}
    # Even an empty target gets next-step guidance.
    assert any("classify" in s.lower() or "propose" in s.lower() for s in report.next_steps)


# ---------------------------------------------------------------------------
# audit_claim (unit-level — no MCP)
# ---------------------------------------------------------------------------


def test_audit_supports_well_evidenced_claim() -> None:
    inv = build_source_inventory(target="smoke", sources=[FIXTURES / "radiance_gluon_gpu"])
    report = audit_claim(
        claimed_targetgen_styles=["runtime_hal"],
        claimed_source_styles=["chipyard_generator"] if "chipyard_project" in inv.detected_source_kinds else [],
        inventory=inv,
    )
    # runtime_hal is broadly supported by HAL/RTL/Chipyard kinds — at least one
    # supporting finding should fire on the radiance fixture.
    supported = [f for f in report.findings if f.conclusion == "supported"]
    assert supported, f"expected supported findings, got {report.findings}"


def test_audit_flags_unsupported_claim() -> None:
    inv = build_source_inventory(target="smoke", sources=[FIXTURES / "external_mlir_cuda_tile"])
    # The cuda_tile fixture has MLIR but not LLVM intrinsic TableGen, so claiming
    # llvm_ukernel should be unsupported.
    report = audit_claim(
        claimed_targetgen_styles=["llvm_ukernel"],
        claimed_source_styles=[],
        inventory=inv,
    )
    unsupported = [f for f in report.findings if f.conclusion in {"unsupported", "contradicted"}]
    assert unsupported, f"expected unsupported finding, got {report.findings}"
    assert report.overall_status in {"warn", "fail"}


def test_audit_rejects_unknown_targetgen_style() -> None:
    inv = build_source_inventory(target="smoke", sources=[FIXTURES / "radiance_gluon_gpu"])
    report = audit_claim(
        claimed_targetgen_styles=["nonexistent_style"],
        claimed_source_styles=[],
        inventory=inv,
    )
    assert report.overall_status == "fail"
    assert any(f.severity == "error" and "Unknown TargetGen style" in f.note for f in report.findings)


# ---------------------------------------------------------------------------
# MCP dispatch — explore_target + propose_modifications
# ---------------------------------------------------------------------------


def test_dispatch_explore_target_returns_evidence() -> None:
    result = dispatch_tool(
        "targetgen_explore_target",
        {
            "target_name": "smoke",
            "source_paths": [str(FIXTURES / "radiance_gluon_gpu")],
        },
    )
    assert result["target"] == "smoke"
    assert "directory_summaries" in result
    assert "findings_by_kind" in result
    assert "next_steps" in result
    # findings are returned as full records (path/kind/evidence/confidence)
    assert all({"kind", "path", "evidence", "confidence"} <= set(f) for f in result["findings"])


def test_dispatch_propose_modifications_returns_modmap_and_audit(tmp_path: Path) -> None:
    out = tmp_path / "draft.yaml"
    result = dispatch_tool(
        "targetgen_propose_modifications",
        {
            "target_name": "agent_radiance",
            "source_paths": [str(FIXTURES / "radiance_gluon_gpu")],
            "targetgen_styles": ["runtime_hal", "post_global_plugin"],
            "source_styles": ["chipyard_generator"],
            "primary_integration": "runtime_hal",
            "rationale": "Reads as a Chipyard-attached accelerator with HAL needs.",
            "confidence": 0.7,
            "out_path": str(out),
        },
    )
    assert result["target"] == "agent_radiance"
    assert result["agent_classification"]["primary_integration"] == "runtime_hal"
    assert result["modification_map"]["target"] == "agent_radiance"
    assert len(result["modification_map"]["stages"]) == 9
    assert result["audit"]["overall_status"] in {"pass", "warn", "fail"}
    assert result["capability_path"] == str(out)
    assert out.is_file()


def test_propose_modifications_rejects_primary_not_in_styles() -> None:
    with pytest.raises(ToolError, match="primary_integration"):
        dispatch_tool(
            "targetgen_propose_modifications",
            {
                "target_name": "x",
                "source_paths": [str(FIXTURES / "radiance_gluon_gpu")],
                "targetgen_styles": ["runtime_hal"],
                "primary_integration": "post_global_plugin",
            },
        )


def test_propose_modifications_rejects_unknown_targetgen_style() -> None:
    with pytest.raises(ToolError, match="unknown targetgen_styles"):
        dispatch_tool(
            "targetgen_propose_modifications",
            {
                "target_name": "x",
                "source_paths": [str(FIXTURES / "radiance_gluon_gpu")],
                "targetgen_styles": ["bogus_style"],
                "primary_integration": "bogus_style",
            },
        )


def test_propose_modifications_audit_flags_evidence_gap() -> None:
    """Claim a style with no supporting evidence → audit should warn."""
    result = dispatch_tool(
        "targetgen_propose_modifications",
        {
            "target_name": "agent_cuda_misclass",
            "source_paths": [str(FIXTURES / "external_mlir_cuda_tile")],
            # Cuda_tile fixture has MLIR but no HAL/runtime evidence
            # — claiming runtime_hal should fail audit.
            "targetgen_styles": ["runtime_hal"],
            "primary_integration": "runtime_hal",
        },
    )
    assert result["audit"]["overall_status"] in {"warn", "fail"}
    findings = result["audit"]["findings"]
    assert any(f["conclusion"] in {"unsupported", "contradicted"} and "runtime_hal" in f["claim"] for f in findings)


def test_propose_modifications_refuses_overwrite_by_default(tmp_path: Path) -> None:
    out = tmp_path / "draft.yaml"
    out.write_text("preexisting\n")
    with pytest.raises(ToolError, match="overwrite=true"):
        dispatch_tool(
            "targetgen_propose_modifications",
            {
                "target_name": "x",
                "source_paths": [str(FIXTURES / "radiance_gluon_gpu")],
                "targetgen_styles": ["runtime_hal"],
                "primary_integration": "runtime_hal",
                "out_path": str(out),
            },
        )
    assert out.read_text() == "preexisting\n"
