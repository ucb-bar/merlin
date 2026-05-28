"""Baseline-comparison tests.

The deterministic ``classify_inventory`` is the regression baseline for
``propose_modifications``. Two contracts are gated here:

1. **Self-consistency**: feeding ``propose_modifications`` the *exact*
   classification ``classify_inventory`` would have returned must produce
   the same modification_map as ``get_modification_map``. Drift between
   the two paths is a bug.

2. **Divergence is reported, not silent**: when the agent claims something
   different from the baseline, ``baseline_comparison`` records the
   delta with non-zero counts and a ``primary_integration_match=False``.
   This is the metric the paper uses to compare arms.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import pytest  # noqa: E402
from targetgen.baseline import (  # noqa: E402
    BaselineComparison,
    compare_to_deterministic,
)
from targetgen.intake import build_source_inventory, classify_inventory  # noqa: E402
from targetgen_mcp import dispatch_tool  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
ALL_FIXTURES = (
    "external_mlir_cuda_tile",
    "chipyard_gemmini_rocc",
    "radiance_gluon_gpu",
    "fft_generator_mmio",
)


# ---------------------------------------------------------------------------
# Self-consistency: agent-driven path must reproduce deterministic output
# when given the same input.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_self_consistency_modmap_matches_deterministic(fixture: str) -> None:
    """propose_modifications, given the deterministic classification, must
    return the same modification_map as the deterministic path. This pins
    the agent-driven path's behavior to the regression-tested baseline."""
    src = FIXTURES / fixture
    inv = build_source_inventory(target=fixture, sources=[src])
    classification = classify_inventory(inv)

    # Agent path: feed the deterministic classification back through.
    agent = dispatch_tool(
        "targetgen_propose_modifications",
        {
            "target_name": fixture,
            "source_paths": [str(src)],
            "targetgen_styles": list(classification.targetgen_styles),
            "source_styles": list(classification.source_styles),
            "primary_integration": classification.primary_integration,
            "confidence": classification.confidence,
        },
    )

    # Deterministic path: build a draft, ask for the modification map.
    deterministic = dispatch_tool(
        "targetgen_create_capability_draft",
        {
            "target_name": fixture,
            "source_paths": [str(src)],
            "out_path": str((src / "_self_consistency_draft.yaml").resolve()),
            "overwrite": True,
        },
    )
    det_modmap = dispatch_tool(
        "targetgen_get_modification_map",
        {"capability_path": deterministic["capability_path"]},
    )
    Path(deterministic["capability_path"]).unlink(missing_ok=True)

    agent_stages = {s["stage"]: s for s in agent["modification_map"]["stages"]}
    det_stages = {s["stage"]: s for s in det_modmap["stages"]}
    assert agent_stages.keys() == det_stages.keys()
    for stage_name in agent_stages:
        a, d = agent_stages[stage_name], det_stages[stage_name]
        assert a["applies"] == d["applies"], stage_name
        assert sorted(a["write_paths"]) == sorted(d["write_paths"]), (
            f"{fixture}/{stage_name}: agent and baseline write_paths diverged.\n"
            f"agent: {a['write_paths']}\nbaseline: {d['write_paths']}"
        )

    cmp = agent["baseline_comparison"]
    assert cmp["primary_integration_match"] is True
    assert cmp["targetgen_styles_agreement"] == 1.0
    assert cmp["source_styles_agreement"] == 1.0
    assert cmp["overall_write_path_jaccard"] == 1.0


# ---------------------------------------------------------------------------
# Divergence: when the agent claims something different, the comparison
# must report it.
# ---------------------------------------------------------------------------


def test_divergent_claim_is_visible_in_comparison() -> None:
    """Cuda_tile fixture's deterministic primary is ``post_global_plugin``;
    claiming ``runtime_hal`` instead must surface as primary_match=False
    and non-zero stage deltas."""
    src = FIXTURES / "external_mlir_cuda_tile"
    result = dispatch_tool(
        "targetgen_propose_modifications",
        {
            "target_name": "agent_cuda_misclass",
            "source_paths": [str(src)],
            "targetgen_styles": ["runtime_hal"],
            "source_styles": [],
            "primary_integration": "runtime_hal",
        },
    )
    cmp = result["baseline_comparison"]
    assert cmp["primary_integration_match"] is False
    assert cmp["agent_primary_integration"] == "runtime_hal"
    assert cmp["baseline_primary_integration"] != "runtime_hal"
    # At least one stage must show a write-path delta.
    diffs = [s for s in cmp["stage_deltas"] if s["only_in_agent"] or s["only_in_baseline"]]
    assert diffs, "expected stage deltas when classifications diverge"


# ---------------------------------------------------------------------------
# compare_to_deterministic as a library function (no MCP)
# ---------------------------------------------------------------------------


def test_compare_to_deterministic_returns_structured_summary() -> None:
    src = FIXTURES / "radiance_gluon_gpu"
    cmp = compare_to_deterministic(
        target_name="probe",
        source_paths=[src],
        agent_targetgen_styles=["runtime_hal"],
        agent_source_styles=["chipyard_generator"],
        agent_primary_integration="runtime_hal",
    )
    assert isinstance(cmp, BaselineComparison)
    # Without an agent_modification_map, deltas stay empty but classification
    # comparison is populated.
    assert cmp.stage_deltas == []
    assert 0.0 <= cmp.targetgen_styles_agreement <= 1.0
    assert 0.0 <= cmp.source_styles_agreement <= 1.0
    assert cmp.baseline_targetgen_styles  # classifier returned something


def test_baseline_comparison_jaccard_is_reflexive() -> None:
    """Jaccard of identical sets is 1.0; disjoint sets is 0.0."""
    src = FIXTURES / "radiance_gluon_gpu"
    inv = build_source_inventory(target="probe", sources=[src])
    classification = classify_inventory(inv)

    same = compare_to_deterministic(
        target_name="probe",
        source_paths=[src],
        agent_targetgen_styles=list(classification.targetgen_styles),
        agent_source_styles=list(classification.source_styles),
        agent_primary_integration=classification.primary_integration,
    )
    assert same.targetgen_styles_agreement == 1.0
    assert same.source_styles_agreement == 1.0

    different = compare_to_deterministic(
        target_name="probe",
        source_paths=[src],
        agent_targetgen_styles=["llvm_ukernel"],  # not what the classifier would say
        agent_source_styles=["llvm_backend_extension"],
        agent_primary_integration="llvm_ukernel",
    )
    # The radiance fixture's classifier never returns llvm_ukernel as primary,
    # so jaccard must be < 1.0 on at least one axis.
    assert different.targetgen_styles_agreement < 1.0 or different.source_styles_agreement < 1.0
