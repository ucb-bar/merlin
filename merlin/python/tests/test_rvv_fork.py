"""Fork generator faithfulness + versioned minting/lineage.

The generator must be faithful (re-rendering hand_v0 knobs == its verbatim schedule), or a fork
would differ from its parent by more than the intended knob. Minting must produce a new
lineage-stamped package (never mutate the parent).
"""
import os

from merlin.rvvgen import load_rvv_package
from merlin.rvvgen.fork import mint_run_id
from merlin.rvvgen.from_strategy import render_schedule, mint_fork

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HAND_V0 = os.path.join(ROOT, "generated_targets", "rvv", "hand_v0")


def test_render_reproduces_hand_v0_verbatim():
    pkg = load_rvv_package(HAND_V0)
    assert render_schedule(pkg.knobs) == pkg.schedule_text


def test_run_id_naming_convention():
    assert mint_run_id("rvv", 2, 3, "20260101T000000") == "rvv_tuned_v2_d3_20260101T000000"


def test_contraction_strategy_knob_changes_only_that_line():
    pkg = load_rvv_package(HAND_V0)
    knobs = dict(pkg.knobs)
    knobs["contraction_strategy"] = "outerproduct"
    out = render_schedule(knobs)
    assert 'lower_contraction lowering_strategy = "outerproduct"' in out
    # the tile/vectorize lines are untouched
    assert "tile_sizes [4, 8, 1]" in out
    assert out.count("transform.structured.vectorize") == pkg.schedule_text.count(
        "transform.structured.vectorize")


def test_mint_fork_writes_lineage_and_preserves_parent(tmp_path):
    fork_dir = mint_fork(
        HAND_V0, {"contraction_strategy": "outerproduct"},
        version=1, depth=1, timestamp="20260101T000000",
        source_evidence=["scalar_broadcast_fma", "xnnpack:f32-gemm-1x4v"],
        lever="lowering_pattern", out_root=tmp_path)
    assert fork_dir.name == "rvv_tuned_v1_d1_20260101T000000"
    fork = load_rvv_package(fork_dir)
    assert fork.knobs["contraction_strategy"] == "outerproduct"
    assert fork.manifest["lineage"]["parent_run_id"] == "hand_v0"
    assert fork.manifest["lineage"]["lever"] == "lowering_pattern"
    assert "scalar_broadcast_fma" in fork.manifest["lineage"]["source_evidence"]
    # parent untouched (different knobs, still verbatim)
    parent = load_rvv_package(HAND_V0)
    assert "contraction_strategy" not in parent.knobs
