"""The optimization loop must be scored against a MEASURED destination, not only its last revision.

Each negative case is a mutation of the positive one, so a gap that cannot report a gap is visible.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.perf.target_reference import (
    REFERENCE_LEDGER_NAME,
    ReferenceError,
    estimate_cycles,
    find_reference,
    load_references,
    score_against_reference,
    structural_gap,
)

DESIGN = "FireSimGemminiRocketConfig"

_REFERENCES = {
    "model_fast": {
        "model": "model", "design": DESIGN, "status": "achieved",
        "measured": {"whole_model_cycles": 1_000_000, "host_share_lower_bound": 0.93,
                     "cycles_per_host_operation_anchor": 2.5},
        "emitted_structure": {"commands": 10, "mesh_regions": 4, "host_lane_regions": 6,
                              "weight_prepack_recipes": 4, "kernel_abi_args": 20,
                              "opcodes": {"PACK": 4, "CONV2D": 5, "COMMIT": 1}},
        "caveats": ["the anchor is a whole-window average"],
    },
    "model_slow": {
        "model": "model", "design": DESIGN, "status": "achieved",
        "measured": {"whole_model_cycles": 2_000_000},
    },
    "model_other_design": {
        "model": "model", "design": "another_design", "status": "achieved",
        "measured": {"whole_model_cycles": 10},
    },
    "model_superseded": {
        "model": "model", "design": DESIGN, "status": "superseded",
        "measured": {"whole_model_cycles": 1},
    },
}


def _buffer(*, commands=10, mesh=4, host=6, prepack=4, args=20,
            opcodes=("PACK",) * 4 + ("CONV2D",) * 5 + ("COMMIT",)):
    return {
        "commands": [{"opcode": opcode} for opcode in opcodes][:commands],
        "kernel_abi": {"args": [{"tensor": f"t{i}"} for i in range(args)]},
        "params": {"mesh_regions": [{}] * mesh, "host_lane_regions": [{}] * host,
                   "weight_prepack_recipes": [{}] * prepack},
    }


def test_the_tracked_ledger_parses_and_declares_the_reference_we_target() -> None:
    """The shipped ledger is the destination; a malformed one silently removes it."""
    references = load_references()
    name, reference = find_reference("resnet50", DESIGN, references=references)
    assert reference["measured"]["whole_model_cycles"] == 1_316_619_699
    assert reference["emitted_structure"]["commands"] == 109
    assert reference["emitted_structure"]["opcodes"]["CONV2D"] == 53
    raw = yaml.safe_load((merlin_dir() / REFERENCE_LEDGER_NAME).read_text())
    assert raw["references"][name]["design"] == DESIGN


def test_the_best_achieved_reference_is_the_fewest_cycles() -> None:
    name, reference = find_reference("model", DESIGN, references=_REFERENCES)
    assert name == "model_fast"
    assert reference["measured"]["whole_model_cycles"] == 1_000_000


def test_a_superseded_point_is_not_a_target() -> None:
    """`model_superseded` has the fewest cycles of all and must still never be selected."""
    name, _ = find_reference("model", DESIGN, references=_REFERENCES)
    assert name != "model_superseded"


def test_a_design_mismatch_is_refused_not_approximated() -> None:
    with pytest.raises(ReferenceError) as caught:
        find_reference("model", "another_design_entirely", references=_REFERENCES)
    assert "not comparable across designs" in str(caught.value)


def test_an_unknown_model_is_refused_and_says_what_is_known() -> None:
    with pytest.raises(ReferenceError) as caught:
        find_reference("no_such_model", DESIGN, references=_REFERENCES)
    assert "no_such_model" in str(caught.value) and "model" in str(caught.value)


def test_a_matching_candidate_reports_every_gap_as_zero() -> None:
    gap = structural_gap(_buffer(), _REFERENCES["model_fast"])
    assert gap["matches"] is True
    assert gap["unmatched"] == []
    assert all(row["gap"] == 0 for row in gap["fields"].values())


def test_gaps_are_signed_and_name_the_field() -> None:
    """The direction is the whole point: -4 mesh regions is a different job than +4."""
    gap = structural_gap(_buffer(mesh=0, host=20), _REFERENCES["model_fast"])
    assert gap["matches"] is False
    assert gap["fields"]["mesh_regions"]["gap"] == -4
    assert gap["fields"]["host_lane_regions"]["gap"] == +14
    assert "mesh_regions" in gap["unmatched"]


def test_a_missing_opcode_class_is_a_named_gap() -> None:
    gap = structural_gap(_buffer(opcodes=("PACK",) * 4 + ("COMMIT",)), _REFERENCES["model_fast"])
    assert gap["opcodes"]["CONV2D"]["gap"] == -5
    assert "opcode:CONV2D" in gap["unmatched"]


def test_an_opcode_the_reference_never_emits_is_also_reported() -> None:
    gap = structural_gap(_buffer(opcodes=("MYSTERY",) * 10), _REFERENCES["model_fast"])
    assert gap["opcodes"]["MYSTERY"]["reference"] == 0
    assert gap["opcodes"]["MYSTERY"]["gap"] == 10


def test_a_reference_without_structure_yields_no_verdict_rather_than_a_pass() -> None:
    gap = structural_gap(_buffer(), _REFERENCES["model_slow"])
    assert gap["matches"] is None
    assert gap["status"] == "reference_declares_no_structure"


def test_a_relative_cycle_estimate_is_anchored_on_the_measurement() -> None:
    """Half the host operations of a reference whose host share is 0.93 -> 1 - 0.93/2 of cycles."""
    estimate = estimate_cycles(500, _REFERENCES["model_fast"],
                               baseline_host_dynamic_operations=1000)
    assert estimate["basis"] == "relative_to_reference_measurement"
    assert estimate["estimated_whole_model_cycles"] == round(1_000_000 * (0.07 + 0.93 * 0.5))
    assert estimate["cycles_above_reference"] < 0


def test_an_equal_op_count_estimates_the_reference_exactly() -> None:
    estimate = estimate_cycles(1000, _REFERENCES["model_fast"],
                               baseline_host_dynamic_operations=1000)
    assert estimate["estimated_whole_model_cycles"] == 1_000_000
    assert estimate["cycles_above_reference"] == 0
    assert estimate["fraction_of_reference"] == 1.0


def test_a_reference_with_no_anchor_says_so_instead_of_guessing() -> None:
    assert estimate_cycles(10, _REFERENCES["model_slow"])["status"] == (
        "reference_carries_no_cycle_anchor")


def test_every_cycle_estimate_is_licensed_as_an_estimate() -> None:
    """A structural match must never be quotable as a measurement."""
    estimate = estimate_cycles(1000, _REFERENCES["model_fast"],
                               baseline_host_dynamic_operations=1000)
    assert estimate["status"] == "estimated"
    assert "never a measurement" in estimate["licence"]
    gap = structural_gap(_buffer(), _REFERENCES["model_fast"])
    assert "not that it runs as fast" in gap["licence"]


def test_the_whole_record_carries_reference_structure_and_caveats() -> None:
    record = score_against_reference(_buffer(), model="model", design=DESIGN,
                                     host_dynamic_operations=1000,
                                     baseline_host_dynamic_operations=1000,
                                     references=_REFERENCES)
    assert record["schema"] == "perf_reference_gap_v1"
    assert record["reference"] == "model_fast"
    assert record["structural"]["matches"] is True
    assert record["cycles"]["cycles_above_reference"] == 0
    assert record["caveats"] == ["the anchor is a whole-window average"]
