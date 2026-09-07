"""Command-buffer diagnostics expose representations while refusing invented occupancy."""
from merlin.perf.command_buffer_diagnostics import representation_activity


def test_representation_directives_and_unknown_timeline_are_both_explicit() -> None:
    audit = representation_activity({
        "tensors": {
            "weight": {"shape": [8, 8], "dtype": "i8", "role": "weight"},
            "output": {"shape": [8, 8], "dtype": "i32", "role": "output",
                       "physical": {"unstack": 2}},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "weight", "dst": "resident"},
             "attributes": {"layout": "packed"}},
            {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "output"},
             "attributes": {"output_dtype": "i32", "epilogue": ["relu"]}},
        ],
    })

    assert audit["command_counts"] == {"COMMIT": 1, "RES_PACK": 1}
    assert audit["representation_directive_count"] == 2
    assert audit["tensors"]["output"]["physical_status"] == "declared"
    assert audit["tensors"]["weight"]["physical_status"] == "UNKNOWN"
    assert audit["occupancy"]["status"] == "UNKNOWN"
    assert audit["emitted_encoding_transitions"]["count"] is None


def test_declined_whole_model_carries_placement_without_becoming_zero_work() -> None:
    audit = representation_activity({
        "tensors": {}, "commands": [],
        "declined": {"op": "host_lane", "reason": "cannot roll the full model"},
        "params": {"lane_placement": [
            {"region": "a", "family": "contraction", "lane": "array"},
            {"region": "b", "family": "elementwise", "lane": "vector"},
            {"region": "c", "family": "contraction", "lane": "array"},
        ]},
    })

    assert audit["lowering"]["status"] == "declined"
    assert audit["placement"]["lane_counts"] == {"array": 2, "vector": 1}
    assert audit["placement"]["adjacent_lane_transitions"] == 2
