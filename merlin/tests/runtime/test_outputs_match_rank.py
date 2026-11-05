"""``outputs_match`` compares values in row-major order, not the bracket nesting a console flattens."""

from merlin.runtime.reference import outputs_match


def test_rank_one_and_rank_three_match_their_flattened_console_form():
    assert outputs_match({"y": [1, 2, 3]}, {"y": [[1, 2, 3]]})
    assert outputs_match({"y": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]}, {"y": [[1, 2], [3, 4], [5, 6], [7, 8]]})


def test_values_count_and_roster_still_have_to_agree():
    assert not outputs_match({"y": [1, 2, 3]}, {"y": [[1, 3, 2]]})
    assert not outputs_match({"y": [1, 2, 3]}, {"y": [[1, 2]]})
    assert not outputs_match({"y": [1]}, {"y": [1], "z": [2]})
