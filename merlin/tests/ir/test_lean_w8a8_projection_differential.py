"""Execute complete generalized W8A8 outputs against Lean, native MLIR and the ordered recipe."""

from __future__ import annotations

import pytest
from lean_w8a8_projection_cases import CASES, changed_source_pair
from test_lean_w8a8_bounded_chain import check_chain, lowered
from test_lean_w8a8_projection import BOUNDS
from test_lean_w8a8_views import compare_outputs, words

pytestmark = pytest.mark.usefixtures("lean_oracle")


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("integer_left", [False, True])
def test_complete_projection_output_matches_ordered_recipe_and_mlir(
    tmp_path, lean_tools, fidelity_tools, name, integer_left
):
    import numpy as np

    case = CASES[name]
    before = case.source(integer_left=integer_left)
    report, documents, _ = check_chain([before, lowered(before)], BOUNDS, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report
    rng = np.random.default_rng(20261026)
    samples, expected = [], []
    for zero in (True, False):
        if integer_left:
            left = (
                np.zeros(case.shape(case.left), dtype=np.int8)
                if zero
                else rng.choice(np.array([-128, -1, 0, 1, 127], dtype=np.int8), size=case.shape(case.left))
            )
            left_words = [str(int(value) % 256) for value in left.flat]
        else:
            left = (
                np.zeros(case.shape(case.left), dtype=np.float32)
                if zero
                else rng.uniform(-3, 3, case.shape(case.left)).astype(np.float32)
            )
            left_words = words(left)
        right = (
            np.zeros(case.shape(case.right), dtype=np.float32)
            if zero
            else rng.uniform(-5, 5, case.shape(case.right)).astype(np.float32)
        )
        samples.append([left_words, words(right)])
        expected.append(words(case.recipe(left, right, integer_left=integer_left)))
    compare_outputs(documents[-1], samples, expected, documents[-1].parent, fidelity_tools)


def test_source_map_change_has_observable_numerical_effects(tmp_path, lean_tools, fidelity_tools):
    import numpy as np

    first, second = changed_source_pair("left")
    rng = np.random.default_rng(20261027)
    left = rng.uniform(-3, 3, first.shape(first.left)).astype(np.float32)
    right = rng.uniform(-5, 5, first.shape(first.right)).astype(np.float32)
    expected = [words(case.recipe(left, right)) for case in (first, second)]
    assert expected[0] != expected[1]
    for index, case in enumerate((first, second)):
        work = tmp_path / str(index)
        work.mkdir()
        before = case.source()
        report, documents, _ = check_chain([before, lowered(before)], BOUNDS, work, lean_tools)
        assert report["conditional_initialized_policy_refinement"], report
        compare_outputs(
            documents[-1], [[words(left), words(right)]], [expected[index]], documents[-1].parent, fidelity_tools
        )
