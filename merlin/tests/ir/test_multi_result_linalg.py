"""The normalizer must un-parenthesise a linalg result list and touch nothing else.

Two things in the same text spell `->` followed by `(` and are CORRECT as written: every
`affine_map<(d0) -> (d0)>`, and a multi-result `func.func` signature (which xDSL already parses).
Rewriting either would corrupt a valid module, so the discriminator -- an arrow that follows a
region close -- is the property under test, not an implementation detail.
"""
from __future__ import annotations

from merlin.frontends.multi_result_linalg import normalize_multi_result_linalg as normalize


def test_a_parenthesised_result_list_loses_exactly_its_parentheses():
    got, n = normalize("} -> (tensor<1xi64>, tensor<1xi64>)")
    assert got == "} -> tensor<1xi64>, tensor<1xi64>"
    assert n == 1


def test_a_single_result_arrow_is_untouched():
    for src in ("} -> tensor<4xf32>", "} -> i32"):
        assert normalize(src) == (src, 0)


def test_an_affine_map_is_never_rewritten():
    src = ("linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, "
           "affine_map<(d0, d1) -> (d1)>], iterator_types = [\"parallel\"]}")
    assert normalize(src) == (src, 0)


def test_a_multi_result_function_signature_is_never_rewritten():
    src = "func.func @f(%a : i32) -> (i32, i32) {"
    assert normalize(src) == (src, 0)


def test_the_arrow_may_be_separated_from_the_brace_by_whitespace_or_newline():
    got, n = normalize("}\n    -> (i8, i8)")
    assert got == "}\n    -> i8, i8" and n == 1


def test_several_occurrences_are_all_rewritten_and_counted():
    got, n = normalize("} -> (i8, i8)\nx\n} -> (i16, i16)")
    assert got == "} -> i8, i8\nx\n} -> i16, i16"
    assert n == 2


def test_nested_parentheses_in_the_type_list_are_balanced_correctly():
    got, n = normalize("} -> (tensor<2xf32>, tensor<3xf32>) trailing")
    assert got == "} -> tensor<2xf32>, tensor<3xf32> trailing" and n == 1


def test_an_unbalanced_list_is_left_for_the_parser_to_report():
    src = "} -> (tensor<1xi64>, tensor<1xi64>"
    assert normalize(src) == (src, 0), "a truncated module must not be silently patched"


def test_a_graph_with_nothing_to_do_is_returned_byte_identical():
    src = "module { func.func @f() { func.return } }"
    got, n = normalize(src)
    assert got == src and n == 0


def test_the_real_failing_shape_from_a_min_dim_lowering():
    # aten.min.dim returns values AND indices; this is the op that made a whole graph uncompilable.
    src = ('    linalg.yield %3999, %4000 : i64, i64\n'
           '    } -> (tensor<1xi64>, tensor<1xi64>)\n'
           '    %4001 = tensor.expand_shape %3991 [[0 : i64, 1 : i64]] output_shape [1, 1]')
    got, n = normalize(src)
    assert n == 1
    assert "} -> tensor<1xi64>, tensor<1xi64>" in got
    assert "[[0 : i64, 1 : i64]] output_shape [1, 1]" in got, "the next op must be untouched"
