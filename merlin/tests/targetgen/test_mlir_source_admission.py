"""Lexical budgets are explicit parser inputs, never semantic qualification."""

import pytest

from merlin.targetgen.contract.mlir_source_admission import MlirSourceUnavailable, admit_mlir_source


def admit(text, **selections):
    selected = {
        "max_source_bytes": 1000,
        "max_nesting": 64,
        "max_integer_bits": 64,
        "allow_dense": False,
        "allow_dense_resource": False,
    }
    selected.update(selections)
    return admit_mlir_source(text, **selected)


@pytest.mark.parametrize(
    "selection",
    [
        {"max_source_bytes": True},
        {"max_source_bytes": 0},
        {"max_nesting": True},
        {"max_nesting": 0},
        {"max_integer_bits": True},
        {"max_integer_bits": 0},
        {"allow_dense": 0},
        {"allow_dense_resource": None},
    ],
)
def test_explicit_lexical_selections_cannot_be_supplied_as_flags_or_absent_limits(selection):
    with pytest.raises(MlirSourceUnavailable, match="bounded lexical"):
        admit("i8", **selection)


def test_complete_integer_width_and_nesting_budget_are_selected_by_each_reader():
    assert admit("array<[i000064, ui32, si8]>") is None
    with pytest.raises(MlirSourceUnavailable, match="scalar width"):
        admit("array<[i000065]>")
    with pytest.raises(MlirSourceUnavailable, match="syntax nesting"):
        admit("array<[i8]>", max_nesting=1)
    assert admit("array<[i65]>", max_integer_bits=65, max_nesting=2) is None


def test_large_explicit_budgets_have_no_guessed_target_or_shape_ceiling():
    assert admit("i8", max_source_bytes=10**9, max_nesting=10**9, max_integer_bits=10**9) is None
    assert admit("i8", max_integer_bits=10**5000) is None


def test_aggregate_flags_are_distinct_explicit_choices_without_a_parser_or_allocation_grant():
    source = "dense<0> : tensor<100000000xi8>"
    assert admit(source, allow_dense=True) is None
    with pytest.raises(MlirSourceUnavailable, match="aggregate literals"):
        admit("dense_resource<unavailable> : tensor<100000000xi8>", allow_dense=True)
    assert admit("dense_resource<unavailable> : tensor<100000000xi8>", allow_dense_resource=True) is None


def test_lexical_admission_is_not_parse_validity_or_a_semantic_disposition():
    assert admit("not_a_verified_operation(%undefined)") is None
    assert admit('// dense<0> i10000000000 [[[\n"dense<0> i10000000000 [[["') is None
    with pytest.raises(MlirSourceUnavailable, match="delimiters"):
        admit("array<[i8]}")


def test_unprefixed_identifier_width_screening_is_an_explicit_conservative_input_domain():
    with pytest.raises(MlirSourceUnavailable, match="scalar width"):
        admit("untyped_attribute {i512 = 0 : i8}")
    assert admit('untyped_attribute {"i512" = 0 : i8}') is None
