"""Exact rational readout declarations and execution agree at half ties."""

import copy

import pytest

from merlin.common.paths import repo_root
from merlin.runtime.exact_readout import requantize_int32
from merlin.targetgen.software_spec import load_software_spec, validate_numerical_semantics


def semantics():
    return load_software_spec(repo_root() / 'examples/gemmini/target/software-spec.yaml')['numerical_semantics']


def test_declared_exact_readout_matches_rne_on_positive_and_negative_ties():
    declared = semantics()
    assert declared['readout']['requant_rounding'] == 'half_even'
    assert requantize_int32([1,3,5,-1,-3,-5], scale_numerator=1,
                           scale_denominator=2).tolist() == [0,2,2,0,-2,-2]


@pytest.mark.parametrize('rounding', ['half_up', 'toward_zero', None])
def test_inconsistent_or_missing_exact_readout_rounding_refuses(rounding):
    declared = copy.deepcopy(semantics())
    if rounding is None:
        declared['readout'].pop('requant_rounding')
    else:
        declared['readout']['requant_rounding'] = rounding
    with pytest.raises(ValueError, match='requant_rounding: half_even'):
        validate_numerical_semantics(declared)
