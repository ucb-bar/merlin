import copy

import pytest

from merlin.llvmlower import quantized_affine_domain as domain
from merlin.llvmlower import quantized_affine_pair as pair

SOURCE = dict(
    lhs_scale=0.020491356030106544, rhs_scale=0.017152275890111923, output_scale=0.030738085508346558, relu=True
)
PREDICTOR = dict(p=135, q=113, scale=0.004938124679028988)


def test_proved_nonnegative_operand_preserves_every_admitted_pair():
    certificate = domain.derive(SOURCE, PREDICTOR, [[-128, 127], [0, 127]])
    assert domain.validate(certificate) == certificate
    assert certificate["admitted_pairs"] == certificate["excluded_pairs"] == 32768
    assert certificate["admitted_source_sha256"] == certificate["admitted_prediction_sha256"]
    assert certificate["exact_for_all_admitted_pairs"]
    assert not certificate["exact_for_complete_type_domain"]
    assert certificate["excluded_mismatches"] > 0
    assert pair.derive(**SOURCE, **PREDICTOR)["mismatched_pairs"] == certificate["excluded_mismatches"]


def test_full_type_domain_cannot_be_silently_reused():
    with pytest.raises(ValueError, match="within admitted"):
        domain.derive(SOURCE, PREDICTOR, [[-128, 127], [-128, 127]])


@pytest.mark.parametrize(
    "intervals",
    [[], [[0, 1]], [[-129, 127], [0, 127]], [[0, -1], [0, 127]], [[0.0, 127], [0, 127]], [[False, 127], [0, 127]]],
)
def test_invalid_source_domains_refuse(intervals):
    with pytest.raises(ValueError):
        domain.derive(SOURCE, PREDICTOR, intervals)


@pytest.mark.parametrize("field", ["admitted_pairs", "source_table_sha256", "excluded_mismatches"])
def test_mutated_certificate_refuses(field):
    certificate = copy.deepcopy(domain.derive(SOURCE, PREDICTOR, [[-128, 127], [0, 127]]))
    certificate[field] = "changed"
    with pytest.raises(ValueError):
        domain.validate(certificate)


def test_finite_source_result_interval_is_only_caller_bound_fact():
    certificate = domain.source_result_interval(SOURCE)
    assert certificate["minimum"] == 0 and certificate["maximum"] == 127
    assert certificate["caller_binding_required"]
    assert certificate["pairs"] == 65536
    signed = domain.source_result_interval(dict(SOURCE, relu=False))
    assert signed["minimum"] == -128
