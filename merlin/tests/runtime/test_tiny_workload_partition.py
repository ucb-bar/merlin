"""Bound Tiny column shards and independently check BF16 tensor-scale choices."""
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np
import pytest

pytestmark = pytest.mark.target('atlas')
from merlin.common.paths import repo_root
sys.path.insert(0, str(repo_root() / 'examples/atlas/comparisons'))
from tiny_workload_inventory import column_shards
from tiny_workload_block import tensor_scale_from_exact_BF16_views


def test_twelve_programs_cover_all_logical_columns_once():
    specs = [(2048, 2048), (2048, 256), (2048, 256), (2048, 2048), (2048, 5632), (2048, 5632), (5632, 2048)]
    assert sum((len(column_shards(k, n)) for k, n in specs)) == 12
    for k, n in specs:
        shards = column_shards(k, n)
        columns = [column for shard in shards for column in range(*shard['columns'])]
        assert columns == list(range(n))
        assert all(
            shard['source_array_bytes'] < 32 * 1024 ** 2 - 4096 and shard['dram_bytes'] < 64 * 1024 ** 2
            for shard in shards
        )
        assert all((shard['physical_mkn'][1] == k for shard in shards))


@pytest.mark.parametrize('shape', [(2048, 33), (8192, 2048, 1048576), (2048, 32000, 1048576)])
def test_bad_source_or_output_extent_refused(shape):
    with pytest.raises(ValueError):
        column_shards(*shape)


def reference(bits):
    magnitude = bits & 32767
    exponent = magnitude >> 7 & 255
    fraction = magnitude & 127
    if not magnitude:
        return 0
    if exponent == 255:
        raise ValueError('nonfinite')
    def pow2(e):
        return Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)
    value = Fraction(128 + fraction, 128) * pow2(exponent - 127) if exponent else Fraction(fraction, 128) * pow2(-126)
    exp = value.numerator.bit_length() - value.denominator.bit_length() - 8
    while value > 448 * pow2(exp):
        exp += 1
    while value <= 448 * pow2(exp - 1):
        exp -= 1
    if not -126 <= exp <= 127:
        raise ValueError('not admitted')
    return exp


def test_scale_boundaries_signs_and_ties_match_exact_rationals():
    samples = list(range(0, 32768, 37)) + [0, 1, 127, 128, 129, 16256, 16352, 16353, 32639, 32640]
    for magnitude in samples:
        for sign in (0, 32768):
            bits = magnitude | sign
            array = (np.array([bits], dtype=np.uint32) << 16).view(np.float32)
            try:
                expected = reference(bits)
            except ValueError:
                with pytest.raises(ValueError):
                    tensor_scale_from_exact_BF16_views(array)
            else:
                assert tensor_scale_from_exact_BF16_views(array) == expected


def test_non_BF16_float32_views_and_nonfinite_refused():
    for array in [np.array([1.00001], np.float32), np.array([float('inf')], np.float32), np.array([1.0], np.float64)]:
        with pytest.raises(ValueError):
            tensor_scale_from_exact_BF16_views(array)


def test_chunkwise_B_scale_must_be_declared_when_global_differs():
    weights = np.ones((32, 64), np.float32)
    weights[:, 32:] = 256
    assert tensor_scale_from_exact_BF16_views(weights) == 0
    assert tensor_scale_from_exact_BF16_views(weights[:, :32].copy()) == -8
    assert tensor_scale_from_exact_BF16_views(weights[:, 32:].copy()) == 0
