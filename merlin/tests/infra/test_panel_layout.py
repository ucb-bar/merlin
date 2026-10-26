"""Column-panel weight packing: exact permutation, zero tails, refused reads."""

import numpy as np
import pytest

from merlin.compile.scheduling.panel_layout import PanelLayout


@pytest.mark.parametrize("k,n,width", [(7, 24, 8), (5, 19, 8), (3, 5, 8), (9, 1, 4)])
def test_pack_is_a_permutation_with_zero_tail(k, n, width):
    rng = np.random.default_rng(k * 31 + n)
    weight = rng.integers(-128, 128, (k, n), dtype=np.int8)
    layout = PanelLayout(k, n, width)
    packed = layout.pack(weight)
    assert packed.shape == (-(-n // width), k, width)
    flat = packed.reshape(-1)
    for row in range(k):
        for col in range(n):
            assert flat[layout.offset(row, col)] == weight[row, col]
    np.testing.assert_array_equal(layout.unpack(packed), weight)
    assert np.count_nonzero(flat) == np.count_nonzero(weight)


def test_source_inverts_reads_and_refuses_padding_or_crossing():
    layout = PanelLayout(4, 13, 8)
    assert layout.source(layout.offset(2, 9), 4) == (2, 9)
    assert layout.source(layout.offset(3, 0), 8) == (3, 0)
    with pytest.raises(ValueError):
        layout.source(layout.offset(1, 6), 4)  # crosses into the next panel
    with pytest.raises(ValueError):
        layout.source(layout.offset(1, 10), 4)  # reads padding lanes past N
    with pytest.raises(ValueError):
        layout.source(layout.shape[0] * 4 * 8, 1)


def test_nonzero_padding_and_bad_widths_are_refused():
    layout = PanelLayout(2, 3, 4)
    packed = layout.pack(np.ones((2, 3), np.int8))
    packed[-1, 0, 3] = 1
    with pytest.raises(ValueError):
        layout.unpack(packed)
    with pytest.raises(ValueError):
        PanelLayout(2, 3, 0)
