"""Batched numerical proofs retain every threshold and the original conflict order."""

import numpy as np
import pytest

from merlin.xdsl_dialects.lowering import output_fusion as OF


@pytest.mark.parametrize("batch", [1, 2, 7, 16, 64])
@pytest.mark.parametrize("target_scale", [1.0, 1.5])
def test_interval_matches_exhaustive_integer_offsets(batch, target_scale):
    scale = np.array([0.5, 0.25, 0.125], dtype=np.float32)
    bias = np.array([3, -4, 0], dtype=np.float32)

    def source(x):
        return np.clip(np.rint((x.astype(np.float32) + bias) * scale), -16, 16)

    def target(x):
        return np.clip(np.rint(x.astype(np.float32) * scale * target_scale), -16, 16)

    lower, upper, conflict = OF.offset_interval(
        source, target, 3, -20, 20, iter(range(-16, 18)), (-64, 63), level_batch=batch
    )
    sequential = OF.offset_interval(source, target, 3, -20, 20, range(-16, 18), (-64, 63))
    for result, expected in zip((lower, upper, conflict), sequential):
        np.testing.assert_array_equal(result, expected)
    domain = np.arange(-20, 21)[:, None]
    for channel in range(3):
        exact = [
            offset for offset in range(-44, 44)
            if np.array_equal(source(domain)[:, channel], target(domain + offset)[:, channel])
        ]
        admitted = list(range(int(lower[channel]), int(upper[channel]) + 1))
        assert admitted == exact


def test_large_channel_batch_has_bounded_temporaries(monkeypatch):
    shapes = []
    original = OF._threshold

    def observe(fn, q, lo, hi, channels):
        shapes.append(np.broadcast_shapes((channels,), np.shape(q)))
        return original(fn, q, lo, hi, channels)

    monkeypatch.setattr(OF, "_threshold", observe)
    channels = 32768
    result = OF.offset_interval(
        lambda x: x, lambda x: x, channels, -1, 1, range(-2, 3), (-4, 4), level_batch=64
    )
    assert all(np.prod(shape) <= 262144 for shape in shapes)
    np.testing.assert_array_equal(result[0], np.zeros(channels, dtype=np.int64))
    np.testing.assert_array_equal(result[1], np.zeros(channels, dtype=np.int64))


@pytest.mark.parametrize("batch", [0, -1, True, 1.5])
def test_invalid_batch_refuses(batch):
    with pytest.raises(ValueError, match="positive integer"):
        OF.offset_interval(lambda x: x, lambda x: x, 1, -1, 1, [0], (-2, 2), level_batch=batch)
