"""Retained rows preserve source identity, lifetime, two readouts and padding."""

from dataclasses import replace

import numpy as np
import pytest
from test_rolling_spatial import Integers, make_plan

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.rolling_spatial import emit_rolling_spatial_producer
from merlin.compile.scheduling.spatial_readout_cache import (
    emit_cached_spatial_max,
    propose_spatial_readout_cache,
)
from merlin.llvmlower.requantization import quantized


@pytest.mark.parametrize("height,width,epoch", [(9, 11, 5), (9, 11, 2), (6, 7, 2), (3, 3, 2)])
def test_actual_frontier_effects_preserve_original_rows_and_all_outputs(height, width, epoch):
    source = make_plan(height, width, SpatialMaxReadout((height, width), (3, 3), (2, 2), (1, 1, 1, 1), -128))
    p, s = source.packed.source.producer, source.packed.source.spatial
    row_bytes = p.primitive * 4
    cache = propose_spatial_readout_cache(
        source,
        AddressWindow("acc", 0, 64 * row_bytes),
        AddressWindow("acc", 64 * row_bytes, 192 * row_bytes),
        row_bytes=row_bytes,
    )
    assert cache is not None
    cache = replace(cache, output_rows=min(epoch, s.output_hw[0]))
    source_chains = source.packed.source.readouts
    source = replace(
        source,
        packed=replace(
            source.packed,
            source=replace(
                source.packed.source,
                readouts=tuple(
                    replace(c, pool=replace(c.pool, scales=(0.75 if i == 0 else 0.625,)))
                    for i, c in enumerate(source_chains)
                ),
            ),
        ),
    )
    cache = replace(cache, source=source)

    def q2(n):
        return source.packed.source.readouts[n // p.column_tile].pool.scales

    q1 = (np.arange(2 * height * width * p.output_channels) % 128).reshape(2, height, width, p.output_channels)
    scratch = np.zeros(source.required_bytes, dtype=np.int8)
    acc = np.zeros((p.accumulator_rows, p.column_tile), dtype=np.int32)
    identities = np.full_like(acc, -1)
    oh, ow = s.output_hw
    result = np.full((2 * oh * ow, p.output_channels), -128, dtype=np.int8)
    writes = np.zeros_like(result, dtype=int)
    expected = np.full_like(result, -128)
    integers = Integers()
    reload_bytes = 0
    for b in range(2):
        for y in range(oh):
            for x in range(ow):
                for ky in range(3):
                    for kx in range(3):
                        iy, ix = y * 2 - 1 + ky, x * 2 - 1 + kx
                        if 0 <= iy < height and 0 <= ix < width:
                            for n in range(p.output_channels):
                                expected[(b * oh + y) * ow + x, n] = max(
                                    expected[(b * oh + y) * ow + x, n], quantized(int(q1[b, iy, ix, n]), q2(n))
                                )

    def lower(kind, f):
        nonlocal reload_bytes
        if kind in ("begin_transport", "await_transport", "consumer_issued"):
            return object()
        if kind == "produce_packed_panel":
            # A completed producer may clobber EVERY byte of its assigned ACC
            # window. Retained readout rows must survive this actual lifetime.
            acc[:64] = -999
            for r in range(f["rows"]):
                x = min(max(f["output_x"] * 2 - 1 + f["first_halo_column"] + r, 0), width - 1)
                start = f["stripe_offset"] + r * p.output_channels
                scratch[start : start + p.output_channels] = q1[f["image"], f["producer_y"], x]
        elif kind == "reload_intermediate":
            reload_bytes += f["rows"] * f["columns"]
            for r in range(f["rows"]):
                start = f["scratch_offset"] + r * f["scratch_row_stride"]
                acc[f["accumulator_row"] + r, : f["columns"]] = scratch[start : start + f["columns"]]
                identities[f["accumulator_row"] + r, : f["columns"]] = 1
        elif kind == "store_spatial_max":
            assert f["upad"] == 0 and f["local_height"] == 3
            for x in range(f["output_pixels"]):
                for n in range(f["columns"]):
                    value = -128
                    for ky in range(3):
                        local_y = ky - f["upad"]
                        for kx in range(3):
                            if 0 <= local_y < f["local_height"]:
                                row = f["accumulator_row"] + local_y * f["local_width"] + x * 2 + kx
                                assert identities[row, n] == 1
                                value = max(value, quantized(int(acc[row, n]), q2(f["output_column"] + n)))
                    result[f["output_row"] + x, f["output_column"] + n] = value
                    writes[f["output_row"] + x, f["output_column"] + n] += 1

    emit_rolling_spatial_producer(
        source,
        integers,
        lower,
        synchronous_completion=True,
        pool_consumer=lambda **fields: emit_cached_spatial_max(
            cache, integers, lower, synchronous_completion=True, **fields
        ),
    )
    np.testing.assert_array_equal(result, expected)
    assert np.all(writes == 1)
    assert reload_bytes == cache.report()["retained_reload_bytes"]
    assert reload_bytes < cache.report()["original_reload_bytes"]


def test_unknown_or_overlapping_capacity_refuses_and_insufficient_capacity_retains_old_route():
    source = make_plan(9, 11, SpatialMaxReadout((9, 11), (3, 3), (2, 2), (1, 1, 1, 1), -128))
    producer, cache = AddressWindow("acc", 0, 2048), AddressWindow("acc", 2048, 2048)
    plan = propose_spatial_readout_cache(source, producer, cache, row_bytes=16)
    assert plan is not None
    for changed in (replace(cache, start=0), replace(cache, space="unknown"), replace(cache, start=2049)):
        with pytest.raises(ValueError):
            replace(plan, cache=changed)
    with pytest.raises(ValueError):
        replace(plan, row_bytes=8)
    assert propose_spatial_readout_cache(source, producer, replace(cache, size=16), row_bytes=16) is None
    with pytest.raises(ValueError):
        emit_cached_spatial_max(
            plan,
            Integers(),
            lambda *_: None,
            image=0,
            pool_y=0,
            output_x=0,
            output_pixels=100,
            synchronous_completion=True,
        )
    with pytest.raises(ValueError):
        emit_cached_spatial_max(
            plan,
            Integers(),
            lambda *_: None,
            image=0,
            pool_y=0,
            output_x=0,
            output_pixels=1,
            synchronous_completion=False,
        )


def test_optional_cache_does_not_reject_other_valid_original_pool_geometry():
    source = make_plan(6, 7, SpatialMaxReadout((6, 7), (3, 3), (2, 2), (0, 0, 0, 0), -128))
    assert (
        propose_spatial_readout_cache(
            source, AddressWindow("acc", 0, 2048), AddressWindow("acc", 2048, 2048), row_bytes=16
        )
        is None
    )
