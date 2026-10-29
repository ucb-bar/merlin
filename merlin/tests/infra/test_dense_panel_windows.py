"""Caller-selected physical bank placement changes actual checked row operands."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER
from test_dense_panel_pipeline import setup

from merlin.compile.scheduling.dense_panel_pipeline import (
    build_dense_panel_pipeline,
    select_dense_panel_window_offsets,
)
from merlin.compile.scheduling.schedule_ir import (
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
)
from merlin.semantic_compiler.allocation_types import StorageBank


def banks(storage):
    result = []
    for view in (storage.lhs, storage.rhs):
        middle = view.base + view.capacity // 2
        for slot, low, high in ((0, view.base, middle), (1, middle, view.base + view.capacity)):
            result.append(
                StorageBank(
                    f"{view.name}_{slot}",
                    view.backing,
                    high * view.unit_bytes,
                    "bytes",
                    view.unit_bytes,
                    low * view.unit_bytes,
                )
            )
    return tuple(result)


def test_typed_windows_roundtrip_drive_actual_addresses_and_preserve_original_source():
    from merlin.compile.scheduling.dense_panel_pipeline import schedule_dense_panel_pipeline

    p, r, storage, block, admission = setup()
    offsets = select_dense_panel_window_offsets(p, block, r, storage, banks=banks(storage))
    assert offsets == ((0, 0), (6, 6))
    default = build_dense_panel_pipeline(p, block, r, storage, ADAPTER, name="pipeline", dtypes=("i8", "i8", "i32"))
    adjacent = build_dense_panel_pipeline(
        p, block, r, storage, ADAPTER, name="pipeline", dtypes=("i8", "i8", "i32"), window_offsets=((0, 0), (4, 6))
    )
    assert adjacent == default
    plan = make_dense_schedule_plan(
        p,
        r,
        storage,
        ADAPTER,
        source_identity="c" * 64,
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        block_tiles=(block.m, block.k, block.n),
        pipeline_prefetch=True,
        pipeline_issue_order="prefetch_rhs_between_columns",
        pipeline_window_offsets=offsets,
    )
    plan = parse_dense_schedule_plan(dense_schedule_text(plan))
    result = schedule_dense_panel_pipeline(
        p,
        r,
        storage,
        ADAPTER,
        admission,
        name="pipeline",
        dtypes=("i8", "i8", "i32"),
        schedule_plan=plan,
        source_identity="c" * 64,
        external_buffers_disjoint=True,
    )
    assert result.kernel.digest() != default.digest()
    assert result.realization["schedule_ir"]["pipeline_window_offsets"] == offsets
    assert "pipeline_window_offsets" in result.realization["schedule_ir"]["decisions_consumed"]
    assert result.realization["source"]["output_tiles"] == p.tiles[0] * p.tiles[2]
    result.materialize_scf().verify()


@pytest.mark.parametrize("offsets", [((0, 0), (0, 6)), ((0, 0), (9, 6)), ((0, 0), (True, 6)), ((0, 0), (6,))])
def test_invalid_aliasing_escaping_or_untyped_windows_refuse(offsets):
    p, r, storage, block, _ = setup()
    with pytest.raises(ValueError, match="panel window"):
        build_dense_panel_pipeline(
            p, block, r, storage, ADAPTER, name="bad", dtypes=("i8", "i8", "i32"), window_offsets=offsets
        )


def test_alignment_and_unknown_bank_geometry_cannot_widen_original_assignment():
    p, r, storage, block, _ = setup()
    storage = replace(storage, lhs=replace(storage.lhs, alignment=2))
    with pytest.raises(ValueError, match="misalign"):
        build_dense_panel_pipeline(
            p, block, r, storage, ADAPTER, name="bad", dtypes=("i8", "i8", "i32"), window_offsets=((1, 0), (6, 6))
        )
    physical = banks(storage)
    with pytest.raises(ValueError, match="overlapping canonical"):
        select_dense_panel_window_offsets(p, block, r, storage, banks=(*physical, replace(physical[0], name="alias")))
    with pytest.raises(ValueError, match="search budget"):
        select_dense_panel_window_offsets(p, block, r, storage, banks=physical, max_candidates=1)
    with pytest.raises(ValueError, match="no two disjoint"):
        select_dense_panel_window_offsets(
            p,
            block,
            r,
            storage,
            banks=tuple(
                StorageBank(
                    view.name,
                    view.backing,
                    (view.base + view.capacity) * view.unit_bytes,
                    "bytes",
                    view.unit_bytes,
                    view.base * view.unit_bytes,
                )
                for view in (storage.lhs, storage.rhs)
            ),
        )


@pytest.mark.parametrize("biased,narrow", [(False, False), (True, False), (True, True), (False, "tiled")])
def test_bank_windows_preserve_every_output_with_reversed_completion_bias_and_readout(monkeypatch, biased, narrow):
    import test_dense_panel_pipeline as existing

    original = existing.build_dense_panel_pipeline

    def placed(p, block, resources, storage, adapter, **kwargs):
        offsets = select_dense_panel_window_offsets(p, block, resources, storage, banks=banks(storage))
        return original(p, block, resources, storage, adapter, window_offsets=offsets, **kwargs)

    monkeypatch.setattr(existing, "build_dense_panel_pipeline", placed)
    existing.test_reversed_completion_with_finite_queue_backpressure_matches_every_output(
        True, biased, narrow, "prefetch_rhs_between_columns"
    )
