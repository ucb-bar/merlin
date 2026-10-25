"""Executable typed loop decisions compose with segmented source/hazard checks."""

from collections import Counter
from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER
from test_dense_segmented_pipeline import chosen, inputs

from merlin.compile.scheduling.dense_panel_pipeline import DRAIN, dense_panel_rhs_stripes
from merlin.compile.scheduling.dense_segmented_pipeline import schedule_dense_segmented_pipeline
from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
from merlin.compile.scheduling.schedule_ir import (
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
)
from merlin.sched.ir import Loop, evaluate, instances


def consume(p, r, s, b, admission, requests):
    plan = make_dense_schedule_plan(
        p, r, s, ADAPTER, source_identity="b" * 64, dtypes=("i8", "i8", "i32"), static_budget=128,
        name="segmented", block_tiles=(b.m, b.k, b.n), pipeline_kind="retained_lhs_segmented_k",
        pipeline_prefetch=True, pipeline_issue_order="prefetch_rhs_between_columns", loop_transforms=requests,
    )
    return schedule_dense_segmented_pipeline(
        p, r, s, ADAPTER, admission, name="segmented", dtypes=("i8", "i8", "i32"),
        schedule_plan=parse_dense_schedule_plan(dense_schedule_text(plan)), source_identity="b" * 64,
        external_buffers_disjoint=True,
    )


def epochs(kernel):
    result, pending = [], []
    for call, env in instances(kernel):
        if call.instr == DRAIN:
            result.append(pending)
            pending = []
        else:
            pending.append((call.instr, tuple((k, evaluate(v, env)) for k, v in call.args)))
    assert not pending
    return result


def test_segmented_stripe_decisions_change_actual_commands_preserving_epochs_and_data():
    p, r, s, b, admission = inputs((3, 14, 7), 2)
    original = chosen(p, r, s, b, admission, issue_order="prefetch_rhs_between_columns")
    requests = dense_panel_rhs_stripes(original.kernel, ADAPTER, factor=2)
    actual = consume(p, r, s, b, admission, requests)
    expected = propose_loop_transforms(original.kernel, requests, index_bits=ADAPTER.index_bits)
    assert actual.kernel == expected and actual.kernel != original.kernel
    before, after = epochs(original.kernel), epochs(actual.kernel)
    assert len(before) == len(after)
    assert [Counter(group) for group in before] == [Counter(group) for group in after]
    for left, right in zip(before, after, strict=True):
        assert [c for c in left if c[0] == "compute"] == [c for c in right if c[0] == "compute"]
    assert actual.realization["source"]["output_tiles"] == original.realization["source"]["output_tiles"]
    assert actual.admission == original.admission
    assert actual.realization["loop_transforms"]["requests"]
    assert "loop_transforms" in actual.realization["schedule_ir"]["decisions_consumed"]
    actual.materialize_scf().verify()


def test_segmented_typed_fission_cannot_separate_preload_from_compute():
    p, r, s, b, admission = inputs((3, 14, 7), 2)
    original = chosen(p, r, s, b, admission, issue_order="prefetch_rhs_between_columns")
    paths = []

    def walk(body, prefix=()):
        for statement in body:
            if type(statement) is Loop:
                path = (*prefix, statement.var)
                if (len(statement.body) == 2
                        and [getattr(c, "instr", None) for c in statement.body] == ["preload", "compute"]):
                    paths.append(path)
                walk(statement.body, path)

    walk(original.kernel.body)
    assert paths
    # This is structurally expressible in the dialect, but violates the actual
    # target descriptor pairing. The original verifier must still reject it.
    with pytest.raises(ValueError, match="pair|PRELOAD|preload|descriptor"):
        consume(p, r, s, b, admission, (LoopTransform("fission", paths[0], cut=1),))


def test_segmented_stripes_execute_all_original_scalar_outputs(monkeypatch):
    import test_compact_dense as existing

    original_fixture = existing.fixture

    def capacity_fixture(**kwargs):
        kwargs["shape"] = (3, 14, 7)
        p, r, s = original_fixture(**kwargs)
        s = replace(s, lhs=replace(s.lhs, capacity=256), rhs=replace(s.rhs, capacity=48),
                    accumulator=replace(s.accumulator, capacity=64))
        r = replace(r, operand_bytes=256 * s.lhs.unit_bytes + 48 * s.rhs.unit_bytes,
                    accumulator_bytes=64 * s.accumulator.unit_bytes, max_m_blocks=2, max_n_blocks=2, max_k_blocks=2)
        return p, r, s

    def schedule(p, r, s, **kwargs):
        from merlin.compile.scheduling.dense_panel_pipeline import _serial_projection
        from merlin.transforms.dense_planning import dense_blocks

        b = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, 2, 2))
        *_, admission = inputs()
        original = chosen(p, r, s, b, admission, issue_order="prefetch_rhs_between_columns")
        requests = dense_panel_rhs_stripes(original.kernel, ADAPTER, factor=2)
        actual = consume(p, r, s, b, admission, requests)
        return replace(actual, kernel=_serial_projection(actual.kernel), realization=actual.realization["source"])

    monkeypatch.setattr(existing, "fixture", capacity_fixture)
    monkeypatch.setattr(existing, "schedule", schedule)
    existing.test_tails_reuse_and_independent_execution(("m", "n"), False)


def test_segmented_wide_loads_keep_combined_law_and_check_every_actual_band():
    from test_panel_tile_loads import evaluated_calls

    from merlin.compile.scheduling.dense_segmented_pipeline import admit_segmented_tile_loads
    from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads, expand_panel_tile_loads
    from merlin.compile.scheduling.tile_load_coalescing import (
        TileLoadAddressLaw,
        TileLoadMapping,
        _linear,
        propose_tile_load_coalescing,
    )
    from merlin.sched.ir import Const

    p, r, s, b, admission = inputs((4, 12, 8), 2)
    original = chosen(p, r, s, b, admission, issue_order="prefetch_rhs_between_columns")
    requests = dense_panel_rhs_stripes(original.kernel, ADAPTER, factor=2)
    striped = consume(p, r, s, b, admission, requests)
    laws = {
        "lhs": TileLoadAddressLaw(p.tile_k, p.tile_m, 4 * p.tile_k, 1, p.k, p.tile_m, "test-lhs-address-law"),
        "weight": TileLoadAddressLaw(p.tile_n, p.tile_k, 4 * p.tile_n, 1, p.n, p.tile_k, "test-rhs-address-law"),
    }

    def project(call):
        if call.instr not in ("lhs_load", "weight_load"):
            return None
        op = ADAPTER.project(call)
        view = s.lhs if op.role == "lhs" else s.rhs
        law = laws[op.role]
        return TileLoadMapping(op.role, view.backing, op.dram_row * law.source_pitch_bytes + op.dram_col,
                               op.row, op.rows, op.cols, law)

    def encode(call, columns):
        return replace(call, args=tuple((key, Const(columns) if key == "cols" else value)
                                       for key, value in call.args))

    def band(call, mapping):
        offset, terms = _linear(mapping.source_byte)
        base, base_terms = _linear(project(call).source_byte)
        assert terms == base_terms
        columns = offset - base
        return replace(call, args=tuple((key, mapping.local_row if key == "dst"
                                        else mapping.columns if key == "cols"
                                        else value + columns if key == "src_col"
                                        else value + columns // mapping.law.tile_columns if key == "b" else value)
                                       for key, value in call.args))

    proposal, changes = propose_tile_load_coalescing(striped.kernel, project=project, encode=encode, max_tiles=2)
    assert changes["changes"]
    # Existing one-K-panel admission must not silently acquire a combined law.
    with pytest.raises(ValueError, match="original checked panel"):
        admit_panel_tile_loads(striped, proposal, p, b, r, s, ADAPTER, project=project, encode_band=band)
    actual = admit_segmented_tile_loads(striped, proposal, p, b, r, s, ADAPTER,
                                      project=project, encode_band=band)
    expanded, proof = expand_panel_tile_loads(actual.kernel, project=project, encode_band=band)
    assert evaluated_calls(expanded) == evaluated_calls(striped.kernel)
    assert proof["physical_load_commands"] < proof["expanded_logical_tile_bands"]
    assert actual.admission == original.admission
    assert actual.realization["dynamic_epoch_drains"] == original.realization["dynamic_epoch_drains"]
    assert actual.realization["source"]["output_tiles"] == original.realization["source"]["output_tiles"]
    actual.materialize_scf().verify()

    def corrupt_mapping(call):
        mapping = project(call)
        return None if mapping is None else replace(mapping, source_byte=mapping.source_byte + 1)

    with pytest.raises(ValueError, match="actual wide mapping differs"):
        admit_segmented_tile_loads(striped, proposal, p, b, r, s, ADAPTER,
                                  project=corrupt_mapping, encode_band=band)
