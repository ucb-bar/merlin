"""Original blocked-copy permission/source and direct-validator freshness."""

from dataclasses import replace

import pytest
from test_compact_dense import fixture
from test_compact_load_groups import adapter, encode, project, schedule

from merlin.compile.scheduling import block_schedule as B
from merlin.compile.scheduling.compact_dense import emit_compact_dense
from merlin.compile.scheduling.compact_verification import verify_compact_dense
from merlin.transforms.dense_planning import dense_blocks


@pytest.mark.parametrize("boundary", ["encode", "project"])
def test_group_permission_or_limits_cannot_change_inside_original_callback(boundary):
    p, r, s = fixture()
    good = schedule(p, r, s)
    a = adapter(p)
    calls = []

    def changed(operation):
        calls.append(1)
        if boundary == "encode":
            object.__setattr__(a.load_grouping.limits, "max_row_bytes", 256)
            return encode(operation)
        object.__setattr__(a.load_grouping, "exact_blocked_mapping", False)
        return project(operation)

    a = replace(a, **{boundary: changed})
    with pytest.raises(ValueError, match="permissions or callbacks changed"):
        if boundary == "encode":
            emit_compact_dense(
                p, dense_blocks(p, r)[-1], s, a, name="dense", dtypes=("i8", "i8", "i32"), group_dma=True
            )
        else:
            verify_compact_dense(good.kernel, p, r, s, a, static_budget=150, timeout_ms=20_000)
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["limit_helper", "limit_validator", "limit_defaults"])
def test_original_callback_cannot_replace_direct_transfer_validation(monkeypatch, damage):
    p, r, s = fixture()
    bad_calls = []

    def forbidden(*args, **kwargs):
        bad_calls.append(1)
        return True

    def changed(operation):
        if damage == "limit_helper":
            monkeypatch.setattr(B, "_transfer_integers", forbidden)
        elif damage == "limit_validator":
            monkeypatch.setattr(B.PanelTransferLimits.__post_init__, "__code__", (lambda self: None).__code__)
        else:
            monkeypatch.setattr(B._transfer_integers, "__kwdefaults__", {"positive": True})
        return encode(operation)

    a = adapter(p, encode=changed)
    with pytest.raises(ValueError, match="snapshot implementation changed"):
        emit_compact_dense(p, dense_blocks(p, r)[-1], s, a, name="dense", dtypes=("i8", "i8", "i32"), group_dma=True)
    assert bad_calls == []


def test_mutated_original_group_context_invalidates_issued_result():
    p, r, s = fixture()
    a = adapter(p)
    result = schedule(p, r, s, a=a)
    object.__setattr__(a.load_grouping, "source_base_alignments", (16, 16))
    with pytest.raises(ValueError, match="permissions or callbacks changed"):
        result.revalidate()


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_blocked_mapping_permission_is_exact_boolean(value):
    p, _, _ = fixture()
    with pytest.raises(ValueError, match="blocked mapping law"):
        replace(adapter(p).load_grouping, exact_blocked_mapping=value)


def test_plain_rectangular_copy_limits_do_not_authorize_blocked_writes():
    p, _, _ = fixture()
    with pytest.raises(ValueError, match="original typed law"):
        adapter(p, load_grouping=adapter(p).load_grouping.limits)
    with pytest.raises(ValueError, match="blocked mapping law"):
        replace(adapter(p).load_grouping, mapping="ordinary_affine_copy")


def test_512_grouped_program_is_checked_symbolically_without_product_expansion(monkeypatch):
    import merlin.sched.ir as ir

    p, r, s = fixture(shape=(512, 512, 512), tile=(16, 16, 16))
    r = replace(r, operand_bytes=1 << 20, accumulator_bytes=1 << 20, max_m_blocks=4, max_n_blocks=4, max_k_blocks=2)
    s = replace(
        s,
        lhs=replace(s.lhs, capacity=256),
        rhs=replace(s.rhs, capacity=256),
        accumulator=replace(s.accumulator, capacity=256),
    )
    monkeypatch.setattr(
        ir, "instances", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no dynamic expansion"))
    )
    result = schedule(p, r, s, pe=True)
    assert result is not None
    assert result.realization["output_tiles"] == 1024
    assert result.realization["grouped_dma_commands"] > 0
    assert result.realization["static_instruction_calls"] < 40
    assert result.realization["projected_effect_sites"] < 100
    assert len(result.kernel.text()) < 20_000


def test_recursive_snapshot_owner_binding_cannot_hide_changed_original_limits(monkeypatch):
    from merlin.compile.scheduling import schedule_pipeline as P

    p, r, s = fixture()
    a = adapter(p)
    original = P._rewrite_snapshot
    replacement_calls, encode_calls = [], []

    def false_snapshot(value):
        replacement_calls.append(1)
        if type(value) is int and value == 256:
            return (int, 128)
        return original(value)

    def changed(operation):
        encode_calls.append(1)
        object.__setattr__(a.load_grouping.limits, "max_row_bytes", 256)
        monkeypatch.setattr(P, "_rewrite_snapshot", false_snapshot)
        return encode(operation)

    a = replace(a, encode=changed)
    with pytest.raises(ValueError, match="snapshot implementation changed"):
        emit_compact_dense(p, dense_blocks(p, r)[-1], s, a, name="dense", dtypes=("i8", "i8", "i32"), group_dma=True)
    assert encode_calls == [1] and replacement_calls == []
