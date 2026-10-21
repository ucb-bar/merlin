"""Actual bounded grouped DMA operands and symbolic blocked-layout checks."""

from dataclasses import replace

import pytest
from test_compact_dense import fixture
from test_compact_pe_weights import ADAPTER_PE
from test_compact_pe_weights import encode as primitive_encode
from test_compact_pe_weights import project as primitive_project

from merlin.compile.scheduling.block_schedule import LoadGroup, PanelTransferLimits
from merlin.compile.scheduling.compact_dense import DenseLoadGrouping, emit_compact_dense, schedule_compact_dense
from merlin.compile.scheduling.compact_verification import verify_compact_dense
from merlin.sched.ir import Call, Const, Loop, as_expr
from merlin.transforms.dense_planning import dense_blocks

FIELDS = ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step")


def encode(operation):
    if type(operation) is not LoadGroup:
        return primitive_encode(operation)
    fields = [("count", Const(len(operation.loads)))]
    for index, load in enumerate(operation.loads):
        fields.extend((f"c{index}_{key}", value) for key, value in primitive_encode(load).args)
    return Call(operation.loads[0].role + "_group", tuple(fields))


def project(call):
    if call.instr not in ("lhs_group", "weight_group"):
        return primitive_project(call)
    values = dict(call.args)
    count = values.get("count")
    if type(count) is not Const or not 2 <= count.value <= 128:
        raise ValueError("actual group count must be static and bounded")
    expected = ("count",) + tuple(f"c{i}_{key}" for i in range(count.value) for key in FIELDS)
    if tuple(key for key, _ in call.args) != expected:
        raise ValueError("actual group operands are missing, duplicated or reordered")
    role = call.instr.removesuffix("_group")
    loads = []
    for i in range(count.value):
        loads.append(primitive_project(Call(role + "_load", tuple((key, values[f"c{i}_{key}"]) for key in FIELDS))))
    return LoadGroup(tuple(loads))


def law(p, **changes):
    value = DenseLoadGrouping(
        PanelTransferLimits(64, 128, 1, 1, 1 << 20, 1 << 20, False),
        (p.k * p.lhs_bytes, p.n * p.rhs_bytes),
        (1, 1),
        (p.tile_m, p.tile_k),
        16,
        "stacked_column_tiles",
        True,
        "neutral-blocked-transfer-qualified-v1",
    )
    return replace(value, **changes)


def adapter(p, **changes):
    return replace(ADAPTER_PE, **({"encode": encode, "project": project, "load_grouping": law(p)} | changes))


def schedule(p, r, s, *, selected=True, a=None, pe=False):
    return schedule_compact_dense(
        p,
        r,
        s,
        adapter(p) if a is None else a,
        name="dense",
        dtypes=tuple("i" + str(w * 8) for w in (p.lhs_bytes, p.rhs_bytes, p.accumulator_bytes)),
        static_budget=150,
        external_buffers_disjoint=True,
        verification_timeout_ms=20_000,
        group_dma=selected,
        reuse_pe_weights=pe,
    )


def check(kernel, p, r, s, a=None):
    return verify_compact_dense(kernel, p, r, s, adapter(p) if a is None else a, static_budget=150, timeout_ms=20_000)


def change_calls(kernel, change):
    def body(statements):
        return tuple(replace(s, body=body(s.body)) if type(s) is Loop else change(s) for s in statements)

    return replace(kernel, body=body(kernel.body))


@pytest.mark.parametrize(
    "shape,tile,widths,pe",
    [
        ((7, 13, 25), (2, 3, 4), (1, 1, 4), False),
        ((7, 9, 16), (3, 2, 5), (2, 1, 4), True),
        ((8, 9, 12), (1, 4, 3), (2, 2, 4), False),
    ],
)
def test_bounded_group_commands_have_complete_primitive_effects(shape, tile, widths, pe):
    p, r, s = fixture(shape=shape, tile=tile, widths=widths, padded=True)
    before, after = schedule(p, r, s, selected=False, pe=pe), schedule(p, r, s, pe=pe)
    assert before is not None and after is not None
    assert after.choice.block == before.choice.block
    report = after.realization
    assert report["grouped_dma_commands"] > 0
    assert report["grouped_primitive_load_effects"] > report["grouped_dma_commands"]
    assert report["executed_instruction_calls"] < before.realization["executed_instruction_calls"]
    for key in ("lhs_loads", "rhs_loads", "requested_load_bytes", "requested_store_bytes", "output_tiles"):
        assert report[key] == before.realization[key]
    after.revalidate()


@pytest.mark.parametrize("missing", ["none", "mapping", "alignment", "stride", "pitch"])
def test_unknown_or_incompatible_original_mapping_retains_original_route(missing):
    p, r, s = fixture()
    a = adapter(p)
    value = a.load_grouping
    if missing == "none":
        value = None
    elif missing == "mapping":
        value = replace(value, exact_blocked_mapping=False)
    elif missing == "alignment":
        value = replace(value, limits=replace(value.limits, address_alignment=16))
    elif missing == "stride":
        value = replace(value, source_strides=(p.k + 1, p.n))
    else:
        value = replace(value, destination_block_rows=(p.tile_m + 1, p.tile_k))
    calls = []
    a = replace(a, load_grouping=value, encode=lambda op: calls.append(op) or encode(op))
    assert schedule(p, r, s, a=a) is None
    assert calls == []
    with pytest.raises(ValueError, match="qualified exact blocked"):
        emit_compact_dense(p, dense_blocks(p, r)[-1], s, a, name="dense", dtypes=("i8", "i8", "i32"), group_dma=True)


@pytest.mark.parametrize("damage", ["omitted", "overlap", "source", "rows", "pitch", "capacity"])
def test_actual_group_component_omissions_and_wrong_physical_mapping_refuse(damage):
    p, r, s = fixture(shape=(8, 12, 16))
    result = schedule(p, r, s)
    assert result is not None
    changed = []

    def mutate(call):
        if changed or call.instr not in ("lhs_group", "weight_group"):
            return call
        op = project(call)
        loads = list(op.loads)
        if damage == "omitted":
            # Re-encode only one component: the remaining primitive reader is missing.
            changed.append(1)
            return encode(loads[0])
        if damage == "overlap":
            loads[1] = replace(loads[1], row=loads[0].row)
        elif damage == "source":
            loads[1] = replace(loads[1], dram_col=loads[0].dram_col)
        elif damage == "rows":
            loads[1] = replace(loads[1], rows=as_expr(1))
        elif damage == "capacity":
            view = s.lhs if loads[1].role == "lhs" else s.rhs
            loads[1] = replace(loads[1], row=as_expr(view.base + view.capacity))
        else:
            loads[1] = replace(loads[1], row=loads[1].row + 1)
        changed.append(1)
        return encode(LoadGroup(tuple(loads)))

    with pytest.raises(ValueError):
        check(change_calls(result.kernel, mutate), p, r, s)
    assert changed


def test_limits_reexport_and_default_primitive_emission_remain_unchanged():
    from merlin.compile.scheduling.panel_transfers import PanelTransferLimits as ExistingLimits

    assert ExistingLimits is PanelTransferLimits
    p, r, s = fixture()
    from test_compact_dense import ADAPTER

    kernel = emit_compact_dense(p, dense_blocks(p, r)[-1], s, ADAPTER, name="dense", dtypes=("i8", "i8", "i32"))
    assert kernel.digest() == "404fb27bbfaf87f11c8bdcecc6f4c84a16e9c15d7a2986983b5a490c68a0fd54"
    assert "grouped_dma_commands" not in check(kernel, p, r, s)


@pytest.mark.parametrize("narrow", ["rows", "source_stride", "destination_stride", "row_multiple", "outer_offset"])
def test_narrow_transfer_facts_decline_before_any_group_encoder(narrow):
    p, r, s = fixture(shape=(8, 12, 16))
    original = law(p)
    limits = original.limits
    if narrow == "rows":
        limits = replace(limits, max_rows=1)
    elif narrow == "source_stride":
        limits = replace(limits, max_source_stride_bytes=1)
    elif narrow == "destination_stride":
        limits = replace(limits, max_destination_stride_bytes=1)
    elif narrow == "row_multiple":
        limits = replace(limits, row_bytes_multiple=5)
    else:
        # Runtime bases are originally aligned; full-column tile origins are
        # not all aligned. In particular outer K groups can begin at offset 6.
        limits = replace(limits, address_alignment=4)
        original = replace(original, source_base_alignments=(4, 4))
        s = replace(s, lhs=replace(s.lhs, unit_bytes=4))
        r = replace(r, operand_bytes=96)
    a = adapter(p, load_grouping=replace(original, limits=limits))
    assert schedule(p, r, s, selected=False, a=a) is not None
    groups = []

    def counted(op):
        if type(op) is LoadGroup:
            groups.append(op)
        return encode(op)

    assert schedule(p, r, s, a=replace(a, encode=counted)) is None
    assert groups == []


@pytest.mark.parametrize("width", [5, 6, 7, 8, 12])
def test_actual_transfer_width_caps_groups_without_losing_primitive_tails(width):
    p, r, s = fixture(shape=(8, 12, 16))
    original = law(p)
    a = adapter(p, load_grouping=replace(original, limits=replace(original.limits, max_row_bytes=width)))
    result = schedule(p, r, s, a=a)
    assert result is not None
    groups = []

    def inspect(call):
        operation = project(call)
        if type(operation) is LoadGroup:
            groups.append(operation)
            element_bytes = p.lhs_bytes if operation.loads[0].role == "lhs" else p.rhs_bytes
            assert sum(load.cols.value * element_bytes for load in operation.loads) <= width
        return call

    change_calls(result.kernel, inspect)
    if width >= 8:
        assert groups
    result.revalidate()
