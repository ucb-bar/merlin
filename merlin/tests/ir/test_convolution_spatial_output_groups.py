"""Whole original rolling sources through checked cross-spatial array reuse."""

from dataclasses import replace

import numpy as np
import pytest
from test_convolution_column_retention import law as retention_law
from test_convolution_compute_epochs import epoch_law
from test_convolution_exploration import issue_law
from test_convolution_readout_epochs import readout_law
from test_convolution_rhs_residency import rhs_law
from test_counted_convolution import Integers

from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource,
    ConvolutionRhsPhaseLaw,
    ConvolutionScheduleChoice,
    emit_convolution_schedule,
    plan_convolution_rhs_phases,
    plan_convolution_rhs_residency,
    plan_convolution_schedule,
)
from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_output_group
from merlin.compile.scheduling.dense_array_retention import DenseArrayRetentionAdmission
from merlin.compile.scheduling.window_transfer import ConvolutionResidentInputRows, CountedConvolutionPlan, Window2D


def selected(shape=(1, 5, 11, 5), padding=(1, 1, 1, 1), *, bias=True, slots=3, max_output_rows=1,
             stride=(1, 1), dilation=(1, 1)):
    w = Window2D(shape, (3, 3), padding=padding, stride=stride, dilation=dilation)
    p = CountedConvolutionPlan(w, 9, 4, min(4, w.output_spatial[1]), 4, 4, 1024, 128, 0, 8, 0, bias)
    source = ConvolutionExplorationSource(p, ((0, 4), (4, 4), (8, 1)), "independent-original-source")
    s = plan_convolution_schedule(
        source,
        ConvolutionScheduleChoice(3, True, 200),
        retention_law=replace(retention_law(), independent_output_rows=True, no_intermediate_observers=True),
        issue_law=replace(issue_law(), load_issue_budget=64),
        epoch_law=replace(epoch_law(), max_compute_pairs=200),
    )
    rhs = plan_convolution_rhs_residency(s, operand_rows=1024, rhs_base=512, law=rhs_law())
    inp = ConvolutionResidentInputRows(
        w,
        source.identity,
        4,
        32,
        shape[2] + padding[1] + padding[3] + 2,
        slots,
        32,
        480,
        4,
        halo_value=0 if any(padding) else None,
    )
    group = plan_rolling_convolution_output_group(s, rhs, inp, max_output_rows=max_output_rows)
    array = DenseArrayRetentionAdmission("neutral", "source-i32", "TEST-ONLY-original-fresh-stay")
    readout = replace(readout_law(), max_issued_readouts=48)
    return s, rhs, inp, group, array, readout


def run(s, rhs, inp, group, array, readout, *, pending_limit=0, ready_readout=False, trace=None,
        readout_transfer_law=None, grouped_transport=False,rhs_phase_prefetch_law=None,input_prefetch_law=None,
        batch_input_prefetch_law=None, input_first_use_law=None):
    p = s.source.plan
    shape = p.window.input_shape
    ci = shape[-1]
    co = p.output_channels
    image = (np.arange(np.prod(shape)) % 7 - 3).reshape(shape).astype(np.int8)
    weights = (np.arange(9 * ci * co) % 5 - 2).reshape(9 * ci, co).astype(np.int8)
    bias = np.arange(co, dtype=np.int32) - 4
    sp = {}
    acc = {}
    ids = {}
    written = {}
    products = {}
    events = []
    fresh = []
    retained = None

    def execute(kind, a):
        nonlocal retained
        events.append("issue:" + kind)
        if trace is not None:
            trace.append((kind, dict(a)))
        if kind in ("stage_input_rows", "fill_input_rows"):
            for r in range(a["rows"]):
                for c in range(a["columns"]):
                    sp[a["local_row"] + (c // 4) * a["block_stride_rows"] + r, c % 4] = (
                        a["value"]
                        if kind == "fill_input_rows"
                        else int(image.ravel()[a["first_element"] + r * a["row_stride_elements"] + c])
                    )
        elif kind == "load_weight":
            for r in range(a["rows"]):
                for c in range(a["columns"]):
                    sp[a["local_row"] + r, c] = int(weights[a["reduction_row"] + r, a["output_column"] + c])
            ids[a["local_row"]] = (a["reduction_row"], a["output_column"])
        elif kind == "initialize_bias":
            for r in range(a['rows']):
                for c in range(a['columns']):
                    acc[a['accumulator_row']+r,c]=bias[a['output_column']+c]
        elif kind == "compute":
            identity = ids[a["rhs_row"]], a["depth"], a["columns"]
            if a.get("fresh_weights", True):
                retained = identity
            else:
                assert retained == identity
            fresh.append(a.get("fresh_weights", True))
            lhs = np.array(
                [[sp[a["lhs_row"] + r * inp.row_step, c] for c in range(a["depth"])] for r in range(a["rows"])]
            )
            weight = np.array([[sp[a["rhs_row"] + r, c] for c in range(a["columns"])] for r in range(a["depth"])])
            value = lhs.astype(np.int64) @ weight.astype(np.int64)
            for r in range(a['rows']):
                for c in range(a['columns']):
                    key=a['accumulator_row']+r,c
                    acc[key]=(value[r,c]if a['first_reduction']and not p.bias else acc[key]+value[r,c])
            for r in range(a["rows"]):
                for c in range(a["columns"]):
                    products.setdefault((a["output_row"] + r, a["output_column"] + c), []).append(ids[a["rhs_row"]][0])
        elif kind == "store":
            for r in range(a["rows"]):
                for c in range(a["columns"]):
                    key = a["output_row"] + r, a["output_column"] + c
                    assert key not in written
                    assert products[key] == [tap * ci + k for tap in range(9) for k in range(0, ci, 4)]
                    step = p.primitive if readout_transfer_law is None else readout_transfer_law.block_rows
                    written[key] = acc[a["accumulator_row"] + (c // p.column_tile) * step+r,c % p.column_tile]
        else:
            raise AssertionError(kind)

    pending = []

    def lower(kind, args):
        if not pending_limit:
            execute(kind, args)
        else:
            if len(pending) == pending_limit:
                execute(*pending.pop(0))
            pending.append((kind, args))

    def complete(kind, args):
        while pending:
            execute(*pending.pop(0))
        events.append(kind)
        if trace is not None:
            trace.append((kind, dict(args)))

    class CarriedIntegers(Integers):
        def loop_carried(self,lo,hi,step,initial,body):
            value=initial
            for i in range(lo,hi,step):value=body(i,value)
            return value

    report = emit_convolution_schedule(
        s,
        CarriedIntegers(),
        lower,
        complete=complete,
        rhs=rhs,
        input_rows=inp,
        readout_law=readout,
        output_group=group,
        array_law=array if group else None,
        ready_readout=ready_readout,
        readout_transfer_law=readout_transfer_law,
        grouped_transport=grouped_transport,
        input_prefetch_law=input_prefetch_law,
        batch_input_prefetch_law=batch_input_prefetch_law,
        input_first_use_law=input_first_use_law,
        rhs_phase_prefetch_law=rhs_phase_prefetch_law,
    )
    assert not pending
    oh, ow = p.window.output_spatial
    expected = np.zeros((shape[0], oh, ow, co), dtype=np.int64)
    if p.bias:
        expected += bias
    top, left, bottom, right = p.window.padding
    xp = np.pad(image, ((0, 0), (top, bottom), (left, right), (0, 0)))
    for y in range(3):
        for x in range(3):
            sy, sx = p.window.stride
            dy, dx = p.window.dilation
            source = xp[:, y*dy : y*dy + oh*sy : sy, x*dx : x*dx + ow*sx : sx].astype(np.int64)
            expected += source @ weights.reshape(3, 3, ci, co)[y, x].astype(np.int64)
    assert np.array_equal(
        np.array([written[r, c] for r in range(shape[0] * oh * ow) for c in range(co)]).reshape(expected.shape),
        expected,
    )
    original_k = [tap * ci + c for tap in range(9) for c in range(0, ci, 4)]
    assert all(v == original_k for v in products.values())
    return events, fresh, report


@pytest.mark.parametrize('pending_limit',[0,2])
@pytest.mark.parametrize('ready_readout',[False,True])
@pytest.mark.parametrize('bias',[False,True])
def test_direct_ring_fragments_pack_acc_and_keep_original_products(pending_limit,ready_readout,bias):
    from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_row_fragments
    s,rhs,inp,ordinary,array,readout=selected(shape=(2,7,11,5),slots=7,max_output_rows=3,bias=bias)
    packed=plan_rolling_convolution_row_fragments(s,rhs,inp,max_output_rows=3)
    assert packed.row_blocks==9 and ordinary.row_blocks==9
    # Three rows of11 become nine packed slots; source-row crossings split
    # actual ring reads, including row, band, image, K and output-column tails.
    old_trace=[];new_trace=[]
    run(s,rhs,inp,ordinary,array,readout,pending_limit=pending_limit,ready_readout=ready_readout,
        trace=old_trace,grouped_transport=True)
    _,_,receipt=run(s,rhs,inp,packed,array,readout,pending_limit=pending_limit,ready_readout=ready_readout,
        trace=new_trace,grouped_transport=True)
    for kind in ('stage_input_rows','fill_input_rows','initialize_bias','input_ready','input_last_use'):
        assert sum(k==kind for k,a in old_trace)==sum(k==kind for k,a in new_trace)
    assert receipt['spatial_output_group']['row_fragments']['extra_input_bytes']==0
    assert any(a['accumulator_row']%s.source.plan.primitive for k,a in new_trace if k=='compute')


def test_direct_ring_fragment_choice_compacts_without_extra_compute_for_aligned_bands():
    from merlin.compile.scheduling.convolution_output_groups import (
        plan_rolling_convolution_row_fragments,plan_rolling_convolution_output_group)
    s,rhs,inp,ordinary,array,readout=selected(shape=(1,4,14,5),slots=6,max_output_rows=4)
    source=replace(s.source,plan=replace(s.source.plan,accumulator_rows=256))
    s=plan_convolution_schedule(source,s.choice,retention_law=s.retention_law,
                               issue_law=s.issue_law,epoch_law=s.epoch_law)
    rhs=plan_convolution_rhs_residency(s,operand_rows=1024,rhs_base=512,law=rhs_law())
    inp=replace(inp,source_identity=source.identity)
    ordinary=plan_rolling_convolution_output_group(s,rhs,inp,max_output_rows=4)
    packed=plan_rolling_convolution_row_fragments(s,rhs,inp,max_output_rows=4)
    assert ordinary.row_blocks==16 and packed.row_blocks==14
    old=[];new=[]
    run(s,rhs,inp,ordinary,array,readout,trace=old,grouped_transport=True,pending_limit=2)
    run(s,rhs,inp,packed,array,readout,trace=new,grouped_transport=True,pending_limit=2)
    assert sum(k=='compute'for k,a in old)==sum(k=='compute'for k,a in new)
    assert sum(k=='store'for k,a in new)<sum(k=='store'for k,a in old)


@pytest.mark.parametrize('mutation',['source','gather','capacity'])
def test_direct_ring_fragments_reject_changed_source_or_storage_before_effects(mutation):
    from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_row_fragments
    s,rhs,inp,ordinary,array,readout=selected(shape=(2,7,11,5),slots=7,max_output_rows=3)
    packed=plan_rolling_convolution_row_fragments(s,rhs,inp,max_output_rows=3)
    if mutation=='source':packed=replace(packed,source_identity='0'*64)
    elif mutation=='gather':packed=replace(packed,gather=replace(packed.gather,dilation=(1,2),kernel=(3,2)))
    else:object.__setattr__(packed.panel,'accumulator_capacity_rows',1)
    trace=[]
    with pytest.raises(ValueError):run(s,rhs,inp,packed,array,readout,trace=trace,grouped_transport=True)
    assert not trace


@pytest.mark.parametrize(
    "shape,padding", [((1, 5, 11, 5), (1, 1, 1, 1)), ((2, 6, 14, 8), (0, 0, 0, 0)), ((1, 5, 20, 3), (2, 1, 0, 2))]
)
@pytest.mark.parametrize("bias", [True, False])
def test_all_original_products_outputs_and_ring_lifetimes_survive_spatial_reuse(shape, padding, bias):
    s, rhs, inp, group, array, readout = selected(shape, padding, bias=bias)
    old, old_fresh, _ = run(s, rhs, inp, None, array, readout)
    new, new_fresh, report = run(s, rhs, inp, group, array, readout)
    assert len(new_fresh) == len(old_fresh) and sum(new_fresh) < sum(old_fresh)
    assert new.count("input_ready") == old.count("input_ready")
    assert new.count("input_last_use") == old.count("input_last_use")
    assert new.count("issue:stage_input_rows") == old.count("issue:stage_input_rows")
    assert new.count("issue:store") == old.count("issue:store")
    assert "spatial_group_last_use" in new
    assert report["spatial_output_group"]["layout"]["row_blocks"] == group.row_blocks
    assert "checked spatial group" in report["completion"]


@pytest.mark.parametrize("change", ["array", "budget", "readout", "rhs", "panel"])
def test_missing_or_changed_source_contract_refuses_before_any_preparation(change):
    s, rhs, inp, group, array, readout = selected()
    if change == "array":
        array = replace(array, hardware_identity="other")
    if change == "budget":
        readout = replace(readout, max_issued_readouts=1)
    if change == "readout":
        readout = None
    if change == "rhs":
        rhs = None
    if change == "panel":
        group = replace(group, panel=replace(group.panel, accumulator_capacity_rows=64))
    effects = []
    with pytest.raises(ValueError):
        emit_convolution_schedule(
            s,
            Integers(),
            lambda *a: effects.append(a),
            complete=lambda *a: effects.append(a),
            rhs=rhs,
            input_rows=inp,
            readout_law=readout,
            output_group=group,
            array_law=array,
        )
    assert effects == []


def test_single_row_tile_and_missing_residency_decline_without_changing_default():
    s, rhs, inp, group, array, readout = selected((1, 5, 3, 5))
    assert group is None
    assert plan_rolling_convolution_output_group(s, None, inp) is None
    assert plan_rolling_convolution_output_group(s, rhs, None) is None


@pytest.mark.parametrize("permission", ["independent_output_rows", "no_intermediate_observers"])
def test_spatial_order_is_not_granted_by_channel_retention(permission):
    s, rhs, inp, group, array, readout = selected()
    s = replace(s, retention_law=replace(s.retention_law, **{permission: False}))
    rhs = replace(rhs, schedule=s)
    effects = []
    with pytest.raises(ValueError, match="independent-row and observer"):
        emit_convolution_schedule(
            s,
            Integers(),
            lambda *a: effects.append(a),
            complete=lambda *a: effects.append(a),
            rhs=rhs,
            input_rows=inp,
            readout_law=readout,
            output_group=group,
            array_law=array,
        )
    assert effects == []


@pytest.mark.parametrize("shape,padding,slots,rows", [
    ((1, 5, 11, 5), (1, 1, 1, 1), 4, 2),
    ((2, 7, 9, 3), (0, 0, 0, 0), 5, 3),
    ((1, 6, 6, 8), (2, 1, 0, 2), 6, 4),
])
def test_multirow_delayed_products_preserve_every_cell_tail_and_ring_version(shape, padding, slots, rows):
    s, rhs, inp, group, array, readout = selected(
        shape, padding, slots=slots, max_output_rows=4)
    assert group.output_rows == rows
    old, old_fresh, _ = run(s, rhs, inp, None, array, readout)
    new, new_fresh, report = run(s, rhs, inp, group, array, readout)
    assert len(new_fresh) == len(old_fresh)
    assert sum(new_fresh) < sum(old_fresh)
    for kind in ("input_ready", "input_last_use", "issue:stage_input_rows", "issue:store"):
        assert new.count(kind) == old.count(kind)
    assert report["spatial_output_group"]["layout"]["output_rows"] == rows
    assert "ring union" in report["spatial_output_group"]["iteration"]


def test_multirow_capacity_and_ring_crossover_retains_one_row_incumbent():
    s, rhs, inp, old, array, readout = selected()
    # Three slots cover one 3x3 row only: a request cannot invent a fourth.
    assert plan_rolling_convolution_output_group(s, rhs, inp, max_output_rows=4) == old
    bigger = replace(inp, slots=4)
    group = plan_rolling_convolution_output_group(s, rhs, bigger, max_output_rows=4)
    assert group.output_rows == 2
    # Four M slots cannot cover two complete 11-wide rows (three tiles each).
    assert plan_rolling_convolution_output_group(
        s, rhs, bigger, max_row_blocks=4, max_output_rows=4).output_rows == 1
    effects = []
    with pytest.raises(ValueError, match="original rolling/RHS"):
        emit_convolution_schedule(s, Integers(), lambda *a: effects.append(a),
            complete=lambda *a: effects.append(a), rhs=rhs, input_rows=inp,
            output_group=group, array_law=array, readout_law=readout)
    assert effects == []


@pytest.mark.parametrize("rows", [0, True, 17, "2"])
def test_output_row_bounds_are_explicit(rows):
    s, rhs, inp, *_ = selected()
    with pytest.raises(ValueError, match="output-row bound"):
        plan_rolling_convolution_output_group(s, rhs, inp, max_output_rows=rows)


def test_dilated_vertical_stride_uses_the_full_source_row_horizon():
    s, rhs, inp, group, array, readout = selected((2, 9, 7, 3), slots=7,
        stride=(2, 1), dilation=(2, 1), max_output_rows=4)
    assert group.output_rows == 2  # span5 + one stride2, not kernel height3
    _, _, report = run(s, rhs, inp, group, array, readout)
    assert report["spatial_output_group"]["ring_read_horizon"]["source_row_span"] == 7
    assert plan_rolling_convolution_output_group(
        s, rhs, replace(inp, slots=6), max_output_rows=4).output_rows == 1


def test_short_source_rows_retain_primitive_sized_disjoint_acc_slots():
    s, rhs, inp, group, array, readout = selected((2, 5, 3, 5), slots=4, max_output_rows=2)
    assert s.source.plan.row_tile == 3 and group.panel.tile == 4
    assert group.output_rows == 2 and group.row_blocks == 2
    assert group.accumulator_row(1, 0) == 12  # three primitive column slots
    run(s, rhs, inp, group, array, readout, pending_limit=2)


@pytest.mark.parametrize("phases", [False, True])
@pytest.mark.parametrize("ready_readout", [False, True])
def test_multirow_uses_actual_counted_cfg_and_real_completion(phases, ready_readout):
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp

    from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
    from merlin.llvmlower.host_builder import FnBuilder

    s, rhs, inp, group, array, readout = phased() if phases else selected(slots=4, max_output_rows=2)
    fn = FnBuilder([])
    computes, stores = set(), []

    def lower(kind, args):
        op = llvm.InlineAsmOp("# " + kind, "~{memory}", [], has_side_effects=True)
        fn.add(op)
        if kind == "compute":
            computes.add(op)
        elif kind == "store":
            stores.append(op)

    emit_convolution_schedule(s, FnBuilderControlAdapter(fn), lower,
        complete=lambda *_: fn.add(llvm.InlineAsmOp("fence", "~{memory}", [], has_side_effects=True)),
        rhs=rhs, input_rows=inp, readout_law=readout, output_group=group, array_law=array,
        ready_readout=ready_readout)
    module = ModuleOp([llvm.FuncOp("kernel", llvm.LLVMFunctionType([], llvm.LLVMVoidType()),
        linkage=llvm.LinkageAttr("external"), body=fn.finish())])
    module.verify()
    assert any(isinstance(op, llvm.CondBrOp) for op in module.walk())
    assert any(isinstance(op, llvm.InlineAsmOp) for op in module.walk())
    if ready_readout:
        blocks = module.body.block.first_op.body.blocks
        # Every actual store block has a final-K conditional edge directly
        # after its actual producer effect, rather than after the K loops.
        assert stores
        for store in stores:
            assert any(isinstance(b.last_op, llvm.CondBrOp) and store.parent in b.last_op.successors
                       and b.last_op.prev_op in computes for b in blocks)


def phased(*, multirow=True):
    s, _, inp, _, array, readout = selected(slots=4 if multirow else 3)
    law = ConvolutionRhsPhaseLaw(s.source.identity, "TEST-ONLY original phase completion",
        independent_output_columns=True, immutable_inputs=True, no_intermediate_observers=True,
        full_completion_before_reprepare=True)
    rhs = plan_convolution_rhs_phases(s, operand_rows=320, law=rhs_law(), phase_law=law, input_rows=inp)
    groups = plan_rolling_convolution_output_group(s, rhs, inp, max_output_rows=2 if multirow else 1)
    return s, rhs, inp, groups, array, readout


@pytest.mark.parametrize("multirow", [False, True])
def test_active_rhs_phases_keep_column_tail_relocations_and_all_real_drains(multirow):
    s, rhs, inp, groups, array, readout = phased(multirow=multirow)
    assert [phase.column_intervals for phase in rhs.phases] == [((0, 4), (4, 4)), ((8, 1),)]
    assert [group.panel.column_blocks for group in groups] == [2, 1]
    assert all(group.panel.operand_capacity_rows == 320 for group in groups)
    assert all(group.output_rows == (2 if multirow else 1) for group in groups)
    old, old_fresh, _ = run(s, rhs, inp, None, array, readout, pending_limit=2)
    new, new_fresh, report = run(s, rhs, inp, groups, array, readout, pending_limit=2)
    assert len(new_fresh) == len(old_fresh) and sum(new_fresh) < sum(old_fresh)
    assert new.count("rhs_phase_last_use") == new.count("weight_preparation") == 2
    assert len(report["phase_emissions"]) == 2
    for result, group, phase in zip(report["phase_emissions"], groups, rhs.phases, strict=True):
        assert result["spatial_output_group"]["layout"]["panel"]["rhs_rows"] == phase.layout.rhs_rows
        assert result["spatial_output_group"]["layout"]["row_blocks"] == group.row_blocks
    for kind in ("input_ready", "input_last_use", "issue:store"):
        assert new.count(kind) == old.count(kind)


def test_unselected_phase_stays_ordinary_and_partial_member_is_not_an_entry():
    s, rhs, inp, groups, array, readout = phased()
    events, _, report = run(s, rhs, inp, (None, groups[1]), array, readout, pending_limit=2)
    assert "spatial_output_group" not in report["phase_emissions"][0]
    assert "spatial_output_group" in report["phase_emissions"][1]
    assert events.count("rhs_phase_last_use") == 2


@pytest.mark.parametrize("change", ["reverse", "missing", "late_budget", "partial"])
def test_invalid_phase_group_mapping_refuses_before_any_target_preparation(change):
    s, rhs, inp, groups, array, readout = phased()
    if change == "reverse":
        groups = groups[::-1]
    elif change == "missing":
        groups = groups[:-1]
    elif change == "late_budget":
        # The unchanged first phase fits; the second group's six issued stores
        # exceed five. No first-phase preparation may leak before the refusal.
        groups = (None, groups[1])
        readout = replace(readout, max_issued_readouts=5)
    else:
        rhs = rhs.phases[0]
        groups = groups[0]
    effects = []
    with pytest.raises(ValueError):
        emit_convolution_schedule(s, Integers(), lambda *a: effects.append(a),
            complete=lambda *a: effects.append(a), rhs=rhs, input_rows=inp,
            output_group=groups, array_law=array, readout_law=readout)
    assert effects == []


@pytest.mark.parametrize("kind", ["one_row", "multirow", "short_row", "phases"])
def test_ready_readout_follows_each_last_original_producer_and_keeps_final_drains(kind):
    if kind == "phases":
        s, rhs, inp, group, array, readout = phased()
    elif kind == "short_row":
        s, rhs, inp, group, array, readout = selected((2, 5, 3, 5), slots=4, max_output_rows=2)
    else:
        s, rhs, inp, group, array, readout = selected(slots=4, max_output_rows=2 if kind == "multirow" else 1)
    old_trace, new_trace = [], []
    old, _, _ = run(s, rhs, inp, group, array, readout, pending_limit=2, trace=old_trace)
    new, _, report = run(s, rhs, inp, group, array, readout, pending_limit=2,
                         ready_readout=True, trace=new_trace)
    for effect in ("issue:compute", "issue:load_weight", "issue:stage_input_rows", "issue:store",
                   "input_ready", "input_last_use", "compute", "spatial_group_last_use", "rhs_phase_last_use"):
        assert old.count(effect) == new.count(effect)
    # Same complete original stores and unchanged compute order, with stores
    # interleaved only after their complete original reduction. A later compute
    # must never overwrite that stored ACC tile before the real group drain.
    def key(a):
        return a["output_row"], a["output_column"], a["rows"], a["columns"]
    assert [a for k, a in old_trace if k == "compute"] == [a for k, a in new_trace if k == "compute"]
    assert sorted(key(a) for k, a in old_trace if k == "store") == sorted(
        key(a) for k, a in new_trace if k == "store")
    compute_after_store = False
    stored = []
    previous = None
    for effect, a in new_trace:
        if effect == "store":
            assert previous[0] == "compute" and key(previous[1]) == key(a)
            stored.append((a["accumulator_row"], a["accumulator_row"] + a["rows"]))
        elif effect == "compute":
            compute_after_store |= bool(stored)
            assert all(a["accumulator_row"] + a["rows"] <= lo or hi <= a["accumulator_row"] for lo, hi in stored)
        elif effect == "spatial_group_last_use":
            stored.clear()
        previous = effect, a
    assert compute_after_store
    children = report.get("phase_emissions", [report])
    assert all(c["spatial_output_group"]["ready_readout"]["applied"] for c in children)


@pytest.mark.parametrize("missing", ["local_read_after_write", "accumulator_result_visibility",
                                    "ordered_configuration_capture", "finite_queue_progress"])
def test_ready_frontier_cannot_replace_missing_store_dependency_law(missing):
    s, rhs, inp, group, array, readout = selected(slots=4, max_output_rows=2)
    effects = []
    with pytest.raises(ValueError):
        emit_convolution_schedule(s, Integers(), lambda *a: effects.append(a),
            complete=lambda *a: effects.append(a), rhs=rhs, input_rows=inp, output_group=group,
            array_law=array, readout_law=replace(readout, **{missing: False}), ready_readout=True)
    assert effects == []


@pytest.mark.parametrize("phases", [False, True])
@pytest.mark.parametrize("ready", [False, True])
def test_spatial_wide_stores_preserve_products_ring_lifetimes_and_phase_tails(phases, ready):
    from test_convolution_readout_transfers import law

    s, rhs, inp, groups, array, readout = phased() if phases else selected(slots=4, max_output_rows=2)
    old_trace, new_trace = [], []
    old, _, _ = run(s, rhs, inp, groups, array, readout, trace=old_trace, pending_limit=2)
    new, _, report = run(s, rhs, inp, groups, array, readout, trace=new_trace, pending_limit=2,
                         ready_readout=ready, readout_transfer_law=law(s))
    # The interpreter verifies every original reduction and output value,
    # including row/channel tails and finite queued execution. Only publication
    # width/order may differ; all original operand transport remains intact.
    assert new.count("issue:store") < old.count("issue:store")
    for kind in ("input_ready", "input_last_use", "weight_preparation", "rhs_phase_last_use",
                 "issue:compute", "issue:stage_input_rows", "spatial_group_last_use"):
        assert new.count(kind) == old.count(kind)
    children = report.get("phase_emissions", [report])
    assert children[0]["readout_transfers"]["physical_stores_per_spatial_tile"] < children[0][
        "readout_transfers"]["original_stores_per_spatial_tile"]


def test_short_row_receipt_uses_actual_primitive_acc_slots_and_preserves_numerical_boundaries():
    from test_convolution_readout_transfers import law

    s, rhs, inp, group, array, readout = selected((2, 5, 3, 5), slots=4, max_output_rows=2)
    assert s.source.plan.row_tile == 3 and group.panel.tile == 4
    selected_law = replace(law(s), block_rows=4)
    _, _, report = run(s, rhs, inp, group, array, readout, readout_transfer_law=selected_law)
    assert report["readout_transfers"]["physical_stores_per_spatial_tile"] == 2
    assert [t["accumulator_row"] for t in report["readout_transfers"]["blocks"][0][0]] == [0, 4]
    _, _, split = run(s, rhs, inp, group, array, readout,
        readout_transfer_law=replace(selected_law, column_states=((0, 4, "a" * 64), (4, 5, "b" * 64))))
    assert split["readout_transfers"]["physical_stores_per_spatial_tile"] == 2
    assert len(split["readout_transfers"]["blocks"][0][0]) == 1


def test_spatial_wide_stores_reject_changed_target_before_preparation():
    from test_convolution_readout_transfers import law

    s, rhs, inp, group, array, readout = selected(slots=4, max_output_rows=2)
    effects = []
    with pytest.raises(ValueError):
        emit_convolution_schedule(s, Integers(), lambda *args: effects.append(args),
            complete=lambda *args: effects.append(args), rhs=rhs, input_rows=inp,
            output_group=group, array_law=array, readout_law=readout,
            readout_transfer_law=replace(law(s), hardware_identity="different-source"))
    assert not effects


@pytest.mark.parametrize("phases", [False, True])
@pytest.mark.parametrize("wide", [False, True])
def test_grouped_transport_preserves_all_source_work_and_removes_empty_row_waits(phases, wide):
    from test_convolution_readout_transfers import law

    s, rhs, inp, groups, array, readout = phased() if phases else selected(slots=4, max_output_rows=2)
    options = dict(readout_transfer_law=law(s)) if wide else {}
    old, _, _ = run(s, rhs, inp, groups, array, readout, pending_limit=2, **options)
    new, _, report = run(s, rhs, inp, groups, array, readout, pending_limit=2,
                         grouped_transport=True, **options)
    for kind in ("issue:compute", "issue:store", "issue:stage_input_rows", "issue:fill_input_rows",
                 "weight_preparation", "rhs_phase_last_use", "spatial_group_last_use"):
        assert old.count(kind) == new.count(kind)
    assert new.count("input_ready") < old.count("input_ready")
    assert new.count("input_last_use") == new.count("input_ready")
    for child in report.get("phase_emissions", [report]):
        rolling = child["rolling_transport"]
        assert rolling["consumer_groups"] == new.count("input_ready") // (2 if phases else 1)
        assert rolling["source_row_union"] <= inp.slots


@pytest.mark.parametrize('pending_limit',[0,2])
@pytest.mark.parametrize('width',[11,21])
def test_invocation_input_prefetch_uses_actual_original_k_run_continuations(pending_limit,width):
    from merlin.compile.scheduling.rolling_lookahead import RollingLookaheadAdmission
    args=selected(shape=(2,9,width,5),slots=6,max_output_rows=2)
    law=RollingLookaheadAdmission('neutral','TEST-ONLY-complete-source-effects',True,True,True,True,True,True)
    trace=[]
    run(*args,pending_limit=pending_limit,grouped_transport=True,input_prefetch_law=law,trace=trace)
    kinds=[kind for kind,_ in trace]
    first=kinds.index('compute');last=len(kinds)-1-kinds[::-1].index('compute')
    assert any(kind in ('stage_input_rows','fill_input_rows')for kind in kinds[first+1:last])


def test_invocation_input_prefetch_refuses_missing_ring_horizon_before_rhs_preparation():
    from merlin.compile.scheduling.rolling_lookahead import RollingLookaheadAdmission
    s,rhs,inp,group,array,readout=selected(shape=(1,9,11,5),slots=4,max_output_rows=2)
    law=RollingLookaheadAdmission('neutral','TEST-ONLY-complete-source-effects',True,True,True,True,True,True)
    trace=[]
    with pytest.raises(ValueError):
        run(s,rhs,inp,group,array,readout,grouped_transport=True,input_prefetch_law=law,trace=trace)
    assert not trace


@pytest.mark.parametrize('pending_limit',[0,2])
@pytest.mark.parametrize('image_windows',[1,2])
def test_invocation_batch_frontier_reuses_only_source_dead_input_rows(pending_limit,image_windows):
    from merlin.compile.scheduling.rolling_lookahead import RollingBatchFrontierAdmission
    args=list(selected(shape=(2,7,9,5),padding=(0,0,0,0),slots=7,max_output_rows=5))
    if image_windows>1:args[2]=replace(args[2],slots=14,image_windows=image_windows)
    assert args[3].output_rows==5
    law=RollingBatchFrontierAdmission('neutral','source-i32','TEST-only original source effects',
                                     True,True,True,True,True,True)
    trace=[]
    run(*args,pending_limit=pending_limit,grouped_transport=True,batch_input_prefetch_law=law,trace=trace)
    kinds=[k for k,_ in trace]
    first=kinds.index('compute');last=len(kinds)-1-kinds[::-1].index('compute')
    assert any(k in ('stage_input_rows','fill_input_rows')for k in kinds[first+1:last])


def test_grouped_transport_admits_actual_transfer_count_before_any_effect():
    s, rhs, inp, groups, array, readout = selected(slots=4, max_output_rows=2, bias=False)
    # The old first row needs15 transfers and the compute-side bound5;
    # a two-row union needs20 transfers, beyond this original admission.
    s = replace(s, issue_law=replace(s.issue_law, load_issue_budget=inp.initial_load_commands))
    rhs = replace(rhs, schedule=s)
    effects = []
    with pytest.raises(ValueError, match="transfer issue bound"):
        emit_convolution_schedule(s, Integers(), lambda *a: effects.append(a),
            complete=lambda *a: effects.append(a), rhs=rhs, input_rows=inp,
            output_group=groups, array_law=array, readout_law=readout, grouped_transport=True)
    assert not effects
