"""Source quantization, halo lifetimes, tails and real compact LLVM emission."""

import operator
from dataclasses import replace

import numpy as np
import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, StringAttr, UnregisteredOp

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.spatial_producer import (SpatialProducerPlan, SpatialProducerBand,SpatialProducerBandAdmission,
    emit_spatial_producer,emit_spatial_producer_band,spatial_producer_band_choices)
from merlin.compile.scheduling.spatial_readout import (
    ReadoutCompositionUnavailable,
    RoundedReadout,
    SpatialReadoutChain,
    prove_spatial_readout,
)
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
from merlin.llvmlower.host_builder import FnBuilder
from merlin.llvmlower.requantization import f32, quantized


class Integers:
    def op(self, name, a, b, dtype="i64"):
        return {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "udiv": operator.floordiv,
            "urem": operator.mod,
            "and": operator.and_,
        }[name](a, b)

    def cmp(self, name, a, b):
        if name != "slt":
            a %= 1 << 64
            b %= 1 << 64
        return {"slt": operator.lt, "ult": operator.lt, "ule": operator.le, "eq": operator.eq}[name](a, b)

    def select(self, predicate, a, b):
        return a if predicate else b

    def loop(self, start, stop, step, body):
        for index in range(start, stop, step):
            body(index)

    def branch(self, predicate, yes, no):
        (yes if predicate else no)()


def chain(scale=0.25, pool=1.0, *, raw=False, relu=False):
    first = RoundedReadout((scale,), relu)
    second = RoundedReadout((pool,))
    return SpatialReadoutChain(None if raw else first, second, second if raw else first)


@pytest.mark.parametrize(
    "readout",
    [
        chain(),
        chain(raw=True, pool=0.5),
        SpatialReadoutChain(RoundedReadout((1.0,)), RoundedReadout((2.0,)), RoundedReadout((2.0,))),
    ],
)
def test_complete_readout_composition_legal_cases(readout):
    proof = prove_spatial_readout(readout)
    assert proof["exact"] and proof["domain"] == [-(1 << 31), (1 << 31) - 1]
    assert proof["checked_partition_endpoints"] <= 1022


@pytest.mark.parametrize(
    "scale", [0.001407368341460824, 0.0020583588629961014, 0.0011235915590077639, 0.0023733836133033037]
)
def test_resnet_independent_rounded_scales_cannot_be_multiplied(scale):
    pool = 0.9900199770927429
    readout = SpatialReadoutChain(
        RoundedReadout((scale,), True), RoundedReadout((pool,)), RoundedReadout((f32(f32(scale) * f32(pool)),), True)
    )
    with pytest.raises(ReadoutCompositionUnavailable) as error:
        prove_spatial_readout(readout)
    witness = error.value.witness
    first = quantized(witness["accumulator"], (f32(scale),), True)
    assert witness["source"] == quantized(first, (f32(pool),))
    assert witness["source"] != witness["replacement"]


def small_plan(*, raw=False, bias=True, staged=False):
    window = Window2D((2, 6, 7, 3), (3, 3), padding=(1, 1, 1, 1))
    producer = CountedConvolutionPlan(window, 5, 4, 4, 4, 3, 32, 30, 0, 8, 3, bias)
    spatial = SpatialMaxReadout((6, 7), (3, 3), (2, 2), (1, 1, 1, 1), -128)
    readouts = (chain(raw=raw, relu=staged), chain(0.125, raw=raw, relu=staged))
    if staged:
        readouts = tuple(replace(q, pool=RoundedReadout((0.9900199770927429,)), replacement=None) for q in readouts)
    return SpatialProducerPlan(
        producer,
        spatial,
        readouts,
        3,
        readout_mode="staged_readouts" if staged else "single_readout",
        intermediate=AddressWindow("invocation_private_i8", 0, 84) if staged else None,
    )


@pytest.mark.parametrize("raw,staged", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("biased", [False, True])
def test_actual_effects_equal_original_convolution_and_pool(raw, biased, staged, band_rows=1,rhs_mode=False,
                                                           selected=None,selected_rhs=None,trace=None,force_band=False):
    plan = small_plan(raw=raw, bias=biased, staged=staged)if selected is None else selected
    if band_rows>1 and selected is None:
        plan=replace(plan,producer=replace(plan.producer,accumulator_rows=80),
                     intermediate=replace(plan.intermediate,size=280))
    if rhs_mode and selected is None:plan=replace(plan,producer=replace(plan.producer,operand_rows=128))
    band=SpatialProducerBand(plan,band_rows)if band_rows>1 or force_band else None
    storage_rows=band.temporary_rows if band else plan.temporary_rows
    p, s = plan.producer, plan.spatial
    image = (np.arange(np.prod(p.window.input_shape)) % 17 - 8).astype(np.int8).reshape(p.window.input_shape)
    weight = (np.arange(3 * 3 * 3 * 5) % 13 - 6).astype(np.int8).reshape(3, 3, 3, 5)
    bias = np.array([-(1 << 31), (1 << 31) - 1, -19, 0, 31], dtype=np.int32)
    lhs = np.zeros((p.row_tile, p.channel_tile), dtype=np.int8)
    acc = np.zeros((p.accumulator_rows, p.column_tile), dtype=np.int32)
    raw_written = np.zeros_like(acc, dtype=bool)
    scratch = np.zeros((storage_rows, p.column_tile), dtype=np.int8)
    scratch_writes = np.zeros_like(scratch, dtype=int)
    oh, ow = s.output_hw
    output = np.zeros((2 * oh * ow, 5), dtype=np.int8)
    writes = np.zeros_like(output, dtype=int)
    rhs = {}

    def lower(kind, a):
        if trace is not None:trace.append((kind,dict(a)))
        if kind == "initialize_bias":
            row, col = a["accumulator_row"], a["output_column"]
            acc[row : row + a["rows"], : a["columns"]] = bias[col : col + a["columns"]]
        elif kind == "load_input":
            for row in range(a["rows"]):
                first = a["first_element"] + row * a["row_stride_elements"]
                lhs[a["local_row"] + row, : a["columns"]] = (
                    0 if a["zero"] else image.ravel()[first : first + a["columns"]]
                )
        elif kind == "load_weight":
            tile = weight.reshape(-1, 5)[
                a["reduction_row"] : a["reduction_row"] + a["rows"],
                a["output_column"] : a["output_column"] + a["columns"],
            ]
            for r in range(a['rows']):rhs[a['local_row']+r]=tile[r]
        elif kind == "compute":
            row, rows, columns = a["accumulator_row"], a["rows"], a["columns"]
            assert p.accumulator_row <= row < row + rows <= p.accumulator_row + storage_rows
            weights=np.stack([rhs[a['rhs_row']+r]for r in range(a['depth'])])
            product = lhs[:rows, : a["depth"]].astype(np.int32) @ weights.astype(np.int32)
            if a["first_reduction"] and not a["bias"]:
                acc[row : row + rows, :columns] = product
            else:
                acc[row : row + rows, :columns] += product
            raw_written[row : row + rows, :columns] = True
        elif kind == "finish_raw_tile":
            pass  # The last compute's synchronous completion preserves private raw rows.
        elif kind == "store_intermediate":
            row = a["accumulator_row"]
            local = (a["scratch_offset"] - plan.intermediate.start) // p.column_tile
            q = plan.readouts[a["readout_tile"]].producer
            tile = acc[row : row + a["rows"], : a["columns"]]
            scratch[local : local + a["rows"], : a["columns"]] = np.array(
                [quantized(int(value), q.scales, q.relu) for value in tile.ravel()], dtype=np.int8
            ).reshape(tile.shape)
            scratch_writes[local : local + a["rows"], : a["columns"]] += 1
        elif kind == "reload_intermediate":
            row = a["accumulator_row"]
            local = (a["scratch_offset"] - plan.intermediate.start) // p.column_tile
            # Every first readout must finish before any raw row is overwritten.
            assert np.all(scratch_writes[: a["stripe_rows"], : a["columns"]] == 1)
            acc[row : row + a["rows"], : a["columns"]] = scratch[local : local + a["rows"], : a["columns"]]
        elif kind == "store_spatial_max":
            height, width = a["local_height"], a["local_width"]
            start = a["accumulator_row"]
            assert np.all(raw_written[start : start + height * width, : a["columns"]])
            readout = plan.readouts[a["readout_tile"]]
            q = readout.pool if staged else readout.replacement
            tile = acc[start : start + height * width, : a["columns"]].reshape(height, width, a["columns"])
            values = np.array([quantized(int(x), q.scales, q.relu) for x in tile.ravel()], dtype=np.int8).reshape(
                tile.shape
            )
            for y in range(a.get('output_rows',1)):
                for x in range(a["output_pixels"]):
                    row, col = a["output_row"] + y*ow+x, a["output_column"]
                    maximum = values[y*s.stride[0]:y*s.stride[0]+s.window[0],
                        x*s.stride[1]:x*s.stride[1]+s.window[1]].max(axis=(0, 1))
                    output[row, col : col + a["columns"]] = maximum
                    writes[row, col : col + a["columns"]] += 1
            raw_written.fill(False)
            scratch_writes.fill(0)
        else:
            raise AssertionError(kind)

    admission=(SpatialProducerBandAdmission(plan.identity,'neutral-test','TEST-ONLY-rectangular-pool',
        3,3,7,7,81,True,True,True,True)if band else None)
    resident=selected_rhs
    if rhs_mode and resident is None:
        from merlin.compile.scheduling.convolution_exploration import (
            ConvolutionExplorationSource,ConvolutionScheduleChoice,ConvolutionRhsResidencyLaw,
            plan_convolution_schedule,plan_convolution_rhs_residency)
        src=ConvolutionExplorationSource(p,((0,4),(4,1)),plan.identity)
        schedule=plan_convolution_schedule(src,ConvolutionScheduleChoice())
        law=ConvolutionRhsResidencyLaw('neutral-test','original-i32',src.identity,
            'TEST-ONLY-synchronous-immutable-RHS',True,True,True,True,True)
        resident=plan_convolution_rhs_residency(schedule,operand_rows=p.operand_rows,law=law)
        assert resident is not None
    receipt=(emit_spatial_producer_band(band,Integers(),lower,synchronous_completion=True,admission=admission,
        rhs=resident,rhs_complete=lambda *_:None)if band else
             emit_spatial_producer(plan, Integers(), lower, synchronous_completion=True))
    expected = np.zeros((2, 6, 7, 5), dtype=np.int32)
    padded = np.pad(image.astype(np.int64), ((0, 0), (1, 1), (1, 1), (0, 0)))
    for b in range(2):
        for y in range(6):
            for x in range(7):
                expected[b, y, x] = (
                    (padded[b, y : y + 3, x : x + 3, :, None] * weight).sum(axis=(0, 1, 2))
                    + (bias.astype(np.int64) if biased else 0)
                ).astype(np.int32)
    quantized_image = np.empty_like(expected, dtype=np.int8)
    for index in np.ndindex(expected.shape):
        q = plan.readouts[index[-1] // p.column_tile]
        value = int(expected[index])
        if q.producer is not None:
            value = quantized(value, q.producer.scales, q.producer.relu)
        quantized_image[index] = quantized(value, q.pool.scales, q.pool.relu)
    padded_q = np.pad(quantized_image, ((0, 0), (1, 1), (1, 1), (0, 0)), constant_values=-128)
    pooled = np.stack(
        [
            padded_q[b, y * 2 : y * 2 + 3, x * 2 : x * 2 + 3].max(axis=(0, 1))
            for b in range(2)
            for y in range(oh)
            for x in range(ow)
        ]
    )
    assert np.array_equal(output, pooled) and np.all(writes == 1)
    assert receipt["full_intermediate_image_bytes"] == receipt["host_payload_operations"] == 0


@pytest.mark.parametrize('rows',[2,3])
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('resident_rhs',[False,True])
def test_bands_retain_both_readouts_original_products_and_output_tails(rows,bias,resident_rhs):
    test_actual_effects_equal_original_convolution_and_pool(False,bias,True,rows,resident_rhs)


def test_band_choices_bind_actual_scratch_acc_and_recomputation_before_effects():
    p=small_plan(staged=True)
    assert [b.output_rows for b in spatial_producer_band_choices(p,max_output_rows=3)]==[1]
    roomy=replace(p,producer=replace(p.producer,accumulator_rows=80),
                  intermediate=replace(p.intermediate,size=280))
    choices=spatial_producer_band_choices(roomy,max_output_rows=3)
    assert [b.output_rows for b in choices]==[1,2,3]
    assert choices[1].report()['producer_halo_pixels']<choices[0].report()['producer_halo_pixels']
    with pytest.raises(ValueError,match='capacity'):SpatialProducerBand(p,2)
    with pytest.raises(ValueError,match='readouts'):SpatialProducerBand(small_plan(),2)
    band=choices[1];events=[]
    object.__setattr__(band.source.intermediate,'size',1)
    with pytest.raises(ValueError):emit_spatial_producer_band(band,Integers(),lambda *a:events.append(a),synchronous_completion=True,admission=None)
    assert not events


@pytest.mark.parametrize('change',['rectangle','tracking','source','readouts'])
def test_band_target_law_refuses_before_real_source_effects(change):
    p=small_plan(staged=True)
    p=replace(p,producer=replace(p.producer,accumulator_rows=80),intermediate=replace(p.intermediate,size=280))
    band=SpatialProducerBand(p,2)
    admission=SpatialProducerBandAdmission(p.identity,'neutral-test','TEST-ONLY-completed-pool',
        2,3,5,7,54,True,True,True,True)
    if change=='rectangle':admission=replace(admission,rectangular_pool_mapping=False)
    elif change=='tracking':admission=replace(admission,maximum_pool_row_requests=53)
    elif change=='source':admission=replace(admission,source_identity='changed-source')
    else:admission=replace(admission,original_q1_q2=False)
    events=[]
    with pytest.raises(ValueError,match='admission'):
        emit_spatial_producer_band(band,Integers(),lambda *a:events.append(a),
                                  synchronous_completion=True,admission=admission)
    assert not events


def test_spatial_registry_combines_band_and_rhs_orders_and_deduplicates_actual_effects():
    import json,hashlib
    from merlin.compile.scheduling.spatial_producer import spatial_producer_pass_space,explore_spatial_producer
    from merlin.compile.scheduling.convolution_exploration import (
        ConvolutionExplorationSource,ConvolutionScheduleChoice,ConvolutionRhsResidencyLaw,
        plan_convolution_schedule,plan_convolution_rhs_residency)
    from merlin.transforms.pass_exploration import EmittedProgram
    from merlin.transforms.pipeline_state import PipelineContext
    source=small_plan(staged=True)
    source=replace(source,producer=replace(source.producer,accumulator_rows=80,operand_rows=128),
                   intermediate=replace(source.intermediate,size=280))
    original=ConvolutionExplorationSource(source.producer,((0,4),(4,1)),source.identity)
    schedule=plan_convolution_schedule(original,ConvolutionScheduleChoice())
    law=ConvolutionRhsResidencyLaw('neutral-test','original-i32',original.identity,
        'TEST-ONLY-synchronous-immutable-RHS',True,True,True,True,True)
    rhs=plan_convolution_rhs_residency(schedule,operand_rows=128,law=law)
    def realize(value):
        effects=[]
        test_actual_effects_equal_original_convolution_and_pool(False,True,True,value.band.output_rows,
            value.rhs is not None,selected=value.band.source,selected_rhs=value.rhs,trace=effects,force_band=True)
        return EmittedProgram(json.dumps(effects).encode(),'actual-original-spatial-effects',value.identity)
    space=spatial_producer_pass_space(source,output_row_limits=(1,2),rhs_choices=(None,rhs),
        verify_source=lambda p:None,realize=realize,max_candidates=8)
    seen=[]
    def feedback(value,receipt):
        seen.append(value.data)
        return dict(program_sha256=hashlib.sha256(value.data).hexdigest(),cycles=None)
    results=list(explore_spatial_producer(source,space,context=PipelineContext(),verify_source=lambda p:None,
        feedback=feedback))
    assert len(results)==8 and all(r.evaluation.status=='compiled'for r in results)
    assert len(seen)==4 and len({r.emission.data for r in results})==4


@pytest.mark.parametrize(
    "change,match",
    [
        ({"external_observers": ("producer_return",)}, "external observers"),
        ({"producer": replace(small_plan().producer, accumulator_rows=23)}, "capacity"),
        ({"readouts": (chain(),)}, "channel tile"),
    ],
)
def test_unknown_source_or_resources_refuse(change, match):
    with pytest.raises(ValueError, match=match):
        replace(small_plan(), **change)


def test_completion_and_source_mutation_refuse():
    p = small_plan()
    with pytest.raises(ValueError, match="synchronous"):
        emit_spatial_producer(p, Integers(), lambda *_: None, synchronous_completion=False)

    def mutate(*_):
        object.__setattr__(p, "external_observers", ("additional_live_use",))

    with pytest.raises(ValueError, match="external observers"):
        emit_spatial_producer(p, Integers(), mutate, synchronous_completion=True)


def test_unknown_padding_and_mutated_readout_refuse():
    p = small_plan()
    with pytest.raises(ValueError, match="minimum-valued"):
        replace(p, spatial=replace(p.spatial, pad_value=0))
    with pytest.raises(ValueError, match="completely padded"):
        replace(p, spatial=replace(p.spatial, padding=(3, 1, 3, 1)))
    object.__setattr__(p.readouts[0].pool, "relu", "unknown")
    with pytest.raises(ValueError, match="explicit activation"):
        emit_spatial_producer(p, Integers(), lambda *_: None, synchronous_completion=True)


def test_staged_readouts_require_complete_explicit_private_scratch():
    p = small_plan(staged=True)
    with pytest.raises(ValueError, match="explicit private"):
        replace(p, intermediate=None)
    with pytest.raises(ValueError, match="complete i8 halo"):
        replace(p, intermediate=AddressWindow("private", 0, 83))
    with pytest.raises(ValueError, match="collapsed replacement"):
        replace(p, readouts=(replace(p.readouts[0], replacement=RoundedReadout((1.0,))), p.readouts[1]))
    with pytest.raises(ValueError, match="rounded i8 producer"):
        replace(p, readouts=(replace(p.readouts[0], producer=None), p.readouts[1]))


def test_large_spatial_plan_emits_real_compact_llvm_without_payload_operations():
    window = Window2D((1, 224, 224, 3), (7, 7), (2, 2), padding=(3, 3, 3, 3))
    producer = CountedConvolutionPlan(window, 64, 16, 16, 16, 3, 16384, 1024, 0, 16, 0, True)
    spatial = SpatialMaxReadout((112, 112), (3, 3), (2, 2), (1, 1, 1, 1), -128)
    plan = SpatialProducerPlan(producer, spatial, (chain(raw=True),) * 4, 16)
    builder = FnBuilder([])
    index = FnBuilderControlAdapter(builder)
    effects = []

    def lower(kind, fields):
        operands = [
            index.value(value) if type(value) is int else value for value in fields.values() if type(value) is not bool
        ]
        effects.append(kind)
        builder.add(
            UnregisteredOp.with_name("halo_test.effect").create(
                operands=operands, attributes={"kind": StringAttr(kind)}
            )
        )

    receipt = emit_spatial_producer(plan, index, lower, synchronous_completion=True)
    function = llvm.FuncOp("halo", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish())
    module = ModuleOp([function])
    module.verify()
    names = [op.name for op in module.walk()]
    assert "llvm.load" not in names and "llvm.store" not in names
    assert len(effects) < 160 and len(list(module.walk())) < 6000
    assert receipt["private_accumulator_rows"] == 3 * 33
    assert receipt["output_elements"] == 56 * 56 * 64
    assert receipt["full_intermediate_image_bytes"] == 0
