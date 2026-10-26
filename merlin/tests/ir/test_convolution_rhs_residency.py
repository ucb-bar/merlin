"""Invocation-local immutable RHS preparation retains original reductions."""
import json
from dataclasses import replace

import numpy as np
import pytest
from test_convolution_column_retention import law as retention_law
from test_convolution_compute_epochs import epoch_law
from test_convolution_exploration import issue_law
from test_counted_convolution import Integers

from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource,
    ConvolutionRhsResidencyLaw,
    ConvolutionScheduleChoice,
    convolution_pass_space,
    emit_convolution_schedule,
    explore_convolution_schedules,
    plan_convolution_rhs_residency,
    plan_convolution_schedule,
)
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
from merlin.transforms.pass_exploration import EmittedProgram
from merlin.transforms.pipeline_state import PipelineContext


def rhs_law():
    return ConvolutionRhsResidencyLaw("neutral", "source-i32", "original-immutable-weights",
        "separate-source-bound-transfer-and-preservation", weight_transfer_only=True,
        immutable_mapped_disjoint_buffers=True, other_effects_preserve_rhs=True,
        exclusive_private_rows=True, entry_quiescence_and_complete_preparation=True)


def source(bias=False):
    p = CountedConvolutionPlan(Window2D((2, 3, 5, 5), (3, 2), padding=(1, 1, 1, 0)),
        9, 4, 4, 4, 3, 512, 12, 0, 8, 0, bias)
    return ConvolutionExplorationSource(p, ((0, 4), (4, 4), (8, 1)), "original-ci-and-n-tails")


def schedule(bias=False, pairs=4):
    return plan_convolution_schedule(source(bias), ConvolutionScheduleChoice(3, True, pairs),
        retention_law=retention_law(), issue_law=issue_law(), epoch_law=epoch_law())


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("pairs", [1, 4])
@pytest.mark.parametrize("base", [None, 192])
def test_resident_rhs_preserves_complete_source_outputs_and_original_per_output_k_order(bias, pairs, base):
    s = schedule(bias, pairs)
    rhs = plan_convolution_rhs_residency(s, operand_rows=512, law=rhs_law(), rhs_base=base)
    assert rhs.layout.rhs_rows == 6*2*3*3  # each kernel's CI tail has its own slot
    image = (np.arange(150) % 17-8).astype(np.int8).reshape(2, 3, 5, 5)
    weights = (np.arange(270) % 13-6).astype(np.int8).reshape(30, 9)
    biases = np.arange(9, dtype=np.int32)-4
    oh, ow = s.source.plan.window.output_spatial
    actual = np.zeros((2*oh*ow, 9), dtype=np.int32)
    written = np.zeros_like(actual)
    spad, acc, pending, data = {}, {}, [], []
    preparation = []

    def execute(kind, a):
        if kind == "load_weight":
            for row in range(a["rows"]):
                spad[a["local_row"]+row] = weights[a["reduction_row"]+row,
                    a["output_column"]:a["output_column"]+a["columns"]].copy()
        elif kind == "load_input":
            for row in range(a["rows"]):
                first = a["first_element"]+row*a["row_stride_elements"]
                spad[a["local_row"]+row] = (np.zeros(a["columns"], dtype=np.int8) if a["zero"]
                    else image.ravel()[first:first+a["columns"]].copy())
        elif kind == "initialize_bias":
            acc[a["accumulator_row"]] = np.broadcast_to(biases[
                a["output_column"]:a["output_column"]+a["columns"]], (a["rows"], a["columns"])).copy()
        elif kind == "compute":
            lhs = np.stack([spad[a["lhs_row"]+r][:a["depth"]] for r in range(a["rows"])])
            rhs_values = np.stack([spad[a["rhs_row"]+r] for r in range(a["depth"])])
            value = lhs.astype(np.int32) @ rhs_values.astype(np.int32)
            key = a["accumulator_row"]
            acc[key] = value if a["first_reduction"] and not bias else acc[key]+value
        elif kind == "store":
            r, c = a["output_row"], a["output_column"]
            actual[r:r+a["rows"], c:c+a["columns"]] = acc[a["accumulator_row"]]
            written[r:r+a["rows"], c:c+a["columns"]] += 1
        else:
            raise AssertionError(kind)

    def issue(kind, fields):
        data.append((kind, dict(fields)))
        if len(pending) == 2:
            execute(*pending.pop(0))
        pending.append((kind, dict(fields)))

    def complete(kind, fields):
        while pending:
            execute(*pending.pop(0))
        if kind == "weight_preparation":
            preparation.append(len(data))
            assert all(k == "load_weight" for k, _ in data)

    emit_convolution_schedule(s, Integers(), issue, complete=complete, rhs=rhs)
    assert not pending and preparation == [36]
    assert sum(k == "load_weight" for k, _ in data) == 36
    expected = np.zeros_like(actual)
    for b in range(2):
        for y in range(oh):
            for x in range(ow):
                value = biases.astype(np.int64).copy() if bias else np.zeros(9, dtype=np.int64)
                for ky in range(3):
                    for kx in range(2):
                        iy, ix = y+ky-1, x+kx-1
                        if 0 <= iy < 3 and 0 <= ix < 5:
                            value += image[b, iy, ix].astype(np.int64) @ weights.reshape(3, 2, 5, 9)[ky, kx]
                expected[(b*oh+y)*ow+x] = value.astype(np.int32)
    assert np.array_equal(actual, expected) and np.all(written == 1)


def test_capacity_declines_but_source_law_and_ownership_refuse():
    s = schedule()
    assert plan_convolution_rhs_residency(s, operand_rows=32, law=rhs_law()) is None
    with pytest.raises(ValueError, match="explicitly owned"):
        plan_convolution_rhs_residency(s, operand_rows=8, law=rhs_law())
    with pytest.raises(ValueError, match="preservation/completion"):
        plan_convolution_rhs_residency(s, operand_rows=512,
            law=replace(rhs_law(), other_effects_preserve_rhs=False))
    rhs = plan_convolution_rhs_residency(s, operand_rows=512, law=rhs_law())
    with pytest.raises(ValueError, match="differs"):
        replace(rhs, layout=replace(rhs.layout, rhs_base=rhs.layout.rhs_base+3))
    object.__setattr__(rhs.law, "source_identity", "changed")
    with pytest.raises(ValueError, match="changed after planning"):
        rhs.verify_current()
    for base in (3, 10, 512, True):
        with pytest.raises(ValueError, match="aligned and disjoint"):
            plan_convolution_rhs_residency(s, operand_rows=512, law=rhs_law(), rhs_base=base)
    assert plan_convolution_rhs_residency(s, operand_rows=512, law=rhs_law(), rhs_base=480) is None


def test_registry_preparation_is_optional_and_capacity_fallback_deduplicates_actual_bytes():
    def realize(s, *, rhs=None):
        events = []
        emit_convolution_schedule(s, Integers(), lambda k, a: events.append((k, a)),
            complete=lambda k, a: events.append(("drain", k)), rhs=rhs)
        return EmittedProgram(json.dumps(events, sort_keys=True).encode(), "actual-effects",
            rhs.identity if rhs is not None else s.identity)

    for rows, unique in ((32, 1), (512, 2)):
        space = convolution_pass_space(source(), verify_source=lambda _: None, realize=realize,
            retention_law=retention_law(), issue_law=issue_law(), epoch_law=epoch_law(),
            live_tile_limits=(3,), issue_modes=(True,), compute_pair_limits=(4,),
            rhs_modes=(False, True), rhs_law=rhs_law(), private_operand_rows=rows, max_candidates=2)
        results = list(explore_convolution_schedules(source(), space, context=PipelineContext()))
        assert all(r.evaluation.status == "compiled" for r in results)
        assert len({r.emission.data for r in results}) == unique
        assert all(len(r.evaluation.result.receipt["passes"]) == 6 for r in results)
