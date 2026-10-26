"""RTL numeric boundaries, actual runtime leaves, and preserved split source."""

import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from merlin_experiments.phase0 import fusion_reference as F
from merlin_experiments.phase0 import runtime_scale_reference as R

from merlin.runtime.exact_readout import f32_readout


def capsule(*, relu=False, siblings=True):
    scheme = dict(
        scale_source="runtime_input",
        weight_scale_granularity="tensor",
        readout_strategy="f32_product_rne",
        scale_value="arbitrary_f32",
        columns_per_scale=16,
        operand_sum_rounding="single_rounding",
    )
    case = dict(
        producer="contraction",
        exit="accumulator" if siblings else "requant",
        multipliers=[99.0],
        runtime_scale_input="Scale",
        readout_stages=["convert", "scale", "requant"],
        accumulator_stages=[],
    )
    if relu:
        case["readout_stages"].insert(1, "activation")
    if siblings:
        case["siblings"] = [dict(exit="requant"), dict(exit="accumulator")]
    return dict(
        inputs=[
            dict(name="A", shape=[1, 1], dtype="i8", role="input"),
            dict(name="W0", shape=[1, 4], dtype="i8", role="input"),
            *([dict(name="W1", shape=[1, 3], dtype="i8", role="input")] if siblings else []),
            dict(name="Scale", shape=[1], dtype="f32", role="input"),
        ],
        operation=dict(
            attributes=dict(
                M=1, K=1, N=4, case=case, quantization_scheme=scheme, outputs=["q", "raw"] if siblings else ["q"]
            )
        ),
    )


@pytest.mark.parametrize(
    "bits",
    [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x00800000,
        0x3F000000,
        0xBF000000,
        0x3E999999,
        0x3E99999A,
        0x3E99999B,
        0x7F7FFFFF,
        0xFF7FFFFF,
    ],
)
@pytest.mark.parametrize("relu", [False, True])
def test_three_rounding_boundaries_match_rtl(bits, relu):
    scale = np.array([bits], np.uint32).view(np.float32)[0]
    values = np.array(
        [
            -(2**31),
            -(2**24) - 1,
            -(2**24),
            -(2**24) + 1,
            -259,
            -257,
            -255,
            -5,
            -3,
            -1,
            0,
            1,
            3,
            5,
            253,
            255,
            257,
            2**24 - 1,
            2**24,
            2**24 + 1,
            2**31 - 1,
        ],
        np.int64,
    )
    exact = R.apply_runtime_readout(values, scale, relu=relu)
    assert np.array_equal(exact, f32_readout(values, scale, "i8", relu=relu))


def test_fraction_rounder_handles_subnormal_ties_and_overflow():
    tiny = Fraction(2) ** -149
    assert R.round_binary32(tiny / 2) == 0
    assert R.round_binary32(tiny * Fraction(3, 2)) == 2 * tiny
    assert R.round_binary32(-tiny * Fraction(3, 2)) == -2 * tiny
    maximum = (2 - Fraction(2) ** -23) * 2**127
    assert R.round_binary32(maximum) == maximum
    assert R.round_binary32(maximum + 2**103) == float("inf")
    assert R.round_binary32(2**24 + 1) == 2**24
    assert R.round_binary32(2**24 + 3) == 2**24 + 4


def test_scale_input_is_not_case_multiplier_metadata():
    cap = capsule()
    env = dict(
        A=np.array([[1]], np.int8),
        W0=np.array([[-5, -3, 3, 5]], np.int8),
        W1=np.array([[1, -3, 7]], np.int8),
        Scale=np.array([0.5], np.float32),
    )
    exact, events = F.exact_reference(cap, env)
    assert exact == F.unfused_reference(cap, env) == dict(q=[[-2, -2, 2, 2]], raw=[[1, -3, 7]])
    assert events["rne_tie"]
    cap["operation"]["attributes"]["case"]["multipliers"] = [-100.0]
    assert F.exact_reference(cap, env)[0] == exact
    env["Scale"] = np.array([0x80000000], np.uint32).view(np.float32)
    assert R.scale_bits(F.runtime_multiplier(cap, env)) == 0x80000000
    assert F.exact_reference(cap, env)[0] == dict(q=[[0, 0, 0, 0]], raw=[[1, -3, 7]])


@pytest.mark.parametrize(
    "bad",
    [
        np.array([0.5], np.float64),
        np.array([1], np.int32),
        np.array([0.5, 0.5], np.float32),
        np.array(0.5, np.float32),
        np.array([np.nan], np.float32),
        np.array([np.inf], np.float32),
    ],
)
def test_runtime_scale_rejects_wrong_storage_and_nonfinite(bad):
    cap = capsule()
    with pytest.raises(ValueError, match="f32"):
        R.runtime_multiplier(cap, dict(Scale=bad))


@pytest.mark.parametrize(
    "label,value",
    [
        ("aligned", 0.25),
        ("partial", 0.25),
        ("rne_saturation", 0.5),
        ("runtime_scale_zero", 0),
        ("runtime_scale_quarter", 0.25),
        ("runtime_scale_half", 0.5),
        ("runtime_scale_nonpow2", 0.3),
    ],
)
def test_runtime_stimuli_are_typed_inputs(label, value):
    env = F.stimuli(capsule(siblings=False), label)
    assert env["Scale"].dtype == np.float32
    assert env["Scale"][0] == np.float32(value)
    assert env["A"].dtype.kind == "i"


def test_case_generation_keeps_runtime_argument_after_split_and_merge():
    from merlin_experiments.phase0.fusion_cases import _merge_sources, _print_source, _split_readout

    from merlin.common import mlir_query as mq
    from merlin.targetgen.composition_capsules import _Source

    cap = capsule(relu=True, siblings=False)
    source = _Source()
    source.input("acc", [1, 4], "i32")
    value = source.readout("readout", "acc", 1, 4, [99.0], 4, relu=True, runtime_scale="Scale")
    module = mq.parse(source.render([value]))
    function = next(op for op in module.body.block.ops if op.name == "func.func")
    reader = next(
        op
        for op in function.body.block.ops
        if getattr(op.attributes.get("prov.composition_stage"), "data", "") == "readout"
    )
    scale_arg = function.body.block.args[1]
    _split_readout(cap, reader)
    module.verify()
    multiply = next(
        op
        for op in function.body.block.ops
        if op.name == "linalg.generic" and any(inner.name == "arith.mulf" for inner in op.walk())
    )
    assert multiply.operands[1] is scale_arg
    assert str(multiply.properties["indexing_maps"].data[1].data.results[0]) == "0"
    ops = list(function.body.block.ops)
    activation = next(op for op in ops if any(inner.name == "arith.maximumf" for inner in op.walk()))
    assert ops.index(activation) < ops.index(multiply)
    assert "99" not in _print_source(module)
    # A synthetic readout-only capsule isolates clone/merge SSA binding from scheduling.
    cap["inputs"] = [
        dict(name="A", shape=[1, 4], dtype="i32", role="input"),
        dict(name="Scale", shape=[1], dtype="f32", role="input"),
    ]
    merged, inputs = _merge_sources([(cap, _print_source(module)), (cap, _print_source(module))], True)
    merged_module = mq.parse(merged)
    merged_func = next(op for op in merged_module.body.block.ops if op.name == "func.func")
    assert [i["name"] for i in inputs] == ["A", "node0_Scale", "node1_Scale"]
    uses = [
        op.operands[1]
        for op in merged_func.body.block.ops
        if op.name == "linalg.generic" and any(inner.name == "arith.mulf" for inner in op.walk())
    ]
    assert uses == list(merged_func.body.block.args)[1:]
    assert [R.typed_result_dtype(capsule(), name) for name in ("q", "raw")] == ["i8", "i32"]


def test_spike_negative_scale_relu_probe_reports_oracle_mismatch(tmp_path):
    # Report classification is unit-tested without executing Spike.
    extension = tmp_path / "fixture.so"
    extension.write_bytes(b"fixture identity only")
    report = R.oracle_probe_report(extension, 2)
    assert report["status"] == "oracle_mismatch" and report["expected"] == 0
    assert report["extension_sha256"]
    assert R.oracle_probe_report(extension, 0, repaired=True)["status"] == "parity"
    with pytest.raises(ValueError, match="repaired Spike"):
        R.oracle_probe_report(extension, 2, repaired=True)


@pytest.mark.skipif(
    not os.environ.get("RUNTIME_SCALE_SPIKE_PROBE_REPORT"), reason="optional real Spike probe not requested"
)
def test_optional_real_spike_negative_scale_relu_probe():
    saved = json.loads(Path(os.environ["RUNTIME_SCALE_SPIKE_PROBE_REPORT"]).read_text())
    report = R.oracle_probe_report(saved["extension"], saved["actual"], repaired=saved.get("repaired", False))
    assert saved["extension_sha256"] == report["extension_sha256"]
    assert saved["status"] == report["status"]
