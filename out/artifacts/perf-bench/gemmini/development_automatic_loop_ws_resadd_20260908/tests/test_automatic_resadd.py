"""Automatic, exact Gemmini LOOP_WS residual-add placement."""
from pathlib import Path

import numpy as np
import pytest

from mlir_oot.frontend.gemmini_resadd import IntegerResAdd, recognize
from mlir_oot.frontend.parse import parse_module
from mlir_oot.gemmini_opt import Pipeline, _print
from mlir_oot.tables import loop_ws


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "tests/fixtures/exact_identity_resadd_i8.mlir").read_text()


def _source(m: int = 2, n: int = 17, *, relu: bool = False) -> str:
    return (TEMPLATE.replace("@M@", str(m)).replace("@N@", str(n))
            .replace("@LO@", "0" if relu else "-128"))


def _candidate(source: str):
    module = parse_module(source)
    return next(op for op in module.walk()
                if getattr(op.attributes.get("prov.region_id"), "data", None) == "resadd")


@pytest.mark.parametrize("relu", [False, True])
def test_recognizes_only_exact_identity_saturating_i8_add(relu: bool) -> None:
    result = recognize(_candidate(_source(relu=relu)))
    assert isinstance(result, IntegerResAdd)
    assert result.shape == (2, 17)
    assert result.relu is relu
    assert result.a_scale == result.b_scale == result.c_scale == 1.0
    assert result.numeric_contract == (
        "clamp_i8(add_i32(sign_extend_i8(a),sign_extend_i8(b)))")


@pytest.mark.parametrize("old,new", [
    ("arith.addi %x32, %y32", "arith.subi %x32, %y32"),
    ("arith.maxsi %sum, %lo", "arith.minsi %sum, %lo"),
    ("arith.trunci %clamped", "arith.trunci %sum"),
    ("affine_map<(d0, d1) -> (d0, d1)>", "affine_map<(d0, d1) -> (d1, d0)>")
])
def test_changed_operator_clamp_or_layout_fails_closed(old: str, new: str) -> None:
    assert recognize(_candidate(_source().replace(old, new, 1))) is None


@pytest.mark.parametrize("m,n,relu", [(1, 1, False), (17, 33, True), (64, 257, False)])
def test_exact_add_is_automatically_placed_and_encoded_as_loop_ws_resadd(
        m: int, n: int, relu: bool) -> None:
    pipe = Pipeline(_source(m, n, relu=relu)).run()
    assert pipe.declined is None
    assert pipe.mixed_declined is None
    receipt = pipe.plan.command_buffer["params"]["gemmini_resadd_placement"]
    assert receipt["candidate_count"] == receipt["selected_count"] == 1
    assert receipt["refused"] == []
    tasks = pipe.plan.command_buffer["params"]["global_program_plan"]["tasks"]
    assert [task["kind"] for task in tasks] == ["resadd"]
    loops = [ins for ins in pipe.instrs if ins.kind == "loop_ws_resadd"]
    assert loops
    assert all(ins.attrs["relu"] is relu for ins in loops)
    target = _print(pipe.artifact)
    launch_funct = loop_ws.opcode("k_LOOP_WS")
    assert target.count(f"0x{launch_funct:x}") >= len(loops)


def test_nonexact_add_stays_on_host_and_records_precise_refusal() -> None:
    source = _source().replace("arith.maxsi %sum, %lo", "arith.minsi %sum, %lo")
    pipe = Pipeline(source).run()
    assert pipe.declined is None
    assert pipe.mixed_declined is None
    receipt = pipe.plan.command_buffer["params"]["gemmini_resadd_placement"]
    assert receipt["selected_count"] == 0
    assert receipt["refused"] == [{
        "source_op_index": 1,
        "reason": "integer add is not the exact identity-scale saturating-i8 LOOP_WS contract",
    }]
    assert all(ins.kind != "loop_ws_resadd" for ins in pipe.instrs)


@pytest.mark.parametrize("relu", [False, True])
def test_identity_resadd_contract_is_exhaustively_equal_for_all_i8_pairs(relu: bool) -> None:
    values = np.arange(-128, 128, dtype=np.int32)
    lhs, rhs = np.meshgrid(values, values, indexing="ij")
    source = np.clip(lhs + rhs, 0 if relu else -128, 127).astype(np.int8)
    # Gemmini: each input is independently RNE-scaled and clamped on MVIN, then the
    # accumulator sum is identity RNE-scaled/clamped on MVOUT. Identity is exact for i8.
    hw_lhs = np.clip(np.rint(lhs.astype(np.float32)), -128, 127).astype(np.int32)
    hw_rhs = np.clip(np.rint(rhs.astype(np.float32)), -128, 127).astype(np.int32)
    hardware = np.clip(
        np.rint((hw_lhs + hw_rhs).astype(np.float32)),
        0 if relu else -128, 127).astype(np.int8)
    assert np.array_equal(hardware, source)


def test_resadd_descriptor_sets_k_zero_and_is_resadd_bit() -> None:
    records = loop_ws.loop_ws_resadd_static(
        rows=17, cols=33, row_stride_a=48, row_stride_b=48, row_stride_c=48)
    bounds = next(row for row in records
                  if row[0] == loop_ws.opcode("k_LOOP_WS_CONFIG_BOUNDS"))
    launch = next(row for row in records if row[0] == loop_ws.opcode("k_LOOP_WS"))
    assert bounds[2] >> 32 == 0  # K tile count
    assert bounds[1] >> 32 == 0  # pad_K
    assert launch[2] & 0b100


def test_canonical_resnet_census_refuses_all_16_float_residuals() -> None:
    import json
    receipt = json.loads((ROOT / "validation/resnet50_resadd_census.json").read_text())
    assert receipt["source_sha256"] == (
        "76c26171096661650e2ee440fb545926936b1e770d2bfd0206a6509e6e835e94")
    assert receipt["candidate_count"] == 16
    assert receipt["selected_count"] == 0
    assert receipt["approximation"] is False
    assert [row["source_op_index"] for row in receipt["float_residual_refused"]] == [
        109, 180, 251, 332, 403, 474, 545, 626,
        697, 768, 839, 910, 981, 1062, 1133, 1204,
    ]
    assert all(row["selected"] is False for row in receipt["float_residual_refused"])
    assert "no unique exact roundeven/clamp i8 sink" in receipt["float_residual_refused"][-1][
        "reasons"][0]
