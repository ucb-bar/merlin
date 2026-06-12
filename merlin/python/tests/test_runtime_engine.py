"""The Merlin-owned runtime engine performs real integer arithmetic and metrics."""
from __future__ import annotations

from merlin.runtime import (Tensor, simulate, reference_outputs, outputs_match,
                            COMMON_METRIC_NAMES)


def _cb(reuse=3, tensors_extra=None, params=None):
    tensors = {"W": {"shape": [8, 6], "dtype": "i8", "role": "weight"},
               "bias": {"shape": [6], "dtype": "i32", "role": "bias"}}
    cmds = [{"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
             "attributes": {"layout": "packed_rhs"}}]
    for i in range(reuse):
        tensors[f"A{i}"] = {"shape": [5, 8], "dtype": "i8", "role": "input"}
        cmds.append({"opcode": "MATMUL_RESIDENT",
                     "operands": {"lhs": f"A{i}", "rhs": "W_res", "dst": f"acc{i}"}})
        cmds.append({"opcode": "COMMIT",
                     "operands": {"src": f"acc{i}", "dst": f"Y{i}", "bias": "bias"},
                     "attributes": {"epilogue": ["bias_add", "requant", "relu"],
                                    "requant_shift": 4, "output_dtype": "i8"}})
    cmds.append({"opcode": "EVICT", "operands": {"handle": "W_res"}})
    return {"abi_version": "0.1", "target": "toy_npu", "backend": "simulator",
            "tensors": tensors, "commands": cmds, "params": params or {"requant_shift": 4}}


def test_tensor_matmul_is_correct():
    a = Tensor((2, 2), [1, 2, 3, 4], "i8")
    ident = Tensor((2, 2), [1, 0, 0, 1], "i8")
    assert a.matmul(ident).to_list() == [[1, 2], [3, 4]]
    # A @ A
    assert a.matmul(a).to_list() == [[7, 10], [15, 22]]


def test_epilogue_ops():
    t = Tensor((1, 3), [10, -5, 100], "i32")
    assert t.add_bias(Tensor((3,), [1, 1, 1], "i32")).to_list() == [[11, -4, 101]]
    assert t.relu().to_list() == [[10, 0, 100]]
    assert t.requant(1).to_list() == [[5, -2, 50]]   # rounding shift
    assert Tensor((1, 2), [200, -200], "i32").to_i8().to_list() == [[127, -128]]


def test_execution_matches_reference():
    cb = _cb(reuse=4)
    res = simulate(cb)
    ref = reference_outputs(cb)
    assert outputs_match(res["outputs"], ref)
    assert set(res["metrics"]) == set(COMMON_METRIC_NAMES)
    assert res["metrics"]["pack_count"] == 1
    assert res["metrics"]["resident_hits"] == 4
    assert res["metrics"]["accumulator_commits"] == 4
    assert res["metrics"]["evictions"] == 1
    assert res["metrics"]["cycles"] > 0


def test_end_to_end_known_values():
    # Identity weight + relu-only epilogue => output equals the (clamped) activation.
    cb = {
        "abi_version": "0.1", "target": "toy_npu", "backend": "simulator",
        "tensors": {"W": {"shape": [2, 2], "dtype": "i8"},
                    "A0": {"shape": [2, 2], "dtype": "i8"}},
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"},
             "attributes": {"epilogue": ["relu"], "output_dtype": "i8"}},
            {"opcode": "EVICT", "operands": {"handle": "W_res"}},
        ],
    }
    inputs = {"W": [[1, 0], [0, 1]], "A0": [[3, 5], [7, 9]]}
    res = simulate(cb, inputs)
    assert res["outputs"]["Y0"] == [[3, 5], [7, 9]]


def test_trace_records_events():
    res = simulate(_cb(reuse=2))
    names = [e["name"] for e in res["trace"]]
    assert "resident_pack" in names
    assert names.count("resident_hit") == 2
    assert "accumulator_commit" in names
    assert "eviction" in names
