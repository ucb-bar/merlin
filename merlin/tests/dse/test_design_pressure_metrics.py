"""Unit tests for the design-pressure metrics on small hand-built regions."""
from merlin.design_pressure import region as R
from merlin.design_pressure.metrics.lifetimes import metric_lifetimes
from merlin.design_pressure.metrics.memory import metric_memory
from merlin.design_pressure.metrics.packing import metric_packing
from merlin.design_pressure.metrics.reuse import metric_reuse
from merlin.design_pressure.metrics.shapes import metric_shapes


def _region(epilogue=False, reuse=8):
    ops = ["matmul"] + (["bias_add", "requant", "relu"] if epilogue else [])
    tensors = {
        "A": {"shape": [4, 256], "dtype": "i8", "lifetime": "single_use"},
        "W": {"shape": [256, 64], "dtype": "i8", "mutable": False,
              "lifetime": "reused_across_region", "reuse_count": reuse},
        "Y": {"shape": [4, 64], "dtype": "i8", "lifetime": "single_use"},
    }
    if epilogue:
        tensors["bias"] = {"shape": [64], "dtype": "i32", "mutable": False}
    return {"name": "t", "ops": ops, "op_sequence": ops, "tensors": tensors,
            "reuse": {"rhs_reuse_count": reuse, "rhs_mutable": False}}


def test_shapes_mnk_and_epilogue():
    m = metric_shapes(_region(epilogue=True))
    assert (m["M"], m["K"], m["N"]) == (4, 256, 64)
    assert m["op"] == "matmul"
    assert m["has_epilogue"] is True
    assert m["macs"] == 4 * 256 * 64


def test_reuse_facts():
    m = metric_reuse(_region(reuse=8))
    assert m["rhs_reuse_count"] == 8
    assert m["rhs_mutable"] is False


def test_lifetimes_accumulator_live_only_with_epilogue():
    assert metric_lifetimes(_region(epilogue=True))["accumulator_live_across_epilogue"] is True
    assert metric_lifetimes(_region(epilogue=False))["accumulator_live_across_epilogue"] is False
    # state bytes = packed W footprint = 256*64*1 byte
    assert metric_lifetimes(_region())["state_bytes_per_step"] == 256 * 64


def test_packing_amortization():
    m = metric_packing(_region(reuse=8))
    assert m["pack_count_baseline"] == 8
    assert m["pack_count_resident"] == 1
    assert m["pack_bytes"] == 256 * 64


def test_memory_intermediate_vs_output_bytes():
    m = metric_memory(_region(epilogue=True, reuse=8))
    # i32 accumulator intermediate is 4x the i8 output, per step.
    assert m["intermediate_i32_bytes_step"] == 4 * (4 * 64)
    assert m["final_output_bytes_step"] == 1 * (4 * 64)
    # baseline moves the weight every step; resident moves it once -> baseline > resident.
    assert m["dram_traffic_bytes_baseline"] > m["dram_traffic_bytes_resident"]


def test_dtype_bytes():
    assert R.dtype_bytes("i8") == 1
    assert R.dtype_bytes("i32") == 4
    assert R.dtype_bytes("bf16") == 2
