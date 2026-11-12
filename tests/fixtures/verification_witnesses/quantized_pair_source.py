from copy import deepcopy

import pytest

from merlin.development.targetgen import capsule_golden as CG
from merlin.development.targetgen import corpus_spec as CS
from merlin.development.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.development.targetgen.contract.schemas import validate_capsule


def source(bias=3, skip=1, side=0):
    metadata = dict(
        kind="op", source_role="handauthored_compiler_test", source_reference="public exact source test", label="public"
    )
    producer = dict(
        metadata,
        name="producer",
        op="conv2d",
        Himg=2,
        Wimg=2,
        ci=1,
        N=1,
        kh=1,
        kw=1,
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        epilogue=["bias_add", "acc_scale"],
        acc_scale=0.5,
        output_dtype="i8",
    )
    residual = dict(
        metadata,
        name="sum",
        op="residual_add",
        M=4,
        N=1,
        epilogue=["relu"],
        lhs_scale=1.0,
        rhs_scale=1.0,
        source_scales=[1.0] * 3,
        bound_lsb=1,
    )
    skip_name = "sum_" + ("X1", "X0")[side]
    entry = dict(
        metadata,
        name="pair",
        kind="model_slice",
        schema="connected_quantized_sum_entry_v1",
        op="quantized_producer_sum",
        components=[producer, residual],
        producer_result_operand=side,
        input_palette=dict(
            schema="merlin.component_input_palette.v1",
            inputs=[
                dict(name=n, axis="linear", values=[v], offset=0)
                for (n, v) in [("producer_IFM", 0), ("producer_W", 0), ("producer_B", bias), (skip_name, skip)]
            ],
        ),
    )
    binding = CS.CorpusBinding(
        "fixture", 16, "int8", "i32", True, ["L0", "L1", "L2", "L3"], "exact_int", readout_inputs={}
    )
    return (entry, binding)


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "CG",
    "CS",
    "deepcopy",
    "parse_interface_mlir",
    "pytest",
    "source",
    "validate_capsule",
]
