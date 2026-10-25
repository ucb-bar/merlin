"""The public pair writer keeps two original rounded source operations."""

from copy import deepcopy

import pytest
from merlin_experiments.phase0.writer import _integer_reference_bound

from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.contract.schemas import validate_capsule


def source():
    metadata = dict(
        kind="op", source_role="handauthored_compiler_test", source_reference="public exact source test", label="public"
    )
    producer = dict(
        metadata,
        name="producer",
        op="conv2d",
        Himg=3,
        Wimg=3,
        ci=1,
        N=1,
        kh=1,
        kw=1,
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        epilogue=["bias_add", "acc_scale", "relu"],
        acc_scale=0.5,
        output_dtype="i8",
    )
    pool = dict(
        metadata,
        name="pool",
        op="maxpool2d",
        M=9,
        N=1,
        pool_in_dims=[3, 3],
        pool_size=[3, 3],
        pool_stride=[2, 2],
        pool_padding=[1, 1, 1, 1],
        pool_pad_value=-128,
        epilogue=["acc_scale", "maxpool"],
        acc_scale=0.9900199770927429,
        output_dtype="i8",
    )
    entry = dict(
        metadata,
        name="pair",
        kind="model_slice",
        op="spatial_producer_pool",
        components=[producer, pool],
        input_palette={
            "schema": "merlin.component_input_palette.v1",
            "inputs": [
                dict(name="producer_IFM", axis="linear", values=[0], offset=0),
                dict(name="producer_W", axis="linear", values=[0], offset=0),
                dict(name="producer_B", axis="linear", values=[3], offset=0),
            ],
        },
    )
    binding = CS.CorpusBinding(
        "fixture", 16, "int8", "i32", True, ["L0", "L1", "L2", "L3"], "exact_int", readout_inputs={}
    )
    return entry, binding


def test_public_builder_and_golden_retain_the_observed_rounding_boundary():
    entry, binding = source()
    cap, text = CS.build(entry, binding)
    validate_capsule(cap)
    cb = parse_interface_mlir(text)
    conv, pool = cb["commands"][1], cb["commands"][-1]
    assert conv["attributes"]["acc_scale"] == 0.5
    assert pool["attributes"]["acc_scale"] == 0.9900199770927429
    assert pool["operands"]["src"] == conv["operands"]["dst"]
    assert cap["expected"]["modes"] == {"conv2d": True, "k_accumulate": True, "maxpool": True}
    # 3 -> RNE(1.5)=2 -> RNE(1.980...)=2. A single product scale yields 1.
    assert CG.golden(cap) == {"pool_Y0": [[2], [2], [2], [2]]}
    broken = deepcopy(cap)
    broken["operation"]["attributes"]["producer"]["attributes"]["acc_scale"] *= entry["components"][1]["acc_scale"]
    broken["operation"]["attributes"]["pool"]["attributes"]["acc_scale"] = 1.0
    assert CG.golden(broken) == {"pool_Y0": [[1], [1], [1], [1]]}


def test_public_source_graph_is_typed_ssa_with_the_original_complete_edge():
    from xdsl.dialects import func

    from merlin.llvmlower.connected_spatial import group_connected_spatial, public_spatial_source_graph

    entry, binding = source()
    before = deepcopy(entry)
    module, rewrite, receipt = public_spatial_source_graph(entry, binding)
    module.verify()
    calls = [op for op in module.walk() if isinstance(op, func.CallOp)]
    assert len(calls) == 2 and calls[1].arguments[0] is calls[0].res[0]
    assert tuple(module.body.block.first_op.function_type.outputs)[0].get_shape() == (4, 1)
    assert rewrite.entries["producer"]["acc_scale"] == 0.5
    assert rewrite.entries["pool"]["acc_scale"] == 0.9900199770927429
    grouped = group_connected_spatial(module, rewrite)
    assert len(grouped.receipt["groups"]) == 1
    assert len(grouped.rewrite.call_buffers[grouped.receipt["groups"][0]["symbol"]]) == 4
    assert receipt["target_execution"] is False and entry == before


def test_public_source_graph_refuses_ambiguous_private_callees():
    from merlin.llvmlower.connected_spatial import public_spatial_source_graph

    entry, binding = source()
    entry["components"][1]["name"] = entry["components"][0]["name"]
    with pytest.raises(ValueError, match="distinct private"):
        public_spatial_source_graph(entry, binding)


def test_composed_modes_retain_complete_region_properties_and_refuse_conflicts():
    from merlin.targetgen.corpus_spatial_pair import _composition_modes

    assert _composition_modes({"movement": True}, {"movement": True, "maxpool": True}) == {
        "movement": True, "maxpool": True
    }
    with pytest.raises(ValueError, match="conflicting mode"):
        _composition_modes({"dataflow": "one"}, {"dataflow": "different"})


def test_pair_integer_width_screen_uses_original_convolution_and_bias():
    entry, binding = source()
    cap, _ = CS.build(entry, binding)
    entry["numerical_semantics"] = {
        "internal_arithmetic": {
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
            "mac_result_bits": 20,
            "signed_operand_bits": 8,
        }
    }
    bound = _integer_reference_bound(entry, cap)
    assert bound["status"] == "proven_safe" and bound["bound"] >= 3
    cap["operation"]["attributes"]["producer"]["op"] = "unknown"
    with pytest.raises(ValueError, match="convolution"):
        _integer_reference_bound(entry, cap)


def test_mismatched_original_source_edge_refuses_before_writing():
    entry, binding = source()
    entry["components"][1]["N"] = 2
    with pytest.raises(ValueError, match="edge"):
        CS.build(entry, binding)


def test_original_connected_observers_bind_only_the_final_logical_result():
    from merlin.llvmlower.device_shim import logical_buffers

    entry, binding = source()
    _, text = CS.build(entry, binding)
    ordinary, error = logical_buffers(text)
    assert not error and len([b for b in ordinary if b.kind == "output"]) == 2
    connected, error = logical_buffers(text, observed_outputs=("pool_Y0",))
    assert not error
    assert [(b.name, b.role) for b in connected] == [
        ("producer_IFM", "input_0"),
        ("producer_W", "weight_0"),
        ("producer_B", "bias_0"),
        ("pool_Y0", "out_0"),
    ]
    assert logical_buffers(text, observed_outputs=("absent",))[0] is None
