from copy import deepcopy

import pytest
from merlin_experiments.phase1.feedback.private_group_provenance import (
    convolution_input_preparations,
    routed_kernel_symbols,
)


def fixture():
    return {
        "granularity": "group",
        "signatures": {"kernel": []},
        "routed": [{"symbol": "kernel", "parallel": [4, 16], "reduction": [48], "dtypes": ["i8", "i8", "i32"]}],
        "entries": {
            "kernel": {
                "op": "conv2d",
                "Himg": 8,
                "Wimg": 8,
                "ci": 3,
                "kh": 4,
                "kw": 4,
                "N": 16,
                "stride": [4, 4],
                "padding": [0, 0, 0, 0],
            }
        },
        "call_buffers": {
            "kernel": [
                {"role": "input_0", "shape": [1, 8, 8, 3], "dtype": "i8"},
                {"role": "weight_0", "shape": [48, 16], "dtype": "i8"},
                {"role": "out_0", "shape": [4, 16], "dtype": "i32"},
            ]
        },
    }


def test_conv_empty_signature_requires_exact_geometry_and_typed_logical_abi():
    emitted = fixture()
    emitted["programs"] = {"kernel": {"entry": deepcopy(emitted["entries"]["kernel"])}}
    assert routed_kernel_symbols(emitted) == {"kernel"}
    mutations = [
        lambda e: e["entries"]["kernel"].update(stride=[3, 3]),
        lambda e: e["entries"]["kernel"].update(dilation=[2, 2]),
        lambda e: e["entries"]["kernel"].update(padding=[True, 0, 0, 0]),
        lambda e: e["call_buffers"]["kernel"][0].update(shape=[1, 3, 8, 8]),
        lambda e: e["call_buffers"]["kernel"][1].update(shape=[16, 48]),
        lambda e: e["call_buffers"]["kernel"][2].update(dtype="i8"),
        lambda e: e["call_buffers"]["kernel"].pop(),
        lambda e: e["signatures"].clear(),
        lambda e: e["entries"]["kernel"].update(op="matmul"),
    ]
    for mutate in mutations:
        changed = deepcopy(emitted)
        mutate(changed)
        with pytest.raises(ValueError):
            routed_kernel_symbols(changed)


def test_source_stated_fused_convolution_requires_matching_bias_and_readout():
    emitted = fixture()
    entry = emitted["entries"]["kernel"]
    entry.update(epilogue=["bias_add", "acc_scale"], acc_scale=0.125)
    emitted["programs"] = {"kernel": {"entry": deepcopy(entry)}}
    emitted["call_buffers"]["kernel"].insert(2, {"role": "bias_0", "shape": [16], "dtype": "i32"})
    emitted["call_buffers"]["kernel"][-1]["dtype"] = "i8"
    emitted["routed"][0]["dtypes"][-1] = "i8"
    assert routed_kernel_symbols(emitted) == {"kernel"}
    for mutate in (
        lambda e: e["call_buffers"]["kernel"][2].update(shape=[4]),
        lambda e: e["call_buffers"]["kernel"][2].update(dtype="i8"),
        lambda e: e["entries"]["kernel"].update(acc_scale=0.25),
        lambda e: e["entries"]["kernel"].update(epilogue=["relu", "bias_add", "acc_scale"]),
    ):
        changed = deepcopy(emitted)
        mutate(changed)
        with pytest.raises(ValueError):
            routed_kernel_symbols(changed)


def _source_bound(emitted):
    row = emitted["routed"][0]
    row.update(group=0, source_region_id="r0", source_node_ids=["n0"])
    return {
        "eligible_groups": [0],
        "eligible_group_provenance": [{"source_group": 0, "source_region_id": "r0", "source_node_ids": ["n0"]}],
        "source_group_contractions": {0: {"parallel": [4, 16], "reduction": [48], "dtypes": ["i8", "i8", "i32"]}},
    }


@pytest.mark.parametrize("biased", [False, True])
def test_source_bound_relu_column_readout_matches_without_copy_exemption(biased):
    emitted = fixture()
    entry = emitted["entries"]["kernel"]
    entry.update(
        epilogue=(["bias_add"] if biased else []) + ["acc_scale", "relu"],
        acc_scale_by_column=[0.125] * 8 + [0.25] * 8,
        output_dtype="i8",
    )
    if biased:
        emitted["call_buffers"]["kernel"].insert(2, {"role": "bias_0", "shape": [16], "dtype": "i32"})
    emitted["call_buffers"]["kernel"][-1]["dtype"] = "i8"
    emitted["routed"][0]["dtypes"][-1] = "i8"
    emitted["programs"] = {"kernel": {"entry": deepcopy(entry)}}
    source = _source_bound(emitted)
    assert routed_kernel_symbols(emitted) == {"kernel"}
    # The original contraction accumulates i32; the source-bound readout is
    # separately typed i8. No ordinary input-copy exemption is granted to it.
    assert convolution_input_preparations(emitted, source) == {}
    mutations = (
        lambda e: e["entries"]["kernel"]["acc_scale_by_column"].__setitem__(0, 0.5),
        lambda e: e["entries"]["kernel"].update(output_dtype="i32"),
        lambda e: e["programs"]["kernel"]["entry"].update(epilogue=["acc_scale"]),
        lambda e: e["call_buffers"]["kernel"][-1].update(dtype="i32"),
        lambda e: e["routed"][0].update(parallel=[8, 16]),
    )
    for mutate in mutations:
        changed = deepcopy(emitted)
        mutate(changed)
        with pytest.raises(ValueError):
            routed_kernel_symbols(changed)


@pytest.mark.parametrize("scales", [[0.125], [True] * 16, [float("nan")] * 16, [float("inf")] * 16])
def test_even_matching_program_cannot_supply_invalid_column_scale_metadata(scales):
    emitted = fixture()
    entry = emitted["entries"]["kernel"]
    entry.update(epilogue=["acc_scale", "relu"], acc_scale_by_column=scales, output_dtype="i8")
    emitted["programs"] = {"kernel": {"entry": deepcopy(entry)}}
    emitted["call_buffers"]["kernel"][-1]["dtype"] = "i8"
    emitted["routed"][0]["dtypes"][-1] = "i8"
    with pytest.raises(ValueError):
        routed_kernel_symbols(emitted)


def test_bare_convolution_keeps_original_accumulator_dtype_obligation():
    emitted = fixture()
    emitted["programs"] = {"kernel": {"entry": deepcopy(emitted["entries"]["kernel"])}}
    source = _source_bound(emitted)
    assert "kernel" in convolution_input_preparations(emitted, source)
    emitted["call_buffers"]["kernel"][-1]["dtype"] = "i8"
    emitted["routed"][0]["dtypes"][-1] = "i8"
    with pytest.raises(ValueError, match="original source contraction"):
        convolution_input_preparations(emitted, source)
