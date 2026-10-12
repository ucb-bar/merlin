from copy import deepcopy

import pytest
from merlin_experiments.phase1.feedback.private_group_provenance import routed_kernel_symbols


def fixture():
    return {"granularity": "group", "signatures": {"kernel": []},
            "routed": [{"symbol": "kernel", "parallel": [4, 16], "reduction": [48], "dtypes": ["i8", "i8", "i32"]}],
            "entries": {"kernel": {"op": "conv2d", "Himg": 8, "Wimg": 8, "ci": 3, "kh": 4, "kw": 4,
                                    "N": 16, "stride": [4, 4], "padding": [0, 0, 0, 0]}},
            "call_buffers": {"kernel": [{"role": "input_0", "shape": [1, 8, 8, 3], "dtype": "i8"},
                                        {"role": "weight_0", "shape": [48, 16], "dtype": "i8"},
                                        {"role": "out_0", "shape": [4, 16], "dtype": "i32"}]}}


def test_conv_empty_signature_requires_exact_geometry_and_typed_logical_abi():
    emitted = fixture()
    emitted["programs"] = {"kernel": {"entry": deepcopy(emitted["entries"]["kernel"])}}
    assert routed_kernel_symbols(emitted) == {"kernel"}
    mutations = [lambda e: e["entries"]["kernel"].update(stride=[3, 3]),
                 lambda e: e["entries"]["kernel"].update(dilation=[2, 2]),
                 lambda e: e["entries"]["kernel"].update(padding=[True, 0, 0, 0]),
                 lambda e: e["call_buffers"]["kernel"][0].update(shape=[1, 3, 8, 8]),
                 lambda e: e["call_buffers"]["kernel"][1].update(shape=[16, 48]),
                 lambda e: e["call_buffers"]["kernel"][2].update(dtype="i8"),
                 lambda e: e["call_buffers"]["kernel"].pop(),
                 lambda e: e["signatures"].clear(),
                 lambda e: e["entries"]["kernel"].update(op="matmul")]
    for mutate in mutations:
        changed = deepcopy(emitted)
        mutate(changed)
        with pytest.raises(ValueError):
            routed_kernel_symbols(changed)
