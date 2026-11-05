"""Standalone quantized spatial maximum reaches the ordinary backend boundary."""

import pytest

from merlin.common import mlir_query as mq
from merlin.llvmlower.device_offload import rewrite_groups_to_device


@pytest.mark.parametrize("padding", [0, 1])
def test_standalone_pool_is_one_input_logical_call(tmp_path, monkeypatch, padding):
    from logical_device_target import install
    from test_device_shim_logical_abi import _Capture, _write_capture

    install(monkeypatch)
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_composition

    factory = CG.TargetOracle

    def oracle(target):
        result = factory(target)
        result.families = (*result.families, "movement")
        result.standalone = ("movement",)
        result.routes = ()
        result.absorbs_sequence = lambda kinds, *, unit: group_composition.absorbs_sequence(result, kinds, unit=unit)
        return result

    monkeypatch.setattr(CG, "TargetOracle", oracle)
    c = _Capture()
    shape = [1, 3, 6, 6]
    x = c.arg("x", shape, "i8")
    dq = c.dequantize(x, shape, 0.5)
    pool, out = c.maxpool(dq, shape, 3, 2, padding)
    q = c.quantize(pool, out, 0.25)
    text = c.module(q, out)
    _write_capture(tmp_path / "capture", text, {})
    # Complete-model selection must use the closed integer pool datapath, not
    # the internal dequantized floating pool operands.
    from merlin.compile import complete_group_routing

    pairs = []
    monkeypatch.setattr(complete_group_routing, "_accum_from_facts",
                        lambda target, lhs, rhs: pairs.append((lhs, rhs)) or "i32")
    selected = complete_group_routing.routing(
        "gemmini", text, "original-package", capture=tmp_path / "capture",
        model="", linked_elf_admission=object(),
    )
    assert pairs == [("int8", "int8")]
    assert selected.require_complete_groups and selected.select(None)
    module = mq.parse(text)
    result = rewrite_groups_to_device(
        module, "gemmini", select=lambda _: True, capture=tmp_path / "capture/model.mlir", require_complete_groups=True
    )
    module.verify()
    assert result.moved == 1 and not result.skipped
    entry = next(iter(result.entries.values()))
    assert entry["op"] == "maxpool2d"
    assert entry["acc_scale"] == 2.0
    assert entry["pool_padding"] == [padding] * 4
    assert entry["M"] == 36 and entry["N"] == 3
    if padding:
        assert entry["pool_pad_value"] == -128


def test_maxpool_interface_preserves_negative_padding_and_scale():
    from merlin.runtime.reference import reference_outputs
    from merlin.runtime.simulator import simulate
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir

    attrs = {
        "epilogue": ["acc_scale", "maxpool"],
        "acc_scale": 0.5,
        "output_dtype": "i8",
        "pool_in_dims": [2, 2],
        "pool_size": [2, 2],
        "pool_stride": [1, 1],
        "pool_padding": [1, 1, 1, 1],
        "pool_pad_value": -128,
    }
    cb = {
        "abi_version": "0.1",
        "target": "synthetic",
        "tensors": {
            "X": {"shape": [4, 1], "dtype": "i8", "role": "input"},
            "Y": {"shape": [9, 1], "dtype": "i8", "role": "output"},
        },
        "commands": [{"opcode": "MAXPOOL2D", "operands": {"src": "X", "dst": "Y"}, "attributes": attrs}],
    }
    parsed = parse_interface_mlir(emit_interface_mlir(cb))
    assert parsed["commands"][0]["attributes"]["pool_pad_value"] == -128
    values = [[-9], [-5], [-3], [-7]]
    # Every window contains a real negative value: zero padding would be wrong.
    expected = [[-4], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-4]]
    assert reference_outputs(parsed, {"X": values})["Y"] == expected
    assert simulate(parsed, {"X": values})["outputs"]["Y"] == expected


def test_standalone_pool_saturates_before_spatial_padding():
    from merlin.runtime.reference import reference_outputs
    from merlin.runtime.simulator import simulate

    cb = {"abi_version": "0.1", "target": "synthetic", "tensors": {
        "X": {"shape": [1, 2], "dtype": "i8", "role": "input"},
        "Y": {"shape": [1, 2], "dtype": "i8", "role": "output"},
    }, "commands": [{"opcode": "MAXPOOL2D", "operands": {"src": "X", "dst": "Y"}, "attributes": {
        "epilogue": ["acc_scale", "maxpool"], "acc_scale": 2.0, "output_dtype": "i8",
        "pool_in_dims": [1, 1], "pool_size": [3, 3], "pool_stride": [1, 1],
        "pool_padding": [1, 1, 1, 1], "pool_pad_value": -128,
    }}]}
    values = {"X": [[-128, 127]]}
    assert reference_outputs(cb, values)["Y"] == [[-128, 127]]
    result = simulate(cb, values)
    assert result["outputs"]["Y"] == [[-128, 127]]


def test_pool_capsule_builder_uses_the_same_interface_and_readout():
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir
    from merlin.targetgen.corpus_spec import CorpusBinding, build_maxpool2d

    binding = CorpusBinding("synthetic", 16, "int8", "int32", True, [], "exact_int")
    cap, text = build_maxpool2d(
        {
            "name": "pool",
            "kind": "op",
            "source_role": "model_derived",
            "source_reference": "test",
            "M": 16,
            "N": 3,
            "acc_scale": 0.75,
            "epilogue": ["acc_scale", "maxpool"],
            "pool_in_dims": [4, 4],
            "pool_size": [2, 2],
            "pool_stride": [2, 2],
            "pool_padding": [0, 0, 0, 0],
        },
        binding,
    )
    cb = parse_interface_mlir(text)
    assert cb["tensors"]["Y0"]["shape"] == [4, 3]
    assert cb["commands"][0]["opcode"] == "MAXPOOL2D"
    assert cap["inputs"][0]["dtype"] == "i8"
    assert cap["numeric_policy"]["compare"] == "exact_int"
