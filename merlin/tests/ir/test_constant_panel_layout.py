"""Immutable source payload, physical call type and complete interface map join."""

from types import SimpleNamespace

import numpy as np
import pytest
from xdsl.dialects.builtin import ModuleOp, TensorType, i8
from xdsl.ir import Block

from merlin.compile.scheduling.tensor_storage import column_panel_storage
from merlin.llvmlower.constant_panel_layout import (
    constant_panel_request, prepare_constant_panels, request_constant_panels,
)
from merlin.llvmlower.group_host_call import _logical_group_call
from merlin.llvmlower.device_shim import logical_buffers
from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir
from merlin.targetgen.corpus_spec import CorpusBinding, build_conv2d


def entry(columns=6, taps=3):
    return dict(name="source", kind="layer", source_role="source", source_reference="typed source",
                op="conv2d", Himg=5, Wimg=5, kh=1, kw=taps, ci=4, N=columns,
                stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1])


@pytest.mark.parametrize("columns,panel", [(6, 2), (15, 3), (8, 4)])
def test_original_source_bytes_bind_physical_call_and_interface(columns, panel):
    source = entry(columns)
    original = np.arange(12 * columns, dtype=np.int8).reshape(12, columns)
    pair = Block(arg_types=[TensorType(i8, [1, 5, 5, 4]), TensorType(i8, original.shape)]).args
    group = SimpleNamespace(window_pool=None, window_mean=None)
    stated = SimpleNamespace(stored_operand=object(), transposed=False, column_order=None)
    bound = []

    def bind(array, dtype):
        bound.append(array)
        return Block(arg_types=[TensorType(dtype, array.shape)]).args[0]

    operands, constants, roles, why = _logical_group_call(
        group, stated, source, pair, {"weight": {"array": "fixed"}}, {"fixed": original}, "",
        bind_weight=bind, constant_panel_columns=panel)
    storage = column_panel_storage(12, columns, panel, "i8")
    assert not why and not constants and roles == ["input_0", "weight_0"]
    assert np.array_equal(bound[0], original.reshape(12, columns // panel, panel).transpose(1, 0, 2))
    assert operands[1].type.get_shape() == storage.physical_shape
    cap, interface = build_conv2d(source, CorpusBinding("neutral", panel, "int8", "i32", True, [], "exact_int"))
    cb = parse_interface_mlir(interface)
    assert cb["tensors"]["W"]["storage_encoding"] == storage.to_dict()
    assert parse_interface_mlir(emit_interface_mlir(cb)) == cb
    buffers, why = logical_buffers(interface)
    assert not why
    assert next(b for b in buffers if b.role == "weight_0").shape == storage.physical_shape
    assert cap["inputs"][0]["shape"] == list(storage.physical_shape)
    assert source["weight_storage_encoding"] == storage.to_dict()


def test_source_without_original_immutable_payload_cannot_pack():
    with pytest.raises(ValueError, match="original immutable"):
        prepare_constant_panels(entry(), None, 2, i8)
    with pytest.raises(ValueError, match="complete original"):
        prepare_constant_panels(entry(), np.zeros((3, 6), np.int8), 2, i8)
    with pytest.raises(ValueError, match="storage width"):
        prepare_constant_panels(entry(), np.zeros((12, 6), np.uint8), 2, i8)


def test_unsupported_sources_retain_original_bytes():
    for source, original in [(entry(5), np.arange(60, dtype=np.int8).reshape(12, 5)),
                             (entry(6, 1), np.arange(24, dtype=np.int8).reshape(4, 6)),
                             (dict(entry(), ci=1), np.arange(18, dtype=np.int8).reshape(3, 6))]:
        actual, storage = prepare_constant_panels(source, original, 2, i8, minimum_input_channels=2)
        assert actual is original and storage is None


def test_request_is_explicit_and_repeated_or_invalid_request_refuses():
    module = ModuleOp([])
    assert constant_panel_request(module) is None
    request_constant_panels(module, 3)
    assert constant_panel_request(module) == 3
    with pytest.raises(ValueError, match="already contains"):
        request_constant_panels(module, 3)
    for bad in [True, 0, -1, 1 << 21]:
        with pytest.raises(ValueError, match="bounded positive"):
            request_constant_panels(ModuleOp([]), bad)


def test_input_channel_capability_is_independent_of_output_panel_width():
    source = dict(entry(), ci=1)
    original = np.arange(18, dtype=np.int8).reshape(3, 6)
    packed, storage = prepare_constant_panels(source, original, 3, i8, minimum_input_channels=1)
    assert storage is not None and packed.shape == (2, 3, 3)
    retained, storage = prepare_constant_panels(source, original, 3, i8, minimum_input_channels=2)
    assert retained is original and storage is None


def test_inconsistent_physical_interface_encoding_refuses():
    source = entry()
    source["weight_storage_encoding"] = column_panel_storage(12, 6, 2, "i8").to_dict()
    _, interface = build_conv2d(source, CorpusBinding("neutral", 2, "int8", "i32", True, [], "exact_int"))
    with pytest.raises(ValueError, match="complete physical type"):
        parse_interface_mlir(interface.replace("tensor<3x12x2xi8>", "tensor<12x6xi8>"))


def test_original_layout_policy_gates_and_replays_request(tmp_path):
    from dataclasses import replace
    from test_host_graph_policy import select
    from test_host_graph_optimizations import _source
    from merlin.frontends.linalg_mlir import parse_mlir_text

    policy = replace(select(tmp_path), constant_panel_columns=3)
    selected, report = policy.optimize(_source())
    assert constant_panel_request(parse_mlir_text(selected)) == 3
    assert policy.optimize(_source()) == (selected, report)
    # Request also works when no independent activation transpose savings
    # qualify; it is still a separate original-software layout permission.
    selected, _ = replace(policy, minimum_transpose_bytes_saved=1000).optimize(_source())
    assert constant_panel_request(parse_mlir_text(selected)) == 3
    selected, _ = replace(policy, enabled=False).optimize(_source())
    assert selected == _source()
