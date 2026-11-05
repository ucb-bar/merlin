from copy import deepcopy

import pytest
from test_output_fusion import _path
from test_sibling_components import effects
from test_sibling_quantized_readout import shared_readout_source

from merlin.llvmlower.device_shim import logical_buffers
from merlin.llvmlower.sibling_components import prepare_sibling_components
from merlin.llvmlower.sibling_readout_commands import require_writer_interface
from merlin.llvmlower.sibling_source_route import _command_buffer
from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir


def prepared(*, observer=False):
    return prepare_sibling_components(
        shared_readout_source(raw_observer=observer), effects=effects(), output_path=_path()
    )


def test_original_bias_input_positions_and_exact_typed_result_buffers():
    plan = prepared()
    component = plan.components[0]
    rows = component.receipt["readouts"]
    cb, _ = _command_buffer(component.core, "selected-target", readouts=rows)
    cb = parse_interface_mlir(emit_interface_mlir(cb))
    buffers, reason = logical_buffers(emit_interface_mlir(cb))
    assert not reason
    assert [(b.name, b.dtype) for b in buffers] == [
        ("input_0", "i8"),
        ("weight_0", "i8"),
        ("weight_1", "i8"),
        ("bias_0", "i32"),
        ("bias_1", "i32"),
        ("out_0", "i8"),
        ("out_1", "i8"),
    ]
    assert [b.shape for b in buffers[-4:]] == [(32,), (32,), (3, 32), (3, 32)]
    commits = [c for c in cb["commands"] if c["opcode"] == "COMMIT"]
    assert [c["attributes"]["bias"] for c in commits] == ["bias_0", "bias_1"]
    assert [c["attributes"]["acc_scale_by_column"] for c in commits] == [
        row["attributes"]["acc_scale_by_column"] for row in rows
    ]
    assert component.writer.result_arguments == (5, 6)
    require_writer_interface(plan.module, component.writer, buffers)


def test_independent_raw_observer_retains_mixed_exit_and_ordered_bias_abi():
    plan = prepared(observer=True)
    component = plan.components[0]
    rows = component.receipt["readouts"]
    assert [row["output_dtype"] for row in rows] == ["i32", "i8"]
    cb, _ = _command_buffer(component.core, "selected-target", readouts=rows)
    buffers, reason = logical_buffers(emit_interface_mlir(cb))
    assert not reason
    assert [b.name for b in buffers] == ["input_0", "weight_0", "weight_1", "bias_1", "out_0", "out_1"]
    assert [b.dtype for b in buffers[-2:]] == ["i32", "i8"]
    assert component.writer.result_arguments == (4, 5)
    require_writer_interface(plan.module, component.writer, buffers)
    with pytest.raises(ValueError, match="pointer type or order"):
        require_writer_interface(plan.module, component.writer, (*buffers[:4], buffers[5], buffers[4]))


@pytest.mark.parametrize("damage", ["scale", "bias", "bias_position", "member", "missing"])
def test_changed_readout_transport_refuses(damage):
    component = prepared().components[0]
    rows = deepcopy(component.receipt["readouts"])
    if damage == "scale":
        rows[0]["attributes"]["acc_scale_by_column"][0] *= 2
    elif damage == "bias":
        rows[0]["bias_values"][0] += 1
    elif damage == "bias_position":
        rows[0]["bias_argument"] += 1
    elif damage == "member":
        rows.reverse()
    else:
        rows.pop()
    with pytest.raises(ValueError):
        _command_buffer(component.core, "selected-target", readouts=rows)
