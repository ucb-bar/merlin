"""Two devices in one module: two command buffers, each carrying only its own commands.

A configuration with two accelerators is a normal one -- two cores with different accelerators, or one
core with a mesh and a vector engine -- and the IR has always been able to describe it:
`command_buffer.create` takes its device as an OPERAND and `append` takes its buffer as one, so which
device runs which commands is already written down. Only the emit stage refused, with a flat
"expected exactly one device.get and one command_buffer.create".

Lifting that is a correctness fix on its own. The old code collected EVERY append in the module
regardless of which buffer it was appended to, so relaxing the cap without fixing the grouping would
not have produced two buffers -- it would have produced two buffers each carrying all of both their
commands, which runs, and silently issues every command to both devices.
"""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import runtime as r
from merlin.xdsl_dialects.lowering.emit_command_buffer import (emit_command_buffer,
                                                               emit_command_buffers)
from merlin.xdsl_dialects.lowering.interface_lowering import LoweringError

pytest.importorskip("xdsl")


def _module(specs):
    """One func holding a (device, create, appends...) group per spec."""
    from xdsl.dialects.builtin import (ArrayAttr, DictionaryAttr, FunctionType, ModuleOp,
                                       StringAttr)
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    blk = Block()
    for dev_name, target, tensors, outs, cmds in specs:
        dev = r.DeviceGetOp(result_types=[r.DeviceType()], properties={
            "device": StringAttr(dev_name), "backend": r.BackendAttr(r.Backend.SIMULATOR)})
        cb = r.CommandBufferCreateOp(
            operands=[dev.dev], result_types=[r.CommandBufferType()],
            properties={"target": StringAttr(target),
                        "tensors": DictionaryAttr({k: StringAttr(v) for k, v in tensors.items()}),
                        "outputs": ArrayAttr([StringAttr(o) for o in outs])})
        blk.add_ops([dev, cb])
        for opcode, args in cmds:
            blk.add_op(r.CommandBufferAppendOp(operands=[cb.cb], properties={
                "opcode": StringAttr(opcode),
                "args": DictionaryAttr({k: StringAttr(v) for k, v in args.items()})}))
    blk.add_op(ReturnOp())
    return ModuleOp([FuncOp("main", FunctionType.from_lists([], []), Region([blk]))])


_ONE = [("dev0", "alpha", {"W": "4x4:i8", "Y": "4x4:i32"}, ["Y"],
         [("RES_PACK", {"src": "W", "dst": "W_res"})])]
_TWO = _ONE + [("dev1", "beta", {"V": "8x8:f32", "Z": "8x8:f32"}, ["Z"],
                [("VECTOR_MAP", {"src": "V", "dst": "Z"})])]


def test_one_device_is_unchanged():
    buffers = emit_command_buffers(_module(_ONE))
    assert len(buffers) == 1
    assert buffers[0]["target"] == "alpha"
    assert emit_command_buffer(_module(_ONE)) == buffers[0], "the single-device caller is untouched"


def test_two_devices_emit_two_buffers():
    buffers = emit_command_buffers(_module(_TWO))
    assert [b["target"] for b in buffers] == ["alpha", "beta"]


def test_each_buffer_carries_only_its_own_commands():
    """The latent bug: appends were collected module-wide, not per buffer."""
    a, b = emit_command_buffers(_module(_TWO))
    assert [c["opcode"] for c in a["commands"]] == ["RES_PACK"]
    assert [c["opcode"] for c in b["commands"]] == ["VECTOR_MAP"]


def test_each_buffer_carries_only_its_own_tensors_and_outputs():
    a, b = emit_command_buffers(_module(_TWO))
    assert set(a["tensors"]) == {"W", "Y"} and set(b["tensors"]) == {"V", "Z"}
    assert a.get("outputs") == ["Y"] and b.get("outputs") == ["Z"]


def test_the_single_buffer_caller_says_what_it_found_instead_of_calling_it_malformed():
    """A module describing two devices is a real thing, not a broken module."""
    with pytest.raises(LoweringError, match="2 command buffers"):
        emit_command_buffer(_module(_TWO))


def test_a_module_with_no_buffer_is_still_an_error():
    with pytest.raises(LoweringError, match="no runtime.command_buffer.create"):
        emit_command_buffers(_module([]))
