"""The readback's element width comes from the DECLARED OUTPUT dtype — never from an input, never a default.

A capsule whose output container is WIDER than its input is the whole difficulty here: an ``inputs: [X
bf16]`` capsule whose ``operation.attributes.output_dtype`` is ``f32`` has a 2-byte input and a 4-byte
result, so any code that takes the result's width from the input, or from a baked default, is wrong by a
factor of two or four. That mis-sizing does not surface as a wrong number; it surfaces as
``ValueError: cannot reshape array of size 4096 into shape (32,32)`` from inside numpy, which the ladder
books as ``tool_crash`` on the L3 plane with the integers redacted out of the verdict — naming neither the
tensor, nor the dtype, nor which side of the readback disagreed.

Two places resolve that dtype and both are locked here:

* :func:`merlin.targetgen.capsule_dram.output_tensor` — the CAPSULE's own declaration. It read
  ``attributes.output_dtype or numeric_policy.dtype or "bf16"``: a chain that silently picks one when the
  two declarations disagree, and invents ``bf16`` when a capsule declares neither.
* :func:`merlin.targetgen.program_oracle._resolve_out_specs` — the COMMAND BUFFER's declaration, which
  sizes the DRAM read window and the decode. It read ``tensors[name]["dtype"]`` alone, defaulting to
  ``"bf16"``, while every other harness reader of the same buffer resolves the width through
  :func:`merlin.runtime.commandbuffer.declared_output_dtypes` (which lets a command's own
  ``attributes.output_dtype`` win for the destination it names). One buffer, two resolution rules.

Both now fail closed with a diagnostic that NAMES what could not be resolved, and the decode's existing
byte-level behaviour for every dtype it already handled is pinned so the hardening cannot move a number.
"""

from __future__ import annotations

import base64
import struct

import numpy as np
import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen import capsule_dram as D
from merlin.targetgen import program_oracle as PO
from merlin.runtime.commandbuffer import declared_output_dtypes


# ---- the CAPSULE's declaration: derived, conflict-refusing, never defaulted ----------------------


def _capsule(*, output_dtype=None, policy_dtype=None, dtype_in="bf16", shape=(16, 16)):
    cap: dict = {
        "name": "SYNTH_movement",
        "inputs": [{"name": "X", "role": "input", "shape": list(shape), "dtype": dtype_in}],
        "operation": {"op": "movement", "attributes": {"out": "Y0", "src": "X"}},
    }
    if output_dtype is not None:
        cap["operation"]["attributes"]["output_dtype"] = output_dtype
    if policy_dtype is not None:
        cap["numeric_policy"] = {"compare": "tolerance_float", "dtype": policy_dtype}
    return cap


def test_output_tensor_takes_the_declared_output_dtype_not_the_input_dtype():
    # The shape mirrors the input; the WIDTH does not. This is the SY_movement_bf16_* / AF*_pt shape:
    # a bf16 operand widened into an f32 result container.
    ot = D.output_tensor(_capsule(output_dtype="f32", policy_dtype="f32"))
    assert ot == {"name": "Y0", "shape": [16, 16], "dtype": "f32"}
    # ... and the DRAM layout sizes the result from that declared width, not from the operand's.
    assert D.tensor_nbytes(ot["shape"], ot["dtype"]) == 16 * 16 * 4


def test_output_tensor_refuses_two_disagreeing_declarations_and_names_both():
    # `attributes.output_dtype` and `numeric_policy.dtype` are two declarations of ONE fact. When they
    # disagree the capsule is self-contradictory: picking either silently produces a readback window that
    # is right for one declaration and wrong for the other. Fail closed, naming both.
    cap = _capsule(output_dtype="f32", policy_dtype="bf16")
    with pytest.raises(D.OutputDtypeUnresolved) as exc:
        D.output_tensor(cap)
    msg = str(exc.value)
    assert "output_dtype" in msg and "numeric_policy" in msg
    assert "'f32'" in msg and "'bf16'" in msg  # both spellings, so the conflict is actionable
    assert "Y0" in msg  # and WHICH tensor it is about


def test_output_tensor_refuses_a_capsule_that_declares_no_output_dtype():
    # The old chain ended in `or "bf16"`. A capsule declaring neither then got a 2-byte result container
    # invented for it -- and for an f32 result that is a 2x mis-sized window that reads as a tool crash.
    with pytest.raises(D.OutputDtypeUnresolved) as exc:
        D.output_tensor(_capsule())
    assert "bf16" not in str(exc.value), "the refusal must not name a substituted default"
    assert "Y0" in str(exc.value)


def test_output_tensor_accepts_either_declaration_alone():
    # Only one declared is not a conflict -- 172 shipped capsules declare numeric_policy.dtype only.
    assert D.output_tensor(_capsule(policy_dtype="f32"))["dtype"] == "f32"
    assert D.output_tensor(_capsule(output_dtype="i32"))["dtype"] == "i32"


def test_a_role_output_input_entry_still_wins_and_is_not_conflict_checked():
    # An explicit `role: output` entry IS the declaration; the operation attributes then describe a
    # different thing and must not be cross-checked against it.
    cap = _capsule(output_dtype="f32", policy_dtype="bf16")
    cap["inputs"].append({"name": "Y0", "role": "output", "shape": [16, 16], "dtype": "i8"})
    assert D.output_tensor(cap) == {"name": "Y0", "shape": [16, 16], "dtype": "i8"}


def test_every_shipped_capsule_declares_a_resolvable_non_conflicting_output_dtype():
    """The corpus invariant this hardening states rather than imposes.

    Measured over the 633 shipped ``capsule.yaml`` files: 461 declare both ``output_dtype`` and
    ``numeric_policy.dtype`` and ALL 461 agree, 172 declare only the policy dtype, and ZERO would have
    reached the old ``"bf16"`` default. So the refusals above cannot change a single shipped verdict --
    and if a future capsule is authored with a conflict, this is where it is caught.
    """
    root = merlin_dir() / "contract" / "capsules"
    resolved = 0
    for cy in sorted(root.rglob("capsule.yaml")):
        cap = yaml.safe_load(cy.read_text(encoding="utf-8"))
        if not isinstance(cap, dict):
            continue
        ot = D.output_tensor(cap)  # must not raise for any shipped capsule
        if ot is not None:
            assert ot["dtype"], f"{cy}: resolved an empty output dtype"
            resolved += 1
    assert resolved > 500, f"expected the shipped corpus, resolved only {resolved}"


# ---- the COMMAND BUFFER's declaration: one resolution rule, shared with every other reader --------


def _cb(y0_dtype="f32", *, cmd_output_dtype=None, base=0x80002000, shape=(16, 16)):
    cb: dict = {
        "target": "atlas",
        "tensors": {
            "X": {"shape": list(shape), "dtype": "bf16", "role": "input", "base": 0x80000000},
            "Y0": {"shape": list(shape), "role": "output", "base": base},
        },
        "commands": [{"opcode": "MOVEMENT", "operands": {"src": "X", "dst": "Y0"}, "attributes": {}}],
    }
    if y0_dtype is not None:
        cb["tensors"]["Y0"]["dtype"] = y0_dtype
    if cmd_output_dtype is not None:
        cb["commands"][0]["attributes"]["output_dtype"] = cmd_output_dtype
    return cb


def test_out_specs_resolve_the_commands_declared_output_dtype_over_the_tensors_own():
    # This is the resolution order `declared_output_dtypes` already applies everywhere else in the
    # harness: a command that names a destination may re-declare the container its result lands in.
    # The program oracle read `tensors["Y0"]["dtype"]` alone, so a buffer that widened via the command
    # attribute was read back at the NARROW width -- a 2x short window on the same tensor.
    cb = _cb(y0_dtype="bf16", cmd_output_dtype="f32")
    assert declared_output_dtypes(cb)["Y0"] == "f32", "the shared reader resolves f32"
    spec = PO._resolve_out_specs("atlas", cb, {})["Y0"]
    assert spec["dtype"] == "f32", "the program oracle must resolve the SAME dtype"
    assert PO._out_nbytes(spec) == 16 * 16 * 4


def test_out_specs_keep_the_tensors_own_dtype_when_no_command_redeclares_it():
    spec = PO._resolve_out_specs("atlas", _cb(y0_dtype="bf16"), {})["Y0"]
    assert spec["dtype"] == "bf16" and PO._out_nbytes(spec) == 16 * 16 * 2


def test_out_specs_refuse_an_output_whose_dtype_is_declared_nowhere():
    # `t.get("dtype", "bf16")`: a buffer that declared no element type for its result got a 2-byte one
    # invented for it. An undeclared width is UNKNOWN -- say so, and name the tensor.
    with pytest.raises(PO.OracleUnavailable) as exc:
        PO._resolve_out_specs("atlas", _cb(y0_dtype=None), {})
    msg = str(exc.value)
    assert "Y0" in msg
    assert "bf16" not in msg, "the refusal must not name a substituted default"


# ---- the decode itself: byte-identical for every dtype it already handled ------------------------


@pytest.mark.parametrize("dtype", ["bf16", "torch.bfloat16"])
def test_bf16_decode_is_unchanged(dtype):
    # bf16 is f32's exponent field with a short mantissa: the stored halfword widens by a 16-bit shift.
    vals = [1.0, -2.5, 0.0, 3.5]
    raw = b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", v))[0] >> 16) for v in vals)
    got = PO._decode_output(raw, [2, 2], dtype, None, name="Y0")
    assert got.dtype == np.float32
    assert got.reshape(-1).tolist() == vals


@pytest.mark.parametrize("dtype", ["i32", "int32", "torch.int32"])
def test_i32_decode_is_unchanged(dtype):
    vals = [1, -2, 0, 2**30]
    raw = np.asarray(vals, dtype="<i4").tobytes()
    got = PO._decode_output(raw, [2, 2], dtype, None, name="Y0")
    assert got.reshape(-1).tolist() == vals
    assert PO._element_bytes(dtype) == 4


@pytest.mark.parametrize("dtype", ["int8", "i8", "torch.int8"])
def test_int8_decode_is_unchanged(dtype):
    vals = [1, -2, 0, 127]
    raw = np.asarray(vals, dtype=np.int8).tobytes()
    got = PO._decode_output(raw, [2, 2], dtype, None, name="Y0")
    assert got.reshape(-1).tolist() == vals
    assert PO._element_bytes(dtype) == 1


@pytest.mark.parametrize("dtype", ["f32", "float32", "torch.float32"])
def test_f32_decode_reads_four_bytes_per_element(dtype):
    # THE CASE THAT CRASHED: 4 bytes are read per element and 4 are decoded, so the window and the
    # decode agree and the reshape onto the declared shape is exact. Sized at 4 and decoded at 1, a
    # 16x16 f32 result produced 1024 elements for a 256-element tensor.
    vals = [1.5, -2.25, 0.0, 1e-8]
    raw = np.asarray(vals, dtype="<f4").tobytes()
    assert PO._element_bytes(dtype) == 4
    got = PO._decode_output(raw, [2, 2], dtype, None, name="Y0")
    assert got.dtype == np.float32
    assert got.reshape(-1).tolist() == pytest.approx(vals)


def test_the_physical_row_unstack_is_unchanged():
    # The atlas MXU writes an [2R, C] result as two stacked R-row banks; the un-stack is declared by the
    # emitting backend, and it happens AFTER the reshape.
    raw = np.arange(8, dtype="<i4").tobytes()
    got = PO._decode_output(raw, [4, 2], "i32", {"unstack_row_halves": 2}, name="Y0")
    assert got.tolist() == [[0, 1, 4, 5], [2, 3, 6, 7]]


def test_an_irreconcilable_byte_count_is_a_named_refusal_not_a_bare_reshape_error():
    # `arr.reshape(dims)`'s ValueError reached the verdict as
    #   "L3 invocation failed: cannot reshape array of size # into shape (#,#)"
    # -- integers redacted, tensor unnamed, dtype unnamed, and classified `tool_crash` on the L3 plane.
    # The refusal must carry every number a reader needs to see which side is wrong.
    raw = b"\x00" * 1000  # 1000 B for a 256-element f32 tensor
    with pytest.raises(PO.OracleUnavailable) as exc:
        PO._decode_output(raw, [16, 16], "f32", None, name="Y0")
    msg = str(exc.value)
    assert "'Y0'" in msg  # the tensor
    assert "[16, 16]" in msg  # its declared shape
    assert "f32" in msg  # the resolved dtype
    assert "1024" in msg  # the bytes the declaration needs
    assert "1000" in msg  # the bytes actually read
    assert "cannot reshape array of size" not in msg, "must not be numpy's own bare phrasing"


def test_an_unregistered_output_dtype_is_refused_rather_than_read_as_bytes():
    with pytest.raises(PO.OracleUnavailable):
        PO._element_bytes("quaternion48")
