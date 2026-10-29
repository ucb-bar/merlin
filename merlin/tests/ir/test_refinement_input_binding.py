"""Typed shared-input admission must run before SMT encoding or solver invocation."""

from types import SimpleNamespace

import pytest

from merlin.verify import cb_semantics, refine
from merlin.verify.smt_semantics import Tensor, UnsupportedSemantics


class NoNewSymbols:
    def symbolic_tensor(self, *args):
        pytest.fail("input admission must not create replacement symbolic inputs")


def command_buffer(dtype="i8", *, role="input", shape=(2, 3)):
    return {"tensors": {"A": {"dtype": dtype, "shape": list(shape), "role": role}}}


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("role", ["input", "weight"])
def test_matching_integer_inputs_reuse_the_same_symbols(width, role):
    tensor = Tensor(2, 3, width, {})
    cb = command_buffer(f"i{width}", role=role)
    bound = refine._bind_leaves([tensor], cb)
    assert bound["A"] is tensor
    encoder = cb_semantics.CommandBufferEncoder(NoNewSymbols(), cb)
    assert encoder.declare_leaves(shared=bound)["A"] is tensor


@pytest.mark.parametrize("entry", ["binding", "shared_encoder"])
@pytest.mark.parametrize(
    "dtype,width",
    [
        ("i16", 8),
        ("i32", 8),
        ("i64", 8),
        ("i8", 16),
        ("f32", 32),
        ("f64", 64),
        ("bogus", 8),
        ("", 8),
        (None, 8),
        (["i8"], 8),
        (8, 8),
    ],
)
def test_mismatched_or_unsupported_shared_dtypes_refuse(entry, dtype, width):
    tensor = Tensor(2, 3, width, {})
    cb = command_buffer(dtype)
    with pytest.raises(UnsupportedSemantics, match="width|dtype|bitvector"):
        if entry == "binding":
            refine._bind_leaves([tensor], cb)
        else:
            encoder = cb_semantics.CommandBufferEncoder(NoNewSymbols(), cb)
            encoder.declare_leaves(shared={"A": tensor})


@pytest.mark.parametrize("shape", [(3, 2), (2,), (1, 2, 3), ()])
def test_shared_encoder_checks_declared_shape(shape):
    encoder = cb_semantics.CommandBufferEncoder(NoNewSymbols(), command_buffer(shape=shape))
    with pytest.raises(UnsupportedSemantics, match="shape|rank"):
        encoder.declare_leaves(shared={"A": Tensor(2, 3, 8, {})})


@pytest.mark.parametrize("dtype", ["f32", "f64", "bogus", "", None, ["i8"]])
def test_nonshared_inputs_still_refuse_unsupported_dtypes(dtype):
    encoder = cb_semantics.CommandBufferEncoder(NoNewSymbols(), command_buffer(dtype))
    with pytest.raises(UnsupportedSemantics, match="dtype|bitvector"):
        encoder.declare_leaves()


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_nonshared_inputs_keep_their_declared_integer_width(width):
    declarations = []

    class RecordingEncoder:
        def symbolic_tensor(self, name, rows, cols, elem_width):
            declarations.append((name, rows, cols, elem_width))
            return Tensor(rows, cols, elem_width, {})

    encoder = cb_semantics.CommandBufferEncoder(RecordingEncoder(), command_buffer(f"i{width}"))
    assert encoder.declare_leaves()["A"].width == width
    assert declarations == [("A", 2, 3, width)]


@pytest.mark.parametrize("dtype,width", [("i16", 8), ("f32", 8), ("f32", 32), ("bogus", 8)])
def test_invalid_compilation_inputs_refuse_before_target_encoding_or_solver(monkeypatch, dtype, width):
    pytest.importorskip("xdsl")
    tensor = Tensor(2, 3, width, {})
    source = SimpleNamespace(inputs=[tensor], outputs={"commit0": tensor})
    monkeypatch.setattr(refine, "encode_interface", lambda *args, **kwargs: source)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached target encoding or solver invocation")

    monkeypatch.setattr(cb_semantics, "encode_command_buffer", forbidden)
    monkeypatch.setattr(refine, "check_module", forbidden)
    with pytest.raises(UnsupportedSemantics, match="width|dtype|bitvector"):
        refine.validate_compilation(object(), command_buffer(dtype))
