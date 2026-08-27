"""Operand derivation must read the ABI a command buffer DECLARES, not only infer it from opcodes.

``args_from_cb`` derives the operands the reference harness feeds a submitted kernel. It did so by
reverse-engineering them from what each opcode MEANS -- one hand-written branch per op class, each one
requiring rank-2 operands. Two consequences, both measured on a live run:

  * a backend emitting an op class with no branch here (``VECTOR_MAP``, ``EVICT``) could not be graded at
    all, and neither could one emitting a batched (rank-3) operand;
  * both were reported to the submitter as ``no canonical_inputs`` -- which is not the cause, since the
    runner attaches those from the golden. 8 of 19 failures on one run, every one named against the wrong
    thing.

The command buffer already states the answer: ``kernel_abi = {weight, lhs, outputs}`` is the very ABI this
module documents itself as building (``[weight] ++ [lhs] ++ [output]``). Reading the declaration needs no
opcode vocabulary and no rank limit.

What is pinned here is the ORDER and the FAIL-CLOSED edges. The declared path is a strict FALLBACK: the
opcode path is tried first and its answer always kept, so no command buffer that derives today can change.
Feeding operands cannot make a wrong kernel pass -- the golden is computed independently -- so the only
thing this can turn is an ungradeable capsule into a graded one.
"""
from __future__ import annotations

import pytest

from merlin.runtime.backends.base import get_backend

@pytest.fixture(scope="module")
def mh():
    """The evicted SIMT backend's harness module, resolved through plugin discovery (never by path)."""
    import importlib
    get_backend("muon")                      # registers the out-of-tree backend package
    return importlib.import_module("merlin._oot_backends.muon.muon_harness")


def _cb(*, leaves=None, **kw):
    """A command buffer. ``leaves`` declares the leaf tensors a real capsule carries, so operand VALUES
    resolve through the same deterministic materialization the golden uses."""
    base = {"abi_version": "0.1", "version": "0.1", "target": "t", "commands": [], "tensors": {}}
    base.update(kw)
    for name, shape in (leaves or {}).items():
        base["tensors"][name] = {"shape": list(shape), "dtype": "f32"}
    return base


# ----------------------------------------------------------------- the declaration itself

def test_a_declared_abi_is_read_in_weight_then_lhs_then_output_order(mh):
    """The generic kernel_abi this module documents: [weight] ++ [lhs] ++ [output]."""
    order = mh.declared_abi_order(_cb(kernel_abi={"weight": "W", "lhs": ["A0"], "outputs": ["Y0"]}))
    assert order == (["W", "A0"], ["Y0"])


def test_a_slot_may_hold_one_name_or_several(mh):
    """A backend may declare one weight or a list of them; both are the same declaration."""
    assert mh.declared_abi_order(_cb(kernel_abi={"weight": ["W0", "W1"], "lhs": "A0",
                                                 "outputs": "Y0"})) == (["W0", "W1", "A0"], ["Y0"])


def test_a_half_written_declaration_is_ignored_rather_than_half_honoured(mh):
    for abi in ({}, {"lhs": ["A0"]}, {"outputs": ["Y0"]}, {"weight": None, "lhs": [], "outputs": ["Y0"]}):
        assert mh.declared_abi_order(_cb(kernel_abi=abi)) is None


def test_no_declaration_at_all_is_not_an_error(mh):
    assert mh.declared_abi_order(_cb()) is None
    assert mh.declared_abi_order(_cb(kernel_abi="not-a-mapping")) is None


# ----------------------------------------------------------------- precedence: opcode path wins

def test_the_opcode_path_is_tried_first_and_its_answer_is_kept(mh, monkeypatch):
    """The load-bearing guarantee: every command buffer that derives today derives identically tomorrow."""
    sentinel = ([mh.TensorArg("from_opcode", 1, 1, [7.0], "f32")], [])
    monkeypatch.setattr(mh, "_args_from_cb_by_opcode", lambda cb: sentinel)
    got = mh.args_from_cb(_cb(kernel_abi={"lhs": ["A0"], "outputs": ["Y0"]},
                              operand_shapes={"A0": [2, 2], "Y0": [2, 2]}))
    assert got is sentinel, "a declared ABI must never override a derivation the opcode path made"


def test_a_crash_in_the_opcode_path_falls_through_instead_of_aborting(mh, monkeypatch):
    """An unfamiliar cb must reach the fallback, not take the harness down with it."""
    def _boom(cb):
        raise ValueError("unfamiliar command buffer")
    monkeypatch.setattr(mh, "_args_from_cb_by_opcode", _boom)
    ins, outs = mh.args_from_cb(_cb(kernel_abi={"lhs": ["A0"], "outputs": ["Y0"]},
                                    operand_shapes={"A0": [2, 2], "Y0": [2, 2]},
                                    leaves={"A0": [2, 2]}))
    assert [a.name for a in ins] == ["A0"] and [a.name for a in outs] == ["Y0"]


def test_neither_path_resolving_still_returns_None(mh, monkeypatch):
    """Fail closed is unchanged -- the fallback adds an answer, it never invents one."""
    monkeypatch.setattr(mh, "_args_from_cb_by_opcode", lambda cb: None)
    assert mh.args_from_cb(_cb()) is None


# ----------------------------------------------------------------- what the fallback unblocks

def _derive(mh, monkeypatch, cb):
    monkeypatch.setattr(mh, "_args_from_cb_by_opcode", lambda _cb: None)
    return mh.args_from_cb(cb)


def test_an_opcode_this_harness_never_models_is_gradeable_via_the_declaration(mh, monkeypatch):
    """The point: feeding operands needs the buffers, not the opcode's meaning."""
    assert not mh.models_opcode("VECTOR_MAP"), "fixture assumes this stays unmodelled"
    ins, outs = _derive(mh, monkeypatch, _cb(
        commands=[{"opcode": "VECTOR_MAP", "operands": {"src": "A0", "dst": "Y0"}}],
        kernel_abi={"lhs": ["A0", "A1"], "outputs": ["Y0"]},
        operand_shapes={"A0": [16, 16], "A1": [16], "Y0": [16, 16]},
        leaves={"A0": [16, 16], "A1": [16]}))
    assert [(a.name, a.rows, a.cols) for a in ins] == [("A0", 16, 16), ("A1", 1, 16)]
    assert [(a.name, a.rows, a.cols) for a in outs] == [("Y0", 16, 16)]


def test_a_batched_operand_folds_to_the_same_row_major_bytes(mh, monkeypatch):
    """Rank > 2 was rejected outright. (b, m, n) occupies exactly the bytes of (b*m, n), so folding is a
    relabelling -- which is what lets a batched operand be fed with no batched code path."""
    ins, outs = _derive(mh, monkeypatch, _cb(
        kernel_abi={"lhs": ["A0", "A1"], "outputs": ["Y0"]},
        operand_shapes={"A0": [2, 16, 16], "A1": [2, 16, 1], "Y0": [2, 16, 1]},
        leaves={"A0": [2, 16, 16], "A1": [2, 16, 1]}))
    assert [(a.name, a.rows, a.cols) for a in ins] == [("A0", 32, 16), ("A1", 32, 1)]
    assert [(a.name, a.rows, a.cols) for a in outs] == [("Y0", 32, 1)]
    assert all(len(a.values) == a.rows * a.cols for a in ins)


def test_a_rank_1_operand_is_a_row_vector(mh, monkeypatch):
    ins, _ = _derive(mh, monkeypatch, _cb(kernel_abi={"lhs": ["A0"], "outputs": ["Y0"]},
                                          operand_shapes={"A0": [16], "Y0": [16]},
                                          leaves={"A0": [16]}))
    assert (ins[0].rows, ins[0].cols) == (1, 16)


def test_the_goldens_operands_are_used_when_the_runner_attached_them(mh, monkeypatch):
    """Values must match the independent golden bit-for-bit, so canonical operands outrank materialization."""
    ins, _ = _derive(mh, monkeypatch, _cb(
        kernel_abi={"lhs": ["A0"], "outputs": ["Y0"]},
        operand_shapes={"A0": [2, 2], "Y0": [2, 2]},
        canonical_inputs={"A0": {"values": [[1.0, 2.0], [3.0, 4.0]]}}))
    assert ins[0].values == [1.0, 2.0, 3.0, 4.0], "nested canonical operands must deep-flatten"


def test_a_declaration_that_contradicts_its_own_operand_is_refused(mh, monkeypatch):
    """A shape the values cannot fill is a broken declaration; feeding it would silently mis-measure."""
    assert _derive(mh, monkeypatch, _cb(
        kernel_abi={"lhs": ["A0"], "outputs": ["Y0"]},
        operand_shapes={"A0": [4, 4], "Y0": [4, 4]},
        canonical_inputs={"A0": {"values": [1.0, 2.0]}})) is None


def test_an_input_with_no_resolvable_shape_is_refused(mh, monkeypatch):
    assert _derive(mh, monkeypatch, _cb(kernel_abi={"lhs": ["A0"], "outputs": ["Y0"]},
                                        operand_shapes={"Y0": [2, 2]})) is None


def test_shapes_declared_but_no_operand_values_available_is_refused(mh, monkeypatch):
    """A declaration alone is not enough. With no leaf to materialize and no canonical operand attached,
    there is nothing to feed -- and feeding zeros would silently mis-measure rather than fail."""
    assert _derive(mh, monkeypatch, _cb(kernel_abi={"lhs": ["A0"], "outputs": ["Y0"]},
                                        operand_shapes={"A0": [2, 2], "Y0": [2, 2]})) is None


# --------------------------------------------------------------- when the declaration must win
# The opcode branches predate block-scaled formats, so for a microscaling capsule they return a
# PLAUSIBLE answer that is missing the E8M0 scale streams entirely -- and a plausible answer wins under
# "opcode first", which would leave a kernel fed half its operands and the whole scale contract inert.
# The override is deliberately narrow: only when every operand the declaration ADDS is a scale.

def _mx_cb(scale_role="scale"):
    return _cb(kernel_abi={"weight": ["W", "W_scale"], "lhs": ["A0", "A0_scale"], "outputs": ["Y0"]},
               operand_shapes={"W": [2, 2], "A0": [2, 2], "W_scale": [1, 2], "A0_scale": [1, 2],
                               "Y0": [2, 2]},
               leaves={"W": [2, 2], "A0": [2, 2], "W_scale": [1, 2], "A0_scale": [1, 2]},
               operands=[{"name": "W", "role": "weight"}, {"name": "A0", "role": "input"},
                         {"name": "W_scale", "role": scale_role},
                         {"name": "A0_scale", "role": scale_role},
                         {"name": "Y0", "role": "output"}])


def test_declared_scales_are_fed_even_when_the_opcode_path_answered(mh, monkeypatch):
    """The op branch returns the elements alone; the declaration adds the scales, so it wins."""
    monkeypatch.setattr(mh, "_args_from_cb_by_opcode",
                        lambda cb: ([mh.TensorArg("W", 2, 2, [0.0] * 4, "f32"),
                                     mh.TensorArg("A0", 2, 2, [0.0] * 4, "f32")],
                                    [mh.TensorArg("Y0", 2, 2, [0.0] * 4, "f32")]))
    ins, _ = mh.args_from_cb(_mx_cb())
    assert [a.name for a in ins] == ["W", "W_scale", "A0", "A0_scale"], (
        "a kernel fed fewer operands than its capsule declares is not given a smaller answer, "
        "it is given the wrong one")


def test_extra_operands_that_are_not_scales_do_not_override(mh, monkeypatch):
    """Narrow on purpose: a cb may declare operands an op branch folds in for other reasons, and
    overriding on those changed 9 of 35 real command buffers when it was tried."""
    monkeypatch.setattr(mh, "_args_from_cb_by_opcode",
                        lambda cb: ([mh.TensorArg("W", 2, 2, [0.0] * 4, "f32"),
                                     mh.TensorArg("A0", 2, 2, [0.0] * 4, "f32")],
                                    [mh.TensorArg("Y0", 2, 2, [0.0] * 4, "f32")]))
    ins, _ = mh.args_from_cb(_mx_cb(scale_role="input"))
    assert [a.name for a in ins] == ["W", "A0"], "only a SCALE may override the opcode derivation"


def test_the_scale_role_is_read_from_the_command_buffers_own_declaration(mh):
    cb = _mx_cb()
    assert mh._is_scale_operand(cb, "W_scale") and mh._is_scale_operand(cb, "A0_scale")
    assert not mh._is_scale_operand(cb, "W") and not mh._is_scale_operand(cb, "A0")
    assert not mh._is_scale_operand(cb, "nonexistent")
