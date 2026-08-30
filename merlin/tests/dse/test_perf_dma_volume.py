"""Movement-volume prediction must degrade to a floor, and must never read a size by position.

Two failures these pin, both of which return a plausible number instead of raising:

* summing only the descriptors that resolved. That understates the footprint in the FLATTERING
  direction -- a smaller predicted volume makes the compiler look better and the model look more
  accurate at once -- so any unresolved descriptor must demote the whole kernel to a lower bound.
* reading the size from a fixed operand position. It works on the form you tested and silently
  returns the wrong field on the next one.
"""
from __future__ import annotations

from merlin.perf.dma_volume import (Descriptor, compare_to_measured, kernel_volume,
                                    propagate_constants, size_field_for)


class _Isa:
    """A stand-in ISA model exposing only the layout accessor this module is allowed to use."""

    def __init__(self, layouts):
        self._l = layouts

    def fields_of(self, mnemonic):
        return self._l[mnemonic]


def _desc(i, direction="read", size=64, reason=None, field="len"):
    return Descriptor(index=i, form="move", channel=0, direction=direction,
                      size_bytes=size, size_field=field if size is not None else None,
                      unresolved_reason=reason)


def test_a_resolved_kernel_reports_a_total() -> None:
    v = kernel_volume("k", [_desc(0, "read", 64), _desc(1, "write", 32)])
    assert (v.read_bytes, v.write_bytes, v.total_bytes) == (64, 32, 96)
    assert v.is_lower_bound is False
    assert "AT LEAST" not in v.claim()


def test_one_unresolved_descriptor_demotes_the_whole_kernel_to_a_floor() -> None:
    """The flattering-direction guard: a partial sum must not present as a total."""
    v = kernel_volume("k", [_desc(0, "read", 64),
                            _desc(1, "read", None, reason="size came from a runtime value")])
    assert v.is_lower_bound is True
    assert v.total_bytes == 64                     # the resolved part is still reported
    assert "AT LEAST" in v.claim()                 # but never as a total
    assert v.unresolved and "runtime value" in v.unresolved[0]


def test_a_floor_below_its_measurement_is_consistent_not_a_match() -> None:
    """A lower bound cannot agree with a measurement; every unresolved descriptor could close the gap."""
    v = kernel_volume("k", [_desc(0, "read", 64), _desc(1, "read", None, reason="unresolved")])
    out = compare_to_measured(v, 4096)
    assert out["verdict"] == "consistent_lower_bound"
    assert "cannot match" in out["note"]


def test_a_floor_above_its_measurement_falsifies_an_input() -> None:
    """The same rule the headline experiment applies: a bound over its measurement is a refutation."""
    v = kernel_volume("k", [_desc(0, "read", 8192), _desc(1, "read", None, reason="unresolved")])
    out = compare_to_measured(v, 4096)
    assert out["verdict"] == "bound_violated"
    assert "falsifies" in out["note"]


def test_the_size_field_comes_from_the_declared_layout() -> None:
    isa = _Isa({"move.a": {"rd": [7], "rs1": [15], "len": [20, 21]}})
    assert size_field_for(isa, "move.a") == "len"


def test_a_form_that_declares_no_size_is_unknown_not_a_guess() -> None:
    """The pick-by-position bug: with no size field declared, the answer is None, not 'operand 2'."""
    isa = _Isa({"move.b": {"rd": [7], "rs1": [15], "vd": [20]}})
    assert size_field_for(isa, "move.b") is None


def test_the_layout_decides_which_field_even_when_forms_differ() -> None:
    """Two forms of one family may carry the size in different fields; position would confuse them."""
    isa = _Isa({"move.a": {"rd": [7], "len": [20]}, "move.b": {"rd": [7], "rs2": [20]}})
    assert size_field_for(isa, "move.a") == "len"
    assert size_field_for(isa, "move.b") == "rs2"


def test_propagation_kills_a_register_it_cannot_evaluate() -> None:
    """A register rewritten by an opaque instruction must go UNKNOWN, never keep a stale constant."""
    prog = [{"form": "li", "operands": {"rd": 2, "imm": 1024}},
            {"form": "opaque", "operands": {"rd": 2}},
            {"form": "move", "operands": {"rd": 0, "rs2": 2}}]
    states = propagate_constants(prog, immediate_forms={"li": "imm"})
    assert states[0][2] == 1024
    assert states[2][2] is None, "a stale constant survived an intervening write"


def test_a_backward_branch_invalidates_every_constant() -> None:
    """Inside a loop the 'constant' differs per iteration, so it is not a constant."""
    prog = [{"form": "li", "operands": {"rd": 2, "imm": 1024}},
            {"form": "br", "operands": {}, "branches_backward": True},
            {"form": "move", "operands": {"rd": 0, "rs2": 2}}]
    states = propagate_constants(prog, immediate_forms={"li": "imm"})
    assert states[2].get(2) is None, "a loop-carried value was treated as a constant"


def test_register_zero_is_never_marked_unknown() -> None:
    """Writing x0 discards; it must not poison the state map."""
    prog = [{"form": "opaque", "operands": {"rd": 0}}]
    assert propagate_constants(prog, immediate_forms={})[0].get(0) is None
