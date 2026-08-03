"""The behavioral program-terminator derivation in isa_introspect: an op is a terminator IFF its semantic
method's ENTIRE effect is to raise one boolean machine-state flag to True (no register/memory method calls,
no other writes). Hermetic — synthetic op classes, no real target, no model venv. Proves the derivation is
behavioral (never a mnemonic/opcode literal) and that it separates a terminator from a barrier and from a
datapath op.
"""
from __future__ import annotations

from merlin.targetgen.oracle_helpers import isa_introspect as I


class _Halt:
    """Sole effect: assert the machine finish flag (name discovered, not assumed)."""
    def exec(self, state):
        state.halted = True


class _Barrier:
    """A fence: no state effect at all — must NOT be a terminator."""
    def exec(self, state):
        pass


class _Compute:
    """A datapath op: reads an (unset) operand and calls a state write method — must NOT be a terminator."""
    def exec(self, state):
        state.write_reg(self.rd, state.read_reg(self.rs1) + 1)


class _OtherHalt:
    """A second terminator raising the SAME flag — consensus picks this flag."""
    def exec(self, state):
        state.halted = True


class _FlagButAlsoWrites:
    """Raises the flag but ALSO calls a state method — not a pure terminator, so not counted."""
    def exec(self, state):
        state.write_reg(0, 0)
        state.halted = True


def test_terminator_flag_detects_pure_halt():
    assert I._terminator_flag(_Halt) == "halted"


def test_barrier_is_not_a_terminator():
    assert I._terminator_flag(_Barrier) is None


def test_compute_op_is_not_a_terminator():
    assert I._terminator_flag(_Compute) is None


def test_flag_plus_side_effect_is_not_a_pure_terminator():
    assert I._terminator_flag(_FlagButAlsoWrites) is None


def test_halt_ops_consensus_over_a_synthetic_module():
    import types
    mod = types.ModuleType("_fake_isa")
    for name, cls in {"HALT": _Halt, "BARRIER": _Barrier, "COMPUTE": _Compute,
                      "HALT2": _OtherHalt}.items():
        setattr(mod, name, cls)
    by_mnem = {"HALT": {}, "BARRIER": {}, "COMPUTE": {}, "HALT2": {}}
    names, flag = I._halt_ops(mod, by_mnem)
    assert names == ["HALT", "HALT2"]                 # both terminators, sorted
    assert flag == "halted"                           # the discovered consensus flag


def test_no_terminator_yields_empty_not_a_guess():
    import types
    mod = types.ModuleType("_fake_isa2")
    mod.BARRIER = _Barrier
    mod.COMPUTE = _Compute
    names, flag = I._halt_ops(mod, {"BARRIER": {}, "COMPUTE": {}})
    assert names == [] and flag is None               # fail-closed: honest empty, never a false halt op
