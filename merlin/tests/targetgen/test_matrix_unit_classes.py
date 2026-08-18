"""A matrix extension's coverage expectation, derived from its own encodings.

The command-buffer OPU corpus derives `expected.instruction_classes` to `[]`, because that endpoint has
no instruction decode — and an empty required set is satisfied by a submission that emits nothing, so
coverage cannot fail. The instructions DO exist on the other surface (the four reserved vector-opcode
slots), they are just read by a different deriver than either regime `_classes_source` knew about.

These tests pin three things: the bridge produces the derived instruction NAMES (no invented
vocabulary), it fails closed in every way that would otherwise yield an empty list, and the two targets
that already had a class source are untouched.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import isa_taxonomy as IT


class _Enc:
    def __init__(self, name):
        self.mnemonic = name


def _derivation(names=("OPMACC", "OPMVIN", "OPMVINBCAST", "OPMVOUT"), *, ok=True, gaps=(),
                crosschecks=({"agrees": True},)):
    return SimpleNamespace(encodings={n: _Enc(n) for n in names}, gaps=tuple(gaps),
                           crosschecks=tuple(crosschecks), ok=ok)


def _contract(roles=None):
    return SimpleNamespace(kernel_roles=roles or {"accumulate": "OPMACC", "broadcast": "OPMVINBCAST",
                                                  "readout": "OPMVOUT", "operand_load": "OPMVIN"})


@pytest.fixture()
def unit(monkeypatch):
    """A stand-in matrix unit, so these tests need no RTL checkout."""
    def _install(derivation=None, contract=None):
        import types
        fake = types.ModuleType("merlin.llvmlower.opu_shim")
        fake.load_contract = lambda u, path=None: contract or _contract()
        fake.derive_encodings = lambda uc: derivation or _derivation()
        import merlin.llvmlower as _ll
        monkeypatch.setattr(_ll, "opu_shim", fake, raising=False)
        import sys
        monkeypatch.setitem(sys.modules, "merlin.llvmlower.opu_shim", fake)
    return _install


# --------------------------------------------------------------------------- the derived class names
def test_the_classes_are_the_units_own_instruction_names(unit):
    unit()
    assert IT.matrix_unit_role_classes("u") == {
        "matmul": ["OPMACC"], "weight_load": ["OPMVINBCAST"],
        "memory": ["OPMVIN"], "acc_readout": ["OPMVOUT"]}


def test_a_matmul_capsule_requires_load_push_multiply_and_readout(unit):
    unit()
    got = IT.matrix_unit_classes_for("u")(op="matmul", output_dtype="i32")
    assert got == ["OPMVIN", "OPMVINBCAST", "OPMACC", "OPMVOUT"]


def test_a_movement_capsule_requires_only_the_memory_op(unit):
    unit()
    assert IT.matrix_unit_classes_for("u")(op="movement", movement=True) == ["OPMVIN"]


def test_an_epilogue_adds_nothing_because_this_unit_has_no_unary_instruction(unit):
    """Recorded, not assumed: the relu slot wants a tensor-unary role, and a matrix extension that has
    none simply does not contribute one — the epilogue runs on the vector unit. If a future unit ships
    one, this test is the place that says the behaviour changed."""
    unit()
    f = IT.matrix_unit_classes_for("u")
    assert f(op="matmul", output_dtype="i32", epilogue=("relu",)) == f(op="matmul", output_dtype="i32")


# --------------------------------------------------------------------------- failing closed
def test_an_ungrounded_derivation_refuses(unit):
    unit(derivation=_derivation(ok=False, gaps=("OPMACC: no funct6",)))
    with pytest.raises(ValueError, match="not fully derived"):
        IT.matrix_unit_role_classes("u")


def test_a_crosscheck_disagreement_refuses(unit):
    unit(derivation=_derivation(ok=False, crosschecks=({"agrees": False},)))
    with pytest.raises(ValueError, match="not fully derived"):
        IT.matrix_unit_role_classes("u")


def test_a_role_naming_an_instruction_the_rtl_lacks_refuses(unit):
    """The contract and the RTL disagreeing is exactly what must not pass silently."""
    unit(derivation=_derivation(names=("OPMACC", "OPMVIN", "OPMVOUT")))
    with pytest.raises(ValueError, match="does not contain"):
        IT.matrix_unit_role_classes("u")


def test_no_mappable_role_refuses_rather_than_returning_empty(unit):
    unit(contract=_contract(roles={"something_else": "OPMACC"}))
    with pytest.raises(ValueError, match="no declared kernel role"):
        IT.matrix_unit_classes_for("u")


def test_an_op_that_resolves_to_no_class_refuses(unit):
    """A capsule whose coverage expectation is empty cannot fail, so it is refused."""
    unit(contract=_contract(roles={"accumulate": "OPMACC"}))     # no memory role at all
    f = IT.matrix_unit_classes_for("u")
    with pytest.raises(ValueError, match="NO instruction classes"):
        f(op="movement", movement=True)


# --------------------------------------------------------------------------- routing, and no regression
def test_a_declared_unit_is_read_from_the_compute_unit(unit):
    unit()
    f = CS._classes_source(None, {"compute_units": [{"name": "u", "matrix_unit": "some_unit"}]})
    assert f(op="matmul", output_dtype="i32") == ["OPMVIN", "OPMVINBCAST", "OPMACC", "OPMVOUT"]


def test_no_declaration_leaves_the_existing_regimes_alone():
    assert CS._declared_matrix_unit({"compute_units": [{"name": "u"}]}) is None
    assert CS._declared_matrix_unit({}) is None


def test_the_rocc_encoding_regime_is_unchanged():
    """The command regime must keep deriving its semantic classes from the encoding map."""
    contract = {"encoding": {"semantic_class": {"1": "MVIN", "2": "MVOUT", "3": "CONFIG",
                                               "4": "COMPUTE_PRELOADED", "5": "PRELOAD"},
                             "config_subtype": {"a": "CONFIG_EX", "b": "CONFIG_LD"}}}
    got = CS._classes_source(None, contract)(op="matmul")
    assert got == ["CONFIG_EX", "CONFIG_LD", "MVIN", "PRELOAD", "COMPUTE_PRELOADED", "MVOUT"]
