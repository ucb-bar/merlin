"""``capacity_fit`` is a contract predicate, and it must be evaluated, attributed, and charged.

The interface already required it -- `contract.require {requires = ["rhs_immutable", "capacity_fit"]}`
-- and nobody evaluated it. So a backend whose lowering assumes unbounded on-chip storage did not fail
its contract; it aborted the simulator three layers away, and the abort was indistinguishable from an
unreachable oracle. Measured: the graded gemmini backend tiles the ITERATION space correctly but
addresses all kt*nt weight tiles as simultaneously resident, so 512x512 needs 16384 scratchpad rows
against 16384 and spike dies in _M_range_check.
"""
from __future__ import annotations

import pytest

from merlin.compile_cli import capacity_fit


def test_the_predicate_is_one_the_contract_actually_names():
    """If this is our own heuristic rather than a contract obligation, it cannot be charged to anyone."""
    from merlin.xdsl_dialects.contract import ASSUMPTION_KINDS, KNOWN_PREDICATES
    assert "capacity_fit" in KNOWN_PREDICATES
    assert "capacity_fit" in ASSUMPTION_KINDS


@pytest.mark.parametrize("m,k,n,holds", [
    (8, 128, 128, True),        # a real small_llama layer — ran on the mesh
    (16, 256, 512, True),       # measured OK against the oracle
    (16, 512, 256, True),       # measured OK against the oracle
    (16, 512, 512, False),      # measured: spike aborts, __n 16384 >= size 16384
])
def test_the_obligation_matches_what_the_oracle_did(m, k, n, holds):
    v = capacity_fit("gemmini", m, k, n, "int8", 16)
    if v["holds"] is None:
        pytest.skip("gemmini declares no scratchpad capacity in its RTL facts")
    assert v["holds"] is holds, v


def test_an_undeclared_capacity_is_unknown_not_satisfied():
    """Fail closed: a target that declares no capacity must not have the obligation assumed true."""
    v = capacity_fit("definitely_not_a_target", 16, 512, 512, "int8", 16)
    assert v["holds"] is None, "unknown capacity must be None, never True"


def test_a_violation_is_charged_to_the_backend_on_the_graded_path():
    """The check is evaluated by the mesh ENTRY POINT, not inside one endpoint's handler.

    It used to live in the RoCC/oot-cert path, which is where it was first needed. That left every
    other endpoint unchecked -- a self-hosted-ISA target routes its layers through the program oracle,
    so its oversized layers declined with no obligation evaluated at all. Asserting it here (rather
    than inside the handler) is the point: one evaluation covering all three paths. Behavioural
    coverage of both endpoints lives in ``test_capacity_fit_second_target.py``.
    """
    import inspect

    from merlin import compile_cli
    src = inspect.getsource(compile_cli.run_matmul_on_mesh)
    assert "capacity_fit_check" in src, "the obligation must be evaluated before any mesh path runs"
    assert "contract_violation" in src, "a decline that the obligation predicted must be named as one"
    assert "capacity_fit_check" not in inspect.getsource(compile_cli._matmul_via_oot_cert), (
        "one evaluation at the entry point, not a per-endpoint copy that other endpoints lack")


def test_runtime_discharge_is_attributed_not_hidden():
    """Blocking host-side keeps whole-model work moving, but a result produced that way must not read
    as evidence the backend handles the layer."""
    import inspect

    from merlin import compile_cli
    from merlin.targetgen import capsule_runner
    assert "discharged_by" in inspect.getsource(compile_cli.run_matmul_on_mesh)
    assert "capacity_fit_delegated_to_runtime" in inspect.getsource(compile_cli.compile_rvv)
    # `_grade_model_capsule` is the budget wrapper; the grade itself is `_grade_model_capsule_inline`.
    assert "contract_obligations" in inspect.getsource(capsule_runner._grade_model_capsule_inline)
