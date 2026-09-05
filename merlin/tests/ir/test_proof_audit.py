"""`contract.prove` tokens are classified by evidence, never read as proofs.

The dialect's verifier compares two strings; that is all it can do. These tests lock the distinction
between a token existing and a token being backed by a verification result, because conflating them
is exactly how "we have a proof obligation system" becomes an overclaim.
"""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


@pytest.fixture(scope="module")
def contract_module():
    from merlin.xdsl_dialects.lowering import pipeline
    return pipeline.lower_repeated_rhs_matmul(reuse=2, m=2, k=2, n=2).contract_module


def test_tokens_are_asserted_when_nothing_has_discharged_them(contract_module):
    """With no verification evidence, every token must be `asserted` — never upgraded by default."""
    from merlin.verify.proofs import ASSERTED, audit_proofs, summarize

    proofs = audit_proofs(contract_module)
    assert proofs, "the reference workload should carry proof tokens"
    assert all(p.status == ASSERTED for p in proofs), summarize(proofs)
    assert summarize(proofs)["verified"] == 0


def test_a_discharged_requirement_becomes_verified(contract_module):
    from merlin.verify.proofs import audit_proofs, summarize

    proofs = audit_proofs(contract_module)
    one = proofs[0]
    discharged = {one.producer_pass: {one.requirement}}
    after = summarize(audit_proofs(contract_module, discharged=discharged))
    assert after["verified"] == 1
    assert after["asserted"] == len(proofs) - 1


def test_evidence_is_scoped_to_the_producing_pass(contract_module):
    """A requirement discharged for a DIFFERENT pass must not credit this one."""
    from merlin.verify.proofs import ASSERTED, audit_proofs

    proofs = audit_proofs(contract_module)
    one = proofs[0]
    wrong = {"some-other-pass": {one.requirement}}
    assert all(p.status == ASSERTED for p in audit_proofs(contract_module, discharged=wrong))
